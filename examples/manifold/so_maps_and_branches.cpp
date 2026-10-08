// This file is part of fdaPDE, a C++ library for physics-informed
// spatial and functional data analysis.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

// compile from the repository root with c++ -std=c++20 -O2 -pthread -I.
// examples/manifold/so_maps_and_branches.cpp
#include <fdaPDE/manifold_geometry.h>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <stdexcept>

using namespace fdapde;
using namespace fdapde::manifold;

namespace {
using Geometry = SOGeometry<double, 3>;
using Point = Geometry::Point;
using Dense = Matrix<double, 3, 3>;

/// @brief names the geometric or numerical status reported by a selected logarithm branch
const char* branch_status(RotationLogStatus status) {
    switch (status) {
    case RotationLogStatus::Regular:
        return "regular";
    case RotationLogStatus::NearCut:
        return "near cut locus";
    case RotationLogStatus::Ambiguous:
        return "ambiguous";
    case RotationLogStatus::Unresolved:
        return "unresolved";
    }
    return "unknown";
}
}   // namespace

// relates body tangents to ambient velocities and makes the logarithm branch explicit at a half turn
int main() {
    const Geometry geometry;
    constexpr double angle = .35;
    const Point from(Dense({std::cos(angle), -std::sin(angle), 0., std::sin(angle), std::cos(angle), 0., 0., 0., 1.}));
    auto direction = geometry.zero_tangent(from);
    direction(0, 1) = -.2;
    direction(0, 2) = .1;
    direction(1, 2) = -.15;

    // a packed skew tangent represents Q Omega rather than a full ambient matrix
    const auto ambient = geometry.to_ambient(from, direction);
    const auto restored = geometry.from_ambient(from, ambient);
    const double body_norm = geometry.norm(from, direction);
    const double expected_norm = std::sqrt(2 * (.2 * .2 + .1 * .1 + .15 * .15));
    // counting both skew triangles independently verifies the full-Frobenius normalization
    fdapde_strong_assert(
      std::isfinite(body_norm) && std::abs(body_norm - expected_norm) < 1e-13, std::runtime_error,
      "body tangent metric mismatch");
    // multiplying by an orthogonal base and projecting back must preserve the original body coordinates
    fdapde_strong_assert(
      (restored - direction).norm() < 1e-13, std::runtime_error, "body and ambient coordinates disagree");

    constexpr double step = .75;
    const auto to = geometry.exponential(from, direction, step);
    const auto recovered = geometry.logarithm(from, to);
    const double logarithm_error = (recovered - step * direction).norm();
    const auto regular_branch = geometry.minimum_logarithm(from, to);
    // a short geodesic lies inside the unique logarithm branch and recovers its known initial velocity
    fdapde_strong_assert(
      regular_branch.diagnostics.regular() && logarithm_error < 1e-12, std::runtime_error,
      "short exponential and logarithm disagree");
    auto probe = geometry.zero_tangent(from);
    probe(0, 1) = .1;
    probe(0, 2) = -.3;
    probe(1, 2) = .2;
    const auto transported = geometry.transport(from, to, probe);
    const double transport_error = std::abs(geometry.norm(to, transported) - geometry.norm(from, probe));
    // transporting a different body direction along this geodesic preserves its metric norm
    fdapde_strong_assert(transport_error < 1e-12, std::runtime_error, "parallel transport changed the metric norm");

    std::cout << std::scientific << std::setprecision(6) << "body norm (full Frobenius): " << body_norm << '\n'
              << "ambient velocity norm: " << ambient.norm() << '\n'
              << "short-step distance: " << geometry.distance(from, to) << '\n'
              << "exp/log error: " << logarithm_error << '\n'
              << "transport norm error: " << transport_error << '\n'
              << "short-step branch: " << branch_status(regular_branch.diagnostics.status) << '\n';

    // an exact planar half turn has two shortest orientations and no unique minimum logarithm
    const auto identity = Point::Identity();
    const Point half_turn(Dense({-1., 0., 0., 0., -1., 0., 0., 0., 1.}));
    bool ordinary_log_rejected = false;
    try {
        (void)geometry.logarithm(identity, half_turn);
    } catch (const std::domain_error&) { ordinary_log_rejected = true; }
    // the ordinary logarithm refuses to silently choose one of the two shortest half-turn orientations
    fdapde_strong_assert(ordinary_log_rejected, std::runtime_error, "cut-locus logarithm selected a hidden branch");
    const auto branch = geometry.minimum_logarithm(identity, half_turn);
    // exact minus-one eigenvalues report mathematical ambiguity instead of a regular logarithm
    fdapde_strong_assert(
      branch.diagnostics.status == RotationLogStatus::Ambiguous, std::runtime_error,
      "half turn did not report its ambiguous branch");
    const auto curve = geometry.geodesic(identity, half_turn, branch);
    const Point midpoint(curve(.5)), endpoint(curve(1.));
    const double midpoint_distance = geometry.distance(identity, midpoint);
    // the explicitly selected branch reaches the supplied endpoint without changing its rotation coefficients
    fdapde_strong_assert(
      (endpoint - half_turn).norm() < 1e-12, std::runtime_error, "selected branch missed its endpoint");
    // full-Frobenius distance along a half-time half turn is pi divided by the square root of two
    fdapde_strong_assert(
      std::isfinite(midpoint_distance) && std::abs(midpoint_distance - std::numbers::pi / std::sqrt(2.)) < 1e-12,
      std::runtime_error, "selected branch midpoint has the wrong distance");
    std::cout << "half-turn branch: " << branch_status(branch.diagnostics.status) << '\n'
              << "ordinary logarithm rejected: " << std::boolalpha << ordinary_log_rejected << '\n'
              << "explicit branch midpoint distance: " << midpoint_distance << '\n'
              << "branch reconstruction residual: " << branch.diagnostics.residual << '\n';
}
