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

#include <fdaPDE/geometric_finite_elements.h>

#include <array>
#include <cmath>
#include <limits>
#include <span>
#include <stdexcept>

namespace {

struct Geometry {
    using Point = double;
    using Tangent = double;

    std::size_t dimension() const { return 1; }
    double inner_product(const Point&, const Tangent& u, const Tangent& v) const { return u * v; }
    double norm(const Point&, const Tangent& value) const { return std::abs(value); }
    Tangent project(const Point&, const Tangent& value) const { return value; }
    Tangent zero_tangent(const Point&) const { return 0; }
    Tangent linear_combination(const Point&, double alpha, const Tangent& u, double beta, const Tangent& v) const {
        return alpha * u + beta * v;
    }
    Point retract(const Point& point, const Tangent& tangent, double step) const { return point + step * tangent; }
    Point exponential(const Point& point, const Tangent& tangent, double step) const { return point + step * tangent; }
    Tangent logarithm(const Point& from, const Point& to) const { return to - from; }
    double distance(const Point& from, const Point& to) const { return std::abs(to - from); }
};

template <typename Function> bool throws_invalid_argument(Function&& function) {
    try {
        function();
    } catch (const std::invalid_argument&) { return true; } catch (...) {
        return false;
    }
    return false;
}

}   // namespace

int main() {
    const Geometry geometry;
    const std::array<double, 2> values {
      {2, 7}
    };
    const std::array<double, 2> invalid_weights {
      {2, 2}
    };
    try {
        static_cast<void>(fdapde::gfe::p1_geodesic_value(
          geometry, std::span<const double>(values), std::span<const double>(invalid_weights), 0));
        return 1;
    } catch (const std::invalid_argument&) { }

    const std::array<double, 2> vertex_weights {
      {0, 1}
    };
    const auto result = fdapde::gfe::p1_geodesic_value(
      geometry, std::span<const double>(values), std::span<const double>(vertex_weights), 0);
    if (
      !result.converged() || result.value != 7 ||
      result.stop_reason != fdapde::manifold::BarycenterStopReason::closed_form) {
        return 2;
    }

    using LogGeometry = fdapde::manifold::LogEuclideanSPDGeometry<double, 3>;
    using LogPoint = LogGeometry::Point;
    using LogTangent = LogGeometry::Tangent;
    fdapde::linalg::Matrix<double, 3, 3> identity;
    fdapde::linalg::Matrix<double, 3, 3> two_identity;
    identity.set_zero();
    two_identity.set_zero();
    for (int i = 0; i < 3; ++i) {
        identity(i, i) = 1;
        two_identity(i, i) = 2;
    }

    const LogGeometry log_geometry;
    const std::array<LogPoint, 2> log_values {
      {LogPoint(identity, fdapde::linalg::checked), LogPoint(two_identity, fdapde::linalg::checked)}
    };
    const std::array<double, 2> log_weights {
      {0.5, 0.5}
    };
    const auto linearization = fdapde::gfe::p1_geodesic_linearization(
      log_geometry, std::span<const LogPoint>(log_values), std::span<const double>(log_weights));
    if (!linearization.result().converged()) return 3;

    const std::array<double, 2> nonzero_sum_direction {
      {0.5, 0.25}
    };
    if (!throws_invalid_argument(
          [&] { static_cast<void>(linearization.weight_jvp(std::span<const double>(nonzero_sum_direction))); })) {
        return 4;
    }

    LogTangent zero;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j <= i; ++j) zero(i, j) = 0;
    }
    const std::array<LogTangent, 2> nodal_directions {
      {zero, zero}
    };
    if (!throws_invalid_argument([&] {
            static_cast<void>(linearization.nodal_jvp(std::span<const LogTangent>(nodal_directions.data(), 1)));
        })) {
        return 5;
    }

    LogTangent nonfinite = zero;
    nonfinite(0, 0) = std::numeric_limits<double>::infinity();
    const std::array<LogTangent, 2> nonfinite_nodal_directions {
      {nonfinite, zero}
    };
    if (!throws_invalid_argument([&] {
            static_cast<void>(linearization.nodal_jvp(std::span<const LogTangent>(nonfinite_nodal_directions)));
        })) {
        return 6;
    }
    if (!throws_invalid_argument([&] { static_cast<void>(linearization.nodal_vjp(nonfinite)); })) return 7;

    const std::array<double, 2> valid_weight_direction {
      {0.5, -0.5}
    };
    static_cast<void>(linearization.weight_jvp(std::span<const double>(valid_weight_direction)));
    static_cast<void>(linearization.nodal_jvp(std::span<const LogTangent>(nodal_directions)));
    if (linearization.nodal_vjp(zero).size() != log_values.size()) return 8;
    return 0;
}
