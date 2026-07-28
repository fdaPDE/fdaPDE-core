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
    return 0;
}
