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

#ifndef __FDAPDE_MANIFOLD_POWER_GEOMETRY_H__
#define __FDAPDE_MANIFOLD_POWER_GEOMETRY_H__

#include "header_check.h"

namespace fdapde {
namespace manifold {

template <FirstOrderGeometry ComponentGeometry> class PowerGeometry {
   public:
    using ComponentPoint = point_t<ComponentGeometry>;
    using ComponentTangent = tangent_t<ComponentGeometry>;
    using Point = std::vector<ComponentPoint>;
    using Tangent = std::vector<ComponentTangent>;

    explicit PowerGeometry(std::size_t factor_count, ComponentGeometry geometry = {}) :
        factor_count_(factor_count), geometry_(std::move(geometry)) {
        if (factor_count_ == 0) { throw std::invalid_argument("A power geometry needs at least one factor."); }
    }

    std::size_t factor_count() const { return factor_count_; }
    std::size_t dimension() const {
        const std::size_t component_dimension = geometry_.dimension();
        if (component_dimension != 0 && factor_count_ > std::numeric_limits<std::size_t>::max() / component_dimension) {
            throw std::overflow_error("Power-geometry dimension exceeds the supported range.");
        }
        return factor_count_ * component_dimension;
    }
    const ComponentGeometry& component_geometry() const& { return geometry_; }
    const ComponentGeometry& component_geometry() const&& = delete;

    double inner_product(const Point& point, const Tangent& u, const Tangent& v) const {
        check_point_(point);
        check_tangent_(u);
        check_tangent_(v);
        double result = 0;
        for (std::size_t i = 0; i < factor_count_; ++i) { result += geometry_.inner_product(point[i], u[i], v[i]); }
        return result;
    }

    double norm(const Point& point, const Tangent& tangent) const {
        check_point_(point);
        check_tangent_(tangent);
        double result = 0;
        for (std::size_t i = 0; i < factor_count_; ++i) {
            result = std::hypot(result, geometry_.norm(point[i], tangent[i]));
        }
        return result;
    }

    Tangent project(const Point& point, const Tangent& ambient) const {
        check_point_(point);
        check_tangent_(ambient);
        Tangent result;
        result.reserve(factor_count_);
        for (std::size_t i = 0; i < factor_count_; ++i) { result.push_back(geometry_.project(point[i], ambient[i])); }
        return result;
    }

    Tangent zero_tangent(const Point& point) const {
        check_point_(point);
        Tangent result;
        result.reserve(factor_count_);
        for (const auto& component : point) { result.push_back(geometry_.zero_tangent(component)); }
        return result;
    }

    Tangent
    linear_combination(const Point& point, double alpha, const Tangent& u, double beta, const Tangent& v) const {
        check_point_(point);
        check_tangent_(u);
        check_tangent_(v);
        Tangent result;
        result.reserve(factor_count_);
        for (std::size_t i = 0; i < factor_count_; ++i) {
            result.push_back(geometry_.linear_combination(point[i], alpha, u[i], beta, v[i]));
        }
        return result;
    }

    Point retract(const Point& point, const Tangent& tangent, double step) const {
        check_point_(point);
        check_tangent_(tangent);
        Point result;
        result.reserve(factor_count_);
        for (std::size_t i = 0; i < factor_count_; ++i) {
            result.push_back(geometry_.retract(point[i], tangent[i], step));
        }
        return result;
    }

    Tangent transport(const Point& from, const Point& to, const Tangent& tangent) const
        requires VectorTransportGeometry<ComponentGeometry>
    {
        check_point_(from);
        check_point_(to);
        check_tangent_(tangent);
        Tangent result;
        result.reserve(factor_count_);
        for (std::size_t i = 0; i < factor_count_; ++i) {
            result.push_back(geometry_.transport(from[i], to[i], tangent[i]));
        }
        return result;
    }
   private:
    void check_point_(const Point& point) const {
        if (point.size() != factor_count_) { throw std::invalid_argument("Power-geometry point size mismatch."); }
    }
    void check_tangent_(const Tangent& tangent) const {
        if (tangent.size() != factor_count_) { throw std::invalid_argument("Power-geometry tangent size mismatch."); }
    }

    std::size_t factor_count_;
    ComponentGeometry geometry_;
};

}   // namespace manifold
}   // namespace fdapde

#endif   // __FDAPDE_MANIFOLD_POWER_GEOMETRY_H__
