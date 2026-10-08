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

#ifndef __FDAPDE_MANIFOLD_GEOMETRY_COMMON_H__
#define __FDAPDE_MANIFOLD_GEOMETRY_COMMON_H__

#include "../header_check.h"

namespace fdapde::manifold::internals {

/// @brief rejects nonfinite arithmetic results before returning an owning value
template <typename Scalar_> Scalar_ checked_geometry_result(Scalar_ value) {
    if (!std::isfinite(value)) throw std::domain_error("SPD geometry produced a nonfinite result.");
    return value;
}

/// @brief checks a public double coefficient before narrowing to the geometry scalar
template <typename Scalar_> Scalar_ geometry_coefficient(double value) {
    if (!std::isfinite(value) || std::abs(static_cast<long double>(value)) > std::numeric_limits<Scalar_>::max()) {
        throw std::invalid_argument("SPD geometry coefficient is not finite or representable.");
    }
    return static_cast<Scalar_>(value);
}

}   // namespace fdapde::manifold::internals

#endif   // __FDAPDE_MANIFOLD_GEOMETRY_COMMON_H__
