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

#ifndef __FDAPDE_LINALG_QUATERNION_H__
#define __FDAPDE_LINALG_QUATERNION_H__

#include "header_check.h"
#include "kernels/so3.h"

namespace fdapde {

/// @brief owns a unit quaternion representing an orientation in three dimensions
/// @details scalar-first Hamilton coefficients compose in the same order as rotation matrix products
template <typename S> class UnitQuaternion {
   public:
    static_assert(
      std::is_floating_point_v<S> && std::same_as<S, std::remove_cv_t<S>>,
      "quaternion scalar must be unqualified floating point");
    using Scalar = S;
    /// @brief constructs the identity orientation
    UnitQuaternion() = default;
    /// @brief normalizes finite nonzero Hamilton coefficients with overflow-safe scaling
    UnitQuaternion(S real, S x, S y, S z) :
        data_(internals::normalized_quaternion(std::array<S, 4> {real, x, y, z})) { }
    /// @brief borrows read-only scalar-first unit coefficients
    const std::array<S, 4>& coefficients() const& { return data_; }
    /// @brief rejects coefficient references escaping a temporary quaternion
    void coefficients() const&& = delete;
    /// @brief constructs a principal-hemisphere quaternion from a verified three-dimensional rotation
    template <RotationLike Q> static UnitQuaternion from_rotation(const Q& rotation) {
        fdapde_strong_assert(rotation.rows() == 3, std::invalid_argument, "quaternion rotation must have order three");
        const auto value = internals::so3_quaternion(rotation);
        return UnitQuaternion(value[0], value[1], value[2], value[3]);
    }
    /// @brief maps a finite three-component axis-angle vector to a unit quaternion
    template <typename Xpr> static UnitQuaternion from_rotation_vector(const MatrixExpr<Xpr>& vector) {
        fdapde_strong_assert(
          vector.rows() == 3 && vector.cols() == 1, std::invalid_argument,
          "quaternion rotation vector must have three components");
        const auto value = internals::so3_quaternion_exp(
          std::array<S, 3> {S(vector.derived()(0, 0)), S(vector.derived()(1, 0)), S(vector.derived()(2, 0))});
        return UnitQuaternion(value[0], value[1], value[2], value[3]);
    }
    /// @brief materializes a verified rotation with independently selected matrix caches
    template <typename P = RotationCache::None> auto rotation() const {
        return RotationMatrix<S, 3, P>(internals::so3_quaternion_matrix(data_));
    }
    /// @brief returns the conjugate quaternion representing the inverse orientation
    UnitQuaternion inverse() const { return UnitQuaternion(data_[0], -data_[1], -data_[2], -data_[3]); }
    /// @brief composes orientations through the Hamilton product and restores unit normalization
    UnitQuaternion operator*(const UnitQuaternion& right) const {
        const auto& a = data_;
        const auto& b = right.data_;
        return UnitQuaternion(
          a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3], a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
          a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1], a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]);
    }
   private:
    std::array<S, 4> data_ {S(1), S(0), S(0), S(0)};
};
}   // namespace fdapde
#endif
