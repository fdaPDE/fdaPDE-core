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

#ifndef __FDAPDE_JACOBIAN_H__
#define __FDAPDE_JACOBIAN_H__

#include "header_check.h"

namespace fdapde {

template <typename Derived_> class Jacobian : public MatrixFieldBase<Derived_::StaticInputSize, Jacobian<Derived_>> {
    fdapde_static_assert(Derived_::Cols == 1, JACOBIAN_OPERATOR_IS_FOR_VECTOR_FIELDS_ONLY);
   public:
    using Derived = Derived_;
    template <typename T> using Meta = Jacobian<T>;
    using Base = MatrixFieldBase<Derived::StaticInputSize, Jacobian<Derived>>;
    using FunctorType = PartialDerivative<
      internals::xpr_or_scalar_wrap_t<
        Derived, Derived::StaticInputSize,
        std::decay_t<decltype(std::declval<Derived_>().operator[](std::declval<int>()))>>,
      1>;
    using InputType = typename Derived::InputType;
    using Scalar = typename Derived::Scalar;
    static constexpr int StaticInputSize = Derived::StaticInputSize;
    static constexpr int Rows = Derived::StaticInputSize;
    static constexpr int Cols = Derived::Rows;
    static constexpr int NestAsRef = 0;
    static constexpr int XprBits = Derived::XprBits;

    explicit constexpr Jacobian(const Derived_& xpr) : Base(), xpr_(xpr) {
        if constexpr (StaticInputSize == Dynamic) data_.resize(xpr_.rows() * xpr_.input_size());
        for (int i = 0; i < xpr_.input_size(); ++i) {
            for (int j = 0; j < xpr_.rows(); ++j) { data_[i * Cols + j] = FunctorType(xpr_[j], i); }
        }
    }
    // getters
    constexpr const FunctorType& operator()(int i, int j) { return data_[i * Cols + j]; }
    constexpr Scalar eval(int i, int j, const InputType& p) const { return data_[i * Cols + j](p); }
    constexpr int rows() const { return xpr_.input_size(); }
    constexpr int cols() const { return xpr_.rows(); }
    constexpr int input_size() const { return xpr_.input_size(); }
    constexpr int size() const { return rows() * cols(); }
    constexpr const Derived& derived() const { return xpr_; }
    // evaluation at point
    constexpr auto operator()(const InputType& p) const { return Base::call_(p); }
   private:
    using StorageType = typename std::conditional_t<
      Derived::StaticInputSize == Dynamic, std::vector<FunctorType>, std::array<FunctorType, Rows * Cols>>;
    StorageType data_;
    internals::ref_select_t<const Derived> xpr_;
};

template <typename XprType>
Jacobian<XprType> constexpr grad(const XprType& xpr)
    requires(internals::is_matrix_field_v<XprType> && XprType::Cols == 1) {
    return Jacobian<XprType>(xpr);
}

}   // namespace fdapde

#endif // __FDAPDE_JACOBIAN_H__
