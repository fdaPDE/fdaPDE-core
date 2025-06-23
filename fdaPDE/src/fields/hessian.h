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

#ifndef __FDAPDE_HESSIAN_H__
#define __FDAPDE_HESSIAN_H__

#include "header_check.h"

namespace fdapde {

template <typename Derived_> class Hessian : public MatrixFieldBase<Derived_::StaticInputSize, Hessian<Derived_>> {
   public:
    using Derived = std::decay_t<Derived_>;
    template <typename T> using Meta = Hessian<T>;
    using Base = MatrixFieldBase<Derived::StaticInputSize, Hessian<Derived>>;
    using FunctorType =
      PartialDerivative<internals::xpr_or_scalar_wrap_t<Derived, Derived::StaticInputSize, Derived>, 2>;
    using InputType = typename Derived::InputType;
    using Scalar = typename Derived::Scalar;
    static constexpr int StaticInputSize = Derived::StaticInputSize;
    static constexpr int Rows = Derived::StaticInputSize;
    static constexpr int Cols = Derived::StaticInputSize;
    static constexpr int NestAsRef = 0;
    static constexpr int XprBits = Derived::XprBits;

    explicit constexpr Hessian(const Derived_& xpr) : Base(), derived_(xpr), xpr_(), data_() {
        if constexpr (StaticInputSize == Dynamic) data_.resize(xpr.input_size(), xpr.input_size(), xpr.input_size());
        for (int i = 0; i < xpr.input_size(); ++i) {
            for (int j = 0; j <= i; ++j) { data_(i, j) = PartialDerivative<std::decay_t<Derived_>, 2>(xpr, i, j); }
        }
	xpr_ = data_.template symmetric_view<Lower>();
    }
    // getters
    constexpr const FunctorType& operator()(int i, int j) { return xpr_(i, j); }
    constexpr Scalar eval(int i, int j, const InputType& p) const { return xpr_.eval(i, j, p); }
    template <typename Dest> constexpr void eval_at(const InputType& p, Dest& dest) const {
        xpr_.eval_at(p, dest);
        return;
    }
    constexpr int rows() const { return data_.rows(); }
    constexpr int cols() const { return data_.cols(); }
    constexpr int input_size() const { return data_.input_size(); }
    constexpr int size() const { return data_.size(); }
    constexpr const Derived& derived() const { return derived_; }
    // evaluation at point
    constexpr auto operator()(const InputType& p) const { return Base::call_(p); }
   private:
    internals::ref_select_t<Derived> derived_;
    MatrixFieldSymmetricView<MatrixField<StaticInputSize, Rows, Cols, FunctorType>, Lower> xpr_;
    MatrixField<StaticInputSize, Rows, Cols, FunctorType> data_;
};

// symbolic function objects
// 1D spaces
template <typename XprType> constexpr auto dxx(const XprType& xpr) {
    fdapde_static_assert(XprType::StaticInputSize >= 1, THIS_FUNCTION_IS_FOR_ONE_OR_HIGHER_DIMENSIONAL_SPACES_ONLY);
    return Hessian<XprType>(xpr)(0, 0);
}
// 2D spaces
template <typename XprType> constexpr auto dyy(const XprType& xpr) {
    fdapde_static_assert(XprType::StaticInputSize >= 2, THIS_FUNCTION_IS_FOR_TWO_OR_HIGHER_DIMENSIONAL_SPACES_ONLY);
    return Hessian<XprType>(xpr)(1, 1);
}
template <typename XprType> constexpr auto dxy(const XprType& xpr) {
    fdapde_static_assert(XprType::StaticInputSize >= 2, THIS_FUNCTION_IS_FOR_TWO_OR_HIGHER_DIMENSIONAL_SPACES_ONLY);
    return Hessian<XprType>(xpr)(0, 1);
}
template <typename XprType> constexpr auto dyx(const XprType& xpr) {
    fdapde_static_assert(XprType::StaticInputSize >= 2, THIS_FUNCTION_IS_FOR_TWO_OR_HIGHER_DIMENSIONAL_SPACES_ONLY);
    return Hessian<XprType>(xpr)(1, 0);
}
// 3D spaces
template <typename XprType> constexpr auto dzz(const XprType& xpr) {
    fdapde_static_assert(XprType::StaticInputSize == 3, THIS_FUNCTION_IS_FOR_THREE_DIMENSIONAL_SPACES_ONLY);
    return Hessian<XprType>(xpr)(2, 2);
}
template <typename XprType> constexpr auto dxz(const XprType& xpr) {
    fdapde_static_assert(XprType::StaticInputSize == 3, THIS_FUNCTION_IS_FOR_THREE_DIMENSIONAL_SPACES_ONLY);
    return Hessian<XprType>(xpr)(0, 2);
}
template <typename XprType> constexpr auto dzx(const XprType& xpr) {
    fdapde_static_assert(XprType::StaticInputSize == 3, THIS_FUNCTION_IS_FOR_THREE_DIMENSIONAL_SPACES_ONLY);
    return Hessian<XprType>(xpr)(2, 0);
}
template <typename XprType> constexpr auto dyz(const XprType& xpr) {
    fdapde_static_assert(XprType::StaticInputSize == 3, THIS_FUNCTION_IS_FOR_THREE_DIMENSIONAL_SPACES_ONLY);
    return Hessian<XprType>(xpr)(1, 2);
}
template <typename XprType> constexpr auto dzy(const XprType& xpr) {
    fdapde_static_assert(XprType::StaticInputSize == 3, THIS_FUNCTION_IS_FOR_THREE_DIMENSIONAL_SPACES_ONLY);
    return Hessian<XprType>(xpr)(2, 1);
}

}   // namespace fdapde

#endif // __FDAPDE_HESSIAN_H__
