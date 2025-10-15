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

#ifndef __FDAPDE_GRADIENT_H__
#define __FDAPDE_GRADIENT_H__

#include "header_check.h"

namespace fdapde {

template <typename Derived_> class Gradient : public MatrixFieldBase<Derived_::StaticInputSize, Gradient<Derived_>> {
   public:
    using Derived = std::decay_t<Derived_>;
    template <typename T> using Meta = Gradient<T>;
    using Base = MatrixFieldBase<Derived::StaticInputSize, Gradient<Derived>>;
    using FunctorType =
      PartialDerivative<internals::xpr_or_scalar_wrap_t<Derived, Derived::StaticInputSize, Derived>, 1>;
    using InputType = typename FunctorType::InputType;
    using Scalar = typename FunctorType::Scalar;
    static constexpr int StaticInputSize = Derived::StaticInputSize;
    static constexpr int Rows = Derived::StaticInputSize;
    static constexpr int Cols = 1;
    static constexpr int NestAsRef = 0;
    static constexpr int XprBits = FunctorType::XprBits;

    explicit constexpr Gradient(const Derived& xpr) : Base(), xpr_(xpr) {
        if constexpr (StaticInputSize == Dynamic) data_.resize(xpr_.input_size());
        for (int i = 0; i < xpr_.input_size(); ++i) { data_[i] = PartialDerivative<std::decay_t<Derived>, 1>(xpr_, i); }
    }
    // getters
    constexpr const FunctorType& operator()(int i, int j) { return data_[i * cols() + j]; }
    constexpr const FunctorType& operator[](int i) { return data_[i]; }
    constexpr Scalar eval(int i, int j, const InputType& p) const { return data_[i * cols() + j](p); }
    constexpr Scalar eval(int i, const InputType& p) const { return data_[i](p); }
    constexpr int rows() const { return xpr_.input_size(); }
    constexpr int cols() const { return Cols; }
    constexpr int input_size() const { return xpr_.input_size(); }
    constexpr int size() const { return Rows; }
    constexpr const Derived& derived() const { return xpr_; }
    // evaluation at point
    constexpr auto operator()(const InputType& p) const { return Base::call_(p); }
   private:
    using StorageType = std::conditional_t<
      StaticInputSize == Dynamic, std::vector<FunctorType>,
      std::array<FunctorType, StaticInputSize >= 0 ? StaticInputSize : 0>>;   // avoid -Wc++11-narrowing
    StorageType data_;
    internals::ref_select_t<Derived> xpr_;
};

template <typename XprType>
constexpr Gradient<XprType> grad(const XprType& xpr) requires(internals::is_scalar_field_v<XprType>) {
    return Gradient<XprType>(xpr);
}
// symbolic function objects
// 1D spaces
template <typename XprType> constexpr auto dx(const XprType& xpr) {
    fdapde_static_assert(XprType::StaticInputSize >= 1, THIS_FUNCTION_IS_FOR_ONE_OR_HIGHER_DIMENSIONAL_SPACES_ONLY);
    return Gradient<XprType>(xpr)[0];
}
// 2D spaces
template <typename XprType> constexpr auto dy(const XprType& xpr) {
    fdapde_static_assert(XprType::StaticInputSize >= 2, THIS_FUNCTION_IS_FOR_TWO_OR_HIGHER_DIMENSIONAL_SPACES_ONLY);
    return Gradient<XprType>(xpr)[1];
}
// 3D spaces
template <typename XprType> constexpr auto dz(const XprType& xpr) {
    fdapde_static_assert(XprType::StaticInputSize == 3, THIS_FUNCTION_IS_FOR_THREE_DIMENSIONAL_SPACES_ONLY);
    return Gradient<XprType>(xpr)[2];
}

  
}   // namespace fdapde

#endif // __FDAPDE_GRADIENT_H__
