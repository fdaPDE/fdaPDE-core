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

#ifndef __GRADIENT_H__
#define __GRADIENT_H__

#include "matrix_field.h"
#include "scalar_field.h"

namespace fdapde {

template <typename Derived_> class Gradient : public fdapde::MatrixBase<Derived_::StaticInputSize, Gradient<Derived_>> {
   public:
    using Derived = Derived_;
    using Base = MatrixBase<Derived::StaticInputSize, Gradient<Derived>>;
    using FunctorType = PartialDerivative<std::decay_t<Derived>, 1>;
    using InputType = typename FunctorType::InputType;
    using Scalar = typename FunctorType::Scalar;
    static constexpr int StaticInputSize = Derived::StaticInputSize;
    static constexpr int Rows = Derived::StaticInputSize;
    static constexpr int Cols = 1;
    static constexpr int NestAsRef = 0;
    static constexpr int XprBits = FunctorType::XprBits;
    using Base::operator();

    explicit constexpr Gradient(const Derived& xpr) : Base(), xpr_(xpr) {
        if constexpr (StaticInputSize == Dynamic) data_.resize(xpr_.input_size());
        for (int i = 0; i < xpr_.input_size(); ++i) { data_[i] = PartialDerivative<std::decay_t<Derived>, 1>(xpr_, i); }
    }
    // getters
    constexpr const PartialDerivative<std::decay_t<Derived>, 1>& operator()(int i, int j) {
        return data_[i * cols() + j];
    }
    constexpr const PartialDerivative<std::decay_t<Derived>, 1>& operator[](int i) { return data_[i]; }
    constexpr Scalar eval(int i, int j, const InputType& p) const { return data_[i * cols() + j](p); }
    constexpr Scalar eval(int i, const InputType& p) const { return data_[i](p); }
    constexpr int rows() const { return Rows; }
    constexpr int cols() const { return Cols; }
    constexpr int input_size() const { return xpr_.input_size(); }
    constexpr int size() const { return Rows; }
    constexpr const Derived& derived() const { return xpr_; }
   private:
    using StorageType = typename std::conditional_t<
      Derived::StaticInputSize == Dynamic, std::vector<FunctorType>, std::array<FunctorType, StaticInputSize>>;
    StorageType data_;
    typename internals::ref_select<Derived>::type xpr_;
};

template <typename XprType> constexpr Gradient<XprType> grad(const XprType& xpr) { return Gradient<XprType>(xpr); }

}   // namespace fdapde

#endif // __GRADIENT_H__
