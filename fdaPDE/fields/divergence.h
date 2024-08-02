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

#ifndef __DIVERGENCE_H__
#define __DIVERGENCE_H__

#include "scalar_field.h"

namespace fdapde {

template <typename Derived> class Divergence : public ScalarBase<Derived::StaticInputSize, Divergence<Derived>> {
    fdapde_static_assert(Derived::Cols == 1, DIVERGENCE_OPERATOR_IS_FOR_VECTOR_FIELDS_ONLY);
   public:
    using Base = ScalarBase<Derived::StaticInputSize, Divergence<Derived>>;
    using FunctorType =
      PartialDerivative<std::decay_t<decltype(std::declval<Derived>().operator[](std::declval<int>()))>, 1>;
    using InputType = typename Derived::InputType;
    using Scalar = typename Derived::Scalar;
    static constexpr int StaticInputSize = Derived::StaticInputSize;
    static constexpr int NestAsRef = 0;

    explicit constexpr Divergence(const Derived& xpr) : Base(), data_(), xpr_(xpr) {
        if constexpr (StaticInputSize == Dynamic) data_.resize(xpr_.rows());
        for (int i = 0; i < xpr_.rows(); ++i) { data_[i] = FunctorType(xpr_[i], i); }
    }
    constexpr Scalar operator()(const InputType& p) const {
        Scalar div_ = 0;
        for (int i = 0; i < xpr_.rows(); ++i) { div_ += data_[i](p); }
        return div_;
    }
    constexpr int input_size() const { return xpr_.input_size(); }
   private:
    using StorageType = typename std::conditional<
      Derived::StaticInputSize == Dynamic, std::vector<FunctorType>, std::array<FunctorType, StaticInputSize>>::type;
    StorageType data_;
    typename internals::ref_select<Derived>::type xpr_;
};

template <typename XprType> Divergence<XprType> constexpr div(const XprType& xpr) { return Divergence<XprType>(xpr); }

}   // namespace fdapde

#endif // __DIVERGENCE_H__
