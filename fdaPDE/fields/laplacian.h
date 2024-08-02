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

#ifndef __LAPLACIAN_H__
#define __LAPLACIAN_H__

#include "scalar_field.h"

namespace fdapde {

template <typename Derived> class Laplacian : public ScalarBase<Derived::StaticInputSize, Laplacian<Derived>> {
   public:
    using Base = ScalarBase<Derived::StaticInputSize, Laplacian<Derived>>;
    using FunctorType = PartialDerivative<Derived, 2>;
    using InputType = typename Derived::InputType;
    using Scalar = typename Derived::Scalar;
    static constexpr int StaticInputSize = Derived::StaticInputSize;
    static constexpr int NestAsRef = 0;

    constexpr Laplacian(const Derived& xpr) : Base(), xpr_(xpr) {
        if constexpr (StaticInputSize == Dynamic) data_.resize(xpr_.input_size());
        for (int i = 0; i < xpr_.input_size(); ++i) { data_[i] = PartialDerivative<Derived, 2>(xpr_, i, i); }
    }
    constexpr Scalar operator()(const InputType& p) const {
        Scalar res = 0;
        for (int i = 0; i < xpr_.input_size(); ++i) { res += data_[i](p); }
        return res;
    }
    constexpr int input_size() const { return xpr_.input_size(); }
   private:
    using StorageType = typename std::conditional_t<
      StaticInputSize == Dynamic, std::vector<FunctorType>, std::array<FunctorType, StaticInputSize>>;
    StorageType data_;
    typename internals::ref_select<const Derived>::type xpr_;
};

template <typename XprType> constexpr Laplacian<XprType> laplacian(const XprType& xpr) {
    return Laplacian<XprType>(xpr);
}

}   // namespace fdapde

#endif   // __LAPLACIAN_H__
