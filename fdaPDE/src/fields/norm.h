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

#ifndef __FDAPDE_NORM_H__
#define __FDAPDE_NORM_H__

#include "header_check.h"

namespace fdapde {

template <int Order, typename Derived_, int PowerFlag = 0>
class MatrixFieldNorm : public ScalarFieldBase<Derived_::StaticInputSize, MatrixFieldNorm<Order, Derived_, PowerFlag>> {
   public:
    using Derived = Derived_;
    template <typename T> using Meta = MatrixFieldNorm<Order, T, PowerFlag>;
    using Base = ScalarFieldBase<Derived::StaticInputSize, MatrixFieldNorm<Order, Derived, PowerFlag>>;
    using InputType = typename Derived::InputType;
    using Scalar = typename Derived::Scalar;
    static constexpr int StaticInputSize = Derived::StaticInputSize;
    static constexpr int NestAsRef = 0;
    static constexpr int XprBits = Derived::XprBits;

    explicit constexpr MatrixFieldNorm(const Derived& xpr) : Base(), xpr_(xpr) { }
    constexpr Scalar operator()(const InputType& p) const {
        if constexpr (StaticInputSize == Dynamic) fdapde_assert(p.size() == xpr_.input_size());
        Scalar norm_ = 0;
        for (int i = 0; i < xpr_.rows(); ++i) {
            for (int j = 0; j < xpr_.cols(); ++j) {
                Scalar coeff = xpr_.eval(i, j, p);
                if constexpr (Order == 1) norm_ += std::abs(coeff);
                if constexpr (Order == 2) norm_ += coeff * coeff;
                if constexpr (Order >= 3) norm_ += std::pow(coeff, Order);
            }
        }
	if constexpr (PowerFlag || Order == 1) return norm_; 
	// compute p-th square root otherwise
        if constexpr (Order == 2) return std::sqrt(norm_);
        if constexpr (Order >= 3) return std::pow(norm_, 1. / Order);
    }
    constexpr int input_size() const { return xpr_.input_size(); }
    constexpr const Derived& derived() const { return xpr_; }
   private:
    internals::ref_select_t<const Derived> xpr_;
};

}   // namespace fdapde

#endif // __FDAPDE_NORM_H__
