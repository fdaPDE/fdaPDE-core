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

#ifndef __DOT_PRODUCT_H__
#define __DOT_PRODUCT_H__

#include "../utils/symbols.h"
#include "../utils/traits.h"
#include "scalar_expressions.h"

namespace fdapde {
namespace core {

// A functor to represent an inner product between two vector expressions T1 and T2.
template <int N, typename T1, typename T2> class DotProduct : public ScalarExpr<N, DotProduct<N, T1, T2>> {
   private:
    T1 op1_; T2 op2_;
    typedef typename static_dynamic_vector_selector<N>::type InnerVectorType;
    static_assert(
      // both are dynamic expressions
      (T1::static_inner_size == Dynamic && T2::static_inner_size == Dynamic) ||
      // or they are both static and have the same number of rows
      (T1::static_inner_size != Dynamic && T2::static_inner_size != Dynamic) &&
      ((T1::cols == T2::cols == 1) && (T1::rows == T2::rows)) || (T1::cols == T2::rows) || (T1::rows == T2::cols));

    int dot_product_outer_size() const {
        if (N == Dynamic) return op1_.outer_size();
        if (T1::cols == T2::cols == 1) {
            return T1::rows;
        } else {
            return (T1::cols == T2::rows) ? T1::cols : T1::rows;
        }
    };
    // value evaluates to true if T.operator[](std::size_t) returns a double
    template <typename T> struct subscript_to_double {
        static constexpr bool value = std::is_same<typename subscript_result_of<T, std::size_t>::type, double>::value;
    };
   public:
    // constructor
    DotProduct(const T1& op1, const T2& op2) : op1_(op1), op2_(op2) { }
    inline double operator()(const InnerVectorType& x) const {
        if constexpr (N == Dynamic) {
            fdapde_assert(
              (x.rows() == op1_.inner_size() && x.rows() == op2_.inner_size() &&
               op1_.outer_size() == op2_.outer_size()));
        }
        double result = 0;
        for (int i = 0; i < dot_product_outer_size(); ++i) {
            if constexpr (!subscript_to_double<T1>::value && !subscript_to_double<T2>::value) {
                result += (op1_[i] * op2_[i])(x);
            } else {
                if constexpr (subscript_to_double<T1>::value && !subscript_to_double<T2>::value) {
                    result += op1_[i] * op2_[i](x);
                } else {
                    result += op1_[i](x) * op2_[i];
                }
            }
        }
        return result;
    }
    template <typename T> const DotProduct<N, T1, T2>& forward(T i) {
        op1_.forward(i);
        op2_.forward(i);
        return *this;
    }
};

}   // namespace core
}   // namespace fdapde

#endif   // __DOT_PRODUCT_H__
