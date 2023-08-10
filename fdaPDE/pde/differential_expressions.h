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

#ifndef __DIFFERENTIAL_OPERATORS_EXPRESSIONS_H__
#define __DIFFERENTIAL_OPERATORS_EXPRESSIONS_H__

#include <tuple>

#include "../mesh/element.h"
#include "../utils/symbols.h"
#include "../utils/traits.h"

namespace fdapde {
namespace core {

#define FDAPDE_DEFINE_DIFFERENTIAL_EXPR_OPERATOR(OPERATOR, FUNCTOR)                                                    \
    template <typename E1, typename E2>                                                                                \
    DifferentialBinOp<E1, E2, FUNCTOR> OPERATOR(const DifferentialExpr<E1>& op1, const DifferentialExpr<E2>& op2) {    \
        return DifferentialBinOp<E1, E2, FUNCTOR> {op1.get(), op2.get(), FUNCTOR()};                                   \
    }

// forward declaration
template <typename E> class DifferentialNegateOp;

// base type for differential operator expression templates
template <typename E> struct DifferentialExpr {
    // call integration method on base type
    template <typename... Args> auto integrate(const std::tuple<Args...>& mem_buffer) const {
        return static_cast<const E&>(*this).integrate(mem_buffer);
    }
    const E& get() const { return static_cast<const E&>(*this); }
    // returns the set of types associated with this expression
    std::tuple<E> get_operator_type() const { return std::make_tuple(static_cast<const E&>(*this)); }
    static constexpr bool is_space_varying = E::is_space_varying;
    static constexpr bool is_symmetric = E::is_symmetric;
    // map operator- to a DifferentialNegateOp node
    DifferentialNegateOp<E> operator-() const { return DifferentialNegateOp<E>(get()); }
};

// a generic binary operation node
template <typename OP1, typename OP2, typename BinaryOperation>
class DifferentialBinOp : public DifferentialExpr<DifferentialBinOp<OP1, OP2, BinaryOperation>> {
   private:
    typedef typename std::remove_reference<OP1>::type OP1_;
    typedef typename std::remove_reference<OP2>::type OP2_;
    OP1_ op1_;            // first  operand
    OP2_ op2_;            // second operand
    BinaryOperation f_;   // operation to apply
   public:
    // constructor
    DifferentialBinOp() = default;   // let default constructible to allow assignment
    DifferentialBinOp(const OP1& op1, const OP2& op2, BinaryOperation f) : op1_(op1), op2_(op2), f_(f) {};
    // integrate method. Apply the functor f_ to the result of integrate() applied to both operands.
    template <typename... Args> auto integrate(const std::tuple<Args...>& mem_buffer) const {
        return f_(op1_.integrate(mem_buffer), op2_.integrate(mem_buffer));
    }
    auto get_operator_type() const { return std::tuple_cat(op1_.get_operator_type(), op2_.get_operator_type()); }
    enum {
        is_space_varying = OP1::is_space_varying || OP2::is_space_varying,
        is_symmetric = OP1::is_symmetric && OP2::is_symmetric
    };

    // returns a bilinear form obtained from this one by removing the effect of operator T
    template <template <typename> typename T> auto remove_operator() const {
        // if the bilinear form has no operator T, return a reference to this form as it is
        if constexpr (!has_instance_of<T, decltype(get_operator_type())>::value) return *this;
        // end of recursion (leafs of the expression tree reached)
        else if constexpr (is_instance_of<OP1_, T>::value)
            return op2_;
        else if constexpr (is_instance_of<OP2_, T>::value)
            return op1_;
        // only one of the two operands is an internal expression node
        else if constexpr (
          is_instance_of<OP1_, DifferentialBinOp>::value && !is_instance_of<OP2_, DifferentialBinOp>::value)
            return f_(op1_.template remove_operator<T>(), op2_);
        else if constexpr (
          !is_instance_of<OP1_, DifferentialBinOp>::value && is_instance_of<OP2_, DifferentialBinOp>::value)
            return f_(op1_, op2_.template remove_operator<T>());
        else   // both operands are expression nodes
            return f_(op1_.template remove_operator<T>(), op2_.template remove_operator<T>());
    }
};
FDAPDE_DEFINE_DIFFERENTIAL_EXPR_OPERATOR(operator+, std::plus<>)
FDAPDE_DEFINE_DIFFERENTIAL_EXPR_OPERATOR(operator-, std::minus<>)

// node representing a single scalar in an expression
class DifferentialScalar : public DifferentialExpr<DifferentialScalar> {
   private:
    double value_;
   public:
    // constructor
    DifferentialScalar() = default;
    DifferentialScalar(double value) : value_(value) { }
    // integrate method returns the stored value
    template <typename... Args> auto integrate(const std::tuple<Args...>& mem_buffer) const { return value_; }
    std::tuple<DifferentialScalar> get_operator_type() const { return std::make_tuple(*this); }
    enum { is_space_varying = false, is_symmetric = true };
};
// allow scalar*operator expressions
template <typename E>
DifferentialBinOp<DifferentialScalar, E, std::multiplies<>> operator*(double op1, const DifferentialExpr<E>& op2) {
    return DifferentialBinOp<DifferentialScalar, E, std::multiplies<>>(
      DifferentialScalar(op1), op2.get(), std::multiplies<>());
}

// unary negation node
template <typename OP> class DifferentialNegateOp : public DifferentialExpr<DifferentialNegateOp<OP>> {
   private:
    typedef typename std::remove_reference<OP>::type OP_;
    OP_ op_;
   public:
    // constructor
    DifferentialNegateOp() = default;
    DifferentialNegateOp(const OP& op) : op_(op) {};
    // integrate method negates the result of integrate()
    template <typename... Args> auto integrate(const std::tuple<Args...>& mem_buffer) const {
        return -(op_.integrate(mem_buffer));
    }
    auto get_operator_type() const { return op_.get_operator_type(); }
    enum { is_space_varying = OP::is_space_varying, is_symmetric = OP::is_symmetric };
};

}   // namespace core
}   // namespace fdapde

#endif   // __DIFFERENTIAL_OPERATORS_EXPRESSIONS_H__
