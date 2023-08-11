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

#ifndef __SCALAR_EXPRESSIONS_H__
#define __SCALAR_EXPRESSIONS_H__

#include <type_traits>

#include "../utils/symbols.h"

namespace fdapde {
namespace core {

// Base class for any ScalarField type
struct ScalarBase { };

// macro for the definition of binary arithmetic operators between scalar fields
#define DEF_SCALAR_EXPR_OPERATOR(OPERATOR, FUNCTOR)                                                                    \
    template <int N, typename E1, typename E2>                                                                         \
    ScalarBinOp<N, E1, E2, FUNCTOR> OPERATOR(const ScalarExpr<N, E1>& op1, const ScalarExpr<N, E2>& op2) {             \
        return ScalarBinOp<N, E1, E2, FUNCTOR> {op1.get(), op2.get(), FUNCTOR()};                                      \
    }                                                                                                                  \
                                                                                                                       \
    template <int N, typename E>                                                                                       \
    ScalarBinOp<N, E, Scalar<N>, FUNCTOR> OPERATOR(const ScalarExpr<N, E>& op1, double op2) {                          \
        return ScalarBinOp<N, E, Scalar<N>, FUNCTOR>(op1.get(), Scalar<N>(op2), FUNCTOR());                            \
    }                                                                                                                  \
                                                                                                                       \
    template <int N, typename E>                                                                                       \
    ScalarBinOp<N, Scalar<N>, E, FUNCTOR> OPERATOR(double op1, const ScalarExpr<N, E>& op2) {                          \
        return ScalarBinOp<N, Scalar<N>, E, FUNCTOR> {Scalar<N>(op1), op2.get(), FUNCTOR()};                           \
    }                                                                                                                  \
// macro for the definition of unary operators on scalar fields
#define DEF_SCALAR_UNARY_OPERATOR(OPERATOR, FUNCTION)                                                                  \
    template <int N, typename E1>                                                                                      \
    ScalarUnOp<N, E1, std::function<double(double)>> OPERATOR(const ScalarExpr<N, E1>& op1) {                          \
        std::function<double(double)> OPERATOR_ = [](double x) -> double { return FUNCTION(x); };                      \
        return ScalarUnOp<N, E1, std::function<double(double)>> {op1.get(), OPERATOR_};                                \
    }

// forward declaration
template <int N, typename E> class ScalarNegationOp;

// Base class for scalar field expressions
template <int N, typename E> struct ScalarExpr : public ScalarBase {
    // call operator() on the base type E
    inline double operator()(const SVector<N>& p) const { return static_cast<const E&>(*this)(p); }
    const E& get() const { return static_cast<const E&>(*this); }
    // forward i to all nodes of the expression. Does nothing if not redefined in E
    template <typename T> void forward(T i) const { return; }
    // map unary operator- to a ScalarNegationOp expression node
    ScalarNegationOp<N, E> operator-() const { return ScalarNegationOp<N, E>(get()); }

    // expose compile time informations
    static constexpr int rows = 1;
    static constexpr int cols = 1;
    static constexpr int base = N;   // dimensionality of base space
};

// an expression node representing a scalar value
template <int N> class Scalar : public ScalarExpr<N, Scalar<N>> {
   private:
    double value_;
   public:
    Scalar(double value) : value_(value) { }
    // call operator, return always the stored value
    inline double operator()(const SVector<N>& p) const { return value_; };
};

// wraps an n_rows x 1 vector of data, acts as a double once forwarded the matrix row
template <int N> class ScalarDataWrapper : public ScalarExpr<N, ScalarDataWrapper<N>> {
   private:
    DMatrix<double> data_;
    double value_;
   public:
    ScalarDataWrapper() = default;
    ScalarDataWrapper(const DMatrix<double>& data) : data_(data) {};
    double operator()(const SVector<N>& p) const { return value_; }
    void forward(std::size_t i) { value_ = data_(i, 0); }   // fix value_ to the i-th coefficient of data_
};

// expression template based arithmetic
template <int N, typename OP1, typename OP2, typename BinaryOperation>
class ScalarBinOp : public ScalarExpr<N, ScalarBinOp<N, OP1, OP2, BinaryOperation>> {
   private:
    typename std::remove_reference<OP1>::type op1_;   // first  operand
    typename std::remove_reference<OP2>::type op2_;   // second operand
    BinaryOperation f_;                               // operation to apply
   public:
    // constructor
    ScalarBinOp(const OP1& op1, const OP2& op2, BinaryOperation f) : op1_(op1), op2_(op2), f_(f) {};
    // call operator, performs the expression evaluation
    double operator()(const SVector<N>& p) const { return f_(op1_(p), op2_(p)); }
    // call parameter evaluation on operands
    template <typename T> const ScalarBinOp<N, OP1, OP2, BinaryOperation>& forward(T i) {
        op1_.forward(i);
        op2_.forward(i);
        return *this;
    }
};
DEF_SCALAR_EXPR_OPERATOR(operator+, std::plus<>)
DEF_SCALAR_EXPR_OPERATOR(operator-, std::minus<>)
DEF_SCALAR_EXPR_OPERATOR(operator*, std::multiplies<>)
DEF_SCALAR_EXPR_OPERATOR(operator/, std::divides<>)

// definition of unary operation nodes
template <int N, typename OP1, typename UnaryOperation>
class ScalarUnOp : public ScalarExpr<N, ScalarUnOp<N, OP1, UnaryOperation>> {
   private:
    typename std::remove_reference<OP1>::type op1_;   // operand
    UnaryOperation f_;                                // operation to apply
   public:
    // constructor
    ScalarUnOp(const OP1& op1, UnaryOperation f) : op1_(op1), f_(f) {};
    // call operator, performs the expression evaluation
    double operator()(const SVector<N>& p) const { return f_(op1_(p)); }
};
DEF_SCALAR_UNARY_OPERATOR(sin, std::sin)
DEF_SCALAR_UNARY_OPERATOR(cos, std::cos)
DEF_SCALAR_UNARY_OPERATOR(tan, std::tan)
DEF_SCALAR_UNARY_OPERATOR(exp, std::exp)
DEF_SCALAR_UNARY_OPERATOR(log, std::log)

// unary negation operation
template <int N, typename OP> class ScalarNegationOp : public ScalarExpr<N, ScalarNegationOp<N, OP>> {
   private:
    typename std::remove_reference<OP>::type op_;
   public:
    // constructor
    ScalarNegationOp(const OP& op) : op_(op) {};
    double operator()(const SVector<N>& p) const { return -op_(p); }
    // call parameter evaluation on stored operand
    template <typename T> const ScalarNegationOp<N, OP>& forward(T i) const {
        op_.forward(i);
        return *this;
    }
};

}   // namespace core
}   // namespace fdapde

#endif   // __SCALAR_EXPRESSIONS_H__
