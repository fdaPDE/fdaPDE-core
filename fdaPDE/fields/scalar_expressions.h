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
#define DEFINE_SCALAR_BINARY_OPERATOR(OPERATOR, FUNCTOR)                                                               \
    template <int N, typename E1, typename E2>                                                                         \
    ScalarBinOp<N, E1, E2, FUNCTOR> OPERATOR(const ScalarExpr<N, E1>& op1, const ScalarExpr<N, E2>& op2) {             \
        return ScalarBinOp<N, E1, E2, FUNCTOR> {op1.get(), op2.get(), FUNCTOR()};                                      \
    }                                                                                                                  \
    template <int N, typename E>                                                                                       \
    ScalarBinOp<N, E, Scalar<N>, FUNCTOR> OPERATOR(const ScalarExpr<N, E>& op1, double op2) {                          \
        return ScalarBinOp<N, E, Scalar<N>, FUNCTOR>(op1.get(), Scalar<N>(op2, op1.get().inner_size()), FUNCTOR());    \
    }                                                                                                                  \
    template <int N, typename E>                                                                                       \
    ScalarBinOp<N, Scalar<N>, E, FUNCTOR> OPERATOR(double op1, const ScalarExpr<N, E>& op2) {                          \
        return ScalarBinOp<N, Scalar<N>, E, FUNCTOR>(Scalar<N>(op1, op2.get().inner_size()), op2.get(), FUNCTOR());    \
    }                                                                                                                  \
// macro for the definition of unary operators on scalar fields
#define DEFINE_SCALAR_UNARY_OPERATOR(OPERATOR, FUNCTION)                                                               \
    template <int N, typename E>                                                                                       \
    ScalarUnOp<N, E, std::function<double(double)>> OPERATOR(const ScalarExpr<N, E>& op1) {                            \
        std::function<double(double)> OPERATOR_ = [](double x) -> double { return FUNCTION(x); };                      \
        return ScalarUnOp<N, E, std::function<double(double)>>(op1.get(), OPERATOR_, op1.get().inner_size());          \
    }

// forward declaration
template <int N, typename E> class ScalarNegationOp;
template <int N, typename E> class ScalarExprGradient;
template <int N, typename E> class ScalarExprHessian;
  
// Base class for scalar field expressions
template <int N, typename E> class ScalarExpr : public ScalarBase {
   protected:
    int dynamic_inner_size_ = 0;   // run-time base space dimension
    double h_ = 1e-3;              // step size used in derivative approximation
   public:
    typedef typename static_dynamic_vector_selector<N>::type VectorType;
    static constexpr int rows = 1;
    static constexpr int cols = 1;
    static constexpr int static_inner_size = N;   // dimensionality of base space (can be Dynamic)

    ScalarExpr() = default;
    ScalarExpr(int dynamic_inner_size) : dynamic_inner_size_(dynamic_inner_size) { }

    // call operator() on the base type E
    inline double operator()(const VectorType& p) const { return static_cast<const E&>(*this)(p); }
    const E& get() const { return static_cast<const E&>(*this); }
    // forward i to all nodes of the expression. Does nothing if not redefined in E
    template <typename T> void forward(T i) const { return; }
    // map unary operator- to a ScalarNegationOp expression node
    ScalarNegationOp<N, E> operator-() const { return ScalarNegationOp<N, E>(get(), dynamic_inner_size_); }
    inline constexpr int inner_size() const { return (N == Dynamic) ? dynamic_inner_size_ : static_inner_size; }
    // dynamic resizing of base space dimension only allowed for Dynamic expressions
    template <int N_ = N> typename std::enable_if<N_ == Dynamic, void>::type resize(int n) { dynamic_inner_size_ = n; }

    void set_step(double h) { h_ = h; }   // set step size in derivative approximation
    ScalarExprGradient<N, E> derive() const { return ScalarExprGradient<N, E>(get(), h_); }
    ScalarExprHessian<N, E> derive_twice() const { return ScalarExprHessian<N, E>(get(), h_); }
};

// an expression node representing a scalar value
template <int N> class Scalar : public ScalarExpr<N, Scalar<N>> {
   private:
    typedef ScalarExpr<N, Scalar<N>> Base;
    double value_;
   public:
    Scalar(double value, int n) : Base(n), value_(value) { }
    inline double operator()(const typename Base::VectorType& p) const { return value_; };
};

// wraps an n_rows x 1 vector of data, acts as a double once forwarded the matrix row
template <int N> class DiscretizedScalarField : public ScalarExpr<N, DiscretizedScalarField<N>> {
   private:
    DMatrix<double, Eigen::RowMajor>* data_;
    double value_;
   public:
    DiscretizedScalarField() = default;
    DiscretizedScalarField(DMatrix<double, Eigen::RowMajor>& data) : data_(&data) {};
    double operator()(const SVector<N>& p) const { return value_; }
    void forward(Eigen::Index i) { value_ = data_->operator()(i, 0); }   // fix value_ to the i-th coefficient of data_
};

// expression template based arithmetic
template <int N, typename OP1, typename OP2, typename BinaryOperation>
class ScalarBinOp : public ScalarExpr<N, ScalarBinOp<N, OP1, OP2, BinaryOperation>> {
    static_assert(OP1::static_inner_size == OP2::static_inner_size, "you mixed fields with different base dimension");
   private:
    typedef ScalarExpr<N, ScalarBinOp<N, OP1, OP2, BinaryOperation>> Base;
    typename std::remove_reference<OP1>::type op1_;   // first  operand
    typename std::remove_reference<OP2>::type op2_;   // second operand
    BinaryOperation f_;                               // operation to apply
   public:
    static constexpr int static_inner_size = OP1::static_inner_size;
    // constructor
    ScalarBinOp(const OP1& op1, const OP2& op2, BinaryOperation f) :
        Base(op1.inner_size()), op1_(op1), op2_(op2), f_(f) {
        if constexpr (N == Dynamic) { fdapde_assert(op1_.inner_size() == op2_.inner_size()); }
    };
    double operator()(const typename Base::VectorType& p) const {
        if constexpr (N == Dynamic) { fdapde_assert(p.rows() == Base::inner_size()); }
        return f_(op1_(p), op2_(p));
    }
    // forward to child nodes
    template <typename T> const ScalarBinOp<N, OP1, OP2, BinaryOperation>& forward(T i) {
        op1_.forward(i); op2_.forward(i);
        return *this;
    }
};
DEFINE_SCALAR_BINARY_OPERATOR(operator+, std::plus<>)
DEFINE_SCALAR_BINARY_OPERATOR(operator-, std::minus<>)
DEFINE_SCALAR_BINARY_OPERATOR(operator*, std::multiplies<>)
DEFINE_SCALAR_BINARY_OPERATOR(operator/, std::divides<>)

// definition of unary operation nodes
template <int N, typename OP1, typename UnaryOperation>
class ScalarUnOp : public ScalarExpr<N, ScalarUnOp<N, OP1, UnaryOperation>> {
   private:
    typedef ScalarExpr<N, ScalarUnOp<N, OP1, UnaryOperation>> Base;
    typename std::remove_reference<OP1>::type op1_;   // operand
    UnaryOperation f_;                                // operation to apply
   public:
    // constructor
    ScalarUnOp(const OP1& op1, UnaryOperation f, int n) : Base(n), op1_(op1), f_(f) {};
    double operator()(const typename Base::VectorType& p) const { return f_(op1_(p)); }
};
DEFINE_SCALAR_UNARY_OPERATOR(sin, std::sin)
DEFINE_SCALAR_UNARY_OPERATOR(cos, std::cos)
DEFINE_SCALAR_UNARY_OPERATOR(tan, std::tan)
DEFINE_SCALAR_UNARY_OPERATOR(exp, std::exp)
DEFINE_SCALAR_UNARY_OPERATOR(log, std::log)

// unary negation operation
template <int N, typename OP> class ScalarNegationOp : public ScalarExpr<N, ScalarNegationOp<N, OP>> {
   private:
    typedef ScalarExpr<N, ScalarNegationOp<N, OP>> Base;
    typename std::remove_reference<OP>::type op_;
   public:
    // constructor
    ScalarNegationOp(const OP& op, int n) : Base(n), op_(op) {};  
    double operator()(const typename Base::VectorType& p) const { return -op_(p); }
    // call parameter evaluation on stored operand
    template <typename T> const ScalarNegationOp<N, OP>& forward(T i) {
        op_.forward(i);
        return *this;
    }
};

}   // namespace core
}   // namespace fdapde

#endif   // __SCALAR_EXPRESSIONS_H__
