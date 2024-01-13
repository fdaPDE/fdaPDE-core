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

#ifndef __VECTOR_EXPRESSIONS_H__
#define __VECTOR_EXPRESSIONS_H__

#include <functional>

#include "../utils/symbols.h"
#include "../utils/assert.h"

namespace fdapde {
namespace core {

// Base class for any VectorField type
struct VectorBase { };

#define DEF_VECT_EXPR_OPERATOR(OPERATOR, FUNCTOR)                                                                      \
    template <int M, int N, typename E1, typename E2>                                                                  \
    VectorBinOp<M, N, E1, E2, FUNCTOR> OPERATOR(const VectorExpr<M, N, E1>& op1, const VectorExpr<M, N, E2>& op2) {    \
        return VectorBinOp<M, N, E1, E2, FUNCTOR>(op1.get(), op2.get(), FUNCTOR());                                    \
    }                                                                                                                  \
    template <int M, int N, typename E>                                                                                \
    VectorBinOp<M, N, VectorConst<M, N>, E, FUNCTOR> OPERATOR(SVector<N> op1, const VectorExpr<M, N, E>& op2) {        \
        return VectorBinOp<M, N, VectorConst<M, N>, E, FUNCTOR>(                                                       \
          VectorConst<M, N>(op1, op2.inner_size(), op2.outer_size()), op2.get(), FUNCTOR());                           \
    }                                                                                                                  \
    template <int M, int N, typename E>                                                                                \
    VectorBinOp<M, N, E, VectorConst<M, N>, FUNCTOR> OPERATOR(const VectorExpr<M, N, E>& op1, SVector<N> op2) {        \
        return VectorBinOp<M, N, E, VectorConst<M, N>, FUNCTOR>(                                                       \
          op1.get(), VectorConst<M, N>(op2, op1.inner_size(), op1.outer_size()), FUNCTOR());                           \
    }

// forward declarations
template <int M, typename T1, typename T2> class DotProduct;
template <int M, int N> class VectorConst;
template <int M, int N, typename E> class VectorNegationOp;

// Base class for vectorial expressions
template <int M, int N, typename E> class VectorExpr : public VectorBase {
   protected:
    int inner_size_ = M;   // \mathbb{R}^M
    int outer_size_ = N;   // \mathbb{R}^N
   public:
    typedef typename static_dynamic_vector_selector<M>::type InnerVectorType;
    typedef typename static_dynamic_vector_selector<N>::type OuterVectorType;
    static constexpr int rows = N;
    static constexpr int cols = 1;
    static constexpr int static_inner_size = M;

    VectorExpr() = default;
    VectorExpr(int inner_size, int outer_size) : inner_size_(inner_size), outer_size_(outer_size) {};

    // call operator[] on the base type E
    auto operator[](std::size_t i) const { return static_cast<const E&>(*this)[i]; }
    const E& get() const { return static_cast<const E&>(*this); }
    inline constexpr int inner_size() const { return (M == Dynamic) ? inner_size_ : static_inner_size; }
    inline constexpr int outer_size() const { return (N == Dynamic) ? outer_size_ : rows; }
    // evaluate the expression at point x
    OuterVectorType operator()(const InnerVectorType& x) const {
        if constexpr (M == Dynamic) fdapde_assert(inner_size_ == x.size());
        OuterVectorType result(outer_size());
        for (size_t i = 0; i < outer_size_; ++i) { result[i] = operator[](i)(x); }
        return result;
    }
    // VectorExpr - InnerVectorType dot product
    DotProduct<M, E, VectorConst<M, N>> dot(const InnerVectorType& op) const {
        return DotProduct<M, E, VectorConst<M, N>>(get(), VectorConst<M, N>(op, inner_size(), outer_size()));
    }
    // VectorExpr - VectorExpr dot product
    template <typename F> DotProduct<M, E, F> dot(const VectorExpr<M, N, F>& op) const {
        return DotProduct<M, E, F>(get(), op.get());
    }
    // evaluate parametric nodes in the expression, does nothing if not redefined in derived classes
    template <typename T> void forward(T i) const { return; }
    // map unary operator- to a VectorNegationOp expression node
    VectorNegationOp<M, N, E> operator-() const { return VectorNegationOp<M, N, E>(get()); }
};
  
// an expression node representing a constant vector
template <int M, int N> class VectorConst : public VectorExpr<M, N, VectorConst<M, N>> {
   private:
    typedef VectorExpr<M, N, VectorConst<M, N>> Base;
    typename Base::OuterVectorType value_;
   public:
    VectorConst(const typename Base::OuterVectorType& value, int m, int n) : Base(m, n), value_(value) {};
    double operator[](std::size_t i) const { return value_[i]; }
};

// wraps a n_rows x N matrix of data, acts as an SVector<N> once fixed the matrix row, assuming each row contains
// the vector to map
template <int M, int N> class DiscretizedVectorField : public VectorExpr<M, N, DiscretizedVectorField<M, N>> {
   private:
    typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> DataType;
    DataType* data_;
    Eigen::Map<SVector<N>> value_;
   public:
    DiscretizedVectorField() : value_(NULL) {};
    DiscretizedVectorField(DataType& data) : data_(&data), value_(NULL) {};
    double operator[](std::size_t i) const { return value_[i]; }
    void forward(std::size_t i) {
        new (&value_) Eigen::Map<SVector<M>>(data_->data() + (i * N));   // construct map in place
    }
};
  
// a generic binary operation node
template <int M, int N, typename OP1, typename OP2, typename BinaryOperation>
class VectorBinOp : public VectorExpr<M, N, VectorBinOp<M, N, OP1, OP2, BinaryOperation>> {
    static_assert(OP1::static_inner_size == OP2::static_inner_size, "you mixed fields with different base dimension");
   private:
    typedef VectorExpr<M, N, VectorBinOp<M, N, OP1, OP2, BinaryOperation>> Base;
    typename std::remove_reference<OP1>::type op1_;   // first  operand
    typename std::remove_reference<OP2>::type op2_;   // second operand
    BinaryOperation f_;                               // operation to apply
   public:
    static constexpr int static_inner_size = OP1::static_inner_size;

    // constructor
    VectorBinOp(const OP1& op1, const OP2& op2, BinaryOperation f) :
        Base(op1.inner_size(), op1.outer_size()), op1_(op1), op2_(op2), f_(f) {
        if constexpr (M == Dynamic || N == Dynamic) {
            fdapde_assert(op1_.inner_size() == op2_.inner_size() && op1_.outer_size() == op2_.outer_size());
        }
    };
    auto operator[](std::size_t i) const { return f_(op1_[i], op2_[i]); }
    // forward to child nodes
    template <typename T> const VectorBinOp<M, N, OP1, OP2, BinaryOperation>& forward(T i) {
        op1_.forward(i); op2_.forward(i);
        return *this;
    }
};
DEF_VECT_EXPR_OPERATOR(operator+, std::plus<>)
DEF_VECT_EXPR_OPERATOR(operator-, std::minus<>)

// support for double*VectorExpr: multiplies each element of VectorExpr by the scalar
// node representing a scalar value in a vectorial expression.
template <int M, int N> class VectorScalar : public VectorExpr<M, N, VectorScalar<M, N>> {
   private:
    typedef VectorExpr<M, N, VectorScalar<M, N>> Base;
    double value_;
   public:
    VectorScalar(double value, int m, int n) : Base(m, n), value_(value) {};
    double operator[](size_t i) const { return value_; }
};
template <int M, int N, typename E>
VectorBinOp<M, N, VectorScalar<M, N>, E, std::multiplies<>> operator*(double op1, const VectorExpr<M, N, E>& op2) {
    return VectorBinOp<M, N, VectorScalar<M, N>, E, std::multiplies<>>(
      VectorScalar<M, N>(op1, op2.inner_size(), op2.outer_size()), op2.get(), std::multiplies<>());
}
template <int M, int N, typename E>
VectorBinOp<M, N, E, VectorScalar<M, N>, std::multiplies<>> operator*(const VectorExpr<M, N, E>& op1, double op2) {
    return VectorBinOp<M, N, E, VectorScalar<M, N>, std::multiplies<>>(
      op1.get(), VectorScalar<M, N>(op2, op1.inner_size(), op1.outer_size()), std::multiplies<>());
}

// unary negation operation
template <int M, int N, typename E> class VectorNegationOp : public VectorExpr<M, N, VectorNegationOp<M, N, E>> {
   private:
    typename std::remove_reference<E>::type op_;
   public:
    // constructor
    VectorNegationOp(const E& op) : op_(op) {};
    // subscript operator
    auto operator[](std::size_t i) const { return -(op_[i]); }
    // call parameter evaluation on stored operand
    template <typename T> const VectorNegationOp<M, N, E>& forward(T i) {
        op_.forward(i);
        return *this;
    }
};

}   // namespace core
}   // namespace fdapde

#endif   // __VECTOR_EXPRESSIONS_H__
