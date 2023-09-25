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

namespace fdapde {
namespace core {

// Base class for any VectorField type
struct VectorBase { };

#define DEF_VECT_EXPR_OPERATOR(OPERATOR, FUNCTOR)                                                                      \
    template <int M, int N, typename E1, typename E2>                                                                  \
    VectorBinOp<M, N, E1, E2, FUNCTOR> OPERATOR(const VectorExpr<M, N, E1>& op1, const VectorExpr<M, N, E2>& op2) {    \
        return VectorBinOp<M, N, E1, E2, FUNCTOR>(op1.get(), op2.get(), FUNCTOR());                                    \
    }                                                                                                                  \
                                                                                                                       \
    template <int M, int N, typename E>                                                                                \
    VectorBinOp<M, N, VectorConst<M, N>, E, FUNCTOR> OPERATOR(SVector<N> op1, const VectorExpr<M, N, E>& op2) {        \
        return VectorBinOp<M, N, VectorConst<M, N>, E, FUNCTOR>(VectorConst<M, N>(op1), op2.get(), FUNCTOR());         \
    }                                                                                                                  \
                                                                                                                       \
    template <int M, int N, typename E>                                                                                \
    VectorBinOp<M, N, E, VectorConst<M, N>, FUNCTOR> OPERATOR(const VectorExpr<M, N, E>& op1, SVector<N> op2) {        \
        return VectorBinOp<M, N, E, VectorConst<M, N>, FUNCTOR>(op1.get(), VectorConst<M, N>(op2), FUNCTOR());         \
    }

// forward declarations
template <int M, typename T1, typename T2> class DotProduct;
template <int M, int N> class VectorConst;
template <int M, int N, typename E> class VectorNegationOp;

// Base class for vectorial expressions
// M: input space dimension, N x 1: output dimension
template <int M, int N, typename E> struct VectorExpr : public VectorBase {
    // call operator[] on the base type E
    auto operator[](std::size_t i) const { return static_cast<const E&>(*this)[i]; }
    const E& get() const { return static_cast<const E&>(*this); }
    // evaluate the expression at point p
    SVector<N> operator()(const SVector<M>& p) const {
        SVector<N> result;
        for (size_t i = 0; i < N; ++i) {
            // trigger evaluation, call subscript of the underyling type. This will produce along the dimension i
            // a callable object, evaluate this passing the point p to get a double
            result[i] = operator[](i)(p);
        }
        return result;
    }
    // dot product between VectorExpr and SVector
    virtual DotProduct<M, E, VectorConst<M, N>> dot(const SVector<N>& op) const;
    // VectorExpr - VectorExpr dot product
    template <typename F> DotProduct<M, E, F> dot(const VectorExpr<M, N, F>& op) const;
    // evaluate parametric nodes in the expression, does nothing if not redefined in derived classes
    template <typename T> void forward(T i) const { return; }
    // map unary operator- to a VectorNegationOp expression node
    VectorNegationOp<M, N, E> operator-() const { return VectorNegationOp<M, N, E>(get()); }

    // expose compile time informations
    static constexpr int rows = N;
    static constexpr int cols = 1;
    static constexpr int base = M;   // dimensionality of base space
};

// an expression node representing a constant vector
template <int M, int N> class VectorConst : public VectorExpr<M, N, VectorConst<M, N>> {
   private:
    SVector<N> value_;
   public:
    VectorConst(SVector<N> value) : value_(value) {};
    // return the stored value along direction i
    double operator[](std::size_t i) const { return value_[i]; }
};

// wraps a n_rows x N matrix of data, acts as an SVector<N> once fixed the matrix row, assuming each row contains
// the vector to map
template <int M, int N = M> class VectorDataWrapper : public VectorExpr<M, N, VectorDataWrapper<M, N>> {
   private:
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> data_;
    Eigen::Map<SVector<N>> value_;
   public:
    VectorDataWrapper() : value_(NULL) {};
    VectorDataWrapper(const DMatrix<double>& data) : data_(data), value_(NULL) {};
    double operator[](std::size_t i) const { return value_[i]; }
    void forward(std::size_t i) {
        new (&value_) Eigen::Map<SVector<M>>(data_.data() + (i * N));   // construct map in place
    }
};

// a generic binary operation node
template <int M, int N, typename OP1, typename OP2, typename BinaryOperation>
class VectorBinOp : public VectorExpr<M, N, VectorBinOp<M, N, OP1, OP2, BinaryOperation>> {
   private:
    typename std::remove_reference<OP1>::type op1_;   // first  operand
    typename std::remove_reference<OP2>::type op2_;   // second operand
    BinaryOperation f_;                               // operation to apply
   public:
    // constructor
    VectorBinOp(const OP1& op1, const OP2& op2, BinaryOperation f) : op1_(op1), op2_(op2), f_(f) {};
    // subscript operator. Let compiler to infer the return type (generally a FieldExpr)
    auto operator[](std::size_t i) const { return f_(op1_[i], op2_[i]); }
    // call parameter evaluation on operands
    template <typename T> const VectorBinOp<M, N, OP1, OP2, BinaryOperation>& forward(T i) {
        op1_.forward(i);
        op2_.forward(i);
        return *this;
    }
};
DEF_VECT_EXPR_OPERATOR(operator+, std::plus<>)
DEF_VECT_EXPR_OPERATOR(operator-, std::minus<>)

// support for double*VectorExpr: multiplies each element of VectorExpr by the scalar
// node representing a scalar value in a vectorial expression.
template <int M, int N> class VectorScalar : public VectorExpr<M, N, VectorScalar<M, N>> {
   private:
    double value_;
   public:
    VectorScalar(double value) : value_(value) {};
    double operator[](size_t i) const { return value_; }
};
template <int M, int N, typename E>
VectorBinOp<M, N, VectorScalar<M, N>, E, std::multiplies<>> operator*(double op1, const VectorExpr<M, N, E>& op2) {
    return VectorBinOp<M, N, VectorScalar<M, N>, E, std::multiplies<>>(
      VectorScalar<M, N>(op1), op2.get(), std::multiplies<>());
}
template <int M, int N, typename E>
VectorBinOp<M, N, E, VectorScalar<M, N>, std::multiplies<>> operator*(const VectorExpr<M, N, E>& op1, double op2) {
    return VectorBinOp<M, N, E, VectorScalar<M, N>, std::multiplies<>>(
      op1.get(), VectorScalar<M, N>(op2), std::multiplies<>());
}

// dot product between a VectorExpr and an (eigen) SVector.
template <int M, int N, typename E>
DotProduct<M, E, VectorConst<M, N>> VectorExpr<M, N, E>::dot(const SVector<N>& op) const {
    return DotProduct<M, E, VectorConst<M, N>>(this->get(), VectorConst<M, N>(op));
}
// dot product between a VectorExpr and a VectorExpr
template <int M, int N, typename E>
template <typename F>
DotProduct<M, E, F> VectorExpr<M, N, E>::dot(const VectorExpr<M, N, F>& op) const {
    return DotProduct<M, E, F>(this->get(), op.get());
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
