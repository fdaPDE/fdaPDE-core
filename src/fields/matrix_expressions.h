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

#ifndef __MATRIX_EXPRESSIONS_H__
#define __MATRIX_EXPRESSIONS_H__

#include <type_traits>

#include "../utils/symbols.h"
#include "scalar_field.h"
#include "vector_expressions.h"

namespace fdapde {
namespace core {

// forward declarations
template <int N, int M, int K, typename F> class MatrixField;
template <int N, int M, int K, typename E> struct MatrixExpr;
template <int N, int M, int K, typename E> class MatrixNegationOp;

// Base class for any MatrixField type
struct MatrixBase { };

// an expression node representing a matrix*vector operation
template <int N, int M, int K, typename T1, typename T2>
class MatrixVectorProduct : public VectorExpr<N, M, MatrixVectorProduct<N, M, K, T1, T2>> {
    static_assert(std::is_base_of<MatrixBase, T1>::value && std::is_base_of<VectorBase, T2>::value);
   private:
    T1 op1_;
    T2 op2_;
   public:
    MatrixVectorProduct(const T1& op1, const T2& op2) : op1_(op1), op2_(op2) {};
    // the i-th element of the vector is given as the dot product of the i-th column of T1 and T2
    inline DotProduct<N, decltype(op1_.row(std::size_t())), T2> operator[](std::size_t i) const {
        return DotProduct<N, decltype(op1_.row(std::size_t())), T2>(op1_.row(i), op2_);
    }
    // call parameter evaluation on operands
    template <typename T> const MatrixVectorProduct<N, M, K, T1, T2>& eval_parameters(T i) {
        op1_.eval_parameters(i);
        op2_.eval_parameters(i);
        return *this;
    }
};

// an expression node representing a matrix*matrix operation
template <int N, int M, int K, typename T1, typename T2>
class MatrixMatrixProduct : public MatrixExpr<N, M, K, MatrixMatrixProduct<N, M, K, T1, T2>> {
    static_assert(std::is_base_of<MatrixBase, T1>::value && std::is_base_of<MatrixBase, T2>::value);
   private:
    T1 op1_;
    T2 op2_;
   public:
    MatrixMatrixProduct(const T1& op1, const T2& op2) : op1_(op1), op2_(op2) {};
    // the (i,j)-th element of the resulting matrix is given by the dot product of the i-th column of T1 and
    // the j-th column of T2
    inline DotProduct<N, decltype(op1_.row(std::size_t())), decltype(op2_.col(std::size_t()))>
    operator()(std::size_t i, std::size_t j) const {
        return DotProduct<N, decltype(op1_.row(std::size_t())), decltype(op2_.col(std::size_t()))>(
          op1_.row(i), op2_.col(j));
    }
    // call parameter evaluation on operands
    template <typename T> const MatrixMatrixProduct<N, M, K, T1, T2>& eval_parameters(T i) {
        op1_.eval_parameters(i);
        op2_.eval_parameters(i);
        return *this;
    }
};

// a node representing a single row of a matrix expression
template <int N, int M, int K, typename E> class MatrixRow : public VectorExpr<N, M, MatrixRow<N, M, K, E>> {
   private:
    E expr_;
    std::size_t row_;
   public:
    MatrixRow(const E& expr, std::size_t row) : expr_(expr), row_(row) {};
    // subscripting the i-th element of a row triggers evaluation of the expression
    auto operator[](std::size_t i) const { return expr_.coeff(row_, i); }
    // call parameter evaluation on stored expression
    template <typename T> const MatrixRow<N, M, K, E>& eval_parameters(T i) {
        expr_.eval_parameters(i);
        return *this;
    }
    // expose number of rows and columns
    static constexpr int rows = 1;
    static constexpr int cols = K;
};
// a node representing a single column of a matrix expression
template <int N, int M, int K, typename E> class MatrixCol : public VectorExpr<N, M, MatrixCol<N, M, K, E>> {
   private:
    E expr_;
    std::size_t col_;
   public:
    MatrixCol(const E& expr, std::size_t col) : expr_(expr), col_(col) {};
    // subscripting the i-th element of a column triggers evaluation of the expression
    auto operator[](std::size_t i) const { return expr_.coeff(i, col_); }
    // call parameter evaluation on stored expression
    template <typename T> const MatrixCol<N, M, K, E>& eval_parameters(T i) {
        expr_.eval_parameters(i);
        return *this;
    }
    // expose number of rows and columns
    static constexpr int rows = M;
    static constexpr int cols = 1;
};
// rows and cols are typically wrapped by DotProduct to perform more general operations

// macro for the definition of standard operations between matrix fields
#define DEF_MATRIX_EXPR_OPERATOR(OPERATOR, FUNCTOR)                                                                    \
    template <int N, int M, int K, typename E1, typename E2>                                                           \
    MatrixBinOp<N, M, K, E1, E2, FUNCTOR> OPERATOR(                                                                    \
      const MatrixExpr<N, M, K, E1>& op1, const MatrixExpr<N, M, K, E2>& op2) {                                        \
        return MatrixBinOp<N, M, K, E1, E2, FUNCTOR>(op1.get(), op2.get(), FUNCTOR());                                 \
    }                                                                                                                  \
                                                                                                                       \
    template <int N, int M, int K, typename E>                                                                         \
    MatrixBinOp<N, M, K, MatrixConst<N, M, K>, E, FUNCTOR> OPERATOR(                                                   \
      SMatrix<M, K> op1, const MatrixExpr<N, M, K, E>& op2) {                                                          \
        return MatrixBinOp<N, M, K, MatrixConst<N, M, K>, E, FUNCTOR>(                                                 \
          MatrixConst<N, M, K>(op1), op2.get(), FUNCTOR());                                                            \
    }                                                                                                                  \
                                                                                                                       \
    template <int N, int M, int K, typename E>                                                                         \
    MatrixBinOp<N, M, K, E, MatrixConst<N, M, K>, FUNCTOR> OPERATOR(                                                   \
      const MatrixExpr<N, M, K, E>& op1, SMatrix<M, K> op2) {                                                          \
        return MatrixBinOp<N, M, K, E, MatrixConst<N, M, K>, FUNCTOR>(                                                 \
          op1.get(), MatrixConst<N, M, K>(op2), FUNCTOR());                                                            \
    }

// base class for matrix expressions
template <int N, int M, int K, typename E> struct MatrixExpr : public MatrixBase {
    // access operator on (i,j)-th element on the base type E
    auto coeff(std::size_t i, std::size_t j) const { return static_cast<const E&>(*this).coeff(i, j); }
    // get underyling type composing the expression node
    const E& get() const { return static_cast<const E&>(*this); }
    // evaluate the expression at point p
    SMatrix<M, K> operator()(const SVector<N>& p) const {
        SMatrix<M, K> result;
        for (std::size_t i = 0; i < M; ++i) {
            for (std::size_t j = 0; j < K; ++j) {
                // trigger evaluation on each element of the expression template.
                // This will produce for the (i,j)-th element a callable object, evaluate it on p
                result(i, j) = coeff(i, j)(p);
            }
        }
        return result;
    }
    // block access to i-th row/column of MatrixExpr
    MatrixRow<N, M, K, E> row(std::size_t i) const;
    MatrixCol<N, M, K, E> col(std::size_t i) const;
    // allow rhs multiplication by a VectorExpr
    template <typename F> MatrixVectorProduct<N, M, K, E, F> operator*(const VectorExpr<N, K, F>& op) const;
    // allow rhs multiplication by constant SVector
    MatrixVectorProduct<N, M, K, E, VectorConst<N, K>> operator*(const SVector<K>& op) const;
    // evaluate parametric nodes in the expression, does nothing if not redefined in derived classes
    template <typename T> void eval_parameters(T i) const { return; }
    // map unary operator- to a MatrixNegationOp expression node
    MatrixNegationOp<M, N, K, E> operator-() const { return MatrixNegationOp<M, N, K, E>(get()); }

    // expose compile time informations
    static constexpr int rows = M;
    static constexpr int cols = K;
    static constexpr int base = N;   // dimensionality of base space
};

// access i-th row of MatrixExpr
template <int N, int M, int K, typename E> MatrixRow<N, M, K, E> MatrixExpr<N, M, K, E>::row(std::size_t i) const {
    return MatrixRow<N, M, K, E>(this->get(), i);
}
// access i-th column of MatrixExpr
template <int N, int M, int K, typename E> MatrixCol<N, M, K, E> MatrixExpr<N, M, K, E>::col(std::size_t i) const {
    return MatrixCol<N, M, K, E>(this->get(), i);
}

// an expression node representing a constant matrix
template <unsigned int N, unsigned int M, unsigned int K>
class MatrixConst : public MatrixExpr<N, M, K, MatrixConst<N, M, K>> {
   private:
    SMatrix<M, K> value_;
   public:
    // constructor
    MatrixConst() = default;
    MatrixConst(SMatrix<M, K> value) : value_(value) { }
    double coeff(std::size_t i, std::size_t j) const { return value_(i, j); }
    // assignment operator
    MatrixConst<N, M, K>& operator=(const SMatrix<M, K>& value) {
        value_ = value;
        return *this;
    }
};

// a parameter node
template <unsigned int N, unsigned int M, unsigned int K, typename F, typename T>
class MatrixParam : public MatrixExpr<N, M, K, MatrixParam<N, M, K, F, T>> {
    // check F is callable with type T and returns an SMatrix<M,K>
    static_assert(std::is_same<decltype(std::declval<F>().operator()(T())), SMatrix<M, K>>::value);
   private:
    // be sure that pointed data are alive for the whole life of this object
    const typename std::remove_reference<F>::type* f_;
    SMatrix<M, K> value_;
   public:
    // default constructor
    MatrixParam() = default;
    MatrixParam(const F& f) : f_(&f) {};
    double coeff(std::size_t i, std::size_t j) const { return value_(i, j); }
    void eval_parameters(T i) { value_ = f_->operator()(i); }
};

// a generic binary operation node
template <int N, int M, int K, typename OP1, typename OP2, typename BinaryOperation>
class MatrixBinOp : public MatrixExpr<N, M, K, MatrixBinOp<N, M, K, OP1, OP2, BinaryOperation>> {
   private:
    typename std::remove_reference<OP1>::type op1_;   // first  operand
    typename std::remove_reference<OP2>::type op2_;   // second operand
    BinaryOperation f_;                               // operation to apply
   public:
    // constructor
    MatrixBinOp(const OP1& op1, const OP2& op2, BinaryOperation f) : op1_(op1), op2_(op2), f_(f) {};
    // access operator. Apply the functor to each accessed element. This returns a callable object
    auto coeff(std::size_t i, std::size_t j) const { return f_(op1_.coeff(i, j), op2_.coeff(i, j)); }
    // call parameter evaluation on operands
    template <typename T> const MatrixBinOp<N, M, K, OP1, OP2, BinaryOperation>& eval_parameters(T i) {
        op1_.eval_parameters(i);
        op2_.eval_parameters(i);
        return *this;
    }
};
DEF_MATRIX_EXPR_OPERATOR(operator+, std::plus<>)
DEF_MATRIX_EXPR_OPERATOR(operator-, std::minus<>)

// MatrixExpr<N,M,K> times VectorExpr<N,K>
template <int N, int M, int K, typename E>
template <typename F>
MatrixVectorProduct<N, M, K, E, F> MatrixExpr<N, M, K, E>::operator*(const VectorExpr<N, K, F>& op) const {
    return MatrixVectorProduct<N, M, K, E, F>(this->get(), op.get());
}
// SMatrix<M,K> times VectorExpr<N,K>
template <int N, int M, int K, typename F>
MatrixVectorProduct<N, M, K, MatrixConst<N, M, K>, F>
operator*(const Eigen::Matrix<double, M, K>& op1, const VectorExpr<N, K, F>& op2) {
    return MatrixVectorProduct<N, M, K, MatrixConst<N, M, K>, F>(MatrixConst<N, M, K>(op1), op2.get());
}
// MatrixExpr<N,M,K> times SVector<K>
template <int N, int M, int K, typename E>
MatrixVectorProduct<N, M, K, E, VectorConst<N, K>> MatrixExpr<N, M, K, E>::operator*(const SVector<K>& op) const {
    return MatrixVectorProduct<N, M, K, E, VectorConst<N, K>>(this->get(), VectorConst<N, K>(op));
}

// element wise multiplication of MatrixExpr by scalar
// class to represent a single scalar node in an MatrixExpr.
template <int N, int M, int K> class MatrixScalar : public MatrixExpr<N, M, K, MatrixScalar<N, M, K>> {
   private:
    double value_;
   public:
    MatrixScalar(double value) : value_(value) { }
    double coeff(std::size_t i, std::size_t j) const { return value_; }
};
template <int N, int M, int K, typename E>
MatrixBinOp<N, M, K, MatrixScalar<N, M, K>, E, std::multiplies<>>
operator*(double op1, const MatrixExpr<N, M, K, E>& op2) {
    return MatrixBinOp<N, M, K, MatrixScalar<N, M, K>, E, std::multiplies<>>(
      MatrixScalar<N, M, K>(op1), op2.get(), std::multiplies<>());
}
template <int N, int M, int K, typename E>
MatrixBinOp<N, M, K, E, MatrixScalar<N, M, K>, std::multiplies<>>
operator*(const MatrixExpr<N, M, K, E>& op1, double op2) {
    return MatrixBinOp<N, M, K, E, MatrixScalar<N, M, K>, std::multiplies<>>(
      op1.get(), MatrixScalar<N, M, K>(op2), std::multiplies<>());
}

// MatrixExpr * MatrixExpr
template <int N, int M, int K, int H, typename E1, typename E2>
MatrixMatrixProduct<N, M, H, E1, E2> operator*(const MatrixExpr<N, M, K, E1>& op1, const MatrixExpr<N, K, H, E2>& op2) {
    return MatrixMatrixProduct<N, M, H, E1, E2>(op1.get(), op2.get());
}

// unary negation operation
template <int M, int N, int K, typename E>
class MatrixNegationOp : public MatrixExpr<M, N, K, MatrixNegationOp<M, N, K, E>> {
   private:
    typename std::remove_reference<E>::type op_;
   public:
    // constructor
    MatrixNegationOp(const E& op) : op_(op) {};
    auto coeff(std::size_t i, std::size_t j) const { return -(op_.coeff(i, j)); }
    // call parameter evaluation on operands
    template <typename T> const MatrixNegationOp<N, M, K, E>& eval_parameters(T i) {
        op_.eval_parameters(i);
        return *this;
    }
};

}   // namespace core
}   // namespace fdapde

#endif   // __MATRIX_EXPRESSIONS_H__
