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

#ifndef __FIELD_DERIVATIVES_H__
#define __FIELD_DERIVATIVES_H__

#include <type_traits>
#include "../utils/symbols.h"

namespace fdapde {
namespace core {

// forward declarations
template <int M, int N, int K, typename F> class MatrixExpr;
template <int M, int N, typename F> class VectorExpr;
template <int M, typename E> class ScalarExpr;

// partial derivative of order R for the field OP
template <int N, int R, typename OP> class FieldPartialDerivative { };

template <int N, typename OP>
class FieldPartialDerivative<N, 1, OP> : public ScalarExpr<N, FieldPartialDerivative<N, 1, OP>> {
   private:
    typedef typename static_dynamic_vector_selector<N>::type VectorType;
    typedef ScalarExpr<N, FieldPartialDerivative<N, 1, OP>> Base;
    typename std::remove_reference<OP>::type op_;
    int i_ = 0;
    double h_;
   public:
    FieldPartialDerivative(const OP& op, int i, double h, int inner_size) : Base(inner_size), op_(op), i_(i), h_(h) {}
    inline double operator()(VectorType x) const {   // must pass by copy
        double result;
        x[i_] = x[i_] + h_;
        result = op_(x);   // f(x + h)
        x[i_] = x[i_] - 2 * h_;
        result -= op_(x);   // f(x - h)
        return result / (2 * h_);
    }
};

// a functor representing the numerical approximation of df^2/(dx_i dx_j)
template <int N, typename OP>
class FieldPartialDerivative<N, 2, OP> : public ScalarExpr<N, FieldPartialDerivative<N, 2, OP>> {
   private:
    typedef typename static_dynamic_vector_selector<N>::type VectorType;
    typedef ScalarExpr<N, FieldPartialDerivative<N, 2, OP>> Base;
    typename std::remove_reference<OP>::type op_;
    int i_ = 0, j_ = 0;
    double h_;
   public:
    FieldPartialDerivative(const OP& op, int i, int j, double h, int inner_size) :
        Base(inner_size), op_(op), i_(i), j_(j), h_(h) { }
    inline double operator()(VectorType x) const {   // must pass by copy
        double result;
        if (i_ != j_) {   // df^2/(dx_i dx_j)
            x[i_] = x[i_] + h_; x[j_] = x[j_] + h_;
            result = op_(x);
            x[j_] = x[j_] - 2 * h_;
            result -= op_(x);
            x[i_] = x[i_] - 2 * h_;
            result += op_(x);
            x[j_] = x[j_] + 2 * h_;
            result -= op_(x);
	    // (f(x + h_i + h_j) - f(x + h_i - h_j) - f(x - h_i + h_j) + f(x - h_i - h_j)) / (4 * h^2)
            return result / (4 * h_ * h_);
        } else {   // df^2/dx_i^2
            x[i_] = x[i_] + 2 * h_;
            result = -op_(x);
            x[i_] = x[i_] - h_;
            result += 16 * op_(x);
            x[i_] = x[i_] - h_;
            result -= 30 * op_(x);
            x[i_] = x[i_] - h_;
            result += 16 * op_(x);
            x[i_] = x[i_] - h_;
            result -= op_(x);
	    // (-f(x+2*h) + 16*f(x+h) - 30*f(x) + 16f(x-h) - f(x-2*h))/(12*h^2)
            return result / (12 * h_ * h_);
        }
    }
};

// vectorial expression encoding the gradient of a scalar expression
template <int N, typename OP> class ScalarExprGradient : public VectorExpr<N, N, ScalarExprGradient<N, OP>> {
   public:
    typedef VectorExpr<N, N, ScalarExprGradient<N, OP>> Base;
    typename std::remove_reference<OP>::type op_;
    double h_ = 1e-3;   // step size used in central finite differences
    ScalarExprGradient(const OP& op) : Base(op.inner_size(), op.inner_size()), op_(op) { }
    ScalarExprGradient(const OP& op, double h) : Base(op.inner_size(), op.inner_size()), op_(op), h_(h) { }
    // laxy evaluation for d^1(op_)/dx_i
    FieldPartialDerivative<N, 1, OP> operator[](int i) const {
        return FieldPartialDerivative<N, 1, OP>(op_, i, h_, op_.inner_size());
    }
};

// matrix expression encoding the hessian of a scalar expression
template <int N, typename OP> class ScalarExprHessian : public MatrixExpr<N, N, N, ScalarExprHessian<N, OP>> {
   public:
    typedef MatrixExpr<N, N, N, ScalarExprHessian<N, OP>> Base;
    typename std::remove_reference<OP>::type op_;
    double h_ = 1e-3;   // step size used in central finite differences
    ScalarExprHessian(const OP& op) : Base(op.inner_size(), op.inner_size(), op.inner_size()), op_(op) { }
    ScalarExprHessian(const OP& op, double h) :
        Base(op.inner_size(), op.inner_size(), op.inner_size()), op_(op), h_(h) { }
    // return second order partial derivative functor for d^2(op_)/(dx_i dx_j)
    FieldPartialDerivative<N, 2, OP> coeff(int i, int j) const {
        return FieldPartialDerivative<N, 2, OP>(op_, i, j, h_, op_.inner_size());
    }
};

}   // namespace core
}   // namespace fdapde

#endif   // __FIELD_DERIVATIVES_H__
