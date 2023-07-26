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

#ifndef __SPACE_VARYING_H__
#define __SPACE_VARYING_H__

#include "../../fields/matrix_expressions.h"
#include "../../fields/scalar_expressions.h"
#include "../../fields/vector_expressions.h"

namespace fdapde {
namespace core {
  
// M: input space dimension, N x K: dimension of the returned diffusion tensor.
template <int M, int N = M, int K = M> class NonConstantDiffusionWrapper {
   private:
    // coeff is a matrix (num_elements*num_nodes) x (N*K), i.e., a row of this matrix corresponds to a 1D
    // expansion of the N x K diffusion tensor.
    DMatrix<double> coeff_;
   public:
    // constructor
    NonConstantDiffusionWrapper() = default;
    NonConstantDiffusionWrapper(const DMatrix<double>& coeff) : coeff_(coeff) { }
    // set data (allow to set the coefficient after the expression template has been built)
    void set_coefficient(const DMatrix<double>& coeff) { coeff_ = coeff; }
    // call operator, handles the conversion from 1D expansion to N x K matrix
    inline SMatrix<N, K> operator()(std::size_t i) const {
        SMatrix<N, K> result;
        for (std::size_t j = 0; j < N; ++j) result.row(j) = coeff_.row(i).segment<K>(j * K);
        return result;
    }
    // convert 
    MatrixParam<M, N, K, NonConstantDiffusionWrapper<M, N, K>, std::size_t> to_expr() const {
        return MatrixParam<M, N, K, NonConstantDiffusionWrapper<M, N, K>, std::size_t>(*this);
    }
};

// M: input space dimension, N x 1 : dimension of the returned transport vector.
template <int M, int N = M> class NonConstantAdvectionWrapper {
   private:
    DMatrix<double> coeff_;
   public:
    // constructor
    NonConstantAdvectionWrapper() = default;
    NonConstantAdvectionWrapper(const DMatrix<double>& coeff) : coeff_(coeff) { }
    // set data (allow to set the coefficient after the expression template has been built)
    void set_coefficient(const DMatrix<double>& coeff) { coeff_ = coeff; }
    inline SVector<N> operator()(std::size_t i) const { return coeff_.block<1, N>(i, 0); }
    // return this object as compatible with the expression template mechanism of parametric expressions
    VectorParam<M, N, NonConstantAdvectionWrapper<M, N>, std::size_t> to_expr() const {
        return VectorParam<M, N, NonConstantAdvectionWrapper<M, N>, std::size_t>(*this);
    }
};

// M : input space dimension
template <int M> class NonConstantReactionWrapper {
   private:
    DMatrix<double> coeff_;
   public:
    // constructor
    NonConstantReactionWrapper() = default;
    NonConstantReactionWrapper(const DMatrix<double>& coeff) : coeff_(coeff) { }
    // set data (allow to set the coefficient after the expression template has been built)
    void set_coefficient(const DMatrix<double>& coeff) { coeff_ = coeff; }
    inline double operator()(std::size_t i) const { return coeff_(i, 0); }
    // return this object as compatible with the expression template mechanism of parametric expressions
    ScalarParam<M, NonConstantReactionWrapper<M>, std::size_t> to_expr() const {
        return ScalarParam<M, NonConstantReactionWrapper<M>, std::size_t>(*this);
    }
};

}   // namespace core
}   // namespace fdapde

#endif   // __SPACE_VARYING_H__
