#ifndef __SPACE_VARYING_FUNCTORS_H__
#define __SPACE_VARYING_FUNCTORS_H__

// include expression templates engine
#include "../../utils/fields/expressions/Expressions.h"

namespace fdaPDE {
namespace core {
namespace FEM {

  // a set of functors to implement non-constant coefficients PDEs.
  // non-constant coefficients are given as matrices. evaluating one of the following functors at index i, returns
  // the value of the coefficient at the i-th quadrature node.

  // in the following M is dimension of the space where the expression of which a functor is part of is evaluated (you can
  // think of it as the dimension of the reference element where finite elements integral are evaluated)
  
  // N x K: dimension of the returned diffusion tensor.
  template <unsigned int M, unsigned int N = M, unsigned int K = M>
  class SpaceVaryingDiffusion {
  private:
    // coeff is a matrix (num_elements*num_nodes) x (N*K). This means that a row of this matrix corresponds to a 1D expansion
    // of the N x K diffusion tensor.
    DMatrix<double> coeff_;
  public:
    // constructor
    SpaceVaryingDiffusion() = default;
    SpaceVaryingDiffusion(const DMatrix<double>& coeff) : coeff_(coeff) {}
    // set data (allow to set the coefficient after the expression template has been built)
    void setData(const DMatrix<double>& coeff) { coeff_ = coeff; }
    // call operator, handles the conversion from 1D expansion to N x K matrix
    inline SMatrix<N,K> operator()(std::size_t i) const {
      SMatrix<N,K> result;
      for(std::size_t j = 0; j < N; ++j)
	result.row(j) = coeff_.row(i).segment<K>(j*K);
      return result;
    }
    // return this object as compatible with the expression template mechanism of parametric expressions
    MatrixParam<M,N,K, SpaceVaryingDiffusion<M,N,K>,std::size_t> asParameter() const {
      return MatrixParam<M,N,K, SpaceVaryingDiffusion<M,N,K>,std::size_t>(*this);
    }
  };

  // N : dimension of the returned advection vector.
  template <unsigned int M, unsigned int N = M>
  class SpaceVaryingAdvection {
  private:
    DMatrix<double> coeff_;
  public:
    // constructor
    SpaceVaryingAdvection() = default;
    SpaceVaryingAdvection(const DMatrix<double>& coeff) : coeff_(coeff) {}
    // set data (allow to set the coefficient after the expression template has been built)
    void setData(const DMatrix<double>& coeff) { coeff_ = coeff; }
    // call operator
    inline SVector<N> operator()(std::size_t i) const {
      return coeff_.block<1,N>(i,0);
    }
    // return this object as compatible with the expression template mechanism of parametric expressions
    VectorParam<M,N, SpaceVaryingAdvection<M,N>,std::size_t> asParameter() const {
      return VectorParam<M,N, SpaceVaryingAdvection<M,N>,std::size_t>(*this);
    }
  };

  class SpaceVaryingReaction {
  private:
    DMatrix<double> coeff_;
  public:
    // constructor
    SpaceVaryingReaction() = default;
    SpaceVaryingReaction(const DMatrix<double>& coeff) : coeff_(coeff) {}
    // set data (allow to set the coefficient after the expression template has been built)
    void setData(const DMatrix<double>& coeff) { coeff_ = coeff; }

    // call operator
    inline double operator()(std::size_t i) const {
      return coeff_(i,0);
    }
    // return this object as compatible with the expression template mechanism of parametric expressions
    ScalarParam<SpaceVaryingReaction,std::size_t> asParameter() const {
      return ScalarParam<SpaceVaryingReaction,std::size_t>(*this);
    }
  };
  
}}}

#endif // __SPACE_VARYING_FUNCTORS_H__
