#ifndef __VECTOR_SPACE_H__
#define __VECTOR_SPACE_H__

#include "../utils/Symbols.h"
#include <vector>

namespace fdaPDE {
namespace core{
namespace NLA{

  // a template class to perform geometric operations in general vector and affine spaces.
  // N is the dimension of the vectors generating the space
  template <unsigned int N>
  class VectorSpace {
  private:
    std::vector<SVector<N>> basis_{}; // the set of vectors generating the vector space
    // a point throught which the vector space passes. If this is not zero, the modeled space is affine (does not pass throught zero)
    SVector<N> offset_{};

    // orthonormalizes the given set of vectors (implementation of the modified gram schmidt method)
    void orthonormalize();  
  public:
    // constructor
    VectorSpace() = default;
    VectorSpace(const std::vector<SVector<N>>& basis, const SVector<N>& offset) : basis_(basis), offset_(offset) {
      orthonormalize();
    };
  
    // project the point x onto this space
    DVector<double> projectOnto(const SVector<N>& x);
    // project the point x into this space
    SVector<N> projectInto(const SVector<N>& x);
    // returns the euclidean distance between the point x and this space
    double distance(const SVector<N>& x);
  };

  // orthonormalize a given set of vectors, producing an orthonormal basis for the vector space spanned by
  // the set of vectors passed as input (implementation of the modified gram-schmidt method)
  template <unsigned int N>
  void VectorSpace<N>::orthonormalize(){
    std::vector<SVector<N>> orthonormalBasis;
    orthonormalBasis.reserve(basis_.size());
  
    // returns the orthogonal projection of v over the space spanned by u
    std::function<SVector<N>(SVector<N>, SVector<N>)> projector = [](SVector<N> u, SVector<N> v) -> SVector<N> {
      double a = v.dot(u)/u.squaredNorm();
      // eigen cannot multiply a matrix by a constant. To allow element operations transform u into an array, perform
      // the multiplication and cast back to an eigen matrix
      return (a*(u.array())).matrix();
    };

    // take the first vector of the input basis as it is
    orthonormalBasis.push_back(basis_[0]/(basis_[0].norm()));
    // build orthonormal vectors following the modified gram-schmidt method
    for(int i = 1; i < basis_.size(); ++i){
      orthonormalBasis.push_back(basis_[i]);
      for(int j = 0; j < i; ++j){
	orthonormalBasis[i] -= projector(orthonormalBasis[j], basis_[i]);
      }
      orthonormalBasis[i] /= orthonormalBasis[i].norm();
    }
    // set new basis
    basis_ = orthonormalBasis;
    return;
  }

  // projects the point x on the space spanned by the vectors contained in basis. this returns a vector of the same dimension
  // of the space where we are projecting written as linear combination of the elements of the set of basis vectors
  template <unsigned int N>
  DVector<double> VectorSpace<N>::projectOnto(const SVector<N> &x){
    // build the projection onto the space spanned by the basis set
    DVector<double> projection;
    projection.resize(basis_.size(),1);

    // values of projection[i] are the coefficients of the linear combination of spaceBasis vectors which gives the input vector x
    for(size_t i = 0; i < basis_.size(); ++i){
      projection[i] = x.dot(basis_[i])/(basis_[i].norm());
    }
    return projection;
  }

  // project an N-dimensional point x on the space spanned by the set of basis vectors, returns an N-dimensional vector corresponding to
  // the projected point (written with respect to the canonical basis of the N-dimensional space where x lies).
  template <unsigned int N>
  SVector<N> VectorSpace<N>::projectInto(const SVector<N> &x){
    // build the projection operator on the space spanned by the basis
    Eigen::Matrix<double, N, Eigen::Dynamic> A;
    A.resize(N, basis_.size());
    // values of projection[i] are the coefficients of the linear combination of basis vectors which gives the input vector x
    for(size_t i = 0; i < basis_.size(); ++i){
      A.col(i) = basis_[i];
    }
    // given the projection operator A*A^T, the projection of x is computed as (A*A^T)*x
    return (A*A.transpose())*x;
  }

  // compute the euclidean distance from x
  template <unsigned int N>
  double VectorSpace<N>::distance(const SVector<N> &x) {
    // project point on subspace spanned by the basis vector, compute the distance between the point and its projection
    SVector<N> projection = projectInto(x-offset_);
    return ((x-offset_) - projection).squaredNorm();
  }
}}}
  
#endif // __VECTOR_SPACE_H__
