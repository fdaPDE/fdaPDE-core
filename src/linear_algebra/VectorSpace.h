#ifndef __VECTOR_SPACE_H__
#define __VECTOR_SPACE_H__

#include "../utils/Symbols.h"
#include <vector>

namespace fdaPDE {
namespace core{
namespace NLA{

  // a template class to perform geometric operations in general vector and affine spaces.
  // M is the vector space dimension, N is the dimension of the embedding space
  template <unsigned int M, unsigned int N>
  class VectorSpace {
  private:
    std::array<SVector<N>, M> basis_{}; // the set of vectors generating the vector space
    // a point throught which the vector space passes. If this is not zero, the modeled space is affine (does not pass throught zero)
    SVector<N> offset_{};

    // orthonormalizes the given set of vectors (implementation of the modified gram schmidt method)
    void orthonormalize();  
  public:
    // constructor
    VectorSpace() = default;
    // affine space constructor
    VectorSpace(const std::array<SVector<N>, M>& basis, const SVector<N>& offset) : basis_(basis), offset_(offset) {
      orthonormalize();
    };
    // space passing by zero
    VectorSpace(const std::array<SVector<N>, M>& basis) : VectorSpace(basis, SVector<N>::Zero()) {};
  
    // project the point x onto this space
    SVector<M> projectOnto(const SVector<N>& x);
    // project the point x into this space
    SVector<N> projectInto(const SVector<N>& x);
    // returns the euclidean distance between the point x and this space
    double distance(const SVector<N>& x);
    // returns an element of the vector space written as linear combination of the basis vectors with respect to a given set
    // of coefficients.
    SVector<N> operator()(const std::array<double, M>& coeffs) const;
  };

#include "VectorSpace.tpp"
}}}
  
#endif // __VECTOR_SPACE_H__
