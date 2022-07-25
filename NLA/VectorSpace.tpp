// orthonormalize a given set of vectors, producing an orthonormal basis for the vector space spanned by
// the set of vectors passed as input (implementation of the modified gram-schmidt method)
template <unsigned int M, unsigned int N>
void VectorSpace<M, N>::orthonormalize(){
  std::array<SVector<N>, M> orthonormalBasis;
  
  // returns the orthogonal projection of v over the space spanned by u
  std::function<SVector<N>(SVector<N>, SVector<N>)> projector = [](SVector<N> u, SVector<N> v) -> SVector<N> {
    double a = v.dot(u)/u.squaredNorm();
    // eigen cannot multiply a matrix by a constant. To allow element operations transform u into an array, perform
    // the multiplication and cast back to an eigen matrix
    return (a*(u.array())).matrix();
  };

  // take the first vector of the input basis as it is
  orthonormalBasis[0] = (basis_[0]/(basis_[0].norm()));
  // build orthonormal vectors following the modified gram-schmidt method
  for(int i = 1; i < basis_.size(); ++i){
    orthonormalBasis[i] = basis_[i];
    for(int j = 0; j < i; ++j){
      orthonormalBasis[i] -= projector(orthonormalBasis[j], basis_[i]);
    }
    orthonormalBasis[i] /= orthonormalBasis[i].norm();
  }
  // set new basis
  basis_ = orthonormalBasis;
  return;
}

// projects the point x onto the space. this returns a vector of the same dimension of the space where we are projecting
// written as linear combination of the basis vectors
template <unsigned int M, unsigned int N>
SVector<M> VectorSpace<M, N>::projectOnto(const SVector<N> &x){
  // build the projection onto the space spanned by the basis set
  SVector<M> projection;

  // values of projection[i] are the coefficients of the linear combination of spaceBasis vectors which gives the input vector x
  for(size_t i = 0; i < basis_.size(); ++i){
    projection[i] = (x-offset_).dot(basis_[i])/(basis_[i].norm());
  }
  return projection;
}

// project an N-dimensional point x into the space
template <unsigned int M, unsigned int N>
SVector<N> VectorSpace<M, N>::projectInto(const SVector<N> &x){
  // build the projection operator on the space spanned by the basis
  Eigen::Matrix<double, N, Eigen::Dynamic> A;
  A.resize(N, basis_.size());
  // values of projection[i] are the coefficients of the linear combination of basis vectors which gives the input vector x
  for(size_t i = 0; i < basis_.size(); ++i){
    A.col(i) = basis_[i];
  }
  // given the projection operator A*A^T, the projection of x is computed as (A*A^T)*(x-offset_) + offset_ (cover also affine spaces)
  return (A*A.transpose())*(x-offset_) + offset_;
}

// compute the euclidean distance from x
template <unsigned int M, unsigned int N>
double VectorSpace<M, N>::distance(const SVector<N> &x) {
  // project point on subspace spanned by the basis vector, compute the distance between the point and its projection
  SVector<N> projection = projectInto(x);
  return (x - projection).squaredNorm();
}

// develops the linear combination of basis vectors with respect to the given vector of coefficients.
template <unsigned int M, unsigned int N>
SVector<N> VectorSpace<M, N>::operator()(const std::array<double, M>& coeffs) const{
  SVector<N> result = offset_;
  for(std::size_t i = 0; i < M; ++i){
    result += coeffs[i]*basis_[i];
  }
  return result;
}
