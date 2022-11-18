// constructor
template <unsigned int M, unsigned int N, unsigned int R>
Element<M,N,R>::Element(std::size_t ID, const std::array<std::size_t, ct_nvertices(M)>& nodeIDs, const std::array<SVector<N>, ct_nvertices(M)>& coords,
			  const std::vector<int>& neighbors, const std::array<std::size_t, ct_nvertices(M)>& boundary) :
  ID_(ID), nodeIDs_(nodeIDs), coords_(coords), neighbors_(neighbors), boundary_(boundary) {
  // precompute barycentric coordinate matrix for fast access
  // use first point as reference
  SVector<N> ref = coords_[0];
  for(std::size_t j = 0; j < M; ++j){
    barycentricMatrix_.col(j) = coords_[j+1] - ref;
  }
    
  // find barycentric coordinates is equivalent to solve a linear system. for efficiecy reasons caching the inverse of the barycentric
  // matrix can be usefull expetially if there is the need to continuously access to barycentric coordinates.
  if constexpr(N == M){
    invBarycentricMatrix_ = barycentricMatrix_.inverse();
  }else{
    // returns the generalized inverse of the barycentric matrix (which is a rectangular matrix for manifold meshes)
    invBarycentricMatrix_ = (barycentricMatrix_.transpose()*barycentricMatrix_).inverse()*barycentricMatrix_.transpose();
  }

  // precompute element measure
  if constexpr(M == N){ // non-manifold case
    measure_ = std::abs(barycentricMatrix_.determinant())/(ct_factorial(M));
  }else{
    if constexpr(M == 2) // surface element, compute area of 3D triangle
      measure_ = 0.5 * barycentricMatrix_.col(0).cross(barycentricMatrix_.col(1)).norm();
    if constexpr(M == 1) // network element, compute length of 2D segment
      measure_ = barycentricMatrix_.col(0).norm();
  }

};

// returns the barycentric coordinates of point x with respect to this element
template <unsigned int M, unsigned int N, unsigned int R>
SVector<M+1> Element<M,N,R>::toBarycentricCoords(const SVector<N>& x) const {
  // solve linear system barycenitrcMatrix_*z = (x-ref) by using the precomputed inverse of the barycentric matrix
  SVector<M> z = invBarycentricMatrix_*(x - coords_[0]);
  // compute barycentric coordinate of reference element
  double z0 = 1 - z.sum();  
  SVector<M+1> result;
  result << SVector<1>(z0), z;

  return result;
}

// returns the midpoint of the element (dimension of the returned point is the same of the embedding dimension)
template <unsigned int M, unsigned int N, unsigned int R>
SVector<N> Element<M,N,R>::midPoint() const {
  // a remarkable property of barycentric coordinates is that the center of gravity of an element has all its
  // barycentric coordinates equal to 1/(M+1). In order to compute the midpoint of an element we hence map this
  // point back in cartesian coordinates
  SVector<M> barycentricMidPoint;
  barycentricMidPoint.fill(1.0/(M+1)); // avoid implicit conversion to int

  return barycentricMatrix_*barycentricMidPoint + coords_[0];
}

// returns the bounding box of the element
template <unsigned int M, unsigned int N, unsigned int R>
std::pair<SVector<N>, SVector<N>> Element<M,N,R>::boundingBox() const{
  // define lower-left and upper-right corner of bounding box
  SVector<N> ll, ur;
  // project each vertex coordinate on the reference axis
  std::array<std::array<double, ct_nvertices(M)>, N> projCoords;
  
  for(std::size_t j = 0; j < ct_nvertices(M); ++j){
    for(std::size_t dim = 0; dim < N; ++dim){
      projCoords[dim][j] = coords_[j][dim];
    }
  }
  // take minimum and maximum value along each dimension, those values define the lower-left and
  // upper-right corner of the bounding box
  for(std::size_t dim = 0; dim < N; ++dim){
    ll[dim] = *std::min_element(projCoords[dim].begin(), projCoords[dim].end());      
    ur[dim] = *std::max_element(projCoords[dim].begin(), projCoords[dim].end());
  }

  return std::make_pair(ll, ur);
}

// check if a point is contained in the element
template <unsigned int M, unsigned int N, unsigned int R>
template <bool is_manifold>
typename std::enable_if<!is_manifold, bool>::type
Element<M,N,R>::contains(const SVector<N> &x) const {
  // you can prove that a point is inside the element if all its barycentric coordinates are positive
  // if the point is on the boundary of the element it might happen that some barycentric coordinate is negative by some
  // really small value. This is the reason why we don't check for exactly zero positive entries but allow for a small tolerance
  
  // get barycentric coordinates of input point
  SVector<N+1> baryCoord = toBarycentricCoords(x);
  // use Eigen visitor to check for positiveness of elements
  return (baryCoord.array() >= -10*std::numeric_limits<double>::epsilon()).all();
}

template <unsigned int M, unsigned int N, unsigned int R>
VectorSpace<M, N> Element<M,N,R>::spannedSpace() const {
  // build a basis for the space containing the element,
  std::array<SVector<N>, M> basis;
  for(size_t i = 0; i < M; ++i){
    basis[i] = (coords_[i+1] - coords_[0]);
  }
  return VectorSpace<M, N>(basis, coords_[0]);
}

// specialization for manifold elements of contains() routine. 
template <unsigned int M, unsigned int N, unsigned int R>
template <bool is_manifold>
typename std::enable_if<is_manifold, bool>::type
  Element<M,N,R>::contains(const SVector<N>& x) const {
  // we start checking if the point is contained in the affine space spanned by the mesh element
  VectorSpace<M, N> vs = spannedSpace();
  // if the distance between the point projection into the plane and the point itself is larger than 0
  // return false, the point does not belong to the plane and therefore cannot belong to the surface element
  if(vs.distance(x) > 10*std::numeric_limits<double>::epsilon()){
    return false;
  }
  // if the point belongs to the spanned space, check if its barycentric coordinates are all positive
  SVector<N> baryCoord = toBarycentricCoords(x);
  return (baryCoord.array() >= 0).all(); // use Eigen visitor to check for positiveness of elements
}

// return true if the element has at least one vertex on the boundary of the domain
template <unsigned int M, unsigned int N, unsigned int R>
bool Element<M,N,R>::isOnBoundary(void) const{
  for(size_t i = 0; i < ct_nvertices(M); ++i){
    if(boundary_[i] == 1) return true;
  }
  return false;
}

template <unsigned int M, unsigned int N, unsigned int R>
std::vector<std::pair<std::size_t, SVector<N>>> Element<M,N,R>::boundaryNodes() const {
  std::vector<std::pair<std::size_t, SVector<N>>> result{};
  for(std::size_t i = 0; i < ct_nvertices(M); ++i){
    if(boundary_[i] == 1){
      result.push_back(std::make_pair(nodeIDs_[i], coords_[i]));
    }
  }
  return result;
}
