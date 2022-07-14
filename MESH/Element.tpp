// constructor
template <unsigned int M, unsigned int N>
Element<M,N>::Element(int ID_, std::array<std::pair<unsigned, SVector<N>>, N_VERTICES(M,N)> FEsupport_,
		      std::array<SVector<N>, N_VERTICES(M,N)> coords_, std::array<unsigned int, M+1> neighbors_,
		        std::array<std::pair<unsigned, unsigned>, N_VERTICES(M,N)> boundaryMarkers_) :
  ID(ID_), FEsupport(FEsupport_), coords(coords_), neighbors(neighbors_), boundaryMarkers(boundaryMarkers_) {

  // precompute barycentric coordinate matrix for fast access
  // use first point as reference
  SVector<N> ref = coords[0];
  for(size_t j = 0; j < M; ++j){
    baryMatrix.col(j) = coords[j+1] - ref;
  }
    
  // find barycentric coordinates is equivalent to solve a linear system.
  // for efficiecy reasons caching the inverse of baryMatrix can be usefull
  // expetially if there is the need to continuously access to barycentric
  // coordinates. Moreover Eigen inverse() is fast for very small matrices
  // (at most 4x4). See documentation to learn what system we are trying to solve
  invBaryMatrix = computeInvBaryMatrix(baryMatrix);
};

// returns the barycentric coordinates of point x with respect to this element
template <unsigned int M, unsigned int N>
SVector<M+1> Element<M, N>::computeBarycentricCoordinates(const SVector<N>& x) const {
  // solve linear system baryMatrix*z = (x - ref) by using the precomputed inverse of baryMatrix
  SVector<M> z = invBaryMatrix*(x - coords[0]);
  // compute barycentric coordinate of reference element
  double z0 = 1 - z.sum();
  
  SVector<M+1> result;
  result << SVector<1>(z0), z;

  return result;
}

// returns the midpoint of the element (dimension of the returned point is the same of the embedding dimension)
template <unsigned int M, unsigned int N>
SVector<N> Element<M, N>::computeMidPoint() const {
  // a remarkable property of barycentric coordinates is that the center of gravity of an element has all its
  // barycentric coordinates equal to 1/(N+1). In order to compute the midpoint of an element we hence map this
  // point back in cartesian coordinates
  SVector<M> barycentricMidPoint;
  barycentricMidPoint.fill(1/(N+1));
  
  return baryMatrix*barycentricMidPoint + coords[0];
}

// returns the bounding box of the element
template <unsigned int M, unsigned int N>
std::pair<SVector<N>, SVector<N>> Element<M,N>::computeBoundingBox() const{

  // define lower-left and upper-right corner of bounding box
  SVector<N> ll, ur;

  // projection of each vertex coordinate on reference axis
  std::array<std::array<double, N_VERTICES(M,N)>, N> projCoords;
  
  for(size_t j = 0; j < N_VERTICES(M,N); ++j){
    for(size_t dim = 0; dim < N; ++dim){
      projCoords[dim][j] = coords[j][dim];
    }
  }

  // take minimum and maximum value along each dimension, those values define the lower-left and
  // upper-right corner of the bounding box
  for(size_t dim = 0; dim < N; ++dim){
    ll[dim] = *std::min_element(projCoords[dim].begin(), projCoords[dim].end());      
    ur[dim] = *std::max_element(projCoords[dim].begin(), projCoords[dim].end());
  }

  return std::make_pair(ll, ur);
}

// check if a point is contained in the element
template <unsigned int M, unsigned int N>
template <bool is_manifold>
typename std::enable_if<!is_manifold, bool>::type
Element<M, N>::contains(const SVector<N> &x) const {
  // you can prove that a point is inside the element if all its barycentric coordinates are positive
  
  // get barycentric coordinates of input point
  SVector<N+1> baryCoord = computeBarycentricCoordinates(x);

  // use Eigen visitor to check for positiveness of elements
  return (baryCoord.array() >= 0).all();
}

// specialization for 2.5D domains (surfaces)
template <> 
Eigen::Matrix<double, 2, 3> SurfaceElement::computeInvBaryMatrix(const Eigen::Matrix<double, 3, 2>& baryMatrix) {
  // returns the generalized inverse of baryMatrix (which is a rectangular for surface elements)
  return (baryMatrix.transpose()*baryMatrix).inverse()*baryMatrix.transpose();
}

// specialization for manifold elements of contains() routine. 
template <unsigned int M, unsigned int N>
template <bool is_manifold>
typename std::enable_if<is_manifold, bool>::type
Element<M,N>::contains(const SVector<N>& x) const {
  // we start checking if the point is contained in the affine space spanned by the mesh element

  // build a basis for the space containing the element,
  // Observe that the space spanned by this set of basis is a vector space in the proper sense
  // (the plane passes throught zero). To cope with affine spaces getL2Distance of class Geometry
  // accepts an offset parameter representing the point the space passes throught
  std::vector<SVector<N>> basis;
  for(size_t i = 0; i < M; ++i){
    basis.push_back(coords[i+1] - coords[0]);
  }
  
  // if the distance between the point projection into the plane and the point itself is larger than 0
  // return false, the point does not belong to the plane and therefore cannot belong to the surface element
  if(Geometry<N>::getL2Distance(basis, coords[0], x) > std::numeric_limits<double>::epsilon()){
    return false;
  }
  
  // if the point belongs to the spanned space, check if its barycentric coordinates are all positive
  SVector<N> baryCoord = computeBarycentricCoordinates(x);

  // use Eigen visitor to check for positiveness of elements
  return (baryCoord.array() >= 0).all();
}

// return true if the element has at least one vertex on the boundary of the domain
template <unsigned int M, unsigned int N>
bool Element<M,N>::isOnBoundary(void) const{
  for(size_t i = 0; i < N_VERTICES(M,N); ++i){
    if(boundaryMarkers[i].second == 1) return true;
  }
  return false;
}
