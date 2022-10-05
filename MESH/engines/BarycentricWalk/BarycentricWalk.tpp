// applies a barycentric walk search
template <unsigned int M, unsigned int N, unsigned int R>
template <typename... Args>
std::shared_ptr<Element<M,N,R>> BarycentricWalk<M,N,R>::search(const SVector<N>& point, Args&... args) const {
  // define uniform distribution over the ID space
  std::random_device rng{};
  std::uniform_int_distribution<uint32_t> uniform_int = std::uniform_int_distribution<uint32_t>(0, mesh_.elements()-1);;
  
  // start from an element at random
  std::shared_ptr<Element<M,N,R>> element = mesh_.element(uniform_int(rng)); 

  if(element->contains(point)){
    (args(element, point), ...); // parameter pack expansion to call functor on the pair (element, point).
    return element;
  }
  
  while(!element->contains(point)){
    // compute barycantric coordinates with respect to the element
    SVector<N+1> baryCoord = element->toBarycentricCoords(point);
  
    // Pick the vertices corresponding to the n highest coordinates, and move into the adjacent element that
    // shares those vertices. This is equivalent to find the minimum baricentric coordinate and move to
    // the element adjacent to the face opposite to this point
    unsigned int minBaryCoordIndex;
    baryCoord.minCoeff(&minBaryCoordIndex);

    // by construction barycentric coordinate at position i is relative to vertex i of the mesh element
    // we can move to the next element in O(1) exploiting the memory representation of mesh in memory
    unsigned int nextID = element->neighbors()[minBaryCoordIndex];
    element = mesh_.element(nextID);
  }
  
  (args(element, point), ...); // parameter pack expansion to call functor on the pair (element, point).
  return element;
}
