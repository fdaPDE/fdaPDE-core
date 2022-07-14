// applies a barycentric walk search
template <unsigned int M, unsigned int N>
std::shared_ptr<Element<M,N>> BarycentricWalk<M,N>::search(const SVector<N>& point){

  // start from an element at random
  std::shared_ptr<Element<M,N>> element = mesh_.requestElementById(uniform_int(rng)); 

  if(element->contains(point)){
    return element;
  }
  
  while(!element->contains(point)){
    // compute barycantric coordinates with respect to the element
    SVector<N+1> baryCoord = element->computeBarycentricCoordinates(point);
  
    // Pick the vertices corresponding to the n highest coordinates, and move into the adjacent element that
    // shares those vertices. This is equivalent to find the minimum baricentric coordinate and move to
    // the element adjacent to the face opposite to this point
    unsigned int minBaryCoordIndex;
    baryCoord.minCoeff(&minBaryCoordIndex);

    // by construction barycentric coordinate at position i is relative to vertex i of the mesh element
    // we can move to the next element in O(1) exploiting the memory representation of mesh in memory
    unsigned int nextID = element->getNeighbors()[minBaryCoordIndex];
    element = mesh_.requestElementById(nextID);
  }
  
  return element;
}
