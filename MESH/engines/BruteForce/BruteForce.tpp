// scans the whole mesh searching for the element containing the given point
template <unsigned int M, unsigned int N, unsigned int R>
template <typename... Args>
std::shared_ptr<Element<M,N,R>> BruteForce<M,N,R>::search(const SVector<N>& point, Args&... args) const {
  // loop over all mesh.
  for(const auto& element : mesh_){
    if(element->contains(point)){
      (args(element, point), ...); // parameter pack expansion to call functor on the pair (element, point).
      return element;
    }
  }
  // no element in mesh found
  return nullptr;
}
