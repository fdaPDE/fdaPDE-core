// scans the whole mesh searching for the element containing the given point
template <unsigned int M, unsigned int N>
template <typename... Args>
std::unique_ptr<Element<M, N>> BruteForce<M, N>::search(const SVector<N>& point, Args&... args) {
  // loop over all mesh. Cycle using auto to let copy elision on returned value
  for(auto element : mesh_){
    if(element->contains(point)){
      (args(element, point), ...); // parameter pack expansion to call functor on the pair (element, point).
      return element;
    }
  }
  // no element in mesh found
  return nullptr;
}
