template <unsigned int N>
bool Query<N>::intersect(const rectangle<N> &rect) const {
  // keep track along which dimension query and rect intersects
  std::array<bool, N> boolVector{};  
  for(size_t dim = 0; dim < N; ++dim){
    // load here once, access it fast
    double qs = queryRange_.second[dim], qf = queryRange_.first[dim];
    double rs = rect.second[dim],        rf = rect.first[dim];
    
    if(// partially overlapping sides
       (qs > rf && rs > qf) || (rs > qf && qs > rf) ||
       // queryRange_ contained into rect or viceversa 
       (rf < qf && qs < rs) || (qf < rf && rs < rs)){

      boolVector[dim] = true; // query intersect rectangle along this dimension
    }
  }
  // query and rect intersects if and only if they intersects along each dimension
  bool result = true;
  for(bool b : boolVector) result &= b;
  
  return result;
}

template <unsigned int N>
bool Query<N>::contains(const SVector<N> &point) const {
  // cycle over dimensions
  for(size_t dim = 0; dim < N; ++dim){
    // return false if point lies outside 
    if(point[dim] > queryRange_.second[dim] || point[dim] < queryRange_.first[dim])
      return false;
  }
  return true;
}
