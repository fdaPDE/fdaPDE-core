#include "SplineBasis.h" // implementation of SplineBasis.h

SplineBasis::SplineBasis(const std::vector<double>& knotsVector, int order){
  // pad the knot vector to obtain a full basis for the whole knot span [u_0, u_n]
  for(std::size_t i = 0; i < order; ++i) 
    knotsVector_.push_back(knotsVector[0]);
  knotsVector_.insert(knotsVector_.end(), knotsVector.begin(), knotsVector.end());
  for(std::size_t i = 0; i < order; ++i)
    knotsVector_.push_back(knotsVector[knotsVector.size()-1]);

  // build spline basis
  for(std::size_t k = 0; k < knotsVector_.size() - order; ++k){ // create spline at each iteration
    basis_.push_back(Spline(knotsVector_, k, order));
  }
}

// expose iterators
typename std::vector<Spline>::const_iterator
SplineBasis::begin() const {
  return basis_.cbegin();
}
typename std::vector<Spline>::const_iterator
SplineBasis::end() const {
  return basis_.cend();
}
