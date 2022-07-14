#ifndef __SPLINE_BASIS_H__
#define __SPLINE_BASIS_H__

#include "../../utils/DataStructures/Tree.h"
using fdaPDE::core::node_ptr;
using fdaPDE::core::LinkDirection;
using fdaPDE::core::Tree;
#include "Spline.h"

#include <vector>

// a SplineBasis is just a collection of Spline objects. This class is compliant with the functional basis concept adopted in the library.
class SplineBasis{
private:
  std::vector<double> knotsVector_{};  // vector of knots
  std::vector<Spline> basis_{};        // the spline basis

public:
  // constructor
  SplineBasis(const std::vector<double>& knotsVector, int order) {
    
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

  const Spline& operator[](std::size_t i) const { return basis_[i]; }   // return i-th element of the basis
  int size() const { return basis_.size(); }                            // return the number of basis elements
};

#endif // __SPLINE_BASIS_H__
