#ifndef __SPLINE_BASIS_H__
#define __SPLINE_BASIS_H__

#include "../../utils/DataStructures/Tree.h"
using fdaPDE::core::node_ptr;
using fdaPDE::core::LinkDirection;
using fdaPDE::core::Tree;
#include "Spline.h"
using fdaPDE::core::FEM::Spline;
#include <vector>

namespace fdaPDE{
namespace core{
namespace FEM{

  // a SplineBasis is just a collection of Spline objects. This class is compliant with the functional basis concept adopted in the library.
  class SplineBasis{
  private:
    std::vector<double> knotsVector_{};  // vector of knots
    std::vector<Spline> basis_{};        // the spline basis
  public:
    using const_iterator = typename std::vector<Spline>::const_iterator;
    // constructor
    SplineBasis(const std::vector<double>& knotsVector, int order);
    // return i-th element of the basis
    const Spline& operator[](std::size_t i) const { return basis_[i]; }
    // return the number of basis elements
    int size() const { return basis_.size(); }
    // allow range-for over basis elements
    const_iterator begin() const;
    const_iterator end() const;
  };

#include "SplineBasis.tpp"

}}}
#endif // __SPLINE_BASIS_H__
