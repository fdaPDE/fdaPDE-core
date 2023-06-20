#ifndef __BASIS_TABLE_H__
#define __BASIS_TABLE_H__

namespace fdaPDE {
namespace core {
namespace FEM {

  // a functor representing a finite element over a pyhsical element e written as function of a reference basis B
  template<unsigned int M, unsigned int N, unsigned int R, typename B>
  class FiniteElement {
  private:
    using element_type = typename B::element_type;
    
    std::size_t node_;        // if the element is written as \phi_i, i is the value of node_
    const Element<M,N,R>& e_; // physical element over which the basis is defined
    const element_type& referenceBasis_; // basis wrt which this finite element is written
  public:
    FiniteElement(std::size_t node, const Element<M,N,R>& e, const element_type& referenceBasis)
      : node_(node), e_(e), referenceBasis_(referenceBasis) {};

    // compiler is better at inlining functors than general std::function<>
    inline double operator()(const SVector<N>& x) const {
      // map x into reference element
      SVector<N> p = e_.invBarycentricMatrix()*(x - e_.coords()[0]);
      return referenceBasis_(p); // evaluate reference basis at p
    }
    std::size_t node() const { return node_; }
  };

  // data structure used as cache for basis elements built over the PDE's domain.
  // the i-th element of this cache returns the set of basis functions built over the i-th element of the mesh
  template <unsigned int M,  unsigned int N, unsigned int R, typename B>
  using BASIS_TABLE = std::vector<std::vector<FiniteElement<M,N,R,B>>>;

}}}

#endif // __BASIS_TABLE_H__
