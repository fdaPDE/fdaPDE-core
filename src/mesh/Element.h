#ifndef __ELEMENT_H__
#define __ELEMENT_H__

#include "../utils/Symbols.h"
#include "../utils/CompileTime.h"
#include "../NLA/VectorSpace.h"
#include <cstddef>
using fdaPDE::core::NLA::VectorSpace;

#include <Eigen/LU>
#include <limits>
#include <array>

namespace fdaPDE{
namespace core{
namespace MESH{

  // M local dimension, N embedding dimension, R order of the element (defaulted to linear finite elements)
  template <unsigned int M, unsigned int N, unsigned int R = 1> class Element; // forward declaration

  // using declarations for manifold specialization
  template <unsigned int R> using SurfaceElement = Element<2,3,R>;
  template <unsigned int R> using NetworkElement = Element<1,2,R>;

  // compile time evaluation of the number of degrees of freedom associated to an element of dimension M and order R
  constexpr unsigned int ct_nnodes(const unsigned int M, const unsigned int R) {
    return ct_factorial(M+R)/(ct_factorial(M)*ct_factorial(R));
  }
  // number of vertices of an M-dimensional simplex
  constexpr unsigned int ct_nvertices(const unsigned int M) { return M+1; }
  // number of edges of an M-dimensional simplex
  constexpr unsigned int ct_nedges(const unsigned int M) { return (M*(M+1))/2; }
  
  // A single mesh element. This object represents the main **geometrical** abstraction of a physical element.
  // No functional information is carried by instances of this object.
  template <unsigned int M, unsigned int N, unsigned int R>
  class Element{
  private:
    std::size_t ID_; // ID of this element
    std::array<std::size_t, ct_nvertices(M)> nodeIDs_{}; // ID of nodes composing the element
    // coordinates of nodes as N-dimensional points. The i-th element in this array refers to node with ID nodeIDs_[i]
    std::array<SVector<N>,  ct_nvertices(M)> coords_{};
    // ID of the neighboring elements, use std::vector since number of neighboring elements is not always known at compile time
    std::vector<int> neighbors_{};
    bool boundary_; // true if the element has at least one vertex on the boundary
        
    // matrices defining the affine transformation from cartesian to barycentric coordinates and viceversa
    Eigen::Matrix<double, N, M> barycentricMatrix_{};
    Eigen::Matrix<double, M, N> invBarycentricMatrix_{};
    // measure of the element (precomputed and cached at construction time)
    double measure_ = 0;
  public:
    // constructor
    Element() = default;  
    Element(std::size_t ID, const std::array<std::size_t, ct_nvertices(M)>& nodeIDs, const std::array<SVector<N>, ct_nvertices(M)>& coords,
	    const std::vector<int>& neighbors, bool boundary);
    
    // getters (read-only mode)
    const std::array<SVector<N>, ct_nvertices(M)> coords() const { return coords_; }
    const std::vector<int> neighbors() const { return neighbors_; }
    const unsigned int ID() const { return ID_; }
    Eigen::Matrix<double, N, M> barycentricMatrix() const { return barycentricMatrix_; }
    Eigen::Matrix<double, M, N> invBarycentricMatrix() const { return invBarycentricMatrix_; }
    std::array<std::size_t, ct_nvertices(M)> nodeIDs() const { return nodeIDs_; }
    
    // computes the baricentric coordinates of a point with respect to this element
    SVector<M+1> toBarycentricCoords(const SVector<N>& x) const;
    SVector<N> midPoint() const; // computes the midpoint of the element

    // check if a given point is inside the element
    template <bool is_manifold = (N!=M)>
    typename std::enable_if<!is_manifold, bool>::type
    contains(const SVector<N>& x) const;
    // specialization for manifold case
    template <bool is_manifold = (N!=M)>
    typename std::enable_if< is_manifold, bool>::type
    contains(const SVector<N>& x) const;

    // compute bounding box of this element. A bounding box is the smallest rectangle containing the element
    // this construction is usefull in searching problems.
    std::pair<SVector<N>, SVector<N>> boundingBox() const;
    bool isOnBoundary() const { return boundary_; } // true if the element has at least one node on the boundary
    VectorSpace<M, N> spannedSpace() const; // vector space passing throught this element
    double measure() const { return measure_; } // measure of the element
    
    // subscript operator
    SVector<N> operator[](std::size_t i) const {
      static_assert(i < ct_nvertices(M));
      return coords_[i];
    };
    // allow range for over element's coordinates
    typename std::array<SVector<N>, ct_nvertices(M)>::const_iterator begin() const { return coords_.cbegin(); }
    typename std::array<SVector<N>, ct_nvertices(M)>::const_iterator end() const { return coords_.cend(); }
    
    // expose compile time informations
    static constexpr unsigned int nodes = ct_nnodes(M,R);
    static constexpr unsigned int vertices = ct_nvertices(M);
    static constexpr unsigned int local_dimension = M;
    static constexpr unsigned int embedding_dimension = N;
    static constexpr unsigned int order = R;
    
  };

#include "Element.tpp"
}}}

#endif // __ELEMENT_H__
