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

  // compile time evaluation of the number of nodes of a given element given its local dimension and order
  constexpr unsigned int ct_nnodes(const unsigned int M, const unsigned int R) {
    return ct_factorial(M+R)/(ct_factorial(M)*ct_factorial(R));
  }
  // compile time evaluation of the number of vertices of a given element
  constexpr unsigned int ct_nvertices(const unsigned int M) { return M+1; }

  // A single mesh element ( defaulted to linear finite element)
  // we assume that the first M+1 entries of any internal data structure handled by Element refer to the **vertices** of the element,
  // any other entry after the first (M+1)s are here to support functional informations, i.e. to build a functional basis of the proper order
  // over the element. Moreover, observe that the ordering of the vertices is of no importance and we simply adopt the same ordering coming
  // from mesh data (e.g. informations stored in .csv files or directly coming from front-ends).
  // Geometrical operations handled by the MESH module are only based on vertex nodes (there is no extra-knowledge from the perspective of the MESH
  // module in nodes which are not vertices). As such any internal implementation regarding the interface exposed by Element doesn't make
  // use of any node which is not a vertex one.
  template <unsigned int M, unsigned int N, unsigned int R>
  class Element{
  private:
    std::size_t ID_; // ID of this element
    std::array<std::size_t, ct_nvertices(M)> nodeIDs_{}; // ID of nodes composing the element
    // coordinates of nodes as N-dimensional points. The i-th element in this array refers to node with ID nodeIDs_[i]
    std::array<SVector<N>,  ct_nvertices(M)> coords_{};
    // ID of the neighboring elements, use std::vector since number of neighboring elements is not always known at compile time
    std::vector<int> neighbors_{};
    // boundary informations. boundary_[i] == 1 <-> node with ID nodeIDs_[i] is on boundary
    std::array<std::size_t, ct_nvertices(M)> boundary_{};
        
    // matrices defining the affine transformation from cartesian to barycentric coordinates and viceversa
    Eigen::Matrix<double, N, M> barycentricMatrix_{};
    Eigen::Matrix<double, M, N> invBarycentricMatrix_{};
    // measure of the element (precomputed and cached at construction time)
    double measure_ = 0;
  public:
    // constructor
    Element() = default;  
    Element(std::size_t ID, const std::array<std::size_t, ct_nvertices(M)>& nodeIDs, const std::array<SVector<N>, ct_nvertices(M)>& coords,
	    const std::vector<int>& neighbors, const std::array<std::size_t, ct_nvertices(M)>& boundary);
    
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

    // returns true if the element has at least one node on the boundary
    bool isOnBoundary(void) const;
    // returns a vector of pairs: <node id, node coordinates> for any node of the element on the boundary of the mesh
    std::vector<std::pair<std::size_t, SVector<N>>> boundaryNodes() const;
    // returns the vector space passing throught this element
    VectorSpace<M, N> spannedSpace() const;
    // returns the measure of the element
    double measure() const { return measure_; }
    
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
