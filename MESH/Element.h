#ifndef __ELEMENT_H__
#define __ELEMENT_H__

#include "../utils/Symbols.h"
#include "../NLA/VectorSpace.h"
using fdaPDE::core::NLA::VectorSpace;

#include <Eigen/LU>
#include <limits>
#include <array>

namespace fdaPDE{
namespace core{
namespace MESH{
  
  template <unsigned int M, unsigned int N> class Element; // forward declaration

  // using declarations for manifold specialization
  using SurfaceElement = Element<2,3>;
  using NetworkElement = Element<1,2>;

  // compile time evaluation of the number of vertices of a given element
  constexpr unsigned int N_VERTICES(const unsigned int M, const unsigned int N) {
    return M+1;
  }

  // A single mesh element
  template <unsigned int M, unsigned int N>
  class Element{
  private:
    std::size_t ID_; // ID of this element
    std::array<std::size_t, N_VERTICES(M,N)> nodeIDs_{}; // ID of nodes composing the element
    // coordinates of nodes as N-dimensional points. The i-th element in this array refers to node with ID nodeIDs_[i]
    std::array<SVector<N>,  N_VERTICES(M,N)> coords_{};
    // ID of the neighboring elements, use std::vector since number of neighboring elements is not always known at compile time
    std::vector<int> neighbors_{};
    // boundary informations. boundary_[i] == 1 <-> node with ID nodeIDs_[i] is on boundary
    std::array<std::size_t, N_VERTICES(M,N)> boundary_{};
    
    // Functional information to assist the FEM module. FEsupport contains pairs <ID, point> where
    //    * ID:    is the global ID of the node in the mesh (as row index in the points_ table)
    //    * point: are the node's coordinates
    // Observe that 'FESupport' is different from 'coords' for elements of order greater than one: in this case
    // the nodes to support a functional basis are more than the geometrical vertices of the element. From here
    // the need to keep the geometrical informations separate from the functional ones.
    std::array<std::pair<unsigned, SVector<N>>, N_VERTICES(M,N)> FEsupport_; // *************** ?????
    
    // matrices defining the affine transformation from cartesian to barycentric coordinates and viceversa
    Eigen::Matrix<double, N, M> barycentricMatrix_{};
    Eigen::Matrix<double, M, N> invBarycentricMatrix_{};

  public:
    using FESupport = std::array<std::pair<unsigned, SVector<N>>, N_VERTICES(M,N)>;

    // constructor
    Element() = default;  
    Element(std::size_t ID, const std::array<std::size_t, N_VERTICES(M,N)>& nodeIDs, const std::array<SVector<N>, N_VERTICES(M,N)>& coords,
	    const std::vector<int>& neighbors, const std::array<std::size_t, N_VERTICES(M,N)>& boundary, const FESupport& FEsupport);
    
    // getters (read-only mode)
    const std::array<SVector<N>, N_VERTICES(M,N)> coords() const { return coords_; }
    const std::vector<int> neighbors() const { return neighbors_; }
    const unsigned int ID() const { return ID_; }

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
    VectorSpace<N> spannedSpace() const;

    // ?????????????? parte dell'interfaccia ancora da definire, non coperta da tests
    //Eigen::Matrix<double, N, M> getBaryMatrix() const { return baryMatrix; }
    //Eigen::Matrix<double, M, N> getInvBaryMatrix() const { return invBaryMatrix; }
    FESupport getFESupport() const { return FEsupport_; }

  };

#include "Element.tpp"
}}}

#endif // __ELEMENT_H__
