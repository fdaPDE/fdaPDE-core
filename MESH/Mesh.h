#ifndef __MESH_H__
#define __MESH_H__

#include <Eigen/Core>
#include <type_traits>
#include <vector>
#include <memory>
#include <array>
#include <unordered_set>

#include "Element.h"
using fdaPDE::core::MESH::ct_nvertices;
#include "ReferenceElement.h"
using fdaPDE::core::MESH::ReferenceElement;
#include "../utils/IO/CSVReader.h"
#include "../utils/Symbols.h"

namespace fdaPDE{
namespace core{
namespace MESH{

  // Mesh is the access point to mesh informations (triangulated domains). It offers an abstraction layer allowing to reason
  // on the mesh from a geometrical perspective, i.e. without considering its internal representation in memory. The class is able to transparently
  // handle manifold and non manifold meshes, exposing the same interface in any case. Three non-type template parameters are used:
  //     * M: local dimension of the mesh (dimension of the space to which a mesh element belongs)
  //     * N: embeddding dimension of the mesh (dimension of the space where the whole mesh lies)
  //     * R: mesh order, the order of a finite element basis function which can be defined on the mesh
  // if M != N the mesh is a manifold. Currently are implemented:
  //     * 1.5D meshes (linear newtorks,    M=1, N=2)
  //     * 2D meshes   (planar domains,     M=2, N=2)
  //     * 2.5D meshes (surfaces,           M=2, N=3)
  //     * 3D meshes   (volumetric domains, M=3, N=3)
  
  // NB about internal implementaiton: special care is needed in the development of linear networks, since differently from other cases the number
  // of neighboing elements is not known at compile time. This implies the usage of specialized data structures

  // trait to detect if the mesh is a manifold
  template <unsigned int M, unsigned int N>
  struct is_manifold{
    static constexpr bool value = (M != N);
  };
  
  // trait to detect if the mesh is a linear network
  template <unsigned int M, unsigned int N>
  struct is_linear_network{
    static constexpr bool value = std::conditional<
      (M == 1 && N == 2), std::true_type, std::false_type
      >::type::value;
  };

  // trait to select a proper neighboring storage structure depending on the type of mesh. In case of linear networks this information is stored as
  // a sparse matrix where entry (i,j) is set to 1 if and only if elements i and j are neighbors
  template <unsigned int M, unsigned int N>
  struct neighboring_structure{
    using type = typename std::conditional<
      is_linear_network<M, N>::value, SpMatrix<int>, DMatrix<int>
      >::type;
  };
  
  template <unsigned int M, unsigned int N, unsigned int R = 1>
  class Mesh{
  private:
    // structural informations related to the construction of a mesh of order R
    static constexpr unsigned int n_vertices = ct_nvertices(M);
    static constexpr unsigned int n_edges = ct_nedges(M);
    static constexpr unsigned int n_dof_per_element = ct_nnodes(M,R);
    static constexpr unsigned int n_dof_per_edge = R-1;
    static constexpr unsigned int n_dof_internal = n_dof_per_element - (M+1) - n_edges*(R-1); // > 0 \iff R > 1
    
    // coordinates of points costituting the vertices of mesh elements
    DMatrix<double> points_{};
    unsigned int numNodes_ = 0;
    // matrix of elements in the triangulation. Each row of the matrix contains the points which made the triangle as row number in the points_
    // matrix. For a mesh of order 1 this matrix coincides with the element matrix passed as input.
    // For higher order meshes this matrix assumes the following format:
    // 
    //    | ----- M + 1 columns ----- | ---- ct_nnodes(M,R) - (M+1) columns ---- |
    //    |            ...            |                    ...                   | dof_table
    //    | ------------------------- | ---------------------------------------- |
    //                                 ct_nnodes(M,R)
    //
    // * the first M+1 columns contain geometrical informations required to define an Element objects, i.e. the vertices of the element itself
    // * the next (ct_nnodes(M,R)-(M+1)) columns contain the ID of nodes required for the definition of a finite element basis of order R, i.e.
    //   are the extra degrees of freedom needed for the support of a functional basis of order higher than 1. Observe that:
    //     * for mesh of order 1 this portion of the elements_ table is absent
    //     * in case R>1, MESH will not pyhsically create nodes supporting the functional basis. Only an enumeration of those nodes
    //       coherent with the mesh topology is generated at construction time.
    Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> elements_{};
    unsigned int numElements_ = 0;
    // store boundary informations. This is a vector of binary coefficients such that, boundary_[j] = 1 \iff node j is on boundary 
    DMatrix<int> boundary_{};
    std::size_t dof_; // degrees of freedom, i.e. the maximmum ID in the dof_table
    // build an enumeration of nodes coherent with the mesh topology, update the boundary structure and dof_ to reflect the enumeration
    void DOFenumerate(const DMatrix<int>& boundary);
    
    // in case of non linear-networks neighbors_ is a dense matrix where row i contains the indexes as row number in triangles_ matrix of the
    // neighboring triangles to triangle i (all triangles in the triangulation which share an edge with i). In case of linear-newtorks neighbors_
    // is a sparse matrix where entry (i,j) is set to 1 iff i and j are neighbors
    typename neighboring_structure<M, N>::type neighbors_{};
    
    // store min-max values for each dimension of the mesh
    std::array<std::pair<double, double>, N> range_{};

    // elements informations are computed once and cached here for fast re-access
    std::vector<std::shared_ptr<Element<M,N,R>>> cache_{};
    void fill_cache();
  public:
    Mesh() = default;
    // construct from .csv files, strings are names of file where raw data is contained
    Mesh(const std::string& points,    const std::string& edges, const std::string& triangles,
	 const std::string& neighbors, const std::string& boundary);

    // construct directly from eigen matrices
    Mesh(const DMatrix<double>& points, const DMatrix<int>& edges, const DMatrix<int>& elements,
	 const typename neighboring_structure<M, N>::type& neighbors, const DMatrix<int>& boundary);
    
    // returns an element object given its ID (its row number in the elements_ matrix) from raw (matrix-like) informations
    std::shared_ptr<Element<M,N,R>> element(unsigned int ID) const;
    // return the coordinate of a node given its ID (its row number in the points_ matrix)
    SVector<N> node(unsigned int ID) const;
    // return true if the given node is on boundary, false otherwise
    bool isOnBoundary(size_t j) const { return boundary_(j) == 1; }
    
    // getters
    unsigned int elements() const { return numElements_; }
    unsigned int nodes() const { return numNodes_; }
    std::array<std::pair<double, double>, N> range() const { return range_; }

    // support for the definition of finite element basis over a mesh object
    std::size_t dof() const { return dof_; } // number of degrees of freedom
    const Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& dof_table() const { return elements_; }
    // coordinates of points supporting a finite element basis
    DMatrix<double> dofCoords() const; 

    // iterators support
    #include "MeshIterators.h"
    // provide begin() and end() methods
    iterator begin() const { return iterator(this, 0); }
    iterator end()   const { return iterator(this, elements_.rows()); }
    // access to boundary iterators
    boundary_iterator boundary_begin() const { return boundary_iterator(this, 0); }
    boundary_iterator boundary_end()   const { return boundary_iterator(this, dof_); }
    
    // expose compile time informations to outside
    static constexpr bool manifold = is_manifold<M, N>::value;
    static constexpr unsigned int local_dimension = M;
    static constexpr unsigned int embedding_dimension = N;
    static constexpr unsigned int order = R;
  };

  // export some aliases
  template <unsigned int R=1> using Mesh2D = Mesh<2,2,R>;
  template <unsigned int R=1> using Mesh3D = Mesh<3,3,R>;
  // manifold cases
  template <unsigned int R=1> using SurfaceMesh = Mesh<2,3,R>;
  template <unsigned int R=1> using NetworkMesh = Mesh<1,2,R>;

#include "Mesh.tpp"
}}}
  
#endif // __MESH_H__
