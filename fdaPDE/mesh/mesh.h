// This file is part of fdaPDE, a C++ library for physics-informed
// spatial and functional data analysis.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#ifndef __MESH_H__
#define __MESH_H__

#include <Eigen/Core>
#include <array>
#include <memory>
#include <type_traits>
#include <unordered_set>
#include <vector>

#include "../utils/IO/CSVReader.h"
#include "../utils/symbols.h"
#include "element.h"
#include "reference_element.h"

namespace fdapde {
namespace core {

// trait to detect if the mesh is a manifold
template <int M, int N> struct is_manifold {
    static constexpr bool value = (M != N);
};

// trait to detect if the mesh is a linear network
template <int M, int N> struct is_linear_network {
    static constexpr bool value = std::conditional<(M == 1 && N == 2), std::true_type, std::false_type>::type::value;
};

// trait to select a proper neighboring storage structure depending on the type of mesh. In case of linear networks this
// information is stored as a sparse matrix where entry (i,j) is set to 1 if and only if elements i and j are neighbors
template <int M, int N> struct neighboring_structure {
    using type = typename std::conditional<is_linear_network<M, N>::value, SpMatrix<int>, DMatrix<int>>::type;
};

// access to domain's triangulation, M: tangent space dimension, N: embedding space dimension, R: FEM mesh order
template <int M, int N, int R = 1> class Mesh {
   private:
    // coordinates of points costituting the vertices of mesh elements
    DMatrix<double> points_ {};
    int num_nodes_ = 0;
    // identifiers of points (as row indexes in points_ matrix) composing each element, by row
    Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> elements_ {};
    int num_elements_ = 0;
    // vector of binary coefficients such that, boundary_[j] = 1 \iff node j is on boundary
    DMatrix<int> boundary_ {};
    std::size_t dof_;   // degrees of freedom, i.e. the maximmum ID in the dof_table
    // build an enumeration of nodes coherent with the mesh topology, update the boundary structure and elements_ table
    void dof_enumerate(const DMatrix<int>& boundary);
    typename neighboring_structure<M, N>::type neighbors_ {};
    std::array<std::pair<double, double>, N> range_ {};   // mesh bounding box

    // elements informations are computed once and cached here for fast re-access
    std::vector<Element<M, N, R>> cache_ {};
    void fill_cache();
   public:
    Mesh() = default;
    // construct from .csv files, strings are names of file where raw data is contained
    Mesh(const std::string& points, const std::string& triangles, const std::string& neighbors,
         const std::string& boundary);

    // construct directly from eigen matrices
    Mesh(const DMatrix<double>& points, const DMatrix<int>& elements,
         const typename neighboring_structure<M, N>::type& neighbors, const DMatrix<int>& boundary);

    // getters
    const Element<M, N, R>& element(int ID) const { return cache_[ID]; }
    Element<M, N, R>& element(int ID) { return cache_[ID]; }
    SVector<N> node(int ID) const { return points_.row(ID); }
    bool is_on_boundary(size_t j) const { return boundary_(j) == 1; }
    int elements() const { return num_elements_; }
    int nodes() const { return num_nodes_; }
    std::array<std::pair<double, double>, N> range() const { return range_; }

    // support for the definition of a finite element basis
    std::size_t dof() const { return dof_; }   // number of degrees of freedom
    const Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& dof_table() const { return elements_; }
    DMatrix<double> dof_coords() const;   // coordinates of points supporting a finite element basis

    // iterators support
    struct iterator {   // range-for loop over mesh elements
       private:
        friend Mesh;
        const Mesh* mesh_container_;   // pointer to mesh object
        int index_;                    // keep track of current iteration during for-loop

        // constructor
        iterator(const Mesh* container, int index) : mesh_container_(container), index_(index) {};
       public:
        // just increment the current iteration and return this iterator
        iterator& operator++() {
            ++index_;
            return *this;
        }
        const Element<M, N, R>& operator*() { return mesh_container_->element(index_); }
        friend bool operator!=(const iterator& lhs, const iterator& rhs) { return lhs.index_ != rhs.index_; }
        const Element<M, N, R>& operator*() const { return mesh_container_->element(index_); }
    };

    struct boundary_iterator {   // range-for loop over boundary nodes
       private:
        friend Mesh;
        const Mesh* mesh_container_;
        int index_;   // current boundary node

        // constructor
        boundary_iterator(const Mesh* container, int index) : mesh_container_(container), index_(index) {};
       public:
        // fetch next boundary node
        boundary_iterator& operator++() {
            index_++;
            // scan until all nodes have been visited or a boundary node is not found
            for (; index_ < mesh_container_->dof_ && mesh_container_->is_on_boundary(index_) != true; ++index_)
                ;
            return *this;
        }
        int operator*() const { return index_; }
        friend bool operator!=(const boundary_iterator& lhs, const boundary_iterator& rhs) {
            return lhs.index_ != rhs.index_;
        }
    };
    iterator begin() const { return iterator(this, 0); }
    iterator end() const { return iterator(this, elements_.rows()); }
    boundary_iterator boundary_begin() const { return boundary_iterator(this, 0); }
    boundary_iterator boundary_end() const { return boundary_iterator(this, dof_); }

    // compile time informations
    static constexpr bool is_manifold = is_manifold<M, N>::value;
    enum {
        local_dimension = M,
        embedding_dimension = N,
        order = R,
        n_vertices = ct_nvertices(M),
        n_edges = ct_nedges(M),
        n_dof_per_element = ct_nnodes(M, R),
        n_dof_per_edge = R - 1,
        n_dof_internal = n_dof_per_element - (M + 1) - n_edges * (R - 1)   // > 0 \iff R > 2
    };
};

// alias exports
template <int R = 1> using Mesh2D = Mesh<2, 2, R>;
template <int R = 1> using Mesh3D = Mesh<3, 3, R>;
// manifold cases
template <int R = 1> using SurfaceMesh = Mesh<2, 3, R>;
template <int R = 1> using NetworkMesh = Mesh<1, 2, R>;

// implementative details

// builds a node enumeration for the support of a basis of order R. This fills both the elements_ table
// and recompute the boundary informations. (support only for order 2 basis)
template <int M, int N, int R>
void Mesh<M, N, R>::dof_enumerate(const DMatrix<int>& boundary) {
    // algorithm initialization
    int next = num_nodes_;   // next valid ID to assign
    // map of already assigned IDs
    std::unordered_map<std::pair<int, int>, std::array<int, n_dof_per_edge>, fdapde::pair_hash> assigned;
    std::size_t col = 0;   // current column of the elements_ table to change
    std::vector<bool> on_boundary(ct_nnodes(M, R) * num_nodes_, false);

    // start enumeration
    for (std::size_t elem = 0; elem < elements_.rows(); ++elem) {
        // consider all pairs of nodes (all edges)
        for (std::size_t i = 0; i < n_vertices; ++i) {
            for (std::size_t j = i + 1; j < n_vertices; ++j) {
                // check if edge (elements_[i], elements_[j]) has an already assigned number
                std::pair<int, int> edge = std::minmax(elements_(elem, i), elements_(elem, j));
                auto it = assigned.find(edge);
                if (it != assigned.end()) {   // there is an already assigned index
                    for (std::size_t z = 0; z < n_dof_per_edge; ++z, ++col) {
                        elements_(elem, M + 1 + col) = it->second[z];
                    }
                } else {
                    for (std::size_t z = 0; z < n_dof_per_edge; ++z, ++col, ++next) {
                        elements_(elem, M + 1 + col) = next;
                        assigned[edge][z] = next;
                        // new node is on boundary iff both endpoints of its edge are on boundary
                        if (boundary(edge.first, 0) && boundary(edge.second, 0)) on_boundary[next] = true;
                    }
                }
            }
        }
        // insert not-shared dofs if required (nodes internal to the current element)
        for (std::size_t i = 0; i < n_dof_internal; ++i, ++col, ++next) { elements_(elem, M + 1 + col) = next; }
        col = 0;   // reset column counter
    }
    dof_ = next;   // store degrees of freedom
    // adjust boundary informations
    boundary_.resize(dof_, 1);
    boundary_.topRows(boundary.rows()) = boundary;
    for (std::size_t i = num_nodes_; i < dof_; ++i) {   // adjust for new nodes of the enumeration
        if (on_boundary[i])
            boundary_(i, 0) = 1;
        else
            boundary_(i, 0) = 0;
    }
    return;
}

// construct directly from raw eigen matrix (used from wrappers)
template <int M, int N, int R>
Mesh<M, N, R>::Mesh(
  const DMatrix<double>& points, const DMatrix<int>& elements,
  const typename neighboring_structure<M, N>::type& neighbors, const DMatrix<int>& boundary) :
    points_(points) {
    // realign indexes (we assume index coming from mesh generator to be greater or equal to 1, C++ starts count from 0)
    if constexpr (!is_linear_network<M, N>::value)
        neighbors_ = (neighbors.array() - 1).matrix();
    else
        neighbors_ = neighbors;   // adjacency matrix is directly given as input as sparse matrix

    // compute dof_table
    elements_.resize(elements.rows(), ct_nnodes(M, R));
    elements_.leftCols(elements.cols()) = (elements.array() - 1).matrix();
    // store number of nodes and number of elements
    num_nodes_ = points_.rows();
    num_elements_ = elements_.rows();
    if constexpr (R > 1)
        dof_enumerate(boundary);
    else {
        // for order 1 meshes the functional basis is built over the same vertices which define the mesh geometry,
        // nothing to do set boundary structure as coming from data and dof as number of mesh nodes
        boundary_ = boundary;
        dof_ = num_nodes_;
    }

    // compute mesh limits
    for (size_t dim = 0; dim < N; ++dim) {
        range_[dim].first = points_.col(dim).minCoeff();
        range_[dim].second = points_.col(dim).maxCoeff();
    }
    // scan the whole mesh and precompute here once all elements' abstractions for fast access
    fill_cache();
    // end of initialization
    return;
}

// construct a mesh from .csv files
template <int M, int N, int R>
Mesh<M, N, R>::Mesh(
  const std::string& points, const std::string& elements, const std::string& neighbors, const std::string& boundary) {
    // open and parse CSV files
    CSVReader<double> d_reader;
    CSVReader<int> i_reader;
    CSVFile<double> points_data = d_reader.parseFile(points);
    CSVFile<int> elements_data = i_reader.parseFile(elements);
    CSVFile<int> boundary_data = i_reader.parseFile(boundary);
    // in the following subtract 1 for index realignment

    // load neighboring informations
    typename std::conditional<!is_linear_network<M, N>::value, CSVFile<int>, CSVSparseFile<int>>::type neighbors_data;
    if constexpr (!is_linear_network<M, N>::value) {
        neighbors_data = i_reader.parseFile(neighbors);
        // move parsed file to eigen dense matrix, recall that a negative value means no neighbor
        neighbors_ = neighbors_data.toEigen();
        neighbors_ = (neighbors_.array() - 1).matrix();
    } else {
        // handle sparse matrix storage of neighboring information in case of linear network
        neighbors_data = i_reader.parseSparseFile(neighbors);
        neighbors_ =
          neighbors_data.toEigen();   // .toEigen() of CSVSparseFile already subtract 1 to indexes for reaglignment
    }

    // bring parsed informations to matrix-like structures
    points_ = points_data.toEigen();
    num_nodes_ = points_.rows();
    // compute mesh range
    for (size_t dim = 0; dim < N; ++dim) {
        range_[dim].first = points_.col(dim).minCoeff();
        range_[dim].second = points_.col(dim).maxCoeff();
    }

    // compute dof_table
    elements_.resize(elements_data.rows(), ct_nnodes(M, R));
    elements_.leftCols(elements_data.cols()) = (elements_data.toEigen().array() - 1).matrix();
    // store number of elements
    num_elements_ = elements_.rows();
    if constexpr (R > 1)
        dof_enumerate(boundary_data.toEigen());
    else {
        // for order 1 meshes the functional basis is built over the same vertices which define the mesh geometry,
        // nothing to do set boundary structure as coming from data and dof as number of mesh nodes
        boundary_ = boundary_data.toEigen();
        dof_ = num_nodes_;
    }
    // scan the whole mesh and precompute here once all elements' abstractions for fast access
    fill_cache();
    // end of initialization
    return;
}

// fill the cache_ data structure with pointers to element objects
template <int M, int N, int R> void Mesh<M, N, R>::fill_cache() {
    // reserve space for cache
    cache_.reserve(num_elements_);

    // cycle over all possible elements' ID
    for (std::size_t ID = 0; ID < num_elements_; ++ID) {
        // degrees of freedom associated with this element
        auto point_data = elements_.row(ID);
        auto neighboring_data = neighbors_.row(ID);   // neighboring structure

        // prepare element
        std::array<std::size_t, ct_nvertices(M)> node_ids {};
        std::array<SVector<N>, ct_nvertices(M)> coords {};
        // number of neighbors may be not known at compile time in case linear network elements are employed, use a
        // dynamic data structure to handle 1.5D case as well transparently
        std::vector<int> neighbors {};
        // boundary informations, the element is on boundary <-> at least one node with ID point_data[i] is on boundary
        bool boundary = false;

        for (size_t i = 0; i < ct_nvertices(M); ++i) {
            SVector<N> node(points_.row(point_data[i]));   // coordinates of node
            coords[i] = node;
            // global ID of the node in the mesh
            node_ids[i] = point_data[i];
            boundary |= (boundary_(point_data[i]) == 1);

            if constexpr (!is_linear_network<M, N>::value) {
                // from triangle documentation: The first neighbor of triangle i is opposite the first corner of
                // triangle i, and so on. by storing neighboring informations as they come from triangle we have that
                // neighbor[0] is the triangle adjacent to the face opposite to coords[0]. This is true for any mesh
                // different from a network mesh
                neighbors.push_back(neighboring_data[i]);
            }
        }
        // fill neighboring information for the linear network element case
        if constexpr (is_linear_network<M, N>::value) {
            for (Eigen::SparseMatrix<int>::InnerIterator sp_mat_it(neighbors_, ID); sp_mat_it; ++sp_mat_it) {
                neighbors.push_back(sp_mat_it.row());   // neighbors_ is stored in ColumnMajor mode
            }
        }
        // cache constructed element
        cache_.emplace_back(ID, node_ids, coords, neighbors, boundary);
    }
}

// produce the matrix of dof coordinates
template <int M, int N, int R> DMatrix<double> Mesh<M, N, R>::dof_coords() const {
    if constexpr (R == 1)
        return points_;   // for order 1 meshes dofs coincide with vertices
    else {
        // allocate space
        DMatrix<double> coords;
        coords.resize(dof_, N);
        coords.topRows(num_nodes_) = points_;      // copy coordinates of elements' vertices
        std::unordered_set<std::size_t> visited;   // set of already visited dofs
        // define reference element
        std::array<SVector<M + 1>, ct_nnodes(M, R)> ref_coords = ReferenceElement<M, R>().bary_coords;

        // cycle over all mesh elements
        for (std::size_t i = 0; i < elements_.rows(); ++i) {
            // extract dofs related to element with ID i
            auto dofs = elements_.row(i);
            auto e = cache_[i];   // take reference to current physical element
            for (std::size_t j = ct_nvertices(M); j < ct_nnodes(M, R); ++j) {   // cycle only on non-vertex points
                if (visited.find(dofs[j]) == visited.end()) {                   // not yet mapped dof
                    // map points from reference to physical element
                    coords.row(dofs[j]) = e.barycentric_matrix() * ref_coords[j].template tail<M>() + e.coords()[0];
                    visited.insert(dofs[j]);
                }
            }
        }
        return coords;
    }
}

}   // namespace core
}   // namespace fdapde

#endif   // __MESH_H__
