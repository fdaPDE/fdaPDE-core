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
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "../utils/combinatorics.h"
#include "../utils/symbols.h"
#include "element.h"
#include "edge.h"
#include "reference_element.h"

namespace fdapde {
namespace core {

// trait to detect if a mesh is a manifold
template <int M, int N> struct is_manifold {
    static constexpr bool value = (M != N);
};

// trait to detect if a mesh is a linear network
template <int M, int N> struct is_linear_network {
    static constexpr bool value = std::conditional<(M == 1 && N == 2), std::true_type, std::false_type>::type::value;
};

// trait to select a proper neighboring storage structure depending on mesh type.
// use a sparse matrix for storage of adjacency matrix for linear networks
template <int M, int N> struct neighboring_structure {
    using type = typename std::conditional<is_linear_network<M, N>::value, SpMatrix<int>, DMatrix<int>>::type;
};
  
// access to domain's triangulation, M: tangent space dimension, N: embedding space dimension
template <int M, int N> class Mesh {
   private:
    // coordinates of points costituting the vertices of mesh elements
    DMatrix<double> nodes_ {};
    int n_nodes_ = 0;
    // identifiers of points (as row indexes in points_ matrix) composing each element, by row
    DMatrix<int, Eigen::RowMajor> elements_ {};
    int n_elements_ = 0;
    // vector of binary coefficients such that, boundary_[j] = 1 \iff node j is on boundary
    DMatrix<int> boundary_ {};
    typename neighboring_structure<M, N>::type neighbors_ {};
    std::array<std::pair<double, double>, N> range_ {};   // mesh bounding box
    // identifiers of points composing each edge of the mesh (linearly stored in a row-major format)
    std::vector<int> edges_ {};
    std::vector<int> edge_map_ {};   // the i-th row refers to the elements' identifiers insisting on the i-th edge
    int n_edges_ = 0;
  
    // precomputed set of elements (cached for fast access)
    std::vector<Element<M, N>> cache_ {};
    void fill_cache();
   public:
    Mesh() = default;
    Mesh(const DMatrix<double>& nodes, const DMatrix<int>& elements,
	 const typename neighboring_structure<M, N>::type& neighbors, const DMatrix<int>& boundary);

    // getters
    const Element<M, N>& element(int ID) const { return cache_[ID]; }
    Element<M, N>& element(int ID) { return cache_[ID]; }
    SVector<N> node(int ID) const { return nodes_.row(ID); }
    bool is_on_boundary(size_t j) const { return boundary_(j) == 1; }
    const DMatrix<double>& nodes() const { return nodes_; }
    const DMatrix<int, Eigen::RowMajor>& elements() const { return elements_; }
    const typename neighboring_structure<M, N>::type& neighbors() const { return neighbors_; }
    const DMatrix<int>& boundary() const { return boundary_; }
    int n_elements() const { return n_elements_; }
    int n_nodes() const { return n_nodes_; }
    std::array<std::pair<double, double>, N> range() const { return range_; }
    Eigen::Map<const DMatrix<int, Eigen::RowMajor>> edges() const {
        return Eigen::Map<const DMatrix<int, Eigen::RowMajor>>(edges_.data(), n_edges_, n_vertices_per_edge);
    }
    int n_edges() const { return n_edges_; }
    Edge<M, N> edge(int ID) const;
  
    // iterators support
    struct iterator {   // range-for loop over mesh elements
       private:
        friend Mesh;
        const Mesh* mesh_;
        int index_;   // current element
        iterator(const Mesh* mesh, int index) : mesh_(mesh), index_(index) {};
       public:
        // increment current iteration index and return this iterator
        iterator& operator++() {
            ++index_;
            return *this;
        }
        const Element<M, N>& operator*() { return mesh_->element(index_); }
        friend bool operator!=(const iterator& lhs, const iterator& rhs) { return lhs.index_ != rhs.index_; }
        const Element<M, N>& operator*() const { return mesh_->element(index_); }
    };

    struct boundary_iterator {   // range-for loop over boundary nodes
       private:
        friend Mesh;
        const Mesh* mesh_;
        int index_;   // current boundary node
        boundary_iterator(const Mesh* mesh, int index) : mesh_(mesh), index_(index) {};
       public:
        // fetch next boundary node
        boundary_iterator& operator++() {
            index_++;
            // scan until all nodes have been visited or a boundary node is not found
            for (; index_ < mesh_->n_nodes_ && mesh_->is_on_boundary(index_) != true; ++index_);
            return *this;
        }
        int operator*() const { return index_; }
        friend bool operator!=(const boundary_iterator& lhs, const boundary_iterator& rhs) {
            return lhs.index_ != rhs.index_;
        }
    };

    struct edge_iterator {   // range-for over edges
       private:
        friend Mesh;
        const Mesh* mesh_;
        int index_;
        edge_iterator(const Mesh* mesh, int index) : mesh_(mesh), index_(index) {};
       public:
        // increment current iteration index and return this iterator
        edge_iterator& operator++() {
            ++index_;
            return *this;
        }
        Edge<M, N> operator*() { return mesh_->edge(index_); }
        friend bool operator!=(const edge_iterator& lhs, const edge_iterator& rhs) { return lhs.index_ != rhs.index_; }
        Edge<M, N> operator*() const { return mesh_->edge(index_); }
    };

    iterator begin() const { return iterator(this, 0); }
    iterator end() const { return iterator(this, elements_.rows()); }
    boundary_iterator boundary_begin() const { return boundary_iterator(this, 0); }
    boundary_iterator boundary_end() const { return boundary_iterator(this, n_nodes_); }
    edge_iterator edge_begin() const { return edge_iterator(this, 0); }
    edge_iterator edge_end() const { return edge_iterator(this, n_edges_); }
  
    // compile time informations
    static constexpr bool is_manifold = is_manifold<M, N>::value;
    enum {
        local_dimension = M,
        embedding_dimension = N,
        n_vertices = ct_nvertices(M),
        n_edges_per_element = ct_nedges(M),
        n_neighbors = ct_nedges(M),
        n_vertices_per_edge = M, // generalize wrt other dimensionalities
	n_elements_per_edge = 2  // same here...
    };
};

// implementative details

// construct from raw matrices (NB: matrix enumeration is assumed to start from 0)
template <int M, int N>
Mesh<M, N>::Mesh(
  const DMatrix<double>& nodes, const DMatrix<int>& elements,
  const typename neighboring_structure<M, N>::type& neighbors, const DMatrix<int>& boundary) :
    nodes_(nodes), neighbors_(neighbors), elements_(elements), boundary_(boundary) {
    // store number of nodes and number of elements
    n_nodes_ = nodes_.rows();
    n_elements_ = elements_.rows();

    // compute mesh limits
    for (size_t dim = 0; dim < N; ++dim) {
        range_[dim].first = nodes_.col(dim).minCoeff();
        range_[dim].second = nodes_.col(dim).maxCoeff();
    }
    // scan the whole mesh and precompute elements informations for fast access
    fill_cache();

    // compute edges informations
    // edges_ are contigously stored in memory as a std::vector<int>, in a row major format
    auto edge_pattern = combinations<n_vertices_per_edge, n_vertices>();
    std::unordered_map<std::array<int, n_vertices_per_edge>, int, std_array_hash<int, n_vertices_per_edge>> visited;

    std::array<int, n_vertices_per_edge> edge;
    for (int i = 0; i < n_elements_; ++i) {
        for (int j = 0; j < edge_pattern.rows(); ++j) {
            // construct edge
            for (int k = 0; k < n_vertices_per_edge; ++k) { edge[k] = elements_(i, edge_pattern(j, k)); }
            // check if edge already processed
            std::sort(edge.begin(), edge.end());
            auto it = visited.find(edge);
            if (it != visited.end()) {
                // free memory (only two elements share the same edge) and update edge to element structure
                *(edge_map_.begin() + 2 * (it->second) + 1) = i;
                visited.erase(it);
            } else {
                // store edge and update edge to element information
                for (int k = 0; k < n_vertices_per_edge; ++k) { edges_.emplace_back(edge[k]); }
                visited.insert({edge, n_edges_});
                n_edges_++;
                edge_map_.insert(edge_map_.end(), {i, -1});   // -1 flags no incident element at edge
            }
        }
    }
    return;
}

// fill the cache_ data structure with pointers to element objects
template <int M, int N> void Mesh<M, N>::fill_cache() {
    // reserve space for cache
    cache_.reserve(n_elements_);

    // cycle over all possible elements' ID
    for (std::size_t ID = 0; ID < n_elements_; ++ID) {
        auto point_data = elements_.row(ID);
        auto neighboring_data = neighbors_.row(ID);
        // prepare element
        std::array<std::size_t, ct_nvertices(M)> node_ids {};
        std::array<SVector<N>, ct_nvertices(M)> coords {};
        // number of neighbors may be not known at compile time in case linear network elements are employed, use a
        // dynamic data structure to handle 1.5D case as well transparently
        std::vector<int> neighbors {};
        // boundary informations, the element is on boundary <-> at least one node with ID point_data[i] is on boundary
        bool boundary = false;

        for (size_t i = 0; i < ct_nvertices(M); ++i) {
            SVector<N> node(nodes_.row(point_data[i]));   // coordinates of node
            coords[i] = node;
            // global ID of the node in the mesh
            node_ids[i] = point_data[i];
            boundary |= (boundary_(point_data[i]) == 1);
            if constexpr (!is_linear_network<M, N>::value) {
                // the first neighbor of triangle i is opposite the first corner of triangle i, and so on
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

template <int M, int N> Edge<M, N> Mesh<M, N>::edge(int ID) const {
    // fetch nodes informations
    std::array<SVector<N>, n_vertices_per_edge> coords {}; 
    std::array<int, n_vertices_per_edge> node_ids {};
    bool on_boundary = true;
    for (std::size_t i = 0; i < n_vertices_per_edge; ++i) {
        coords[i] = SVector<N>(nodes_.row(i));
        node_ids[i] = *(edges_.begin() + ID * n_vertices_per_edge + i);
	on_boundary &= is_on_boundary(node_ids[i]);
    }
    // elements adjacent to this edge
    std::array<int, 2> elements_ids {*(edge_map_.begin() + 2 * ID), *(edge_map_.begin() + 2 * ID + 1)};
    return Edge<M, N>(ID, node_ids, coords, elements_ids, on_boundary);
}

// alias exports
using Mesh2D = Mesh<2, 2>;
using Mesh3D = Mesh<3, 3>;
using SurfaceMesh = Mesh<2, 3>;
using NetworkMesh = Mesh<1, 2>;
  
}   // namespace core
}   // namespace fdapde

#endif   // __MESH_H__
