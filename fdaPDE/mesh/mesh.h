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
#include "mesh_utils.h"
#include "reference_element.h"
#include "point_location/point_location_base.h"
#include "point_location/adt.h"

namespace fdapde {
namespace core {

// trait to select a proper neighboring storage structure depending on mesh type.
template <int M, int N> struct neighboring_structure {
    using type = typename std::conditional<is_network<M, N>::value, SpMatrix<int>, DMatrix<int>>::type;
};

template <int M, int N> class Mesh {
   protected:
    // physical coordinates of mesh's vertices
    DMatrix<double> nodes_ {};
    int n_nodes_ = 0;
    // identifiers of nodes (as row indexes in nodes_ matrix) composing each element, in a RowMajor format
    DMatrix<int, Eigen::RowMajor> elements_ {};
    int n_elements_ = 0;
    // vector of binary coefficients such that, boundary_[j] = 1 \iff node j is on boundary
    DMatrix<int> boundary_ {};
    SMatrix<2, N> range_ {};   // mesh bounding box (column i maps to the i-th dimension)
    typename neighboring_structure<M, N>::type neighbors_ {};
    // identifiers of nodes composing each facet of the mesh (linearly stored in a row-major format)
    std::vector<int> facets_ {};
    std::unordered_map<int, std::vector<int>> facet_to_element_ {};   // map from facet id to elements insisting on it
    int n_facets_ = 0;
    // identifiers of nodes composing each edge (for 2D and 2.5D, edges coincide with facets)
    std::vector<int> edges_ {};
    std::unordered_map<int, std::vector<int>> edge_to_element_ {};   // map from edge id to elements insisting on it
    int n_edges_ = 0;

    // precomputed set of elements
    std::vector<Element<M, N>> elements_cache_ {};
    mutable std::shared_ptr<PointLocationBase<M, N>> point_location_ = nullptr;
    using DefaultLocationPolicy = ADT<M, N>;
   public:
    Mesh() = default;
    // 2D, 2.5D, 3D constructor
    template <int M_ = M, int N_ = N,
	      typename std::enable_if<!is_network<M_, N_>::value, int>::type = 0>
    Mesh(const DMatrix<double>& nodes, const DMatrix<int>& elements, const DMatrix<int>& boundary);
    // linear network (1.5D) specialized constructor
    template <int M_ = M, int N_ = N,
	      typename std::enable_if< is_network<M_, N_>::value, int>::type = 0>
    Mesh(const DMatrix<double>& nodes, const DMatrix<int>& elements, const DMatrix<int>& boundary);
  
    // getters
    const Element<M, N>& element(int ID) const { return elements_cache_[ID]; }
    Element<M, N>& element(int ID) { return elements_cache_[ID]; }
    SVector<N> node(int ID) const { return nodes_.row(ID); }
    bool is_on_boundary(int ID) const { return boundary_(ID) == 1; }
    const DMatrix<double>& nodes() const { return nodes_; }
    const DMatrix<int, Eigen::RowMajor>& elements() const { return elements_; }
    const typename neighboring_structure<M, N>::type& neighbors() const { return neighbors_; }
    const DMatrix<int>& boundary() const { return boundary_; }
    int n_elements() const { return n_elements_; }
    int n_nodes() const { return n_nodes_; }
    SMatrix<2, N> range() const { return range_; }
    Eigen::Map<const DMatrix<int, Eigen::RowMajor>> facets() const {
        return Eigen::Map<const DMatrix<int, Eigen::RowMajor>>(facets_.data(), n_facets_, n_vertices_per_facet);
    }
    int n_facets() const { return n_facets_; }
    Facet<M, N> facet(int ID) const;

    int n_edges() const {
        static_assert(!is_network<M, N>::value);
        return is_3d<M, N>::value ? n_facets_ : n_edges_;
    }
    Eigen::Map<const DMatrix<int, Eigen::RowMajor>> edges() const {
        static_assert(!is_network<M, N>::value);
        if constexpr (is_3d<M, N>::value) {
            return Eigen::Map<const DMatrix<int, Eigen::RowMajor>>(edges_.data(), n_edges_, n_vertices_per_edge);
        } else {
            return facets();
        }
    }
    DVector<int> locate(const DMatrix<double>& points) const {
        if (point_location_ == nullptr) point_location_ = std::make_shared<DefaultLocationPolicy>(*this);
        return point_location_->locate(points);
    }
    // getter and iterator on edges

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
            for (; index_ < mesh_->n_nodes_ && mesh_->is_on_boundary(index_) != true; ++index_)
                ;
            return *this;
        }
        int operator*() const { return index_; }
        friend bool operator!=(const boundary_iterator& lhs, const boundary_iterator& rhs) {
            return lhs.index_ != rhs.index_;
        }
    };

    struct facet_iterator {   // range-for over facets
       private:
        friend Mesh;
        const Mesh* mesh_;
        int index_;
        facet_iterator(const Mesh* mesh, int index) : mesh_(mesh), index_(index) {};
       public:
        // increment current iteration index and return this iterator
        facet_iterator& operator++() {
            ++index_;
            return *this;
        }
        Facet<M, N> operator*() const { return mesh_->facet(index_); }
        friend bool operator!=(const facet_iterator& lhs, const facet_iterator& rhs) {
            return lhs.index_ != rhs.index_;
        }
    };

    iterator begin() const { return iterator(this, 0); }
    iterator end() const { return iterator(this, elements_.rows()); }
    boundary_iterator boundary_begin() const { return boundary_iterator(this, 0); }
    boundary_iterator boundary_end() const { return boundary_iterator(this, n_nodes_); }
    facet_iterator facet_begin() const { return facet_iterator(this, 0); }
    facet_iterator facet_end() const { return facet_iterator(this, n_facets_); }

    // setters
    template <template <int, int> typename PointLocationPolicy_>
    void set_point_location_policy() {
        point_location_ = std::make_shared<PointLocationPolicy_<M, N>>(*this);
    }
  
    // compile time informations
    static constexpr bool is_manifold = (M != N);
    enum {
        local_dimension = M,
        embedding_dimension = N,
        n_vertices = ct_nvertices(M),
        n_vertices_per_edge = 2,
        n_vertices_per_facet = M,
        n_facets_per_element = ct_nfacets(M),
        n_neighbors_per_element = ct_nneighbors(M),
        n_elements_per_facet = 2
    };
};

// implementative details

template <int M, int N>
template <int M_, int N_, typename std::enable_if<!is_network<M_, N_>::value, int>::type>
Mesh<M, N>::Mesh(const DMatrix<double>& nodes, const DMatrix<int>& elements, const DMatrix<int>& boundary) :
    nodes_(nodes), elements_(elements), boundary_(boundary) {
    // store number of nodes and number of elements
    n_nodes_ = nodes_.rows();
    n_elements_ = elements_.rows();
    // compute mesh limits
    range_.row(0) = nodes_.colwise().minCoeff();
    range_.row(1) = nodes_.colwise().maxCoeff();

    // reserve neighbors_ storage (-1 in column i implies no neighbor adjacent to the facet opposite to vertex i)
    neighbors_ = DMatrix<int>::Constant(n_elements_, n_neighbors_per_element, -1);
    // compute facets informations and neighboring structure
    auto facet_pattern = combinations<n_vertices_per_facet, n_vertices>();
    std::unordered_map<std::array<int, n_vertices_per_facet>, int, std_array_hash<int, n_vertices_per_facet>> visited;
    std::array<int, n_vertices_per_facet> facet;

    // cycle over all elements
    for (int i = 0; i < n_elements_; ++i) {
        for (int j = 0; j < facet_pattern.rows(); ++j) {
            // construct facet
            for (int k = 0; k < n_vertices_per_facet; ++k) { facet[k] = elements_(i, facet_pattern(j, k)); }
            std::sort(facet.begin(), facet.end());   // normalize wrt node ordering
            auto it = visited.find(facet);
            if (it != visited.end()) {
                // update face to element bounding
                facet_to_element_[it->second][1] = i;
                // update neighboring informations (each face is shared by two, and only two, adjacent elements)
                for (int h = 0; h < n_elements_per_facet; ++h) {
                    int element_id = facet_to_element_[it->second][h];
                    if (element_id >= 0) {   // not a boundary face
                        // search point opposite to this face (the j-th node which is not a node of this face)
                        int j = 0;
                        for (; j < n_vertices; ++j) {
                            bool found = false;
                            for (int k = 0; k < n_vertices_per_facet; ++k) {
                                if (it->first[k] == elements_(element_id, j)) { found = true; }
                            }
                            if (!found) break;
                        }
                        neighbors_(element_id, j) = facet_to_element_[it->second][(h + 1) % n_elements_per_facet];
                    }
                }
                // free memory
                visited.erase(it);
            } else {
                // store facet and update face to element bounding
                for (int k = 0; k < n_vertices_per_facet; ++k) { facets_.emplace_back(facet[k]); }
                visited.insert({facet, n_facets_});
                facet_to_element_[n_facets_].insert(facet_to_element_[n_facets_].end(), {i, -1});
                n_facets_++;
            }
        }
    }
    // compute edges for 3D domains
    if constexpr (is_3d<M, N>::value) {
        auto edge_pattern = combinations<n_vertices_per_edge, n_vertices>();
        std::unordered_map<std::array<int, n_vertices_per_edge>, int, std_array_hash<int, n_vertices_per_edge>> visited;
        std::array<int, n_vertices_per_edge> edge;
        // cycle over all elements
        for (int i = 0; i < n_elements_; ++i) {
            for (int j = 0; j < edge_pattern.rows(); ++j) {
                // construct facet
                for (int k = 0; k < n_vertices_per_edge; ++k) { edge[k] = elements_(i, edge_pattern(j, k)); }
                std::sort(edge.begin(), edge.end());   // normalize wrt node ordering
                auto it = visited.find(edge);
                if (it != visited.end()) {
                    edge_to_element_[it->second].push_back(i);
                } else {
                    // store facet and update face to element bounding
                    for (int k = 0; k < n_vertices_per_edge; ++k) { edges_.emplace_back(edge[k]); }
                    visited.insert({edge, n_edges_});
                    edge_to_element_[n_edges_].push_back(i);
                    n_edges_++;
                }
            }
        }
    }
    // precompute elements informations for fast access
    elements_cache_.reserve(n_elements_);
    std::array<int, ct_nvertices(M)> node_ids {};
    std::array<SVector<N>, ct_nvertices(M)> coords {};
    std::array<int, ct_nneighbors(M)> neighbors {};
    for (int i = 0; i < n_elements_; ++i) {
        bool boundary = false;   // element on boundary \iff at least one of its nodes is on boundary
        for (int j = 0; j < ct_nvertices(M); ++j) {
            int node_id = elements_(i, j);
            SVector<N> node(nodes_.row(node_id));    // physical coordinate of the node
            coords[j] = node;
            node_ids[j] = node_id;                   // global id of node in the mesh
            boundary |= (boundary_(node_id) == 1);   // boundary status
            neighbors[j] = neighbors_(i, j);         // neighboring element on the facet opposite to node_id
        }
        // cache element
        elements_cache_.emplace_back(i, node_ids, coords, neighbors, boundary);
    }
    return;
}

// constructor specialization for linear networks (1.5D domains)
template <int M, int N>
template <int M_, int N_, typename std::enable_if< is_network<M_, N_>::value, int>::type>
Mesh<M, N>::Mesh(const DMatrix<double>& nodes, const DMatrix<int>& elements, const DMatrix<int>& boundary) :
    nodes_(nodes), elements_(elements), boundary_(boundary) {
    // store number of nodes and number of elements
    n_nodes_ = nodes_.rows();
    n_elements_ = elements_.rows();
    // compute mesh limits
    range_.row(0) = nodes_.colwise().minCoeff();
    range_.row(1) = nodes_.colwise().maxCoeff();
    // compute facets and facet to elements boundings (for linear networks, facets coincide with nodes)
    for (std::size_t i = 0; i < n_elements_; ++i) {
        facet_to_element_[elements_(i, 0)].push_back(i);
        facet_to_element_[elements_(i, 1)].push_back(i);
    }
    for (auto& [key, value] : facet_to_element_) { facets_.emplace_back(key); }
    n_facets_ = facets_.size();

    // recover adjacency matrix
    SpMatrix<int> adjoint_neighbors;
    std::vector<Eigen::Triplet<int>> triplets;
    for (const auto& e : facet_to_element_) {
        for (std::size_t i = 0; i < e.second.size(); ++i) {
            for (std::size_t j = i + 1; j < e.second.size(); ++j) triplets.emplace_back(e.second[j], e.second[i], 1);
        }
    }
    adjoint_neighbors.resize(n_elements_, n_elements_);
    adjoint_neighbors.setFromTriplets(triplets.begin(), triplets.end());
    neighbors_ = adjoint_neighbors.selfadjointView<Eigen::Lower>();   // symmetrize neighboring relation

    // precompute elements informations for fast access
    elements_cache_.reserve(n_elements_);
    std::array<int, ct_nvertices(1)> node_ids {};
    std::array<SVector<2>, ct_nvertices(1)> coords {};
    for (int i = 0; i < n_elements_; ++i) {
        std::vector<int> neighbors {};
        bool boundary = false;   // element on boundary \iff at least one of its nodes is on boundary
        for (int j = 0; j < ct_nvertices(1); ++j) {
            int node_id = elements_(i, j);
            SVector<2> node(nodes_.row(node_id));    // physical coordinate of the node
            coords[j] = node;
            node_ids[j] = node_id;                   // global id of node in the mesh
            boundary |= (boundary_(node_id) == 1);   // boundary status
        }
        for (SpMatrix<int>::InnerIterator sp_mat_it(neighbors_, i); sp_mat_it; ++sp_mat_it) {
            neighbors.push_back(sp_mat_it.row());   // neighbors_ organized in ColumnMajor mode
        }
        // cache element
        elements_cache_.emplace_back(i, node_ids, coords, neighbors, boundary);
    }
};

template <int M, int N> Facet<M, N> Mesh<M, N>::facet(int ID) const {
    // fetch facet nodes informations
    std::array<SVector<N>, n_vertices_per_facet> coords {};
    std::array<int, n_vertices_per_facet> node_ids {};
    bool on_boundary = true;   // facet is a boundary facet \iff all its nodes are ob boundary
    for (int i = 0; i < n_vertices_per_facet; ++i) {
        coords[i] = SVector<N>(nodes_.row(i));
        node_ids[i] = *(facets_.begin() + ID * n_vertices_per_facet + i);
        on_boundary &= is_on_boundary(node_ids[i]);
    }
    return Facet<M, N>(ID, node_ids, coords, facet_to_element_.at(ID), on_boundary);
}

// template specialization for 1D meshes (bounded intervals)
template <> class Mesh<1, 1> {
   protected:
    // pyhisical coordinates of nodes on the line
    DVector<double> nodes_;
    int n_nodes_ = 0;
    // identifiers of nodes (as row indexes in nodes_ matrix) composing each element, in a RowMajor format
    DMatrix<int, Eigen::RowMajor> elements_ {};
    int n_elements_ = 0;
    SVector<2> range_ {};   // mesh bounding box (minimum and maximum coordinates of interval)
    DMatrix<int> neighbors_ {};

    // precomputed set of elements
    std::vector<Element<1, 1>> elements_cache_ {};
   public:
    Mesh() = default;
    Mesh(const DVector<double>& nodes) : nodes_(nodes) {
        // store number of nodes and elements
        n_nodes_ = nodes_.rows();
        n_elements_ = n_nodes_ - 1;
        // compute mesh limits
        range_[0] = nodes_[0];
        range_[1] = nodes_[n_nodes_ - 1];

        // build elements and neighboring structure
        elements_.resize(n_elements_, 2);
        for (std::size_t i = 0; i < n_nodes_ - 1; ++i) {
            elements_(i, 0) = i;
            elements_(i, 1) = i + 1;
        }
        neighbors_ = DMatrix<int>::Constant(n_elements_, n_neighbors_per_element, -1);
        neighbors_(0, 1) = 1;
        for (std::size_t i = 1; i < n_elements_ - 1; ++i) {
            neighbors_(i, 0) = i - 1;
            neighbors_(i, 1) = i + 1;
        }
        neighbors_(n_elements_ - 1, 0) = n_elements_ - 2;

        // precompute elements informations for fast access
        elements_cache_.reserve(n_elements_);
        for (int i = 0; i < n_elements_; ++i) {
            bool boundary = i == 0 || i == (n_elements_ - 1);   // element on boundary \iff its ID is 0 or n_elements_-1
            elements_cache_.emplace_back(
              i, std::array<int, 2> {i, i + 1}, std::array<SVector<1>, 2> {node(i), node(i + 1)},
              std::array<int, 2> {neighbors_(i, 0), neighbors_(i, 1)}, boundary);
        }
        return;
    };
    // construct from interval's bounds [a, b] and the number of subintervals n into which split [a, b]
    Mesh(double a, double b, std::size_t n) : Mesh(DVector<double>::LinSpaced(n + 1, a, b)) { }

    // getters
    const Element<1, 1>& element(int ID) const { return elements_cache_[ID]; }
    Element<1, 1>& element(int ID) { return elements_cache_[ID]; }
    SVector<1> node(int ID) const { return SVector<1>(nodes_[ID]); }
    bool is_on_boundary(int ID) const { return (ID == 0 || ID == (n_nodes_ - 1)); }
    const DVector<double>& nodes() const { return nodes_; }
    const DMatrix<int, Eigen::RowMajor>& elements() const { return elements_; }
    const DMatrix<int>& neighbors() const { return neighbors_; }
    int n_elements() const { return n_elements_; }
    int n_nodes() const { return n_nodes_; }
    SVector<2> range() const { return range_; }

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
        const Element<1, 1>& operator*() { return mesh_->element(index_); }
        friend bool operator!=(const iterator& lhs, const iterator& rhs) { return lhs.index_ != rhs.index_; }
        const Element<1, 1>& operator*() const { return mesh_->element(index_); }
    };
    iterator begin() const { return iterator(this, 0); }
    iterator end() const { return iterator(this, elements_.rows()); }

    // localize element containing point using a O(log(n)) time-complexity binary search strategy
    DVector<int> locate(const DVector<double>& points) const {
        // allocate space
        DVector<int> result;
        result.resize(points.rows());
        // start search
        for (std::size_t i = 0; i < points.rows(); ++i) {
            // check if point is inside
            if (points[i] < range_[0] || points[i] > range_[1]) {
                result[i] = -1;
            } else {
                // search by binary search strategy
                int h_min = 0, h_max = n_nodes_;
                while (true) {
                    int j = h_min + std::floor((h_max - h_min) / 2);
                    if (points[i] >= nodes_[j] && points[i] < nodes_[j + 1]) {
                        result[i] = j;
                        break;
                    } else {
                        if (points[i] < nodes_[j]) {
                            h_max = j;   // search on the left
                        } else {
                            h_min = j;   // search on the right
                        }
                    }
                }
            }
        }
        return result;
    }

    // compile time informations
    static constexpr bool is_manifold = false;
    enum {
        local_dimension = 1,
        embedding_dimension = 1,
        n_vertices = 2,
        n_neighbors_per_element = 2
    };
};

// alias exports
typedef Mesh<1, 1> Mesh1D;
typedef Mesh<2, 2> Mesh2D;
typedef Mesh<3, 3> Mesh3D;
typedef Mesh<2, 3> SurfaceMesh;
typedef Mesh<1, 2> NetworkMesh;
  
}   // namespace core
}   // namespace fdapde

#endif   // __MESH_H__
