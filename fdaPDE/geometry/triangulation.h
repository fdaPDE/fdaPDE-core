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

#ifndef __TRIANGULATION_H__
#define __TRIANGULATION_H__

#include <array>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "../utils/combinatorics.h"
#include "../utils/symbols.h"
#include "element.h"
#include "utils.h"

namespace fdapde {
namespace core {

template <int M, int N> class Triangulation {
   public:
    using NeighborsContainerType = std::conditional_t<is_network<M, N>::value, SpMatrix<int>, DMatrix<int>>;
    using MeshType = Triangulation<M, N>;
    using ElementType = Element<MeshType>;
    // type-erasure wrapper for point location strategy
    struct PointLocation__ {
        template <typename T> using fn_ptrs = mem_fn_ptrs<&T::locate>;
        // solves the point location problem for a set of points, outputs -1 if element not found
        DVector<int> locate(const DMatrix<double>& points) const {
            fdapde_assert(points.cols() == N);
            DVector<int> result(points.rows());
            for (int i = 0; i < points.rows(); ++i) {
                auto e = invoke<const ElementType*, 0>(*this, SVector<N>(points.row(i)));
                result[i] = e ? e->ID() : -1;
            }
            return result;
        }
    };
    // compile time informations
    static constexpr int local_dim = M;
    static constexpr int embed_dim = N;
    static constexpr bool is_manifold = (local_dim != embed_dim);
    static constexpr int n_vertices = ct_nvertices(local_dim);
    static constexpr int n_vertices_per_facet = local_dim;
    static constexpr int n_facets_per_element = ct_nfacets(local_dim);
    static constexpr int n_neighbors_per_element = ct_nneighbors(local_dim);
    static constexpr int n_elements_per_facet = 2;

    Triangulation() = default;
    // 2D, 2.5D, 3D constructor
    Triangulation(const DMatrix<double>& nodes, const DMatrix<int>& elements, const DMatrix<int>& boundary)
        requires(!is_network<M, N>::value)
        : nodes_(nodes), elements_(elements), boundary_(boundary) {
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
        std::unordered_map<   // for each facet, one of the elements insisting on it
          std::array<int, n_vertices_per_facet>, int, std_array_hash<int, n_vertices_per_facet>>
          visited;
        std::array<int, n_vertices_per_facet> facet;
        // search vertex of e opposite to facet f (the j-th vertex of e which is not a node of f)
        auto vertex_opposite_to_facet = [this](const std::array<int, n_vertices_per_facet>& f, int e) -> int {
            int j = 0;
            for (; j < n_vertices; ++j) {
                bool found = false;
                for (int k = 0; k < n_vertices_per_facet; ++k) {
                    if (f[k] == elements_(e, j)) { found = true; }
                }
                if (!found) break;
            }
            return j;
        };
        // cycle over all elements
        for (int i = 0; i < n_elements_; ++i) {
            for (int j = 0; j < facet_pattern.rows(); ++j) {
                // construct facet
                for (int k = 0; k < n_vertices_per_facet; ++k) { facet[k] = elements_(i, facet_pattern(j, k)); }
                std::sort(facet.begin(), facet.end());   // normalize wrt node ordering
                auto it = visited.find(facet);
                if (it == visited.end()) {
                    facets_.insert(facets_.end(), &facet[0], &facet[n_vertices_per_facet]);
                    visited.insert({facet, i});   // store facet and ID of element insisting on it
                    n_facets_++;
                } else {
                    // update neighboring informations (each facet is shared by two, and only two, adjacent elements)
                    neighbors_(it->second, vertex_opposite_to_facet(facet, it->second)) = i;
                    neighbors_(i, vertex_opposite_to_facet(facet, i)) = it->second;   // exploit symmetry of relation
                    visited.erase(it);
                }
            }
        }
        cache_.resize(n_elements_);
        return;
    }
    // linear network (1.5D) specialized constructor
    Triangulation(const DMatrix<double>& nodes, const DMatrix<int>& elements, const DMatrix<int>& boundary)
        requires(is_network<M, N>::value)
        : nodes_(nodes), elements_(elements), boundary_(boundary) {
        // store number of nodes and number of elements
        n_nodes_ = nodes_.rows();
        n_elements_ = elements_.rows();
        // compute mesh limits
        range_.row(0) = nodes_.colwise().minCoeff();
        range_.row(1) = nodes_.colwise().maxCoeff();
        // compute facets and neighboring structure
	std::unordered_map<int, std::vector<int>> node_connection; // for each node, the elements insisting on it
        for (int i = 0; i < n_elements_; ++i) {
            node_connection[elements_(i, 0)].push_back(i);
            node_connection[elements_(i, 1)].push_back(i);
        }
        for (auto& [key, value] : node_connection) { facets_.emplace_back(key); }
        n_facets_ = facets_.size();
        // recover adjacency matrix
        SpMatrix<int> adjoint_neighbors;
        std::vector<Eigen::Triplet<int>> adj;
        for (const auto& e : node_connection) {
            for (std::size_t i = 0; i < e.second.size(); ++i) {
                for (std::size_t j = i + 1; j < e.second.size(); ++j) adj.emplace_back(e.second[j], e.second[i], 1);
            }
        }
        adjoint_neighbors.resize(n_elements_, n_elements_);
        adjoint_neighbors.setFromTriplets(adj.begin(), adj.end());
        neighbors_ = adjoint_neighbors.selfadjointView<Eigen::Lower>();   // symmetrize neighboring relation
        cache_.resize(n_elements_);
    }

    // getters
    const ElementType& element(int id) const {
        if (!cache_[id]) cache_[id] = ElementType(id, this);
        return cache_[id];
    }
    ElementType& element(int id) {
        if (!cache_[id]) cache_[id] = ElementType(id, this);
        return cache_[id];
    }
    SVector<N> node(int id) const { return nodes_.row(id); }
    bool is_on_boundary(int id) const { return boundary_(id) == 1; }
    const DMatrix<double>& nodes() const { return nodes_; }
    const DMatrix<int, Eigen::RowMajor>& elements() const { return elements_; }
    const NeighborsContainerType& neighbors() const { return neighbors_; }
    const DMatrix<int>& boundary() const { return boundary_; }
    int n_elements() const { return n_elements_; }
    int n_nodes() const { return n_nodes_; }
    SMatrix<2, N> range() const { return range_; }
    Eigen::Map<const DMatrix<int, Eigen::RowMajor>> facets() const {
        return Eigen::Map<const DMatrix<int, Eigen::RowMajor>>(facets_.data(), n_facets_, n_vertices_per_facet);
    }
    int n_facets() const { return n_facets_; }
    DVector<int> locate(const DMatrix<double>& points) const {
        if (!point_location_) point_location_ = TreeSearch(*this);   // fallback to tree-based search strategy
        return point_location_.locate(points);
    }

    // iterators support
    struct element_iterator {   // range-for loop over mesh elements
       private:
        friend Triangulation;
        const Triangulation* mesh_;
        int index_;   // current element
       public:
        element_iterator(const Triangulation* mesh, int index) : mesh_(mesh), index_(index) {};
        element_iterator& operator++() {
            ++index_;
            return *this;
        }
        const ElementType& operator*() { return mesh_->element(index_); }
        const ElementType& operator*() const { return mesh_->element(index_); }
        friend bool operator!=(const element_iterator& lhs, const element_iterator& rhs) {
            return lhs.index_ != rhs.index_;
        }
    };
    element_iterator begin() const { return element_iterator(this, 0); }
    element_iterator end() const { return element_iterator(this, elements_.rows()); }

    struct boundary_iterator {   // range-for loop over boundary nodes
       private:
        friend Triangulation;
        const Triangulation* mesh_;
        int index_;   // current boundary node
       public:
        boundary_iterator(const Triangulation* mesh, int index) : mesh_(mesh), index_(index) {};
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
    boundary_iterator boundary_begin() const { return boundary_iterator(this, 0); }
    boundary_iterator boundary_end() const { return boundary_iterator(this, n_nodes_); }

    // setters
    template <typename PointLocation> void set_point_location(PointLocation&& point_location) {
        point_location_ = point_location;
    }
   protected:
    // physical coordinates of mesh's vertices
    DMatrix<double> nodes_ {};
    int n_nodes_ = 0;
    // identifiers of nodes (as row indexes in nodes_ matrix) composing each element, in a RowMajor format
    DMatrix<int, Eigen::RowMajor> elements_ {};
    int n_elements_ = 0;
    DMatrix<int> boundary_ {};   // vector of binary coefficients such that, boundary_[j] = 1 \iff node j is on boundary
    SMatrix<2, N> range_ {};     // mesh bounding box (column i maps to the i-th dimension)
    NeighborsContainerType neighbors_ {};
    std::vector<int> facets_ {};   // nodes composing each facet of the mesh (linearly stored in a row-major format)
    int n_facets_ = 0;
    mutable std::vector<ElementType> cache_ {};
    mutable erase<heap_storage, PointLocation__> point_location_ {};
};

// template specialization for 1D meshes (bounded intervals)
template <> class Triangulation<1, 1> {
   public:
    using NeighborsContainerType = DMatrix<int>;
    using MeshType = Triangulation<1, 1>;
    using ElementType = Element<MeshType>;
    // compile time informations
    static constexpr int local_dim = 1;
    static constexpr int embed_dim = 1;
    static constexpr bool is_manifold = false;
    static constexpr int n_vertices = 2;
    static constexpr int n_vertices_per_facet = local_dim;
    static constexpr int n_facets_per_element = 2;
    static constexpr int n_neighbors_per_element = 2;

    Triangulation() = default;
    Triangulation(const DVector<double>& nodes) : nodes_(nodes) {
        // store number of nodes and elements
        n_nodes_ = nodes_.rows();
        n_elements_ = n_nodes_ - 1;
        // compute mesh limits
        range_[0] = nodes_[0];
        range_[1] = nodes_[n_nodes_ - 1];
        // build elements and neighboring structure
        elements_.resize(n_elements_, 2);
        for (int i = 0; i < n_nodes_ - 1; ++i) {
            elements_(i, 0) = i;
            elements_(i, 1) = i + 1;
        }
        neighbors_ = DMatrix<int>::Constant(n_elements_, n_neighbors_per_element, -1);
        neighbors_(0, 1) = 1;
        for (int i = 1; i < n_elements_ - 1; ++i) {
            neighbors_(i, 0) = i - 1;
            neighbors_(i, 1) = i + 1;
        }
        neighbors_(n_elements_ - 1, 0) = n_elements_ - 2;
	// set first and last nodes as boundary nodes
	boundary_ = DMatrix<int>::Zero(n_nodes_, 1);
	boundary_(0, 0) = 1;
	boundary_(n_nodes_ - 1, 0) = 1;
        cache_.resize(n_elements_);
    };
    // construct from interval's bounds [a, b] and the number of subintervals n into which split [a, b]
    Triangulation(double a, double b, int n) : Triangulation(DVector<double>::LinSpaced(n + 1, a, b)) { }

    // getters
    const ElementType& element(int ID) const {
        if (!cache_[ID]) cache_[ID] = ElementType(ID, this);
        return cache_[ID];
    }
    ElementType& element(int ID) {
        if (!cache_[ID]) cache_[ID] = ElementType(ID, this);
        return cache_[ID];
    }
    SVector<1> node(int ID) const { return SVector<1>(nodes_[ID]); }
    bool is_on_boundary(int ID) const { return (ID == 0 || ID == (n_nodes_ - 1)); }
    const DVector<double>& nodes() const { return nodes_; }
    const DMatrix<int, Eigen::RowMajor>& elements() const { return elements_; }
    const DMatrix<int>& neighbors() const { return neighbors_; }
    const DMatrix<int>& boundary() const { return boundary_; }
    int n_elements() const { return n_elements_; }
    int n_nodes() const { return n_nodes_; }
    SVector<2> range() const { return range_; }

    // iterators support
    struct element_iterator {   // range-for loop over mesh elements
       private:
        friend Triangulation;
        const Triangulation* mesh_;
        int index_;   // current element
        element_iterator(const Triangulation* mesh, int index) : mesh_(mesh), index_(index) {};
       public:
        // increment current iteration index and return this element_iterator
        element_iterator& operator++() {
            ++index_;
            return *this;
        }
        const ElementType& operator*() { return mesh_->element(index_); }
        friend bool operator!=(const element_iterator& lhs, const element_iterator& rhs) {
            return lhs.index_ != rhs.index_;
        }
        const ElementType& operator*() const { return mesh_->element(index_); }
    };
    element_iterator begin() const { return element_iterator(this, 0); }
    element_iterator end() const { return element_iterator(this, elements_.rows()); }

    // localize element containing point using a O(log(n)) time-complexity binary search strategy
    DVector<int> locate(const DVector<double>& points) const {
        // allocate space
        DVector<int> result;
        result.resize(points.rows());
        // start search
        for (int i = 0; i < points.rows(); ++i) {
            // check if point is inside
            if (points[i] < range_[0] || points[i] > range_[1]) {
                result[i] = -1;
            } else {
                // binary search strategy
                int h_min = 0, h_max = n_nodes_;
                while (true) {
                    int j = h_min + std::floor((h_max - h_min) / 2);
                    if (points[i] >= nodes_[j] && points[i] < nodes_[j + 1]) {
                        result[i] = j;
                        break;
                    } else {
                        if (points[i] < nodes_[j]) {   // search on the left
                            h_max = j;
                        } else {   // search on the right
                            h_min = j;
                        }
                    }
                }
            }
        }
        return result;
    }
   protected:
    // pyhisical coordinates of nodes on the line
    DVector<double> nodes_;
    int n_nodes_ = 0;
    // identifiers of nodes (as row indexes in nodes_ matrix) composing each element, in a RowMajor format
    DMatrix<int, Eigen::RowMajor> elements_ {};
    int n_elements_ = 0;
    DMatrix<int> boundary_ {};   // vector of binary coefficients such that, boundary_[j] = 1 \iff node j is on boundary
    SVector<2> range_ {};        // mesh bounding box (minimum and maximum coordinates of interval)
    DMatrix<int> neighbors_ {};
    mutable std::vector<ElementType> cache_ {};
};

// alias exports
using Triangulation1D = Triangulation<1, 1>;
using Triangulation2D = Triangulation<2, 2>;
using Triangulation3D = Triangulation<3, 3>;
using SurfaceMeshTriangulation = Triangulation<2, 3>;
using NetworkMesh = Triangulation<1, 2>;

}   // namespace core
}   // namespace fdapde

#endif   // __TRIANGULATION_H__
