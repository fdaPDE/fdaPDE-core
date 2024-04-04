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
#include "../linear_algebra/binary_matrix.h"
#include "element.h"
#include "utils.h"

namespace fdapde {
namespace core {

// face-based storage
template <int M, int N> class Triangulation {
   public:
    static constexpr int local_dim = M;
    static constexpr int embed_dim = N;
    static constexpr int n_vertices_per_face = ct_nvertices(local_dim);
    static constexpr int n_vertices_per_edge = local_dim;
    static constexpr int n_edges_per_face = ct_nedges(local_dim);
    static constexpr int n_neighbors_per_face = ct_nneighbors(local_dim);
    static constexpr int n_faces_per_edge = 2;
    static constexpr bool is_manifold = !(local_dim == embed_dim);
    using MeshType = Triangulation<M, N>;
    using FaceType = Element<MeshType>;
    using NodeType = SVector<N>;
    // type-erasure wrapper for point location strategy
    struct PointLocation__ {
        template <typename T> using fn_ptrs = mem_fn_ptrs<&T::locate>;
        // solves the point location problem for a set of points, outputs -1 if element not found
        DVector<int> locate(const DMatrix<double>& points) const {
            fdapde_assert(points.cols() == N);
            DVector<int> result(points.rows());
            for (int i = 0; i < points.rows(); ++i) {
                auto e = invoke<const FaceType*, 0>(*this, NodeType(points.row(i)));
                result[i] = e ? e->ID() : -1;
            }
            return result;
        }
    };

    Triangulation() = default;
    Triangulation(const DMatrix<double>& nodes, const DMatrix<int>& faces, const DMatrix<int>& boundary)
        requires(!is_network<M, N>::value)
        : nodes_(nodes), faces_(faces), nodes_markers_(boundary) {
        // store number of nodes and number of faces
        n_nodes_ = nodes_.rows();
        n_faces_ = faces_.rows();
        // compute mesh limits
        range_.row(0) = nodes_.colwise().minCoeff();
        range_.row(1) = nodes_.colwise().maxCoeff();
	// compute number of edges
        auto edge_pattern = combinations<n_vertices_per_edge, n_vertices_per_face>();
	using edge_t = std::array<int, n_vertices_per_edge>;
	using hash_t = fdapde::std_array_hash<int, n_vertices_per_edge>;
	std::unordered_set<edge_t, hash_t> edges_set;
	edge_t edge;
        for (int i = 0; i < n_faces_; ++i) {
            for (int j = 0; j < edge_pattern.rows(); ++j) {
                // construct edge
                for (int k = 0; k < n_vertices_per_edge; ++k) { edge[k] = faces_(i, edge_pattern(j, k)); }
                std::sort(edge.begin(), edge.end());   // normalize wrt node ordering
                if (edges_set.find(edge) == edges_set.end()) edges_set.insert(edge);
            }
        }
	n_edges_ = edges_set.size();
	edges_set.clear();
	// allocate storage
	edges_markers_ = BinaryVector<Dynamic>::Ones(n_edges_);
	edges_.resize(n_edges_, n_vertices_per_edge);
	// -1 in neighbors_'s column i implies no neighbor adjacent to the edge opposite to vertex i
        neighbors_ = DMatrix<int>::Constant(n_faces_, n_neighbors_per_face, -1);
	struct edge_info {
            int edge_id, face_id;   // for each face, its ID and the ID of one of the faces insisting on it
        };
        std::unordered_map<edge_t, edge_info, hash_t> edges_map;
        // search vertex of face f opposite to edge e (the j-th vertex of f which is not a node of e)
        auto vertex_opposite_to_edge = [this](int e, int f) -> int {
            int j = 0;
            for (; j < n_vertices_per_face; ++j) {
                bool found = false;
                for (int k = 0; k < n_vertices_per_edge; ++k) {
                    if (edges_(e, k) == faces_(f, j)) { found = true; }
                }
                if (!found) break;
            }
            return j;
        };
	int edge_id = 0;
        for (int i = 0; i < n_faces_; ++i) {
            for (int j = 0; j < edge_pattern.rows(); ++j) {
                // construct edge
                for (int k = 0; k < n_vertices_per_edge; ++k) { edge[k] = faces_(i, edge_pattern(j, k)); }
                std::sort(edge.begin(), edge.end());   // normalize wrt node ordering
                auto it = edges_map.find(edge);
                if (it == edges_map.end()) {   // never processed edge
                    for (int k = 0; k < n_vertices_per_edge; ++k) { edges_(edge_id, k) = edge[k]; }
                    edges_map.emplace(edge, {edge_id, i});
                    edge_id++;
                } else {
                    int k = it->second.face_id;
		    int h = it->second.edge_id;
		    // elements k and i are neighgbors (they share a face)
                    neighbors_(k, vertex_opposite_to_edge(h, k)) = i;
                    neighbors_(i, vertex_opposite_to_edge(h, i)) = k;
                    edges_markers_.clear(h);   // edge_id-th edge cannot be on boundary
                    edges_map.erase(it);
                }
            }
        }
	return;
    }
    // getters
    FaceType face(int id) const { return FaceType(id, this); }
    NodeType node(int id) const { return nodes_.row(id); }
    bool is_node_on_boundary(int id) const { return nodes_markers_[id]; }
    bool is_edge_on_boundary(int id) const { return edges_markers_[id]; }
    const DMatrix<double>& nodes() const { return nodes_; }
    const DMatrix<int, Eigen::RowMajor>& faces() const { return faces_; }
    const DMatrix<int, Eigen::RowMajor>& neighbors() const { return neighbors_; }
    const DMatrix<int, Eigen::RowMajor>& edges() const { return edges_; }
    const BinaryVector<Dynamic>& boundary_nodes() const { return nodes_markers_; }
    const BinaryVector<Dynamic>& boundary_edges() const { return edges_markers_; }
    int n_faces() const { return n_faces_; }
    int n_nodes() const { return n_nodes_; }
    int n_edges() const { return n_edges_; }
    SMatrix<2, N> range() const { return range_; }
    DVector<int> locate(const DMatrix<double>& points) const {
        if (!point_locator_) point_locator_ = TreeSearch(*this);   // fallback to tree-based search strategy
        return point_locator_.locate(points);
    }
  
    // iterators
    struct face_iterator {
       private:
        using This = face_iterator;
        const Triangulation* mesh_;
        int index_;   // current element
       public:
        face_iterator(const Triangulation* mesh, int index) : mesh_(mesh), index_(index) { }
        This& operator++() {
            ++index_;
            return *this;
        }
        FaceType operator*() const { return mesh_->face(index_); }
        friend bool operator!=(const This& lhs, const This& rhs) { return lhs.index_ != rhs.index_; }
        friend bool operator==(const This& lhs, const This& rhs) { return lhs.index_ == rhs.index_; }
    };
    face_iterator faces_begin() const { return face_iterator(this, 0); }
    face_iterator faces_end() const { return face_iterator(this, n_faces_); }

    struct boundary_node_iterator {
       private:
        using This = boundary_node_iterator;
        const Triangulation* mesh_;
        int index_;   // current boundary node
       public:
        boundary_node_iterator(const Triangulation* mesh, int index) : mesh_(mesh), index_(index) { }
        boundary_node_iterator& operator++() {
            index_++;
            // scan until all nodes have been visited or a boundary node is not found
            for (; index_ < mesh_->n_nodes() && mesh_->is_on_boundary(index_) != true; ++index_);
            return *this;
        }
        int operator*() const { return index_; }   // ID of the boundary node
        friend bool operator!=(const This& lhs, const This& rhs) { return lhs.index_ != rhs.index_; }
        friend bool operator==(const This& lhs, const This& rhs) { return lhs.index_ == rhs.index_; }
    };
    boundary_node_iterator boundary_nodes_begin() const { return boundary_node_iterator(this, 0); }
    boundary_node_iterator boundary_nodes_end() const { return boundary_node_iterator(this, n_nodes_); }

    // geometrical view (e.g., as Simplex instances) of the boundary edges
    struct boundary_edge_iterator {
       private:
        using This = boundary_edge_iterator;
        const Triangulation* mesh_;
        int index_;   // current boundary face
       public:
        boundary_edge_iterator(const Triangulation* mesh, int index) : mesh_(mesh), index_(index) { }
        boundary_edge_iterator& operator++() {
            index_++;
            for (; index_ < mesh_->n_edges() && mesh_->edges_markers_[index_]; ++index_);
            return *this;
        }
        Simplex<local_dim - 1, embed_dim> operator*() const {
            SMatrix<embed_dim, local_dim> coords;
            for (int i = 0; i < local_dim; ++i) { coords.col(i) = mesh_->nodes_.row(mesh_->edges_(index_, i)); }
            return Simplex<local_dim - 1, embed_dim>(coords);
        };
        friend bool operator!=(const This& lhs, const This& rhs) { return lhs.index_ != rhs.index_; }
        friend bool operator==(const This& lhs, const This& rhs) { return lhs.index_ == rhs.index_; }
    };
    boundary_edge_iterator boundary_edges_begin() const { return boundary_edge_iterator(this, 0); }
    boundary_edge_iterator boundary_edges_end() const { return boundary_edge_iterator(this, n_edges_); }
  
    // setters
    template <typename PointLocation_> void set_point_locator(PointLocation_&& point_locator) {
        point_locator_ = point_locator;
    }
   protected:
    DMatrix<double> nodes_ {};                         // physical coordinates of mesh's vertices
    DMatrix<int, Eigen::RowMajor> faces_ {};           // nodes (as row indexes in nodes_ matrix) composing each face
    DMatrix<int, Eigen::RowMajor> edges_ {};           // nodes composing each edge
    DMatrix<int, Eigen::RowMajor> neighbors_ {};       // ids of faces adjacent to a given face (-1 if no adjacent face)
    BinaryVector<fdapde::Dynamic> nodes_markers_ {};   // j-th element is 1 \iff node j is on boundary
    BinaryVector<fdapde::Dynamic> edges_markers_ {};   // j-th element is 1 \iff edge j is on boundary
    SMatrix<2, N> range_ {};                           // mesh bounding box (column i maps to the i-th dimension)
    int n_nodes_ = 0, n_faces_ = 0, n_edges_ = 0;
    mutable erase<heap_storage, PointLocation__> point_locator_ {};
};

// template specialization for 1D meshes (bounded intervals)
template <> class Triangulation<1, 1> {
   public:
    using NeighborsContainerType = DMatrix<int>;
    using MeshType = Triangulation<1, 1>;
    using FaceType = Element<MeshType>;
    // compile time informations
    static constexpr int local_dim = 1;
    static constexpr int embed_dim = 1;
    static constexpr bool is_manifold = false;
    static constexpr int n_vertices_per_element = 2;
    static constexpr int n_vertices_per_face = local_dim;
    static constexpr int n_faces_per_element = 2;
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
    const FaceType& element(int ID) const {
        if (!cache_[ID]) cache_[ID] = FaceType(ID, this);
        return cache_[ID];
    }
    FaceType& element(int ID) {
        if (!cache_[ID]) cache_[ID] = FaceType(ID, this);
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
        const FaceType& operator*() { return mesh_->element(index_); }
        friend bool operator!=(const element_iterator& lhs, const element_iterator& rhs) {
            return lhs.index_ != rhs.index_;
        }
        const FaceType& operator*() const { return mesh_->element(index_); }
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
    mutable std::vector<FaceType> cache_ {};
};

    // linear network (1.5D) specialized constructor
    // Triangulation(const DMatrix<double>& nodes, const DMatrix<int>& faces, const DMatrix<int>& boundary)
    //     requires(is_network<M, N>::value)
    //     : nodes_(nodes), faces_(faces), boundary_(boundary) {
    //     // store number of nodes and number of elements
    //     // n_nodes_ = nodes_.rows();
    //     // n_faces_ = faces_.rows();
    //     // // compute mesh limits
    //     // range_.row(0) = nodes_.colwise().minCoeff();
    //     // range_.row(1) = nodes_.colwise().maxCoeff();
    //     // // compute faces and neighboring structure
    // 	// std::unordered_map<int, std::vector<int>> node_connection; // for each node, the elements insisting on it
    //     // for (int i = 0; i < n_elements_; ++i) {
    //     //     node_connection[faces_(i, 0)].push_back(i);
    //     //     node_connection[faces_(i, 1)].push_back(i);
    //     // }
    //     // for (auto& [key, value] : node_connection) { edges_.emplace_back(key); }
    //     // n_faces_ = faces_.size();
    //     // // recover adjacency matrix
    //     // SpMatrix<int> adjoint_neighbors;
    //     // std::vector<Eigen::Triplet<int>> adj;
    //     // for (const auto& e : node_connection) {
    //     //     for (std::size_t i = 0; i < e.second.size(); ++i) {
    //     //         for (std::size_t j = i + 1; j < e.second.size(); ++j) adj.emplace_back(e.second[j], e.second[i], 1);
    //     //     }
    //     // }
    //     // adjoint_neighbors.resize(n_elements_, n_elements_);
    //     // adjoint_neighbors.setFromTriplets(adj.begin(), adj.end());
    //     // neighbors_ = adjoint_neighbors.selfadjointView<Eigen::Lower>();   // symmetrize neighboring relation
    //     // cache_.resize(n_elements_);
    // }


  
// alias exports
using Triangulation1D = Triangulation<1, 1>;
using Triangulation2D = Triangulation<2, 2>;
using Triangulation3D = Triangulation<3, 3>;
using SurfaceMeshTriangulation = Triangulation<2, 3>;
using NetworkMesh = Triangulation<1, 2>;

}   // namespace core
}   // namespace fdapde

#endif   // __TRIANGULATION_H__
