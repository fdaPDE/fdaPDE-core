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
#include <unordered_map>
#include <vector>

#include "../linear_algebra/binary_matrix.h"
#include "../utils/combinatorics.h"
#include "../utils/symbols.h"
#include "triangle.h"
#include "tetrahedron.h"
#include "tree_search.h"
#include "utils.h"

namespace fdapde {
namespace core {

template <int M, int N> class Triangulation;
template <int M, int N, typename Derived> class TriangulationBase {
   public:
    // can speek only about cells and vertices here and neighbors
    static constexpr int local_dim = M;
    static constexpr int embed_dim = N;
    static constexpr int n_nodes_per_cell = local_dim + 1;
    static constexpr int n_neighbors_per_cell = local_dim + 1;
    static constexpr bool is_manifold = !(local_dim == embed_dim);
    using CellType = std::conditional_t<M == 2, Triangle<Derived>, Tetrahedron<Derived>>;
    using NodeType = SVector<embed_dim>;

    TriangulationBase() = default;
    TriangulationBase(const DMatrix<double>& nodes, const DMatrix<int>& cells, const DMatrix<int>& boundary) :
        nodes_(nodes), cells_(cells), nodes_markers_(boundary) {
        // store number of nodes and number of cells
        n_nodes_ = nodes_.rows();
        n_cells_ = cells_.rows();
        // compute mesh limits
        range_.row(0) = nodes_.colwise().minCoeff();
        range_.row(1) = nodes_.colwise().maxCoeff();
        // -1 in neighbors_'s column i implies no neighbor adjacent to the edge opposite to vertex i
        neighbors_ = DMatrix<int>::Constant(n_cells_, n_neighbors_per_cell, -1);
    }
    // getters
    CellType cell(int id) const { return CellType(id, static_cast<const Derived*>(this)); }
    NodeType node(int id) const { return nodes_.row(id); }
    bool is_node_on_boundary(int id) const { return nodes_markers_[id]; }
    const DMatrix<double>& nodes() const { return nodes_; }
    const DMatrix<int, Eigen::RowMajor>& cells() const { return cells_; }
    const DMatrix<int, Eigen::RowMajor>& neighbors() const { return neighbors_; }
    const BinaryVector<Dynamic>& boundary_nodes() const { return nodes_markers_; }
    int n_cells() const { return n_cells_; }
    int n_nodes() const { return n_nodes_; }
    int n_boundary_nodes() const { return nodes_markers_.count(); }
    SMatrix<2, N> range() const { return range_; }
  
    // iterators over cells
    class cell_iterator : public index_based_iterator<cell_iterator, CellType> {
        using Base = index_based_iterator<cell_iterator, CellType>;
        using Base::index_;
        friend Base;
        const Derived* mesh_;
        cell_iterator& operator()(int i) {
            Base::val_ = mesh_->cell(i);
            return *this;
        }
       public:
        cell_iterator(int index, const Derived* mesh) : Base(index, 0, mesh->n_cells_), mesh_(mesh) {
            if (index_ < mesh_->n_cells_) operator()(index_);
        }
    };
    cell_iterator cells_begin() const { return cell_iterator(0, static_cast<const Derived*>(this)); }
    cell_iterator cells_end() const { return cell_iterator(n_cells_, static_cast<const Derived*>(this)); }
    // iterator over boundary nodes
    class boundary_node_iterator : public index_based_iterator<boundary_node_iterator, int> {
        using Base = index_based_iterator<boundary_node_iterator, int>;
        using Base::index_;
        const Derived* mesh_;
       public:
        boundary_node_iterator(int index, const Derived* mesh) : Base(index, 0, mesh->n_nodes_), mesh_(mesh) {
            for (; index_ < mesh_->n_nodes_ && !mesh_->nodes_markers_[index_] != true; ++index_);
            this->val_ = index_;
        }
        boundary_node_iterator& operator++() {
            index_++;
            for (; index_ < mesh_->n_nodes_ && !mesh_->nodes_markers_[index_] != true; ++index_);
            this->val_ = index_;
            return *this;
        }
        boundary_node_iterator& operator--() {
            --index_;
            for (; index_ >= 0 && !mesh_->nodes_markers_[index_] != true; --index_);
            this->val_ = index_;
            return *this;
        }
    };
    boundary_node_iterator boundary_nodes_begin() const {
        return boundary_node_iterator(0, static_cast<const Derived*>(this));
    }
    boundary_node_iterator boundary_nodes_end() const {
        return boundary_node_iterator(n_nodes_, static_cast<const Derived*>(this));
    }
   protected:
    DMatrix<double> nodes_ {};                         // physical coordinates of mesh's vertices
    DMatrix<int, Eigen::RowMajor> cells_ {};           // nodes (as row indexes in nodes_ matrix) composing each cell
    DMatrix<int, Eigen::RowMajor> neighbors_ {};       // ids of cells adjacent to a given cell (-1 if no adjacent cell)
    BinaryVector<fdapde::Dynamic> nodes_markers_ {};   // j-th element is 1 \iff node j is on boundary
    SMatrix<2, embed_dim> range_ {};                   // mesh bounding box (column i maps to the i-th dimension)
    int n_nodes_ = 0, n_cells_ = 0;
};

// face-based storage
template <int N> class Triangulation<2, N> : public TriangulationBase<2, N, Triangulation<2, N>> {
    fdapde_static_assert(N == 2 || N == 3, THIS_CLASS_IS_FOR_2D_OR_3D_TRIANGULATIONS_ONLY);
   public:
    using Base = TriangulationBase<2, N, Triangulation<2, N>>;
    static constexpr int n_nodes_per_edge = 2;
    static constexpr int n_edges_per_cell = 3;
    static constexpr int n_faces_per_edge = 2;
    using EdgeType = typename Base::CellType::EdgeType;
    using LocationPolicy = TreeSearch<Triangulation<2, N>>;
    using Base::cells_;      // N \times 3 matrix of node identifiers for each triangle
    using Base::embed_dim;   // dimensionality of the ambient space
    using Base::local_dim;   // dimensionality of the tangent space
    using Base::n_cells_;    // N: number of triangles

    Triangulation() = default;
    Triangulation(const DMatrix<double>& nodes, const DMatrix<int>& faces, const DMatrix<int>& boundary) :
        Base(nodes, faces, boundary) {
        using edge_t = std::array<int, n_nodes_per_edge>;
        using hash_t = fdapde::std_array_hash<int, n_nodes_per_edge>;
        struct edge_info {
            int edge_id, face_id;   // for each edge, its ID and the ID of one of the cells insisting on it
        };
        auto edge_pattern = combinations<n_nodes_per_edge, Base::n_nodes_per_cell>();
        std::unordered_map<edge_t, edge_info, hash_t> edges_map;
	std::vector<bool> edges_markers;
        edge_t edge;
        cell_to_edges_.resize(n_cells_, n_edges_per_cell);
        // search vertex of face f opposite to edge e (the j-th vertex of f which is not a node of e)
        auto node_opposite_to_edge = [this](int e, int f) -> int {
            int j = 0;
            for (; j < Base::n_nodes_per_cell; ++j) {
                bool found = false;
                for (int k = 0; k < n_nodes_per_edge; ++k) {
                    if (edges_[e * n_nodes_per_edge + k] == cells_(f, j)) { found = true; }
                }
                if (!found) break;
            }
            return j;
        };
        int edge_id = 0;
        for (int i = 0; i < n_cells_; ++i) {
            for (int j = 0; j < edge_pattern.rows(); ++j) {
                // construct edge
                for (int k = 0; k < n_nodes_per_edge; ++k) { edge[k] = cells_(i, edge_pattern(j, k)); }
                std::sort(edge.begin(), edge.end());   // normalize wrt node ordering
                auto it = edges_map.find(edge);
                if (it == edges_map.end()) {   // never processed edge
                    edges_.insert(edges_.end(), edge.begin(), edge.end());
		    edges_markers.push_back(true);
                    edges_map.emplace(edge, edge_info {edge_id, i});
                    cell_to_edges_(i, j) = edge_id;
                    edge_id++;
                } else {
                    const auto& [h, k] = it->second;
                    // elements k and i are neighgbors (they share an edfe)
                    this->neighbors_(k, node_opposite_to_edge(h, k)) = i;
                    this->neighbors_(i, node_opposite_to_edge(h, i)) = k;
                    cell_to_edges_(i, j) = h;
                    edges_markers[h] = false;   // edge_id-th edge cannot be on boundary
                    edges_map.erase(it);
                }
            }
        }
        n_edges_ = edges_.size() / n_nodes_per_edge;
        edges_markers_ = BinaryVector<fdapde::Dynamic>(edges_markers.begin(), edges_markers.end(), n_edges_);
        return;
    }
    // getters
    bool is_edge_on_boundary(int id) const { return edges_markers_[id]; }
    Eigen::Map<const DMatrix<int, Eigen::RowMajor>> edges() const {
        return Eigen::Map<const DMatrix<int, Eigen::RowMajor>>(edges_.data(), n_edges_, n_nodes_per_edge);
    }
    const DMatrix<int, Eigen::RowMajor>& cell_to_edges() const { return cell_to_edges_; }
    const BinaryVector<Dynamic>& boundary_edges() const { return edges_markers_; }
    int n_edges() const { return n_edges_; }
    int n_boundary_edges() const { return edges_markers_.count(); }
    // iterators over edges
    class edge_iterator : public index_based_iterator<edge_iterator, EdgeType> {
       protected:
        using Base = index_based_iterator<edge_iterator, EdgeType>;
        using Base::index_;
        const Triangulation* mesh_;
        BinaryVector<fdapde::Dynamic> filter_;
        // fetch next edge
        void next_() {
            for (; index_ < mesh_->n_edges_ && !filter_[index_]; ++index_);
            if (index_ == mesh_->n_edges_) return;
	    Base::val_ = EdgeType(index_, mesh_);
            index_++;
        }
       public:
        edge_iterator(int index, const Triangulation* mesh, const BinaryVector<fdapde::Dynamic>& filter) :
            Base(index, 0, mesh->n_edges_), mesh_(mesh), filter_(filter) {
            next_();
        }
        edge_iterator(int index, const Triangulation* mesh) :
            edge_iterator(index, mesh, BinaryVector<fdapde::Dynamic>(mesh->n_edges_)) { }
        edge_iterator& operator++() {
            next_();
            return *this;
        }
        edge_iterator& operator--() {
            // fetch previous edge
            for (; index_ >= 0 && !filter_[index_]; --index_);
            if (index_ == -1) return *this;
            Base::val_ = EdgeType(index_, mesh_);
            index_--;
            return *this;
        }
    };
    // iterator over boundary edges
    struct boundary_edge_iterator : public edge_iterator {
        boundary_edge_iterator(int index, const Triangulation* mesh) :
            edge_iterator(index, mesh, mesh->edges_markers_) { }
    };
    boundary_edge_iterator boundary_edges_begin() const { return boundary_edge_iterator(0, this); }
    boundary_edge_iterator boundary_edges_end() const { return boundary_edge_iterator(n_edges_, this); }

    // point location
    DVector<int> locate(const DMatrix<double>& points) const {
        if (!location_policy_.has_value()) location_policy_ = LocationPolicy(this);
        return location_policy_->locate(points);
    }
   protected:
    std::vector<int> edges_ {};                        // nodes (as row indexes in nodes_ matrix) composing each edge
    DMatrix<int, Eigen::RowMajor> cell_to_edges_ {};   // ids of edges composing each face
    BinaryVector<fdapde::Dynamic> edges_markers_ {};   // j-th element is 1 \iff edge j is on boundary
    int n_edges_ = 0;
    mutable std::optional<LocationPolicy> location_policy_ {};
};

// face-based storage
template <> class Triangulation<3, 3> : public TriangulationBase<3, 3, Triangulation<3, 3>> {
   private:
    // basic iterator type
    template <typename Iterator, typename ValueType> class iterator : public index_based_iterator<Iterator, ValueType> {
       protected:
        using Base = index_based_iterator<Iterator, ValueType>;
        using Base::index_;
        const Triangulation* mesh_;
        BinaryVector<fdapde::Dynamic> filter_;
        void next_() {
            for (; index_ < Base::end_ && !filter_[index_]; ++index_);
            if (index_ == Base::end_) return;
            Base::val_ = ValueType(index_, mesh_);
            index_++;
        }
       public:
        iterator(
          int index, int begin, int end, const Triangulation* mesh, const BinaryVector<fdapde::Dynamic>& filter) :
            Base(index, begin, end), mesh_(mesh), filter_(filter) {
            next_();
        }
        iterator(int index, int begin, int end, const Triangulation* mesh) :
            iterator(index, begin, end, mesh, BinaryVector<fdapde::Dynamic>(mesh->n_edges_)) { }
        Iterator& operator++() {
            next_();
            return static_cast<Iterator&>(*this);
        }
        Iterator& operator--() {
            for (; index_ >= Base::begin_ && !filter_[index_]; --index_);
            if (index_ == -1) return static_cast<Iterator&>(*this);
            Base::val_ = ValueType(index_, mesh_);
            index_--;
            return static_cast<Iterator&>(*this);
        }
    };
   public:
    using Base = TriangulationBase<3, 3, Triangulation<3, 3>>;
    static constexpr int n_nodes_per_face = 3;
    static constexpr int n_nodes_per_edge = 2;
    static constexpr int n_edges_per_face = 3;
    static constexpr int n_faces_per_cell = 4;
    using FaceType = typename Base::CellType::FaceType;
    using EdgeType = typename Base::CellType::EdgeType;
    using LocationPolicy = TreeSearch<Triangulation<3, 3>>;
    using Base::embed_dim;
    using Base::local_dim;

    Triangulation() = default;
    Triangulation(const DMatrix<double>& nodes, const DMatrix<int>& cells, const DMatrix<int>& boundary) :
        Base(nodes, cells, boundary) {
        using face_t = std::array<int, n_nodes_per_face>;
        using edge_t = std::array<int, n_nodes_per_edge>;
        struct face_info {
            int face_id, cell_id;   // for each face, its ID and the ID of one of the faces insisting on it
        };
	typedef int edge_info;
        auto face_pattern = combinations<n_nodes_per_face, n_nodes_per_cell>();
        auto edge_pattern = combinations<n_nodes_per_edge, n_nodes_per_face>();
        std::unordered_map<edge_t, edge_info, fdapde::std_array_hash<int, n_nodes_per_edge>> edges_map;
        std::unordered_map<face_t, face_info, fdapde::std_array_hash<int, n_nodes_per_face>> faces_map;
	std::vector<bool> faces_markers, edges_markers;
        face_t face;
        edge_t edge;
        cell_to_faces_.resize(n_cells_, n_faces_per_cell);
        // search vertex of face f opposite to edge e (the j-th vertex of f which is not a node of e)
        auto node_opposite_to_face = [this](int e, int f) -> int {
            int j = 0;
            for (; j < Base::n_nodes_per_cell; ++j) {
                bool found = false;
                for (int k = 0; k < n_nodes_per_face; ++k) {
                    if (faces_[e * n_nodes_per_face + k] == cells_(f, j)) { found = true; }
                }
                if (!found) break;
            }
            return j;
        };
        int face_id = 0, edge_id = 0;
        for (int i = 0; i < n_cells_; ++i) {
            for (int j = 0; j < face_pattern.rows(); ++j) {
                // construct face
                for (int k = 0; k < n_nodes_per_face; ++k) { face[k] = cells_(i, face_pattern(j, k)); }
                std::sort(face.begin(), face.end());   // normalize wrt node ordering
                auto it = faces_map.find(face);
                if (it == faces_map.end()) {   // never processed face
                    faces_.insert(faces_.end(), face.begin(), face.end());
		    faces_markers.push_back(true);
                    faces_map.emplace(face, face_info {face_id, i});
                    cell_to_faces_(i, j) = face_id;
                    face_id++;
                    // compute for each face the ids of its edges
                    for (int k = 0; k < n_edges_per_face; ++k) {
                        for (int h = 0; h < n_nodes_per_edge; ++h) { edge[h] = face[edge_pattern(k, h)]; }
                        std::sort(edge.begin(), edge.end());
                        auto it = edges_map.find(edge);
                        if (it == edges_map.end()) {
                            edges_.insert(edges_.end(), edge.begin(), edge.end());
                            face_to_edges_.push_back(edge_id);
                            edges_map.emplace(edge, edge_id);
			    edges_markers.push_back(nodes_markers_[edge[0]] && nodes_markers_[edge[1]]);
                            edge_id++;
                        } else {
                            face_to_edges_.push_back(edges_map.at(edge));
                        }
                    }
                } else {
                    const auto& [h, k] = it->second;
                    // elements k and i are neighgbors (they share a face)
                    neighbors_(k, node_opposite_to_face(h, k)) = i;
                    neighbors_(i, node_opposite_to_face(h, i)) = k;
                    cell_to_faces_(i, j) = h;
		    faces_markers[h] = false;
                    faces_map.erase(it);
                }
            }
        }
        n_faces_ = faces_.size() / n_nodes_per_face;
        n_edges_ = edges_.size() / n_nodes_per_edge;
        faces_markers_ = BinaryVector<fdapde::Dynamic>(faces_markers.begin(), faces_markers.end(), n_faces_);
	edges_markers_ = BinaryVector<fdapde::Dynamic>(edges_markers.begin(), edges_markers.end(), n_edges_);
        return;
    }
    // getters
    bool is_face_on_boundary(int id) const { return faces_markers_[id]; }
    const DMatrix<int, Eigen::RowMajor>& neighbors() const { return neighbors_; }
    Eigen::Map<const DMatrix<int, Eigen::RowMajor>> faces() const {
        return Eigen::Map<const DMatrix<int, Eigen::RowMajor>>(faces_.data(), n_faces_, n_nodes_per_face);
    }
    Eigen::Map<const DMatrix<int, Eigen::RowMajor>> edges() const {
        return Eigen::Map<const DMatrix<int, Eigen::RowMajor>>(edges_.data(), n_edges_, n_nodes_per_edge);
    }
    const DMatrix<int, Eigen::RowMajor>& cell_to_faces() const { return cell_to_faces_; }
    Eigen::Map<const DMatrix<int, Eigen::RowMajor>> face_to_edges() const {
        return Eigen::Map<const DMatrix<int, Eigen::RowMajor>>(face_to_edges_.data(), n_faces_, n_edges_per_face);
    }
    const BinaryVector<Dynamic>& boundary_faces() const { return faces_markers_; }
    int n_faces() const { return n_faces_; }
    int n_edges() const { return n_edges_; }
    int n_boundary_faces() const { return faces_markers_.count(); }
    int n_boundary_edges() const { return edges_markers_.count(); }
    // iterators over edges
    struct edge_iterator : public iterator<edge_iterator, EdgeType> {
        edge_iterator(int index, const Triangulation* mesh, const BinaryVector<fdapde::Dynamic>& filter) :
            iterator<edge_iterator, EdgeType>(index, 0, mesh->n_edges_, mesh, filter) { }
        edge_iterator(int index, const Triangulation* mesh) :
            iterator<edge_iterator, EdgeType>(index, 0, mesh->n_edges_, mesh) { }
    };
    edge_iterator edges_begin() const { return edge_iterator(0, this); }
    edge_iterator edges_end() const { return edge_iterator(n_edges_, this); }
    // iterators over faces
    struct face_iterator : public iterator<face_iterator, FaceType> {
        face_iterator(int index, const Triangulation* mesh, const BinaryVector<fdapde::Dynamic>& filter) :
            iterator<face_iterator, FaceType>(index, 0, mesh->n_faces_, mesh, filter) { }
        face_iterator(int index, const Triangulation* mesh) :
            iterator<face_iterator, FaceType>(index, 0, mesh->n_faces_, mesh) { }
    };
    // iterator over boundary faces
    struct boundary_face_iterator : public face_iterator {
        boundary_face_iterator(int index, const Triangulation* mesh) :
            face_iterator(index, mesh, mesh->faces_markers_) { }
    };
    boundary_face_iterator boundary_faces_begin() const { return boundary_face_iterator(0, this); }
    boundary_face_iterator boundary_faces_end() const { return boundary_face_iterator(n_faces_, this); }

    // point location
    DVector<int> locate(const DMatrix<double>& points) const {
        if (!location_policy_.has_value()) location_policy_ = LocationPolicy(this);
        return location_policy_->locate(points);
    }
    int locate(const SVector<embed_dim>& p) const {
        if (!location_policy_.has_value()) location_policy_ = LocationPolicy(this);
        return location_policy_->locate(p);
    }
   protected:
    std::vector<int> faces_, edges_;   // nodes (as row indexes in nodes_ matrix) composing each face and edge
    DMatrix<int, Eigen::RowMajor> cell_to_faces_ {};   // ids of faces composing each cell
    std::vector<int> face_to_edges_;                   // ids of edges composing each face
    BinaryVector<fdapde::Dynamic> faces_markers_ {};   // j-th element is 1 \iff face j is on boundary
    BinaryVector<fdapde::Dynamic> edges_markers_ {};   // j-th element is 1 \iff edge j is on boundary
    int n_faces_ = 0, n_edges_ = 0;
    mutable std::optional<LocationPolicy> location_policy_ {};
};

}   // namespace core
}   // namespace fdapde

#endif   // __TRIANGULATION_H__
