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

#ifndef __FDAPDE_TETRAHEDRON_H__
#define __FDAPDE_TETRAHEDRON_H__

#include "header_check.h"

namespace fdapde {

template <typename Triangulation>
class Tetrahedron : public Simplex<Triangulation::local_dim, Triangulation::embed_dim> {
    fdapde_static_assert(
      Triangulation::local_dim == 3 && Triangulation::embed_dim == 3, THIS_CLASS_IS_FOR_TETRAHEDRAL_MESHES_ONLY);
   public:
    static constexpr int n_nodes_per_edge = 2;
    static constexpr int n_nodes_per_face = 3;
    // constructor
    Tetrahedron() = default;
    Tetrahedron(int id, const Triangulation* mesh) : id_(id), mesh_(mesh), boundary_(false) {
        int b_matches_ = 0;   // element is on boundary if has at least (n_nodes - 1) nodes on boundary
        for (int j = 0; j < this->n_nodes; ++j) {
            this->coords_.col(j) = mesh_->node(mesh_->cells()(id_, j));
            if (mesh_->is_node_on_boundary(mesh_->cells()(id_, j))) b_matches_++;
        }
        if (b_matches_ >= this->n_nodes - 1) boundary_ = true;
        // compute edges identifiers (guarantees edge_iterator order: (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
        int f1 = mesh_->cell_to_faces()(id_, 0);   // (0 1 2)
        int f2 = mesh_->cell_to_faces()(id_, 1);   // (0 1 3)
        int f3 = mesh_->cell_to_faces()(id_, 2);   // (0 2 3)
        const auto& edge_of = mesh_->face_to_edges();
        edge_ids_ = {edge_of(f1, 0), edge_of(f1, 1), edge_of(f2, 1), edge_of(f1, 2), edge_of(f2, 2), edge_of(f3, 2)};
        this->initialize();
    }
    // a triangulation-aware view of a tetrahedron edge
    class EdgeType : public Simplex<1, Triangulation::embed_dim> {
        using Base = Simplex<1, Triangulation::embed_dim>;
        int edge_id_;
        const Triangulation* mesh_;
       public:
        using CoordsType = Eigen::Matrix<double, Triangulation::embed_dim, 2>;
        EdgeType() = default;
        EdgeType(int edge_id, const Triangulation* mesh) : edge_id_(edge_id), mesh_(mesh) {
            for (int i = 0; i < Base::n_nodes; ++i) { Base::coords_.col(i) = mesh_->node(mesh_->edges()(edge_id_, i)); }
            this->initialize();
        }
        bool on_boundary() const { return mesh_->is_edge_on_boundary(edge_id_); }
        Eigen::Matrix<int, Dynamic, 1> node_ids() const { return mesh_->edges().row(edge_id_); }
        int id() const { return edge_id_; }
        const std::unordered_set<int>& adjacent_cells() const { return mesh_->edge_to_cells().at(edge_id_); }
        int marker() const {   // mesh edge's marker
            return mesh_->edges_markers().size() > edge_id_ ? mesh_->edges_markers()[edge_id_] : Unmarked;
        }
    };
    // a triangulation-aware view of a tetrahedron face
    class FaceType : public Simplex<2, Triangulation::embed_dim> {
        using Base = Simplex<2, Triangulation::embed_dim>;
        int face_id_;
        const Triangulation* mesh_;
       public:
        using CoordsType = Eigen::Matrix<double, Triangulation::embed_dim, 3>;
        FaceType() = default;
        FaceType(int face_id, const Triangulation* mesh) : face_id_(face_id), mesh_(mesh) {
            for (int i = 0; i < Base::n_nodes; ++i) { Base::coords_.col(i) = mesh_->node(mesh_->faces()(face_id_, i)); }
            this->initialize();
        }
        bool on_boundary() const { return mesh_->is_face_on_boundary(face_id_); }
        Eigen::Matrix<int, Dynamic, 1> node_ids() const { return mesh_->faces().row(face_id_); }
        Eigen::Matrix<int, Dynamic, 1> edge_ids() const { return mesh_->face_to_edges().row(face_id_); }
        int id() const { return face_id_; }
        EdgeType edge(int n) const { return EdgeType(mesh_->face_to_edges()(face_id_, n), mesh_); }
        Eigen::Matrix<int, Dynamic, 1> adjacent_cells() const { return mesh_->face_to_cells().row(face_id_); }
        int marker() const {   // mesh face's marker
            return mesh_->faces_markers().size() > face_id_ ? mesh_->faces_markers()[face_id_] : Unmarked;
        }
    };

    // getters
    int id() const { return id_; }
    Eigen::Matrix<int, Dynamic, 1> neighbors() const { return mesh_->neighbors().row(id_); }
    Eigen::Matrix<int, Dynamic, 1> node_ids() const { return mesh_->cells().row(id_); }
    bool on_boundary() const { return boundary_; }
    operator bool() const { return mesh_ != nullptr; }
    EdgeType edge(int n) const { return EdgeType(edge_ids_[n], mesh_); }
    FaceType face(int n) const { return FaceType(mesh_->cell_to_faces()(id_, n), mesh_); }
    // cell marker
    int marker() const { return mesh_->cells_markers().size() > id_ ? mesh_->cells_markers()[id_] : Unmarked; }

    // iterator over tetrahedron edges
    class edge_iterator : public internals::index_iterator<edge_iterator, EdgeType> {
        using Base = internals::index_iterator<edge_iterator, EdgeType>;
        using Base::index_;
        friend Base;
        const Tetrahedron* t_;
        // access to i-th edge
        edge_iterator& operator()(int i) {
            Base::val_ = t_->edge(i);
            return *this;
        }
       public:
        edge_iterator(int index, const Tetrahedron* t) : Base(index, 0, t->n_edges), t_(t) {
            if (index_ < t_->n_edges) operator()(index_);
        }
    };
    edge_iterator edges_begin() const { return edge_iterator(0, this); }
    edge_iterator edges_end() const { return edge_iterator(this->n_edges, this); }
    // iterator over tetrahedron faces
    class face_iterator : public internals::index_iterator<face_iterator, FaceType> {
        using Base = internals::index_iterator<face_iterator, FaceType>;
        using Base::index_;
        friend Base;
        const Tetrahedron* t_;
        // access to i-th face
        face_iterator& operator()(int i) {
            Base::val_ = t_->face(i);
            return *this;
        }
       public:
        face_iterator(int index, const Tetrahedron* t) : Base(index, 0, t->n_faces), t_(t) {
            if (index_ < t_->n_faces) operator()(index_);
        }
    };
    face_iterator faces_begin() const { return face_iterator(0, this); }
    face_iterator faces_end() const { return face_iterator(this->n_faces, this); }
   private:
    int id_ = 0;                    // tetrahedron identifier in the physical mesh
    std::array<int, 6> edge_ids_;   // edges identifiers int the physical mesh
    const Triangulation* mesh_ = nullptr;
    bool boundary_ = false;   // true if the element has at least one vertex on the boundary
};

}   // namespace fdapde

#endif   // __FDAPDE_TETRAHEDRON_H__
