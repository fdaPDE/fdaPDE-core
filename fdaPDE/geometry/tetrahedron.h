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

#ifndef __TETRAHEDRON_H__
#define __TETRAHEDRON_H__

#include "../utils/symbols.h"
#include "simplex.h"

namespace fdapde {
namespace core {

template <typename Triangulation>
class Tetrahedron : public Simplex<Triangulation::local_dim, Triangulation::embed_dim> {
    fdapde_static_assert(
      Triangulation::local_dim == 3 && Triangulation::embed_dim == 3, THIS_CLASS_IS_ONLY_FOR_TETRAHEDRAL_MESHES_ONLY);
   public:
    // constructor
    Tetrahedron() = default;
    Tetrahedron(int id, const Triangulation* mesh) : id_(id), mesh_(mesh), boundary_(false) {
        int b_matches_ = 0;   // element is on boundary if has at least (n_nodes - 1) nodes on boundary
        for (int j = 0; j < this->n_nodes; ++j) {
            this->coords_.col(j) = mesh_->node(mesh_->cells()(id_, j));
	    if (mesh_->is_node_on_boundary(mesh_->cells()(id_, j))) b_matches_++;
        }
	if (b_matches_ >= this->n_nodes - 1) boundary_ = true;
	// compute edges identifiers
	std::unordered_set<int> edge_ids;
        for (int i = 0; i < this->n_faces; ++i) {
            int face_id = mesh_->cell_to_faces()(id_, i);
            for (int j = 0; j < 3; ++j) { edge_ids.insert(mesh_->face_to_edges()(face_id, j)); }
        }
	int i = 0;
	for(const int& id : edge_ids) edge_ids_[i++] = id;
        this->initialize();
    }
    // a triangulation-aware view of a tetrahedron edge
    class EdgeType : public Simplex<1, Triangulation::embed_dim> {
      using Base = Simplex<1, Triangulation::embed_dim>;
      int edge_id_;
      const Triangulation* mesh_;
    public:
      using CoordsType = SMatrix<Triangulation::embed_dim, 2>;
      EdgeType() = default;
      EdgeType(int edge_id, const Triangulation* mesh) : edge_id_(edge_id), mesh_(mesh) {
        for (int i = 0; i < Base::n_nodes; ++i) { Base::coords_.col(i) = mesh_->node(mesh_->edges()(edge_id_, i)); }
        this->initialize();
      }
      bool on_boundary() const { return mesh_->is_edge_on_boundary(edge_id_); }
      DVector<int> node_ids() const { return mesh_->edges().row(edge_id_); }
      int id() const { return edge_id_; }
      const std::unordered_set<int>& adjacent_cells() const { return mesh_->edge_to_cells().at(edge_id_); }
    };
    // a triangulation-aware view of a tetrahedron face
    class FaceType : public Simplex<2, Triangulation::embed_dim> {
      using Base = Simplex<2, Triangulation::embed_dim>;
      int face_id_;
      const Triangulation* mesh_;
     public:
      using CoordsType = SMatrix<Triangulation::embed_dim, 3>;
      FaceType() = default;
      FaceType(int face_id, const Triangulation* mesh) : face_id_(face_id), mesh_(mesh) {
        for (int i = 0; i < Base::n_nodes; ++i) { Base::coords_.col(i) = mesh_->node(mesh_->faces()(face_id_, i)); }
        this->initialize();
      }
      bool on_boundary() const { return mesh_->is_face_on_boundary(face_id_); }
      DVector<int> node_ids() const { return mesh_->faces().row(face_id_); }
      DVector<int> edge_ids() const { return mesh_->face_to_edges().row(face_id_); }
      int id() const { return face_id_; }
      EdgeType edge(int n) const { return EdgeType(mesh_->face_to_edges()(face_id_, n), mesh_); }
      DVector<int> adjacent_cells() const { return mesh_->face_to_cells().row(face_id_); }
    };

    // getters
    int id() const { return id_; }
    DVector<int> neighbors() const { return mesh_->neighbors().row(id_); }
    DVector<int> node_ids() const { return mesh_->cells().row(id_); }
    bool on_boundary() const { return boundary_; }
    operator bool() const { return mesh_ != nullptr; }
    EdgeType edge(int n) const { return EdgeType(edge_ids_[n], mesh_); }
    FaceType face(int n) const { return FaceType(mesh_->cell_to_faces()(id_, n), mesh_); }

    // iterator over tetrahedron edges
    class edge_iterator : public index_based_iterator<edge_iterator, EdgeType> {
        using Base = index_based_iterator<edge_iterator, EdgeType>;
        using Base::index_;
        friend Base;
        const Tetrahedron* t_;
        // access to i-th edge
        edge_iterator& operator()(int i) {
            Base::val_ = t_->edge(i);
            return *this;
        }
       public:
        edge_iterator(int index, const Tetrahedron* t) : Base(index, 0, t_->n_edges), t_(t) {
            if (index_ < t_->n_edges) operator()(index_);
        }
    };
    edge_iterator edges_begin() const { return edge_iterator(0, this); }
    edge_iterator edges_end() const { return edge_iterator(this->n_edges, this); }
    // iterator over tetrahedron faces
    class face_iterator : public index_based_iterator<face_iterator, FaceType> {
        using Base = index_based_iterator<face_iterator, FaceType>;
        using Base::index_;
        friend Base;
        const Tetrahedron* t_;
        // access to i-th face
        face_iterator& operator()(int i) {
            Base::val_ = t_->face(i);
            return *this;
        }
       public:
        face_iterator(int index, const Tetrahedron* t) : Base(index, 0, t_->n_faces), t_(t) {
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

}   // namespace core
}   // namespace fdapde

#endif   // __TETRAHEDRON_H__
