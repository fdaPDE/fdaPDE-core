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

#ifndef __ELEMENT_H__
#define __ELEMENT_H__

#include "../utils/symbols.h"
#include "simplex.h"

namespace fdapde {
namespace core {
    
  // an element is a geometric object bounded to a mesh. It carries both geometrical and connectivity informations
  template <typename MeshType> class Element : public Simplex<MeshType::local_dim, MeshType::embed_dim> {
   public:
    // constructor
    Element() = default;
    Element(int id, const MeshType* mesh) : id_(id), mesh_(mesh), boundary_(false) {
        int b_matches_ = 0;   // element is on boundary if has at least one face (n_vertices - 1 vertices) on boundary
        for (int j = 0; j < this->n_vertices; ++j) {
            this->coords_.col(j) = mesh_->node(mesh_->faces()(id_, j));
	    if (mesh_->is_node_on_boundary(mesh_->faces()(id_, j))) b_matches_++;
        }
	if (b_matches_ >= this->n_vertices - 1) boundary_ = true;
	this->initialize();
    }
    // getters
    int id() const { return id_; }
    DVector<int> neighbors() const { return mesh_->neighbors().row(id_); }
    DVector<int> node_ids() const { return mesh_->faces().row(id_); } // change in vertices_id
    bool on_boundary() const { return boundary_; }
    operator bool() const { return mesh_ != nullptr; }

    // a triangulation-aware view of an element face
    class Face : public Simplex<MeshType::local_dim, MeshType::embed_dim>::FaceType {
    private:
      int face_id_;
      const MeshType* mesh_;
    public:
      using CoordsType = SMatrix<MeshType::embed_dim, MeshType::local_dim + 1>;
      Face() = default;
      Face(int face_id, const MeshType* mesh) : face_id_(face_id), mesh_(mesh) {
	for (int i = 0; i < this->n_vertices; ++i) { this->coords_.col(i) = mesh_->node(mesh_->edges()(face_id_, i)); }
	this->initialize();
      }
      bool on_boundary() const { return mesh_->is_edge_on_boundary(face_id_); }
      DVector<int> node_ids() const { return mesh_->edges().row(face_id_); }
      int id() const { return face_id_; }
    };
    Face face(int n) const {
      fdapde_assert(n < this->n_faces);
      return Face(mesh_->face_to_edges()(id_, n), mesh_);
    }

    struct face_iterator {
     private:
      int index_;
      const Element* element_;
      Face f_;
     public:
      using value_type        = Face;
      using pointer           = const Face*;
      using reference         = const Face&;
      using size_type         = std::size_t;
      using difference_type   = std::ptrdiff_t;
      using iterator_category = std::forward_iterator_tag;

      face_iterator(int index, const Element* element) : index_(index), element_(element) {
          if (index_ < element_->n_faces) f_ = element_->face(index_);
      }
      reference operator*() const { return f_; }
      pointer operator->() const { return &f_; }
      face_iterator& operator++() {
	  index_++;
	  if (index_ < element_->n_faces) f_ = element_->face(index_);
	  return *this;
      }
      face_iterator operator++(int) {
	  face_iterator tmp(index_, this);
	  ++(*this);
	  return tmp;
      }
      friend bool operator!=(const face_iterator& lhs, const face_iterator& rhs) { return lhs.index_ != rhs.index_; }
      friend bool operator==(const face_iterator& lhs, const face_iterator& rhs) { return lhs.index_ == rhs.index_; }
    };
    face_iterator faces_begin() const { return face_iterator(0, this); }
    face_iterator faces_end() const { return face_iterator(this->n_faces, this); }
    
   private:
    int id_ = 0;   // ID of this element in the physical mesh
    const MeshType* mesh_ = nullptr;
    bool boundary_ = false;   // true if the element has at least one vertex on the boundary
};

}   // namespace core
}   // namespace fdapde

#endif   // __ELEMENT_H__
