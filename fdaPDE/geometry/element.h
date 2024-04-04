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
            this->coords_.col(j) = mesh_->node(mesh_->elements()(id_, j));
	    if (mesh_->is_on_boundary(mesh_->elements()(id_, j))) b_matches_++;
        }
	if (b_matches_ >= this->n_vertices - 1) boundary_ = true;
    }
    // getters
    int ID() const { return id_; }
    auto neighbors() const { return mesh_->neighbors().row(id_); }
    auto node_ids() const { return mesh_->elements().row(id_); }
    bool is_on_boundary() const { return boundary_; }
    operator bool() const { return mesh_ != nullptr; }
   private:
    int id_ = 0;   // ID of this element in the physical mesh
    const MeshType* mesh_ = nullptr;
    bool boundary_ = false;   // true if the element has at least one vertex on the boundary
};

}   // namespace core
}   // namespace fdapde

#endif   // __ELEMENT_H__
