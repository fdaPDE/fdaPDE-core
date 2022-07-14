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

#ifndef __FDAPDE_SEGMENT_H__
#define __FDAPDE_SEGMENT_H__

#include "header_check.h"

namespace fdapde {

// a mesh aware view of a segment
template <typename Triangulation> class Segment : public Simplex<Triangulation::local_dim, Triangulation::embed_dim> {
    fdapde_static_assert(Triangulation::local_dim == 1, THIS_CLASS_IS_FOR_INTERVAL_AND_LINEAR_NETWORK_MESHES_ONLY);
   public:
    // constructor
    Segment() = default;
    Segment(int id, const Triangulation* mesh) : id_(id), mesh_(mesh), boundary_(false) {
        int b_matches_ = 0;   // element is on boundary if has at least (n_nodes - 1) nodes on boundary
        for (int j = 0; j < this->n_nodes; ++j) {
            this->coords_.col(j) = mesh_->node(mesh_->cells()(id_, j));
	    if (mesh_->is_node_on_boundary(mesh_->cells()(id_, j))) b_matches_++;
        }
	if (b_matches_ >= this->n_nodes - 1) boundary_ = true;
	this->initialize();
    }
    // getters
    int id() const { return id_; }
    Eigen::Matrix<int, Dynamic, 1> neighbors() const { return mesh_->neighbors().row(id_); }
    Eigen::Matrix<int, Dynamic, 1> node_ids() const { return mesh_->cells().row(id_); }
    bool on_boundary() const { return boundary_; }
    operator bool() const { return mesh_ != nullptr; }
   protected:
    int id_ = 0;   // segment ID in the physical mesh
    const Triangulation* mesh_ = nullptr;
    bool boundary_ = false;   // true if the element has at least one vertex on the boundary
};

}   // namespace fdapde

#endif   // __FDAPDE_SEGMENT_H__
