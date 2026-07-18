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

#ifndef __FDAPDE_DOF_TRIANGLE_H__
#define __FDAPDE_DOF_TRIANGLE_H__

#include "header_check.h"

namespace fdapde {

// definition of dof-informed triangle, i.e. a triangle with attached dofs
template <typename DofHandler>
class DofTriangle : public Triangle<typename DofHandler::TriangulationType> {
    fdapde_static_assert(DofHandler::TriangulationType::local_dim == 2, THIS_CLASS_IS_FOR_TRIANGULAR_MESHES_ONLY);
    using Base = Triangle<typename DofHandler::TriangulationType>;
    const DofHandler* dof_handler_;
   public:
    using TriangulationType = typename DofHandler::TriangulationType;
    static constexpr int local_dim = TriangulationType::local_dim;
    static constexpr int embed_dim = TriangulationType::embed_dim;
  
    class EdgeType : public Base::EdgeType {
        Eigen::Matrix<int, Dynamic, 1> dofs_;
        const DofHandler* dof_handler_;

        void initialize_dofs(int cell_id, int local_edge_id) {
            const int n_node_dofs =
              dof_handler_->n_dofs_per_node() * TriangulationType::n_nodes_per_edge;
            const int n_edge_dofs = dof_handler_->n_dofs_per_edge();
            const int n_scalar_dofs = n_node_dofs + n_edge_dofs;
            dofs_.resize(n_scalar_dofs * dof_handler_->dof_multiplicity());

            int out = 0;
            const int n_unique_dofs = dof_handler_->n_unique_dofs();
            if (dof_handler_->dof_sharing()) {
                for (int component = 0; component < dof_handler_->dof_multiplicity(); ++component) {
                    if (dof_handler_->n_dofs_per_node() > 0) {
                        for (int node : this->node_ids()) dofs_[out++] = node + component * n_unique_dofs;
                    }
                    for (int k = 0; k < n_edge_dofs; ++k) {
                        dofs_[out++] = dof_handler_->edge_to_dofs().at(this->id())[k] + component * n_unique_dofs;
                    }
                }
                return;
            }

            const auto cell_dofs = dof_handler_->active_dofs(cell_id);
            const int n_dofs_per_component = dof_handler_->n_dofs_per_cell();
            const int edge_offset =
              dof_handler_->n_dofs_per_node() * TriangulationType::n_nodes_per_cell + local_edge_id * n_edge_dofs;
            for (int component = 0; component < dof_handler_->dof_multiplicity(); ++component) {
                const int component_offset = component * n_dofs_per_component;
                for (int endpoint = 0; endpoint < TriangulationType::n_nodes_per_edge; ++endpoint) {
                    for (int k = 0; k < dof_handler_->n_dofs_per_node(); ++k) {
                        const int local_node = TriangulationType::edge_pattern(local_edge_id, endpoint);
                        dofs_[out++] = cell_dofs[
                          component_offset + local_node * dof_handler_->n_dofs_per_node() + k];
                    }
                }
                for (int k = 0; k < n_edge_dofs; ++k) {
                    dofs_[out++] = cell_dofs[component_offset + edge_offset + k];
                }
            }
        }
       public:
        EdgeType() = default;
        EdgeType(int edge_id, const DofHandler* dof_handler) :
            Base::EdgeType(edge_id, dof_handler->triangulation()), dof_handler_(dof_handler) {
            int cell_id = 0;
            int local_edge_id = 0;
            if (!dof_handler_->dof_sharing()) {
                auto adjacent_cells = this->adjacent_cells();
                cell_id = adjacent_cells[0] >= 0 ? adjacent_cells[0] : adjacent_cells[1];
                local_edge_id = dof_handler_->triangulation()->cell(cell_id).local_edge_id(edge_id);
            }
            initialize_dofs(cell_id, local_edge_id);
        }
        EdgeType(int edge_id, int cell_id, int local_edge_id, const DofHandler* dof_handler) :
            Base::EdgeType(edge_id, dof_handler->triangulation()), dof_handler_(dof_handler) {
            initialize_dofs(cell_id, local_edge_id);
        }
        const Eigen::Matrix<int, Dynamic, 1>& dofs() const { return dofs_; }
        Eigen::Matrix<int, Dynamic, 1> dofs_markers() const { return dof_handler_->dof_markers()(dofs()); }
        BinaryVector<Dynamic> boundary_dofs() const {
            BinaryVector<Dynamic> boundary(dofs_.size());
            int i = 0;
            for (int dof : dofs_) {
                if (dof_handler_->is_dof_on_boundary(dof)) boundary.set(i);
                ++i;
            }
            return boundary;
        }
    };
    // constructor
    DofTriangle() = default;
    DofTriangle(int cell_id, const DofHandler* dof_handler) :
        Base(cell_id, dof_handler->triangulation()), dof_handler_(dof_handler) { }
    Eigen::Matrix<int, Dynamic, 1> dofs() const { return dof_handler_->active_dofs(Base::id()); }
    Eigen::Matrix<int, Dynamic, 1> dofs_markers() const { return dof_handler_->dof_markers()(dofs()); }
    BinaryVector<Dynamic> boundary_dofs() const {
        Eigen::Matrix<int, Dynamic, 1> tmp = dofs();
        BinaryVector<Dynamic> boundary(tmp.size());
        int i = 0;
        for (int dof : tmp) {
            if (dof_handler_->is_dof_on_boundary(dof)) boundary.set(i);
            ++i;
        }
        return boundary;
    }
    // overload geometric edge getter to return dof-informed edge structure
    EdgeType edge(int n) const {
        fdapde_assert(n < Base::n_edges);
        return EdgeType(dof_handler_->triangulation()->cell_to_edges()(Base::id(), n), Base::id(), n, dof_handler_);
    }
    class edge_iterator : public internals::index_iterator<edge_iterator, EdgeType> {
        using Base = internals::index_iterator<edge_iterator, EdgeType>;
        using Base::index_;
        friend Base;
        const DofTriangle* t_;
        // access to i-th triangle edge
        edge_iterator& operator()(int i) {
            Base::val_ = t_->edge(i);
            return *this;
        }
       public:
        edge_iterator(int index, const DofTriangle* t) : Base(index, 0, t_->n_edges), t_(t) {
            if (index_ < t_->n_edges) operator()(index_);
        }
    };
    edge_iterator edges_begin() const { return edge_iterator(0, this); }
    edge_iterator edges_end() const { return edge_iterator(this->n_edges, this); }
};

}   // namespace fdapde

#endif // __FDAPDE_DOF_TRIANGLE_H__
