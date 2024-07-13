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

#ifndef __DOF_TRIANGLE_H__
#define __DOF_TRIANGLE_H__

namespace fdapde {
namespace core {

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
        DVector<int> dofs_;
        const DofHandler* dof_handler_;
       public:
        EdgeType() = default;
        EdgeType(int edge_id, const DofHandler* dof_handler) :
            Base::EdgeType(edge_id, dof_handler->triangulation()), dof_handler_(dof_handler) {
            // if you query a DofTriangle for its edge, most likely you want to access its dofs. compute and cache
            dofs_ = DVector<int>(TriangulationType::n_nodes_per_edge + dof_handler_->n_dofs_per_edge());
            int j = 0;
            for (int d : this->node_ids()) dofs_[j++] = d;   // at nodes, dof numbering == mesh numbering
            for (int k = 0; k < dof_handler_->n_dofs_per_edge(); ++k) {
                dofs_[j++] = dof_handler_->edge_to_dofs().at(this->id())[k];
            }
        }
        const DVector<int>& dofs() const { return dofs_; }
        DVector<short> dofs_markers() const { return dof_handler_->dof_markers()(dofs()); }
        BinaryVector<fdapde::Dynamic> boundary_dofs() const {
            BinaryVector<fdapde::Dynamic> boundary(dofs_.size());
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
    DVector<int> dofs() const { return dof_handler_->active_dofs(Base::id()); }
    DVector<short> dofs_markers() const { return dof_handler_->dof_markers()(dofs()); }
    BinaryVector<fdapde::Dynamic> boundary_dofs() const {
        DVector<int> tmp = dofs();
        BinaryVector<fdapde::Dynamic> boundary(tmp.size());
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
        return EdgeType(dof_handler_->triangulation()->cell_to_edges()(Base::id(), n), dof_handler_);
    }
    class edge_iterator : public index_based_iterator<edge_iterator, EdgeType> {
        using Base = index_based_iterator<edge_iterator, EdgeType>;
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

}   // namespace core
}   // namespace fdapde

#endif // __DOF_TRIANGLE_H__
