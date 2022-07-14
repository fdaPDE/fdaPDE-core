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

#ifndef __FDAPDE_DOF_SEGMENT_H__
#define __FDAPDE_DOF_SEGMENT_H__

#include "header_check.h"

namespace fdapde {

// definition of dof-informed triangle, i.e. a triangle with attached dofs
template <typename DofHandler>
class DofSegment : public Segment<typename DofHandler::TriangulationType> {
    fdapde_static_assert(
      DofHandler::TriangulationType::local_dim == 1, THIS_CLASS_IS_FOR_INTERVAL_AND_LINEAR_NETWORK_MESHES_ONLY);
    using Base = Segment<typename DofHandler::TriangulationType>;
    const DofHandler* dof_handler_;
   public:
    using TriangulationType = typename DofHandler::TriangulationType;
    static constexpr int local_dim = TriangulationType::local_dim;
    static constexpr int embed_dim = TriangulationType::embed_dim;
    // constructor
    DofSegment() = default;
    DofSegment(int cell_id, const DofHandler* dof_handler) :
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
};

}   // namespace fdapde

#endif // __FDAPDE_DOF_SEGMENT_H__
