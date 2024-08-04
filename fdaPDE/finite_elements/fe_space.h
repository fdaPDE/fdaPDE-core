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

#ifndef __FE_SPACE_H__
#define __FE_SPACE_H__

#include "../utils/symbols.h"
#include "dof_handler.h"

namespace fdapde {

template <typename TriangulationType, typename FeType_, int NComponents> class FiniteElementSpace {
   public:
    static constexpr int local_dim = TriangulationType::local_dim;
    static constexpr int embed_dim = TriangulationType::embed_dim;
    using FeType = std::decay<FeType_>::type;
    using cell_dof_descriptor = FeType::template cell_dof_descriptor<local_dim>;
    using face_dof_descriptor = FeType::template face_dof_descriptor<local_dim>;
    using ReferenceCell = typename cell_dof_descriptor::ReferenceCell;
    using BasisType = typename cell_dof_descriptor::BasisType;
    using BasisElementType = typename cell_dof_descriptor::BasisElementType;
    static constexpr int n_dofs_per_edge = cell_dof_descriptor::n_dofs_per_edge;
    static constexpr int n_dofs_per_face = cell_dof_descriptor::n_dofs_per_face;
    static constexpr int n_dofs_internal = cell_dof_descriptor::n_dofs_internal;
    static constexpr int n_dofs_per_cell = cell_dof_descriptor::n_dofs_per_cell;
    static constexpr int n_components = NComponents;
    static constexpr bool is_vector_fe = n_components != 1;

    // scalar element constructor
    FiniteElementSpace(const TriangulationType& triangulation, FeType_ fe) :
        triangulation_(&triangulation), dof_handler_(triangulation) {
        dof_handler_.enumerate(fe);
        cell_basis_ = BasisType(unit_cell_dofs_.dofs_phys_coords());
        face_basis_ = typename face_dof_descriptor::BasisType(unit_face_dofs_.dofs_phys_coords());
    }
    // vector element constructor (see CWG 1591)
    FiniteElementSpace(const TriangulationType& triangulation, [[maybe_unused]] const FeType_ (&fe)[NComponents]) :
      triangulation_(&triangulation), dof_handler_(triangulation) {
        fdapde_static_assert(n_components > 0, DEFINITION_OF_EMPTY_FINITE_ELEMENT_SPACE_IS_ILL_FORMED);

        dof_handler_.enumerate(FeType {});
        cell_basis_ = BasisType(unit_cell_dofs_.dofs_phys_coords());
        face_basis_ = typename face_dof_descriptor::BasisType(unit_face_dofs_.dofs_phys_coords());
    }

    // getters
    const TriangulationType* triangulation() const { return triangulation_; }
    const DofHandler<local_dim, embed_dim>& dof_handler() const { return dof_handler_; }
    int n_basis() const { return n_components * cell_basis_.size(); }
    int n_basis_face() const { return n_components * face_basis_.size(); }
    // scalar finite elements API
    double eval(int i, const SVector<local_dim>& p) const requires(n_components == 1) { return cell_basis_[i](p); }
    SVector<embed_dim> eval_grad(int i, const SVector<local_dim>& p) const requires(n_components == 1) {
        return cell_basis_[i].gradient()(p);
    }
    double face_eval(int i, const SVector<local_dim>& p) const requires(n_components == 1) { return face_basis_[i](p); }
    SVector<embed_dim> face_eval_grad(int i, const SVector<local_dim>& p) const requires(n_components == 1) {
        return face_basis_[i].gradient()(p);
    }
    const BasisElementType& basis_function(int i) const requires(n_components == 1) { return cell_basis_[i]; }
   private:
    const TriangulationType* triangulation_;
    DofHandler<local_dim, embed_dim> dof_handler_;
    cell_dof_descriptor unit_cell_dofs_;
    face_dof_descriptor unit_face_dofs_;
    BasisType cell_basis_;
    typename face_dof_descriptor::BasisType face_basis_;
};

// CTAD for scalar finite elements
template <typename TriangulationType, typename FEType_>
FiniteElementSpace(const TriangulationType&, FEType_) -> FiniteElementSpace<TriangulationType, FEType_, 1>;

// double eval_component(int i, int j, const SVector<local_dim>& p) const { return basis_[i].gradient()[j](p); }
// must provide access to the basis evaluation (also vectorial) on reference cell
// same also on reference face (what about 1D/1.5D?)

// // vector finite element
// const PolynomialType& operator()(int i, int j) const requires (n_components > 1) {    // (\psi_i)_j
//     return (i % n_components == j) ? basis_[j] : zero_;
// }
// double eval(int i, int j, const SVector<local_dim>& p) const requires(n_components > 1) {   // (\psi_i)_j(p)
//     return (i % n_components == j) ? basis_[j](p) : 0;
// }
// // \nabla[(\psi_i)_j] (gradient of the j-th component of \psi_i)
// SVector<local_dim> eval_grad(int i, int j, const SVector<local_dim>& p) const requires(n_components > 1) {
//     return (i % n_components == j) ? basis_[j].derive()(p) : SVector<local_dim>::Zero();
// }
  
}   // namespace fdapde

#endif   // __FE_SPACE_H__
