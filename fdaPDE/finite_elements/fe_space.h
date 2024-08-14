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

template <typename Triangulation_, typename FeType_, int Size_> class FiniteElementSpace {
   public:
    using Triangulation = std::decay_t<Triangulation_>;
    using FeType = std::decay_t<FeType_>;
    static constexpr int local_dim = Triangulation::local_dim;
    static constexpr int embed_dim = Triangulation::embed_dim;
    using cell_dof_descriptor = FeType::template cell_dof_descriptor<local_dim>;
    using face_dof_descriptor = FeType::template face_dof_descriptor<local_dim>;
    using ReferenceCell = typename cell_dof_descriptor::ReferenceCell;
    using BasisType = typename cell_dof_descriptor::BasisType;
    using ElementType = typename cell_dof_descriptor::ElementType;
    static constexpr int n_components = Size_;
    static constexpr bool is_vector_fe = (n_components != 1);

    // scalar element constructor
    FiniteElementSpace(const Triangulation_& triangulation, FeType_ fe) :
        triangulation_(&triangulation), dof_handler_(triangulation) {
        dof_handler_.enumerate(fe);
        cell_basis_ = BasisType(unit_cell_dofs_.dofs_phys_coords());
        face_basis_ = typename face_dof_descriptor::BasisType(unit_face_dofs_.dofs_phys_coords());
    }
    // vector element constructor (see CWG 1591)
    FiniteElementSpace(const Triangulation_& triangulation, [[maybe_unused]] const FeType_ (&fe)[Size_]) :
      triangulation_(&triangulation), dof_handler_(triangulation) {
        fdapde_static_assert(n_components > 0, DEFINITION_OF_EMPTY_FINITE_ELEMENT_SPACE_IS_ILL_FORMED);
        dof_handler_.enumerate(FeType {});
        cell_basis_ = BasisType(unit_cell_dofs_.dofs_phys_coords());
        face_basis_ = typename face_dof_descriptor::BasisType(unit_face_dofs_.dofs_phys_coords());
    }
    // getters
    const Triangulation& triangulation() const { return *triangulation_; }
    const DofHandler<local_dim, embed_dim>& dof_handler() const { return dof_handler_; }
    DofHandler<local_dim, embed_dim>& dof_handler() { return dof_handler_; }
    constexpr int n_basis() const { return n_components * cell_basis_.size(); }
    constexpr int n_basis_face() const { return n_components * face_basis_.size(); }
    int n_dofs() const { return dof_handler_.n_dofs(); }
    // scalar finite elements API
    template <typename InputType> constexpr auto eval(int i, const InputType& p) const {
        fdapde_static_assert(n_components == 1, THIS_METHOD_IS_ONLY_FOR_SCALAR_FINITE_ELEMENTS);
        return cell_basis_[i](p);
    }
    template <typename InputType> constexpr auto eval_grad(int i, const InputType& p) const {
        fdapde_static_assert(n_components == 1, THIS_METHOD_IS_ONLY_FOR_SCALAR_FINITE_ELEMENTS);
        return cell_basis_[i].gradient()(p);
    }
    template <typename InputType> constexpr auto face_eval(int i, const InputType& p) const {
        fdapde_static_assert(n_components == 1, THIS_METHOD_IS_ONLY_FOR_SCALAR_FINITE_ELEMENTS);
        return face_basis_[i](p);
    }
    template <typename InputType> constexpr auto face_eval_grad(int i, const InputType& p) const {
        fdapde_static_assert(n_components == 1, THIS_METHOD_IS_ONLY_FOR_SCALAR_FINITE_ELEMENTS);
        return face_basis_[i].gradient()(p);
    }
    constexpr const ElementType& basis_function(int i) const { return cell_basis_[i]; }
   private:
    const Triangulation* triangulation_;
    DofHandler<local_dim, embed_dim> dof_handler_;
    cell_dof_descriptor unit_cell_dofs_;
    face_dof_descriptor unit_face_dofs_;
    BasisType cell_basis_;
    typename face_dof_descriptor::BasisType face_basis_;
};

// CTAD for scalar finite elements
template <typename Triangulation, typename FEType_>
FiniteElementSpace(const Triangulation&, FEType_) -> FiniteElementSpace<Triangulation, FEType_, 1>;

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
