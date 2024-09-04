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

template <typename Triangulation_, typename FeType_> class FeSpace {
   public:
    using Triangulation = std::decay_t<Triangulation_>;
    using FeType = std::decay_t<FeType_>;
    static constexpr int local_dim = Triangulation::local_dim;
    static constexpr int embed_dim = Triangulation::embed_dim;
    static constexpr int n_components  = FeType::n_components;
    static constexpr bool is_vector_fe = FeType::is_vector_fe;
    using cell_dof_descriptor = FeType::template cell_dof_descriptor<local_dim>;
    using face_dof_descriptor = FeType::template face_dof_descriptor<local_dim>;
    using ReferenceCell = typename cell_dof_descriptor::ReferenceCell;
    using BasisType = typename cell_dof_descriptor::BasisType;
    using ElementType = typename cell_dof_descriptor::ElementType;

    FeSpace() = default;
    FeSpace(const Triangulation_& triangulation, FeType_ fe) :
        triangulation_(&triangulation), dof_handler_(triangulation) {
        dof_handler_.enumerate(fe);
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
    template <typename InputType>
    constexpr auto eval(int i, const InputType& p) const requires(n_components == 1) { return cell_basis_[i](p); }
    template <typename InputType>
    constexpr auto eval_grad(int i, const InputType& p) const requires(n_components == 1) {
        return cell_basis_[i].gradient()(p);
    }
    template <typename InputType>
    constexpr auto face_eval(int i, const InputType& p) const requires(n_components == 1) { return face_basis_[i](p); }
    template <typename InputType>
    constexpr auto face_eval_grad(int i, const InputType& p) const requires(n_components == 1) {
        return face_basis_[i].gradient()(p);
    }
    // vector finite elements API (\psi_i = [ \psi_{i, 1}, ..., \psi_{i, j}, ..., \psi_{i, n_components} ])
    // here we observe that for a vector valued basis system, \psi_i[j] = 0 \iff i != j, and \psi_j[j] = \psi_j, being
    // \psi_j the j-th scalar basis function defined on the reference cell
    template <typename InputType>
    constexpr auto eval(int i, int j, const InputType& p) const requires(n_components > 1) {
        return (i % n_components == j) ? cell_basis_[j](p) : 0.0;
    }
    template <typename InputType>
    constexpr auto eval_grad(int i, int j, const InputType& p) const
        requires(n_components > 1) {
        using OutputType =
          std::decay_t<decltype(cell_basis_.operator[](std::declval<int>()).operator()(std::declval<InputType>()))>;
        return (i % n_components == j) ? cell_basis_[i].gradient()(p) : OutputType::Zero();
    }
    template <typename InputType>
    constexpr auto face_eval(int i, int j, const InputType& p) const requires(n_components > 1) {
        return (i % n_components == j) ? face_basis_[j](p) : 0.0;
    }
    template <typename InputType>
    constexpr auto face_eval_grad(int i, int j, const InputType& p) const
        requires(n_components > 1) {
        using OutputType =
          std::decay_t<decltype(face_basis_.operator[](std::declval<int>()).operator()(std::declval<InputType>()))>;
        return (i % n_components == j) ? face_basis_[i].gradient()(p) : OutputType::Zero();
    }
   private:
    const Triangulation* triangulation_;
    DofHandler<local_dim, embed_dim> dof_handler_;
    cell_dof_descriptor unit_cell_dofs_;
    face_dof_descriptor unit_face_dofs_;
    BasisType cell_basis_;
    typename face_dof_descriptor::BasisType face_basis_;
};
  
}   // namespace fdapde

#endif   // __FE_SPACE_H__
