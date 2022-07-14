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

#ifndef __FDAPDE_FE_SPACE_H__
#define __FDAPDE_FE_SPACE_H__

#include "header_check.h"

namespace fdapde {

// forward declarations
template <typename FeSpace_> class FeFunction;
namespace internals {

template <typename Triangulation_, typename Form_, int Options_, typename... Quadrature_>
class fe_bilinear_form_assembly_loop;
template <typename Triangulation_, typename Form_, int Options_, typename... Quadrature_>
class fe_linear_form_assembly_loop;

}   // namespace internals

// definition of finite element space
template <typename Triangulation_, typename FeType_> class FeSpace {
    template <typename T> struct subscript_t_impl {
        using type = std::decay_t<decltype(std::declval<T>().operator[](std::declval<int>()))>;
    };
    template <typename T> using subscript_t = typename subscript_t_impl<T>::type;
   public:
    using Triangulation = std::decay_t<Triangulation_>;
    using FeType = std::decay_t<FeType_>;
    static constexpr int local_dim = Triangulation::local_dim;
    static constexpr int embed_dim = Triangulation::embed_dim;
    using cell_dof_descriptor = FeType::template cell_dof_descriptor<local_dim>;
    using face_dof_descriptor = FeType::template face_dof_descriptor<local_dim>;
    using ReferenceCell = typename cell_dof_descriptor::ReferenceCell;
    using CellBasisType = typename cell_dof_descriptor::BasisType;
    using CellShapeFunctionType = subscript_t<CellBasisType>;
    using FaceBasisType = typename face_dof_descriptor::BasisType;
    using FaceShapeFunctionType = subscript_t<FaceBasisType>;
    using DofHandlerType = DofHandler<local_dim, embed_dim, finite_element_tag>;
    // vector finite element descriptors
    static constexpr int n_components  = FeType::n_components;
    static constexpr bool is_vector_fe = (n_components > 1);
    // definition of assembly loops
    using discretization_category = finite_element_tag;
    constexpr int sobolev_regularity() const { return 1; }
    template <typename Triangulation__, typename Form__, int Options__, typename... Quadrature__>
    using bilinear_form_assembly_loop =
      internals::fe_bilinear_form_assembly_loop<Triangulation__, Form__, Options__, Quadrature__...>;
    template <typename Triangulation__, typename Form__, int Options__, typename... Quadrature__>
    using linear_form_assembly_loop =
      internals::fe_linear_form_assembly_loop  <Triangulation__, Form__, Options__, Quadrature__...>;

    FeSpace() = default;
    FeSpace(const Triangulation_& triangulation, FeType_ fe) :
        triangulation_(std::addressof(triangulation)), dof_handler_(triangulation) {
        dof_handler_.enumerate(fe);
        cell_basis_ = CellBasisType(unit_cell_dofs_.dofs_phys_coords());
        // degenerate case for 1D boundary integration uses default initialization
        if constexpr (local_dim - 1 != 0) { face_basis_ = FaceBasisType(unit_face_dofs_.dofs_phys_coords()); }
    }
    // observers
    const Triangulation& triangulation() const { return *triangulation_; }
    const DofHandlerType& dof_handler() const { return dof_handler_; }
    DofHandlerType& dof_handler() { return dof_handler_; }
    constexpr int n_shape_functions() const { return n_components * cell_basis_.size(); }
    constexpr int n_shape_functions_face() const { return n_components * face_basis_.size(); }
    int n_dofs() const { return dof_handler_.n_dofs(); }
    const CellBasisType& cell_basis() const { return cell_basis_; }
    const FaceBasisType& face_basis() const { return face_basis_; }
    // evaluation
    template <typename InputType>
        requires(std::is_invocable_v<CellShapeFunctionType, InputType>)
    constexpr auto eval_shape_value(int i, const InputType& p) const {
        return cell_basis_[i](p);
    }
    template <typename InputType>
        requires(std::is_invocable_v<decltype(std::declval<CellShapeFunctionType>().gradient()), InputType>)
    constexpr auto eval_shape_grad(int i, const InputType& p) const {
        return cell_basis_[i].gradient()(p);
    }
    template <typename InputType>
        requires(std::is_invocable_v<decltype(std::declval<CellShapeFunctionType>().divergence()), InputType>)
    constexpr auto eval_shape_div(int i, const InputType& p) const {
        fdapde_static_assert(n_components > 1, THIS_METHOD_IS_FOR_VECTOR_FINITE_ELEMENTS_ONLY);
        return cell_basis_[i].divergence()(p);
    }
    template <typename InputType>
        requires(std::is_invocable_v<FaceShapeFunctionType, InputType>)
    constexpr auto eval_face_shape_value(int i, const InputType& p) const {
        return face_basis_[i](p);
    }
    template <typename InputType>
        requires(std::is_invocable_v<decltype(std::declval<FaceShapeFunctionType>().gradient()), InputType>)    
    constexpr auto eval_face_shape_grad(int i, const InputType& p) const {
        return face_basis_[i].gradient()(p);
    }
    template <typename InputType>
        requires(std::is_invocable_v<decltype(std::declval<FaceShapeFunctionType>().divergence()), InputType>)
    constexpr auto eval_face_shape_div(int i, const InputType& p) const {
        fdapde_static_assert(n_components > 1, THIS_METHOD_IS_FOR_VECTOR_FINITE_ELEMENTS_ONLY);
        return face_basis_[i].divergence()(p);
    }
    // evaluation on physical domain
    // shape function evaluation, skip point location step (cell_id provided as input)
    template <typename InputType> auto eval_cell_value(int i, int cell_id, const InputType& p) const {
        // map p to reference cell and evaluate
        typename DofHandlerType::CellType cell = dof_handler_.cell(cell_id);
        InputType ref_p = cell.invJ() * (p - cell.node(0));
        return eval_shape_value(i, ref_p);
    }
    // evaluate value of the i-th shape function defined on the physical cell containing point p
    template <typename InputType> auto eval_cell_value(int i, const InputType& p) const {
        // localize p in physical domain
        int cell_id = triangulation_->locate(p);
        if (cell_id == -1) return std::numeric_limits<double>::quiet_NaN();
	return eval_cell_value(i, cell_id, p);
    }
    // access i-th basis function on physical domain
    FeFunction<FeSpace<Triangulation_, FeType_>> operator[](int i) {
        fdapde_assert(i < dof_handler_.n_dofs());
	Eigen::Matrix<double, Dynamic, 1> coeff = Eigen::Matrix<double, Dynamic, 1>::Zero(dof_handler_.n_dofs());
	coeff[i] = 1;
        return FeFunction<FeSpace<Triangulation_, FeType_>>(*this, coeff);
    }
    // boundary conditions
    template <typename... Data> void impose_dirichlet_constraint(int on, Data&&... g) {
        dof_handler_.set_dirichlet_constraint(on, g...);
    }
    template <typename Iterator, typename... Data>
        requires(std::input_iterator<Iterator>)
    void impose_dirichlet_constraint(Iterator begin, Iterator end, Data&&... g) {
        for (auto it = begin; it != end; ++it) { dof_handler_.set_dirichlet_constraint(*it, g...); }
    }
    template <typename... Data> void impose_dirichlet_constraint(const std::initializer_list<int>& on, Data&&... g) {
        impose_dirichlet_constraint(on.begin(), on.end(), std::forward<Data>(g)...);
    }
    template <typename... Data> void impose_dirichlet_constraint(Data&&... g) {
        impose_dirichlet_constraint(BoundaryAll, g...);
    }
   private:
    const Triangulation* triangulation_;
    DofHandlerType dof_handler_;
    cell_dof_descriptor unit_cell_dofs_;
    face_dof_descriptor unit_face_dofs_;
    CellBasisType cell_basis_ {};
    FaceBasisType face_basis_ {};
};

// detection trait for finite element spaces
template <typename FuncSpace> struct is_fe_space {
    static constexpr bool value = std::is_same_v<typename FuncSpace::discretization_category, finite_element_tag>;
};
template <typename FuncSpace> static constexpr bool is_fe_space_v = is_fe_space<FuncSpace>::value;

}   // namespace fdapde

#endif   // __FDAPDE_FE_SPACE_H__
