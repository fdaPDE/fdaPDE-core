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

#ifndef __FDAPDE_FE_BILINEAR_FORM_ASSEMBLER_H__
#define __FDAPDE_FE_BILINEAR_FORM_ASSEMBLER_H__

#include "header_check.h"

namespace fdapde {
namespace internals {
  
// galerkin and petrov-galerkin vector finite element assembly loop
template <typename Triangulation_, typename Form_, int Options_, typename... Quadrature_>
class fe_bilinear_form_assembly_loop :
    public fe_assembler_base<Triangulation_, Form_, Options_, Quadrature_...>,
    public assembly_xpr_base<fe_bilinear_form_assembly_loop<Triangulation_, Form_, Options_, Quadrature_...>> {
   public:
    // detect trial and test spaces from bilinear form
    using TrialSpace = trial_space_t<Form_>;
    using TestSpace  = test_space_t <Form_>;
    static_assert(TrialSpace::local_dim == TestSpace::local_dim && TrialSpace::embed_dim == TestSpace::embed_dim);
    static constexpr bool is_galerkin = std::is_same_v<TrialSpace, TestSpace>;
    static constexpr bool is_petrov_galerkin = !is_galerkin;
    using Base = fe_assembler_base<Triangulation_, Form_, Options_, Quadrature_...>;
    using Form = typename Base::Form;
    using DofHandlerType = typename Base::DofHandlerType;
    using discretization_category = typename TestSpace::discretization_category;
    fdapde_static_assert(
      std::is_same_v<discretization_category FDAPDE_COMMA typename TrialSpace::discretization_category>,
      TEST_AND_TRIAL_SPACE_MUST_HAVE_THE_SAME_DISCRETIZATION_CATEGORY);
    static constexpr int local_dim = Base::local_dim;
    static constexpr int embed_dim = Base::embed_dim;
    using Base::form_;
    // as trial and test spaces could be different, we here need to redefine some properties of Base
    // trial space properties
    using TrialFeType = typename TrialSpace::FeType;
    using trial_fe_traits = std::conditional_t<
      Options_ == CellMajor, fe_cell_assembler_traits<TrialSpace, Quadrature_...>,
      fe_face_assembler_traits<TrialSpace, Quadrature_...>>;
    using trial_dof_descriptor = typename trial_fe_traits::dof_descriptor;
    using TrialBasisType = typename trial_dof_descriptor::BasisType;
    static constexpr int n_trial_basis = TrialBasisType::n_basis;
    static constexpr int n_trial_components = TrialSpace::n_components;
    // test space properties
    using TestFeType = typename TestSpace::FeType;
    using test_fe_traits = std::conditional_t<
      Options_ == CellMajor, fe_cell_assembler_traits<TestSpace, Quadrature_...>,
      fe_face_assembler_traits<TestSpace, Quadrature_...>>;
    using test_dof_descriptor = typename test_fe_traits::dof_descriptor;
    using TestBasisType = typename test_dof_descriptor::BasisType;
    static constexpr int n_test_basis = TestBasisType::n_basis;
    static constexpr int n_test_components = TestSpace::n_components;

    using Quadrature = std::conditional_t<
      (is_galerkin || sizeof...(Quadrature_) > 0), typename Base::Quadrature,
      higher_degree_fe_quadrature_t<
        typename TrialFeType::template cell_quadrature_t<local_dim>,
        typename TestFeType ::template cell_quadrature_t<local_dim>>>;
    static constexpr int n_quadrature_nodes = Quadrature::order;
   private:
    // selected Quadrature could be different than Base::Quadrature, evaluate trial and (re-evaluate) test functions
    static constexpr auto test_shape_values_  = Base::template eval_shape_values<Quadrature, test_fe_traits >();
    static constexpr auto trial_shape_values_ = Base::template eval_shape_values<Quadrature, trial_fe_traits>();
    static constexpr auto test_shape_grads_   = Base::template eval_shape_grads <Quadrature, test_fe_traits >();
    static constexpr auto trial_shape_grads_  = Base::template eval_shape_grads <Quadrature, trial_fe_traits>();
    static constexpr auto test_shape_hess_    = Base::template eval_shape_hess  <Quadrature, test_fe_traits >();
    static constexpr auto trial_shape_hess_   = Base::template eval_shape_hess  <Quadrature, trial_fe_traits>();
    // private data members
    const DofHandlerType* trial_dof_handler_;
    Quadrature quadrature_ {};
    constexpr const DofHandlerType* test_dof_handler() const { return Base::dof_handler_; }
    constexpr const DofHandlerType* trial_dof_handler() const {
        return is_galerkin ? Base::dof_handler_ : trial_dof_handler_;
    }
    const TrialSpace* trial_space_;
   public:
    fe_bilinear_form_assembly_loop() = default;
    fe_bilinear_form_assembly_loop(
      const Form_& form, typename Base::fe_traits::geo_iterator begin, typename Base::fe_traits::geo_iterator end,
      const Quadrature_&... quadrature)
        requires(sizeof...(quadrature) <= 1)
        : Base(form, begin, end, quadrature...), trial_space_(&internals::trial_space(form_)) {
        if constexpr (is_petrov_galerkin) { trial_dof_handler_ = &internals::trial_space(form_).dof_handler(); }
        fdapde_assert(test_dof_handler()->n_dofs() != 0 && trial_dof_handler()->n_dofs() != 0);
    }

    Eigen::SparseMatrix<double> assemble() const {
        Eigen::SparseMatrix<double> assembled_mat(test_dof_handler()->n_dofs(), trial_dof_handler()->n_dofs());
        std::vector<Eigen::Triplet<double>> triplet_list;
	assemble(triplet_list);
	// linearity of the integral is implicitly used here, as duplicated triplets are summed up (see Eigen docs)
        assembled_mat.setFromTriplets(triplet_list.begin(), triplet_list.end());
        assembled_mat.makeCompressed();
        return assembled_mat;
    }
    void assemble(std::vector<Eigen::Triplet<double>>& triplet_list) const {
        using iterator = typename Base::fe_traits::dof_iterator;
        iterator begin(Base::begin_.index(), test_dof_handler(), Base::begin_.marker());
        iterator end  (Base::end_.index(),   test_dof_handler(), Base::end_.marker()  );
        // prepare assembly loop
	Eigen::Matrix<int, Dynamic, 1> test_active_dofs, trial_active_dofs;
        MdArray<double, MdExtents<n_test_basis,  n_quadrature_nodes, embed_dim, n_test_components >> test_grads;
        MdArray<double, MdExtents<n_trial_basis, n_quadrature_nodes, embed_dim, n_trial_components>> trial_grads;
        Matrix<double, n_test_basis , n_quadrature_nodes> test_divs;
        Matrix<double, n_trial_basis, n_quadrature_nodes> trial_divs;
        MdArray<double, MdExtents<n_test_basis,  n_quadrature_nodes, n_test_components,  embed_dim, embed_dim>>
	  test_hess;
        MdArray<double, MdExtents<n_trial_basis, n_quadrature_nodes, n_trial_components, embed_dim, embed_dim>>
          trial_hess;

        if constexpr (Form::XprBits & int(fe_assembler_flags::compute_physical_quad_nodes)) {
            Base::distribute_quadrature_nodes(begin, end);
        }
        // start assembly loop
        internals::fe_assembler_packet<embed_dim> fe_packet(n_trial_components, n_test_components);
	// if hessians are zero, assemble physical hessian once and never update
        constexpr bool test_hess_is_zero = std::all_of(
          test_shape_hess_.data(), test_shape_hess_.data() + test_shape_hess_.size(), [](double x) { return x == 0; });
        constexpr bool trial_hess_is_zero =
          std::all_of(trial_shape_hess_.data(), trial_shape_hess_.data() + trial_shape_hess_.size(), [](double x) {
              return x == 0;
          });
        if constexpr (test_hess_is_zero ) { std::fill_n(fe_packet.test_hess.data(), fe_packet.test_hess.size(), 0.0); }
        if constexpr (trial_hess_is_zero) {
            std::fill_n(fe_packet.trial_hess.data(), fe_packet.trial_hess.size(), 0.0);
        }

        int local_cell_id = 0;
        for (iterator it = begin; it != end; ++it) {
            // update fe_packet content based on form requests
            fe_packet.measure = it->measure();
            if constexpr (Form::XprBits & int(geo_assembler_flags::compute_geo_id)) { fe_packet.geo_id = it->id(); }
            if constexpr (Form::XprBits & int(geo_assembler_flags::compute_face_normal)) {
                fdapde_static_assert(Options_ == FaceMajor, BILINEAR_FORM_REQUIRES_A_FACE_MAJOR_ASSEMBLY_LOOP);
                fe_packet.normal.assign_inplace_from(it->normal());
            }
            if constexpr (Form::XprBits & int(fe_assembler_flags::compute_shape_grad)) {
                Base::eval_shape_grads_on_cell(it, test_shape_grads_, test_grads);
                if constexpr (is_petrov_galerkin) Base::eval_shape_grads_on_cell(it, trial_shape_grads_, trial_grads);
            }
            if constexpr (Form::XprBits & int(fe_assembler_flags::compute_shape_div)) {
                fdapde_static_assert(
                  n_test_components != 1 || n_trial_components != 1,
                  DIVERGENCE_OPERATOR_IS_DEFINED_ONLY_FOR_VECTOR_ELEMENTS);
                if constexpr (n_test_components != 1) Base::eval_shape_div_on_cell(it, test_shape_grads_, test_divs);
                if constexpr (is_petrov_galerkin && n_trial_components != 1)
                    Base::eval_shape_div_on_cell(it, trial_shape_grads_, trial_divs);
            }
            if constexpr (Form::XprBits & int(fe_assembler_flags::compute_shape_hess)) {
                if constexpr (!test_hess_is_zero) Base::eval_shape_hess_on_cell(it, test_shape_hess_, test_hess);
                if constexpr (is_petrov_galerkin && !trial_hess_is_zero)
                    Base::eval_shape_hess_on_cell(it, trial_shape_hess_, trial_hess);
            }

            // perform integration of weak form for (i, j)-th basis pair
            test_active_dofs = it->dofs();
            if constexpr (is_petrov_galerkin) { trial_active_dofs = trial_dof_handler()->active_dofs(it->id()); }
            for (int i = 0; i < n_trial_basis; ++i) {      // trial function loop
                for (int j = 0; j < n_test_basis; ++j) {   // test function loop
                    double value = 0;
                    for (int q_k = 0; q_k < n_quadrature_nodes; ++q_k) {
                        if constexpr (Form::XprBits & int(fe_assembler_flags::compute_shape_values)) {
                            fe_packet.trial_value.assign_inplace_from(trial_shape_values_.template slice<0, 1>(i, q_k));
                            fe_packet.test_value .assign_inplace_from(test_shape_values_ .template slice<0, 1>(j, q_k));
                        }
                        if constexpr (Form::XprBits & int(fe_assembler_flags::compute_shape_grad)) {
                            fe_packet.trial_grad.assign_inplace_from(is_galerkin ?
                                test_grads.template slice<0, 1>(i, q_k) : trial_grads.template slice<0, 1>(i, q_k));
                            fe_packet.test_grad .assign_inplace_from(test_grads.template slice<0, 1>(j, q_k));
                        }
                        if constexpr (Form::XprBits & int(fe_assembler_flags::compute_shape_div)) {
                            if constexpr (n_trial_components != 1) {
                                fe_packet.trial_div =
                                  (is_galerkin && n_test_components != 1) ? test_divs(i, q_k) : trial_divs(i, q_k);
                            }
                            if constexpr (n_test_components != 1) fe_packet.test_div = test_divs(j, q_k);
                        }
                        if constexpr (Form::XprBits & int(fe_assembler_flags::compute_shape_hess)) {
                            if constexpr (!trial_hess_is_zero)
                                fe_packet.trial_hess.assign_inplace_from(is_galerkin ?
				    test_hess.template slice<0, 1>(i, q_k) : trial_hess.template slice<0, 1>(i, q_k));
                            if constexpr (!test_hess_is_zero)
                                fe_packet.test_hess.assign_inplace_from(test_hess.template slice<0, 1>(j, q_k));
                        }
                        if constexpr (Form::XprBits & int(fe_assembler_flags::compute_physical_quad_nodes)) {
                            fe_packet.quad_node_id = local_cell_id * n_quadrature_nodes + q_k;
                        }
                        value += Quadrature::weights[q_k] * form_(fe_packet);
                    }
                    triplet_list.emplace_back(
                      test_active_dofs[j], is_galerkin ? test_active_dofs[i] : trial_active_dofs[i],
                      value * fe_packet.measure);
                }
            }
            local_cell_id++;
        }
        return;
    }
    constexpr int n_dofs() const { return trial_dof_handler()->n_dofs(); }
    constexpr int rows() const { return test_dof_handler()->n_dofs(); }
    constexpr int cols() const { return trial_dof_handler()->n_dofs(); }
    constexpr const TrialSpace& trial_space() const { return *trial_space_; }
};


// optimized computation of discretized laplace operator (\int_D (\grad{\psi_i} * \grad{\psi_j})) for scalar elements
template <typename DofHandler, typename FeType> class scalar_fe_grad_grad_assembly_loop {
    static constexpr int local_dim = DofHandler::local_dim;
    static constexpr int embed_dim = DofHandler::embed_dim;
    using Quadrature = typename FeType::template cell_quadrature_t<local_dim>;
    using cell_dof_descriptor = FeType::template cell_dof_descriptor<local_dim>;
    using BasisType = typename cell_dof_descriptor::BasisType;
    static constexpr int n_quadrature_nodes = Quadrature::order;
    static constexpr int n_basis = BasisType::n_basis;
    static constexpr int n_components = FeType::n_components;
    fdapde_static_assert(n_components == 1, THIS_CLASS_IS_FOR_SCALAR_FINITE_ELEMENTS_ONLY);
    // compile-time evaluation of \nabla{\psi_i}(q_j), i = 1, ..., n_basis, j = 1, ..., n_quadrature_nodes
    static constexpr std::array<Matrix<double, local_dim, n_quadrature_nodes>, n_basis> shape_grad_ {[]() {
        std::array<Matrix<double, local_dim, n_quadrature_nodes>, n_basis> shape_grad_ {};
        BasisType basis {cell_dof_descriptor().dofs_phys_coords()};
        for (int i = 0; i < n_basis; ++i) {
            // evaluation of \nabla{\psi_i} at q_j, j = 1, ..., n_quadrature_nodes
            std::array<double, n_quadrature_nodes * local_dim> grad_eval_ {};
            for (int k = 0; k < n_quadrature_nodes; ++k) {
                auto grad = basis[i].gradient()(Quadrature::nodes.row(k).transpose());
                for (int j = 0; j < local_dim; ++j) { grad_eval_[j * n_quadrature_nodes + k] = grad[j]; }
            }
            shape_grad_[i] = Matrix<double, local_dim, n_quadrature_nodes>(grad_eval_);
        }
        return shape_grad_;
    }()};
    DofHandler* dof_handler_;
   public:
    scalar_fe_grad_grad_assembly_loop() = default;
    scalar_fe_grad_grad_assembly_loop(DofHandler& dof_handler) : dof_handler_(&dof_handler) { }
    Eigen::SparseMatrix<double> assemble() {
        if (!dof_handler_) dof_handler_->enumerate(FeType {});
        int n_dofs = dof_handler_->n_dofs();
        Eigen::SparseMatrix<double> assembled_mat(n_dofs, n_dofs);
        // prepare assembly loop
        std::vector<Eigen::Triplet<double>> triplet_list;
        Eigen::Matrix<int, Dynamic, 1> active_dofs;
        std::array<Matrix<double, embed_dim, n_quadrature_nodes>, n_basis> shape_grad;
        for (typename DofHandler::cell_iterator it = dof_handler_->cells_begin(); it != dof_handler_->cells_end();
             ++it) {
            active_dofs = it->dofs();
            // map reference cell shape functions' gradient on physical cell
            for (int i = 0; i < n_basis; ++i) {
                for (int j = 0; j < n_quadrature_nodes; ++j) {
                    shape_grad[i].col(j) =
                      Map<const double, local_dim, embed_dim>(it->invJ().data()).transpose() * shape_grad_[i].col(j);
                }
            }
            for (int i = 0; i < BasisType::n_basis; ++i) {
                for (int j = 0; j < i + 1; ++j) {
                    std::pair<const int&, const int&> minmax(std::minmax(active_dofs[i], active_dofs[j]));
                    double value = 0;
                    for (int k = 0; k < n_quadrature_nodes; ++k) {
                        value += Quadrature::weights[k] * (shape_grad[i].col(k)).dot(shape_grad[j].col(k));
                    }
                    triplet_list.emplace_back(minmax.first, minmax.second, value * it->measure());
                }
            }
        }
        // linearity of the integral is implicitly used here, as duplicated triplets are summed up (see Eigen docs)
        assembled_mat.setFromTriplets(triplet_list.begin(), triplet_list.end());
        assembled_mat.makeCompressed();
	return assembled_mat.selfadjointView<Eigen::Upper>();
    }
    constexpr int n_dofs() const { return dof_handler_->n_dofs(); }
    constexpr int rows() const { return n_dofs(); }
    constexpr int cols() const { return n_dofs(); }  
};
  
}   // namespace internals
}   // namespace fdapde

#endif   // __FDAPDE_FE_BILINEAR_FORM_ASSEMBLER_H__
