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

#ifndef __FDAPDE_SP_BILINEAR_FORM_ASSEMBLER_H__
#define __FDAPDE_SP_BILINEAR_FORM_ASSEMBLER_H__

#include "header_check.h"

namespace fdapde {
namespace internals {
  
// galerkin and petrov-galerkin spline assembly loop
template <typename Triangulation_, typename Form_, int Options_, typename... Quadrature_>
class sp_bilinear_form_assembly_loop :
    public sp_assembler_base<Triangulation_, Form_, Options_, Quadrature_...>,
    public assembly_xpr_base<sp_bilinear_form_assembly_loop<Triangulation_, Form_, Options_, Quadrature_...>> {
   public:
    // detect trial and test spaces from bilinear form
    using TrialSpace = trial_space_t<Form_>;
    using TestSpace  = test_space_t <Form_>;
    static_assert(TrialSpace::local_dim == TestSpace::local_dim && TrialSpace::embed_dim == TestSpace::embed_dim);
    static constexpr bool is_galerkin = std::is_same_v<TrialSpace, TestSpace>;
    static constexpr bool is_petrov_galerkin = !is_galerkin;
    using Base = sp_assembler_base<Triangulation_, Form_, Options_, Quadrature_...>;
    using Form = typename Base::Form;
    using DofHandlerType = typename Base::DofHandlerType;
    using discretization_category = typename TestSpace::discretization_category;
    fdapde_static_assert(
      std::is_same_v<discretization_category FDAPDE_COMMA typename TrialSpace::discretization_category>,
      TEST_AND_TRIAL_SPACE_MUST_HAVE_THE_SAME_DISCRETIZATION_CATEGORY);
    static constexpr int local_dim = Base::local_dim;
    static constexpr int embed_dim = Base::embed_dim;
    using Base::form_;
    using Base::test_space_;
   private:
    // private data members
    const DofHandlerType* trial_dof_handler_;
    constexpr const DofHandlerType* test_dof_handler() const { return Base::dof_handler_; }
    constexpr const DofHandlerType* trial_dof_handler() const {
        return is_galerkin ? Base::dof_handler_ : trial_dof_handler_;
    }
    const TrialSpace* trial_space_;
   public:
    sp_bilinear_form_assembly_loop() = default;
    sp_bilinear_form_assembly_loop(
      const Form_& form, typename Base::geo_iterator begin, typename Base::geo_iterator end,
      const Quadrature_&... quadrature)
        requires(sizeof...(quadrature) <= 1)
        : Base(form, begin, end, quadrature...), trial_space_(std::addressof(internals::trial_space(form_))) {
        if constexpr (is_petrov_galerkin) {
            trial_dof_handler_ = std::addressof(internals::trial_space(form_).dof_handler());
        }
        fdapde_assert(test_dof_handler()->n_dofs() != 0 && trial_dof_handler()->n_dofs() != 0);
        if constexpr (sizeof...(Quadrature_) == 0) {
            // default to higher-order quadrature
            if (test_space_->degree() != trial_space_->degree()) {
                internals::get_sp_quadrature(
                  test_space_->degree() > trial_space_->degree() ? test_space_->degree() : trial_space_->degree(),
                  Base::quad_nodes_, Base::quad_weights_);
            }
        }
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
        using iterator = typename Base::dof_iterator;
        iterator begin(Base::begin_.index(), test_dof_handler(), Base::begin_.marker());
        iterator end  (Base::end_.index()  , test_dof_handler(), Base::end_.marker()  );
	// prepare assembly loop
        std::vector<int> test_active_dofs, trial_active_dofs;
        int n1 = test_space_->degree() + 1, n2 = (is_galerkin ? test_space_->degree() : trial_space_->degree()) + 1;
        int q = Base::n_quadrature_nodes_;
        MdArray<double, MdExtents<Dynamic, Dynamic>> test_shape_values(n1, q), trial_shape_values(n2, q);
        MdArray<double, MdExtents<Dynamic, Dynamic>> test_shape_dx    (n1, q), trial_shape_dx    (n2, q);
        MdArray<double, MdExtents<Dynamic, Dynamic>> test_shape_dxx   (n1, q), trial_shape_dxx   (n2, q);

        std::unordered_map<const void*, Eigen::Matrix<double, Dynamic, Dynamic>> sp_map_buff;
        if constexpr (Form::XprBits & int(sp_assembler_flags::compute_physical_quad_nodes)) {
            Base::distribute_quadrature_nodes(
              sp_map_buff, begin, end);   // distribute quadrature nodes on physical mesh (if required)
        }
	// start assembly loop
        internals::sp_assembler_packet<local_dim> sp_packet {};
        int local_cell_id = 0;
        for (iterator it = begin; it != end; ++it) {
            test_active_dofs = it->dofs();
            if constexpr (is_petrov_galerkin) { trial_active_dofs = trial_dof_handler()->active_dofs(it->id()); }
            // update fe_packet content based on form requests
            sp_packet.cell_measure = it->measure();
            if constexpr (Form::XprBits & int(sp_assembler_flags::compute_shape_values)) {
                Base::eval_shape_values(test_space_->physical_basis(), test_active_dofs, it, test_shape_values);
                Base::eval_shape_values(
                  trial_space_->physical_basis(), is_petrov_galerkin ? trial_active_dofs : test_active_dofs, it,
                  trial_shape_values);
            }
            if constexpr (Form::XprBits & int(sp_assembler_flags::compute_shape_dx)) {
                Base::eval_shape_dx(test_space_->physical_basis(), test_active_dofs, it, test_shape_dx);
                Base::eval_shape_dx(
                  trial_space_->physical_basis(), is_petrov_galerkin ? trial_active_dofs : test_active_dofs, it,
                  trial_shape_dx);
            }
            if constexpr (Form::XprBits & int(sp_assembler_flags::compute_shape_dxx)) {
                Base::eval_shape_dxx(test_space_->physical_basis(), test_active_dofs, it, test_shape_dxx);
                Base::eval_shape_dxx(
                  trial_space_->physical_basis(), is_petrov_galerkin ? trial_active_dofs : test_active_dofs, it,
                  trial_shape_dxx);
            }
            // perform integration of weak form for (i, j)-th basis pair
            for (int i = 0; i < n2; ++i) {
                for (int j = 0; j < n1; ++j) {
                    double value = 0;
                    for (int q_k = 0; q_k < Base::n_quadrature_nodes_; ++q_k) {
                        if constexpr (Form::XprBits & int(sp_assembler_flags::compute_shape_values)) {
                            sp_packet.trial_value = trial_shape_values(i, q_k);
                            sp_packet.test_value  = test_shape_values (j, q_k);
                        }
                        if constexpr (Form::XprBits & int(sp_assembler_flags::compute_shape_dx)) {
                            sp_packet.trial_dx = trial_shape_dx(i, q_k);
                            sp_packet.test_dx  = test_shape_dx (j, q_k);
                        }
                        if constexpr (Form::XprBits & int(sp_assembler_flags::compute_shape_dxx)) {
                            sp_packet.trial_dxx = trial_shape_dxx(i, q_k);
                            sp_packet.test_dxx  = test_shape_dxx (j, q_k);
                        }
                        if constexpr (Form::XprBits & int(sp_assembler_flags::compute_physical_quad_nodes)) {
                            sp_packet.quad_node_id = local_cell_id * Base::n_quadrature_nodes_ + q_k;
                        }
                        value += Base::quad_weights_(q_k, 0) * form_(sp_packet);
                    }
                    triplet_list.emplace_back(
                      test_active_dofs[j], is_galerkin ? test_active_dofs[i] : trial_active_dofs[i],
                      value * sp_packet.cell_measure * 0.5);
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

}   // namespace internals
}   // namespace fdapde

#endif   // __FDAPDE_SP_BILINEAR_FORM_ASSEMBLER_H__
