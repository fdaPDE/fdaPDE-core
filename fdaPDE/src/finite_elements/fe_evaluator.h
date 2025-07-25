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

#ifndef __FDAPDE_FE_EVALUATOR_H__
#define __FDAPDE_FE_EVALUATOR_H__

#include "header_check.h"

namespace fdapde {
namespace internals {

template <typename Form_> struct fe_pointwise_evaluator_loop {
    using FeSpace = test_space_t<Form_>;
    static constexpr int local_dim = FeSpace::local_dim;
    static constexpr int embed_dim = FeSpace::embed_dim;
    static constexpr int n_components = FeSpace::n_components;
   private:
    using is_not_packet_evaluable = decltype([]<typename Xpr>() {
        return !(
          std::is_invocable_v<Xpr, fe_assembler_packet<embed_dim>> ||
          requires(Xpr xpr, int i, int j, fe_assembler_packet<embed_dim> input_type) {
              xpr.eval(i, j, input_type);   // vector case
          });
    });
   public:
    using Form = std::decay_t<decltype(xpr_wrap<FeMap, is_not_packet_evaluable>(std::declval<Form_>()))>;
    using FeType = typename FeSpace::FeType;
    using DofHandlerType = DofHandler<local_dim, embed_dim, finite_element_tag>;
    using discretization_category = typename FeSpace::discretization_category;
    fdapde_static_assert(
      std::is_same_v<discretization_category FDAPDE_COMMA finite_element_tag>,
      THIS_CLASS_IS_FOR_FINITE_ELEMENT_DISCRETIZATION_ONLY);

    fe_pointwise_evaluator_loop() = default;
    fe_pointwise_evaluator_loop(const Form_& form, const Eigen::Matrix<double, Dynamic, Dynamic>& locs) :
        form_(xpr_wrap<FeMap, is_not_packet_evaluable>(form)),
        dof_handler_(&internals::test_space(form_).dof_handler()),
        fe_space_(&internals::test_space(form_)),
        locs_(locs) {
        fdapde_assert(dof_handler_->n_dofs() > 0 && locs.rows() > 0 && locs.cols() == embed_dim);
        cell_ids_ = fe_space_->triangulation().locate(locs_);
    }

    Eigen::SparseMatrix<double> assemble() const {
        Eigen::SparseMatrix<double> evaluation_mat(locs_.rows(), dof_handler_->n_dofs());
        std::vector<Eigen::Triplet<double>> triplet_list;
        assemble(triplet_list);
        evaluation_mat.setFromTriplets(triplet_list.begin(), triplet_list.end());
        evaluation_mat.makeCompressed();
        return evaluation_mat;
    }

    void assemble(std::vector<Eigen::Triplet<double>>& triplet_list) const {
        int n_shape_functions = fe_space_->n_shape_functions();
        int n_locs = locs_.rows();
        triplet_list.reserve(n_locs * n_shape_functions);
        internals::fe_assembler_packet<embed_dim> fe_packet(n_components);
        // build evaluation matrix
        for (int i = 0; i < n_locs; ++i) {
            if (cell_ids_[i] != -1) {   // point falls inside domain
                auto cell = dof_handler_->cell(cell_ids_[i]);
                // map i-th point to reference element
                Matrix<double, embed_dim, 1> ref_p = cell.invJ() * (locs_.row(i).transpose() - cell.node(0));
                for (int h = 0; h < n_shape_functions; ++h) {
                    if constexpr (Form::XprBits & int(fe_assembler_flags::compute_shape_values)) {
                        fe_packet.test_value(0) = fe_space_->eval_cell_value(h, cell_ids_[i], ref_p);
                    }
                    if constexpr (Form::XprBits & int(fe_assembler_flags::compute_shape_grad)) {
                        fe_packet.test_grad.template slice<0>(0).assign_inplace_from(
                          fe_space_->eval_cell_grad(h, cell_ids_[i], ref_p));
                    }
                    if constexpr (Form::XprBits & int(fe_assembler_flags::compute_shape_hess)) {
                        if constexpr (!FeType::is_hess_zero) {
                            fe_packet.test_hess.template slice<0>(0).assign_inplace_from(
                              fe_space_->eval_cell_hess(h, cell_ids_[i], ref_p));
                        }
                    }
                    // evaluate form
                    double value = form_(fe_packet);
                    if (std::abs(value) > 1e-14) { triplet_list.emplace_back(i, cell.dofs()[h], value); }
                }
            }
        }
        return;
    }
   protected:
    Form form_;
    const DofHandlerType* dof_handler_;
    const FeSpace* fe_space_;
    Eigen::Matrix<double, Dynamic, Dynamic> locs_;
    Eigen::Matrix<int, Dynamic, 1> cell_ids_;
};

template <typename Locations> class evaluator_dispatch {
    Locations locs_;
   public:
    evaluator_dispatch() = default;
    evaluator_dispatch(const Locations& locs) : locs_(locs) { }

    template <typename Form> auto operator()(const Form& form) const {
        return fe_pointwise_evaluator_loop<Form>(form, locs_);
    }
};

}   // namespace internals

// main entry point for basis evaluation
template <typename Locations> auto eval_at(const Locations& locs) {
    return internals::evaluator_dispatch<Locations>(locs);
}

}   // namespace fdapde

#endif   // __FDAPDE_FE_EVALUATOR_H__
