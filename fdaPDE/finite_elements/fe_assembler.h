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

#ifndef __FE_ASSEMBLER_H__
#define __FE_ASSEMBLER_H__

#include <unordered_map>

#include "../fields/meta.h"
#include "../linear_algebra/constexpr_matrix.h"

namespace fdapde {

template <typename Derived_> struct FeMap;
  
enum bilinear_bits {
    compute_shape_values        = 0x0001,
    compute_shape_grad          = 0x0002,
    compute_second_derivatives  = 0x0004,
    compute_physical_quad_nodes = 0x0010
};

namespace internals {

// detect trial space from bilinear form
template <typename Xpr> constexpr decltype(auto) trial_space(Xpr xpr) {
    constexpr bool found = meta::xpr_find<
      decltype([]<typename Xpr_>() { return requires { typename Xpr_::TrialSpace; }; }), std::decay_t<Xpr>>();
    fdapde_static_assert(found, NO_TRIAL_SPACE_FOUND_IN_EXPRESSION);
    return meta::xpr_query<
      decltype([]<typename Xpr_>(Xpr_ xpr) { return xpr.fe_space(); }),
      decltype([]<typename Xpr_>() { return requires { typename Xpr_::TrialSpace; }; })>(xpr);
}
// detect test space from bilinear form
template <typename Xpr> constexpr decltype(auto) test_space(Xpr xpr) {
    constexpr bool found = meta::xpr_find<
      decltype([]<typename Xpr_>() { return requires { typename Xpr_::TestSpace; }; }), std::decay_t<Xpr>>();
    fdapde_static_assert(found, NO_TEST_SPACE_FOUND_IN_EXPRESSION);
    return meta::xpr_query<
      decltype([]<typename Xpr_>(Xpr_ xpr) { return xpr.fe_space(); }),
      decltype([]<typename Xpr_>() { return requires { typename Xpr_::TestSpace; }; })>(xpr);
}

template <int LocalDim> struct fe_assembler_packet {
    int quad_node_id;                 // physical active quadrature node index
    double cell_measure;              // measure of active cell
    double cell_diameter;             // diameter of active cell
    double trial_value, test_value;   // \psi_i(q_k), \psi_j(q_k) at active quadrature node
    // \nabla{\psi_i}(q_k), \nabla{\psi_j}(q_k) at active quadrature node
    cexpr::Vector<double, LocalDim> trial_grad, test_grad;
};
  
// optimized computation of mass matrix [A]_{ij} = \int_D (\psi_i * \psi_j)
template <typename DofHandler, typename FeType> class fe_assembler_mass_loop {
    static constexpr int local_dim = DofHandler::local_dim;
    static constexpr int embed_dim = DofHandler::embed_dim;
    using Quadrature = typename FeType::template select_cell_quadrature_t<local_dim>;
    using cell_dof_descriptor = FeType::template cell_dof_descriptor<local_dim>;
    using BasisType = typename cell_dof_descriptor::BasisType;
    static constexpr int n_quadrature_nodes = Quadrature::n_nodes;
    static constexpr int n_basis = BasisType::n_basis;
    // compile-time evaluation of integral \int_{\hat K} \psi_i \psi_j on reference element \hat K
    static constexpr cexpr::Matrix<double, n_basis, n_basis> int_table_ {[]() {
        std::array<double, n_basis * n_basis> int_table_ {};
        BasisType basis {cell_dof_descriptor().dofs_phys_coords()};
        for (int i = 0; i < n_basis; ++i) {
            for (int j = 0; j < n_basis; ++j) {
                for (int k = 0; k < n_quadrature_nodes; ++k) {
                    int_table_[i * n_basis + j] +=
                      Quadrature::weights[k] * (basis[i] * basis[j])(Quadrature::nodes.row(k).transpose());
                }
            }
        }
        return int_table_;
    }};
    DofHandler* dof_handler_;
   public:
    fe_assembler_mass_loop() = default;
    fe_assembler_mass_loop(DofHandler& dof_handler) : dof_handler_(&dof_handler) { }

    SpMatrix<double> run() {
        if (!dof_handler_) dof_handler_->enumerate(FeType {});
        SpMatrix<double> assembled_mat(dof_handler_->n_dofs(), dof_handler_->n_dofs());
        std::vector<Eigen::Triplet<double>> triplet_list;
        DVector<int> active_dofs;
        for (typename DofHandler::cell_iterator it = dof_handler_->cells_begin(); it != dof_handler_->cells_end();
             ++it) {
            active_dofs = it->dofs();
            for (int i = 0; i < BasisType::n_basis; ++i) {
                for (int j = 0; j < i + 1; ++j) {
                    std::pair<const int&, const int&> minmax(std::minmax(active_dofs[i], active_dofs[j]));
                    triplet_list.emplace_back(minmax.first, minmax.second, int_table_(i, j) * it->measure());
                }
            }
        }
        // linearity of the integral is implicitly used here, as duplicated triplets are summed up (see Eigen docs)
        assembled_mat.setFromTriplets(triplet_list.begin(), triplet_list.end());
        assembled_mat.makeCompressed();
        return assembled_mat.selfadjointView<Eigen::Upper>();
    }
};

// optimized computation of discretized Laplace [A]_{ij} = \int_D (\grad{\psi_i} * \grad{\psi_j})
template <typename DofHandler, typename FeType> class fe_assembler_grad_grad_loop {
    static constexpr int local_dim = DofHandler::local_dim;
    static constexpr int embed_dim = DofHandler::embed_dim;
    using Quadrature = typename FeType::template select_cell_quadrature_t<local_dim>;
    using cell_dof_descriptor = FeType::template cell_dof_descriptor<local_dim>;
    using BasisType = typename cell_dof_descriptor::BasisType;
    static constexpr int n_quadrature_nodes = Quadrature::n_nodes;
    static constexpr int n_basis = BasisType::n_basis;
    // compile-time evaluation of \nabla{\psi_i}(q_j), i = 1, ..., n_basis, j = 1, ..., n_quadrature_nodes
    static constexpr std::array<cexpr::Matrix<double, local_dim, n_quadrature_nodes>, n_basis> shape_grad_ {[]() {
        std::array<cexpr::Matrix<double, local_dim, n_quadrature_nodes>, n_basis> shape_grad_ {};
        BasisType basis {cell_dof_descriptor().dofs_phys_coords()};
        for (int i = 0; i < n_basis; ++i) {
            // evaluation of \nabla{\psi_i} at q_j, j = 1, ..., n_quadrature_nodes
            std::array<double, n_quadrature_nodes * local_dim> grad_eval_ {};
            for (int k = 0; k < n_quadrature_nodes; ++k) {
                auto grad = basis[i].gradient()(Quadrature::nodes.row(k).transpose());
                for (int j = 0; j < local_dim; ++j) { grad_eval_[j * n_quadrature_nodes + k] = grad[j]; }
            }
            shape_grad_[i] = cexpr::Matrix<double, local_dim, n_quadrature_nodes>(grad_eval_);
        }
        return shape_grad_;
    }()};
    DofHandler* dof_handler_;
   public:
    fe_assembler_grad_grad_loop() = default;
    fe_assembler_grad_grad_loop(DofHandler& dof_handler) : dof_handler_(&dof_handler) { }

    SpMatrix<double> run() {
        if (!dof_handler_) dof_handler_->enumerate(FeType {});
        SpMatrix<double> assembled_mat(dof_handler_->n_dofs(), dof_handler_->n_dofs());
	// prepare assembly loop
        std::vector<Eigen::Triplet<double>> triplet_list;
        DVector<int> active_dofs;
        std::array<cexpr::Matrix<double, local_dim, n_quadrature_nodes>, n_basis> shape_grad;
        for (typename DofHandler::cell_iterator it = dof_handler_->cells_begin(); it != dof_handler_->cells_end();
             ++it) {
            active_dofs = it->dofs();
	    // map reference cell shape functions' gradient on physical cell
            for (int i = 0; i < n_basis; ++i) {
                for (int j = 0; j < n_quadrature_nodes; ++j) {
                    shape_grad[i].col(j) =
                      cexpr::Map<const double, local_dim, embed_dim>(it->invJ().data()).transpose() *
                      shape_grad_[i].col(j);
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
};

// implementation of galerkin assembly loop for the discretization of arbitrarily bilinear forms 
template <typename Triangulation_, typename Form_, typename... Quadrature_> class fe_assembler_galerkin_loop {
    // detect trial and test spaces from bilinear form
    using TrialSpace = std::decay_t<decltype(trial_space(std::declval<Form_>()))>;
    using TestSpace  = std::decay_t<decltype(test_space (std::declval<Form_>()))>;
    fdapde_static_assert(
      std::is_same_v<TrialSpace FDAPDE_COMMA TestSpace>,
      GALERKIN_ASSEMBLY_LOOP_REQUIRES_TRIAL_AND_TEST_SPACES_TO_BE_EQUAL);
   public:
    using Form = std::decay_t<decltype(meta::xpr_wrap<FeMap, decltype([]<typename Xpr>() {
          return !std::is_invocable_v<
            Xpr, fe_assembler_packet<Xpr::StaticInputSize>>;
        })>(std::declval<Form_>()))>;
    using Triangulation = typename std::decay_t<Triangulation_>;
    using FunctionSpace = TrialSpace;
    using FeType = typename FunctionSpace::FeType;
    static constexpr int local_dim = Triangulation::local_dim;
    static constexpr int embed_dim = Triangulation::embed_dim;
    using Quadrature = decltype([]() {
        if constexpr (sizeof...(Quadrature_) == 0) {
            return typename FeType::template select_cell_quadrature_t<local_dim> {};
        } else {
            return std::get<0>(std::tuple<Quadrature_...>());
        }
    }());
    using cell_dof_descriptor = FeType::template cell_dof_descriptor<local_dim>;
    using BasisType = typename cell_dof_descriptor::BasisType;
    static constexpr int n_quadrature_nodes = Quadrature::n_nodes;
    static constexpr int n_basis = BasisType::n_basis;
    // compile-time evaluation of \psi_i(q_j), i = 1, ..., n_basis, j = 1, ..., n_quadrature_nodes
    static constexpr cexpr::Matrix<double, n_basis, n_quadrature_nodes> shape_values_ {[]() {
        std::array<double, n_basis * n_quadrature_nodes> shape_values_ {};
        BasisType basis {cell_dof_descriptor().dofs_phys_coords()};
        for (int i = 0; i < n_basis; ++i) {
            for (int j = 0; j < n_quadrature_nodes; ++j) {
                shape_values_[i * n_quadrature_nodes + j] = basis[i](Quadrature::nodes.row(j).transpose());
            }
        }
        return shape_values_;
    }()};
    // compile-time evaluation of \nabla{\psi_i}(q_j), i = 1, ..., n_basis, j = 1, ..., n_quadrature_nodes
    static constexpr std::array<cexpr::Matrix<double, local_dim, n_quadrature_nodes>, n_basis> shape_grad_ {[]() {
        std::array<cexpr::Matrix<double, local_dim, n_quadrature_nodes>, n_basis> shape_grad_ {};
        BasisType basis {cell_dof_descriptor().dofs_phys_coords()};
        for (int i = 0; i < n_basis; ++i) {
            // evaluation of \nabla{\psi_i} at q_j, j = 1, ..., n_quadrature_nodes
            std::array<double, n_quadrature_nodes * local_dim> grad_eval_ {};
            for (int k = 0; k < n_quadrature_nodes; ++k) {
                auto grad = basis[i].gradient()(Quadrature::nodes.row(k).transpose());
                for (int j = 0; j < local_dim; ++j) { grad_eval_[j * n_quadrature_nodes + k] = grad[j]; }
            }
            shape_grad_[i] = cexpr::Matrix<double, local_dim, n_quadrature_nodes>(grad_eval_);
        }
        return shape_grad_;
    }()};

    Form form_;
    Quadrature quadrature_ {};
    DofHandler<local_dim, embed_dim> dof_handler_;
    typename Triangulation::cell_iterator begin_, end_;
   public:
    fe_assembler_galerkin_loop() = default;
    fe_assembler_galerkin_loop(
      const Form_& form, typename Triangulation_::cell_iterator begin, typename Triangulation_::cell_iterator end,
      const Quadrature_&... quadrature) requires(sizeof...(quadrature) <= 1) :
        form_(meta::xpr_wrap<FeMap, decltype([]<typename Xpr>() {
                                 return !std::is_invocable_v<Xpr, fe_assembler_packet<Xpr::StaticInputSize>>;
                             })>(form)),
        quadrature_([... quadrature = std::forward<Quadrature_>(quadrature)]() {
            if constexpr (sizeof...(quadrature) == 1) {
                return std::get<0>(std::make_tuple(quadrature...));
            } else {
                return Quadrature {};
            }
        }()),
        dof_handler_(trial_space(form).dof_handler()),
        begin_(begin),
        end_(end) { }

    SpMatrix<double> run() {
        if (!dof_handler_) dof_handler_.enumerate(FeType {});
        // move Triangulation::cell_iterator to DofHandler::cell_iterator
        using iterator = typename DofHandler<local_dim, embed_dim>::cell_iterator;
        iterator begin(begin_.index(), &dof_handler_);
        iterator end(end_.index(), &dof_handler_);

        // prepare assembly loop
        SpMatrix<double> assembled_mat(dof_handler_.n_dofs(), dof_handler_.n_dofs());
        std::vector<Eigen::Triplet<double>> triplet_list;
        DVector<int> active_dofs;
        std::array<cexpr::Matrix<double, local_dim, n_quadrature_nodes>, n_basis>
          shape_grad;   // gradient of shape functions mapped on physical cell
        std::unordered_map<void*, DMatrix<double>> fe_map_buff;   // evaluation of FeMap nodes at quadrature points
	
        // distribute quadrature nodes on physical mesh (if required)
        DMatrix<double> quad_nodes;
        if constexpr (Form::XprBits & bilinear_bits::compute_physical_quad_nodes) {
            quad_nodes.resize(n_quadrature_nodes * (end_.index() - begin_.index()), embed_dim);
            int i = 0;
            for (typename Triangulation::cell_iterator it = begin_; it != end_; ++it) {
                for (int j = 0; j < n_quadrature_nodes; ++j) {
                    quad_nodes.row(i * n_quadrature_nodes + j) =
                      it->J() * Eigen::Map<const SMatrix<n_quadrature_nodes, local_dim>>(quadrature_.nodes.data())
                                  .row(j)
                                  .transpose() +
                      it->node(0);
                }
		i++;
            }
            // evaluate FeMap nodes at quadrature nodes
            meta::xpr_apply_if<
              decltype([]<typename Xpr_, typename... Args>(Xpr_& xpr, Args&&... args) {
                  xpr.subscribe(std::forward<Args>(args)...);
                  return;
              }),
              decltype([]<typename Xpr_>() {
                  return requires(Xpr_ xpr) { xpr.subscribe(fe_map_buff, quad_nodes, begin, end); };
              })>(form_, fe_map_buff, quad_nodes, begin, end);
        }

	// start assembly loop
        internals::fe_assembler_packet<local_dim> fe_packet;
        for (iterator it = begin; it != end; ++it) {
            fe_packet.cell_measure = it->measure();
            // if constexpr (WeakForm::Flags & UpdateCellDiameter) fe_data.cell_diameter = cell->diameter();
            if constexpr (Form::XprBits & bilinear_bits::compute_shape_grad) {
                for (int i = 0; i < n_basis; ++i) {
                    for (int j = 0; j < n_quadrature_nodes; ++j) {
                        shape_grad[i].col(j) =
                          cexpr::Map<const double, local_dim, embed_dim>(it->invJ().data()).transpose() *
                          shape_grad_[i].col(j);
                    }
                }
            }
            // perform integration of weak form for (i, j)-th basis pair
            active_dofs = it->dofs();
            for (int i = 0; i < BasisType::n_basis; ++i) {       // trial function loop
                for (int j = 0; j < BasisType::n_basis; ++j) {   // test function loop
                    double value = 0;
                    for (int q_k = 0; q_k < n_quadrature_nodes; ++q_k) {
                        // update fe_packet
                        fe_packet.trial_value = shape_values_(i, q_k);
                        fe_packet.test_value  = shape_values_(j, q_k);
                        if constexpr (Form::XprBits & bilinear_bits::compute_shape_grad) {
                            fe_packet.trial_grad = shape_grad[i].col(q_k);
                            fe_packet.test_grad  = shape_grad[j].col(q_k);
                        }
                        if constexpr (Form::XprBits & bilinear_bits::compute_physical_quad_nodes) {
                            fe_packet.quad_node_id = it->id() * n_quadrature_nodes + q_k;
                        }
                        value += Quadrature::weights[q_k] * form_(fe_packet);
                    }
                    triplet_list.emplace_back(active_dofs[i], active_dofs[j], value * fe_packet.cell_measure);
                }
            }
        }
        // linearity of the integral is implicitly used here, as duplicated triplets are summed up (see Eigen docs)
        assembled_mat.setFromTriplets(triplet_list.begin(), triplet_list.end());
        assembled_mat.makeCompressed();
        return assembled_mat;
    }
};

}   // namespace internals

template <typename Triangulation, typename... Quadrature> class FeCellAssembler {
    fdapde_static_assert(
      sizeof...(Quadrature) < 2, YOU_CAN_PROVIDE_AT_MOST_ONE_QUADRATURE_FORMULA_TO_A_FE_CELL_ASSEMBLER);
    std::tuple<Quadrature...> quadrature_;
    typename Triangulation::cell_iterator begin_, end_;
   public:
    FeCellAssembler(
      const Triangulation::cell_iterator& begin, const Triangulation::cell_iterator& end,
      const Quadrature&... quadrature) :
        begin_(begin), end_(end), quadrature_(std::make_tuple(quadrature...)) { }
    FeCellAssembler(const Triangulation& triangulation, const Quadrature&... quadrature) :
        FeCellAssembler(triangulation.cells_begin(), triangulation.cells_end(), quadrature...) { }

    template <typename Form>
    internals::fe_assembler_galerkin_loop<Triangulation, Form, Quadrature...> operator()(const Form& form) const {
        if constexpr (sizeof...(Quadrature) == 0) {
            return {form, begin_, end_};
        } else {
            return {form, begin_, end_, std::get<0>(quadrature_)};
        }
    }
};

template <typename Triangulation, typename... Quadrature>
FeCellAssembler<Triangulation, Quadrature...>
integrate(const Triangulation& triangulation, const Quadrature&... quadrature) {
    return FeCellAssembler<Triangulation, Quadrature...>(triangulation, quadrature...);
}

}   // namespace fdapde

#endif   // __FE_ASSEMBLER_H__
