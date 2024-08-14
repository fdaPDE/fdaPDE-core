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

#include "../linear_algebra/constexpr_matrix.h"

namespace fdapde {

enum bilinear_bits {
    compute_shape_values = 0x0001,
    compute_shape_grad   = 0x0002
};
  
namespace internals {

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
    static constexpr std::array<cexpr::Matrix<double, local_dim, n_quadrature_nodes>, n_basis> grad_table_ {[]() {
        std::array<cexpr::Matrix<double, local_dim, n_quadrature_nodes>, n_basis> grad_table_ {};
        BasisType basis {cell_dof_descriptor().dofs_phys_coords()};
        for (int i = 0; i < n_basis; ++i) {
            // evaluation of \nabla{\psi_i} at q_j, j = 1, ..., n_quadrature_nodes
            std::array<double, n_quadrature_nodes * local_dim> grad_eval_ {};
            for (int k = 0; k < n_quadrature_nodes; ++k) {
                auto grad = basis[i].gradient()(Quadrature::nodes.row(k).transpose());
                for (int j = 0; j < local_dim; ++j) { grad_eval_[j * n_quadrature_nodes + k] = grad[j]; }
            }
            grad_table_[i] = cexpr::Matrix<double, local_dim, n_quadrature_nodes>(grad_eval_);
        }
        return grad_table_;
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
        cexpr::Matrix<double, local_dim, embed_dim> invJT;
        for (typename DofHandler::cell_iterator it = dof_handler_->cells_begin(); it != dof_handler_->cells_end();
             ++it) {
            active_dofs = it->dofs();
            invJT = it->invJ().transpose();
            for (int i = 0; i < BasisType::n_basis; ++i) {
                for (int j = 0; j < i + 1; ++j) {
                    std::pair<const int&, const int&> minmax(std::minmax(active_dofs[i], active_dofs[j]));
                    double value = 0;
                    for (int k = 0; k < n_quadrature_nodes; ++k) {
                        value +=
                          Quadrature::weights[k] * (invJT * grad_table_[i].col(k)).dot(invJT * grad_table_[j].col(k)); // to be optimized, invJT * grad_table_ should be computed outside loop (immediately after storing invJT)
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
template <int LocalDim> struct fe_assembler_packet {
  // global quad node index

  // store pointer to cell?
    double cell_measure;                       // measure of active cell
    double cell_diameter;                      // diameter of active cell
    double trial_value, test_value;            // \psi_i(q_k), \psi_j(q_k) at active quadrature node
  cexpr::Vector<double, LocalDim> trial_grad, test_grad;   // \nabla{\psi_i}(q_k), \nabla{\psi_j}(q_k) at active quadrature node
};
  
// finds node in expression tree satisfying boolean predicate F
template <typename F, typename Xpr> class xpr_find_impl {
    static constexpr auto type_ = []() {
        if constexpr (requires(Xpr xpr) {   // binary node
                          typename Xpr::LhsDerived;
                          typename Xpr::RhsDerived;
                      }) {
            // recurse left branch subtree
            if constexpr (xpr_find_impl<F, typename Xpr::LhsDerived>::type::value) {
                return std::type_identity<std::true_type> {};
            } else {   // recurse right branch subtree
                if constexpr (xpr_find_impl<F, typename Xpr::RhsDerived>::type::value) {
                    return std::type_identity<std::true_type> {};
                } else {
                    return std::type_identity<std::false_type> {};   // no node satisfies F in subtree rooted at Xpr
                }
            }
        } else {
            if constexpr (requires(Xpr xpr) {
                              typename Xpr::Derived;
                          }) {   // Xpr is an internal, not leaf, node. recurse down
                return typename xpr_find_impl<F, typename Xpr::Derived>::type {};
            } else {
                // not binary nor unary node, this is a leaf, then, either derives F or not
                if constexpr (F().template operator()<Xpr>()) {
                    return std::type_identity<std::true_type> {};
                } else {
                    return std::type_identity<std::false_type> {};
                }
            }
        }
    };
   public:
    using type = typename decltype(type_())::type;
};
template <typename F, typename Xpr> struct xpr_find : xpr_find_impl<F, Xpr>::type { };
template <typename F, typename Xpr> static constexpr bool xpr_find_v = xpr_find<F, Xpr>::value;

template <typename F, typename Xpr> constexpr auto& find_fe_space(Xpr xpr) {
    if constexpr (F().template operator()<Xpr>()) {
        return xpr.fe_space();
    } else {
        if constexpr (requires(Xpr xpr) {   // binary node
                          typename Xpr::LhsDerived;
                          typename Xpr::RhsDerived;
                      }) {
            if constexpr (xpr_find_v<F, typename Xpr::LhsDerived>) return find_fe_space<F>(xpr.lhs());
            if constexpr (xpr_find_v<F, typename Xpr::RhsDerived>) return find_fe_space<F>(xpr.rhs());
        } else {
            if constexpr (xpr_find_v<F, typename Xpr::Derived>) return find_fe_space<F>(xpr.derived());
        }
    }
}
template <typename Xpr> constexpr auto& trial_space(Xpr xpr) {
    auto TrialSpace = []<typename Xpr_>() { return requires { typename Xpr_::TrialSpace; }; };
    fdapde_static_assert(xpr_find<decltype(TrialSpace) FDAPDE_COMMA Xpr>::value, NO_TRIAL_SPACE_FOUND_IN_EXPRESSION);
    return find_fe_space<decltype(TrialSpace)>(xpr);
}
template <typename Xpr> constexpr auto& test_space(Xpr xpr) {
    auto TestSpace = []<typename Xpr_>() { return requires { typename Xpr_::TestSpace; }; };
    fdapde_static_assert(xpr_find<decltype(TestSpace) FDAPDE_COMMA Xpr>::value, NO_TEST_SPACE_FOUND_IN_EXPRESSION);
    return find_fe_space<decltype(TestSpace)>(xpr);
}

template <typename Triangulation_, typename Form_, typename... Quadrature_> class fe_assembler_galerkin_loop {
    using TrialSpace = std::decay_t<decltype(trial_space(std::declval<Form_>()))>;
    using TestSpace  = std::decay_t<decltype(test_space (std::declval<Form_>()))>;
    fdapde_static_assert(
      std::is_same_v<TrialSpace FDAPDE_COMMA TestSpace>,
      GALERKIN_ASSEMBLY_LOOP_REQUIRES_TRIAL_AND_TEST_SPACES_TO_BE_EQUAL);
   public:
    using Form = typename std::decay_t<Form_>;
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
    DofHandler<local_dim, embed_dim>* dof_handler_;
    typename Triangulation::cell_iterator begin_, end_;
   public:
    fe_assembler_galerkin_loop() = default;
    fe_assembler_galerkin_loop(
      const Form_& form, typename Triangulation_::cell_iterator begin, typename Triangulation_::cell_iterator end,
      const Quadrature_&... quadrature) requires(sizeof...(quadrature) == 1) :
        form_(form),
        quadrature_(std::get<0>(std::make_tuple(quadrature...))),
        dof_handler_(&trial_space(form).dof_handler()),
        begin_(begin),
        end_(end) { }
    fe_assembler_galerkin_loop(
      const Form_& form, Triangulation::cell_iterator begin, Triangulation::cell_iterator end) :
        form_(form), quadrature_(Quadrature {}), dof_handler_(&trial_space(form).dof_handler()), begin_(begin),
        end_(end) { }

    SpMatrix<double> run() const {
        if (!dof_handler_) dof_handler_->enumerate(FeType {});
        // move Triangulation::cell_iterator to DofHandler::cell_iterator
        using iterator = typename DofHandler<local_dim, embed_dim>::cell_iterator;
        iterator begin(begin_.index(), dof_handler_);
        iterator end(end_.index(), dof_handler_);

	// prepare assembly loop
        SpMatrix<double> assembled_mat(dof_handler_->n_dofs(), dof_handler_->n_dofs());
        std::vector<Eigen::Triplet<double>> triplet_list;
        DVector<int> active_dofs;
        cexpr::Matrix<double, local_dim, embed_dim> invJT;
	// gradient of shape functions mapped on physical cell
        std::array<cexpr::Matrix<double, local_dim, n_quadrature_nodes>, n_basis> shape_grad;
        internals::fe_assembler_packet<local_dim> fe_packet;
        for (iterator it = begin; it != end; ++it) {
            fe_packet.cell_measure = it->measure();

            // to be done

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
    FeCellAssembler(const Triangulation& triangulation, const Quadrature&... quadrature) :
        begin_(triangulation.cells_begin()),
        end_(triangulation.cells_end()),
        quadrature_(std::make_tuple(quadrature...)) { }
    FeCellAssembler(
      const Triangulation::cell_iterator& begin, const Triangulation::cell_iterator& end,
      const Quadrature&... quadrature) :
        begin_(begin), end_(end), quadrature_(std::make_tuple(quadrature...)) { }

    template <typename Form>
    internals::fe_assembler_galerkin_loop<Triangulation, Form, Quadrature...> operator()(const Form& form) const {
        if constexpr (sizeof...(Quadrature) == 0) {
            return {form, begin_, end_};
        } else {
            return {form, begin_, end_, std::get<0>(quadrature_)};
        }
    }
    // template <typename WeakForm>
    // internals::fe_assembler_petrov_galerkin_loop operator()(WeakForm&& weak_form) const
    //     requires(!std::is_same_v<typename WeakForm::TrialSpace, typename WeakForm::TestSpace>) {
    //     return internals::fe_assembler_petrov_galerkin_loop(weak_form, begin_, end_, quadrature_);
    // }
};

template <typename Triangulation, typename... Quadrature>
FeCellAssembler<Triangulation, Quadrature...>
integrate(const Triangulation& triangulation, const Quadrature&... quadrature) {
    return FeCellAssembler<Triangulation, Quadrature...>(triangulation, quadrature...);
}

// template <typename TrialSpaceType, typename TestSpaceType> class FeAssembler {
//     fdapde_static_assert(
//       std::is_same_v<typename TrialSpaceType::Triangulation FDAPDE_COMMA typename TestSpaceType::Triangulation>,
//       TRIAL_AND_TEST_SPACES_MUST_BE_DEFINED_ON_THE_SAME_TRIANGULATION_TYPE);
//     using Triangulation = typename TrialSpaceType::Triangulation;
//     static constexpr int local_dim = Triangulation::local_dim;
//     static constexpr int embed_dim = Triangulation::embed_dim;

//     auto eval_basis_at_quad_nodes =
//       []<typename BasisType, typename Quadrature>(BasisType basis, Quadrature quadrature) {
//           int n_quadrature_nodes = Quadrature::n_nodes;
//           int n_basis = BasisType::n_basis;
//           cexpr::Matrix<double, n_nodes, n_basis> eval_;
//           for (int i = 0; i < n_nodes; ++i) {
//               for (int j = 0; j < n_basis; ++j) { eval_(i, j) = basis.eval(j, quadrature.nodes[i]); }
//           }
//           return eval_;
//       };
//     auto eval_basis_grad_at_quad_nodes =
//       []<typename BasisType, typename Quadrature>(BasisType basis, Quadrature quadrature) {
//           constexpr int n_quadrature_nodes = Quadrature::n_nodes;
//           constexpr int n_basis = BasisType::n_basis;
//           std::array<cexpr::Matrix<double, local_dim, n_basis>, n_quadrature_nodes> eval_;
//           for (int i = 0; i < n_quadrature_nodes; ++i) {
//               for (int j = 0; j < n_basis; ++j) { eval_[i].col(j) = basis.eval_grad(j, quadrature.nodes[i]); }
//           }
//           return eval_;
//       };

//     double cell_measure;
//     double cell_diameter;
//     std::vector<SMatrix<LocalDim, BasisSize>> basis_phys_grad;

//     template <int EmbedDim> void update_grad(const SMatrixL<localDim, EmbedDim>& invJ) {
//         int n_quadrature_nodes = basis_phys_grad.size();
//         for (int i = 0; i < n_quadrature_nodes; ++i) {
//             for (int j = 0; j < n_basis; ++j) { basis_phys_grad[i].col(j) = invJ.transpose() * basis_grad[i].col(j); }
//         }
//     }
  
//     template <typename WeakForm, typename Quadrature>
//     SpMatrix<double> galerkin_assembly_loop(WeakForm&& weak_form, const Quadrature& quadrature) const {
//         fdapde_static_assert(
//           std::is_same_v<typename WeakForm::TrialSpace FDAPDE_COMMA typeanem WeakForm::TestSpace>,
//           GALERKIN_ASSEMBLY_LOOP_REQUIRES_TRIAL_AND_TEST_SPACES_TO_BE_EQUAL);
//         using FunctionalSpace = typename WeakForm::TrialSpace;
//         using DofHandler = typename FunctionalSpace::DofHandler;
//         const FunctionalSpace& functional_space = weak_form.trial_space();
//         // evaluate basis functions at quadrature nodes
//         static constexpr auto basis_vals = eval_basis_at_quad_nodes(functional_space, quadrature);
//         static constexpr auto basis_grad = eval_basis_grad_at_quad_nodes(functional_space, quadrature);

//         // initialize dof_handler
//         const DofHandler& dof_handler = functional_space.dof_handler();
//         if (!dof_handler) dof_handler.enumerate_dofs();

//         SpMatrix<double> assembled_mat(dof_handler.n_dofs(), dof_handler.n_dofs());
//         std::vector<Eigen::Triplet<double>> triplet_list;

//         // subscribe trial and test functions to assembly loop
//         weak_form.subscribe(*this);

//         for (typename DofHandler::cell_iterator cell = dof_handler.cells_begin(); cell != dof_handler.cells_end();
//              ++cell) {
//             // update informations relative to this element, depending on weak form flags
//             cell_measure = cell->measure();
//             if constexpr (WeakForm::Flags & UpdateCellDiameter) cell_diameter = cell->diameter();
//             if constexpr (WeakForm::Flags & UpdateGrad) {
//                 for (int i = 0; i < n_quadrature_nodes; ++i) {
//                     for (int j = 0; j < n_basis; ++j) {
//                         basis_phys_grad[i].col(j) = invJ.transpose() * basis_grad[i].col(j);
//                     }
//                 }
//             }
//             // phys_laplacian = ...;

//             DVector<int> active_dofs = cell->dofs();
//             int n_active_dofs = active_dofs.size();
//             for (int i = 0; i < n_active_dofs; ++i) {
//                 for (int j = 0; j < n_active_dofs; ++j) {
//                     // perform integration of weak form for (i, j)-th basis pair
//                     triplet_list.emplace_back(active_dofs[i], active_dofs[j], weak_form.integrate(i, j) * cell_measure);
//                 }
//             }
//         }
//         // linearity of the integral is implicitly used during matrix construction, as duplicated triplets are
//         // summed up, see Eigen docs for more details
//         assembled_mat.setFromTriplets(triplet_list.begin(), triplet_list.end());
//         assembled_mat.makeCompressed();
//         return assembled_mat;
//     }

//     template <typename WeakForm, typename Quadrature>
//     SpMatrix<double> petrov_galerkin_assembly_loop(WeakForm&& weak_form, const Quadrature& quadrature) const {
//         fdapde_static_assert(
//           !std::is_same_v<typename WeakForm::TrialSpace FDAPDE_COMMA typeanem WeakForm::TestSpace>,
//           TRIAL_AND_TEST_SPACES_ARE_EQUAL_BUT_YOU_REQUIRED_A_PETROV_GALERKIN_ASSEMBLY_USE_GALERKIN_INSTEAD);
//         // set up trial space
//         using TrialSpace = typename WeakForm::TrialSpace;
//         using TrialDofHandler = typename TrialSpace::DofHandler;
//         const TrialSpace& trial_space = weak_form.trial_space();
//         const TrialDofHandler& trial_dof_handler = trial_space.dof_handler();
//         if (!trial_dof_handler) trial_dof_handler.enumerate_dofs();
//         // evaluate basis functions at quadrature nodes
//         static constexpr auto trial_basis_vals = eval_basis_at_quad_nodes(trial_space, quadrature);
//         static constexpr auto trial_basis_grad = eval_basis_grad_at_quad_nodes(trial_space, quadrature);

//         // set up test space
//         using TestSpace = typename WeakForm::TestSpace;
//         using TestDofHandler = typename TestSpace::DofHandler;
//         const TestSpace& test_space = weak_form.test_space();
//         const TestDofHandler& test_dof_handler = test_space.dof_handler();
//         if (!test_dof_handler) test_dof_handler.enumerate_dofs();
//         // evaluate basis functions at quadrature nodes
//         static constexpr auto test_basis_vals = eval_basis_at_quad_nodes(test_space, quadrature);
//         static constexpr auto test_basis_grad = eval_basis_grad_at_quad_nodes(test_space, quadrature);

//         SpMatrix<double> assembled_mat(test_dof_handler.n_dofs(), trial_dof_handler.n_dofs());
//         std::vector<Eigen::Triplet<double>> triplet_list;

//         // subscribe trial and test functions to assembly loop
//         weak_form.subscribe(*this);

//         for (typename TrialDofHandler::cell_iterator cell = trial_dof_handler.cells_begin();
//              cell != trial_dof_handler.cells_end(); ++cell) {
//             // update informations relative to this element, depending on weak form flags
//             cell_measure_ = cell->measure();
//             if constexpr (WeakForm::Flags & UpdateCellDiameter) cell_diameter = cell->diameter();
//             if constexpr (WeakForm::Flags & UpdateGrad)
//                 phys_grad = cell->invJ().transpose() *
//                             basis_eval;   // think how to do this... this should be done differently for trial and test
//             // phys_laplacian = ...;

//             DVector<int> trial_active_dofs = cell->dofs();
//             int n_trial_active_dofs =
//               trial_active_dofs.size();   // this can be computed statically in advance out of this loop
//             for (int i = 0; i < n_trial_active_dofs; ++i) {
//                 for (int j = 0; j < n_test_active_dofs; ++j) {
//                     // perform integration of weak form for (i, j)-th basis pair
//                     triplet_list.emplace_back(
//                       test_active_dofs[j], trial_active_dofs[i], weak_form.integrate(i, j) * cell_measure);
//                 }
//             }
//         }
//         // linearity of the integral is implicitly used during matrix construction, as duplicated triplets are
//         // summed up, see Eigen docs for more details
//         assembled_mat.setFromTriplets(triplet_list.begin(), triplet_list.end());
//         assembled_mat.makeCompressed();
//         return assembled_mat;
//     }
// public:
//     FeCellAssembler() = default;

//     // this requires quadrature rule... must be done inside call operator
//     template <typename WeakForm, typename Quadrature>
//     SpMatrix<double> operator()(WeakForm&& weak_form, const Quadrature& quadrature) const {
//         // need to take Trial and Test spaces
//         using TrialSpace = typename WeakForm::TrialSpace;
//         using TestSpace = typename WeakForm::TestSpace;
//         static constexpr bool is_petrov_galerkin = !std::is_same_v<TrialSpace, TestSpace>;

//         if constexpr (is_petrov_galerkin) {
//             return petrov_galerkin_assembly_loop().run(weak_form, quadrature);
//         } else {
//             return galerkin_assembly_loop().run(weak_form, quadrature);
//         }
//     }
// };

//   // TrialFunction<FunctionalSpace> v(Vh); // only knows about fe_assembler_packet<FunctionalSpace>

//   // WeakForm MUST know about Trial and Test space, then can store pointers to fe_assembler_data<TrialSpace>, fe_assembler_data<TestSpace>
  
//   // auto assembler = integrate(begin, end, quadrature); -> FeCellAssembler<LocalDim, EmbedDim, Quadrature>; // geometry + quadrature
//   // auto L = assembler(weak_form); -> FeCellGalerkinAssembler<LocalDim, EmbedDim, Quadrature, WeakForm>     // geometry + quadrature + functional space info (we are ready)
//   // L.compute(); <- this computes the discretization matrix (this is not supposed to be used outside)

//   // another story is the FeFaceAssembler for border integration

//   // Quadrature might be Dynamic, to say, ok deduce you on the base of the space (higher space order)
//   auto a = integrate(square, Quad::P1)(u * v);

//   // Trial and Test still takes some pointer to fe_data<TheirSpace> (ok, we need to sacrify the information about the number of quadrature points, but..... this is the price to pay)
//   auto a = integrate(square)(u * v);

  
  // BilinearForm

  // before was template <typename FeSpace>
  // but FeSpace is known only once we know the weak form......
  //from FeSpace we can get Triangulation and DofHandler (indeed a FeSpace is a function description + triangulation + dofs)
  // FeSpace cannot stay here, first we might have different FeSpaces.... second, FeSpace is something we know once we get
  // the weak form
  
  
      // but if we set the DofHandler in the FeSpace, we can have different DofHandlers... suppose we have a P1 element for
    // the trial function and P2 for the test, then discretization matrix is NOT square, and we have to integrate
    // \psi_{ij} for i = 1, ..., n j = 1, ..., m, with n != m
    // therefore we have two enumerations of dofs over the triangulation...

    // if we loop with

    // for(dof_cell e : dof_handler) {
    //     we need to support on e different enumerations, because then we must do
    //     for(int i : local_dofs_for_trial_function) {
    //         for(int j : local_dofs_for_test_function) {
    //             triplet_list.emplace_back(i, j, integral involving i-th basis of trial space and j-th basis of test space);
    //         }
    //     }
    // }

    // its like, ok you cycle over the mesh geometry, then query the two dof handlers for getting the local to global dof conversion
    // you can loop over one of the dof handlers, and query for the inner

    // for (dof_cell e : test_dof_handler) {
    //     for (int i : test_local_dofs) {
    //         for (int j : trial_local_dofs.at_element(e.id())) {
    //             triplet_list.emplace_back(
    //               i, j, integral involving i - th basis of trial space and j - th basis of test space);
    //         }
    //     }
    // }

    // by doing so we can support Trial and Test functions defined on different spaces, what kind of optimizations can we do?
    // if trial and test functions are defined on the same space, we need just one DofHandler, there is no point in paying
    // twice the price of initialization
    // who must initialize the DofHandler? I think the FeAssembler on initialization (first time it is queried), since
    // only the FeAssembler has informations about the Trial space and the Test space (they do not know each other in
    // isolation). The FeAssembler can then check for the types of elements, and perform the proper initializations

    // also, if Trial and Test spaces are the same, above loop can be written as

    // for (dof_cell e : test_dof_handler) {
    //     for (int i : test_local_dofs) {
    //         for (int j : test_local_dofs) {
    //             triplet_list.emplace_back(
    //               i, j, integral involving i - th basis of trial space and j - th basis of test space);
    //         }
    //     }
    // }

    // decided to drop symmetric optimization (cannot easily infer it from the weak form in general)


  // given iterators over the triangulation, we can cast them to iterators for the dof_handler (we just need the indices of the cells to start the iterators)
  // so that this construction work
  // template <typename Triangulation> FeAssembler<Triangulation::LocalDim, Triangulation::EmbedDim> integrate(begin, end);

  // moreover, if we have boundary_iterators, we know that the integration is on the boundary of the domain (this most likely will call to a specialization of the
  // assembly loop)
  
  // we cannot put Quadrature as a template parameter of FeAssembler, indeed Trial and Test functions need a pointer
  // to the FeAssembler (to get access to FeAssembler computed quantities, which are computed here for optimization
  // purposes (quantites which can be reused are comptued just once)). Trial and Test can infer the type of the
  // FeAssember (as FeAssembler<LocalDim, EmbedDim>) but NOT the type of the quadrature.

  // therefore we can do
  // auto L = integrate(begin, end)(weak_form); // quadrature inferred from the order of finite elements? Quadrature becomes a responsibility of the FeP which
  // must provide the quadrature order which is exact for its Order

  // if you want to select a quadrature, do
  // auto L = integrate(begin, end)(weak_form, quadrature);
  // and this is perfectly fine

  // auto L = integrate(mesh.boundary_begin(), mesh.boundary_end())(weak_form); // integration over the boundary

  // can we sum up bilinear forms?? it would be usefull

  // Trial and Test functions
  // depending on the FeSpace, they are either derived of ScalarBase or MatrixBase
  // u -> expose a call operator operator()(int i) which returns the value of the basis function at i-th quadrature node
  // v -> same
  // the expression template mechanism should work transparently, even with int call operator
  
  // the difference between Trial and Test function is that Trial has a corresponding vector of coefficients of the FE
  // basis expansion of u (since is the solution to the differential problem), while v no

  // given this, we can work with u and v and compose expressions

  // about grad(u), grad(v), we need to redefine differential quantites, since Trial and Test must keep these quantities
  // from the FeAssembler
  // for instance grad(u) exposes an operator()(int i) -> which is something like fe_assembler->basis_function_cell_grad(i);
  // given this, it should work transparently,

  // otherwise, we could define a PartialDerivative<..., 1> and PartialDerivative<..., 2> for Trial and Test functions? in this way, all the differential
  // operators still work transparently (we can check this). What we have to do is to modify the differential operator
  // to take the definition of the PartialDerivative from the Derived type

  // to be done in the afeternoon, and see what is the generated assembly code

        // sure we have to notice that if they are defined on the same FeSpace, some optimizations can be put in place

        // second derivatives

        // we need to analyze the weak form to understand what kind of updates we have to do for each cell
        // compute measure
        // compute (J^-1)^T
        // compute diameter
        // to do this, we use a bit flag mechanism, in particular, the weak form provides an int where each bit has its
        // meaning, and we check here if we have to compute something or not neverthless, FeAssembler must be coded in
        // advantage to handle all possible requests

        // is the weak form symmetric? if yes, just integrate half of the coefficients... actually, for the symmetric
        // components, we can always integrate just half of the coefficients

        // we can start assembly the discretization matrix looping on the DofHandler
        // for(loop on dof_aware cell) {
        //     update informations relative to cell
        //     evaluate integral
        // }

}   // namespace fdapde

#endif   // __FE_ASSEMBLER_H__
