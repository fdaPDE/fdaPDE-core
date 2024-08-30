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
#include "fe_integration.h"

namespace fdapde {

template <typename Derived_> struct FeMap;
  
enum bilinear_bits {
    compute_shape_values        = 0x0001,
    compute_shape_grad          = 0x0002,
    compute_second_derivatives  = 0x0004,
    compute_physical_quad_nodes = 0x0010
};

[[maybe_unused]] static constexpr int CellMajor = 0;
[[maybe_unused]] static constexpr int FaceMajor = 1;
  
namespace internals {

// detect trial space from bilinear form
template <typename Xpr> constexpr decltype(auto) trial_space(Xpr&& xpr) {
    constexpr bool found = meta::xpr_find<
      decltype([]<typename Xpr_>() { return requires { typename Xpr_::TrialSpace; }; }), std::decay_t<Xpr>>();
    fdapde_static_assert(found, NO_TRIAL_SPACE_FOUND_IN_EXPRESSION);
    return meta::xpr_query<
      decltype([]<typename Xpr_>(Xpr_&& xpr) -> auto& { return xpr.fe_space(); }),
      decltype([]<typename Xpr_>() { return requires { typename Xpr_::TrialSpace; }; })>(std::forward<Xpr>(xpr));
}
// detect test space from bilinear form
template <typename Xpr> constexpr decltype(auto) test_space(Xpr&& xpr) {
    constexpr bool found = meta::xpr_find<
      decltype([]<typename Xpr_>() { return requires { typename Xpr_::TestSpace; }; }), std::decay_t<Xpr>>();
    fdapde_static_assert(found, NO_TEST_SPACE_FOUND_IN_EXPRESSION);
    return meta::xpr_query<
      decltype([]<typename Xpr_>(Xpr_&& xpr) -> auto& { return xpr.fe_space(); }),
      decltype([]<typename Xpr_>() { return requires { typename Xpr_::TestSpace; }; })>(std::forward<Xpr>(xpr));
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
template <typename DofHandler, typename FeType> class fe_mass_assembly_loop {
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
    fe_mass_assembly_loop() = default;
    fe_mass_assembly_loop(DofHandler& dof_handler) : dof_handler_(&dof_handler) { }

    SpMatrix<double> assemble() {
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
    constexpr int n_dofs() const { return dof_handler_->n_dofs(); }
};

// optimized computation of discretized Laplace [A]_{ij} = \int_D (\grad{\psi_i} * \grad{\psi_j})
template <typename DofHandler, typename FeType> class fe_grad_grad_assembly_loop {
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
    fe_grad_grad_assembly_loop() = default;
    fe_grad_grad_assembly_loop(DofHandler& dof_handler) : dof_handler_(&dof_handler) { }

    SpMatrix<double> assemble() {
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
template <typename Derived> struct fe_assembly_xpr_base;

// traits for surface integration \int_{\partial D} (...)
template <typename FunctionSpace_, typename... Quadrature_> class fe_face_assembler_traits {
    using FeType = typename FunctionSpace_::FeType;
    static constexpr int local_dim = FunctionSpace_::local_dim;
    static constexpr int embed_dim = FunctionSpace_::embed_dim;
   public:
    using dof_descriptor = FeType::template face_dof_descriptor<local_dim>;
    using Quadrature = decltype([]() {
        if constexpr (sizeof...(Quadrature_) == 0) {
            return typename FeType::template select_face_quadrature_t<local_dim> {};
        } else {
            return std::get<0>(std::tuple<Quadrature_...>());
        }
    }());
    using geo_iterator = typename Triangulation<local_dim, embed_dim>::boundary_iterator;
    using dof_iterator = typename DofHandler<local_dim, embed_dim>::boundary_iterator;
};

// traits for integration \int_D (...)
template <typename FunctionSpace_, typename... Quadrature_> class fe_cell_assembler_traits {
    using FeType = typename FunctionSpace_::FeType;
    static constexpr int local_dim = FunctionSpace_::local_dim;
    static constexpr int embed_dim = FunctionSpace_::embed_dim;
   public:
    using dof_descriptor = FeType::template cell_dof_descriptor<local_dim>;
    using Quadrature = decltype([]() {
        if constexpr (sizeof...(Quadrature_) == 0) {
            return typename FeType::template select_cell_quadrature_t<local_dim> {};
        } else {
            return std::get<0>(std::tuple<Quadrature_...>());
        }
    }());
    fdapde_static_assert(
      internals::is_fe_quadrature_simplex<Quadrature>, SUPPLIED_QUADRATURE_FORMULA_IS_NOT_FOR_SIMPLEX_INTEGRATION);
    using geo_iterator = typename Triangulation<local_dim, embed_dim>::cell_iterator;
    using dof_iterator = typename DofHandler<local_dim, embed_dim>::cell_iterator;
};

// base class for finite element assembly loops (of galerkin type, e.g. trial space == test space)
template <typename Triangulation_, typename Form_, int Options_, typename... Quadrature_>
struct fe_assembler_base {
    // detect test space
    using TestSpace  = std::decay_t<decltype(test_space (std::declval<Form_>()))>;
    using Form = std::decay_t<decltype(meta::xpr_wrap<FeMap, decltype([]<typename Xpr>() {
          return !std::is_invocable_v<
            Xpr, fe_assembler_packet<Xpr::StaticInputSize>>;
        })>(std::declval<Form_>()))>;
    using Triangulation = typename std::decay_t<Triangulation_>;
    static constexpr int local_dim = Triangulation::local_dim;
    static constexpr int embed_dim = Triangulation::embed_dim;
    static constexpr int Options = Options_;
    using FunctionSpace = TestSpace;
    using FeType = typename FunctionSpace::FeType;
    using fe_traits = std::conditional_t<
      Options == CellMajor, fe_cell_assembler_traits<FunctionSpace, Quadrature_...>,
      fe_face_assembler_traits<FunctionSpace, Quadrature_...>>;
    using Quadrature = typename fe_traits::Quadrature;
    using dof_descriptor = typename fe_traits::dof_descriptor;
    using BasisType = typename dof_descriptor::BasisType;
    static constexpr int n_quadrature_nodes = Quadrature::n_nodes;
    static constexpr int n_basis = BasisType::n_basis;
  
    fe_assembler_base() = default;
    fe_assembler_base(
      const Form_& form, typename fe_traits::geo_iterator begin, typename fe_traits::geo_iterator end,
      const Quadrature_&... quadrature) requires(sizeof...(quadrature) <= 1) : 
        form_(meta::xpr_wrap<FeMap, decltype([]<typename Xpr>() {
                                 return !std::is_invocable_v<Xpr, fe_assembler_packet<Xpr::StaticInputSize>>;
                             })>(form)),
        quadrature_([... quadrature = std::forward<const Quadrature_>(quadrature)]() {
            if constexpr (sizeof...(quadrature) == 1) {
                return std::get<0>(std::make_tuple(quadrature...));
            } else {
                return Quadrature {};
            }
        }()),
        dof_handler_(&test_space(form_).dof_handler()),
        begin_(begin),
        end_(end) {
        fdapde_assert(dof_handler_->n_dofs() > 0);
    }
    constexpr int n_dofs() const { return dof_handler_->n_dofs(); }
   protected:
    // compile-time evaluation of \psi_i(q_j), i = 1, ..., n_basis, j = 1, ..., n_quadrature_nodes
    static constexpr cexpr::Matrix<double, n_basis, n_quadrature_nodes> shape_values_ {[]() {
        std::array<double, n_basis * n_quadrature_nodes> shape_values_ {};
        BasisType basis {dof_descriptor().dofs_phys_coords()};
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
        BasisType basis {dof_descriptor().dofs_phys_coords()};
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

    void distribute_quadrature_nodes(
      std::unordered_map<const void*, DMatrix<double>>& fe_map_buff, typename fe_traits::dof_iterator begin,
      typename fe_traits::dof_iterator end) const {
        DMatrix<double> quad_nodes;
        quad_nodes.resize(n_quadrature_nodes * (end_.index() - begin_.index()), embed_dim);
        int local_cell_id = 0;
        for (typename fe_traits::geo_iterator it = begin_; it != end_; ++it) {
            for (int q_k = 0; q_k < n_quadrature_nodes; ++q_k) {
                quad_nodes.row(local_cell_id * n_quadrature_nodes + q_k) =
                  it->J() *
                    Eigen::Map<const SMatrix<n_quadrature_nodes, Quadrature::local_dim>>(quadrature_.nodes.data())
                      .row(q_k)
                      .transpose() +
                  it->node(0);
            }
            local_cell_id++;
        }

        // evaluate FeMap nodes at quadrature nodes
        meta::xpr_apply_if<
          decltype([]<typename Xpr_, typename... Args>(Xpr_& xpr, Args&&... args) {
              xpr.init(std::forward<Args>(args)...);
              return;
          }),
          decltype([]<typename Xpr_>() {
              return requires(Xpr_ xpr) { xpr.init(fe_map_buff, quad_nodes, begin, end); };
          })>(form_, fe_map_buff, quad_nodes, begin, end);
    }
    Form form_;
    Quadrature quadrature_ {};
    const DofHandler<local_dim, embed_dim>* dof_handler_;
    typename fe_traits::geo_iterator begin_, end_;
};

template <typename Triangulation_, typename Form_, int Options_, typename... Quadrature_>
class fe_matrix_assembly_loop :
    public fe_assembler_base<Triangulation_, Form_, Options_, Quadrature_...>,
    public fe_assembly_xpr_base<fe_matrix_assembly_loop<Triangulation_, Form_, Options_, Quadrature_...>> {
    // detect trial and test spaces from bilinear form
    using TrialSpace = std::decay_t<decltype(trial_space(std::declval<Form_>()))>;
    using TestSpace  = std::decay_t<decltype(test_space (std::declval<Form_>()))>;
    fdapde_static_assert(
      std::is_same_v<TrialSpace FDAPDE_COMMA TestSpace>,
      GALERKIN_ASSEMBLY_LOOP_REQUIRES_TRIAL_AND_TEST_SPACES_TO_BE_EQUAL);
    using Base = fe_assembler_base<Triangulation_, Form_, Options_, Quadrature_...>;
    using Form = typename Base::Form;
    static constexpr int local_dim = Base::local_dim;
    static constexpr int embed_dim = Base::embed_dim;
    static constexpr int n_basis = Base::n_basis;
    static constexpr int n_quadrature_nodes = Base::n_quadrature_nodes;
    using Base::dof_handler_;
    using Base::form_;
   public:
    fe_matrix_assembly_loop() = default;
    fe_matrix_assembly_loop(
      const Form_& form, typename Base::fe_traits::geo_iterator begin, typename Base::fe_traits::geo_iterator end,
      const Quadrature_&... quadrature) requires(sizeof...(quadrature) <= 1)
        : Base(form, begin, end, quadrature...) { }

    SpMatrix<double> assemble() const {
        SpMatrix<double> assembled_mat(dof_handler_->n_dofs(), dof_handler_->n_dofs());
        std::vector<Eigen::Triplet<double>> triplet_list;
	assemble(triplet_list);
	// linearity of the integral is implicitly used here, as duplicated triplets are summed up (see Eigen docs)
        assembled_mat.setFromTriplets(triplet_list.begin(), triplet_list.end());
        assembled_mat.makeCompressed();
        return assembled_mat;
    }
    void assemble(std::vector<Eigen::Triplet<double>>& triplet_list) const {
        using iterator = typename Base::fe_traits::dof_iterator;
        iterator begin(Base::begin_.index(), dof_handler_, Base::begin_.marker());
        iterator end  (Base::end_.index(),   dof_handler_, Base::end_.marker());
        // prepare assembly loop
        DVector<int> active_dofs;
        std::array<cexpr::Matrix<double, local_dim, n_quadrature_nodes>, n_basis>
          shape_grad;   // gradient of shape functions mapped on physical cell
        std::unordered_map<const void*, DMatrix<double>> fe_map_buff;
        if constexpr (Form::XprBits & bilinear_bits::compute_physical_quad_nodes) {
            Base::distribute_quadrature_nodes(
              fe_map_buff, begin, end);   // distribute quadrature nodes on physical mesh (if required)
        }
	
        // start assembly loop
        internals::fe_assembler_packet<local_dim> fe_packet;
	int local_cell_id = 0;
        for (iterator it = begin; it != end; ++it) {
            fe_packet.cell_measure = it->measure();
            // if constexpr (WeakForm::Flags & UpdateCellDiameter) fe_data.cell_diameter = cell->diameter();
            if constexpr (Form::XprBits & bilinear_bits::compute_shape_grad) {
                for (int i = 0; i < n_basis; ++i) {
                    for (int j = 0; j < n_quadrature_nodes; ++j) {
                        shape_grad[i].col(j) =
                          cexpr::Map<const double, local_dim, embed_dim>(it->invJ().data()).transpose() *
                          Base::shape_grad_[i].col(j);
                    }
                }
            }

            // perform integration of weak form for (i, j)-th basis pair
            active_dofs = it->dofs();
            for (int i = 0; i < n_basis; ++i) {       // trial function loop
                for (int j = 0; j < n_basis; ++j) {   // test function loop
                    double value = 0;
                    for (int q_k = 0; q_k < n_quadrature_nodes; ++q_k) {
                        // update fe_packet
                        fe_packet.trial_value = Base::shape_values_(i, q_k);
                        fe_packet.test_value  = Base::shape_values_(j, q_k);
                        if constexpr (Form::XprBits & bilinear_bits::compute_shape_grad) {
                            fe_packet.trial_grad = shape_grad[i].col(q_k);
                            fe_packet.test_grad  = shape_grad[j].col(q_k);
                        }
                        if constexpr (Form::XprBits & bilinear_bits::compute_physical_quad_nodes) {
                            fe_packet.quad_node_id = local_cell_id * n_quadrature_nodes + q_k;
                        }
                        value += Base::Quadrature::weights[q_k] * form_(fe_packet);
                    }
                    triplet_list.emplace_back(active_dofs[i], active_dofs[j], value * fe_packet.cell_measure);
                }
            }
	    local_cell_id++;
        }
        return;
    }
};

// assembly loop for the discretization of integrals \int_D f * \psi_i, with \psi_i \in test space
template <typename Triangulation_, typename Form_, int Options_, typename... Quadrature_>
class fe_vector_assembly_loop :
    public fe_assembler_base<Triangulation_, Form_, Options_, Quadrature_...>,
    public fe_assembly_xpr_base<fe_vector_assembly_loop<Triangulation_, Form_, Options_, Quadrature_...>> {
    static constexpr bool trial_space_found = meta::xpr_find<
      decltype([]<typename Xpr_>() { return requires { typename Xpr_::TrialSpace; }; }), std::decay_t<Form_>>();
    fdapde_static_assert(!trial_space_found, IF_YOU_WANT_TO_DISCRETIZE_A_BILINEAR_FORM_USE_THE_GALERKIN_ASSEMBLY_LOOP);
    using Base = fe_assembler_base<Triangulation_, Form_, Options_, Quadrature_...>;
    using Form = typename Base::Form;
    static constexpr int local_dim = Base::local_dim;
    static constexpr int embed_dim = Base::embed_dim;
    static constexpr int n_basis = Base::n_basis;
    static constexpr int n_quadrature_nodes = Base::n_quadrature_nodes;
    using Base::dof_handler_;
    using Base::form_;
   public:
    fe_vector_assembly_loop() = default;
    fe_vector_assembly_loop(
      const Form_& form, typename Base::fe_traits::geo_iterator begin, typename Base::fe_traits::geo_iterator end,
      const Quadrature_&... quadrature) requires(sizeof...(quadrature) <= 1)
        : Base(form, begin, end, quadrature...) { }

    DVector<double> assemble() const {
        DVector<double> assembled_vec(dof_handler_->n_dofs());
        assembled_vec.setZero();
        assemble(assembled_vec);
        return assembled_vec;
    }
    void assemble(DVector<double>& assembled_vec) const {
        using iterator = typename Base::fe_traits::dof_iterator;
        iterator begin(Base::begin_.index(), dof_handler_, Base::begin_.marker());
        iterator end  (Base::end_.index(),   dof_handler_, Base::end_.marker());
        // prepare assembly loop
        DVector<int> active_dofs;
        std::unordered_map<const void*, DMatrix<double>> fe_map_buff;   // evaluation of FeMap nodes at quadrature nodes
        if constexpr (Form::XprBits & bilinear_bits::compute_physical_quad_nodes) {
            Base::distribute_quadrature_nodes(
              fe_map_buff, begin, end);   // distribute quadrature nodes on physical mesh (if required)
        }
	
        // start assembly loop
        internals::fe_assembler_packet<local_dim> fe_packet;
	int local_cell_id = 0;
        for (iterator it = begin; it != end; ++it) {
            fe_packet.cell_measure = it->measure();
            active_dofs = it->dofs();
            for (int i = 0; i < n_basis; ++i) {   // test function loop
                double value = 0;
                for (int q_k = 0; q_k < n_quadrature_nodes; ++q_k) {
                    // update fe_packet
                    fe_packet.test_value = Base::shape_values_(i, q_k);
                    if constexpr (Form::XprBits & bilinear_bits::compute_physical_quad_nodes) {
                        fe_packet.quad_node_id = local_cell_id * n_quadrature_nodes + q_k;
                    }
                    value += Base::Quadrature::weights[q_k] * form_(fe_packet);
                }		
                assembled_vec[active_dofs[i]] += value * fe_packet.cell_measure;
            }
	    local_cell_id++;
        }
        return;
    }
};

// arithmetic between matrix assembly loops
template <typename Derived> struct fe_assembly_xpr_base {
    constexpr const Derived& derived() const { return static_cast<const Derived&>(*this); }
};

template <typename Lhs, typename Rhs>
struct fe_assembly_add_op : public fe_assembly_xpr_base<fe_assembly_add_op<Lhs, Rhs>> {
    fdapde_static_assert(
      std::is_same_v<decltype(std::declval<Lhs>().assemble()) FDAPDE_COMMA decltype(std::declval<Rhs>().assemble())>,
      YOU_ARE_SUMMING_NON_COMPATIBLE_ASSEMBLY_LOOPS);
    using OutputType = decltype(std::declval<Lhs>().assemble());
    fe_assembly_add_op(const Lhs& lhs, const Rhs& rhs) : lhs_(lhs), rhs_(rhs) {
        fdapde_assert(lhs.n_dofs() == rhs.n_dofs());
    }
    OutputType assemble() const {
        if constexpr (std::is_same_v<OutputType, SpMatrix<double>>) {
            SpMatrix<double> assembled_mat(lhs_.n_dofs(), lhs_.n_dofs());
            std::vector<Eigen::Triplet<double>> triplet_list;
            assemble(triplet_list);
            // linearity of the integral is implicitly used here, as duplicated triplets are summed up (see Eigen docs)
            assembled_mat.setFromTriplets(triplet_list.begin(), triplet_list.end());
            assembled_mat.makeCompressed();
            return assembled_mat;
        } else {
            DVector<double> assembled_vec(lhs_.n_dofs());
            assembled_vec.setZero();
            assemble(assembled_vec);
            return assembled_vec;
        }
    }
    template <typename T> void assemble(T& assembly_buff) const {
        lhs_.assemble(assembly_buff);
        rhs_.assemble(assembly_buff);
        return;
    }
    constexpr int n_dofs() const { return lhs_.n_dofs(); }
   private:
    Lhs lhs_;
    Rhs rhs_;
};
template <typename Lhs, typename Rhs>
fe_assembly_add_op<Lhs, Rhs> operator+(const fe_assembly_xpr_base<Lhs>& lhs, const fe_assembly_xpr_base<Rhs>& rhs) {
    return fe_assembly_add_op<Lhs, Rhs>(lhs.derived(), rhs.derived());
}

}   // namespace internals

template <typename Triangulation, int Options, typename... Quadrature> class FeGalerkinAssembler {
    fdapde_static_assert(
      sizeof...(Quadrature) < 2, YOU_CAN_PROVIDE_AT_MOST_ONE_QUADRATURE_FORMULA_TO_A_FE_CELL_ASSEMBLER);
    std::tuple<Quadrature...> quadrature_;
    std::conditional_t<
      Options == CellMajor, typename Triangulation::cell_iterator, typename Triangulation::boundary_iterator>
      begin_, end_;
   public:
    FeGalerkinAssembler() = default;
    template <typename Iterator>
    FeGalerkinAssembler(const Iterator& begin, const Iterator& end, const Quadrature&... quadrature) :
        begin_(begin), end_(end), quadrature_(std::make_tuple(quadrature...)) { }

    template <typename Form> auto operator()(const Form& form) const {
        static constexpr bool has_trial_space = meta::xpr_find<
          decltype([]<typename Xpr_>() { return requires { typename Xpr_::TrialSpace; }; }), std::decay_t<Form>>();
        static constexpr bool has_test_space  = meta::xpr_find<
          decltype([]<typename Xpr_>() { return requires { typename Xpr_::TestSpace; };  }), std::decay_t<Form>>();

        if constexpr (has_trial_space && has_test_space) {   // bilinear form discretization
            if constexpr (sizeof...(Quadrature) == 0) {
                return internals::fe_matrix_assembly_loop<Triangulation, Form, Options> {form, begin_, end_};
            } else {
                return internals::fe_matrix_assembly_loop<
                  Triangulation, Form, Options, std::tuple_element_t<0, std::tuple<Quadrature...>>> {
                  form, begin_, end_, std::get<0>(quadrature_)};
            }
        } else {   // functional (forcing-term like) discretization
            if constexpr (sizeof...(Quadrature) == 0) {
                return internals::fe_vector_assembly_loop<Triangulation, Form, Options> {form, begin_, end_};
            } else {
                return internals::fe_vector_assembly_loop<
                  Triangulation, Form, Options, std::tuple_element_t<0, std::tuple<Quadrature...>>> {
                  form, begin_, end_, std::get<0>(quadrature_)};
            }
        }
    }
};

template <typename Triangulation, typename... Quadrature>
using FeCellGalerkinAssembler = FeGalerkinAssembler<Triangulation, CellMajor, Quadrature...>;
template <typename Triangulation, typename... Quadrature>
using FeFaceGalerkinAssembler = FeGalerkinAssembler<Triangulation, FaceMajor, Quadrature...>;

template <typename Triangulation, typename... Quadrature>
auto integral(const Triangulation& triangulation, Quadrature... quadrature) {
    return FeCellGalerkinAssembler<Triangulation, Quadrature...>(
      triangulation.cells_begin(), triangulation.cells_end(), quadrature...);
}
template <typename Triangulation, typename... Quadrature>
auto integral(
  const CellIterator<Triangulation>& begin, const CellIterator<Triangulation>& end, Quadrature... quadrature) {
    return FeCellGalerkinAssembler<Triangulation, Quadrature...>(begin, end, quadrature...);
}
// boundary integration
template <typename Triangulation, typename... Quadrature>
auto integral(
  const BoundaryIterator<Triangulation>& begin, const BoundaryIterator<Triangulation>& end, Quadrature... quadrature) {
    return FeFaceGalerkinAssembler<Triangulation, Quadrature...>(begin, end, quadrature...);
}
template <typename Triangulation, typename... Quadrature>
auto integral(
  const std::pair<BoundaryIterator<Triangulation>, BoundaryIterator<Triangulation>>& range, Quadrature... quadrature) {
    return FeFaceGalerkinAssembler<Triangulation, Quadrature...>(range.first, range.second, quadrature...);
}

}   // namespace fdapde

#endif   // __FE_ASSEMBLER_H__
