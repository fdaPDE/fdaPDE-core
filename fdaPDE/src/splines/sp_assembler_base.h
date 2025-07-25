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

#ifndef __FDAPDE_SP_ASSEMBLER_BASE_H__
#define __FDAPDE_SP_ASSEMBLER_BASE_H__

#include "header_check.h"

namespace fdapde{

template <typename Derived_> struct SpMap;
  
enum class sp_assembler_flags {
    compute_shape_values        = 0x0001,
    compute_shape_dx            = 0x0002,
    compute_shape_dxx           = 0x0004,
    compute_physical_quad_nodes = 0x0010,
    compute_cell_diameter       = 0x0020
};
  
namespace internals {

// informations sent from the assembly loop to the integrated forms
template <int LocalDim> struct sp_assembler_packet {
    static constexpr int local_dim = LocalDim;
    sp_assembler_packet() = default;
    sp_assembler_packet(sp_assembler_packet&&) noexcept = default;
    sp_assembler_packet(const sp_assembler_packet&) noexcept = default;

    // geometric informations
    int quad_node_id;       // active physical quadrature node index
    double cell_measure;    // active cell measure
    double cell_diameter;   // active cell diameter

    // functional informations
    double trial_value, test_value;   // \psi_i(q_k), \psi_j(q_k)
    double trial_dx, test_dx;         // d{\psi_i}/dx(q_k), d{\psi_j}/dx(q_k)
    double trial_dxx, test_dxx;       // d^2{\psi_i}/dx^2(q_k), d^2{\psi_j}/dx^2(q_k)
};

// base class for spline-based assembly loops
template <typename Triangulation_, typename Form_, int Options_, typename... Quadrature_>
struct sp_assembler_base {
    fdapde_static_assert(sizeof...(Quadrature_) < 2, YOU_CAN_SUPPLY_AT_MOST_ONE_QUADRATURE_RULE_TO_A_SP_ASSEMBLY_LOOP);
    // detect test space (since a test function is always present in a weak form)
    using TestSpace = test_space_t<Form_>;
   protected:
    using is_not_packet_evaluable =
      decltype([]<typename Xpr>() { return !(std::is_invocable_v<Xpr, sp_assembler_packet<Xpr::StaticInputSize>>); });
   public:
    using Form = std::decay_t<decltype(xpr_wrap<SpMap, is_not_packet_evaluable>(std::declval<Form_>()))>;
    using Triangulation = typename std::decay_t<Triangulation_>;
    static constexpr int local_dim = Triangulation::local_dim;
    static constexpr int embed_dim = Triangulation::embed_dim;
    static constexpr int Options = Options_;
    using FunctionSpace = TestSpace;
    using DofHandlerType = DofHandler<local_dim, embed_dim, spline_tag>;
    using Quadrature = decltype([]() {
        if constexpr (sizeof...(Quadrature_) == 0) {
            return void();   // quadrature selcted at run-time provided the actual order of spline basis
        } else {
            return std::get<0>(std::tuple<Quadrature_...>());   // user-defined quadrature
        }
    }());
    using geo_iterator = typename Triangulation::cell_iterator;
    using dof_iterator = typename DofHandlerType::cell_iterator;
    using discretization_category = typename TestSpace::discretization_category;
    fdapde_static_assert(
      std::is_same_v<discretization_category FDAPDE_COMMA spline_tag>, THIS_CLASS_IS_FOR_SPLINE_DISCRETIZATION_ONLY);

    sp_assembler_base() = default;
    sp_assembler_base(
      const Form_& form, const geo_iterator& begin, const geo_iterator& end, const Quadrature_&... quadrature)
        requires(sizeof...(quadrature) <= 1)
        :
        form_(xpr_wrap<SpMap, is_not_packet_evaluable>(form)),
        dof_handler_(std::addressof(internals::test_space(form_).dof_handler())),
        test_space_(std::addressof(internals::test_space(form_))),
        begin_(begin),
        end_(end) {
        fdapde_assert(dof_handler_->n_dofs() > 0);
        // copy quadrature rule
	Eigen::Matrix<double, Dynamic, Dynamic> quad_nodes__;
        if constexpr (sizeof...(quadrature) == 1) {
            auto quad_rule = std::get<0>(std::make_tuple(quadrature...));
            fdapde_assert(local_dim == quad_rule.local_dim);
            quad_nodes__ .resize(quad_rule.order, quad_rule.local_dim);
            quad_weights_.resize(quad_rule.order, 1);
            for (int i = 0; i < quad_rule.order; ++i) {
                quad_weights_(i, 0) = quad_rule.weights[i];
                for (int j = 0; j < local_dim; ++j) { quad_nodes__(i, j) = quad_rule.nodes(i, j); }
	    }
        } else {
            internals::get_sp_quadrature(test_space_->order(), quad_nodes__, quad_weights_);
        }
	// build grid of quadrature nodes on reference domain
	n_quadrature_nodes_ = quad_nodes__.rows();
        int n_cells = end_.index() - begin_.index(), n_src_points = quad_nodes__.rows();
        quad_nodes_.resize(n_cells * n_src_points, local_dim);
        int i = 0;
	// as quadrature nodes are defined on [-1, 1], we map them on each triangulation's subinterval
        for (auto it = begin_; it != end_; ++it) {
            double a = it->nodes()[0], b = it->nodes()[1];
            for (int j = 0; j < quad_nodes__.rows(); ++j) {
                quad_nodes_(i, 0) = (quad_nodes__(j, 0) + 1) * (b - a) * 0.5 + a;
                i++;
            }
        }
    }
    const TestSpace& test_space() const { return *test_space_; }
   protected:
    // evaluation of \psi_i(q_j), i = 1, ..., n_basis, j = 1, ..., n_quadrature_nodes
    template <typename BasisType__, typename IteratorType, typename DstMdArray>
    void eval_shape_values(
      BasisType__&& basis, const std::vector<int>& active_dofs, IteratorType cell, DstMdArray& dst) const {
        int n_basis = active_dofs.size();
        for (int i = 0; i < n_basis; ++i) {
            // evaluation of \psi_i at q_j, j = 1, ..., n_quadrature_nodes
            for (int j = 0; j < n_quadrature_nodes_; ++j) {
                dst(i, j) = basis[active_dofs[i]](quad_nodes_.row(cell->id() * n_quadrature_nodes_ + j).transpose());
            }
        }
        return;
    }
    // evaluation of k-th order derivative of basis function
    template <typename BasisType__, typename IteratorType, typename DstMdArray>
        requires(requires(BasisType__ basis, int i, int k) { basis[i].gradient(k); })
    void eval_shape_derivative(
      BasisType__&& basis, const std::vector<int>& active_dofs, IteratorType cell, DstMdArray& dst, int order) const {
        using BasisType = std::decay_t<BasisType__>;
	using DerivativeType = decltype(std::declval<BasisType>()[std::declval<int>()].gradient(std::declval<int>()));
        int n_basis = active_dofs.size();
        for (int i = 0; i < n_basis; ++i) {
            DerivativeType der = basis[active_dofs[i]].gradient(order);
            // evaluation of d^k\psi_i/dx^k at q_j, j = 1, ..., n_quadrature_nodes
            for (int j = 0; j < n_quadrature_nodes_; ++j) {
                dst(i, j) = der(quad_nodes_.row(cell->id() * n_quadrature_nodes_ + j).transpose());
            }
        }
        return;
    }
    template <typename BasisType__, typename IteratorType, typename DstMdArray>
    void
    eval_shape_dx(BasisType__&& basis, const std::vector<int>& active_dofs, IteratorType cell, DstMdArray& dst) const {
        eval_shape_derivative(basis, active_dofs, cell, dst, 1);
    }
    template <typename BasisType__, typename IteratorType, typename DstMdArray>
    void
    eval_shape_dxx(BasisType__&& basis, const std::vector<int>& active_dofs, IteratorType cell, DstMdArray& dst) const {
        eval_shape_derivative(basis, active_dofs, cell, dst, 2);
    }
    void distribute_quadrature_nodes(
      std::unordered_map<const void*, Eigen::Matrix<double, Dynamic, Dynamic>>& sp_map_buff, dof_iterator begin,
      dof_iterator end) {
        Eigen::Matrix<double, Dynamic, Dynamic> quad_nodes;
        quad_nodes.resize(n_quadrature_nodes_ * (end_.index() - begin_.index()), embed_dim);
        int local_cell_id = 0;
        for (geo_iterator it = begin_; it != end_; ++it) {
            for (int q_k = 0; q_k < n_quadrature_nodes_; ++q_k) {
                quad_nodes.row(local_cell_id * n_quadrature_nodes_ + q_k) =
                  it->J() * quad_nodes_.row(q_k).transpose() + it->node(0);
            }
            local_cell_id++;
        }
        // evaluate Map nodes at quadrature nodes
        xpr_apply_if<
          decltype([]<typename Xpr_, typename... Args>(Xpr_& xpr, Args&&... args) {
              xpr.init(std::forward<Args>(args)...);
              return;
          }),
          decltype([]<typename Xpr_>() {
              return requires(Xpr_ xpr) { xpr.init(sp_map_buff, quad_nodes, begin, end); };
          })>(form_, sp_map_buff, quad_nodes, begin, end);
        return;
    }

    Form form_;
    const DofHandlerType* dof_handler_;
    const TestSpace* test_space_;
    geo_iterator begin_, end_;
    // quadrature
    Eigen::Matrix<double, Dynamic, Dynamic> quad_nodes_, quad_weights_;
    int n_quadrature_nodes_;
};

}   // namespace internals
}   // namespace fdapde

#endif   // __FDAPDE_SP_ASSEMBLER_BASE_H__
