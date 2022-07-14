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

#ifndef __FDAPDE_FE_ASSEMBLER_BASE_H__
#define __FDAPDE_FE_ASSEMBLER_BASE_H__

#include "header_check.h"

namespace fdapde {

template <typename Derived_> struct FeMap;

enum class fe_assembler_flags {
    compute_shape_values        = 0x0001,
    compute_shape_grad          = 0x0002,
    compute_shape_hess          = 0x0004,
    compute_shape_div           = 0x0008,
    compute_physical_quad_nodes = 0x0010
};

namespace internals {

// informations sent from the assembly loop to the integrated forms
template <int LocalDim> struct fe_assembler_packet : geo_assembler_packet<LocalDim> {
    static constexpr int local_dim = LocalDim;
    fe_assembler_packet(int n_trial_components, int n_test_components) :
        trial_value(n_trial_components),
        test_value (n_test_components ),
        trial_grad (n_trial_components),
        test_grad  (n_test_components ),
        trial_hess (n_trial_components),
        test_hess  (n_test_components ) { }
    fe_assembler_packet(int n_components) : fe_assembler_packet(n_components, n_components) { }
    fe_assembler_packet() : fe_assembler_packet(1, 1) { }
    fe_assembler_packet(fe_assembler_packet&&) noexcept = default;
    fe_assembler_packet(const fe_assembler_packet&) noexcept = default;

    int quad_node_id;
    // functional informations (Dynamic stands for number of components)
    MdArray<double, MdExtents<Dynamic>> trial_value, test_value;            // \psi_i(q_k), \psi_j(q_k)
    MdArray<double, MdExtents<Dynamic, local_dim>> trial_grad, test_grad;   // \nabla{\psi_i}(q_k), \nabla{\psi_j}(q_k)
    MdArray<double, MdExtents<Dynamic, local_dim, local_dim>> trial_hess, test_hess;
    double trial_div = 0, test_div = 0;
};

// traits for surface integration \int_{\partial D} (...)
template <typename FeSpace_, typename... Quadrature_> struct fe_face_assembler_traits {
    using FeType = typename FeSpace_::FeType;
    static constexpr int local_dim = FeSpace_::local_dim;
    static constexpr int embed_dim = FeSpace_::embed_dim;
    static constexpr int n_components = FeSpace_::n_components;

    using dof_descriptor = FeType::template face_dof_descriptor<local_dim>;
    using BasisType = typename dof_descriptor::BasisType;
    static constexpr int n_basis = BasisType::n_basis;  
    using Quadrature = decltype([]() {
        if constexpr (sizeof...(Quadrature_) == 0) {
            return typename FeType::template face_quadrature_t<local_dim> {};
        } else {
            using UserQuadrature = std::tuple_element_t<0, std::tuple<Quadrature_...>>;
            fdapde_static_assert(
              UserQuadrature::local_dim == (local_dim - 1),
              SUPPLIED_QUADRATURE_DIMENSION_DOES_NOT_MATCH_FACE_DIMENSION);
            return std::get<0>(std::tuple<Quadrature_...>());
        }
    }());
    static constexpr int n_quadrature_nodes = Quadrature::degree;
    fdapde_static_assert(
      internals::is_fe_quadrature_simplex<Quadrature>, SUPPLIED_QUADRATURE_FORMULA_IS_NOT_FOR_SIMPLEX_INTEGRATION);
    using geo_iterator = typename Triangulation<local_dim, embed_dim>::boundary_iterator;
    using dof_iterator = typename DofHandler<local_dim, embed_dim, finite_element_tag>::boundary_iterator;
};

// traits for integration \int_D (...) over the whole domain D
template <typename FeSpace_, typename... Quadrature_> struct fe_cell_assembler_traits {
    using FeType = typename FeSpace_::FeType;
    static constexpr int local_dim = FeSpace_::local_dim;
    static constexpr int embed_dim = FeSpace_::embed_dim;
    static constexpr int n_components = FeSpace_::n_components;

    using dof_descriptor = FeType::template cell_dof_descriptor<local_dim>;
    using BasisType = typename dof_descriptor::BasisType;
    static constexpr int n_basis = BasisType::n_basis;  
    using Quadrature = decltype([]() {
        if constexpr (sizeof...(Quadrature_) == 0) {
            return typename FeType::template cell_quadrature_t<local_dim> {};
        } else {
            using UserQuadrature = std::tuple_element_t<0, std::tuple<Quadrature_...>>;
            fdapde_static_assert(
              UserQuadrature::local_dim == local_dim, SUPPLIED_QUADRATURE_DIMENSION_DOES_NOT_MATCH_PROBLEM_DIMENSION);
            return std::get<0>(std::tuple<Quadrature_...>());
        }
    }());
    static constexpr int n_quadrature_nodes = Quadrature::degree;
    fdapde_static_assert(
      internals::is_fe_quadrature_simplex<Quadrature>, SUPPLIED_QUADRATURE_FORMULA_IS_NOT_FOR_SIMPLEX_INTEGRATION);
    using geo_iterator = typename Triangulation<local_dim, embed_dim>::cell_iterator;
    using dof_iterator = typename DofHandler<local_dim, embed_dim, finite_element_tag>::cell_iterator;
};
  
// base class for vector finite element assembly loops
template <typename Triangulation_, typename Form_, int Options_, typename... Quadrature_>
struct fe_assembler_base {
    // detect test space (since a test function is always present in a weak form)
    using TestSpace = test_space_t<Form_>;
    using Triangulation = typename std::decay_t<Triangulation_>;
    using Form = std::decay_t<
      decltype(xpr_wrap<FeMap, decltype([]<typename Xpr>() {
	    return !(
	        std::is_invocable_v<Xpr, fe_assembler_packet<Triangulation::embed_dim>> ||
		requires(Xpr xpr, int i, int j, fe_assembler_packet<Triangulation::embed_dim> input_type) {
		    xpr.eval(i, j, input_type);    // vector case
		});
	  })>(std::declval<Form_>()))>;
    static constexpr int local_dim = Triangulation::local_dim;
    static constexpr int embed_dim = Triangulation::embed_dim;
    static constexpr int Options = Options_;
    using FunctionSpace = TestSpace;
    using FeType = typename FunctionSpace::FeType;
    using DofHandlerType = DofHandler<local_dim, embed_dim, finite_element_tag>;
    using fe_traits = std::conditional_t<
      Options == CellMajor, fe_cell_assembler_traits<FunctionSpace, Quadrature_...>,
      fe_face_assembler_traits<FunctionSpace, Quadrature_...>>;
    using Quadrature = typename fe_traits::Quadrature;
    using dof_descriptor = typename fe_traits::dof_descriptor;
    using BasisType = typename dof_descriptor::BasisType;
    static constexpr int n_quadrature_nodes = Quadrature::order;
    static constexpr int n_basis = BasisType::n_basis;
    static constexpr int n_components = FunctionSpace::n_components;
    using discretization_category = typename TestSpace::discretization_category;
    fdapde_static_assert(
      std::is_same_v<discretization_category FDAPDE_COMMA finite_element_tag>,
      THIS_CLASS_IS_FOR_FINITE_ELEMENT_DISCRETIZATION_ONLY);

    fe_assembler_base() = default;
    fe_assembler_base(
      const Form_& form, typename fe_traits::geo_iterator begin, typename fe_traits::geo_iterator end,
      const Quadrature_&... quadrature)
        requires(sizeof...(quadrature) <= 1):
        form_(xpr_wrap<FeMap, decltype([]<typename Xpr>() {
	      return !(
		  std::is_invocable_v<Xpr, fe_assembler_packet<Triangulation::embed_dim>> ||
		  requires(Xpr xpr, int i, int j, fe_assembler_packet<Triangulation::embed_dim> input_type) {
		      xpr.eval(i, j, input_type);    // vector case
		});
	    })>(form)),
        quadrature_([... quadrature = std::forward<const Quadrature_>(quadrature)]() {
            if constexpr (sizeof...(quadrature) == 1) {
                return std::get<0>(std::make_tuple(quadrature...));
            } else {
                return Quadrature {};
            }
        }()),
        dof_handler_(&internals::test_space(form_).dof_handler()),
        test_space_(&internals::test_space(form_)),
        begin_(begin),
        end_(end) {
        fdapde_assert(dof_handler_->n_dofs() > 0);
    }
    const TestSpace& test_space() const { return *test_space_; }
   protected:
    // compile-time evaluation of \psi_i(q_j), i = 1, ..., n_basis, j = 1, ..., n_quadrature_nodes
    template <typename Quadrature__, typename fe_traits__>
    static consteval MdArray<
      double, MdExtents<fe_traits__::n_basis, Quadrature__::order, fe_traits__::n_components>>
    eval_shape_values() {
        fdapde_static_assert(   // check dimensions only for user-provided quadrature
          std::is_same_v<Quadrature__ FDAPDE_COMMA Quadrature> || Quadrature__::local_dim == local_dim,
          SUPPLIED_QUADRATURE_DIMENSION_DOES_NOT_MATCH_PROBLEM_DIMENSION);
        using BasisType = typename fe_traits__::BasisType;
	using dof_descriptor = typename fe_traits__::dof_descriptor;
        constexpr int n_basis = BasisType::n_basis;
        constexpr int n_quadrature_nodes = Quadrature__::order;
	constexpr int n_components = fe_traits__::n_components;
	
        MdArray<double, MdExtents<n_basis, n_quadrature_nodes, n_components>> shape_values_ {};
        BasisType basis {dof_descriptor().dofs_phys_coords()};
        for (int i = 0; i < n_basis; ++i) {
            // evaluation of \psi_i at q_j, j = 1, ..., n_quadrature_nodes
            for (int j = 0; j < n_quadrature_nodes; ++j) {
                const auto value = basis[i](Quadrature__::nodes.row(j).transpose());
                for (int k = 0; k < n_components; ++k) {
                    shape_values_(i, j, k) = scalar_or_kth_component_of(value, k);
                }
            }
        }
        return shape_values_;
    }
    // compile-time evaluation of \nabla{\psi_i}(q_j), i = 1, ..., n_basis, j = 1, ..., n_quadrature_nodes
    // NB: reference shape functions gradients are intentionally stored **by row** instead of **by column** for memory
    // aligness reasons (essentially, to allow contiguous access of components' gradients)
    template <typename Quadrature__, typename fe_traits__>
    static consteval MdArray<
      double, MdExtents<fe_traits__::n_basis, Quadrature__::order, fe_traits__::n_components, local_dim>>
    eval_shape_grads() {
        fdapde_static_assert(   // check dimensions only for user-provided quadrature
          std::is_same_v<Quadrature__ FDAPDE_COMMA Quadrature> || Quadrature__::local_dim == local_dim,
          SUPPLIED_QUADRATURE_DIMENSION_DOES_NOT_MATCH_PROBLEM_DIMENSION);
        using BasisType = typename fe_traits__::BasisType;
	using dof_descriptor = typename fe_traits__::dof_descriptor;
        constexpr int n_basis = BasisType::n_basis;
        constexpr int n_quadrature_nodes = Quadrature__::order;
        constexpr int n_components = fe_traits__::n_components;
	
        MdArray<double, MdExtents<n_basis, n_quadrature_nodes, n_components, local_dim>> shape_grad_ {};
        BasisType basis {dof_descriptor().dofs_phys_coords()};
        for (int i = 0; i < n_basis; ++i) {
            // evaluation of \nabla{\psi_i} at q_j, j = 1, ..., n_quadrature_nodes
            for (int j = 0; j < n_quadrature_nodes; ++j) {
                const auto grad = basis[i].gradient()(Quadrature__::nodes.row(j).transpose());
                for (int k = 0; k < n_components; ++k) {
                    for (int h = 0; h < local_dim; ++h) { shape_grad_(i, j, k, h) = grad(h, k); }
                }
            }
        }
        return shape_grad_;
    }
    // compile-time evaluation of hessian matrix of \psi_i(q_j), i = 1, ..., n_basis, j = 1, ..., n_quadrature_nodes
    template <typename Quadrature__, typename fe_traits__>
    static consteval MdArray<
      double, MdExtents<fe_traits__::n_basis, Quadrature__::order, fe_traits__::n_components, local_dim, local_dim>>
    eval_shape_hess() {
        fdapde_static_assert(   // check dimensions only for user-provided quadrature
          std::is_same_v<Quadrature__ FDAPDE_COMMA Quadrature> || Quadrature__::local_dim == local_dim,
          SUPPLIED_QUADRATURE_DIMENSION_DOES_NOT_MATCH_PROBLEM_DIMENSION);
        using BasisType = typename fe_traits__::BasisType;
	using dof_descriptor = typename fe_traits__::dof_descriptor;
        constexpr int n_basis = BasisType::n_basis;
        constexpr int n_quadrature_nodes = Quadrature__::order;
        constexpr int n_components = fe_traits__::n_components;
	
        MdArray<double, MdExtents<n_basis, n_quadrature_nodes, n_components, local_dim, local_dim>> shape_hess_ {};
        BasisType basis {dof_descriptor().dofs_phys_coords()};
	if constexpr(n_components == 1) {
        for (int i = 0; i < n_basis; ++i) {
            // evaluation of \nabla{\psi_i} at q_j, j = 1, ..., n_quadrature_nodes
            for (int j = 0; j < n_quadrature_nodes; ++j) {
                const auto hess = basis[i].hessian()(Quadrature__::nodes.row(j).transpose());
                for (int k = 0; k < n_components; ++k) {
                    shape_hess_.template slice<0, 1, 2>(i, j, k).assign_inplace_from(hess.data());
                }
            }
        }
	}
        return shape_hess_;
    }

    // test basis functions evaluations
    static constexpr MdArray<double, MdExtents<n_basis, n_quadrature_nodes, n_components>> test_shape_values_ =
      eval_shape_values<Quadrature, fe_traits>();
    static constexpr MdArray<double, MdExtents<n_basis, n_quadrature_nodes, n_components, local_dim>>
      test_shape_grads_ = eval_shape_grads<Quadrature, fe_traits>();
    static constexpr MdArray<double, MdExtents<n_basis, n_quadrature_nodes, n_components, local_dim, local_dim>>
      test_shape_hess_  = eval_shape_hess <Quadrature, fe_traits>();

    void
    distribute_quadrature_nodes(typename fe_traits::dof_iterator begin, typename fe_traits::dof_iterator end) const {
        Eigen::Matrix<double, Dynamic, Dynamic> quad_nodes;
        Eigen::Map<const Eigen::Matrix<
          double, n_quadrature_nodes, Quadrature::local_dim,
          Quadrature::local_dim != 1 ? Eigen::RowMajor : Eigen::ColMajor>>
          ref_quad_nodes(quadrature_.nodes.data());
        quad_nodes.resize(n_quadrature_nodes * (end_.index() - begin_.index()), embed_dim);
        int local_cell_id = 0;
        for (typename fe_traits::geo_iterator it = begin_; it != end_; ++it) {
            for (int q_k = 0; q_k < n_quadrature_nodes; ++q_k) {
                quad_nodes.row(local_cell_id * n_quadrature_nodes + q_k) =
                  it->J() * ref_quad_nodes.row(q_k).transpose() + it->node(0);
            }
            local_cell_id++;
        }
        // evaluate Map nodes at quadrature nodes
        xpr_for_each<
          decltype([]<typename Xpr_, typename... Args>(Xpr_& xpr, Args&&... args) {
              xpr.init(std::forward<Args>(args)...);
              return;
          }),
          decltype([]<typename Xpr_>() { return requires(Xpr_ xpr) { xpr.init(quad_nodes, begin, end); }; })>(
          form_, quad_nodes, begin, end);
        return;
    }
    // moves \nabla{\psi_i}(q_k) from the reference cell to physical cell pointed by it
    template <typename CellIterator, typename SrcMdArray, typename DstMdArray>
    constexpr void eval_shape_grads_on_cell(CellIterator& it, const SrcMdArray& ref_grads, DstMdArray& dst) const {
        constexpr int n_basis_ = SrcMdArray::static_extents[0];
        constexpr int n_quadrature_nodes_ = SrcMdArray::static_extents[1];
        constexpr int n_components_ = SrcMdArray::static_extents[2];
        for (int i = 0; i < n_basis_; ++i) {
            for (int j = 0; j < n_quadrature_nodes_; ++j) {
                // get i-th reference basis gradient evaluted at j-th quadrature node
                auto ref_grad = ref_grads.template slice<0, 1>(i, j).as_matrix();
                Matrix<double, embed_dim, n_components_> mapped_grad;
                for (int k = 0; k < n_components_; ++k) {
                    mapped_grad.col(k) =
                      (ref_grad.row(k) * Map<const double, local_dim, embed_dim>(it->invJ().data()))
                        .transpose();
                }
                dst.template slice<0, 1>(i, j).assign_inplace_from(mapped_grad.data());
            }
        }
        return;
    }
    // computes div(\psi_i)(q_k) on the physical cell pointed by it
    template <typename CellIterator, typename GradMdArray, typename DivMdArray>
    constexpr void eval_shape_div_on_cell(CellIterator& it, const GradMdArray& ref_grads, DivMdArray& dst) const {
        constexpr int n_basis_ = GradMdArray::static_extents[0];
        constexpr int n_quadrature_nodes_ = GradMdArray::static_extents[1];
        constexpr int n_components_ = GradMdArray::static_extents[2];
        fdapde_constexpr_assert(n_components_ == 1 || n_components_ == local_dim);
        for (int i = 0; i < n_basis_; ++i) {
            for (int j = 0; j < n_quadrature_nodes_; ++j) {
                // get i-th reference basis gradient evaluted at j-th quadrature node
                auto ref_grad = ref_grads.template slice<0, 1>(i, j).as_matrix();
		// compute divergence as trace of jacobian matrix
		double div_ = 0;
                for (int k = 0; k < n_components_; ++k) {
                    div_ +=
                      ref_grad.row(k).dot(Map<const double, local_dim, embed_dim>(it->invJ().data()).col(k));
                }
                dst(i, j) = div_;
            }
        }
        return;
    }
    // computes hessian matrix of \psi_i evaluated at quadrature nodes q_k on the physical cell pointed by it
    template <typename CellIterator, typename SrcMdArray, typename DstMdArray>
    constexpr void eval_shape_hess_on_cell(CellIterator& it, const SrcMdArray& ref_hess, DstMdArray& dst) const {
        constexpr int n_basis_ = SrcMdArray::static_extents[0];
        constexpr int n_quadrature_nodes_ = SrcMdArray::static_extents[1];
        constexpr int n_components_ = SrcMdArray::static_extents[2];
        for (int i = 0; i < n_basis_; ++i) {
            for (int j = 0; j < n_quadrature_nodes_; ++j) {
                Matrix<double, embed_dim, embed_dim> mapped_hess;
                for (int k = 0; k < n_components_; ++k) {
                    // move i-th reference basis hessian evaluted at j-th quadrature node on physical cell
                    mapped_hess = Map<const double, local_dim, embed_dim>(it->invJ().data()).transpose() *
                                  ref_hess.template slice<0, 1, 2>(i, j, k).as_matrix();
                    dst.template slice<0, 1, 2>(i, j, k).assign_inplace_from(mapped_hess.data());
                }
            }
        }
        return;
    }

    Form form_;
    Quadrature quadrature_ {};
    const DofHandlerType* dof_handler_;
    const TestSpace* test_space_;
    typename fe_traits::geo_iterator begin_, end_;
};
    
}   // namespace internals
}   // namespace fdapde

#endif   // __FDAPDE_FE_ASSEMBLER_BASE_H__
