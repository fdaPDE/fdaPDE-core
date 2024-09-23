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

#ifndef __FE_ASSEMBLER_BASE_H__
#define __FE_ASSEMBLER_BASE_H__

#include <unordered_map>

#include "../fields/meta.h"
#include "../linear_algebra/constexpr_matrix.h"
#include "../linear_algebra/mdarray.h"
#include "fe_integration.h"

namespace fdapde {

template <typename Derived_> struct FeMap;

enum fe_assembler_flags {
    compute_shape_values        = 0x0001,
    compute_shape_grad          = 0x0002,
    compute_second_derivatives  = 0x0004,
    compute_shape_div           = 0x0008,
    compute_physical_quad_nodes = 0x0010,
    compute_cell_diameter       = 0x0020
};

[[maybe_unused]] static constexpr int CellMajor = 0;
[[maybe_unused]] static constexpr int FaceMajor = 1;
  
namespace internals {

// informations sent from the assembly loop to the integrated forms
template <int LocalDim> struct fe_assembler_packet {
    static constexpr int local_dim = LocalDim;
    fe_assembler_packet(int n_trial_components, int n_test_components) :
        trial_value(n_trial_components), test_value(n_test_components), trial_grad(n_trial_components),
        test_grad(n_test_components) { }
    fe_assembler_packet(int n_components) : fe_assembler_packet(n_components, n_components) { }
    fe_assembler_packet() : fe_assembler_packet(1, 1) { }
    fe_assembler_packet(fe_assembler_packet&&) noexcept = default;
    fe_assembler_packet(const fe_assembler_packet&) noexcept = default;

    // geometric informations
    int quad_node_id;       // active physical quadrature node index
    double cell_measure;    // active cell measure
    double cell_diameter;   // active cell diameter

    // functional informations (Dynamic stands for number of components)
    MdArray<double, MdExtents<Dynamic>> trial_value, test_value;            // \psi_i(q_k), \psi_j(q_k)
    MdArray<double, MdExtents<Dynamic, local_dim>> trial_grad, test_grad;   // \nabla{\psi_i}(q_k), \nabla{\psi_j}(q_k)
    double trial_div = 0, test_div = 0;
};
  
// detect trial space from bilinear form
template <typename Xpr> constexpr decltype(auto) trial_space(Xpr&& xpr) {
    constexpr bool found = meta::xpr_find<
      decltype([]<typename Xpr_>() { return requires { typename Xpr_::TrialSpace; }; }), std::decay_t<Xpr>>();
    fdapde_static_assert(found, NO_TRIAL_SPACE_FOUND_IN_EXPRESSION);
    return meta::xpr_query<
      decltype([]<typename Xpr_>(Xpr_&& xpr) -> auto& { return xpr.fe_space(); }),
      decltype([]<typename Xpr_>() { return requires { typename Xpr_::TrialSpace; }; })>(std::forward<Xpr>(xpr));
}
template <typename Xpr> using trial_space_t = std::decay_t<decltype(trial_space(std::declval<Xpr>()))>;
// detect test space from bilinear form
template <typename Xpr> constexpr decltype(auto)  test_space(Xpr&& xpr) {
    constexpr bool found = meta::xpr_find<
      decltype([]<typename Xpr_>() { return requires { typename Xpr_::TestSpace; }; }), std::decay_t<Xpr>>();
    fdapde_static_assert(found, NO_TEST_SPACE_FOUND_IN_EXPRESSION);
    return meta::xpr_query<
      decltype([]<typename Xpr_>(Xpr_&& xpr) -> auto& { return xpr.fe_space(); }),
      decltype([]<typename Xpr_>() { return requires { typename Xpr_::TestSpace; }; })>(std::forward<Xpr>(xpr));
}
template <typename Xpr> using test_space_t = std::decay_t<decltype(test_space(std::declval<Xpr>()))>;

template <typename T>
    requires(
      std::is_floating_point_v<T> ||
      requires(T t, int k) {
          { t.size() } -> std::convertible_to<std::size_t>;
          { t.operator[](k) } -> std::convertible_to<double>;
      })
constexpr auto scalar_or_kth_component_of(const T& t, std::size_t k) {
    if constexpr (std::is_floating_point_v<T>) {
        return t;
    } else {
        fdapde_constexpr_assert(k < t.size());
        return t[k];
    }
}

// implementation of galerkin assembly loop for the discretization of arbitrarily bilinear forms
template <typename Derived> struct fe_assembly_xpr_base;

// traits for surface integration \int_{\partial D} (...)
template <typename FeSpace_, typename... Quadrature_> struct fe_face_assembler_traits {
    using FeType = typename FeSpace_::FeType;
    static constexpr int local_dim = FeSpace_::local_dim;
    static constexpr int embed_dim = FeSpace_::embed_dim;
    static constexpr int n_components = FeSpace_::n_components;

    using dof_descriptor = FeType::template face_dof_descriptor<local_dim>;
    using Quadrature = decltype([]() {
        if constexpr (sizeof...(Quadrature_) == 0) {
            return typename FeType::template select_face_quadrature_t<local_dim> {};
        } else {
            return std::get<0>(std::tuple<Quadrature_...>());
        }
    }());
    fdapde_static_assert(
      internals::is_fe_quadrature_simplex<Quadrature>, SUPPLIED_QUADRATURE_FORMULA_IS_NOT_FOR_SIMPLEX_INTEGRATION);
    using geo_iterator = typename Triangulation<local_dim, embed_dim>::boundary_iterator;
    using dof_iterator = typename DofHandler<local_dim, embed_dim>::boundary_iterator;
};

// traits for integration \int_D (...) over the whole domain D
template <typename FeSpace_, typename... Quadrature_> struct fe_cell_assembler_traits {
    using FeType = typename FeSpace_::FeType;
    static constexpr int local_dim = FeSpace_::local_dim;
    static constexpr int embed_dim = FeSpace_::embed_dim;
    static constexpr int n_components = FeSpace_::n_components;

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
  
// base class for vector finite element assembly loops
template <typename Triangulation_, typename Form_, int Options_, typename... Quadrature_>
struct fe_assembler_base {
    // detect test space (since a test function is always present in a weak form)
    using TestSpace = test_space_t<Form_>;
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
    static constexpr int n_components = FunctionSpace::n_components;
  
    fe_assembler_base() = default;
    fe_assembler_base(
      const Form_& form, typename fe_traits::geo_iterator begin, typename fe_traits::geo_iterator end,
      const Quadrature_&... quadrature)
        requires(sizeof...(quadrature) <= 1) :
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
   protected:
    // compile-time evaluation of \psi_i(q_j), i = 1, ..., n_basis, j = 1, ..., n_quadrature_nodes
    template <typename Quadrature__, typename dof_descriptor__>
    static consteval MdArray<
      double, MdExtents<dof_descriptor__::BasisType::n_basis, Quadrature__::n_nodes, n_components>>
    eval_shape_values() {
        using BasisType = typename dof_descriptor__::BasisType;
        constexpr int n_basis = BasisType::n_basis;
        constexpr int n_quadrature_nodes = Quadrature__::n_nodes;
        MdArray<double, MdExtents<n_basis, n_quadrature_nodes, n_components>> shape_values_ {};
        BasisType basis {dof_descriptor__().dofs_phys_coords()};
        for (int i = 0; i < n_basis; ++i) {
            // evaluation of \nabla{\psi_i} at q_j, j = 1, ..., n_quadrature_nodes
            for (int j = 0; j < n_quadrature_nodes; ++j) {
                auto value = basis[i](Quadrature::nodes.row(j).transpose());
                for (int k = 0; k < n_components; ++k) {
                    shape_values_(i, j, k) = scalar_or_kth_component_of(value, k);
                }
            }
        }
        return shape_values_;
    }
    // compile-time evaluation of \nabla{\psi_i}(q_j), i = 1, ..., n_basis, j = 1, ..., n_quadrature_nodes
    template <typename Quadrature__, typename dof_descriptor__>
    static consteval MdArray<
      double, MdExtents<dof_descriptor__::BasisType::n_basis, Quadrature__::n_nodes, n_components, local_dim>>
    eval_shape_grads() {
        using BasisType = typename dof_descriptor__::BasisType;
        constexpr int n_basis = BasisType::n_basis;
        constexpr int n_quadrature_nodes = Quadrature__::n_nodes;
        MdArray<double, MdExtents<n_basis, n_quadrature_nodes, n_components, local_dim>> shape_grad_ {};
        BasisType basis {dof_descriptor__().dofs_phys_coords()};
        for (int i = 0; i < n_basis; ++i) {
            // evaluation of \nabla{\psi_i} at q_j, j = 1, ..., n_quadrature_nodes
            for (int j = 0; j < n_quadrature_nodes; ++j) {
                auto grad = basis[i].gradient()(Quadrature::nodes.row(j).transpose());
                for (int k = 0; k < n_components; ++k) {
                    for (int h = 0; h < local_dim; ++h) { shape_grad_(i, j, k, h) = grad(h, k); }
                }
            }
        }
        return shape_grad_;
    }

    // test basis functions evaluations
    static constexpr MdArray<double, MdExtents<n_basis, n_quadrature_nodes, n_components>> test_shape_values_ =
      eval_shape_values<Quadrature, dof_descriptor>();
    static constexpr MdArray<double, MdExtents<n_basis, n_quadrature_nodes, n_components, local_dim>>
      test_shape_grads_ = eval_shape_grads<Quadrature, dof_descriptor>();

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
	return;
    }
    // moves \nabla{\psi_i}(q_k) from the reference cell to physical cell pointed by it
    template <typename CellIterator, typename RefGrads, typename Dst>
    constexpr void eval_shape_grads_on_cell(CellIterator& it, const RefGrads& ref_grads, Dst& dst) const {
        const int n_basis_ = ref_grads.extent(0);
        const int n_quadrature_nodes_ = ref_grads.extent(1);
        const int n_components_ = ref_grads.extent(2);

	for (int i = 0; i < n_basis_; ++i) {
            for (int j = 0; j < n_quadrature_nodes_; ++j) {
                // get i-th reference basis gradient evaluted at j-th quadrature node
                auto ref_grad = ref_grads.template slice<0, 1>(i, j).matrix();
                for (int k = 0; k < n_components_; ++k) {
                    dst.template slice<0, 1, 2>(i, j, k).assign_inplace_from(cexpr::Matrix<double, 1, local_dim>(
                      ref_grad.row(k) * cexpr::Map<const double, local_dim, embed_dim>(it->invJ().data())));
                }
            }
        }
        return;
    }
    // computes div(\psi_i)(q_k) on the physical cell pointer by it
    template <typename CellIterator, typename RefGrads, typename Dst>
    constexpr void eval_shape_div_on_cell(CellIterator& it, const RefGrads& ref_grads, Dst& dst) const {
        const int n_basis_ = ref_grads.extent(0);
        const int n_quadrature_nodes_ = ref_grads.extent(1);
        const int n_components_ = ref_grads.extent(2);
	fdapde_assert(n_components_ == local_dim);
	
        for (int i = 0; i < n_basis_; ++i) {
            for (int j = 0; j < n_quadrature_nodes_; ++j) {
                double div_ = 0;
                for (int k = 0; k < n_components_; ++k) {
                    // get k-th component of i-th reference basis gradient evaluated at j-th quadrature node
                    auto ref_grad = ref_grads.template slice<0, 1, 2>(i, j, k).matrix();
                    div_ += cexpr::Map<const double, local_dim, embed_dim>(it->invJ().data()).col(k).dot(ref_grad);
                }
                dst(i, j) = div_;
            }
        }
        return;
    }

    Form form_;
    Quadrature quadrature_ {};
    const DofHandler<local_dim, embed_dim>* dof_handler_;
    typename fe_traits::geo_iterator begin_, end_;
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
        fdapde_assert(lhs.rows() == rhs.rows() && lhs.cols() == rhs.cols());
    }
    OutputType assemble() const {
        if constexpr (std::is_same_v<OutputType, SpMatrix<double>>) {
            SpMatrix<double> assembled_mat(lhs_.rows(), lhs_.cols());
            std::vector<Eigen::Triplet<double>> triplet_list;
            assemble(triplet_list);
            // linearity of the integral is implicitly used here, as duplicated triplets are summed up (see Eigen docs)
            assembled_mat.setFromTriplets(triplet_list.begin(), triplet_list.end());
            assembled_mat.makeCompressed();
            return assembled_mat;
        } else {
            DVector<double> assembled_vec(lhs_.rows());
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
    constexpr int rows() const { return lhs_.rows(); }
    constexpr int cols() const { return lhs_.cols(); }
   private:
    Lhs lhs_;
    Rhs rhs_;
};
  
template <typename Lhs, typename Rhs>
fe_assembly_add_op<Lhs, Rhs> operator+(const fe_assembly_xpr_base<Lhs>& lhs, const fe_assembly_xpr_base<Rhs>& rhs) {
    return fe_assembly_add_op<Lhs, Rhs>(lhs.derived(), rhs.derived());
}

}   // namespace internals
}   // namespace fdapde

#endif   // __FE_ASSEMBLER_BASE_H__
