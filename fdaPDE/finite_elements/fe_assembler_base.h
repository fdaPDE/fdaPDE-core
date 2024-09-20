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
    compute_physical_quad_nodes = 0x0010,
    compute_cell_diameter       = 0x0020
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
template <typename Xpr> constexpr decltype(auto)  test_space(Xpr&& xpr) {
    constexpr bool found = meta::xpr_find<
      decltype([]<typename Xpr_>() { return requires { typename Xpr_::TestSpace; }; }), std::decay_t<Xpr>>();
    fdapde_static_assert(found, NO_TEST_SPACE_FOUND_IN_EXPRESSION);
    return meta::xpr_query<
      decltype([]<typename Xpr_>(Xpr_&& xpr) -> auto& { return xpr.fe_space(); }),
      decltype([]<typename Xpr_>() { return requires { typename Xpr_::TestSpace; }; })>(std::forward<Xpr>(xpr));
}

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
