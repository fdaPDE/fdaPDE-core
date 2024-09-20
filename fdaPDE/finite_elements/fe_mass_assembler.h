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

#ifndef __FE_MASS_ASSEMBLER_H__
#define __FE_MASS_ASSEMBLER_H__

#include "fe_assembler_base.h"

namespace fdapde {
namespace internals {

// optimized computation of mass matrix [A]_{ij} = \int_D (\psi_i * \psi_j). Handles both scalar and vectorial fems
template <typename LhsFeSpace, typename RhsFeSpace = LhsFeSpace>
    requires(LhsFeSpace::local_dim == RhsFeSpace::local_dim && LhsFeSpace::embed_dim == RhsFeSpace::embed_dim)
class fe_mass_assembly_loop {
    static constexpr int local_dim = LhsFeSpace::local_dim;
    static constexpr int embed_dim = LhsFeSpace::embed_dim;
    // select the quadrature which optimally integrates the highest order finite element
    using Quadrature = higher_order_fe_quadrature_t<
      typename LhsFeSpace::template select_cell_quadrature_t<local_dim>,
      typename RhsFeSpace::template select_cell_quadrature_t<local_dim>>;
    static constexpr int n_quadrature_nodes = Quadrature::n_nodes;
    using lhs_cell_dof_descriptor = LhsFeSpace::template cell_dof_descriptor<local_dim>;
    using rhs_cell_dof_descriptor = RhsFeSpace::template cell_dof_descriptor<local_dim>;
    using LhsBasisType = typename lhs_cell_dof_descriptor::BasisType;
    using RhsBasisType = typename rhs_cell_dof_descriptor::BasisType;
    // number of basis on reference element
    static constexpr int n_lhs_basis = LhsBasisType::n_basis;
    static constexpr int n_rhs_basis = RhsBasisType::n_basis;
    // compile-time evaluation of integral \int_{\hat K} \psi_i \psi_j on reference element \hat K
    static constexpr cexpr::Matrix<double, n_lhs_basis, n_rhs_basis> int_table_ {[]() {
        std::array<double, n_lhs_basis * n_rhs_basis> int_table_ {};
        LhsBasisType lhs_basis {lhs_cell_dof_descriptor().dofs_phys_coords()};
        RhsBasisType rhs_basis {rhs_cell_dof_descriptor().dofs_phys_coords()};
        for (int i = 0; i < n_lhs_basis; ++i) {
            for (int j = 0; j < n_rhs_basis; ++j) {
                for (int k = 0; k < n_quadrature_nodes; ++k) {
                    int_table_[i * n_lhs_basis + j] +=
                      // for scalar elements, a.dot(b) casts to a plain scalar multiplication a * b
                      Quadrature::weights[k] * (lhs_basis[i].dot(rhs_basis[j]))(Quadrature::nodes.row(k).transpose());
                }
            }
        }
        return int_table_;
    }};
    typename LhsFeSpace::DofHandlerType* lhs_dof_handler_;
    typename RhsFeSpace::DofHandlerType* rhs_dof_handler_;
   public:
    fe_mass_assembly_loop() = default;
    fe_mass_assembly_loop(const fe_mass_assembly_loop&) = default;
    fe_mass_assembly_loop(fe_mass_assembly_loop&&) = default;
    // construct from functional spaces
    fe_mass_assembly_loop(const LhsFeSpace& lhs_fe_type, const RhsFeSpace& rhs_fe_type) :
        lhs_dof_handler_(&lhs_fe_type.dof_handler()), rhs_dof_handler_(&rhs_fe_type.dof_handler()) { }
    fe_mass_assembly_loop(const LhsFeSpace& fe_type)
        requires(std::is_same_v<LhsFeSpace, RhsFeSpace>)
        : lhs_dof_handler_(&fe_type.dof_handler()), rhs_dof_handler_(&fe_type.dof_handler()) { }
    // construct from dof handlers
    template <typename LhsDofHandler, typename RhsDofHandler>
        requires(std::is_convertible_v<LhsDofHandler, typename LhsFeSpace::DofHandlerType> &&
                 std::is_convertible_v<RhsDofHandler, typename RhsFeSpace::DofHandlerType>)
    fe_mass_assembly_loop(const LhsDofHandler& lhs_dof_handler, const RhsDofHandler& rhs_dof_handler) :
        lhs_dof_handler_(&lhs_dof_handler), rhs_dof_handler_(&rhs_dof_handler) { }
    template <typename DofHandler>
        requires(std::is_same_v<LhsFeSpace, RhsFeSpace> &&
                 std::is_convertible_v<DofHandler, typename LhsFeSpace::DofHandlerType>)
    fe_mass_assembly_loop(const DofHandler& dof_handler) :
        lhs_dof_handler_(&dof_handler), rhs_dof_handler_(&dof_handler) { }

    SpMatrix<double> assemble() {
        if (!lhs_dof_handler_) lhs_dof_handler_->enumerate(LhsFeSpace {});
        if (!rhs_dof_handler_) rhs_dof_handler_->enumerate(RhsFeSpace {});

        SpMatrix<double> assembled_mat(lhs_dof_handler_->n_dofs(), rhs_dof_handler_->n_dofs());
        std::vector<Eigen::Triplet<double>> triplet_list;
        DVector<int> lhs_active_dofs;
        DVector<int> rhs_active_dofs;
        for (typename LhsFeSpace::DofHandlerType::cell_iterator it = lhs_dof_handler_->cells_begin();
             it != lhs_dof_handler_->cells_end(); ++it) {
            lhs_active_dofs = it->dofs();
            if constexpr (std::is_same_v<LhsFeSpace, RhsFeSpace>) {   // galerkin discretization (square matrix)
                for (int i = 0; i < LhsBasisType::n_basis; ++i) {
                    for (int j = 0; j < i + 1; ++j) {
                        std::pair<const int&, const int&> minmax(std::minmax(lhs_active_dofs[i], lhs_active_dofs[j]));
                        triplet_list.emplace_back(minmax.first, minmax.second, int_table_(i, j) * it->measure());
                    }
                }
            } else {   // petrov-galerkin discretization (possibly rectangular matrix)
                rhs_active_dofs = rhs_dof_handler_->active_dofs(it->id());
                for (int i = 0; i < LhsBasisType::n_basis; ++i) {
                    for (int j = 0; j < RhsBasisType::n_basis; ++j) {
                        triplet_list.emplace_back(
                          lhs_active_dofs[i], rhs_active_dofs[j], int_table_(i, j) * it->measure());
                    }
                }
            }
        }
        // linearity of the integral is implicitly used here, as duplicated triplets are summed up (see Eigen docs)
        assembled_mat.setFromTriplets(triplet_list.begin(), triplet_list.end());
        assembled_mat.makeCompressed();
        if constexpr (std::is_same_v<LhsFeSpace, RhsFeSpace>) {
            return assembled_mat.selfadjointView<Eigen::Upper>();
        } else {
            return assembled_mat;
        }
    }
    constexpr int n_dofs() const { return rhs_dof_handler_->n_dofs(); }
    constexpr int rows() const { return lhs_dof_handler_->n_dofs(); }
    constexpr int cols() const { return rhs_dof_handler_->n_dofs(); }
};

}   // namespace internals
}   // namespace fdapde

#endif // __FE_MASS_ASSEMBLER_H__
