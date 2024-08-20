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

#ifndef __FE_P_H__
#define __FE_P_H__

#include "../geometry/simplex.h"
#include "../linear_algebra/constexpr_matrix.h"
#include "../utils/symbols.h"
#include "fe_integration.h"
#include "lagrange_basis.h"

namespace fdapde {

// representation of the finite element space P_h^K = { v \in H^1(D) : v_{e} \in P^K \forall e \in T_h }
template <int Order> struct FeP {
    static constexpr int order = Order;
    fdapde_static_assert(Order < 4, THIS_CLASS_SUPPORTS_LAGRANGE_ELEMENTS_UP_TO_ORDER_THREE);

    template <int LocalDim> struct dof_descriptor {
        static constexpr int local_dim = LocalDim;
        using ReferenceCell = Simplex<local_dim, local_dim>;   // reference unit simplex
        static constexpr bool dof_sharing = true;              // piecewise continuous finite elements
        static constexpr int n_dofs_per_edge = local_dim > 1 ? (Order - 1 < 0 ? 0 : (Order - 1)) : 0;
        static constexpr int n_dofs_per_face = local_dim > 2 ? (Order - 2 < 0 ? 0 : (Order - 2)) : 0;
        static constexpr int n_dofs_internal = local_dim < 3 ? (Order - local_dim < 0 ? 0 : (Order - local_dim)) : 0;
        static constexpr int n_dofs_per_cell = ReferenceCell::n_nodes + n_dofs_per_edge * ReferenceCell::n_edges +
                                               n_dofs_per_face * ReferenceCell::n_faces + n_dofs_internal;
        using BasisType = LagrangeBasis<local_dim, Order>;
        using ElementType = typename BasisType::PolynomialType;

        constexpr dof_descriptor() : dofs_phys_coords_(), dofs_bary_coords_() {
            // compute dofs physical coordinates on reference cell
            constexpr int n_nodes = local_dim + 1;
            cexpr::Matrix<double, local_dim, n_nodes> reference_simplex;
            reference_simplex.setZero();
            for (int i = 0; i < local_dim; ++i) { reference_simplex(i, i + 1) = 1; }
            dofs_phys_coords_.template topRows<n_nodes>(0) = reference_simplex.transpose();
            int j = n_nodes;
            // constexpr enumeration of reference simplex edges
            auto edge_enumerate = [&, this]() {
                std::vector<bool> bitmask(n_nodes, 0);
                std::fill_n(bitmask.begin(), 2, 1);
                cexpr::Matrix<double, local_dim, 2> edge_coords;
                for (int i = 0; i < ReferenceCell::n_edges; ++i) {
                    for (int k = 0, h = 0; k < n_nodes; ++k) {
                        if (bitmask[k]) edge_coords.col(h++) = reference_simplex.col(k);
                    }
                    for (int k = 0; k < n_dofs_per_edge; ++k) {
                        dofs_phys_coords_.row(j++) =
                          (edge_coords.col(0) + (k + 1) * (edge_coords.col(1) - edge_coords.col(0)) / Order)
                            .transpose();
                    }
                    std::prev_permutation(bitmask.begin(), bitmask.end());
                }
            };
	    // enumerate dofs on reference cell
            if constexpr (local_dim == 1) {
                if constexpr (n_dofs_internal > 0) {
                    for (int i = 0; i < n_dofs_internal; ++i) { dofs_phys_coords_[j++] = (i + 1) * 1. / Order; }
                }
            }
            if constexpr (local_dim == 2) {
                if constexpr (n_dofs_per_edge > 0) { edge_enumerate(); }
                if constexpr (n_dofs_internal > 0) {
                    // add simplex barycenter
                    dofs_phys_coords_.row(j++) =
                      (reference_simplex.col(1) + reference_simplex.col(2)) * 1.0 / (local_dim + 1);
                }
            }
            if constexpr (local_dim == 3) {
                if constexpr (n_dofs_per_edge > 0) { edge_enumerate(); }
                if constexpr (n_dofs_per_face > 0) {
                    // add barycenter of tetrahedron faces
                    std::vector<bool> bitmask(n_nodes, 0);
                    std::fill_n(bitmask.begin(), ReferenceCell::n_nodes_per_face, 1);
                    cexpr::Matrix<double, local_dim, 3> face_coords;
                    for (int i = 0; i < ReferenceCell::n_faces; ++i) {
                        for (int k = 0, h = 0; k < n_nodes; ++k) {
                            if (bitmask[k]) face_coords.col(h++) = reference_simplex.col(k);
                        }
                        cexpr::Matrix<double, local_dim, 2> J;
                        for (int k = 0; k < 2; ++k) J.col(k) = face_coords.col(k + 1) - face_coords.col(0);
                        dofs_phys_coords_.row(j++) =
                          (J * cexpr::Vector<double, 2>::Constant(1.0 / (local_dim + 1))) + face_coords.col(0);
                        std::prev_permutation(bitmask.begin(), bitmask.end());
                    }
                }
            }
            // compute barycentric coordinates
            dofs_bary_coords_.template rightCols<local_dim>(1) = dofs_phys_coords_;
            if constexpr (local_dim == 1) {
                for (int i = 0; i < dofs_bary_coords_.rows(); ++i) {
                    dofs_bary_coords_(i, 0) = 1 - dofs_bary_coords_(i, 1);
                }
            } else {
                for (int i = 0; i < n_dofs_per_cell; ++i) {
                    double sum = 0;
                    for (int j = 0; j < local_dim; ++j) sum += dofs_bary_coords_(i, j);
                    dofs_bary_coords_(i, 0) = 1 - sum;
                }
            }
        }
        // getters
        constexpr const auto& dofs_phys_coords() const { return dofs_phys_coords_; }
        constexpr const auto& dofs_bary_coords() const { return dofs_bary_coords_; }
       private:
        cexpr::Matrix<double, n_dofs_per_cell, local_dim> dofs_phys_coords_;       // dofs physical coordinates
        cexpr::Matrix<double, n_dofs_per_cell, local_dim + 1> dofs_bary_coords_;   // dofs barycentric coordinates
    };

    // select quadrature which optimally integrates (Order + 1) polynomials
    template <int LocalDim> class select_cell_quadrature {
        static constexpr int select_quadrature_() {
            if (LocalDim == 1) return Order == 1 ? 2 : (Order == 2 ? 3 : 3);
            if (LocalDim == 2) return Order == 1 ? 3 : (Order == 2 ? 6 : 6);
            if (LocalDim == 3) return Order == 1 ? 4 : (Order == 2 ? 5 : 5);
        }
       public:
        using type = internals::fe_quadrature_simplex<LocalDim, select_quadrature_()>;
    };
  
    template <int LocalDim> using cell_dof_descriptor = dof_descriptor<LocalDim>;
    template <int LocalDim> using face_dof_descriptor = dof_descriptor<LocalDim - 1>;
    template <int LocalDim> using select_cell_quadrature_t = typename select_cell_quadrature<LocalDim>::type;
    template <int LocalDim> using select_face_quadrature_t = typename select_cell_quadrature<LocalDim - 1>::type;
};
  
// lagrange finite element alias
[[maybe_unused]] static struct P1_ : FeP<1> { } P1;
[[maybe_unused]] static struct P2_ : FeP<2> { } P2;
[[maybe_unused]] static struct P3_ : FeP<3> { } P3;
    
}   // namespace fdapde

#endif   // __FE_P_H__
