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

#include "../utils/symbols.h"
#include "../utils/traits.h"
#include "lagrange_basis.h"

namespace fdapde {
namespace core {

// representation of the finite element space P_h^K = { v \in H^1(D) : v_{e} \in P^K \forall e \in T_h }
template <int Order> struct FE_P {
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
        using BasisElementType = typename BasisType::PolynomialType;

        dof_descriptor() {
            // compute dofs physical coordinates on reference cell
            ReferenceCell reference_simplex = ReferenceCell::Unit();
            dofs_phys_coords_.topRows(ReferenceCell::n_nodes) = reference_simplex.nodes().transpose();
            int j = ReferenceCell::n_nodes;
            if constexpr (local_dim == 1) {
                if constexpr (n_dofs_internal > 0) {
                    for (int i = 0; i < n_dofs_internal; ++i) { dofs_phys_coords_[j++] = (i + 1) * 1. / Order; }
                }
            }
            if constexpr (local_dim == 2) {
                if constexpr (n_dofs_per_edge > 0) {
                    for (typename ReferenceCell::boundary_iterator it = reference_simplex.boundary_begin();
                         it != reference_simplex.boundary_end(); ++it) {
                        for (int i = 0; i < n_dofs_per_edge; ++i) {
                            dofs_phys_coords_.row(j++) = it->node(0) + (i + 1) * (it->node(1) - it->node(0)) / Order;
                        }
                    }
                }
                if constexpr (n_dofs_internal > 0) { dofs_phys_coords_.row(j++) = reference_simplex.barycenter(); }
            }
            if constexpr (local_dim == 3) {
                if constexpr (n_dofs_per_edge > 0) {
                    // cycle over unit tetrahedron edges
                    std::vector<bool> bitmask(ReferenceCell::n_nodes, 0);
                    std::fill_n(bitmask.begin(), 2, 1);   // each edge is made by 2 nodes
                    SMatrix<local_dim, 2> coords;
                    for (int h = 0; h < ReferenceCell::n_edges; ++h) {
                        for (int i = 0, h = 0; i < ReferenceCell::n_nodes; ++i) {
                            if (bitmask[i]) coords.col(h++) = reference_simplex.nodes().col(i);
                        }
                        for (int i = 0; i < n_dofs_per_edge; ++i) {
                            dofs_phys_coords_.row(j++) =
                              coords.col(0) + (i + 1) * (coords.col(1) - coords.col(0)) / Order;
                        }
                        std::prev_permutation(bitmask.begin(), bitmask.end());
                    }
                }
                if constexpr (n_dofs_per_face > 0) {
                    for (typename ReferenceCell::boundary_iterator it = reference_simplex.boundary_begin();
                         it != reference_simplex.boundary_end(); ++it) {
                        dofs_phys_coords_.row(j++) = it->barycenter();
                    }
                }
            }
            // compute barycentric coordinates
            dofs_bary_coords_.rightCols(local_dim) = dofs_phys_coords_;
            if constexpr (local_dim == 1) {
                for (int i = 0; i < dofs_bary_coords_.rows(); ++i) {
                    dofs_bary_coords_(i, 0) = 1 - dofs_bary_coords_(i, 1);
                }
            } else {
                dofs_bary_coords_.col(0) = (1 - dofs_bary_coords_.rowwise().sum().array()).matrix();
            }
        }
        // getters
        const SMatrix<n_dofs_per_cell, local_dim>& dofs_phys_coords() const { return dofs_phys_coords_; }
        const SMatrix<n_dofs_per_cell, local_dim + 1>& dofs_bary_coords() const { return dofs_bary_coords_; }
       private:
        SMatrix<n_dofs_per_cell, local_dim> dofs_phys_coords_;       // dofs physical coordinates over reference simplex
        SMatrix<n_dofs_per_cell, local_dim + 1> dofs_bary_coords_;   // dofs barycentric coordinates
    };

    template <int LocalDim> using cell_dof_descriptor = dof_descriptor<LocalDim>;
    template <int LocalDim> using face_dof_descriptor = dof_descriptor<LocalDim - 1>;
};
  
// lagrange finite element alias
[[maybe_unused]] static struct P1_ : FE_P<1> { } P1;
[[maybe_unused]] static struct P2_ : FE_P<2> { } P2;
[[maybe_unused]] static struct P3_ : FE_P<3> { } P3;
    
}   // namespace core
}   // namespace fdapde

#endif   // __FE_P_H__
