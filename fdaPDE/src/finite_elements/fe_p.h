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

#ifndef __FDAPDE_FE_P_H__
#define __FDAPDE_FE_P_H__

#include "header_check.h"

namespace fdapde {

namespace internals {

// implementation of vector lagrange space (FeP with NComponents > 1). basis are ordered so that, if, for instance
// LocalDim = 2, Order = 1, NComponents = 2, then \psi_0 = [\psi_0, 0], \psi_1 = [\psi_1, 0], \psi_2 = [\psi_2, 0],
// \psi_3 = [0, \psi_0], ... in accordance to how the dof handler replicates its dofs if dof_multiplicity > 1
template <int LocalDim, int Order, int NComponents> struct vector_fe_p_basis_type {
    static constexpr int n_basis = LagrangeBasis<LocalDim, Order>::n_basis * NComponents;

    // implements \boldsymbol{\psi}_i = \psi_{i_ % n_components} \iff i == (i \ n_components), 0 otherwise
    struct PolynomialType : public MatrixFieldBase<LocalDim, PolynomialType> {
       private:
        // the i-th component of the basis vector
        struct Component : public ScalarFieldBase<LocalDim, Component> {
            using Scalar = double;
            using InputType = Vector<Scalar, LocalDim>;
            static constexpr int StaticInputSize = LocalDim;
            static constexpr int NestAsRef = 0;
            static constexpr int XprBits = 0;
            constexpr Component() : xpr_(nullptr), i_(0) { }
            constexpr Component(const PolynomialType* xpr, int i) : xpr_(xpr), i_(i) { }
            constexpr Scalar operator()(const InputType& p) const { return xpr_->eval(i_, p); }
            constexpr int input_size() const { return StaticInputSize; }
           private:
            const PolynomialType* xpr_;
            int i_;
        };
       public:
        using Base = MatrixFieldBase<LocalDim, PolynomialType>;
        using Scalar = double;
        using InputType = Vector<Scalar, LocalDim>;
        static constexpr int StaticInputSize = LocalDim;
        static constexpr int NestAsRef = 1;
        static constexpr int XprBits = 0;
        static constexpr int ReadOnly = 1;
        static constexpr int Rows = NComponents;
        static constexpr int Cols = 1;

        constexpr PolynomialType() = default;
        template <int n_nodes>
            requires(n_nodes == LagrangeBasis<LocalDim, Order>::n_basis)
        constexpr PolynomialType(const Matrix<double, n_nodes, LocalDim>& nodes, int i) :
            basis_(nodes), i_(i) { }
        constexpr Component operator[](int i) const { return Component(this, i); }
        constexpr Scalar eval(int i, [[maybe_unused]] int j, const InputType& p) const {
            constexpr int n_basis_ = LagrangeBasis<LocalDim, Order>::n_basis;
            return (i == (i_ / n_basis_)) ? basis_[i_ % n_basis_](p) : Scalar(0);
        }
        constexpr Scalar eval(int i, const InputType& p) const { return eval(i, 0, p); }
        // evaluation at point
        constexpr auto operator()(const InputType& p) const { return Base::call_(p); }
        constexpr int rows() const { return Rows; }
        constexpr int cols() const { return Cols; }
        constexpr int input_size() const { return StaticInputSize; }
        constexpr int size() const { return Rows * Cols; }
       private:
        LagrangeBasis<LocalDim, Order> basis_;
        int i_;   // the i_-th vector basis function [0, \ldots, \psi_{i_ % n_basis}, \ldots, 0]
    };
    constexpr vector_fe_p_basis_type() = default;
    template <int n_nodes>
        requires(n_nodes == LagrangeBasis<LocalDim FDAPDE_COMMA Order>::n_basis)
    constexpr explicit vector_fe_p_basis_type(const Matrix<double, n_nodes, LocalDim>& nodes) : basis_() {
        for (int i = 0; i < n_basis; ++i) { basis_[i] = PolynomialType(nodes, i); }
    }
    // getters
    constexpr const PolynomialType& operator[](int i) const { return basis_[i]; }
    constexpr int size() const { return n_basis; }
   private:
    std::array<PolynomialType, n_basis> basis_;
};

}   // namespace internals

// representation of the finite element space P_h^K = { v \in H^1(D) : v_{e} \in P^K \forall e \in T_h }
template <int Order, int NComponents> struct FeP {  
    static constexpr int order = Order;
    static constexpr int n_components = NComponents;
    static constexpr bool is_vector_fe = (n_components != 1);
    fdapde_static_assert(n_components > 0, DEFINITION_OF_FINITE_ELEMENT_WITH_ZERO_OR_LESS_COMPONENTS_IS_ILL_FORMED);
    static constexpr bool is_hess_zero = Order <= 1;

    // dummy template argument enables template specialization of this nested template
    template <int LocalDim, typename dummy = void> struct dof_descriptor {
        static constexpr int local_dim = LocalDim;
        using ReferenceCell = Simplex<local_dim, local_dim>;   // reference unit simplex
        static constexpr bool dof_sharing = true;              // piecewise continuous finite element
        static constexpr int fe_order = Order;
        static constexpr int n_dofs_per_node = 1;
        static constexpr int n_dofs_per_edge = local_dim > 1 ? (Order - 1 < 0 ? 0 : (Order - 1)) : 0;
        static constexpr int n_dofs_per_face = local_dim > 2 ? (Order - 2 < 0 ? 0 : (Order - 1) * (Order - 2) / 2) : 0;
        static constexpr int n_dofs_internal = []() {
            if (local_dim == 1) return Order < 1 ? 0 : Order - 1;
            if (local_dim == 2) return Order < 2 ? 0 : (Order - 1) * (Order - 2) / 2;
            if (local_dim == 3) {
                if (Order < 3) return 0;
                int n_dofs_internal_ = 0;
                for (int i = 1; i < Order - 2; ++i) { n_dofs_internal_ += (i + 1) * i / 2; }
                return n_dofs_internal_;
            }
        }();
        static constexpr int n_dofs_per_cell = n_dofs_per_node * ReferenceCell::n_nodes +
                                               n_dofs_per_edge * ReferenceCell::n_edges +
                                               n_dofs_per_face * ReferenceCell::n_faces + n_dofs_internal;
        static constexpr int dof_multiplicity = n_components;
        using BasisType = std::conditional_t<
          NComponents == 1, LagrangeBasis<LocalDim, Order>,
          internals::vector_fe_p_basis_type<LocalDim, Order, NComponents>>;

        constexpr dof_descriptor() : dofs_phys_coords_(), dofs_bary_coords_() {
            // compute dofs physical coordinates on reference cell
            constexpr int n_nodes = local_dim + 1;
            Matrix<double, local_dim, n_nodes> reference_simplex;
            reference_simplex.setZero();
            for (int i = 0; i < local_dim; ++i) { reference_simplex(i, i + 1) = 1; }
            dofs_phys_coords_.template topRows<n_nodes>() = reference_simplex.transpose();
            int j = n_nodes;
            // constexpr enumeration of reference simplex edges
            auto edge_enumerate = [&, this]() {
                std::vector<bool> bitmask(n_nodes, 0);
                std::fill_n(bitmask.begin(), 2, 1);
                Matrix<double, local_dim, 2> edge_coords;
                for (int i = 0; i < ReferenceCell::n_edges; ++i) {
                    for (int k = 0, h = 0; k < n_nodes; ++k) {
		      if (bitmask[k]) { edge_coords.col(h++) = reference_simplex.col(k); }
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
                    // add triangle of equidistant nodes having Order - 2 nodes per side
                    double step = 1.0 / Order;
                    for (int k = 0; k < Order - 2; ++k) {
                        for (int h = 0; h < Order - 2 - k; ++h) {
                            dofs_phys_coords_.row(j++) = Matrix<double, 1, 2>(step * (k + 1), step * (h + 1));
                        }
                    }
                }
            }
            if constexpr (local_dim == 3) {
                if constexpr (n_dofs_per_edge > 0) { edge_enumerate(); }
                if constexpr (n_dofs_per_face > 0) {
                    // add triangle of equidistant nodes having Order - 2 nodes per side
                    double step = 1.0 / Order;
		    // compute barycentric coordinates of nodes to insert on faces
                    Matrix<double, n_dofs_per_face, 2> bary_coords;
                    for (int k = 0, r = 0; k < Order - 2; ++k) {
                        for (int h = 0; h < Order - 2 - k; ++h) {
                            bary_coords.row(r++) = Matrix<double, 1, 2>(step * (k + 1), step * (h + 1));
                        }
                    }
                    std::vector<bool> bitmask(n_nodes, 0);
                    std::fill_n(bitmask.begin(), ReferenceCell::n_nodes_per_face, 1);
                    Matrix<double, local_dim, 3> face_coords;
                    for (int i = 0; i < ReferenceCell::n_faces; ++i) {
                        for (int k = 0, h = 0; k < n_nodes; ++k) {
                            if (bitmask[k]) face_coords.col(h++) = reference_simplex.col(k);
                        }
                        Matrix<double, local_dim, 2> J;
                        for (int k = 0; k < 2; ++k) { J.col(k) = face_coords.col(k + 1) - face_coords.col(0); }
                        for (int k = 0; k < n_dofs_per_face; ++k) {
                            dofs_phys_coords_.row(j++) =
                              (J * bary_coords.row(k).transpose() + face_coords.col(0)).transpose();
                        }
                        std::prev_permutation(bitmask.begin(), bitmask.end());
                    }
                }
                if constexpr (n_dofs_internal > 0) {
                    // add 3D simplex-shaped grid of equidistant points with Order - 3 nodes per side
                    double step = 1.0 / Order;
                    for (int l = 0; l < Order - 3; ++l) {   // z-axis
                        for (int k = 0; k < Order - 3 - l; ++k) {
                            for (int h = 0; h < Order - 3 - k; ++h) {
                                dofs_phys_coords_.row(j++) =
                                  Matrix<double, 1, 3>(step * (k + 1), step * (h + 1), step * (l + 1));
                            }
                        }
                    }
                }
            }
            // compute barycentric coordinates
            dofs_bary_coords_.template rightCols<local_dim>() = dofs_phys_coords_;
            if constexpr (local_dim == 1) {
                for (int i = 0; i < dofs_bary_coords_.rows(); ++i) {
                    dofs_bary_coords_(i, 0) = 1 - dofs_bary_coords_(i, 1);
                }
            } else {
                for (int i = 0; i < n_dofs_per_cell; ++i) {
                    double sum = 0;
                    for (int j = 0; j < local_dim; ++j) sum += dofs_bary_coords_(i, j + 1);
                    dofs_bary_coords_(i, 0) = 1 - sum;
                }
            }
        }
        // getters
        constexpr const auto& dofs_phys_coords() const { return dofs_phys_coords_; }
        constexpr const auto& dofs_bary_coords() const { return dofs_bary_coords_; }
       private:
        Matrix<double, n_dofs_per_cell, local_dim> dofs_phys_coords_;       // dofs physical coordinates
        Matrix<double, n_dofs_per_cell, local_dim + 1> dofs_bary_coords_;   // dofs barycentric coordinates
    };
    // template specialization for boundary integration of 1D elements
    template <typename dummy> struct dof_descriptor<0, dummy> {
        static constexpr int local_dim = 0;
        using ReferenceCell = Simplex<local_dim, local_dim>;
        static constexpr bool dof_sharing = true;
        static constexpr int fe_order = Order;
        static constexpr int n_dofs_per_node = 1;
        static constexpr int n_dofs_per_edge = 0;
        static constexpr int n_dofs_per_face = 0;
        static constexpr int n_dofs_internal = 0;
        static constexpr int n_dofs_per_cell = 1;
        static constexpr int dof_multiplicity = n_components;
        using BasisType = std::conditional_t<
          NComponents == 1, LagrangeBasis<0, Order>, internals::vector_fe_p_basis_type<0, Order, NComponents>>;
    };
  
    // select quadrature which optimally integrates (Order + 1) polynomials
    template <int LocalDim> class select_cell_quadrature {
        static constexpr int select_quadrature_() {
            if (LocalDim == 1) return Order == 1 ? 2 : (Order == 2 ? 3  : 4 );
            if (LocalDim == 2) return Order == 1 ? 3 : (Order == 2 ? 6  : 12);
            if (LocalDim == 3) return Order == 1 ? 4 : (Order == 2 ? 11 : 24);
        }
       public:
        using type = internals::fe_quadrature_simplex<LocalDim, select_quadrature_()>;
    };
    template <int LocalDim> using cell_dof_descriptor = dof_descriptor<LocalDim>;
    template <int LocalDim> using face_dof_descriptor = dof_descriptor<LocalDim - 1>;
    template <int LocalDim> using cell_quadrature_t = typename select_cell_quadrature<LocalDim>::type;
    template <int LocalDim> using face_quadrature_t = typename select_cell_quadrature<LocalDim - 1>::type;
};

// template specialization for P0 elements
template <int NComponents> struct FeP<0, NComponents> {
    static constexpr int order = 0;
    static constexpr int n_components = NComponents;
    static constexpr bool is_vector_fe = (n_components != 1);
    fdapde_static_assert(n_components > 0, DEFINITION_OF_FINITE_ELEMENT_WITH_ZERO_OR_LESS_COMPONENTS_IS_ILL_FORMED);
  
    template <int LocalDim> struct dof_descriptor {
        static constexpr int local_dim = LocalDim;
        using ReferenceCell = Simplex<local_dim, local_dim>;   // reference unit simplex
        static constexpr bool dof_sharing = false;             // discontinuous finite element
        static constexpr int fe_order = 0;
        static constexpr int n_dofs_per_node = 0;
        static constexpr int n_dofs_per_edge = 0;
        static constexpr int n_dofs_per_face = 0;
        static constexpr int n_dofs_internal = 1;
        static constexpr int n_dofs_per_cell = n_dofs_internal;
        static constexpr int dof_multiplicity = n_components;
        using BasisType = std::conditional_t<
          NComponents == 1, LagrangeBasis<LocalDim, 0>, internals::vector_fe_p_basis_type<LocalDim, 0, NComponents>>;

        constexpr dof_descriptor() : dofs_phys_coords_(), dofs_bary_coords_() {
            // compute dofs physical coordinates on reference cell
            constexpr int n_nodes = local_dim + 1;
            Matrix<double, local_dim, n_nodes> reference_simplex;
            reference_simplex.setZero();
            for (int i = 0; i < local_dim; ++i) { reference_simplex(i, i + 1) = 1; }
	    // the unique dof is the simplex barycenter
            if constexpr (local_dim == 1) { dofs_phys_coords_.row(0) = Vector<double, 1>(0.5); }
            if constexpr (local_dim == 2) {
                dofs_phys_coords_.row(0) =
                  (reference_simplex.col(1) + reference_simplex.col(2)).transpose() * 1.0 / (local_dim + 1);
            }
            if constexpr (local_dim == 3) {
                dofs_phys_coords_.row(0) =
                  (reference_simplex.col(1) + reference_simplex.col(2) + reference_simplex.col(3)).transpose() * 1.0 /
                  (local_dim + 1);
            }
            // compute barycentric coordinates
            dofs_bary_coords_.template rightCols<local_dim>(1) = dofs_phys_coords_;
            if constexpr (local_dim == 1) {
                dofs_bary_coords_(0, 0) = 0.5;
            } else {
                double sum = 0;
                for (int j = 0; j < local_dim; ++j) sum += dofs_bary_coords_(0, j + 1);
                dofs_bary_coords_(0, 0) = 1 - sum;
            }
        }
        // getters
        constexpr const auto& dofs_phys_coords() const { return dofs_phys_coords_; }
        constexpr const auto& dofs_bary_coords() const { return dofs_bary_coords_; }
       private:
        Matrix<double, n_dofs_per_cell, local_dim> dofs_phys_coords_;       // dofs physical coordinates
        Matrix<double, n_dofs_per_cell, local_dim + 1> dofs_bary_coords_;   // dofs barycentric coordinates
    };
    template <int LocalDim>
    struct select_cell_quadrature : std::type_identity<internals::fe_quadrature_simplex<LocalDim, 1>> { };
    template <int LocalDim> using cell_dof_descriptor = dof_descriptor<LocalDim>;
    template <int LocalDim> using face_dof_descriptor = dof_descriptor<LocalDim - 1>;
    template <int LocalDim> using cell_quadrature_t = typename select_cell_quadrature<LocalDim>::type;
    template <int LocalDim> using face_quadrature_t = typename select_cell_quadrature<LocalDim - 1>::type;
};

// lagrange finite element aliases (N: number of components, N > 1 \implies vector finite elements)
template <int N> constexpr FeP<0, N> P0 = FeP<0, N> {};
template <int N> constexpr FeP<1, N> P1 = FeP<1, N> {};
template <int N> constexpr FeP<2, N> P2 = FeP<2, N> {};
template <int N> constexpr FeP<3, N> P3 = FeP<3, N> {};
template <int N> constexpr FeP<4, N> P4 = FeP<4, N> {};
template <int N> constexpr FeP<5, N> P5 = FeP<5, N> {};
  
}   // namespace fdapde

#endif   // __FDAPDE_FE_P_H__
