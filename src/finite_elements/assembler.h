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

#ifndef __ASSEMBLER_H__
#define __ASSEMBLER_H__

#include <memory>

#include "../utils/symbols.h"
#include "../utils/compile_time.h"
#include "../fields/vector_field.h"
#include "../fields/scalar_field.h"
#include "../fields/field_ptrs.h"
#include "../mesh/mesh.h"
#include "../mesh/element.h"
#include "integration/integrator.h"
#include "basis/lagrangian_basis.h"
#include "basis/multivariate_polynomial.h"
#include "basis/basis_cache.h"
#include "operators/bilinear_form_traits.h"

namespace fdapde {
namespace core {

// finite element method assembly loop
template <unsigned int M, unsigned int N, unsigned int R, typename B, typename I> class Assembler {
   private:
    constexpr static unsigned n_basis = ct_nnodes(M, R);
    const Mesh<M, N, R>& mesh_;   // mesh
    const I& integrator_;         // quadrature rule used in integrals approzimation
    B reference_basis_ {};        // functional basis over reference N-dimensional unit simplex
    std::size_t dof_;             // overall number of unknowns in the FEM linear system
    const Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& dof_table_;
   public:
    Assembler(const Mesh<M, N, R>& mesh, const I& integrator) :
        mesh_(mesh), integrator_(integrator), dof_(mesh_.dof()), dof_table_(mesh.dof_table()) {};

    // discretization methods
    template <typename E> SpMatrix<double> discretize_operator(const E& op);
    template <typename F> DVector<double>  discretize_forcing (const F& force);
};

// implementative details
  
// assembly for the discretization matrix of a general bilinear form L
template <unsigned int M, unsigned int N, unsigned int R, typename B, typename I>
template <typename E>
SpMatrix<double> Assembler<M, N, R, B, I>::discretize_operator(const E& op) {
    std::vector<Eigen::Triplet<double>> triplet_list;   // store triplets (node_i, node_j, integral_value)
    SpMatrix<double> discretization_matrix;

    // properly preallocate memory to avoid reallocations
    triplet_list.reserve(n_basis * mesh_.elements());
    discretization_matrix.resize(dof_, dof_);

    // prepare space for bilinear form components
    using BasisType = typename B::ElementType;
    using NablaType = decltype(std::declval<BasisType>().derive());
    BasisType buff_psi_i, buff_psi_j;               // basis functions \psi_i, \psi_j
    NablaType buff_nabla_psi_i, buff_nabla_psi_j;   // gradient of basis functions \nabla \psi_i, \nabla \psi_j
    MatrixConst<M, N, M> buff_invJ;   // (J^{-1})^T, being J the inverse of the barycentric matrix relative to element e
    // prepare buffer to be sent to bilinear form
    auto mem_buffer = std::make_tuple(
      ScalarPtr(&buff_psi_i), ScalarPtr(&buff_psi_j), VectorPtr(&buff_nabla_psi_i), VectorPtr(&buff_nabla_psi_j),
      MatrixPtr(&buff_invJ));

    // develop bilinear form expression in an integrable field here once
    auto f = op.integrate(mem_buffer);   // let the compiler deduce the type of the expression template!

    std::size_t current_id;
    // cycle over all mesh elements
    for (const auto& e : mesh_) {
        // update elements related informations: current ID and the affine map from current element to reference element
        buff_invJ = e.inv_barycentric_matrix().transpose();
        current_id = e.ID();
        // consider all pair of nodes
        for (size_t i = 0; i < n_basis; ++i) {
            buff_psi_i = reference_basis_[i];
            buff_nabla_psi_i = buff_psi_i.derive();   // update buffers content
            for (size_t j = 0; j < n_basis; ++j) {
                buff_psi_j = reference_basis_[j];
                buff_nabla_psi_j = buff_psi_j.derive();   // update buffers content
                if constexpr (is_symmetric<decltype(op)>::value) {
                    // compute only half of the discretization matrix if the operator is symmetric
                    if (dof_table_(current_id, i) >= dof_table_(current_id, j)) {
                        double value = integrator_.template integrate<decltype(op)>(e, f);
			
                        // linearity of the integral is implicitlu used during matrix construction, since duplicated
                        // triplets are summed up, see Eigen docs for more details
                        triplet_list.emplace_back(dof_table_(current_id, i), dof_table_(current_id, j), value);
                    }
                } else {
                    // not any optimization to perform in the general case
                    double value = integrator_.template integrate<decltype(op)>(e, f);
                    triplet_list.emplace_back(dof_table_(current_id, i), dof_table_(current_id, j), value);
                }
            }
        }
    }
    // matrix assembled
    discretization_matrix.setFromTriplets(triplet_list.begin(), triplet_list.end());
    discretization_matrix.makeCompressed();

    // return just half of the discretization matrix if the form is symmetric (lower triangular part)
    if constexpr (is_symmetric<decltype(op)>::value)
        return discretization_matrix.selfadjointView<Eigen::Lower>();
    else
        return discretization_matrix;
};

template <unsigned int M, unsigned int N, unsigned int R, typename B, typename I>
template <typename F>
DVector<double> Assembler<M, N, R, B, I>::discretize_forcing(const F& f) {
    // allocate space for result vector
    DVector<double> discretization_vector {};
    discretization_vector.resize(dof_, 1);   // there are as many basis functions as degrees of freedom on the mesh
    discretization_vector.fill(0);           // init result vector to zero

    // build forcing vector
    for (const auto& e : mesh_) {
        for (size_t i = 0; i < n_basis; ++i) {
            // integrate \int_e [f*\psi], exploit integral linearity
            discretization_vector[dof_table_(e.ID(), i)] += integrator_.integrate(e, f, reference_basis_[i]); 
        }
    }
    return discretization_vector;
}

}   // namespace core
}   // namespace fdapde

#endif   // __ASSEMBLER_H__
