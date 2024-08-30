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

#ifndef __DOF_CONSTRAINTS_H__
#define __DOF_CONSTRAINTS_H__

#include "../linear_algebra/binary_matrix.h"

namespace fdapde {

template <int LocalDim, int EmbedDim> class DofHandler;

// managment of affine constraints on degrees of freedom of type \sum_{j} c_ij * dof_ij = b_j
template <int LocalDim, int EmbedDim> class DofConstraints {
   public:
    static constexpr int local_dim = LocalDim;
    static constexpr int embed_dim = EmbedDim;
    static constexpr double eps = 1e30;

    DofConstraints() = default;
    DofConstraints(const DofHandler<local_dim, embed_dim>& dof_handler) : dof_handler_(&dof_handler) { }

    // guarantees that the linear system Ax = b is such that all (affine) constraints are respected
    void enforce_constraints(SpMatrix<double>& A) const {
	BinaryMatrix<Dynamic> A_mask(A.rows(), A.cols());
        for (const fdapde::Triplet<double>& triplet : constraint_pattern_) {
            if (!A_mask(triplet.row(), triplet.col())) {   // guaratees only the first constraint is set
                A.coeffRef(triplet.row(), triplet.col()) = triplet.value() * eps;
                A_mask.set(triplet.row(), triplet.col());
            }
        }
        return;
    }
    void enforce_constraints(DVector<double>& b) const {
	BinaryVector<Dynamic> b_mask(b.rows());
        for (const fdapde::Duplet<double>& duplet : constraint_values_) {
            if (!b_mask[duplet.row()]) {
                b[duplet.row()] = duplet.value() * eps;
                b_mask.set(duplet.row());
            }
        }
        return;
    }
    void enforce_constraints(SpMatrix<double>& A, DVector<double>& b) const {
        fdapde_assert(A.rows() == b.rows());
	enforce_constraints(A);
	enforce_constraints(b);
        return;
    }
    // set dirichlet constraint type on boundary nodes marked on (diagonal constraint pattern)
    template <typename Callable> void set_dirichlet_constraint(int on, const Callable& g) {
        int n_boundary_dofs = dof_handler_->n_boundary_dofs(on);
        fdapde_assert(n_boundary_dofs > 0);
        for (typename DofHandler<local_dim, embed_dim>::boundary_dofs_iterator it =
               dof_handler_->boundary_dofs_begin(on);
             it != dof_handler_->boundary_dofs_end(on); ++it) {
            int dof_id = it->id();
            constraint_pattern_.emplace_back(dof_id, dof_id, 1.0);
            constraint_values_.emplace_back(dof_id, g(it->coord()));
        }
        return;
    }
    template <typename Iterator> void set_pattern_from_triplets(Iterator begin, Iterator end) {
        fdapde_static_assert(
          std::is_same_v<std::decay_t<typename Iterator::value_type> FDAPDE_COMMA fdapde::Triplet<double>>,
          INVALID_ITERATOR_VALUE_TYPE_FOR_CONSTRAINT_PATTERN);
        constraint_pattern_.push_back(constraint_pattern_.end(), begin, end);
        return;
    }
    template <typename Iterator> void set_values_from_duplets(Iterator begin, Iterator end) {
        fdapde_static_assert(
          std::is_same_v<std::decay_t<typename Iterator::value_type> FDAPDE_COMMA fdapde::Duplet<double>>,
          INVALID_ITERATOR_VALUE_TYPE_FOR_CONSTRAINT_VALUES);
        constraint_values_.push_back(constraint_values_.end(), begin, end);
        return;
    }
   private:
    const DofHandler<local_dim, embed_dim>* dof_handler_;
    std::vector<fdapde::Triplet<double>> constraint_pattern_;   // triplets (i, j, c_ij)
    std::vector<fdapde::Duplet<double>>  constraint_values_;    // pairs (i, b_i)
};

}   // namespace fdapde

#endif   // __DOF_CONSTRAINTS_H__
