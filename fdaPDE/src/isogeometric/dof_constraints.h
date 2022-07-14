#ifndef __FDAPDE_ISO_DOF_CONSTRAINTS_H__
#define __FDAPDE_ISO_DOF_CONSTRAINTS_H__

#include "header_check.h"

namespace fdapde {



template <typename DofHandler, typename DiscretizationCategory> class DofConstraints;


// Management of affine constraints of the form ∑_j c_ij * dof_j = b_i
template <typename DofHandler> class DofConstraints<DofHandler, iso_tag> {
    // Pair-like object for constraint values
        template <typename Scalar_> class Duplet {
        public:
        using Index = int;
        using Scalar = Scalar_;
        Duplet() : row_(0), value_() { }
        Duplet(int row, const Scalar& value) : row_(row), value_(value) { }

        Index row() const { return row_; }
        const Scalar& value() const { return value_; }
        Scalar& value() { return value_; }

        private:
        Index row_;
        Scalar value_;
        };
    public:
    using DofHandlerType = std::decay_t<DofHandler>;
    static constexpr int local_dim = DofHandlerType::local_dim;
    static constexpr int embed_dim = DofHandlerType::embed_dim;
    static constexpr double eps = 1e30;

    DofConstraints() = default;
    DofConstraints(const DofHandlerType& dof_handler) : dof_handler_(&dof_handler) { }

    // Enforce constraints on vector or matrix
    template <typename T>
    void enforce_constraints(T&& t) const {
        if constexpr (internals::is_subscriptable<std::decay_t<T>, int>) {
            for (const Duplet<double>& duplet : constraint_values_) {
                t[duplet.row()] = duplet.value() * eps;
            }
        } else {
            for (const Triplet<double>& triplet : constraint_pattern_) {
                //std::cout<<"Triplet value: "<<triplet.value()<<std::endl;
                t.coeffRef(triplet.row(), triplet.col()) = triplet.value() * eps;
            }
        }
    }

    template <typename SystemMatrix, typename SystemRhs>
    void enforce_constraints(SystemMatrix&& A, SystemRhs&& b) const {
        fdapde_assert(A.rows() == b.rows());
        enforce_constraints(A);
        enforce_constraints(b);
    }

    // Set Dirichlet constraint: u = 0 on marker
    void set_hom_dirichlet_constraint(int marker = BoundaryAll) { // for Callable g it can be done :)
        int n_boundary_dofs = dof_handler_->n_boundary_dofs(marker);
         fdapde_assert(marker == BoundaryAll || n_boundary_dofs > 0);

        for (typename DofHandlerType::boundary_dofs_iterator it = dof_handler_->boundary_dofs_begin(marker);
            it != dof_handler_->boundary_dofs_end(marker); ++it) {
            int dof_id = it->id();
            constraint_pattern_.emplace_back(dof_id, dof_id, 1.0);  // fix DOF
            constraint_values_.emplace_back(dof_id, 0.0);           // to 0
        }
    }

    void set_clamped_hom_constraint() {
        // Set Dirichlet constraint: u = 0 on marker, including adjacent boundaries
        int n_dofs = dof_handler_->n_dofs();
        for (int i = 0; i < n_dofs; ++i) {
            if(dof_handler_->is_dof_on_boundary(i) || dof_handler_->is_dof_on_adjacent_boundary(i)) {
                constraint_pattern_.emplace_back(i, i, 1.0); // fix DOF
                constraint_values_.emplace_back(i, 0.0);    // to 0
            }
        }
    }

    // Set periodic constraint: dof_slave = dof_master
    void set_master_slave_constraint(int master_id, int slave_id) {
        constraint_pattern_.emplace_back(slave_id, slave_id, 1.0);   // +u_slave
        constraint_pattern_.emplace_back(slave_id, master_id, -1.0); // -u_master
        constraint_values_.emplace_back(slave_id, 0.0);              // = 0
    }

    private:
    const DofHandlerType* dof_handler_;
    std::vector<Triplet<double>> constraint_pattern_;
    std::vector<Duplet<double>> constraint_values_;
   
};


} // namespace fdapde

#endif // __FDAPDE_ISO_DOF_CONSTRAINTS_H__