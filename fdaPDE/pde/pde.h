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

#ifndef __PDE_H__
#define __PDE_H__

#include <type_traits>
#include <unordered_map>

#include "../mesh/mesh.h"
#include "../utils/symbols.h"
#include "../utils/integration/integrator.h"
#include "differential_expressions.h"
#include "../utils/type_erasure.h"

namespace fdapde {
namespace core {

// for a given resolution strategy S and operator E, selects a proper solver.
// to be partially specialized with respect to S
template <typename S, typename D, typename E, typename F, typename... Ts> struct pde_solver_selector { };

// Description of a Partial Differential Equation Lf = u solved with strategy S
template <typename D,     // domain's triangulation
          typename E,     // differential operator L
          typename F,     // forcing term u
          typename S,     // resolution strategy
          typename... Ts> // parameters forwarded to S
class PDE {
   public:
    typedef D DomainType;   // triangulated domain
    static constexpr int M = DomainType::local_dimension;
    static constexpr int N = DomainType::embedding_dimension;
    typedef E OperatorType;   // differential operator in its strong-formulation
    static_assert(
      std::is_base_of<DifferentialExpr<OperatorType>, OperatorType>::value, "E is not a valid differential operator");
    typedef F ForcingType;   // type of forcing object (either a matrix or a callable object)
    static_assert(
      std::is_same<DMatrix<double>, F>::value || std::is_base_of<ScalarExpr<D::embedding_dimension, F>, F>::value,
      "forcing is not a matrix or a scalar expression || N != F::base");
    typedef typename pde_solver_selector<S, D, E, F, Ts...>::type SolverType;
    typedef typename SolverType::ReferenceBasis ReferenceBasis;   // function space approximating the solution space
    typedef typename SolverType::Quadrature Quadrature;           // quadrature for numerical integral approximations

    // minimal constructor, use below setters to complete the construction of a PDE object
    PDE(const D& domain) : domain_(domain) { }
    PDE(const D& domain, E diff_op) : domain_(domain), diff_op_(diff_op) { };
    void set_forcing(const F& forcing_data) { forcing_data_ = forcing_data; }
    void set_differential_operator(E diff_op) { diff_op_ = diff_op; }
    // full constructors
    PDE(const D& domain, E diff_op, const F& forcing_data) :
        domain_(domain), diff_op_(diff_op), forcing_data_(forcing_data) { }

    // setters
    void set_dirichlet_bc(const DMatrix<double>& data) { boundary_data_ = data; }
    void set_initial_condition(const DVector<double>& data) { initial_condition_ = data; };

    // getters
    const DomainType& domain() const { return domain_; }
    OperatorType differential_operator() const { return diff_op_; }
    const ForcingType& forcing_data() const { return forcing_data_; }
    const DVector<double>& initial_condition() const { return initial_condition_; }
    const DMatrix<double>& boundary_data() const { return boundary_data_; };
    const Quadrature& integrator() const { return solver_.integrator(); }
    const ReferenceBasis& reference_basis() const { return solver_.reference_basis(); }

    template <template <typename> typename EvaluationPolicy>
    std::pair<SpMatrix<double>, DVector<double>> eval_functional_basis(const DMatrix<double>& locs) const {
        return solver_.basis().template eval<EvaluationPolicy>(locs);
    }
    std::size_t n_dofs() const { return solver_.n_dofs(); }
    const DMatrix<double>& solution() const { return solver_.solution(); };   // PDE solution
    const DMatrix<double>& force() const { return solver_.force(); };         // rhs of discrete linear system
    const SpMatrix<double>& R1() const { return solver_.R1(); };              // stiff matrix
    const SpMatrix<double>& R0() const { return solver_.R0(); };              // mass matrix
    DMatrix<double> dof_coords() { return solver_.dofs_coords(domain_); }
    DMatrix<double> quadrature_nodes() const { return integrator().quadrature_nodes(domain_); };
    void init() { solver_.init(*this); };   // initializes the solver
    void solve() {                          // solves the PDE
        if (!is_empty(boundary_data_)) solver_.set_dirichlet_bc(*this);
        solver_.solve(*this);
    }
   private:
    const DomainType& domain_;               // triangulated problem domain
    OperatorType diff_op_;                   // differential operator in its strong formulation
    ForcingType forcing_data_;               // forcing data
    DVector<double> initial_condition_ {};   // initial condition, (for space-time problems only)
    SolverType solver_ {};                   // problem solver
    DMatrix<double> boundary_data_;          // boundary conditions
};

// PDE-detection type trait
template <typename T> struct is_pde {
    static constexpr bool value = fdapde::is_instance_of<T, PDE>::value;
};

enum eval { pointwise, areal };
template <typename T> struct pointwise_evaluation;
template <typename T> struct areal_evaluation;

// type-erasure wrapper (forcing type must be convertible to a DMatrix<double>)
struct I_PDE {
    void init()  { invoke<void, 0>(*this); }
    void solve() { invoke<void, 1>(*this); }
    // getters
    decltype(auto) solution()         const { return invoke<const DMatrix<double>& , 2>(*this); }
    decltype(auto) force()            const { return invoke<const DMatrix<double>& , 3>(*this); }
    decltype(auto) R1()               const { return invoke<const SpMatrix<double>&, 4>(*this); }
    decltype(auto) R0()               const { return invoke<const SpMatrix<double>&, 5>(*this); }
    decltype(auto) quadrature_nodes() const { return invoke<DMatrix<double>        , 6>(*this); }
    decltype(auto) n_dofs()           const { return invoke<std::size_t            , 7>(*this); }
    decltype(auto) dof_coords()       const { return invoke<DMatrix<double>        , 8>(*this); }
    decltype(auto) forcing_data()     const { return invoke<const DMatrix<double>& , 9>(*this); }
  
    struct eval_basis_ret_type { SpMatrix<double> Psi; DVector<double> D; };
    std::optional<eval_basis_ret_type> eval_basis(eval e, const DMatrix<double>& locs) const {
        using RetType = eval_basis_ret_type;
        switch (e) {   // run-time switch based on sampling strategy
        case eval::pointwise:
            return std::optional<RetType>{invoke<RetType, 10>(*this, locs)};
        case eval::areal:
	    return std::optional<RetType>{invoke<RetType, 11>(*this, locs)};
        }
        return std::nullopt;
    }
    // setters
    template <typename F> void set_forcing(const F& data) { fdapde::invoke<void, 12>(*this, DMatrix<double>(data)); }
    void set_dirichlet_bc(const DMatrix<double>& data) { fdapde::invoke<void, 13>(*this, data); }
    void set_initial_condition(const DVector<double>& data) { fdapde::invoke<void, 14>(*this, data); }
    template <typename E> void set_differential_operator(E diff_op) { fdapde::invoke<void, 15>(*this, diff_op); }

    // function pointers forwardings
    template <typename T>
    using fn_ptrs = fdapde::mem_fn_ptrs<
      &T::init, &T::solve,   // initialization and pde solution
      // getters
      &T::solution, &T::force, &T::R1, &T::R0, &T::quadrature_nodes, &T::n_dofs, &T::dof_coords, &T::forcing_data,
      &T::template eval_functional_basis<pointwise_evaluation>, &T::template eval_functional_basis<areal_evaluation>,
      // setters
      &T::set_forcing, &T::set_dirichlet_bc, &T::set_initial_condition, &T::set_differential_operator>;
};
using pde_ptr = fdapde::erase<fdapde::heap_storage, I_PDE>; // type-erased wrapper for PDEs

// factory method
template <typename... Args_, typename... Args> pde_ptr make_pde(Args&&... args) {
    return pde_ptr(PDE<Args_...>(std::forward<Args>(args)...));
}
  
}   // namespace core
}   // namespace fdapde

#endif   // __PDE_H__
