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
#include <optional>

#include "../mesh/mesh.h"
#include "../utils/symbols.h"
#include "../utils/integration/integrator.h"
#include "differential_expressions.h"
#include "../utils/type_erasure.h"
#include "differential_operators.h"
#include "symbols.h"

namespace fdapde {
namespace core {

// Description of a Partial Differential Equation Lf = u solved with strategy S
template <typename D,     // problem domain
          typename E,     // differential operator L
          typename F,     // forcing term u
          typename S,     // resolution strategy
          typename... Ts> // parameters forwarded to S
class PDE {
   public:
    using SpaceDomainType = D;                     // triangulated spatial domain
    using TimeDomainType = DVector<double>;   // time-interval [0,T] (for time-dependent PDEs)
    static constexpr int M = SpaceDomainType::local_dimension;
    static constexpr int N = SpaceDomainType::embedding_dimension;
    using OperatorType = E;   // differential operator in its strong-formulation
    static_assert(
      std::is_base_of<DifferentialExpr<OperatorType>, OperatorType>::value);
    using ForcingType = F;   // type of forcing object (either a matrix or a callable object)
    static_assert(
      std::is_same<DMatrix<double>, ForcingType>::value ||
        std::is_base_of<ScalarExpr<SpaceDomainType::embedding_dimension, ForcingType>, ForcingType>::value);
    using SolverType = typename pde_solver_selector<S, SpaceDomainType, OperatorType, ForcingType, Ts...>::type;
    using FunctionalBasis = typename SolverType::FunctionalBasis;   // function space approximating the solution space
    using Quadrature = typename SolverType::Quadrature;             // quadrature for numerical integral approximations

    // space-only constructors
    fdapde_enable_constructor_if(is_stationary, OperatorType) PDE(const D& domain) :
        domain_(domain), solver_(domain) { }
    fdapde_enable_constructor_if(is_stationary, OperatorType) PDE(const D& domain, E diff_op) :
        domain_(domain), diff_op_(diff_op), solver_(domain) { }
    fdapde_enable_constructor_if(is_stationary, OperatorType)
      PDE(const SpaceDomainType& domain, OperatorType diff_op, const ForcingType& forcing_data) :
        domain_(domain), diff_op_(diff_op), forcing_data_(forcing_data), solver_(domain) { }
    // space-time constructors
    fdapde_enable_constructor_if(is_parabolic, OperatorType) PDE(const D& domain, const TimeDomainType& t) :
        domain_(domain), time_domain_(t), solver_(domain) { }
    fdapde_enable_constructor_if(is_parabolic, OperatorType) PDE(const D& domain, const TimeDomainType& t, E diff_op) :
        domain_(domain), time_domain_(t), diff_op_(diff_op), solver_(domain) { }
    fdapde_enable_constructor_if(is_parabolic, OperatorType) PDE(
      const SpaceDomainType& domain, const TimeDomainType& t, OperatorType diff_op, const ForcingType& forcing_data) :
        domain_(domain), time_domain_(t), diff_op_(diff_op), forcing_data_(forcing_data), solver_(domain) { }

    // setters
    void set_forcing(const ForcingType& forcing_data) { forcing_data_ = forcing_data; }
    void set_differential_operator(OperatorType diff_op) { diff_op_ = diff_op; }
    void set_dirichlet_bc(const DMatrix<double>& data) { boundary_data_ = data; }
    void set_initial_condition(const DVector<double>& data) { initial_condition_ = data; };
    // getters
    const SpaceDomainType& domain() const { return domain_; }
    const DVector<double>& time_domain() const { return time_domain_; }
    OperatorType differential_operator() const { return diff_op_; }
    const ForcingType& forcing_data() const { return forcing_data_; }
    const DVector<double>& initial_condition() const { return initial_condition_; }
    const DMatrix<double>& boundary_data() const { return boundary_data_; };
    const Quadrature& integrator() const { return solver_.integrator(); }
    const FunctionalBasis& basis() const { return solver_.basis(); }
    // evaluates the functional basis defined over the pyhisical domain on a given set of locations
    template <template <typename> typename EvaluationPolicy>
    std::pair<SpMatrix<double>, DVector<double>> eval_functional_basis(const DMatrix<double>& locs) const {
        return solver_.basis().template eval<EvaluationPolicy>(locs);
    }
    std::size_t n_dofs() const { return solver_.n_dofs(); }
    const DMatrix<double>& solution() const { return solver_.solution(); };   // PDE solution
    const DMatrix<double>& force() const { return solver_.force(); };         // rhs of discrete linear system
    const SpMatrix<double>& stiff() const { return solver_.stiff(); };        // stiff matrix
    const SpMatrix<double>& mass() const { return solver_.mass(); };          // mass matrix
    DMatrix<double> dof_coords() { return solver_.dofs_coords(); }
    DMatrix<double> quadrature_nodes() const { return integrator().quadrature_nodes(domain_); };
    void init() { solver_.init(*this); };   // initializes the solver
    void solve() {                          // solves the PDE
        if (!is_empty(boundary_data_)) solver_.set_dirichlet_bc(*this);
        solver_.solve(*this);
    }
   private:
    const SpaceDomainType& domain_;          // triangulated spatial domain
    const TimeDomainType time_domain_;       // time interval [0, T], for space-time PDEs
    OperatorType diff_op_;                   // differential operator in its strong formulation
    ForcingType forcing_data_;               // forcing data
    DVector<double> initial_condition_ {};   // initial condition, for space-time PDEs
    SolverType solver_ {};                   // problem solver
    DMatrix<double> boundary_data_;          // boundary conditions
};

// type-erasure wrapper (we require ForcingType_ to be convertible to a DMatrix<double>)
struct I_PDE {
    void init()  { invoke<void, 0>(*this); }
    void solve() { invoke<void, 1>(*this); }
    // getters
    decltype(auto) solution()          const { return invoke<const DMatrix<double>& , 2>(*this); }
    decltype(auto) force()             const { return invoke<const DMatrix<double>& , 3>(*this); }
    decltype(auto) stiff()             const { return invoke<const SpMatrix<double>&, 4>(*this); }
    decltype(auto) mass()              const { return invoke<const SpMatrix<double>&, 5>(*this); }
    decltype(auto) quadrature_nodes()  const { return invoke<DMatrix<double>        , 6>(*this); }
    decltype(auto) n_dofs()            const { return invoke<std::size_t            , 7>(*this); }
    decltype(auto) dof_coords()        const { return invoke<DMatrix<double>        , 8>(*this); }
    decltype(auto) forcing_data()      const { return invoke<const DMatrix<double>& , 9>(*this); }
    decltype(auto) time_domain()       const { return invoke<const DVector<double>&, 10>(*this); }
    decltype(auto) initial_condition() const { return invoke<const DVector<double>&, 11>(*this); }
  
    struct eval_basis_ret_type { SpMatrix<double> Psi; DVector<double> D; };
    std::optional<eval_basis_ret_type> eval_basis(eval e, const DMatrix<double>& locs) const {
        using RetType = eval_basis_ret_type;
        switch (e) {   // run-time switch based on sampling strategy
        case eval::pointwise:
            return std::optional<RetType>{invoke<RetType, 12>(*this, locs)};
        case eval::areal:
	    return std::optional<RetType>{invoke<RetType, 13>(*this, locs)};
        }
        return std::nullopt;
    }
    // setters
    template <typename F> void set_forcing(const F& data) { fdapde::invoke<void, 14>(*this, DMatrix<double>(data)); }
    void set_dirichlet_bc(const DMatrix<double>& data) { fdapde::invoke<void, 15>(*this, data); }
    void set_initial_condition(const DVector<double>& data) { fdapde::invoke<void, 16>(*this, data); }
    template <typename E> void set_differential_operator(E diff_op) { fdapde::invoke<void, 17>(*this, diff_op); }

    // function pointers forwardings
    template <typename T>
    using fn_ptrs = fdapde::mem_fn_ptrs<
      &T::init, &T::solve,   // initialization and pde solution
      // getters
      &T::solution, &T::force, &T::stiff, &T::mass, &T::quadrature_nodes, &T::n_dofs, &T::dof_coords, &T::forcing_data,
      &T::time_domain, &T::initial_condition,
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
