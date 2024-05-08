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
    using SpaceDomainType = D;                // triangulated spatial domain
    using TimeDomainType = DVector<double>;   // time-interval [0,T] (for time-dependent PDEs)
    static constexpr int M = SpaceDomainType::local_dim;
    static constexpr int N = SpaceDomainType::embed_dim;
    using OperatorType = E;   // differential operator in its strong-formulation
    static_assert(std::is_base_of<DifferentialExpr<OperatorType>, OperatorType>::value);
    using ForcingType = F;   // type of forcing object (either a matrix or a callable object)
    static_assert(
      std::is_same<DMatrix<double>, ForcingType>::value ||
      std::is_base_of<ScalarExpr<N, ForcingType>, ForcingType>::value);
    using SolverType = typename pde_solver_selector<S, SpaceDomainType, OperatorType, ForcingType, Ts...>::type;
    using FunctionalBasis = typename SolverType::FunctionalBasis;   // function space approximating the solution space
    using Quadrature = typename SolverType::Quadrature;             // quadrature for numerical integral approximations
    using ReferenceBasis = typename SolverType::ReferenceBasis;

    // space-only constructors
    PDE(const D& domain) requires(is_stationary<OperatorType>::value) : domain_(domain), solver_(domain) { }
    PDE(const D& domain, E diff_op) requires(is_stationary<OperatorType>::value)
        : domain_(domain), diff_op_(diff_op), solver_(domain) { }
    PDE(const SpaceDomainType& domain, OperatorType diff_op, const ForcingType& forcing_data)
        requires(is_stationary<OperatorType>::value)
        : domain_(domain), diff_op_(diff_op), forcing_data_(forcing_data), solver_(domain) { }
    // space-time constructors
    PDE(const D& domain, const TimeDomainType& t) requires(is_parabolic<OperatorType>::value)
        : domain_(domain), time_domain_(t), solver_(domain) { }
    PDE(const D& domain, const TimeDomainType& t, E diff_op) requires(is_parabolic<OperatorType>::value)
        : domain_(domain), time_domain_(t), diff_op_(diff_op), solver_(domain) { }
    PDE(const SpaceDomainType& domain, const TimeDomainType& t, OperatorType diff_op, const ForcingType& forcing_data)
        requires(is_parabolic<OperatorType>::value)
        : domain_(domain), time_domain_(t), diff_op_(diff_op), forcing_data_(forcing_data), solver_(domain) { }

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
    const ReferenceBasis& reference_basis() const { return solver_.reference_basis(); }
    // evaluates the functional basis defined over the pyhisical domain on a given set of locations
    template <template <typename> typename EvaluationPolicy>
    std::pair<SpMatrix<double>, DVector<double>> eval_functional_basis(const DMatrix<double>& locs) const {
        return solver_.basis().template eval<EvaluationPolicy>(locs);
    }
    int n_dofs() const { return solver_.n_dofs(); }
    const DMatrix<double>& solution() const { return solver_.solution(); };   // PDE solution
    const DMatrix<double>& force() const { return solver_.force(); };         // rhs of discretized linear system
    const SpMatrix<double>& stiff() const { return solver_.stiff(); };
    const SpMatrix<double>& mass() const { return solver_.mass(); };
    DMatrix<double> dof_coords() { return solver_.dofs_coords(); }
    const DMatrix<int>& dofs() const { return solver_.dofs(); }
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

// PDE type-erasure wrapper
struct PDE__ {
    // function pointers forwardings
    template <typename T>
    using fn_ptrs = fdapde::mem_fn_ptrs<
      &T::init, &T::solve,    // initialization and solution algorithm
      &T::solution,           // solution expansion coefficient vector
      &T::force,              // discretized forcing term [b]_i = \int_D f*\psi_i
      &T::stiff,              // discretization matrix of differential operator L
      &T::mass,               // mass matrix [R0]_{ij} = \int_D \psi_i*\psi_j
      &T::quadrature_nodes,   // quadrature nodes on pyhisical domain
      &T::n_dofs,             // number of degrees of freedom (number of basis function over the physical domain)
      &T::dof_coords,         // degrees of freedom physical coordinates
      &T::forcing_data,       // input forcing data
      &T::time_domain, &T::initial_condition,   // for space-time problems, the time interval T and initial condition
      &T::template eval_functional_basis<pointwise_evaluation>,   // computes [\Psi]_{ij} = \psi_j(p_i)
      &T::template eval_functional_basis<areal_evaluation>,       // computes [\Psi]_{ij} = \int_{D_i} \psi_i
      // setters
      &T::set_forcing, &T::set_dirichlet_bc, &T::set_initial_condition, &T::set_differential_operator>;

    void init()  { invoke<void, 0>(*this); }
    void solve() { invoke<void, 1>(*this); }
    // getters
    decltype(auto) solution()          const { return invoke<const DMatrix<double>& , 2> (*this); }
    decltype(auto) force()             const { return invoke<const DMatrix<double>& , 3> (*this); }
    decltype(auto) stiff()             const { return invoke<const SpMatrix<double>&, 4> (*this); }
    decltype(auto) mass()              const { return invoke<const SpMatrix<double>&, 5> (*this); }
    decltype(auto) quadrature_nodes()  const { return invoke<DMatrix<double>        , 6> (*this); }
    decltype(auto) n_dofs()            const { return invoke<int                    , 7> (*this); }
    decltype(auto) dof_coords()        const { return invoke<DMatrix<double>        , 8> (*this); }
    decltype(auto) forcing_data()      const { return invoke<const DMatrix<double>& , 9> (*this); }
    decltype(auto) time_domain()       const { return invoke<const DVector<double>& , 10>(*this); }
    decltype(auto) initial_condition() const { return invoke<const DVector<double>& , 11>(*this); }
    struct EvalReturnType { SpMatrix<double> Psi; DVector<double> D; };
    std::optional<EvalReturnType> eval_basis(int eval_type, const DMatrix<double>& locs) const {
        switch (eval_type) {
        case 0:   // Sampling::pointwise
            return std::optional<EvalReturnType> {invoke<EvalReturnType, 12>(*this, locs)};
        case 1:   // Sampling::areal
            return std::optional<EvalReturnType> {invoke<EvalReturnType, 13>(*this, locs)};
        }
        return std::nullopt;
    }
    // setters
    template <typename ForcingType> void set_forcing(const ForcingType& data)   { invoke<void, 14>(*this, data); }
    void set_dirichlet_bc(const DMatrix<double>& data)      { invoke<void, 15>(*this, data); }
    void set_initial_condition(const DVector<double>& data) { invoke<void, 16>(*this, data); }
    template <typename E> void set_differential_operator(E diff_op) { invoke<void, 17>(*this, diff_op); }
};

// factory method
template <typename... Args_, typename... Args> fdapde::erase<fdapde::heap_storage, PDE__> make_pde(Args&&... args) {
    return fdapde::erase<fdapde::heap_storage, PDE__>(PDE<Args_...>(std::forward<Args>(args)...));
}

}   // namespace core
}   // namespace fdapde

#endif   // __PDE_H__
