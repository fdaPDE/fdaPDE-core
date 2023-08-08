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

namespace fdapde {
namespace core {

// for a given resolution strategy S and operator E, selects a proper solver.
// to be partially specialized with respect to S
template <typename T, typename D, typename E, typename F> struct pde_solver_selector { };

// PDEs base class (used as tag in higher components, allow run-time polymorphism)
// abstract PDEs interface accessible throught a pointer to PDE
struct PDEBase {
    virtual const DMatrix<double>& solution() const  = 0;
    virtual const DMatrix<double>& force() const     = 0;
    virtual const SpMatrix<double>& R1() const       = 0;
    virtual const SpMatrix<double>& R0() const       = 0;
    virtual DMatrix<double> quadrature_nodes() const = 0;
    virtual void init()                              = 0;
    virtual void solve()                             = 0;
};
typedef std::shared_ptr<PDEBase> pde_ptr;

// Description of a Partial Differential Equation Lf = u solved with strategy S
template <typename D,   // triangulated problem domain
          typename E,   // differential operator L
          typename F,   // forcing term u
          typename S>   // resolution strategy
class PDE : public PDEBase {
   public:
    typedef D DomainType;   // triangulated domain
    static constexpr std::size_t M = DomainType::local_dimension;
    static constexpr std::size_t N = DomainType::embedding_dimension;
    static constexpr std::size_t R = DomainType::order;
    typedef E OperatorType;   // differential operator in its strong-formulation
    static_assert(
      std::is_base_of<DifferentialExpr<OperatorType>, OperatorType>::value, "E is not a valid differential operator");
    typedef F ForcingType;   // type of forcing object (either a matrix or a callable object)
    static_assert(
      std::is_same<DMatrix<double>, F>::value || std::is_base_of<ScalarExpr<D::embedding_dimension, F>, F>::value,
      "forcing is not a matrix or a scalar expression || N != F::base");
    typedef typename pde_solver_selector<S, D, E, F>::type SolverType;   // resolution strategy
    typedef typename SolverType::FunctionSpace FunctionSpace;     // function space approximating the solution space
    typedef typename SolverType::FunctionBasis FunctionBasis;     // basis defined on the overall domain
    typedef typename SolverType::QuadratureRule QuadratureRule;   // quadrature for numerical integral approximations

    // minimal constructor, use below setters to complete the construction of a PDE object
    PDE(const D& domain) : domain_(domain) { }
    void set_forcing(const F& forcing_data) { forcing_data_ = forcing_data; }
    void set_differential_operator(E diff_op) { diff_op_ = diff_op; }
    // full constructors
    PDE(const D& domain, E diff_op, const F& forcing_data) :
        domain_(domain), diff_op_(diff_op), forcing_data_(forcing_data) { }

    // setters
    void set_dirichlet_bc(const DMatrix<double>& data) {
        for (auto it = domain_.boundary_begin(); it != domain_.boundary_end(); ++it) {
            boundary_data_[*it] = data.row(*it);   // O(1) complexity
        }
    };
    void set_initial_condition(const DVector<double>& data) { initial_condition_ = data; };

    // getters
    const DomainType& domain() const { return domain_; }
    OperatorType differential_operator() const { return diff_op_; }
    const ForcingType& forcing_data() const { return forcing_data_; }
    const DVector<double>& initial_condition() const { return initial_condition_; }
    const std::unordered_map<std::size_t, DVector<double>>& boundary_data() const { return boundary_data_; };
    const QuadratureRule& integrator() const { return solver_.integrator(); }
    const FunctionSpace& reference_basis() const { return solver_.reference_basis(); }
    const FunctionBasis& basis() const { return solver_.basis(); }

    // pde_ptr accessible interface
    virtual const DMatrix<double>& solution() const { return solver_.solution(); };   // PDE solution
    virtual const DMatrix<double>& force() const { return solver_.force(); };         // rhs of discrete linear system
    virtual const SpMatrix<double>& R1() const { return solver_.R1(); };              // stiff matrix
    virtual const SpMatrix<double>& R0() const { return solver_.R0(); };              // mass matrix
    virtual DMatrix<double> quadrature_nodes() const { return integrator().quadrature_nodes(domain_); };
    virtual void init() { solver_.init(*this); };   // initializes the solver
    virtual void solve() {                          // solves the PDE
        if (!boundary_data_.empty()) solver_.set_dirichlet_bc(*this);
        solver_.solve(*this);
    }

   private:
    const DomainType& domain_;               // triangulated problem domain
    OperatorType diff_op_;                   // differential operator in its strong formulation
    ForcingType forcing_data_;               // forcing data
    DVector<double> initial_condition_ {};   // initial condition, (for space-time problems only)
    SolverType solver_ {};                   // problem solver
    std::unordered_map<std::size_t, DVector<double>> boundary_data_ {};   // (node_id, boundary_value) map
};

// factory for pde_ptr objects
template <typename D, typename E, typename F, typename S>
pde_ptr make_pde(const D& domain, E diff_op, const F& forcing_data) {
    return std::make_shared<PDE<D, decltype(diff_op), F, S>>(domain, diff_op, forcing_data);
}

// PDE-detection type trait
template <typename T> struct is_pde {
    static constexpr bool value = std::is_base_of<PDEBase, T>::value;
};

}   // namespace core
}   // namespace fdapde

#endif   // __PDE_H__
