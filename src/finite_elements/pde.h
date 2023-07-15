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
#include <vector>

#include "../mesh/mesh.h"
#include "../utils/symbols.h"
#include "basis/lagrangian_basis.h"
#include "basis/basis_cache.h"
#include "solvers/fem_solver_base.h"
#include "solvers/fem_standard_space_solver.h"
#include "solvers/fem_standard_spacetime_solver.h"
#include "integration/integrator.h"

namespace fdapde {
namespace core {

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

// Description of a Partial Differential Equation Lf = u
template <unsigned int M,                          // local dimension of the mesh
	  unsigned int N,                          // dimension of the mesh embedding space
	  unsigned int R,                          // order of the mesh
	  typename E,                              // differential operator L
	  typename F,                              // forcing term u
	  typename B = LagrangianBasis<M, R>,      // functional basis
	  typename I = Integrator<M, R>,           // quadrature rule
	  typename S = typename pde_fem_standard_solver_selector<E>::type>
class PDE : public PDEBase {
   private:
    const Mesh<M, N, R>& domain_;   // problem domain
    E bilinear_form_;               // the differential operator of the problem in its weak formulation
    static_assert(std::is_base_of<BilinearFormExpr<E>, E>::value);
    F forcing_data_;   // forcing data
    static_assert(std::is_same<DMatrix<double>, F>::value || std::is_base_of<ScalarExpr<F>, F>::value);
    DVector<double> initial_condition_ {};   // initial condition, used in space-time problems only
    I integrator_ {};                        // integrator used for approximation of integrals
    B reference_basis_ {};                   // basis defined over the reference unit simplex
    BasisCache<M, N, R, B> basis_ {};        // basis built over the whole domain_
    S solver_ {};                            // numerical scheme used to find a solution to this PDE

    // stores index of the boundary node and relative boundary value.
    std::unordered_map<std::size_t, DVector<double>> boundary_data_ {};
    void build_basis();   // initializes functional basis
   public:
    // minimal constructor, use below setters to complete the construction of a PDE object
    PDE(const Mesh<M, N, R>& domain) : domain_(domain) { build_basis(); }
    void set_forcing(const F& forcing_data) { forcing_data_ = forcing_data; }
    void set_bilinear_form(E bilinear_form) { bilinear_form_ = bilinear_form; }
    // full constructors
    PDE(const Mesh<M, N, R>& domain, E bilinear_form, const F& forcing_data) :
        domain_(domain), bilinear_form_(bilinear_form), forcing_data_(forcing_data) {
        build_basis();
    }
    PDE(const Mesh<M, N, R>& domain, E bilinear_form, const F& forcing_data, const B& basis, const I& integrator) :
        domain_(domain), bilinear_form_(bilinear_form), forcing_data_(forcing_data), reference_basis_(basis),
        integrator_(integrator) {
        build_basis();
    }

    // setters
    void set_dirichlet_bc(const DMatrix<double>& data);
    void set_initial_condition(const DVector<double>& data) { initial_condition_ = data; };

    // getters
    const Mesh<M, N, R>& domain() const { return domain_; }
    E bilinear_form() const { return bilinear_form_; }
    const F& forcing_data() const { return forcing_data_; }
    const DVector<double>& initial_condition() const { return initial_condition_; }
    const std::unordered_map<std::size_t, DVector<double>>& boundary_data() const { return boundary_data_; };
    const I& integrator() const { return integrator_; }
    const B& reference_basis() const { return reference_basis_; }
    const BasisCache<M, N, R, B>& basis() const { return basis_; }

    // PDE interface accessible from a pde_ptr
    virtual const DMatrix<double>& solution() const { return solver_.solution(); };   // PDE solution
    virtual const DMatrix<double>& force() const { return solver_.force(); };         // rhs of FEM linear system
    virtual const SpMatrix<double>& R1() const { return solver_.R1(); };              // stiff matrix
    virtual const SpMatrix<double>& R0() const { return solver_.R0(); };              // mass matrix
    virtual DMatrix<double> quadrature_nodes() const { return integrator_.quadrature_nodes(domain_); };

    virtual void init() { solver_.init(*this); };   // computes matrices R1, R0 and forcing vector u
    virtual void solve() {                          // solves the PDE
        if (!boundary_data_.empty()) solver_.set_dirichlet_bc(*this);
        solver_.solve(*this);
    }
  
    // expose compile time informations
    static constexpr std::size_t local_dimension = M;
    static constexpr std::size_t embedding_dimension = N;
    static constexpr std::size_t basis_order = R;
    typedef E BilinearFormType;
    typedef B BasisType;
    typedef I IntegratorType;
    typedef F ForcingType;
};

// template argument deduction rule for PDE object
template <unsigned int M, unsigned int N, unsigned int R, typename E, typename F, typename B, typename I>
PDE(const Mesh<M, N, R>& domain, E bilinear_form, const F& forcing, const B& basis, const I& integrator)
  -> PDE<M, N, R, decltype(bilinear_form), F, B, I>;

// implementative details
  
// basis table cache initialization
template <unsigned int M, unsigned int N, unsigned int R, typename E, typename F, typename B, typename I, typename S>
void PDE<M, N, R, E, F, B, I, S>::build_basis() {
    // preallocate memory for functional basis
    basis_.resize(domain_.elements());
    for (std::size_t i = 0; i < domain_.elements(); ++i) {
        basis_[i].reserve(ct_nnodes(M, R));
        for (std::size_t j = 0; j < ct_nnodes(M, R); ++j) {
            // store the dof_table()(i,j)-th basis function defined over element i and written in terms of the j-th
            // basis over the reference element
            basis_[i].emplace_back(domain_.dof_table()(i, j), domain_.element(i), reference_basis_[j]);
        }
    }
}

template <unsigned int M, unsigned int N, unsigned int R, typename E, typename F, typename B, typename I, typename S>
void PDE<M, N, R, E, F, B, I, S>::set_dirichlet_bc(const DMatrix<double>& data) {
    for (auto it = domain_.boundary_begin(); it != domain_.boundary_end(); ++it) {
        boundary_data_[*it] = data.row(*it);   // O(1) complexity
    }
}

// factory for pde_ptr objects
template <unsigned int M, unsigned int N, unsigned int R, typename E, typename F>
pde_ptr make_pde(const Mesh<M, N, R>& domain, E bilinear_form, const F& forcing_data) {
    return std::make_shared<PDE<M, N, R, decltype(bilinear_form), F>>(domain, bilinear_form, forcing_data);
}

}   // namespace core
}   // namespace fdapde

#endif   // __PDE_H__
