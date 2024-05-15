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

#ifndef __FEM_SOLVER_BASE_H__
#define __FEM_SOLVER_BASE_H__

#include <exception>

#include "../../utils/integration/integrator.h"
#include "../../utils/symbols.h"
#include "../../utils/traits.h"
#include "../../utils/combinatorics.h"
#include "../basis/lagrangian_basis.h"
#include "../fem_assembler.h"
#include "../fem_symbols.h"
#include "../operators/reaction.h"   // for mass-matrix computation
#include "../../pde/symbols.h"

namespace fdapde {
namespace core {
  
// base class for the definition of a general solver based on the Finite Element Method
template <typename D, typename E, typename F, typename... Ts> class FEMSolverBase {
   public:
    typedef std::tuple<Ts...> SolverArgs;
    enum {
        fem_order = std::tuple_element<0, SolverArgs>::type::value,
        n_dof_per_element = ct_nnodes(D::local_dimension, fem_order),
        n_dof_per_edge = fem_order - 1,
        n_dof_internal =
          n_dof_per_element - (D::local_dimension + 1) - D::n_facets_per_element * (fem_order - 1)   // > 0 \iff R > 2
    };
    using DomainType = D;
    using FunctionalBasis = LagrangianBasis<DomainType, fem_order>;
    using ReferenceBasis = typename FunctionalBasis::ReferenceBasis;
    using Quadrature = typename ReferenceBasis::Quadrature;
    using ForceQuadrature = Integrator<FEM, D::local_dimension, 2>;  // quadrature to assemble the force (higher grade)
    // constructor
    FEMSolverBase() = default;
    FEMSolverBase(const DomainType& domain) : basis_(domain) {}
    FEMSolverBase(const DomainType& domain, const DMatrix<short int>& BMtrx) : basis_(domain, BMtrx) {}
    // getters
    const DMatrix<double>& solution() const { return solution_; }
    const DMatrix<double>& force() const { return force_; }
    const DMatrix<double>& force_neumann() const { return force_neumann_; }
    const DMatrix<double>& force_robin() const { return force_robin_; }
    const SpMatrix<double>& stiff() const { return stiff_; }
    const SpMatrix<double>& mass() const { return mass_; }
    const Quadrature& integrator() const { return integrator_; }
    const ForceQuadrature& force_integrator() const { return force_integrator_; }
    const ReferenceBasis& reference_basis() const { return reference_basis_; }
    const FunctionalBasis& basis() const { return basis_; }
    std::size_t n_dofs() const { return n_dofs_; }   // number of degrees of freedom (FEM linear system's unknowns)
    const DMatrix<int>& dofs() const { return dofs_; }
    DMatrix<double> dofs_coords() { return basis_.dofs_coords(); };   // computes the physical coordinates of dofs
    // flags
    bool is_init = false;   // notified true if initialization occurred with no errors
    bool success = false;   // notified true if problem solved with no errors

    template <typename PDE> void init(const PDE& pde);
    template <typename PDE> void set_dirichlet_bc(const PDE& pde);
    template <typename PDE> void build_neumann_force(const PDE& pde);
    template <typename PDE> void set_neumann_bc(const PDE& pde);
    template <typename PDE> void build_robin_force(const PDE& pde);
    template <typename PDE> void set_robin_bc(const PDE& pde);
    void set_stab_param(const double delta) { delta_ = delta; }
    
    struct boundary_dofs_iterator {   // range-for loop over boundary dofs
       private:
        friend FEMSolverBase;
        const FEMSolverBase* fem_solver_;
        int index_;   // current boundary dof
        boundary_dofs_iterator(const FEMSolverBase* fem_solver, int index) : fem_solver_(fem_solver), index_(index) {};
       public:
        // fetch next boundary dof
        boundary_dofs_iterator& operator++() {
            index_++;
            // scan until all nodes have been visited or a boundary node is not found
            for (; index_ < fem_solver_->n_dofs_ && fem_solver_->boundary_dofs_(index_,0) == 0; ++index_);
            return *this;
        }
        int operator*() const { return index_; }
        friend bool operator!=(const boundary_dofs_iterator& lhs, const boundary_dofs_iterator& rhs) {
            return lhs.index_ != rhs.index_;
        }
    };
    struct boundary_dofs_iterator_Dirichlet {   // range-for loop over Dirichlet boundary dofs
       private:
        friend FEMSolverBase;
        const FEMSolverBase* fem_solver_;
        int index_;   // current boundary dof
        boundary_dofs_iterator_Dirichlet(const FEMSolverBase* fem_solver, int index) : fem_solver_(fem_solver), index_(index) {};
       public:
        // fetch next boundary dof
        boundary_dofs_iterator_Dirichlet& operator++() {
            index_++;
            // scan until all nodes have been visited or a boundary node is not found
            for (; index_ < fem_solver_->n_dofs_ && fem_solver_->boundary_dofs_Dirichlet_(index_,0) == 0; ++index_);
            return *this;
        }
        int operator*() const { return index_; }
        friend bool operator!=(const boundary_dofs_iterator_Dirichlet& lhs, const boundary_dofs_iterator_Dirichlet& rhs) {
            return lhs.index_ != rhs.index_;
        }
    };
    struct boundary_dofs_iterator_Neumann {   // range-for loop over Neumann boundary dofs
       private:
        friend FEMSolverBase;
        const FEMSolverBase* fem_solver_;
        int index_;   // current boundary dof
        boundary_dofs_iterator_Neumann(const FEMSolverBase* fem_solver, int index) : fem_solver_(fem_solver), index_(index) {};
       public:
        // fetch next boundary dof
        boundary_dofs_iterator_Neumann& operator++() {
            index_++;
            // scan until all nodes have been visited or a boundary node is not found
            for (; index_ < fem_solver_->n_dofs_ && fem_solver_->boundary_dofs_Neumann_(index_,0) == 0; ++index_);
            return *this;
        }
        int operator*() const { return index_; }
        friend bool operator!=(const boundary_dofs_iterator_Neumann& lhs, const boundary_dofs_iterator_Neumann& rhs) {
            return lhs.index_ != rhs.index_;
        }
    };
    struct boundary_dofs_iterator_Robin {   // range-for loop over Neumann boundary dofs
       private:
        friend FEMSolverBase;
        const FEMSolverBase* fem_solver_;
        int index_;   // current boundary dof
        boundary_dofs_iterator_Robin(const FEMSolverBase* fem_solver, int index) : fem_solver_(fem_solver), index_(index) {};
       public:
        // fetch next boundary dof
        boundary_dofs_iterator_Robin& operator++() {
            index_++;
            // scan until all nodes have been visited or a boundary node is not found
            for (; index_ < fem_solver_->n_dofs_ && fem_solver_->boundary_dofs_Robin_(index_,0) == 0; ++index_);
            return *this;
        }
        int operator*() const { return index_; }
        friend bool operator!=(const boundary_dofs_iterator_Robin& lhs, const boundary_dofs_iterator_Robin& rhs) {
            return lhs.index_ != rhs.index_;
        }
    };

    boundary_dofs_iterator boundary_dofs_begin() const { return boundary_dofs_iterator(this, 0); }
    boundary_dofs_iterator boundary_dofs_end() const { return boundary_dofs_iterator(this, n_dofs_); }
    boundary_dofs_iterator_Dirichlet boundary_dofs_begin_Dirichlet() const {
        auto it = boundary_dofs_iterator_Dirichlet(this, 0);
        if (this->boundary_dofs_Dirichlet_(0,0) == 1) return it;
        else return ++it;
    }
    boundary_dofs_iterator_Dirichlet boundary_dofs_end_Dirichlet() const { return boundary_dofs_iterator_Dirichlet(this, n_dofs_); }
    boundary_dofs_iterator_Neumann boundary_dofs_begin_Neumann() const {
        auto it = boundary_dofs_iterator_Neumann(this, 0);
        if (this->boundary_dofs_Neumann_(0,0) == 1) return it;
        else return ++it;
    }
    boundary_dofs_iterator_Neumann boundary_dofs_end_Neumann() const { return boundary_dofs_iterator_Neumann(this, n_dofs_); }
    boundary_dofs_iterator_Robin boundary_dofs_begin_Robin() const {
        auto it = boundary_dofs_iterator_Robin(this, 0);
        if (this->boundary_dofs_Robin_(0,0) == 1) return it;
        else return ++it;
    }
    boundary_dofs_iterator_Robin boundary_dofs_end_Robin() const { return boundary_dofs_iterator_Robin(this, n_dofs_); }
   protected:
    Quadrature integrator_ {};                   // default to a quadrature rule which is exact for the considered FEM order
    ForceQuadrature force_integrator_ {};        // default to a quadrature rule which is of order 2
    FunctionalBasis basis_ {};                   // basis system defined over the pyhisical domain
    ReferenceBasis reference_basis_ {};          // function basis on the reference unit simplex
    DMatrix<double> solution_;                   // vector of coefficients of the approximate solution
    DMatrix<double> force_;                      // discretized force [u]_i = \int_D f*\psi_i
    DMatrix<double> force_neumann_;              // discretized Neumann force [u]_{ij} = \int_\Gamma_{Neumann} f_neumman(@time=j)*\psi_i
    DMatrix<double> force_robin_;                // discretized Robin force [u]_{ij} = \int_\Gamma_{Robin} (1/b * f_robin(@time=j) * \psi_i)
    SpMatrix<double> stiff_;                     // [stiff_]_{ij} = a(\psi_i, \psi_j), being a(.,.) the bilinear form
    SpMatrix<double> mass_;                      // mass matrix, [mass_]_{ij} = \int_D (\psi_i * \psi_j)
    SpMatrix<double> robin_;                     // Robin b.c.s boundary mass matrix, [robin_]_{ij} = \int_\Gamma_{Robin} (\mu * a/b * \psi_i * \psi_j)
    double delta_ = 1.;                          // stabilization parameter

    std::size_t n_dofs_ = 0;        // degrees of freedom, i.e. the maximum ID in the dof_table_
    DMatrix<int> dofs_;             // for each element, the degrees of freedom associated to it
    DMatrix<int> boundary_dofs_;    // unknowns on the boundary of the domain, for boundary conditions prescription
    DMatrix<int> boundary_dofs_Dirichlet_;   // unknowns on the Dirichlet boundary of the domain, for boundary conditions prescription
    DMatrix<int> boundary_dofs_Neumann_;     // unknowns on the Neumann boundary of the domain, for boundary conditions prescription
    DMatrix<int> boundary_dofs_Robin_;       // unknowns on the Robin boundary of the domain, for boundary conditions prescription
};

// implementative details

// initialize solver
template <typename D, typename E, typename F, typename... Ts>
template <typename PDE>
void FEMSolverBase<D, E, F, Ts...>::init(const PDE& pde) {
    fdapde_static_assert(is_pde<PDE>::value, THIS_METHOD_IS_FOR_PDE_ONLY);
    n_dofs_ = basis_.size();
    dofs_ = basis_.dofs();
    boundary_dofs_ = basis_.boundary_dofs();
    boundary_dofs_Dirichlet_ = basis_.boundary_dofs_Dirichlet();
    boundary_dofs_Neumann_ = basis_.boundary_dofs_Neumann();
    boundary_dofs_Robin_ = basis_.boundary_dofs_Robin();
    
    // assemble stiffness matrix for given operator
    Assembler<FEM, DomainType, ReferenceBasis, Quadrature> assembler_stiff(pde.domain(), integrator_, n_dofs_, dofs_);
    if constexpr(is_nonlinear<E>::value){
      if (is_empty(solution_)) solution_ = pde.initial_condition();
      assembler_stiff.set_f(solution_.col(0));
    }
    stiff_ = assembler_stiff.discretize_operator(pde.differential_operator());
    stiff_.makeCompressed();

    // assemble forcing vector
    std::size_t n = n_dofs_;   // degrees of freedom in space
    std::size_t m;             // number of time points
    Assembler<FEM, DomainType, ReferenceBasis, ForceQuadrature> assembler_force(pde.domain(), force_integrator_, n_dofs_, dofs_);
    if constexpr (!std::is_base_of<ScalarBase, F>::value) {
        m = pde.forcing_data().cols();
        force_.resize(n * m, 1);
        force_.block(0, 0, n, 1) = assembler_force.discretize_forcing(pde.forcing_data().col(0));

        // iterate over time steps if a space-time PDE is supplied
        if constexpr (is_parabolic<E>::value) {
            for (std::size_t i = 1; i < m; ++i) {
                force_.block(n * i, 0, n, 1) = assembler_force.discretize_forcing(pde.forcing_data().col(i));
            }
        }
    } else {
        // TODO: support space-time callable forcing for parabolic problems
        m = 1;
        force_.resize(n * m, 1);
        force_.block(0, 0, n, 1) = assembler_force.discretize_forcing(pde.forcing_data());
    }
    
    // compute mass matrix [mass]_{ij} = \int_{\Omega} \phi_i \phi_j
    mass_ = assembler_stiff.discretize_operator(Reaction<FEM, double>(1.0));

    // assemble stabilization matrix (if needed)
    if constexpr (is_advection<E>::value && !is_parabolic<E>::value) {
        typedef std::tuple<Ts...> SolverArgs;
        SpMatrix<double> stab_(this->n_dofs_, this->n_dofs_);
        stab_.setZero();
        DVector<double> stab_rhs_ = DVector<double>::Zero(this->n_dofs_);
        if constexpr (is_reaction<E>::value) {  // diffusion - advection - reaction equation
            // initialize empty value with the CORRECT TYPE using default constructor
            auto mu_temp = std::tuple_element_t<1, SolverArgs>();
            auto b_temp = std::tuple_element_t<2, SolverArgs>();
            auto c_temp = std::tuple_element_t<3, SolverArgs>();
            // retrieve the values of the PDE parameters from the singleton
            PDEparameters<decltype(mu_temp), decltype(b_temp), decltype(c_temp)> &PDEparams =
                PDEparameters<decltype(mu_temp), decltype(b_temp), decltype(c_temp)>::getInstance(mu_temp, b_temp, c_temp);
            // get the actual parameters
            auto mu = std::get<0>(PDEparams.getData());
            auto b = std::get<1>(PDEparams.getData());
            auto c = std::get<2>(PDEparams.getData());
            // compute first the normb of b to compute the Peclet number
            Assembler<FEM, DomainType, ReferenceBasis, Quadrature> assembler_stab(pde.domain(), integrator_, n_dofs_, dofs_);
            double normb = 0;
            if constexpr (std::is_base_of<VectorBase, decltype(b)>::value) {
                normb = assembler_stab.normVectorField(b);
            } else {
                normb = b.norm();
            }
            double h = std::sqrt(this-> basis_.get_element_size() * 2); // assuming triangles
            // compute the Peclet number
            double Pe = 0;
            if constexpr (std::is_same<decltype(mu), double>::value) {
                Pe = normb * h * 0.5 / mu;
            }else{
                Pe = normb * h * 0.5 / mu.norm();
            }
            if (Pe > 1) {
                auto SUPGtuple = std::make_tuple(mu, b, c, normb, delta_);
                auto StrongStab = SUPG_ADV_DIFF_REACT<FEM, decltype(SUPGtuple)>(SUPGtuple);
                
                stab_ = assembler_stab.discretize_operator(StrongStab);
                stab_rhs_ = assembler_stab.discretize_SUPG_RHS(b, normb, pde.forcing_data(), delta_);
            }

        }  else {   // diffusion - advection equation 
        // TODO: avoid this code repetition
            // initialize empty value with the CORRECT TYPE using default constructor
            auto mu_temp = std::tuple_element_t<1, SolverArgs>();
            auto b_temp = std::tuple_element_t<2, SolverArgs>();

            PDEparameters<decltype(mu_temp), decltype(b_temp)> &PDEparams =
                    PDEparameters<decltype(mu_temp), decltype(b_temp)>::getInstance(mu_temp, b_temp);

            auto mu = std::get<0>(PDEparams.getData());
            auto b = std::get<1>(PDEparams.getData());

            Assembler<FEM, DomainType, ReferenceBasis, Quadrature> assembler(pde.domain(), this->integrator_,
                                                                                this->n_dofs_, this->dofs_);

            double normb = 0;
            if constexpr (std::is_base_of<VectorBase, decltype(b)>::value) {
                normb = assembler.normVectorField(b);
            } else {
                normb = b.norm();
            }

            double h = std::sqrt(this-> basis_.get_element_size() * 2);
            double Pe = 0;
            if constexpr (std::is_same<decltype(mu), double>::value) {
                Pe = normb * h * 0.5 / mu;
            }else{
                Pe = normb * h * 0.5 / mu.norm();
            }

            if (Pe > 1) {
                auto SUPGtuple = std::make_tuple(mu, b, normb, delta_);
                auto StrongStab = SUPG_ADV_DIFF<FEM, decltype(SUPGtuple)>(SUPGtuple);
                
                stab_ = assembler.discretize_operator(StrongStab);
                stab_rhs_ = assembler.discretize_SUPG_RHS(b, normb, pde.forcing_data(), delta_);
            }
        }   // end is_reaction
        // SUPG STABILIZATION HERE!
        stiff_ += stab_;
        force_ += stab_rhs_;
    }   // end is_advection

    // build Neumann and Robin force vectors
    build_neumann_force(pde);
    build_robin_force(pde);

    is_init = true;
    return;
}

// impose dirichlet boundary conditions
template <typename D, typename E, typename F, typename... Ts>
template <typename PDE>
void FEMSolverBase<D, E, F, Ts...>::set_dirichlet_bc(const PDE& pde) {
    fdapde_static_assert(is_pde<PDE>::value, THIS_METHOD_IS_FOR_PDE_ONLY);
    if (!is_init) throw std::runtime_error("solver must be initialized first!");
    for (auto it = boundary_dofs_begin_Dirichlet(); it != boundary_dofs_end_Dirichlet(); ++it) {
        // standard application of Dirichlet boundary conditions
        stiff_.row(*it) *= 0;            // zero all entries of this row
        stiff_.coeffRef(*it, *it) = 1;   // set diagonal element to 1 to impose equation u_j = b_j
        // penalty method for Dirichlet boundary conditions (preferrable with direct solvers for pivoting)
        // stiff_.coeffRef(*it, *it) = 1e30;

        // TODO: currently only space-only case supported (reason of [0] below)
        
        // satndard application of Dirichlet boundary conditions
        force_.coeffRef(*it, 0) = pde.dirichlet_boundary_data()(*it, 0);   // impose boundary value on forcing term
        // penalty method for Dirichlet boundary conditions
        // force_.coeffRef(*it, 0) = 1e30 * pde.dirichlet_boundary_data()(*it, 0);
        
        // iterate over time steps if a space-time PDE is supplied
        if constexpr (is_parabolic<E>::value) {
            std::size_t n = n_dofs_;   // degrees of freedom in space
            std::size_t m = pde.forcing_data().cols();  // number of time points
            for (std::size_t i = 1; i < m; ++i) {
                // standard application of Dirichlet boundary conditions
                force_.coeffRef((*it) + n*i, 0) = pde.dirichlet_boundary_data()(*it, i);   // impose boundary value on forcing term
                // penalty method for Dirichlet boundary conditions
                // force_.coeffRef((*it) + n*i, 0) = 1e30 * pde.dirichlet_boundary_data()(*it, i);
            }
        }
    }
    return;
}

// build Neumann forcing vector
template <typename D, typename E, typename F, typename... Ts>
template <typename PDE>
void FEMSolverBase<D, E, F, Ts...>::build_neumann_force(const PDE& pde) {
    // std::cout << "called Neumann" << std::endl;
    fdapde_static_assert(is_pde<PDE>::value, THIS_METHOD_IS_FOR_PDE_ONLY);

    // allocate space for result vector
    DVector<double> neumann_forcing {};
    neumann_forcing.resize(n_dofs_, 1);   // there are as many basis functions as degrees of freedom on the mesh

    std::size_t n = n_dofs_;   // degrees of freedom in space
    std::size_t m = 1;
    if constexpr (is_parabolic<E>::value) m = pde.forcing_data().cols();  // number of time points
    force_neumann_ = DMatrix<double>::Zero(n, m);

    int i;   // index we need to count the number of facets on the boundary

    if constexpr(D::local_dimension ==1) {
        // for each timestep
        for (std::size_t k=0; k<m; k++){
            i = -1;
            neumann_forcing.fill(0);  // init result vector to zero
            // cycle over all mesh facets
            for (int ID = 0; ID < pde.domain().n_nodes(); ++ID) {
                // check if the facet is on boundary
                if (pde.domain().is_on_boundary(ID)) {
                    i++;                    
                    bool Neumann = (basis_.BMtrx()(ID, 0) == 1);
                    if (Neumann) {
                        // integrate pde.neumann_boundary_data()*phi over the facet (*it)
                        double integral_value = 0;
                        
                        // we assume pde.neumann_boundary_data() given as vector of values with the assumption that
                        // pde.neumann_boundary_data()[boundary_integration_table_.num_nodes*e.ID() + iq] equals the discretized field at the iq-th quadrature
                        // node.
                        integral_value += pde.neumann_boundary_data()(i, k);

                        // correct for measure of domain (facet e)
                        neumann_forcing[ID] += integral_value;
                    }    
                }
            }
            for (auto it = boundary_dofs_begin_Neumann(); it != boundary_dofs_end_Neumann(); ++it) {
                force_neumann_.coeffRef(*it, k) += neumann_forcing(*it, 0);
            }
        }
    }

    else {
        static constexpr int Mb = D::local_dimension - 1;
        static constexpr int Nb = D::embedding_dimension;
        static constexpr int Rb = basis_.R;
        using FunctionalBasis = LagrangianBasis<Mesh<Mb,Nb>, Rb>;
        using ReferenceBasis = typename FunctionalBasis::ReferenceBasis;

        ReferenceBasis boundary_reference_basis(ReferenceElement<Mb, Rb>::nodes);
        static constexpr int Kb = standard_fem_quadrature_rule<Mb, Rb>::K;   // number of quadrature nodes for integration on the boundary
        auto boundary_integration_table_ = IntegratorTable<Mb, Kb>();


        // for each timestep
        for (std::size_t k=0; k<m; k++){
            i = -1;
            neumann_forcing.fill(0);  // init result vector to zero
            // cycle over all mesh facets
            for (auto it = pde.domain().facet_begin(); it != pde.domain().facet_end(); ++it) {
                bool Neumann = false;  // to asses if a facet is a Neumann boundary one
                // check if the facet is on boundary
                if ((*it).on_boundary()) {
                    i++;
                    for (auto j = 0; j < (*it).node_ids().size(); ++j) if (basis_.BMtrx()((*it).node_ids()[j], 0) == 1) Neumann = true;
                    if (Neumann) {
                        for (auto jj=0; jj<boundary_reference_basis.n_basis; ++jj){
                            // integrate pde.neumann_boundary_data()*phi over the facet (*it)
                            double integral_value = 0;
                            for (size_t iq = 0; iq < boundary_integration_table_.num_nodes; ++iq) {
                                const SVector<Mb>& p = boundary_integration_table_.nodes[iq];

                                // we assume pde.neumann_boundary_data() given as vector of values with the assumption that
                                // pde.neumann_boundary_data()[boundary_integration_table_.num_nodes*i + iq] equals the discretized 
                                // field at the iq-th quadrature node.
                                integral_value += (pde.neumann_boundary_data()(boundary_integration_table_.num_nodes * i + iq, k) * boundary_reference_basis[jj](p)) * boundary_integration_table_.weights[iq]; 
                            }
                            // correct for measure of domain (facet e)
                            neumann_forcing[basis_.dof_boundary_table()(i, jj)] += integral_value * (*it).measure();
                        }
                    }    
                }
            }
            for (auto it = boundary_dofs_begin_Neumann(); it != boundary_dofs_end_Neumann(); ++it) {
                force_neumann_.coeffRef(*it, k) += neumann_forcing(*it, 0);
            }
        }
    }

    return;
}


// impose Neumann boundary conditions
template <typename D, typename E, typename F, typename... Ts>
template <typename PDE>
void FEMSolverBase<D, E, F, Ts...>::set_neumann_bc(const PDE& pde) {
    // std::cout << "called Neumann" << std::endl;
    fdapde_static_assert(is_pde<PDE>::value, THIS_METHOD_IS_FOR_PDE_ONLY);
    if (!is_init) throw std::runtime_error("solver must be initialized first!");

    std::size_t n = n_dofs_;   // degrees of freedom in space
    std::size_t m = 1;
    if constexpr (is_parabolic<E>::value) m = pde.forcing_data().cols();  // number of time points

    // for each timestep
    for (std::size_t k=0; k<m; k++){
        for (auto it = boundary_dofs_begin_Neumann(); it != boundary_dofs_end_Neumann(); ++it) {
            force_.coeffRef((*it) + n*k, 0) += force_neumann_.coeffRef(*it, k);
        }
    }

    return;
}


// impose Robin boundary conditions
template <typename D, typename E, typename F, typename... Ts>
template <typename PDE>
void FEMSolverBase<D, E, F, Ts...>::build_robin_force(const PDE& pde) {
    // std::cout << "called Robin" << std::endl;
    fdapde_static_assert(is_pde<PDE>::value, THIS_METHOD_IS_FOR_PDE_ONLY);

    DVector<double> robin_forcing {};
    robin_forcing.resize(n_dofs_, 1);   // there are as many basis functions as degrees of freedom on the mesh
    robin_.resize(n_dofs_, n_dofs_);

    std::size_t n = n_dofs_;   // degrees of freedom in space
    std::size_t m = 1;
    if constexpr (is_parabolic<E>::value) m = pde.forcing_data().cols();  // number of time points
    force_robin_ = DMatrix<double>::Zero(n, m);

    int i;   // index we need to count the number of facets on the boundary

    if constexpr(D::local_dimension == 1) {
         // for each timestep
        for (std::size_t k=0; k<m; k++){
            i = -1;
            robin_forcing.fill(0);   // init result vector to zero
            // cycle over all mesh facets
            for (int ID = 0; ID < pde.domain().n_nodes(); ++ID) {
                // check if the facet is on boundary
                if (pde.domain().is_on_boundary(ID)) {
                    i++;                    
                    bool Robin = (basis_.BMtrx()(ID, 0) == 2);
                    if (Robin) {
                        // integrate pde.robin_boundary_data()*phi over the facet (*it)
                        double integral_value = 0;

                        integral_value += pde.robin_boundary_data()(i, k);

                        // correct for measure of domain (facet e)
                        robin_forcing[ID] += integral_value / pde.robin_constants()[1];
                    }    
                }
            }
            for (auto it = boundary_dofs_begin_Robin(); it != boundary_dofs_end_Robin(); ++it) {
                force_robin_.coeffRef(*it, k) += robin_forcing(*it, 0);
                if (k == 0) { // to do only once
                    robin_.coeffRef((*it), (*it)) += pde.robin_constants()[0]/pde.robin_constants()[1];
                }
            }
        }
    }

    else {
        static constexpr int Mb = D::local_dimension - 1;
        static constexpr int Nb = D::embedding_dimension;
        static constexpr int Rb = basis_.R;
        using FunctionalBasis = LagrangianBasis<Mesh<Mb,Nb>, Rb>;
        using ReferenceBasis = typename FunctionalBasis::ReferenceBasis;

        ReferenceBasis boundary_reference_basis(ReferenceElement<Mb, Rb>::nodes);
        static constexpr int Kb = standard_fem_quadrature_rule<Mb, Rb>::K;   // number of quadrature nodes for integration on the boundary
        auto boundary_integration_table_ = IntegratorTable<Mb, Kb>();

        // for each timestep
        for (std::size_t k=0; k<m; k++){
            i = -1;
            robin_forcing.fill(0);   // init result vector to zero
            // cycle over all mesh facets
            for (auto it = pde.domain().facet_begin(); it != pde.domain().facet_end(); ++it) {
                bool Robin = false;  // to asses if a facet is a Robin boundary one
                // check if the facet is on boundary
                if ((*it).on_boundary()) {
                    i++;
                    for (auto j = 0; j < (*it).node_ids().size(); ++j) {
                        if (basis_.BMtrx()((*it).node_ids()[j], 0) == 1) {Robin = false; break;}
                        if (basis_.BMtrx()((*it).node_ids()[j], 0) == 2) {Robin = true;}
                    }
                    if (Robin) {
                        for (auto jj=0; jj<boundary_reference_basis.n_basis; ++jj){
                            // integrate pde.robin_boundary_data()*phi over the facet (*it)
                            double force_integral_value = 0;
                            for (size_t iq = 0; iq < boundary_integration_table_.num_nodes; ++iq) {
                                const SVector<Mb>& p = boundary_integration_table_.nodes[iq];
  
                                // we assume pde.robin_boundary_data() given as vector of values with the assumption that
                                // pde.robin_boundary_data()[boundary_integration_table_.num_nodes*i + iq] equals the discretized 
                                // field at the iq-th quadrature node.
                                force_integral_value += ((pde.robin_boundary_data()(boundary_integration_table_.num_nodes * i + iq, k)) * boundary_reference_basis[jj](p)) * boundary_integration_table_.weights[iq];
                            }
                            // correct for measure of domain (facet e)
                            robin_forcing[basis_.dof_boundary_table()(i, jj)] += force_integral_value * (*it).measure() / pde.robin_constants()[1];

                            // integrate phi_jj*phi_ii over the facet (*it)
                            if (k == 0){
                                for (auto ii=0; ii<boundary_reference_basis.n_basis; ++ii){
                                    double mass_integral_value = 0;
                                    for (size_t iq = 0; iq < boundary_integration_table_.num_nodes; ++iq) {
                                        const SVector<Mb>& p = boundary_integration_table_.nodes[iq];
                                        mass_integral_value += boundary_reference_basis[jj](p) * boundary_reference_basis[ii](p) * boundary_integration_table_.weights[iq];
                                    }
                                    // correct for measure of domain (facet e)
                                    robin_.coeffRef(basis_.dof_boundary_table()(i, ii), basis_.dof_boundary_table()(i, jj)) += mass_integral_value * (*it).measure() * pde.robin_constants()[0]/pde.robin_constants()[1];
                                }
                            }
                        }
                    }    
                }
            }
            for (auto it = boundary_dofs_begin_Robin(); it != boundary_dofs_end_Robin(); ++it) {
                force_robin_.coeffRef(*it, k) += robin_forcing(*it, 0);
            }
        }
    }

    return;
}


// impose Robin boundary conditions
template <typename D, typename E, typename F, typename... Ts>
template <typename PDE>
void FEMSolverBase<D, E, F, Ts...>::set_robin_bc(const PDE& pde) {
    // std::cout << "called Robin" << std::endl;
    fdapde_static_assert(is_pde<PDE>::value, THIS_METHOD_IS_FOR_PDE_ONLY);
    if (!is_init) throw std::runtime_error("solver must be initialized first!");

    std::size_t n = n_dofs_;   // degrees of freedom in space
    std::size_t m = 1;
    if constexpr (is_parabolic<E>::value) m = pde.forcing_data().cols();  // number of time points

    // for each timestep
    for (std::size_t k=0; k<m; k++){
        for (auto it = boundary_dofs_begin_Robin(); it != boundary_dofs_end_Robin(); ++it) {
            force_.coeffRef((*it) + n*k, 0) += force_robin_.coeffRef(*it, k);
        }
    }
    stiff_ += robin_;

    return;
}

}   // namespace core
}   // namespace fdapde

#endif   // __FEM_SOLVER_BASE_H__