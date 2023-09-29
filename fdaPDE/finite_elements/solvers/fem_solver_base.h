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
#include "../basis/finite_element_basis.h"
#include "../basis/lagrangian_element.h"
#include "../fem_assembler.h"
#include "../fem_symbols.h"
#include "../operators/reaction.h"   // for mass-matrix computation

namespace fdapde {
namespace core {

// forward declaration
template <typename PDE> struct is_pde;
  
// base class for the definition of a general solver based on the Finite Element Method
template <typename D, typename E, typename F, typename... Ts> class FEMSolverBase {
   public:
    typedef std::tuple<Ts...> SolverArgs;
    enum {
        fem_order = std::tuple_element <0, SolverArgs>::type::value,
        n_dof_per_element = ct_nnodes(D::local_dimension, fem_order),
        n_dof_per_edge = fem_order - 1,
        n_dof_internal =
          n_dof_per_element - (D::local_dimension + 1) - D::n_edges_per_element * (fem_order - 1)   // > 0 \iff R > 2
    };
    typedef D DomainType;
    typedef Integrator<DomainType::local_dimension, fem_order> QuadratureRule;
    typedef LagrangianElement<DomainType::local_dimension, fem_order> FunctionSpace;
    typedef FiniteElementBasis<FunctionSpace> FunctionBasis;

    // constructor
    FEMSolverBase() = default;

    // getters
    const DMatrix<double>& solution() const { return solution_; }
    const DMatrix<double>& force() const { return force_; }
    const SpMatrix<double>& R1() const { return R1_; }
    const SpMatrix<double>& R0() const { return R0_; }
    const QuadratureRule& integrator() const { return integrator_; }
    const FunctionSpace& reference_basis() const { return reference_basis; }
    const FunctionBasis& basis() const { return fe_basis_; }
    std::size_t n_dofs() const { return n_dofs_; }   // number of degrees of freedom (FEM linear system's unknowns)
    const DMatrix<int>& dofs() const { return dofs_; }
    DMatrix<double> dofs_coords(const DomainType& mesh);   // computes the physical coordinates of dofs

    // flags
    bool is_init = false;   // notified true if initialization occurred with no errors
    bool success = false;   // notified true if problem solved with no errors

    template <typename PDE> void init(const PDE& pde);
    template <typename PDE> void set_dirichlet_bc(const PDE& pde);
    
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
    boundary_dofs_iterator boundary_dofs_begin() const { return boundary_dofs_iterator(this, 0); }
    boundary_dofs_iterator boundary_dofs_end() const { return boundary_dofs_iterator(this, n_dofs_); }
  
   protected:
    QuadratureRule integrator_ {};       // default to a quadrature rule which is exact for the considered FEM order
    FunctionSpace reference_basis_ {};   // function basis on the reference unit simplex
    FunctionBasis fe_basis_ {};          // basis over the whole domain
    DMatrix<double> solution_;           // vector of coefficients of the approximate solution
    DMatrix<double> force_;              // discretized force [u]_i = \int_D f*\psi_i
    SpMatrix<double> R1_;   // [R1_]_{ij} = a(\psi_i, \psi_j), being a(.,.) the bilinear form of the problem
    SpMatrix<double> R0_;   // mass matrix, [R0_]_{ij} = \int_D (\psi_i * \psi_j)

    std::size_t n_dofs_ = 0;        // degrees of freedom, i.e. the maximum ID in the dof_table_
    DMatrix<int> dofs_;             // for each element, the degrees of freedom associated to it
    DMatrix<int> boundary_dofs_;    // unknowns on the boundary of the domain, for boundary conditions prescription
   private:
    // builds an enumeration of dofs coherent with the mesh topology for a functional basis of order R 
    void enumerate_dofs(const DomainType& mesh);
};

// implementative details

// initialize solver
template <typename D, typename E, typename F, typename... Ts>
template <typename PDE>
void FEMSolverBase<D, E, F, Ts...>::init(const PDE& pde) {
    static_assert(is_pde<PDE>::value, "not a valid PDE");
    // enumerate linear system unknowns
    enumerate_dofs(pde.domain());
    // assemble discretization matrix for given operator    
    Assembler<FEM, D, FunctionSpace, QuadratureRule> assembler(pde.domain(), integrator_, n_dofs_, dofs_);
    R1_ = assembler.discretize_operator(pde.differential_operator());
    R1_.makeCompressed();
    // assemble forcing vector
    std::size_t n = n_dofs_;   // degrees of freedom in space
    std::size_t m;             // number of time points
    if constexpr (!std::is_base_of<ScalarBase, F>::value) {
        m = pde.forcing_data().cols();
        force_.resize(n * m, 1);
        force_.block(0, 0, n, 1) = assembler.discretize_forcing(pde.forcing_data().col(0));

        // iterate over time steps if a space-time PDE is supplied
        if constexpr (is_parabolic<E>::value) {
            for (std::size_t i = 1; i < m; ++i) {
                force_.block(n * i, 0, n, 1) = assembler.discretize_forcing(pde.forcing_data().col(i));
            }
        }
    } else {
        // TODO: support space-time callable forcing for parabolic problems
        m = 1;
        force_.resize(n * m, 1);
        force_.block(0, 0, n, 1) = assembler.discretize_forcing(pde.forcing_data());
    }
    // compute mass matrix [R0]_{ij} = \int_{\Omega} \phi_i \phi_j
    R0_ = assembler.discretize_operator(Reaction<FEM, double>(1.0));
    is_init = true;
    return;
}

// impose dirichlet boundary conditions
template <typename D, typename E, typename F, typename... Ts>
template <typename PDE>
void FEMSolverBase<D, E, F, Ts...>::set_dirichlet_bc(const PDE& pde) {
    static_assert(is_pde<PDE>::value, "not a valid PDE");
    if (!is_init) throw std::runtime_error("solver must be initialized first!");
    for (auto it = boundary_dofs_begin(); it != boundary_dofs_end(); ++it) {
      R1_.row(*it) *= 0;            // zero all entries of this row
      R1_.coeffRef(*it, *it) = 1;   // set diagonal element to 1 to impose equation u_j = b_j
	
      // TODO: currently only space-only case supported (reason of [0] below)
      force_.coeffRef(*it, 0) = pde.boundary_data()(*it, 0);   // impose boundary value on forcing term
    }
    return;
}
  
// builds a node enumeration for the support of a basis of order R. Specialization for 2D domains
template <typename D, typename E, typename F, typename... Ts>
void FEMSolverBase<D, E, F, Ts...>::enumerate_dofs(const D& mesh) {
  if(n_dofs_ != 0) return; // return early if dofs already computed
  if constexpr (fem_order == 1) {
    n_dofs_ = mesh.n_nodes();
    dofs_ = mesh.elements();
    boundary_dofs_ = mesh.boundary(); 
  } else {
    dofs_.resize(mesh.n_elements(), n_dof_per_element);
    dofs_.leftCols(D::n_vertices) = mesh.elements(); // copy dofs associated to geometric vertices

    int next = mesh.n_nodes();   // next valid ID to assign
    auto edge_pattern = combinations<D::n_vertices_per_edge, D::n_vertices>();
    std::set<int> boundary_set;
    
    // cycle over mesh edges
    for(auto edge = mesh.edge_begin(); edge != mesh.edge_end(); ++edge) {
      for(std::size_t i = 0; i < D::n_elements_per_edge; ++i) {
	int element_id = (*edge).adjacent_elements()[i];
	if(element_id >= 0) {
	  // search for dof insertion point
	  std::size_t j = 0;
	  for(; j < edge_pattern.rows(); ++j) {
	    std::array<int, D::n_vertices_per_edge> e {};
	    for(std::size_t k = 0; k < D::n_vertices_per_edge; ++k) {
	      e[k] = mesh.elements()(element_id, edge_pattern(j,k));
	    }
	    std::sort(e.begin(), e.end()); // normalize edge ordering
	    if((*edge).node_ids() == e) break;
	  }
	  dofs_(element_id, D::n_vertices + j) = next;
	  if((*edge).on_boundary()) boundary_set.insert(next);

	  // insert any internal dofs, if any (for cubic or higher order) + insert n_dof_per_edge dofs (for cubic or hiher)
	}	
      }
      next++;
    }

    n_dofs_ = next;   // store number of unknowns
    // update boundary
    boundary_dofs_ = DMatrix<int>::Zero(n_dofs_, 1);
    boundary_dofs_.topRows(mesh.boundary().rows()) = mesh.boundary();
    for (auto it = boundary_set.begin(); it != boundary_set.end(); ++it) {
        boundary_dofs_(*it, 0) = 1;
    }
  }
  return;
}


  
// produce the matrix of dof coordinates
template <typename D, typename E, typename F, typename... Ts>
DMatrix<double> FEMSolverBase<D, E, F, Ts...>::dofs_coords(const D& mesh) {
    enumerate_dofs(mesh);
    if constexpr (fem_order == 1)
        return mesh.nodes();   // for order 1 dofs coincide with mesh vertices
    else {
        // allocate space
        DMatrix<double> coords;
        coords.resize(n_dofs_, D::embedding_dimension);
        coords.topRows(mesh.n_nodes()) = mesh.nodes();       // copy coordinates of elements' vertices
        std::unordered_set<std::size_t> visited;             // set of already visited dofs
        std::array<SVector<D::local_dimension + 1>, n_dof_per_element> ref_coords =
          ReferenceElement<D::local_dimension, fem_order>().bary_coords;

        // cycle over all mesh elements
        for (const auto& e : mesh) {
            // extract dofs related to element with ID i
            auto dofs = dofs_.row(e.ID());
            for (std::size_t j = D::n_vertices; j < n_dof_per_element; ++j) {   // cycle on non-vertex points
                if (visited.find(dofs[j]) == visited.end()) {                   // not yet mapped dof
                    // map points from reference to physical element
		  static constexpr int M = D::local_dimension;
                    coords.row(dofs[j]) = e.barycentric_matrix() * ref_coords[j].template tail<M>() + e.coords()[0];
                    visited.insert(dofs[j]);
                }
            }
        }
        return coords;
    }
}

  
}   // namespace core
}   // namespace fdapde

#endif   // __FEM_SOLVER_BASE_H__
