#ifndef __FEM_STANDARD_SPACE_SOLVER_H__
#define __FEM_STANDARD_SPACE_SOLVER_H__

#include "../../utils/Symbols.h"
#include "../Assembler.h"
using fdaPDE::core::FEM::Assembler;
#include "../PDE.h"
using fdaPDE::core::FEM::PDE;
#include "FEMBaseSolver.h"
using fdaPDE::core::FEM::FEMBaseSolver;

namespace fdaPDE{
namespace core{
namespace FEM{

  struct FEMStandardSpaceSolver : public FEMBaseSolver{
    // constructor
    FEMStandardSpaceSolver() = default;

    // solves the PDE using the classical FEM approach: compute the discretization matrix R1_ relative to a bilinear form E using some finite element
    // basis B and the forcing vector b, then solves the linear system R1_*u = b, where u is the searched PDE approximation
    template <unsigned int M, unsigned int N, unsigned int R, typename E, typename B, typename I> 
    void solve(const PDE<M, N, R, E>& pde, const B& basis, const I& integrator);
  };

  template <unsigned int M, unsigned int N, unsigned int R, typename E, typename B, typename I> 
  void FEMStandardSpaceSolver::solve(const PDE<M, N, R, E>& pde, const B& basis, const I& integrator){
    this->init(pde, basis, integrator); // init solver for this PDE
    // impose boundary conditions
    for(std::size_t i = 0; i < pde.domain().nodes(); ++i){
      if(pde.domain().isOnBoundary(i)){
	// boundaryDatum is a pair (nodeID, boundary value)
	double boundaryDatum = pde.boundaryData().at(i)[0];

	// To impose a Dirichlet boundary condition means to introduce an equation of the kind u_j = b_j where j is the index
	// of the boundary node and b_j is the boundary value we want to impose on this node. This actually removes one degree
	// of freedom from the system. We do so by zeroing out the j-th row of the stiff matrix and set the corresponding
	// diagonal element to 1
	this->R1_.row(i) *= 0; // zero all entries of this row
	this->R1_.coeffRef(i, i) = 1; // set diagonal element to 1 to impose equation u_j = b_j
	this->forcingVector_(i,0) = boundaryDatum; // impose boundary value
      }
    }
    // define eigen system solver, use QR decomposition.
    Eigen::SparseQR<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> solver;
    solver.compute(this->R1_);
    // stop if something was wrong...
    if(solver.info()!=Eigen::Success) {
      this->success = false;
      return;
    }
    // solve FEM linear system: discretizationMatrix_*solution_ = forcingVector_;
    this->solution_ = solver.solve(this->forcingVector_);  
    return;
  }

}}}
#endif // __SPACE_SOLVER_H__
