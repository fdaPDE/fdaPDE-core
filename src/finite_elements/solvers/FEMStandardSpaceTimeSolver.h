#ifndef __FEM_STANDARD_SPACE_TIME_SOLVER_H__
#define __FEM_STANDARD_SPACE_TIME_SOLVER_H__

#include <stdexcept>
#include "../../utils/Symbols.h"
#include "../Assembler.h"
using fdaPDE::core::FEM::Assembler;
#include "FEMBaseSolver.h"
using fdaPDE::core::FEM::FEMBaseSolver;

namespace fdaPDE{
namespace core{
namespace FEM{

  struct FEMStandardSpaceTimeSolver : public FEMBaseSolver {
    // constructor
    FEMStandardSpaceTimeSolver() = default;

    // solves the PDE using a FEM discretization in space and a finite difference discretization in time (forward-euler scheme)
    template <unsigned int M, unsigned int N, unsigned int R, typename E,
	      typename F, typename B, typename I, typename S>
    void solve(const PDE<M,N,R,E,F,B,I,S>& pde, double deltaT);
  };

  // use forward-euler to discretize the time derivative. Under this approximation we get a discretization matrix for the PDE operator
  // equal to K = [M/deltaT + A] (forward Euler scheme)
  template <unsigned int M, unsigned int N, unsigned int R, typename E,
	    typename F, typename B, typename I, typename S>
  void FEMStandardSpaceTimeSolver::solve(const PDE<M,N,R,E,F,B,I,S>& pde, double deltaT) {
    if(!init_) throw std::runtime_error("solver must be initialized first!");
    
    // define eigen system solver, use SparseLU decomposition.
    Eigen::SparseLU<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> solver;
    std::size_t n = pde.domain().dof(); // degrees of freedom in space
    std::size_t m = pde.forcingData().cols(); // number of iterations for time loop
    
    this->solution_.resize(pde.domain().dof(), m-1);
    this->solution_.col(0) = pde.initialCondition(); // impose initial condition
    DVector<double> rhs = ((this->R0_)/deltaT)*pde.initialCondition() + this->force_.block(0,0, n,1);  
  
    // Observe that K is time invariant only for homogeneous boundary conditions. In general we need to recompute K at each time instant,
    // anyway we can avoid the recomputation of K at each iteration by just keeping overwriting it at the boundary indexes positions.
    Eigen::SparseMatrix<double> K = (this->R0_)/deltaT + this->R1_; // build system matrix

    // prepare system matrix to handle dirichlet boundary conditions
    for(auto it = pde.domain().boundary_begin(); it != pde.domain().boundary_end(); ++it){
      K.row(*it) *= 0; // zero all entries of this row
      K.coeffRef(*it,*it) = 1; // set diagonal element to 1 to impose equation u_j = b_j
    }
    // execute temporal loop to solve ODE system via forward-euler scheme
    for(std::size_t i = 1; i < m-1; ++i){
      // impose boundary conditions
      for(auto it = pde.domain().boundary_begin(); it != pde.domain().boundary_end(); ++it){
	rhs[*it] = pde.boundaryData().at(*it)[i];;
      }
      solver.compute(K); // prepare solver
      if(solver.info()!=Eigen::Success){ // stop if something was wrong...
	this->success = false;
	return;
      }
      DVector<double> u_i = solver.solve(rhs); // solve linear system
      this->solution_.col(i) = u_i; // append time step solution to solution matrix
      rhs = ((this->R0_)/deltaT)*u_i + this->force_.block(n*i,0, n,1); // update rhs for next iteration
    }
    return;
  }
  
}}}
#endif // __FEM_STANDARD_SPACE_TIME_SOLVER_H__
