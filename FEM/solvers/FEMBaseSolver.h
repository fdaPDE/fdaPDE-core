#ifndef __FEM_BASE_SOLVER_H__
#define __FEM_BASE_SOLVER_H__

#include "../../utils/Symbols.h"
#include "../PDE.h"

class FEMBaseSolver{
protected:
  DMatrix<double> solution_;                // vector of coefficients of the approximate solution written in terms of the chosen basis
  DMatrix<double> forcingVector_;           // right-hand side of the linear system giving the FEM solution
  Eigen::SparseMatrix<double> R1_;  // result of the discretization of the bilinear form, also known as R1.
  Eigen::SparseMatrix<double> R0_;  // mass matrix, needed by components in higher levels of the architecture, known as R0.

  // some informations about solution error

  // initializes internal FEM solver status
  // M, N and E are parameters releated to the PDE, while B and I indicates respectively the functional basis and the integrator
  // to use during problem solution.
  template <unsigned int M, unsigned int N, typename E, typename B, typename I> 
  void init(const PDE<M, N, E>& pde, const B& basis, const I& integrator);
  
public:
  FEMBaseSolver() = default;
  
  // flag used to notify is something was wrong during computation of solution
  bool success = true;

  // getters
  DMatrix<double> getSolution() const { return solution_; }
  DMatrix<double> getForce() const { return forcingVector_; }
  Eigen::SparseMatrix<double> getR1() const { return R1_; }
  Eigen::SparseMatrix<double> getR0() const { return R0_; }
};

// fill internal data structures required by FEM to solve the problem.
template <unsigned int M, unsigned int N, typename E, typename B, typename I>
void FEMBaseSolver::init(const PDE<M, N, E>& pde, const B& basis, const I& integrator) {
  Assembler<M, N, B, I> assembler(pde.getDomain(), basis, integrator); // create assembler object
  R1_ = assembler.assemble(pde.getBilinearForm());       // fill discretization matrix for current operator
  // SparseQR solver needs its matrix in compressed form (see Eigen documentation for details)
  R1_.makeCompressed();
  forcingVector_.resize(pde.getDomain().nodes(), pde.getForcingData().cols());

  for(std::size_t i = 0; i < pde.getForcingData().cols(); ++i){
    forcingVector_.col(i) = assembler.forcingTerm(pde.getForcingData().col(i)); // fill discretization of rhs for FEM linear system
  }

  // R0_ is a mass matrix ([R0]_{ij} = \int_{\Omega} \phi_i \phi_j). This quantity can be obtained by computing
  // the discretization of the Identity() operator
  R0_ = assembler.assemble(Identity());
  return;
}

#endif // __FEM_BASE_SOLVER_H__
