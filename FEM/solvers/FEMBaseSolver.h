#ifndef __FEM_BASE_SOLVER_H__
#define __FEM_BASE_SOLVER_H__

#include "../../utils/Symbols.h"
#include "../PDE.h"
using fdaPDE::core::FEM::PDE;

namespace fdaPDE{
namespace core{
namespace FEM{

  // base class for the definition of a general solver based on the Finite Element Method
  class FEMBaseSolver{
  protected:
    DMatrix<double> solution_; // vector of coefficients of the approximate solution written in terms of the chosen basis
    DMatrix<double> forcingVector_; // right-hand side of the linear system giving the FEM solution
    Eigen::SparseMatrix<double> R1_; // result of the discretization of the bilinear form, also known as R1.
    Eigen::SparseMatrix<double> R0_; // mass matrix, needed by components in higher levels of the architecture, known as R0.

    // initializes internal FEM solver status
    // M, N and E are parameters releated to the PDE, B and I indicates respectively the functional basis and the integrator
    // to use during the numerical resolution of the problem.
    template <unsigned int M, unsigned int N, unsigned int R, typename E, typename B, typename I> 
    void init(const PDE<M, N, R, E>& pde, const B& basis, const I& integrator);
  public:
        // flag used to notify is something was wrong during computation
    bool success = true;
    // constructor
    FEMBaseSolver() = default;
    // getters
    DMatrix<double> solution() const { return solution_; }
    DMatrix<double> force() const { return forcingVector_; }
    Eigen::SparseMatrix<double> R1() const { return R1_; }
    Eigen::SparseMatrix<double> R0() const { return R0_; }
  };

  // fill internal data structures required by FEM to solve the problem.
  template <unsigned int M, unsigned int N, unsigned int R, typename E, typename B, typename I>
  void FEMBaseSolver::init(const PDE<M, N, R, E>& pde, const B& basis, const I& integrator) {
    Assembler<M, N, R, B, I> assembler(pde.domain(), basis, integrator); // create assembler object
    R1_ = assembler.assemble(pde.bilinearForm());       // fill discretization matrix for current operator
    // SparseQR solver needs its matrix in compressed form (see Eigen documentation for details)
    R1_.makeCompressed();
    forcingVector_.resize(pde.domain().nodes(), pde.forcingData().cols());

    for(std::size_t i = 0; i < pde.forcingData().cols(); ++i){
      forcingVector_.col(i) = assembler.forcingTerm(pde.forcingData().col(i)); // fill discretization of rhs for FEM linear system
    }

    // R0_ is a mass matrix ([R0]_{ij} = \int_{\Omega} \phi_i \phi_j). This quantity can be obtained by computing
    // the discretization of the Identity() operator
    R0_ = assembler.assemble(Identity());
    return;
  }

}}}
#endif // __FEM_BASE_SOLVER_H__
