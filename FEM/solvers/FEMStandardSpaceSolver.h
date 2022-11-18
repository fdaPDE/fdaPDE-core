#ifndef __FEM_STANDARD_SPACE_SOLVER_H__
#define __FEM_STANDARD_SPACE_SOLVER_H__

#include "../../utils/Symbols.h"
#include "../Assembler.h"
using fdaPDE::core::FEM::Assembler;
#include "FEMBaseSolver.h"
using fdaPDE::core::FEM::FEMBaseSolver;

namespace fdaPDE{
namespace core{
namespace FEM{

  struct FEMStandardSpaceSolver : public FEMBaseSolver{
    // constructor
    FEMStandardSpaceSolver() = default;

    // solves the PDE using the classical FEM approach:
    //    * compute the discretization matrix R1_ relative to a bilinear form E using:
    //      - some finite element basis B
    //      - the integrator I for numerical approximation of integrals
    //    * compute the discretization of the forcing vector b
    //    * solve the linear system R1_*u = b
    // at the end u is the searched PDE approximation
    template <unsigned int M, unsigned int N, unsigned int R, typename E, typename F, typename B, typename I, typename S>
    void solve(const PDE<M,N,R,E,F,B,I,S>& pde);
  };

  template <unsigned int M, unsigned int N, unsigned int R, typename E, typename F, typename B, typename I, typename S>
  void FEMStandardSpaceSolver::solve(const PDE<M,N,R,E,F,B,I,S>& pde){
    this->init(pde); // init solver for this PDE
    this->imposeBoundaryConditions(pde); // impose boundary conditions on forcing vector and R1_ matrix

    // define eigen system solver, use sparse LU decomposition.
    Eigen::SparseLU<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> solver;
    solver.compute(*this->R1_);
    // stop if something was wrong...
    if(solver.info()!=Eigen::Success) {
      this->success = false;
      return;
    }
    // solve FEM linear system: R1_*solution_ = force_;
    this->solution_ = std::make_shared<DMatrix<double>>( solver.solve(*this->force_) );
    return;
  }

}}}
#endif // __SPACE_SOLVER_H__
