#ifndef __FEM_BASE_SOLVER_H__
#define __FEM_BASE_SOLVER_H__

#include <memory>
#include "../../utils/Symbols.h"
#include "../Assembler.h"
using fdaPDE::core::FEM::Assembler;
#include "../basis/LagrangianBasis.h"
using fdaPDE::core::FEM::LagrangianBasis;
#include "../integration/Integrator.h"
using fdaPDE::core::FEM::Integrator;

namespace fdaPDE{
namespace core{
namespace FEM{

  // forward declarations
  struct FEMStandardSpaceSolver;
  struct FEMStandardSpaceTimeSolver;
  // trait to select the space-only or the space-time version of the PDE standard solver
  template <typename E> struct pde_standard_solver_selector {
    using type = typename std::conditional<is_parabolic<E>::value,
					   FEMStandardSpaceTimeSolver, FEMStandardSpaceSolver>::type;
  };
  // PDE forward declaration
  template <unsigned int M, unsigned int N, unsigned int R, typename E, typename F, typename B, typename I, typename S> class PDE;

  // base class for the definition of a general solver based on the Finite Element Method
  class FEMBaseSolver{
  protected:
    std::shared_ptr<DMatrix<double>> solution_; // vector of coefficients of the approximate solution written in terms of the chosen basis
    std::shared_ptr<DMatrix<double>> force_; // right-hand side of the linear system giving the FEM solution
    std::shared_ptr<SpMatrix<double>> R1_; // result of the discretization of the bilinear form
    std::shared_ptr<SpMatrix<double>> R0_; // mass matrix, needed by components in higher levels of the architecture

    // impose boundary conditions
    template <unsigned int M, unsigned int N, unsigned int R, typename E, typename F, typename B, typename I, typename S>
    void imposeBoundaryConditions(const PDE<M,N,R,E,F,B,I,S>& pde);    
  public:
    // flag used to notify is something was wrong during computation
    bool success = true;
    // constructor
    FEMBaseSolver() = default;
    // getters
    std::shared_ptr<DMatrix<double>> solution() const { return solution_; }
    std::shared_ptr<DMatrix<double>> force() const { return force_; }
    std::shared_ptr<SpMatrix<double>> R1() const { return R1_; }
    std::shared_ptr<SpMatrix<double>> R0() const { return R0_; }

    // initializes internal FEM solver status
    // M, N and E are parameters releated to the PDE, B and I indicates respectively the functional basis and the integrator
    // to use during the numerical resolution of the problem.
    template <unsigned int M, unsigned int N, unsigned int R, typename E, typename F, typename B, typename I, typename S>
    void init(const PDE<M,N,R,E,F,B,I,S>& pde);
  };

  // fill internal data structures required by FEM to solve the problem.
  template <unsigned int M, unsigned int N, unsigned int R, typename E, typename F, typename B, typename I, typename S>
  void FEMBaseSolver::init(const PDE<M,N,R,E,F,B,I,S>& pde) {
    Assembler<M, N, R, B, I> assembler(pde.domain(), pde.basis(), pde.integrator()); // create assembler object
    // fill discretization matrix for current operator
    R1_ = std::make_shared<SpMatrix<double>>( assembler.assemble(pde.bilinearForm()) );
    R1_->makeCompressed();
    // fill forcing vector
    force_ = std::make_shared<DMatrix<double>>();
    force_->resize(assembler.dof(), 1); //pde.forcingData().cols());
    force_->col(0) = assembler.forcingTerm(pde.forcingData());

    // SPACE-TIME PROBLEMS CURRENTLY NOT SUPPORTED DUE TO THE FACT WE NOW ACCEPT ALSO CALLABLE AS FORCING
    
    // for(std::size_t i = 0; i < pde.forcingData().cols(); ++i){
    //   force_->col(i) = assembler.forcingTerm(pde.forcingData().col(i)); // rhs of FEM linear system
    // }

    // R0_ is a mass matrix ([R0]_{ij} = \int_{\Omega} \phi_i \phi_j). This quantity can be obtained by computing
    // the discretization of the Identity() operator
    R0_ = std::make_shared<SpMatrix<double>>( assembler.assemble(Identity()) );
    return;
  }

  template <unsigned int M, unsigned int N, unsigned int R, typename E, typename F, typename B, typename I, typename S>
  void FEMBaseSolver::imposeBoundaryConditions(const PDE<M,N,R,E,F,B,I,S>& pde){
    // impose homogeneous dirichlet boundary condition by default to remove not necessary degrees of freedom from FEM linear system R1_
    for(size_t i = 0; i < R1_->rows(); ++i){
      if(pde.domain().isOnBoundary(i)){ 
	R1_->row(i) *= 0;       // zero all entries of this row
	R1_->coeffRef(i,i) = 1; // set diagonal element to 1 to impose equation u_j = b_j

	// boundaryDatum is a pair (nodeID, boundary value)
	double boundaryDatum = pde.boundaryData().empty() ? 0 : pde.boundaryData().at(i)[0];
	force_->coeffRef(i,0) = boundaryDatum; // impose boundary value (fallback to homogenenous dirichlet)
      }
    }
    return;
  }

}}}
#endif // __FEM_BASE_SOLVER_H__
