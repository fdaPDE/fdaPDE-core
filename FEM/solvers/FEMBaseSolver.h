#ifndef __FEM_BASE_SOLVER_H__
#define __FEM_BASE_SOLVER_H__

#include <memory>
#include <stdexcept>
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
    using type = typename std::conditional<
      is_parabolic<E>::value, FEMStandardSpaceTimeSolver, FEMStandardSpaceSolver>::type;
  };
  // PDE forward declaration
  template <unsigned int M, unsigned int N, unsigned int R, typename E,
	    typename F, typename B, typename I, typename S>
  class PDE;

  // base class for the definition of a general solver based on the Finite Element Method
  class FEMBaseSolver {
  protected:
    DMatrix<double> solution_; // vector of coefficients of the approximate solution 
    DMatrix<double> force_;    // right-hand side of the FEM linear system
    SpMatrix<double> R1_;      // result of the discretization of the bilinear form
    SpMatrix<double> R0_;      // mass matrix, i.e. discretization of the identity operator
    bool init_ = false;        // set to true by init() at the end of solver initialization
  public:
    // flag used to notify is something was wrong during computation
    bool success = true;
    // constructor
    FEMBaseSolver() = default;
    // getters
    const DMatrix<double>& solution() const { return solution_; }
    const DMatrix<double>& force() const { return force_; }
    const SpMatrix<double>& R1() const { return R1_; }
    const SpMatrix<double>& R0() const { return R0_; }

    // initializes internal FEM solver status
    template <unsigned int M, unsigned int N, unsigned int R, typename E,
	      typename F, typename B, typename I, typename S>
    void init(const PDE<M,N,R,E,F,B,I,S>& pde) {
      Assembler<M, N, R, B, I> assembler(pde.domain(), pde.integrator()); // create assembler object
      // fill discretization matrix for current operator
      R1_ = assembler.assemble(pde.bilinearForm());
      R1_.makeCompressed();
    
      // fill forcing vector
      std::size_t n = pde.domain().dof(); // degrees of freedom in space
      std::size_t m; // number of time points
    
      if constexpr(!std::is_base_of<ScalarBase, F>::value){
	m = pde.forcingData().cols();
	force_.resize(n*m, 1);
	force_.block(0,0, n,1) = assembler.forcingTerm(pde.forcingData().col(0));
      }else{
	// TODO: find a way to allow for space-time callable forcing. We cannot deduce the number of time points
	// from a ScalarField<> object, have to request an additional parameter which contains the number of time points m
	// and must force user to supply a space-time field (a field with an integer parameter t which is incremented by
	// solver during its operations... maybe a TimeField<> specialization of ScalarField<> ?? )
	m = 1;
	force_.resize(n*m, 1);
	force_.block(0,0, n,1) = assembler.forcingTerm(pde.forcingData());
      }
      // iterate over time steps if a space-time PDE is supplied
      if constexpr(is_parabolic<E>::value){
	for(std::size_t i = 1; i < m; ++i){
	  force_.block(n*i,0, n,1) = assembler.forcingTerm(pde.forcingData().col(i));
	}
      }

      // compute mass matrix [R0]_{ij} = \int_{\Omega} \phi_i \phi_j by discretization of the identity operator
      R0_ = assembler.assemble(Identity());
      init_ = true;
      return;
    }

    // impose dirichlet boundary conditions
    template <unsigned int M, unsigned int N, unsigned int R, typename E,
	      typename F, typename B, typename I, typename S>
    void imposeDirichletBC(const PDE<M,N,R,E,F,B,I,S>& pde) {
      if(!init_) throw std::runtime_error("solver must be initialized first!");
      for(auto it = pde.domain().boundary_begin(); it != pde.domain().boundary_end(); ++it){
	R1_.row(*it) *= 0; // zero all entries of this row
	R1_.coeffRef(*it,*it) = 1; // set diagonal element to 1 to impose equation u_j = b_j
	
	// boundaryData is a map (nodeID, boundary value).
	// TODO: currently only space-only case supported (reason of [0] below)
	force_.coeffRef(*it,0) = pde.boundaryData().at(*it)[0]; // impose boundary value on forcing term
      }
      return;
    };
  };

}}}
#endif // __FEM_BASE_SOLVER_H__
