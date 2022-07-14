#ifndef __PDE_H__
#define __PDE_H__

#include "../utils/Symbols.h"
#include "../MESH/Mesh.h"
using fdaPDE::core::MESH::Mesh;
#include "operators/BilinearFormTraits.h"
#include "Assembler.h"
#include "operators/Identity.h"

#include <type_traits>
#include <unordered_map>
#include <vector>

// top level class to describe a partial differential equation. PDE objects are used by solvers to obtain a solution to the problem

// forward declarations
class FEMStandardSpaceTimeSolver;
class FEMStandardSpaceSolver;

// trait to select the space-only or the space-time variant of the PDE standard solver
template <typename E> struct pde_standard_solver_selector {
  using type = typename std::conditional<is_parabolic<E>::value,
					 FEMStandardSpaceTimeSolver, FEMStandardSpaceSolver>::type;
};

// N and M are the problem dimensions: these informations are strictly releated to the mesh used for domain discretization.
// In particular N is the dimension of the problem domain, M is the local dimension of the manifold describing the domain.
template <unsigned int M,  // dimension of the mesh embedding space
	  unsigned int N,  // local dimension of the mesh
	  typename E,      // type of the BilinearFormExpr
	  typename Solver = typename pde_standard_solver_selector<E>::type>
class PDE{
private:
  const Mesh<M,N>& domain_;     // problem domain
  E bilinearForm_;              // the differential operator of the problem in its weak formulation
  DMatrix forcingData_{};       // forcing data, a vector is used to handle space-time problems
  DVector initialCondition_{};  // initial condition, used in space-time problems only
  
  // memorize boundary data in a sparse structure, by storing the index of the boundary node and the relative boundary value.
  // a vector is used as mapped type to handle both space and space-time problems
  std::unordered_map<unsigned, DVector> boundaryData_{};

  Solver solver_{}; // the solver used to solve the PDE
public:
  // expose space dimensions to PDE solver
  static constexpr unsigned M_ = M;
  static constexpr unsigned N_ = N;

  // constructor, a DMatrix is accepted as forcingData to handle also space-time problems
  PDE(const Mesh<M,N>& domain, E bilinearForm, const DMatrix& forcingData) :
    domain_(domain), bilinearForm_(bilinearForm), forcingData_(forcingData) {};

  // setters for boundary and initial conditions
  void setDirichletBC(const DMatrix& data);
  //void setNeumannBC();
  void setInitialCondition(const DVector& data) { initialCondition_ = data; };
  
  // getters
  const Mesh<M, N>& getDomain() const { return domain_; }
  E getBilinearForm() const { return bilinearForm_; }
  const DMatrix& getForcingData() const { return forcingData_; }
  const DVector& getInitialCondition() const { return initialCondition_; }
  const std::unordered_map<unsigned, DVector>& getBoundaryData() const { return boundaryData_; };

  // solution informations produced by call to .solve()
  const DMatrix  getSolution() const { return solver_.getSolution(); };
  const DMatrix  getForce() const { return solver_.getForce(); };
  const SpMatrix getR1() const { return solver_.getR1(); };
  const SpMatrix getR0() const { return solver_.getR0(); };

  // entry point for PDE solver.
  template <typename B, typename I, typename... Args>
  void solve(const B& base, const I& integrator, Args... args);
};

// argument deduction rule for PDE object
template <unsigned int M, unsigned int N, typename E>
PDE(const Mesh<M,N>& domain, E bilinearForm, const DVector& forcingData) -> PDE<M, N, decltype(bilinearForm)>;

#include "PDE.tpp"

#endif // __PDE_H__
