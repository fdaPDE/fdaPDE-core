#ifndef __PDE_H__
#define __PDE_H__

#include "../utils/Symbols.h"
#include "../MESH/Mesh.h"
using fdaPDE::core::MESH::Mesh;
using fdaPDE::core::MESH::ct_nnodes;
#include "basis/LagrangianBasis.h"
using fdaPDE::core::FEM::LagrangianBasis;
#include "basis/BasisTable.h"
using fdaPDE::core::FEM::BASIS_TABLE;
#include "solvers/FEMBaseSolver.h"
#include "solvers/FEMStandardSpaceSolver.h"
#include "solvers/FEMStandardSpaceTimeSolver.h"
using fdaPDE::core::FEM::pde_standard_solver_selector;
using fdaPDE::core::FEM::FEMStandardSpaceSolver;
using fdaPDE::core::FEM::FEMStandardSpaceTimeSolver;
#include "integration/Integrator.h"
using fdaPDE::core::FEM::Integrator;
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace fdaPDE{
namespace core{
namespace FEM{

  // base class for any PDE object (used as tag in higher components of the architecture)
  struct PDEBase {};
  
  // top level class to describe a partial differential equation Lf = u.
  // N and M are the problem dimensions: these informations are strictly releated to the mesh used for domain discretization.
  // In particular N is the dimension of the problem domain, M is the local dimension of the manifold describing the domain.
  template <unsigned int M, // local dimension of the mesh 
	    unsigned int N, // dimension of the mesh embedding space
	    unsigned int R, // order of the mesh
	    typename E, // differential operator L
	    typename F, // forcing term u
	    typename B = LagrangianBasis<M,N,R>, // functional basis
	    typename I = Integrator<M,R>, // quadrature rule used for approximation of integrals
	    typename S = typename pde_standard_solver_selector<E>::type>
  class PDE : public PDEBase {
  private:
    const Mesh<M,N,R>& domain_; // problem domain
    E bilinearForm_; // the differential operator of the problem in its weak formulation
    static_assert(std::is_base_of<BilinearFormExpr<E>, E>::value);
    F forcingData_; // forcing data
    static_assert(std::is_same<DMatrix<double>, F>::value || std::is_base_of<ScalarExpr<F>, F>::value);
    DVector<double> initialCondition_{}; // initial condition, used in space-time problems only
    
    // memorize boundary data in a sparse structure, by storing the index of the boundary node and the relative boundary value.
    // a vector is used as value type to handle both space and space-time problems
    typedef std::unordered_map<unsigned, DVector<double>> boundary_map;
    boundary_map boundaryData_{};

    I integrator_{}; // integrator used for approximation of integrals
    B referenceBasis_{}; // basis defined over the reference unit simplex
    BASIS_TABLE<M,N,R,B> basis_{}; // basis built over the whole domain_
    S solver_{}; // numerical scheme used to find a solution to this PDE

    void buildBasis_(); // initializes BASIS_TABLE
  public:
    // minimal constructor, use below setters to complete the construction of a PDE object
    PDE(const Mesh<M,N,R>& domain) : domain_(domain) { buildBasis_(); }
    void setForcing(const F& forcingData) { forcingData_ = forcingData; }
    void setBilinearForm(E bilinearForm) { bilinearForm_ = bilinearForm; }
    // full constructors
    PDE(const Mesh<M,N,R>& domain, E bilinearForm, const F& forcingData);
    PDE(const Mesh<M,N,R>& domain, E bilinearForm, const F& forcingData, const B& basis, const I& integrator);
    
    // setters for boundary conditions
    void setDirichletBC(const DMatrix<double>& data);
    //void setNeumannBC();
    void setInitialCondition(const DVector<double>& data) { initialCondition_ = data; };
  
    // getters
    const Mesh<M,N,R>& domain() const { return domain_; }
    E bilinearForm() const { return bilinearForm_; }
    const F& forcingData() const { return forcingData_; }
    const DVector<double>& initialCondition() const { return initialCondition_; }
    const boundary_map& boundaryData() const { return boundaryData_; };
    const I& integrator() const { return integrator_; }
    DMatrix<double> quadratureNodes() const { return integrator_.quadratureNodes(domain_); }; // returns all quadrature nodes on the mesh
    
    // solution informations produced by call to .solve()
    std::shared_ptr<DMatrix<double>>  solution() const { return solver_.solution(); };
    std::shared_ptr<DMatrix<double>>  force() const { return solver_.force(); }; // rhs of FEM linear system
    std::shared_ptr<SpMatrix<double>> R1() const { return solver_.R1(); };
    std::shared_ptr<SpMatrix<double>> R0() const { return solver_.R0(); };
    const BASIS_TABLE<M,N,R,B>& basis() const { return basis_; }
    
    void init();  // computes matrices R1, R0 and forcing vector without solving the FEM linear system.
    void solve(); // entry point for PDE solver. Solves the pde (i.e. after this call solution() will contain valid data)

    // expose compile time informations
    static constexpr std::size_t local_dimension = M;
    static constexpr std::size_t embedding_dimension = N;
    static constexpr std::size_t basis_order = R;
  };

  // argument deduction rule for PDE object
  template <unsigned int M, unsigned int N, unsigned int R, typename E, typename F, typename I, typename B>
  PDE(const Mesh<M,N,R>& domain, E bilinearForm, const F& forcing, const B& basis, const I& integrator)
    -> PDE<M, N, R, decltype(bilinearForm), F, I, B>;

#include "PDE.tpp"

}}}
#endif // __PDE_H__
