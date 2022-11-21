#ifndef __ASSEMBLER_H__
#define __ASSEMBLER_H__

#include <Eigen/Sparse>
#include <Eigen/src/Core/util/Constants.h>
#include <cstddef>
#include <limits>
#include <memory>

#include "../utils/CompileTime.h"
#include "../utils/fields/VectorField.h"
using fdaPDE::core::VectorField;
#include "../utils/fields/ScalarField.h"
using fdaPDE::core::ScalarField;
#include "../utils/fields/FieldPtrs.h"
using fdaPDE::core::ScalarPtr;
using fdaPDE::core::VectorPtr;
using fdaPDE::core::MatrixPtr;
#include "../MESH/Mesh.h"
using fdaPDE::core::MESH::Mesh;
#include "../MESH/Element.h"
using fdaPDE::core::MESH::Element;
using fdaPDE::core::MESH::ct_nnodes;
// FEM module includes
#include "integration/Integrator.h"
using fdaPDE::core::FEM::Integrator;
#include "basis/LagrangianBasis.h"
#include "basis/MultivariatePolynomial.h"
using fdaPDE::core::FEM::LagrangianBasis;
using fdaPDE::core::FEM::MultivariatePolynomial;
#include "basis/BasisTable.h"
using fdaPDE::core::FEM::BASIS_TABLE;
#include "operators/BilinearFormTraits.h"
using fdaPDE::core::FEM::is_symmetric;

namespace fdaPDE{
namespace core{
namespace FEM{
  
  // FEM assembler. M local dimension, N embedding dimension, B basis function, I integrator
  template <unsigned int M, unsigned int N, unsigned int R, typename B, typename I>
  class Assembler {
  private:
    constexpr static unsigned n_basis = ct_nnodes(M,R);
    const Mesh<M, N, R>& mesh_; // mesh
    const I& integrator_; // quadrature rule used in integrals approzimation
    B referenceBasis_{}; // functional basis over reference N-dimensional unit simplex
    std::size_t dof_; // overall number of unknowns in the FEM linear system
    const DMatrix<int>& dof_table_; // for each element the associated degrees of freedom
    
  public:
    Assembler(const Mesh<M,N,R>& mesh, const I& integrator) :
      mesh_(mesh), integrator_(integrator), dof_(mesh_.dof()), dof_table_(mesh.dof_table()) {};
    // assemble discretization matrix
    template <typename E>
    Eigen::SparseMatrix<double> assemble(const E& bilinearForm);
    // assemble forcing vector
    template <typename F>
    Eigen::Matrix<double, Eigen::Dynamic, 1> forcingTerm(const F& f);

    std::size_t dof() const { return dof_; }
    
  };

#include "Assembler.tpp"

}}}
#endif // __ASSEMBLER_H__
