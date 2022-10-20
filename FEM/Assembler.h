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
#include "operators/BilinearFormTraits.h"
using fdaPDE::core::FEM::is_symmetric;

namespace fdaPDE{
namespace core{
namespace FEM{

  template <unsigned int N> using BASIS_TABLE = std::vector<std::vector<ScalarField<static_cast<int>(N)>>>;
  
  // FEM assembler. M local dimension, N embedding dimension, B basis function, I integrator
  template <unsigned int M, unsigned int N, unsigned int R, typename B, typename I>
  class Assembler {
  private:
    constexpr static unsigned n_basis = ct_nnodes(M,R);
    const Mesh<M, N, R>& mesh_; // mesh
    const I& integrator_; // quadrature rule to approximate integrals
    B referenceBasis_{}; // functional basis over reference N-dimensional unit simplex
    const BASIS_TABLE<N>& basis_; // basis system over the entire domain
    
  public:
    Assembler(const Mesh<M, N, R>& mesh, const BASIS_TABLE<N>& basis, const I& integrator) :
      mesh_(mesh), basis_(basis), integrator_(integrator) {};
    // assemble discretization matrix
    template <typename E>
    Eigen::SparseMatrix<double> assemble(const E& bilinearForm);
    // assemble forcing vector
    template <typename F>
    Eigen::Matrix<double, Eigen::Dynamic, 1> forcingTerm(const F& f);
  };

#include "Assembler.tpp"

}}}
#endif // __ASSEMBLER_H__
