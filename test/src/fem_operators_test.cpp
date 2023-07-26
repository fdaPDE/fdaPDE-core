#include <cstddef>
#include <gtest/gtest-typed-test.h>
#include <gtest/gtest.h> // testing framework

#include "../../src/utils/symbols.h"
#include "../../src/mesh/element.h"
using fdapde::core::Element;
#include "../../src/fields/matrix_expressions.h"

#include "utils/mesh_loader.h"
using fdapde::testing::MESH_TYPE_LIST;
using fdapde::testing::MeshLoader;
#include "utils/utils.h"
using fdapde::testing::almost_equal;

#include "../../src/utils/compile_time.h"
#include "../../src/finite_elements/integration/integrator.h"
using fdapde::core::Integrator;
#include "../../src/finite_elements/integration/integrator_tables.h"
#include "../../src/finite_elements/basis/lagrangian_basis.h"
using fdapde::core::LagrangianBasis;
#include "../../src/finite_elements/operators/laplacian.h"
using fdapde::core::Laplacian;
#include "../../src/fields/field_ptrs.h"
using fdapde::core::MatrixConst;
using fdapde::core::ScalarPtr;
using fdapde::core::VectorPtr;
using fdapde::core::MatrixPtr;

#include "../../src/finite_elements/operators/divergence.h"
using fdapde::core::Divergence;
using fdapde::core::div;

// tests related to integration of fields produced by bilinear forms

// test integration of Laplacian weak form for a LagrangianBasis of order 2
TEST(BilinearFormsTest, LaplacianQuadraticElement) {
  // load sample mesh, request an order 2 basis support
  MeshLoader<Mesh2D<2>> CShaped("c_shaped");
  auto e = CShaped.mesh.element(175); // reference element for this test
  Integrator<2,2> integrator{}; 

  // define differential operator
  auto L = -Laplacian();
  // define functional space
  LagrangianBasis<2,2> basis{};
  
  using BasisType = typename LagrangianBasis<2,2>::ElementType;
  using NablaType = decltype(std::declval<BasisType>().derive());
  BasisType buff_psi_i, buff_psi_j; // basis functions \psi_i, \psi_j
  NablaType buff_nabla_psi_i, buff_nabla_psi_j; // gradient of basis functions \nabla \psi_i, \nabla \psi_j
  MatrixConst<2,2,2> buff_invJ; // (J^{-1})^T, being J the inverse of the barycentric matrix relative to element e
  // prepare buffer to be sent to bilinear form
  auto mem_buffer = std::make_tuple
    (ScalarPtr(&buff_psi_i), ScalarPtr(&buff_psi_j), VectorPtr(&buff_nabla_psi_i),
     VectorPtr(&buff_nabla_psi_j), MatrixPtr(&buff_invJ));
    
  // develop bilinear form expression in an integrable field
  auto f = L.integrate(mem_buffer); 
  buff_invJ = e.inv_barycentric_matrix().transpose();
  
  std::vector<double> integrals;
  
  for(size_t i = 0; i < basis.size(); ++i){
    buff_psi_i = basis[i];
    buff_nabla_psi_i = buff_psi_i.derive();
    for(size_t j = 0; j < basis.size(); ++j){
      buff_psi_j = basis[j];
      buff_nabla_psi_j = buff_psi_j.derive();
      double value = integrator.template integrate<decltype(L)>(e, f); // perform integration of bilinear form
      integrals.push_back(value);
    }
  }

  // define vector of expected results, each row defines the result of the integration \int_e [\nabla \psi_i * \nabla \psi_j] in the order
  // basis functions are traversed on the reference element. Recall that the enumeration of basis functions is
  // 1 -> (0,0), 2 -> (1,0), 3 -> (0,1), 4 -> (0.5, 0), 5-> (0, 0.5), 6 -> (0.5, 0.5)
  std::vector<double> expected({
      0.7043890316492852,  0.1653830261033185,  0.0694133177797771, -0.6615321044132733, -0.2776532711191089,  0.0000000000000013, // \psi_1 \psi_j
      0.1653830261033185,  0.7043890316492852,  0.0694133177797769, -0.6615321044132735,  0.0000000000000003, -0.2776532711191076, // \psi_2 \psi_j
      0.0694133177797771,  0.0694133177797769,  0.4164799066786617,  0.0000000000000002, -0.2776532711191083, -0.2776532711191075, // \psi_3 \psi_j
     -0.6615321044132733, -0.6615321044132735,  0.0000000000000002,  2.4336772933029756, -0.5553065422382126, -0.5553065422382162, // \psi_5 \psi_j
     -0.2776532711191089,  0.0000000000000003, -0.2776532711191083, -0.5553065422382126,  2.4336772933029738, -1.3230642088265447, // \psi_4 \psi_j
      0.0000000000000013, -0.2776532711191075, -0.2776532711191076, -0.5553065422382162, -1.3230642088265447,  2.4336772933029751  // \psi_6 \psi_j
    }); 

  // check for double equality of all computed integrals
  for(std::size_t i = 0; i < expected.size(); ++i){
    EXPECT_TRUE( almost_equal(integrals[i], expected[i]) );
  }
}
