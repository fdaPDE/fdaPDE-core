#include <cstddef>
#include <gtest/gtest-typed-test.h>
#include <gtest/gtest.h> // testing framework
#include <unsupported/Eigen/SparseExtra>

#include "../../src/utils/symbols.h"
#include "../../src/mesh/element.h"
using fdapde::core::Element;

#include "../../src/finite_elements/basis/lagrangian_basis.h"
using fdapde::core::LagrangianBasis;
#include "../../src/finite_elements/operators/laplacian.h"
#include "../../src/finite_elements/operators/gradient.h"
using fdapde::core::Laplacian;
using fdapde::core::Gradient;
#include "../../src/finite_elements/pde.h"
using fdapde::core::PDE;

#include "utils/mesh_loader.h"
using fdapde::testing::MeshLoader;
#include "utils/utils.h"
using fdapde::testing::almost_equal;
using fdapde::testing::DOUBLE_TOLERANCE;

// test to globally check the correctness of PDE solver

// check error of approximated solution by an order 1 FEM within theoretical expectations
TEST(PDESolutionsTest, LaplacianIsotropicOrder1) {
  // exact solution
  auto solutionExpr = [](SVector<2> x) -> double { 
    return x[0] + x[1];
  };
  // differential operator
  auto differential_operator = -1.0*Laplacian();
  // load sample mesh for order 1 basis
  MeshLoader<Mesh2D<>> unit_square("unit_square");
  
  PDE<2,2,1, decltype(differential_operator), DMatrix<double>> pde_(unit_square.mesh);
  pde_.set_differential_operator(differential_operator);
  
  // compute boundary condition and exact solution
  DMatrix<double> nodes_ = unit_square.mesh.dof_coords();
  DMatrix<double> dirichletBC(nodes_.rows(), 1);
  DMatrix<double> solution_ex(nodes_.rows(), 1);

  for(int i=0; i<nodes_.rows(); ++i){
    dirichletBC(i) = solutionExpr(nodes_.row(i));
    solution_ex(i) = solutionExpr(nodes_.row(i));
  }
  // set dirichlet conditions
  pde_.set_dirichlet_bc(dirichletBC);

  // request quadrature nodes and evaluate forcing on them
  DMatrix<double> quadrature_nodes = pde_.integrator().quadrature_nodes(unit_square.mesh);
  DMatrix<double> f = DMatrix<double>::Zero(quadrature_nodes.rows(), 1);
  pde_.set_forcing(f);
  // init solver and solve differential problem
  pde_.init();
  pde_.solve();

  // check computed error within theoretical expectations
  DMatrix<double> error_ = solution_ex - pde_.solution();
  double error_L2 = (pde_.R0() * error_.cwiseProduct(error_)).sum();
  EXPECT_TRUE(error_L2 < DOUBLE_TOLERANCE);
}

// check error of approximated solution by an order 2 FEM within theoretical expectations
TEST(PDESolutionsTest, LaplacianIsotropicOrder2) {
  // exact solution  
  auto solutionExpr = [](SVector<2> x) -> double { 
    return 1.-x[0]*x[0] - x[1]*x[1];
  };
  // non-zero forcing term
  auto forcingExpr = [](SVector<2> x) -> double { 
    return 4.0 ;
  };
  ScalarField<2> forcing(forcingExpr); // wrap lambda expression in ScalarField object
  // differential operator  
  auto differential_operator = -1.0*Laplacian();
  // load sample mesh for order 2 basis  
  MeshLoader<Mesh2D<2>> unit_square("unit_square");
  
  PDE<2,2,2, decltype(differential_operator), DMatrix<double>> pde_(unit_square.mesh);  
  pde_.set_differential_operator(differential_operator);
  
  // compute boundary condition and exact solution
  DMatrix<double> nodes_ = unit_square.mesh.dof_coords();
  DMatrix<double> dirichletBC(nodes_.rows(), 1);
  DMatrix<double> solution_ex(nodes_.rows(), 1);

  for(int i=0; i<nodes_.rows(); ++i){
    dirichletBC(i) = solutionExpr(nodes_.row(i));
    solution_ex(i) = solutionExpr(nodes_.row(i));
  }
  // set dirichlet conditions    
  pde_.set_dirichlet_bc(dirichletBC);

  // request quadrature nodes and evaluate forcing on them
  DMatrix<double> quadrature_nodes = pde_.integrator().quadrature_nodes(unit_square.mesh);
  DMatrix<double> f = 4.*DMatrix<double>::Ones(quadrature_nodes.rows(), 1);
  pde_.set_forcing(f);
  // init solver and solve differential problem 
  pde_.init();
  pde_.solve();
  
  DMatrix<double> error_ = solution_ex - pde_.solution();
  double error_L2 = (pde_.R0() * error_.cwiseProduct(error_)).sum();

  // check computed error within theoretical expectations
  EXPECT_TRUE(error_L2 < DOUBLE_TOLERANCE);
}

// check error of approximated solution by an order 2 FEM within theoretical expectations
TEST(PDESolutionsTest, LaplacianIsotropicOrder2_CallableForce) {
  // exact solution    
  auto solutionExpr = [](SVector<2> x) -> double { 
    return 1.-x[0]*x[0] - x[1]*x[1];
  };
  // non-zero forcing term
  auto forcingExpr = [](SVector<2> x) -> double { 
    return 4.0;
  };
  ScalarField<2> forcing(forcingExpr); // wrap lambda expression in ScalarField object
  // differential operator
  auto differential_operator = -1.0*Laplacian();
  // load sample mesh for order 2 basis      
  MeshLoader<Mesh2D<2>> unit_square("unit_square");
  
  // initialize PDE with callable forcing term
  PDE<2,2,2, decltype(differential_operator), ScalarField<2>> pde_(unit_square.mesh, differential_operator, forcing);

  // compute boundary condition and exact solution  
  DMatrix<double> nodes_ = unit_square.mesh.dof_coords();
  DMatrix<double> dirichletBC(nodes_.rows(), 1);
  DMatrix<double> solution_ex(nodes_.rows(), 1);

  for(int i=0; i<nodes_.rows(); ++i){
    dirichletBC(i) = solutionExpr(nodes_.row(i));
    solution_ex(i) = solutionExpr(nodes_.row(i));
  }
  // set dirichlet conditions        
  pde_.set_dirichlet_bc(dirichletBC);
  // init solver and solve differential problem    
  pde_.init();
  pde_.solve();

  // check computed error within theoretical expectations  
  DMatrix<double> error_ = solution_ex - pde_.solution();
  double error_L2 = (pde_.R0() * error_.cwiseProduct(error_)).sum();
  EXPECT_TRUE(error_L2 < DOUBLE_TOLERANCE);
}

// check error of approximated solution by an order 1 FEM for the advection-diffusion problem
//    -\Laplacian(u) - \alpha*(du/dx) = \gamma sin(pi*y)   on \Omega
//    u = 0   on boundary
// where \Omega is the [0,1] x [0,1] 2D unit square
TEST(PDESolutionsTest, AdvectionDiffusionIsotropicOrder1) {
  // setup problem parameter
  constexpr double pi = 3.14159265358979323846;
  double alpha_ = 1.0;
  double gamma_ = pi;   

  // define exact solution
  // - \gamma/(\pi^2)*(p*\exp{\lambda_1*x} + (1-p)*\exp{\lambda_2*x} - 1)*sin(\pi*y)
  double lambda1 = -alpha_/2 - std::sqrt((alpha_/2)*(alpha_/2) + pi*pi);
  double lambda2 = -alpha_/2 + std::sqrt((alpha_/2)*(alpha_/2) + pi*pi);

  double p_ = (1-std::exp(lambda2))/(std::exp(lambda1)-std::exp(lambda2));
  
  auto solutionExpr = [&gamma_, &lambda1, &lambda2, &p_](SVector<2> x) -> double { 
    return -gamma_/(pi*pi) * ( p_ * std::exp(lambda1 * x[0]) + (1 - p_) * std::exp(lambda2 * x[0]) - 1. ) * std::sin(pi * x[1]);
  };
  
  // non-zero forcing term
  auto forcingExpr = [&gamma_](SVector<2> x) -> double { 
    return gamma_ * std::sin(pi * x[1]);
  };
  
  // differential operator
  SVector<2> beta_;
  beta_ << -alpha_, 0.;
  auto differential_operator = -1.0*Laplacian() + Gradient(beta_);
  // load sample mesh for order 1 basis
  MeshLoader<Mesh2D<>> unit_square("unit_square");
  PDE<2,2,1, decltype(differential_operator), DMatrix<double>> pde_(unit_square.mesh);
  pde_.set_differential_operator(differential_operator);

  // compute boundary condition and exact solution
  DMatrix<double> nodes_ = unit_square.mesh.dof_coords();
  DMatrix<double> solution_ex(nodes_.rows(), 1);

  for(int i=0; i<nodes_.rows(); ++i){
    solution_ex(i) = solutionExpr(nodes_.row(i));
  }
  // set dirichlet conditions        
  DMatrix<double> dirichletBC = DMatrix<double>::Zero(nodes_.rows(), 1);
  pde_.set_dirichlet_bc(dirichletBC);
  
  // request quadrature nodes and evaluate forcing on them
  DMatrix<double> quadrature_nodes = pde_.integrator().quadrature_nodes(unit_square.mesh);
  DMatrix<double> f = DMatrix<double>::Zero(quadrature_nodes.rows(), 1);
  
  for(int i=0; i<quadrature_nodes.rows(); ++i){
    f(i) = forcingExpr(quadrature_nodes.row(i));
  }
  pde_.set_forcing(f);
  
  // init solver and solve differential problem
  pde_.init();
  pde_.solve();
  
  // check computed error
  DMatrix<double> error_ = solution_ex - pde_.solution();
  double error_L2 = (pde_.R0() * error_.cwiseProduct(error_)).sum();
  EXPECT_TRUE(error_L2 < 1e-5);
}

// check error of approximated solution by an order 2 FEM for the advection-diffusion problem
//    -\Laplacian(u) - \alpha*(du/dx) = \gamma sin(pi*y)   on \Omega
//    u = 0   on boundary
// where \Omega is the [0,1] x [0,1] 2D unit square
TEST(PDESolutionsTest, AdvectionDiffusionIsotropicOrder2) {
  // define problem parameters
  constexpr double pi = 3.14159265358979323846;
  double alpha_ = 1.0;
  double gamma_ = pi;   

  // define exact solution
  // - \gamma/(\pi^2)*(p*\exp{\lambda_1*x} + (1-p)*\exp{\lambda_2*x} - 1)*sin(\pi*y)
  double lambda1 = -alpha_/2 - std::sqrt((alpha_/2)*(alpha_/2) + pi*pi);
  double lambda2 = -alpha_/2 + std::sqrt((alpha_/2)*(alpha_/2) + pi*pi);

  double p_ = (1-std::exp(lambda2))/(std::exp(lambda1)-std::exp(lambda2));
  
  auto solutionExpr = [&gamma_, &lambda1, &lambda2, &p_](SVector<2> x) -> double { 
    return -gamma_/(pi*pi) * ( p_ * std::exp(lambda1 * x[0]) + (1 - p_) * std::exp(lambda2 * x[0]) - 1. ) * std::sin(pi * x[1]);
  };
  
  // non-zero forcing term
  auto forcingExpr = [&gamma_](SVector<2> x) -> double { 
    return gamma_ * std::sin(pi * x[1]);
  };
  ScalarField<2> forcing(forcingExpr); // wrap lambda expression in ScalarField object
  
  // differential operator
  SVector<2> beta_;
  beta_ << -alpha_, 0.;
  auto differential_operator = -Laplacian() + Gradient(beta_);
  // load sample mesh for order 1 basis
  MeshLoader<Mesh2D<2>> unit_square("unit_square");
  
  PDE<2,2,2, decltype(differential_operator), ScalarField<2>> pde_(unit_square.mesh, differential_operator, forcing);
  
  // compute boundary condition and exact solution
  DMatrix<double> nodes_ = unit_square.mesh.dof_coords();
  DMatrix<double> solution_ex(nodes_.rows(), 1);

  for(int i=0; i<nodes_.rows(); ++i){
    solution_ex(i) = solutionExpr(nodes_.row(i));
  }
  // set dirichlet conditions        
  DMatrix<double> dirichletBC = DMatrix<double>::Zero(nodes_.rows(), 1);
  pde_.set_dirichlet_bc(dirichletBC);

  // init solver and solve differential problem
  pde_.init();
  pde_.solve();
  
  // check computed error
  DMatrix<double> error_ = solution_ex - pde_.solution();
  double error_L2 = (pde_.R0() * error_.cwiseProduct(error_)).sum();
  EXPECT_TRUE(error_L2 < 1e-7);
}
