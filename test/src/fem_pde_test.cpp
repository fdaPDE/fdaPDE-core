// This file is part of fdaPDE, a C++ library for physics-informed
// spatial and functional data analysis.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#include <gtest/gtest.h>   // testing framework
#include <cstddef>

#include <fdaPDE/utils.h>
#include <fdaPDE/fields.h>
#include <fdaPDE/mesh.h>
#include <fdaPDE/finite_elements.h>
using fdapde::core::Element;
using fdapde::core::LagrangianElement;
using fdapde::core::PDE;
using fdapde::core::ScalarField;
using fdapde::core::advection;
using fdapde::core::laplacian;
using fdapde::core::FEM;

#include "utils/mesh_loader.h"
using fdapde::testing::MeshLoader;
#include "utils/utils.h"
using fdapde::testing::almost_equal;
using fdapde::testing::DOUBLE_TOLERANCE;

// test to globally check the correctness of PDE solver

// check error of approximated solution by an order 1 FEM within theoretical expectations
TEST(fem_pde_test, laplacian_isotropic_order1) {
    // exact solution
    auto solutionExpr = [](SVector<2> x) -> double { return x[0] + x[1]; };
    
    MeshLoader<Mesh2D<>> unit_square("unit_square");
    auto L = -laplacian<FEM>();
    PDE<decltype(unit_square.mesh), decltype(L), DMatrix<double>, FEM> pde_(unit_square.mesh);
    pde_.set_differential_operator(L);

    // compute boundary condition and exact solution
    DMatrix<double> nodes_ = unit_square.mesh.dof_coords();
    DMatrix<double> dirichletBC(nodes_.rows(), 1);
    DMatrix<double> solution_ex(nodes_.rows(), 1);

    for (int i = 0; i < nodes_.rows(); ++i) {
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
TEST(fem_pde_test, laplacian_isotropic_order2_callable_force) {
    // exact solution
    auto solutionExpr = [](SVector<2> x) -> double { return 1. - x[0] * x[0] - x[1] * x[1]; };
    // non-zero forcing term
    auto forcingExpr = [](SVector<2> x) -> double { return 4.0; };
    ScalarField<2> forcing(forcingExpr);   // wrap lambda expression in ScalarField object

    MeshLoader<Mesh2D<2>> unit_square("unit_square");
    auto L = -laplacian<FEM>();
    // initialize PDE with callable forcing term
    PDE<decltype(unit_square.mesh), decltype(L), ScalarField<2>, FEM> pde_(unit_square.mesh, L, forcing);

    // compute boundary condition and exact solution
    DMatrix<double> nodes_ = unit_square.mesh.dof_coords();
    DMatrix<double> dirichletBC(nodes_.rows(), 1);
    DMatrix<double> solution_ex(nodes_.rows(), 1);

    for (int i = 0; i < nodes_.rows(); ++i) {
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
TEST(fem_pde_test, advection_diffusion_isotropic_order1) {
    // setup problem parameter
    constexpr double pi = 3.14159265358979323846;
    double alpha_ = 1.0;
    double gamma_ = pi;

    // define exact solution
    // - \gamma/(\pi^2)*(p*\exp{\lambda_1*x} + (1-p)*\exp{\lambda_2*x} - 1)*sin(\pi*y)
    double lambda1 = -alpha_ / 2 - std::sqrt((alpha_ / 2) * (alpha_ / 2) + pi * pi);
    double lambda2 = -alpha_ / 2 + std::sqrt((alpha_ / 2) * (alpha_ / 2) + pi * pi);

    double p_ = (1 - std::exp(lambda2)) / (std::exp(lambda1) - std::exp(lambda2));

    auto solutionExpr = [&gamma_, &lambda1, &lambda2, &p_](SVector<2> x) -> double {
        return -gamma_ / (pi * pi) * (p_ * std::exp(lambda1 * x[0]) + (1 - p_) * std::exp(lambda2 * x[0]) - 1.) *
               std::sin(pi * x[1]);
    };

    // non-zero forcing term
    auto forcingExpr = [&gamma_](SVector<2> x) -> double { return gamma_ * std::sin(pi * x[1]); };

    // differential operator
    SVector<2> beta_;
    beta_ << -alpha_, 0.;
    auto L = -laplacian<FEM>() + advection<FEM>(beta_);
    // load sample mesh for order 1 basis
    MeshLoader<Mesh2D<>> unit_square("unit_square");
    PDE<decltype(unit_square.mesh), decltype(L), DMatrix<double>, FEM> pde_(unit_square.mesh);
    pde_.set_differential_operator(L);

    // compute boundary condition and exact solution
    DMatrix<double> nodes_ = unit_square.mesh.dof_coords();
    DMatrix<double> solution_ex(nodes_.rows(), 1);

    for (int i = 0; i < nodes_.rows(); ++i) { solution_ex(i) = solutionExpr(nodes_.row(i)); }
    // set dirichlet conditions
    DMatrix<double> dirichletBC = DMatrix<double>::Zero(nodes_.rows(), 1);
    pde_.set_dirichlet_bc(dirichletBC);

    // request quadrature nodes and evaluate forcing on them
    DMatrix<double> quadrature_nodes = pde_.integrator().quadrature_nodes(unit_square.mesh);
    DMatrix<double> f = DMatrix<double>::Zero(quadrature_nodes.rows(), 1);
    for (int i = 0; i < quadrature_nodes.rows(); ++i) { f(i) = forcingExpr(quadrature_nodes.row(i)); }
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
TEST(fem_pde_test, advection_diffusion_isotropic_order2) {
    // define problem parameters
    constexpr double pi = 3.14159265358979323846;
    double alpha_ = 1.0;
    double gamma_ = pi;

    // define exact solution
    // - \gamma/(\pi^2)*(p*\exp{\lambda_1*x} + (1-p)*\exp{\lambda_2*x} - 1)*sin(\pi*y)
    double lambda1 = -alpha_ / 2 - std::sqrt((alpha_ / 2) * (alpha_ / 2) + pi * pi);
    double lambda2 = -alpha_ / 2 + std::sqrt((alpha_ / 2) * (alpha_ / 2) + pi * pi);

    double p_ = (1 - std::exp(lambda2)) / (std::exp(lambda1) - std::exp(lambda2));

    auto solutionExpr = [&gamma_, &lambda1, &lambda2, &p_](SVector<2> x) -> double {
        return -gamma_ / (pi * pi) * (p_ * std::exp(lambda1 * x[0]) + (1 - p_) * std::exp(lambda2 * x[0]) - 1.) *
               std::sin(pi * x[1]);
    };

    // non-zero forcing term
    auto forcingExpr = [&gamma_](SVector<2> x) -> double { return gamma_ * std::sin(pi * x[1]); };
    ScalarField<2> forcing(forcingExpr);   // wrap lambda expression in ScalarField object

    // differential operator
    SVector<2> beta_;
    beta_ << -alpha_, 0.;
    auto L = -laplacian<FEM>() + advection<FEM>(beta_); // -\Delta + dot(beta_, \nabla)
    // load sample mesh for order 1 basis
    MeshLoader<Mesh2D<2>> unit_square("unit_square");

    PDE<decltype(unit_square.mesh), decltype(L), ScalarField<2>, FEM> pde_(unit_square.mesh, L, forcing);

    // compute boundary condition and exact solution
    DMatrix<double> nodes_ = unit_square.mesh.dof_coords();
    DMatrix<double> solution_ex(nodes_.rows(), 1);

    for (int i = 0; i < nodes_.rows(); ++i) { solution_ex(i) = solutionExpr(nodes_.row(i)); }
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
