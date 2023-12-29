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

#include <fdaPDE/fields.h>
#include <fdaPDE/finite_elements.h>
#include <fdaPDE/mesh.h>
#include <fdaPDE/utils.h>
#include <gtest/gtest.h>   // testing framework

#include <cstddef>
using fdapde::core::advection;
using fdapde::core::dt;
using fdapde::core::Element;
using fdapde::core::FEM;
using fdapde::core::fem_order;
using fdapde::core::laplacian;
using fdapde::core::make_pde;
using fdapde::core::PDE;
using fdapde::core::ScalarField;

#include "utils/mesh_loader.h"
using fdapde::testing::MeshLoader;
#include "utils/utils.h"
using fdapde::testing::almost_equal;
using fdapde::testing::DOUBLE_TOLERANCE;

// test to globally check the correctness of PDE solver

// check error of approximated solution by an order 1 FEM within theoretical expectations
TEST(fem_pde_test, laplacian_isotropic_order1) {
    // exact solution
    auto solution_expr = [](SVector<2> x) -> double { return x[0] + x[1]; };

    MeshLoader<Mesh2D> unit_square("unit_square");
    auto L = -laplacian<FEM>();
    // instantiate a type-erased wrapper for this pde
    PDE<decltype(unit_square.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> pde_(unit_square.mesh, L);
    
    // compute boundary condition and exact solution
    DMatrix<double> nodes_ = pde_.dof_coords();
    DMatrix<double> dirichlet_bc(nodes_.rows(), 1);
    DMatrix<double> solution_ex(nodes_.rows(), 1);

    for (int i = 0; i < nodes_.rows(); ++i) {
        dirichlet_bc(i) = solution_expr(nodes_.row(i));
        solution_ex(i) = solution_expr(nodes_.row(i));
    }
    // set dirichlet conditions
    pde_.set_dirichlet_bc(dirichlet_bc);

    // request quadrature nodes and evaluate forcing on them
    DMatrix<double> quadrature_nodes = pde_.quadrature_nodes();
    DMatrix<double> f = DMatrix<double>::Zero(quadrature_nodes.rows(), 1);
    pde_.set_forcing(f);
    // init solver and solve differential problem
    pde_.init();
    pde_.solve();
    // check computed error within theoretical expectations
    DMatrix<double> error_ = solution_ex - pde_.solution();
    double error_L2 = (pde_.mass() * error_.cwiseProduct(error_)).sum();
    EXPECT_TRUE(error_L2 < DOUBLE_TOLERANCE);
}

// check error of approximated solution by an order 2 FEM within theoretical expectations
TEST(fem_pde_test, laplacian_isotropic_order2_callable_force) {
    // exact solution
    auto solution_expr = [](SVector<2> x) -> double { return 1. - x[0] * x[0] - x[1] * x[1]; };
    // non-zero forcing term
    auto forcing_expr = [](SVector<2> x) -> double { return 4.0; };
    ScalarField<2> forcing(forcing_expr);   // wrap lambda expression in ScalarField object

    MeshLoader<Mesh2D> unit_square("unit_square");
    auto L = -laplacian<FEM>();
    // initialize PDE with callable forcing term
    PDE<decltype(unit_square.mesh), decltype(L), ScalarField<2>, FEM, fem_order<2>> pde_(unit_square.mesh, L, forcing);
    // compute boundary condition and exact solution
    DMatrix<double> nodes_ = pde_.dof_coords();
    DMatrix<double> dirichlet_bc(nodes_.rows(), 1);
    DMatrix<double> solution_ex(nodes_.rows(), 1);

    for (int i = 0; i < nodes_.rows(); ++i) {
        dirichlet_bc(i) = solution_expr(nodes_.row(i));
        solution_ex(i) = solution_expr(nodes_.row(i));
    }
    // set dirichlet conditions
    pde_.set_dirichlet_bc(dirichlet_bc);
    // init solver and solve differential problem
    pde_.init();
    pde_.solve();
    // check computed error within theoretical expectations
    DMatrix<double> error_ = solution_ex - pde_.solution();
    double error_L2 = (pde_.mass() * error_.cwiseProduct(error_)).sum();
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
    MeshLoader<Mesh2D> unit_square("unit_square");
    PDE<decltype(unit_square.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> pde_(unit_square.mesh);
    pde_.set_differential_operator(L);

    // compute boundary condition and exact solution
    DMatrix<double> nodes_ = pde_.dof_coords();
    DMatrix<double> solution_ex(nodes_.rows(), 1);

    for (int i = 0; i < nodes_.rows(); ++i) { solution_ex(i) = solutionExpr(nodes_.row(i)); }
    // set dirichlet conditions
    DMatrix<double> dirichletBC = DMatrix<double>::Zero(nodes_.rows(), 1);
    pde_.set_dirichlet_bc(dirichletBC);

    // request quadrature nodes and evaluate forcing on them
    DMatrix<double> quadrature_nodes = pde_.quadrature_nodes();
    DMatrix<double> f = DMatrix<double>::Zero(quadrature_nodes.rows(), 1);
    for (int i = 0; i < quadrature_nodes.rows(); ++i) { f(i) = forcingExpr(quadrature_nodes.row(i)); }
    pde_.set_forcing(f);

    // init solver and solve differential problem
    pde_.init();
    pde_.solve();

    // check computed error
    DMatrix<double> error_ = solution_ex - pde_.solution();
    double error_L2 = (pde_.mass() * error_.cwiseProduct(error_)).sum();
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
    auto L = -laplacian<FEM>() + advection<FEM>(beta_);   // -\Delta + dot(beta_, \nabla)
    // load sample mesh for order 1 basis
    MeshLoader<Mesh2D> unit_square("unit_square");

    PDE<decltype(unit_square.mesh), decltype(L), ScalarField<2>, FEM, fem_order<2>> pde_(unit_square.mesh, L, forcing);

    // compute boundary condition and exact solution
    DMatrix<double> nodes_ = pde_.dof_coords();
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
    double error_L2 = (pde_.mass() * error_.cwiseProduct(error_)).sum();
    EXPECT_TRUE(error_L2 < 1e-7);
}

// check error of approximated solution by an order 2 FEM for the parabolic problem
//   dt(u) -\Laplacian(u) = f(x,t)   in \Omega
//   u = g                           on boundary
//   u0 = h                          in \Omega at times t0
// where \Omega is the [0,1] x [0,1] 2D unit square,
// f(x, t) = (8*pi*pi-1.)*sin(2*pi*x[0])*sin(2*pi*x[1])*exp(-t)
// g(x, t) = sin(2*pi*x[0])*sin(2*pi*x[1])*std::exp(-t)
// h(x)    = sin(2*pi*x[0])*sin(2*pi*x[1])*std::exp(-t)
TEST(fem_pde_test, parabolic_isotropic_order2) {
    // exact solution
    constexpr double pi = 3.14159265358979323846;
    int M = 101;
    DMatrix<double> times(M, 1);
    double time_max = 1.;
    for (int j = 0; j < M; ++j) { times(j) = time_max / (M - 1) * j; }

    int num_refinements = 1;
    DMatrix<double> error_L2 = DMatrix<double>::Zero(M, num_refinements);

    auto solution_expr = [](SVector<2> x, double t) -> double {
        return std::sin(2 * pi * x[0]) * std::sin(2 * pi * x[1]) * std::exp(-t);
    };
    auto forcing_expr = [](SVector<2> x, double t) -> double {
        return (8 * pi * pi - 1.) * std::sin(2 * pi * x[0]) * std::sin(2 * pi * x[1]) * std::exp(-t);
    };

    MeshLoader<Mesh2D> unit_square("unit_square");
    auto L = dt<FEM>() - laplacian<FEM>();
    PDE<decltype(unit_square.mesh), decltype(L), DMatrix<double>, FEM, fem_order<2>> pde_(unit_square.mesh, times);
    pde_.set_differential_operator(L);

    // compute boundary condition and exact solution
    DMatrix<double> nodes_ = pde_.dof_coords();
    DMatrix<double> dirichlet_bc(nodes_.rows(), M);
    DMatrix<double> solution_ex(nodes_.rows(), M);
    DMatrix<double> initial_condition(nodes_.rows(), 1);

    for (int i = 0; i < nodes_.rows(); ++i) {
        for (int j = 0; j < M; ++j) {
            dirichlet_bc(i, j) = solution_expr(nodes_.row(i), times(j));
            solution_ex(i, j) = solution_expr(nodes_.row(i), times(j));
        }
    }
    // dirichlet_bc = DMatrix<double>::Zero(nodes_.rows(),M);

    for (int i = 0; i < nodes_.rows(); ++i) { initial_condition(i) = solution_expr(nodes_.row(i), times(0)); }
    // set dirichlet conditions
    pde_.set_dirichlet_bc(dirichlet_bc);

    // set initial condition
    pde_.set_initial_condition(initial_condition);

    // request quadrature nodes and evaluate forcing on them
    DMatrix<double> quadrature_nodes = pde_.quadrature_nodes();
    DMatrix<double> f(quadrature_nodes.rows(), M);
    for (int i = 0; i < quadrature_nodes.rows(); ++i) {
        for (int j = 0; j < M; ++j) { f(i, j) = forcing_expr(quadrature_nodes.row(i), times(j)); }
    }
    pde_.set_forcing(f);
    // init solver and solve differential problem
    pde_.init();
    pde_.solve();

    // check computed error within theoretical expectations
    DMatrix<double> error_(nodes_.rows(), 1);
    for (int j = 0; j < M; ++j) {
        error_ = solution_ex.col(j) - pde_.solution().col(j);
        error_L2(j, 0) = (pde_.mass() * error_.cwiseProduct(error_)).sum();
    }

    EXPECT_TRUE(error_L2.maxCoeff() < 1e-7);
}

// check the convergence rate (fixed time step, FEM order 1) for the parabolic problem
//   dt(u) -\Laplacian(u) = f(x,t)   in \Omega
//   u = g                           on boundary
//   u0 = h                          in \Omega at times t0
// where \Omega is the [0,1] x [0,1] 2D unit square,
// f(x, t) = (8*pi*pi-1.)*sin(2*pi*x[0])*sin(2*pi*x[1])*exp(-t)
// g(x, t) = sin(2*pi*x[0])*sin(2*pi*x[1])*std::exp(-t)
// h(x)    = sin(2*pi*x[0])*sin(2*pi*x[1])*std::exp(-t)
TEST(fem_pde_test, parabolic_isotropic_order1_convergence) {
    // exact solution
    constexpr double pi = 3.14159265358979323846;
    int M = 31;
    DMatrix<double> times(M, 1);
    double time_max = 1.;
    for (int j = 0; j < M; ++j) { times(j) = time_max / (M - 1) * j; }

    int num_refinements = 4;
    DMatrix<int> N(num_refinements, 1);   // number of refinements
    N << 16, 32, 64, 128;
    DMatrix<double> order(num_refinements - 1, 1);
    DMatrix<double> error_L2 = DMatrix<double>::Zero(M, num_refinements);

    auto solution_expr = [](SVector<2> x, double t) -> double {
        return std::sin(2 * pi * x[0]) * std::sin(2 * pi * x[1]) * std::exp(-t);
    };
    auto forcing_expr = [](SVector<2> x, double t) -> double {
        return (8 * pi * pi - 1.) * std::sin(2 * pi * x[0]) * std::sin(2 * pi * x[1]) * std::exp(-t);
    };

    for (int n = 0; n < num_refinements; ++n) {
        std::string domain_name = "unit_square_" + std::to_string(N(n));
        MeshLoader<Mesh2D> unit_square(domain_name);
        auto L = dt<FEM>() - laplacian<FEM>();
        PDE<decltype(unit_square.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> pde_(unit_square.mesh, times);
        pde_.set_differential_operator(L);

        // compute boundary condition and exact solution
        DMatrix<double> nodes_ = pde_.dof_coords();
        DMatrix<double> dirichlet_bc(nodes_.rows(), M);
        DMatrix<double> solution_ex(nodes_.rows(), M);
        DMatrix<double> initial_condition(nodes_.rows(), 1);

        for (int i = 0; i < nodes_.rows(); ++i) {
            for (int j = 0; j < M; ++j) {
                dirichlet_bc(i, j) = solution_expr(nodes_.row(i), times(j));
                solution_ex(i, j) = solution_expr(nodes_.row(i), times(j));
            }
        }
        // dirichlet_bc = DMatrix<double>::Zero(nodes_.rows(),M);

        for (int i = 0; i < nodes_.rows(); ++i) { initial_condition(i) = solution_expr(nodes_.row(i), times(0)); }
        // set dirichlet conditions
        pde_.set_dirichlet_bc(dirichlet_bc);

        // set initial condition
        pde_.set_initial_condition(initial_condition);

        // request quadrature nodes and evaluate forcing on them
        DMatrix<double> quadrature_nodes = pde_.quadrature_nodes();
        DMatrix<double> f(quadrature_nodes.rows(), M);
        for (int i = 0; i < quadrature_nodes.rows(); ++i) {
            for (int j = 0; j < M; ++j) { f(i, j) = forcing_expr(quadrature_nodes.row(i), times(j)); }
        }
        pde_.set_forcing(f);
        // init solver and solve differential problem
        pde_.init();
        pde_.solve();

        // check computed error within theoretical expectations
        DMatrix<double> error_(nodes_.rows(), 1);
        for (int j = 0; j < M; ++j) {
            error_ = solution_ex.col(j) - pde_.solution().col(j);
            error_L2(j, n) = std::sqrt((pde_.mass() * error_.cwiseProduct(error_)).sum());
        }
    }

    // check estimated convergence rate
    for (int n = 1; n < num_refinements; ++n) {
        order(n - 1) = std::log2(error_L2(M - 1, n - 1) / error_L2(M - 1, n));
        EXPECT_TRUE(floor(order(n - 1)) == 2);
    }
}
