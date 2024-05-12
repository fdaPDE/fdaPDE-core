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

#include <cmath>
#include <cstddef>
using fdapde::core::advection;
using fdapde::core::dt;
using fdapde::core::Element;
using fdapde::core::FEM;
using fdapde::core::fem_order;
using fdapde::core::laplacian;
using fdapde::core::NonLinearReaction;
using fdapde::core::LagrangianBasis;
using fdapde::core::non_linear_op;
using fdapde::core::diffusion;
using fdapde::core::make_pde;
using fdapde::core::PDE;
using fdapde::core::ScalarField;
using fdapde::core::PDEparameters;

#include "utils/mesh_loader.h"
using fdapde::testing::MeshLoader;
#include "utils/utils.h"
using fdapde::testing::almost_equal;
using fdapde::testing::DOUBLE_TOLERANCE;


TEST(fem_pde_boundary_condition_test, laplacian_dirichlet_neumann_1D) {
    // exact solution
    auto solution_expr = [](SVector<1> x) -> double { return 1. + x[0]*x[0]; };
    auto forcing_expr = [](SVector<1> x) -> double { return -2.; };
    auto neumann_expr = [](SVector<1> x) -> double { return 2*x[0]; };

    // Robin data
    // double a = 17;
    // double b = 21;
    // auto robin_expr = [&](SVector<1> x) -> double { return a*(1. + x[0]*x[0]) + b*2*x[0]; };

    Mesh1D domain(0.0, 1.0, 10);
    // define the Neumann and Dirichlet boundary with a DMatrix (=0 if Dirichlet, =1 if Neumann, =2 if Robin)
    // considering the unit square,
    // we have Neumann boundary
    DMatrix<short int> boundary_matrix = DMatrix<short int>::Zero(domain.n_nodes(), 1) ; // has all zeros
    boundary_matrix(10, 0) = 1;
    // boundary_matrix(10, 0) = 2;

    auto L = -laplacian<FEM>();
    // instantiate a type-erased wrapper for this pde
    PDE<decltype(domain), decltype(L), DMatrix<double>, FEM, fem_order<1>> pde_(domain, L, boundary_matrix);
    
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
    DMatrix<double> quadrature_nodes = pde_.force_quadrature_nodes();
    DMatrix<double> f(quadrature_nodes.rows(), 1);
    for (int i = 0; i < quadrature_nodes.rows(); ++i) {
        f(i) = forcing_expr(quadrature_nodes.row(i));
    }
    pde_.set_forcing(f);
    DMatrix<double> boundary_quadrature_nodes = pde_.boundary_quadrature_nodes();
    DMatrix<double> f_neumann(boundary_quadrature_nodes.rows(), 1);
    for (auto i=0; i< boundary_quadrature_nodes.rows(); ++i){
        f_neumann(i) = neumann_expr(boundary_quadrature_nodes.row(i));
    }
    pde_.set_neumann_bc(f_neumann);
    // Matrix<double> f_robin(boundary_quadrature_nodes.rows(), 1);
    // for (auto i=0; i< boundary_quadrature_nodes.rows(); ++i){
    //     f_robin(i) = robin_expr(boundary_quadrature_nodes.row(i));
    // }
    // DVector<double> robin_constants(2,1);
    // robin_constants << a, b;
    // pde_.set_robin_bc(f_robin, robin_constants);
    // init solver and solve differential problem
    pde_.init();
    pde_.solve();
    // check computed error within theoretical expectations
    DMatrix<double> error_ = solution_ex - pde_.solution();
    double error_L2 = (pde_.mass() * error_.cwiseProduct(error_)).sum();
    // std::cout << std::sqrt(error_L2) << std::endl;

    // std::ofstream file("ND_1D_P1.txt");    //it will be exported in the current build directory
    // if (file.is_open()){
    //     for(int i = 0; i < pde_.solution().rows(); ++i)
    //         file << pde_.solution()(i) << '\n';
    //     file.close();
    // } else {
    //     std::cerr << "test unable to save solution" << std::endl;
    // }

    EXPECT_TRUE(error_L2 < DOUBLE_TOLERANCE);
}

// check error of approximated solution by an order 1 FEM within theoretical expectations
// apply mixed boundary conditions
TEST(fem_pde_boundary_condition_test, laplacian_dirichlet_neumann_2D) {
    // exact solution
    auto solution_expr = [](SVector<2> x) -> double { return 1. + x[0]*x[0] + 2*x[1]*x[1]; };
    auto neumann_expr = [](SVector<2> x) -> double { return 4*x[1]; };
    auto forcing_expr = [](SVector<2> x) -> double { return -6.; };

    // Robin data
    // double a = 7;
    // double b = 3;
    // auto robin_expr = [&](SVector<2> x) -> double { return a*(1 + x[0]*x[0] + 2*x[1]*x[1]) + b*2*x[0]; };

    MeshLoader<Mesh2D> unit_square("unit_square_16");
    // define the Neumann and Dirichlet boundary with a DMatrix (=0 if Dirichlet, =1 if Neumann, =2 if Robin)
    // considering the unit square,
    // we have Dirichlet boundary when x=0 and x=1 (left and right sides)
    // we have Neumann boundary when y=0 and y=1 (upper and lower sides)
    DMatrix<short int> boundary_matrix = DMatrix<short int>::Zero(unit_square.mesh.n_nodes(), 1) ; // has all zeros
    for (size_t j=1; j<16; ++j) boundary_matrix(j, 0) = 1;
    for (size_t j=273; j<288; ++j) boundary_matrix(j, 0) = 1;
    // DMatrix<short int> boundary_matrix = DMatrix<short int>::Ones(unit_square.mesh.n_nodes(), 1)*2;
    // for (size_t j=0; j<17; ++j) boundary_matrix(j, 0) = 0;
    // for (size_t j=272; j<289; ++j) boundary_matrix(j, 0) = 0;

    auto L = -laplacian<FEM>();
    // instantiate a type-erased wrapper for this pde
    PDE<decltype(unit_square.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> pde_(unit_square.mesh, L, boundary_matrix);
    
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
    DMatrix<double> quadrature_nodes = pde_.force_quadrature_nodes();
    DMatrix<double> f(quadrature_nodes.rows(), 1);
    for (int i = 0; i < quadrature_nodes.rows(); ++i) {
        f(i) = forcing_expr(quadrature_nodes.row(i)); 
    }
    pde_.set_forcing(f);
    DMatrix<double> boundary_quadrature_nodes = pde_.boundary_quadrature_nodes();
    DMatrix<double> f_neumann(boundary_quadrature_nodes.rows(), 1);
    for (auto i=0; i< boundary_quadrature_nodes.rows(); ++i){
        f_neumann(i) = neumann_expr(boundary_quadrature_nodes.row(i));
    }
    pde_.set_neumann_bc(f_neumann);
    // DMatrix<double> f_robin(boundary_quadrature_nodes.rows(), 1);
    // for (auto i=0; i< boundary_quadrature_nodes.rows(); ++i){
    //     f_robin(i) = robin_expr(boundary_quadrature_nodes.row(i));
    // }
    // DVector<double> robin_constants(2,1);
    // robin_constants << a, b;
    // pde_.set_robin_bc(f_robin, robin_constants);
    // init solver and solve differential problem
    pde_.init();
    pde_.solve();
    // check computed error within theoretical expectations
    DMatrix<double> error_ = solution_ex - pde_.solution();
    double error_L2 = (pde_.mass() * error_.cwiseProduct(error_)).sum();
    // std::cout << std::sqrt(error_L2) << std::endl;

    // std::ofstream file("ND_2D_P1.txt");    //it will be exported in the current build directory
    // if (file.is_open()){
    //     for(int i = 0; i < pde_.solution().rows(); ++i)
    //         file << pde_.solution()(i) << '\n';
    //     file.close();
    // } else {
    //     std::cerr << "test unable to save solution" << std::endl;
    // }

    EXPECT_TRUE(error_L2 < DOUBLE_TOLERANCE);
}

// check error of approximated solution by an order 2 FEM within theoretical expectations
// apply mixed boundary conditions
TEST(fem_pde_boundary_condition_test, laplacian_dirichlet_neumann_2D_order2) {
    // exact solution
    auto solution_expr = [](SVector<2> x) -> double { return 1. + x[0]*x[0] + 2*x[1]*x[1]; };
    auto neumann_expr = [](SVector<2> x) -> double { return 4*x[1]; };
    auto forcing_expr = [](SVector<2> x) -> double { return -6.; };

    MeshLoader<Mesh2D> unit_square("unit_square_16");
    // define the boundary with a DMatrix (=0 if Dirichlet, =1 if Neumann, 2= if Robin)
    // considering the unit square,
    // we have Dirichlet boundary when x=0 and x=1 (left and right sides)
    // we have Neumann boundary when y=0 and y=1 (upper and lower sides)
    DMatrix<short int> boundary_matrix = DMatrix<short int>::Zero(unit_square.mesh.n_nodes(), 1);
    for (size_t j=1; j<16; ++j) boundary_matrix(j, 0) = 1;
    for (size_t j=273; j<288; ++j) boundary_matrix(j, 0) = 1;

    auto L = -laplacian<FEM>();
    // instantiate a type-erased wrapper for this pde
    PDE<decltype(unit_square.mesh), decltype(L), DMatrix<double>, FEM, fem_order<2>> pde_(unit_square.mesh, L, boundary_matrix);
    
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
    DMatrix<double> quadrature_nodes = pde_.force_quadrature_nodes();
    DMatrix<double> f(quadrature_nodes.rows(), 1);
    for (int i = 0; i < quadrature_nodes.rows(); ++i) {
        f(i) = forcing_expr(quadrature_nodes.row(i)); 
    }
    pde_.set_forcing(f);
    DMatrix<double> boundary_quadrature_nodes = pde_.boundary_quadrature_nodes();
    DMatrix<double> f_neumann(boundary_quadrature_nodes.rows(), 1);
    for (auto i=0; i< boundary_quadrature_nodes.rows(); ++i){
        f_neumann(i) = neumann_expr(boundary_quadrature_nodes.row(i));
    }
    pde_.set_neumann_bc(f_neumann);
    // init solver and solve differential problem
    pde_.init();
    pde_.solve();
    // check computed error within theoretical expectations
    DMatrix<double> error_ = solution_ex - pde_.solution();
    double error_L2 = (pde_.mass() * error_.cwiseProduct(error_)).sum();
    // std::cout << error_L2 << std::endl;

    // std::ofstream file("ND_2D_P2.txt");    //it will be exported in the current build directory
    // if (file.is_open()){
    //     for(int i = 0; i < pde_.solution().rows(); ++i)
    //         file << pde_.solution()(i) << '\n';
    //     file.close();
    // } else {
    //     std::cerr << "test unable to save solution" << std::endl;
    // }

    EXPECT_TRUE(error_L2 < DOUBLE_TOLERANCE);
}

// check error of approximated solution by an order 1 FEM within theoretical expectations in 2.5D
// apply mixed boundary conditions
TEST(fem_pde_boundary_condition_test, laplacian_dirichlet_neumann_surface) {
    // exact solution
    auto solution_expr = [](SVector<3> x) -> double { return 1. + x[0]*x[0] + 2*x[1]*x[1] + 3*x[2]*x[2]; };
    auto neumann_expr = [](SVector<3> x) -> double {
        if (x[1]==0 || x[1]==1) return 4*x[1]; 
        return 2*x[0];};
    auto forcing_expr = [](SVector<3> x) -> double { return -6.; };

    // Robin data
    // double a = 35;
    // double b = 3;
    // auto robin_expr = [&](SVector<3> x) -> double { return a*(1 + x[0]*x[0] + 2*x[1]*x[1] + 3*x[2]*x[2]) + b*4*x[1]; };

    MeshLoader<SurfaceMesh> domain("unit_square_surface");
    // define the Neumann and Dirichlet boundary with a DMatrix (=0 if Dirichlet, =1 if Neumann, =2 if Robin)
    // we have Neumann boundary when y=0 and y=1 (upper and lower sides)
    DMatrix<short int> boundary_matrix = DMatrix<short int>::Zero(domain.mesh.n_nodes(), 1) ; // has all zeros
    for (size_t j=1; j<10; ++j) boundary_matrix(j, 0) = 1;
    for (size_t j=111; j<120; ++j) boundary_matrix(j, 0) = 1;

    auto L = -laplacian<FEM>();
    // instantiate a type-erased wrapper for this pde
    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> pde_(domain.mesh, L, boundary_matrix);
    
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
    DMatrix<double> quadrature_nodes = pde_.force_quadrature_nodes();
    DMatrix<double> f(quadrature_nodes.rows(), 1);
    for (int i = 0; i < quadrature_nodes.rows(); ++i) {
        f(i) = forcing_expr(quadrature_nodes.row(i)); 
    }
    pde_.set_forcing(f);
    DMatrix<double> boundary_quadrature_nodes = pde_.boundary_quadrature_nodes();
    DMatrix<double> f_neumann(boundary_quadrature_nodes.rows(), 1);
    for (auto i = 0; i < boundary_quadrature_nodes.rows(); ++i){
        f_neumann(i) = neumann_expr(boundary_quadrature_nodes.row(i));
    }
    pde_.set_neumann_bc(f_neumann);
    // DMatrix<double> f_robin(boundary_quadrature_nodes.rows(), 1);
    // for (auto i=0; i< boundary_quadrature_nodes.rows(); ++i){
    //     f_robin(i) = robin_expr(boundary_quadrature_nodes.row(i));
    // }
    // DVector<double> robin_constants(2,1);
    // robin_constants << a, b;
    // pde_.set_robin_bc(f_robin, robin_constants);
    // init solver and solve differential problem
    pde_.init();
    pde_.solve();
    // check computed error within theoretical expectations
    DMatrix<double> error_ = solution_ex - pde_.solution();
    double error_L2 = (pde_.mass() * error_.cwiseProduct(error_)).sum();
    // std::cout << std::sqrt(error_L2) << std::endl;
    EXPECT_TRUE(error_L2 < DOUBLE_TOLERANCE);


    // storing solution 
    // std::ofstream file("surface_N.txt");    //it will be exported in the current build directory
    // if (file.is_open()){
    //     for(int i = 0; i < pde_.solution().rows(); ++i)
    //         file << pde_.solution()(i) << '\n';
    //     file.close();
    // } else {
    //     std::cerr << "surface test unable to save solution" << std::endl;
    // }

}

// check error of approximated solution by an order 1 FEM within theoretical expectations in 3D
// apply mixed boundary conditions
TEST(fem_pde_boundary_condition_test, laplacian_dirichlet_neumann_3D) {
    // exact solution
    auto solution_expr = [](SVector<3> x) -> double { return 1. + x[0]*x[0] + 2*x[1]*x[1] + 3*x[2]*x[2]; };
    auto neumann_expr = [](SVector<3> x) -> double { return 6*x[2]; };
    auto forcing_expr = [](SVector<3> x) -> double { return -12.; };

    // Robin data
    // double a = 35;
    // double b = 3;
    // auto robin_expr = [&](SVector<3> x) -> double { return a*(1. + x[0]*x[0] + 2*x[1]*x[1] + 3*x[2]*x[2]) + b*6*x[2]; };

    MeshLoader<Mesh3D> unit_cube("unit_cube_14");
    // define the Neumann and Dirichlet boundary with a DMatrix (=0 if Dirichlet, =1 if Neumann, =2 if Robin)
    // considering the unit cube,
    // we have Neumann boundary when z=0 and z=1 (upper and lower sides)
    DMatrix<short int> boundary_matrix = DMatrix<short int>::Zero(unit_cube.mesh.n_nodes(), 1) ; // has all zeros
    for (size_t j=1; j<225; ++j) boundary_matrix(j, 0) = 1;
    for (size_t j=3152; j<3375; ++j) boundary_matrix(j, 0) = 1;
    // set the Dirichlet nodes at the edges
    for (size_t j=0; j<15; j++) {
        boundary_matrix(15*j, 0) = 0;      // lower face, points with x=0
        boundary_matrix(14 + j*15, 0) = 0; // lower face, points with x=1
        boundary_matrix(j, 0) = 0;         // lower face, points with y=0
        boundary_matrix(210 + j, 0) = 0;   // lower face, points with y=1

        boundary_matrix(3150 + 15*j, 0) = 0; // upper face, points with x=0
        boundary_matrix(3164 + j*15, 0) = 0; // upper face, points with x=1
        boundary_matrix(3150 + j, 0) = 0;    // upper face, points with y=0
        boundary_matrix(3360 + j, 0) = 0;    // upper face, points with y=1
    }

    auto L = -laplacian<FEM>();
    // instantiate a type-erased wrapper for this pde
    PDE<decltype(unit_cube.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> pde_(unit_cube.mesh, L, boundary_matrix);
    
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
    DMatrix<double> quadrature_nodes = pde_.force_quadrature_nodes();
    DMatrix<double> f(quadrature_nodes.rows(), 1);
    for (int i = 0; i < quadrature_nodes.rows(); ++i) {
        f(i) = forcing_expr(quadrature_nodes.row(i)); 
    }
    pde_.set_forcing(f);
    DMatrix<double> boundary_quadrature_nodes = pde_.boundary_quadrature_nodes();
    DMatrix<double> f_neumann(boundary_quadrature_nodes.rows(), 1);
    for (auto i=0; i< boundary_quadrature_nodes.rows(); ++i){
        f_neumann(i) = neumann_expr(boundary_quadrature_nodes.row(i));
    }
    pde_.set_neumann_bc(f_neumann);
    // DMatrix<double> f_robin(boundary_quadrature_nodes.rows(), 1);
    // for (auto i=0; i< boundary_quadrature_nodes.rows(); ++i){
    //     f_robin(i) = robin_expr(boundary_quadrature_nodes.row(i));
    // }
    // DVector<double> robin_constants(2,1);
    // robin_constants << a, b;
    // pde_.set_robin_bc(f_robin, robin_constants);
    // init solver and solve differential problem
    pde_.init();
    pde_.solve();
    // check computed error within theoretical expectations
    DMatrix<double> error_ = solution_ex - pde_.solution();
    double error_L2 = (pde_.mass() * error_.cwiseProduct(error_)).sum();
    // std::cout << std::sqrt(error_L2) << std::endl;
    EXPECT_TRUE(error_L2 < DOUBLE_TOLERANCE);


    //storing solution 
    // std::ofstream file("cube_DR.txt");    //it will be exported in the current build directory
    // if (file.is_open()){
    //     for(int i = 0; i < pde_.solution().rows(); ++i)
    //         file << std::setprecision(15) << pde_.solution()(i) << '\n';
    //     file.close();
    // } else {
    //     std::cerr << "cubic test unable to save solution" << std::endl;
    // }

}

TEST(fem_pde_boundary_condition_test, space_time_dirichlet_robin_1D) {

    // exact solution
    int M = 101;
    DMatrix<double> times(M, 1);
    double time_max = 1e-3;
    for (int j = 0; j < M; ++j) { times(j) = time_max / (M - 1) * j; }

    int num_refinements = 1;
    DMatrix<double> error_L2 = DMatrix<double>::Zero(M, num_refinements);
    constexpr double pi = 3.14159265358979323846;

    auto solution_expr = [](SVector<1> x, double t) -> double {
        return (std::cos(pi*x[0]) + 2)*std::exp(-t);
    };
    auto forcing_expr = [](SVector<1> x, double t) -> double {
        return std::exp(-t) * (-2*std::cos(pi*x[0]) - 4 + pi*pi*std::cos(pi*x[0])) + (std::cos(pi*x[0]) + 2)*(std::cos(pi*x[0]) + 2)*std::exp(-2*t);
    };
    // auto neumann_expr = [](SVector<1> x, double t) -> double {
    //     return - pi*std::sin(pi*x[0])*std::exp(-t);
    // };
    // Robin data
    double a = 5;
    double b = 3;
    auto robin_expr = [&](SVector<1> x, double t) -> double { 
        return a*(std::cos(pi*x[0]) + 2)*std::exp(-t) - b*pi*std::sin(pi*x[0])*std::exp(-t);
    };

    Mesh1D domain(0.0, 1.0, 10);
    // define the Neumann and Dirichlet boundary with a DMatrix (=0 if Dirichlet, =1 if Neumann, =2 if Robin)
    // considering the unit square,
    // we have Neumann boundary
    DMatrix<short int> boundary_matrix = DMatrix<short int>::Zero(domain.n_nodes(), 1) ; // has all zeros
    boundary_matrix(10, 0) = 2;


    // non linear reaction h_(u)*u
    std::function<double(SVector<1>)> h_ = [&](SVector<1> ff) -> double {return 1 - ff[0];};
    // build the non-linearity object # N=1
    NonLinearReaction<1, LagrangianBasis<decltype(domain),1>::ReferenceBasis> non_linear_reaction(h_);

    // differential operator
    auto L = dt<FEM>() - laplacian<FEM>() - non_linear_op<FEM>(non_linear_reaction);

    PDE<decltype(domain), decltype(L), DMatrix<double>, FEM, fem_order<1>> pde_(domain, times, L, h_, boundary_matrix);
    
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
    for (int i = 0; i < nodes_.rows(); ++i) { initial_condition(i) = solution_expr(nodes_.row(i), times(0)); }

    // set dirichlet conditions
    pde_.set_dirichlet_bc(dirichlet_bc);

    // set initial condition
    pde_.set_initial_condition(initial_condition);

    // request quadrature nodes and evaluate forcing on them
    DMatrix<double> quadrature_nodes = pde_.force_quadrature_nodes();
    DMatrix<double> f(quadrature_nodes.rows(), M);
    for (int i = 0; i < quadrature_nodes.rows(); ++i) {
        for (int j = 0; j < M; ++j) { f(i, j) = forcing_expr(quadrature_nodes.row(i), times(j)); }
    }
    pde_.set_forcing(f);
    DMatrix<double> boundary_quadrature_nodes = pde_.boundary_quadrature_nodes();
    // DMatrix<double> f_neumann(boundary_quadrature_nodes.rows(), M);
    // for (int i=0; i< boundary_quadrature_nodes.rows(); ++i){
    //     for (int j = 0; j < M; ++j) { f_neumann(i, j) = neumann_expr(boundary_quadrature_nodes.row(i), times(j)); }
    // }
    // pde_.set_neumann_bc(f_neumann);
    DMatrix<double> f_robin(boundary_quadrature_nodes.rows(), M);
    for (int i=0; i< boundary_quadrature_nodes.rows(); ++i){
        for (int j = 0; j < M; ++j) { f_robin(i, j) = robin_expr(boundary_quadrature_nodes.row(i), times(j)); }
    }
    DVector<double> robin_constants(2,1);
    robin_constants << a, b;
    pde_.set_robin_bc(f_robin, robin_constants);

    // init solver and solve differential problem
    pde_.init();
    pde_.solve();

    // check computed error within theoretical expectations
    // std::cout << "L2 errors:" << std::endl;
    DMatrix<double> error_(nodes_.rows(), 1);
    for (int j = 0; j < M; ++j) {
        error_ = solution_ex.col(j) - pde_.solution().col(j);
        error_L2(j, 0) = (pde_.mass() * error_.cwiseProduct(error_)).sum();
        // std::cout << "t = " << j << " ErrorL2 = " << std::sqrt(error_L2(j,0)) << std::endl;
    }

    // std::ofstream file("CNSI_P1_space_time_RD_1D_h200_T5_dt10.txt");    //it will be exported in the current build directory
    // if (file.is_open()){
    //     for(int i = 0; i < pde_.solution().col(M-1).rows(); ++i)
    //         file << std::setprecision(15) << pde_.solution().col(M-1)(i) << '\n';
    //     file << std::sqrt(error_L2(M-1,0));
    //     file.close();
    // } else {
    //     std::cerr << "test unable to save solution" << std::endl;
    // }

    EXPECT_TRUE(error_L2.maxCoeff() < 1e-7);
}

// space time test with mixed boundary onditions
TEST(fem_pde_boundary_condition_test, space_time_dirichlet_neumann_2D) {

    // exact solution
    int M = 101;
    DMatrix<double> times(M, 1);
    double time_max = 1e-3;
    for (int j = 0; j < M; ++j) { times(j) = time_max / (M - 1) * j; }
    // diffusion coefficient
    SMatrix<2,2> K_{{3., 0.},{0., 3.}};

    int num_refinements = 1;
    DMatrix<double> error_L2 = DMatrix<double>::Zero(M, num_refinements);
    constexpr double pi = 3.14159265358979323846;

    auto solution_expr = [](SVector<2> x, double t) -> double {
        return (std::cos(pi*x[0])*std::cos(pi*x[1]) + 2)*std::exp(-t);
    };
    auto forcing_expr = [&](SVector<2> x, double t) -> double {
        return -4*std::exp(-t) + std::cos(pi*x[0])*std::cos(pi*x[1])*std::exp(-t)*(-2+2*K_(0,0)*pi*pi) + (4 + std::cos(pi*x[0])*std::cos(pi*x[0])*std::cos(pi*x[1])*std::cos(pi*x[1]) + 4*std::cos(pi*x[0])*std::cos(pi*x[1]))*std::exp(-2*t);
    };
    // auto neumann_expr = [](SVector<2> x, double t) -> double { 
    //     return - pi*std::sin(pi*x[0])*std::cos(pi*x[1])*std::exp(-t); };
    // Robin data
    double a = 17;
    double b = 32;
    auto robin_expr = [&](SVector<2> x, double t) -> double { 
        return a*(std::cos(pi*x[0])*std::cos(pi*x[1]) + 2)*std::exp(-t) - b*pi*std::sin(pi*x[0])*std::cos(pi*x[1])*std::exp(-t);
    };

    MeshLoader<Mesh2D> unit_square("unit_square_16");

    // define the Neumann and Dirichlet boundary with a DMatrix (=0 if Dirichlet, =1 if Neumann, =2 if Robin)
    // considering the unit square,
    // we have Dirichlet boundary when y=0 and y=1 (upper and lower sides)
    // we have Neumann boundary when x=0 and x=1 (left and right sides)
    DMatrix<short int> boundary_matrix = DMatrix<short int>::Zero(unit_square.mesh.n_nodes(), 1) ; // has all zeros
    for (size_t j=1; j<16; j++) {
        boundary_matrix(0 + 17*j, 0) = 2;
        boundary_matrix(16 + 17*j, 0) = 2;
    }

    // non linear reaction h_(u)*u
    std::function<double(SVector<1>)> h_ = [&](SVector<1> ff) -> double {return 1 - ff[0];};
    // build the non-linearity object # N=2
    NonLinearReaction<2, LagrangianBasis<decltype(unit_square.mesh),1>::ReferenceBasis> non_linear_reaction(h_);

    // differential operator
    auto L = dt<FEM>() - diffusion<FEM>(K_) - non_linear_op<FEM>(non_linear_reaction);

    PDE<decltype(unit_square.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> pde_(unit_square.mesh, times, L, h_, boundary_matrix);
    
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
    for (int i = 0; i < nodes_.rows(); ++i) { initial_condition(i) = solution_expr(nodes_.row(i), times(0)); }
    // set dirichlet conditions
    pde_.set_dirichlet_bc(dirichlet_bc);
    // set initial condition
    pde_.set_initial_condition(initial_condition);

    // request quadrature nodes and evaluate forcing on them
    DMatrix<double> quadrature_nodes = pde_.force_quadrature_nodes();
    DMatrix<double> f(quadrature_nodes.rows(), M);
    for (int i = 0; i < quadrature_nodes.rows(); ++i) {
        for (int j = 0; j < M; ++j) { f(i, j) = forcing_expr(quadrature_nodes.row(i), times(j)); }
    }
    pde_.set_forcing(f);
    DMatrix<double> boundary_quadrature_nodes = pde_.boundary_quadrature_nodes();
    // DMatrix<double> f_neumann(boundary_quadrature_nodes.rows(), M);
    // for (auto i=0; i< boundary_quadrature_nodes.rows(); ++i){
    //     for (int j = 0; j < M; ++j) f_neumann(i, j) = neumann_expr(boundary_quadrature_nodes.row(i), times(j));
    // }
    // pde_.set_neumann_bc(f_neumann);
    DMatrix<double> f_robin(boundary_quadrature_nodes.rows(), M);
    for (int i=0; i< boundary_quadrature_nodes.rows(); ++i){
        for (int j = 0; j < M; ++j) { f_robin(i, j) = robin_expr(boundary_quadrature_nodes.row(i), times(j)); }
    }
    DVector<double> robin_constants(2,1);
    robin_constants << a, b;
    pde_.set_robin_bc(f_robin, robin_constants, K_(0,0));

    // init solver and solve differential problem
    pde_.init();
    pde_.solve();

    // check computed error within theoretical expectations
    // std::cout << "L2 errors:" << std::endl;
    DMatrix<double> error_(nodes_.rows(), 1);
    for (int j = 0; j < M; ++j) {
        error_ = solution_ex.col(j) - pde_.solution().col(j);
        error_L2(j, 0) = (pde_.mass() * error_.cwiseProduct(error_)).sum();
        // std::cout << "t = " << j << " ErrorL2 = " << std::sqrt(error_L2(j,0)) << std::endl;
    }

    // std::ofstream file("CNSI_P1_space_time_RD_2D_h256_T5_dt10.txt");    //it will be exported in the current build directory
    // if (file.is_open()){
    //     for(int i = 0; i < pde_.solution().col(M-1).rows(); ++i)
    //         file << std::setprecision(15) << pde_.solution().col(M-1)(i) << '\n';
    //     file << std::sqrt(error_L2(M-1,0));
    //     file.close();
    // } else {
    //     std::cerr << "test unable to save solution" << std::endl;
    // }

    EXPECT_TRUE(error_L2.maxCoeff() < 1e-7);
}

TEST(fem_pde_boundary_condition_test, space_time_dirichlet_neumann_3D) {

    // exact solution
    int M = 101;
    DMatrix<double> times(M, 1);
    double time_max = 1e-3;
    for (int j = 0; j < M; ++j) { times(j) = time_max / (M - 1) * j; }

    int num_refinements = 1;
    DMatrix<double> error_L2 = DMatrix<double>::Zero(M, num_refinements);
    SVector<3> b_; b_ << 2., -1., 1.;  // advection coefficient

    auto solution_expr = [](SVector<3> x, double t) -> double {
        return (x[0] + x[1]*x[1] + x[2]*x[2])*t;
    };
    auto forcing_expr = [](SVector<3> x, double t) -> double {
        return x[0] + x[1]*x[1] + x[2]*x[2] - 4*t + (2 - 2*x[1] + 2*x[2])*t + (x[0] + x[1]*x[1] + x[2]*x[2])*t*(1 - (x[0] + x[1]*x[1] + x[2]*x[2])*t);
    };
    // auto neumann_expr = [](SVector<3> x, double t) -> double { 
    //     return 2*x[2]*t;
    // };
    // Robin data
    double a = 7;
    double b = 3;
    auto robin_expr = [&](SVector<3> x, double t) -> double { 
        return a*solution_expr(x,t) + b*2*x[2]*t;
    };

    MeshLoader<Mesh3D> domain("unit_cube_14");
    // define the Neumann and Dirichlet boundary with a DMatrix (=0 if Dirichlet, =1 if Neumann, =2 if Robin)
    // considering the unit cube,
    // we have Neumann boundary when z=0 and z=1 (upper and lower sides)
    DMatrix<short int> boundary_matrix = DMatrix<short int>::Zero(domain.mesh.n_nodes(), 1) ; // has all zeros
    for (size_t j=1; j<225; ++j) boundary_matrix(j, 0) = 2;
    for (size_t j=3152; j<3375; ++j) boundary_matrix(j, 0) = 2;
    // set the Dirichlet nodes at the edges
    for (size_t j=0; j<15; j++) {
        boundary_matrix(15*j, 0) = 0;      // lower face, points with x=0
        boundary_matrix(14 + j*15, 0) = 0; // lower face, points with x=1
        boundary_matrix(j, 0) = 0;         // lower face, points with y=0
        boundary_matrix(210 + j, 0) = 0;   // lower face, points with y=1

        boundary_matrix(3150 + 15*j, 0) = 0; // upper face, points with x=0
        boundary_matrix(3164 + 15*j, 0) = 0; // upper face, points with x=1
        boundary_matrix(3150 + j, 0) = 0;    // upper face, points with y=0
        boundary_matrix(3360 + j, 0) = 0;    // upper face, points with y=1
    }

    // non linear reaction h_(u)*u
    std::function<double(SVector<1>)> h_ = [&](SVector<1> ff) -> double {return 1 - ff[0];};
    // build the non-linearity object # N=3
    NonLinearReaction<3, LagrangianBasis<decltype(domain.mesh),1>::ReferenceBasis> non_linear_reaction(h_);

    // differential operator
    auto L = dt<FEM>() -laplacian<FEM>() + advection<FEM>(b_) + non_linear_op<FEM>(non_linear_reaction);

    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> pde_(domain.mesh, times, L, h_, boundary_matrix);
    
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
    for (int i = 0; i < nodes_.rows(); ++i) { initial_condition(i) = solution_expr(nodes_.row(i), times(0)); }
    // set dirichlet conditions
    pde_.set_dirichlet_bc(dirichlet_bc);
    // set initial condition
    pde_.set_initial_condition(initial_condition);

    // request quadrature nodes and evaluate forcing on them
    DMatrix<double> quadrature_nodes = pde_.force_quadrature_nodes();
    DMatrix<double> f(quadrature_nodes.rows(), M);
    for (int i = 0; i < quadrature_nodes.rows(); ++i) {
        for (int j = 0; j < M; ++j) { f(i, j) = forcing_expr(quadrature_nodes.row(i), times(j)); }
    }
    pde_.set_forcing(f);
    DMatrix<double> boundary_quadrature_nodes = pde_.boundary_quadrature_nodes();
    // DMatrix<double> f_neumann(boundary_quadrature_nodes.rows(), M);
    // for (auto i=0; i< boundary_quadrature_nodes.rows(); ++i){
    //     for (int j = 0; j < M; ++j) f_neumann(i, j) = neumann_expr(boundary_quadrature_nodes.row(i), times(j));
    // }
    // pde_.set_neumann_bc(f_neumann);
    DMatrix<double> f_robin(boundary_quadrature_nodes.rows(), M);
    for (int i=0; i< boundary_quadrature_nodes.rows(); ++i){
        for (int j = 0; j < M; ++j) { f_robin(i, j) = robin_expr(boundary_quadrature_nodes.row(i), times(j)); }
    }
    DVector<double> robin_constants(2,1);
    robin_constants << a, b;
    pde_.set_robin_bc(f_robin, robin_constants);

    // init solver and solve differential problem
    pde_.init();
    pde_.solve();

    // check computed error within theoretical expectations
    // std::cout << "L2 errors:" << std::endl;
    DMatrix<double> error_(nodes_.rows(), 1);
    for (int j = 0; j < M; ++j) {
        error_ = solution_ex.col(j) - pde_.solution().col(j);
        error_L2(j, 0) = (pde_.mass() * error_.cwiseProduct(error_)).sum();
        // std::cout << "t = " << j << " ErrorL2 = " << std::sqrt(error_L2(j,0)) << std::endl;
    }

    // std::ofstream file("ESI_P1_space_time_RD_3D_h20_T20_dt2.txt");    //it will be exported in the current build directory
    // if (file.is_open()){
    //     for(int i = 0; i < pde_.solution().col(M-1).rows(); ++i)
    //         file << std::setprecision(15) << pde_.solution().col(M-1)(i) << '\n';
    //     file << std::sqrt(error_L2(M-1,0));
    //     file.close();
    // } else {
    //     std::cerr << "test unable to save solution" << std::endl;
    // }

    EXPECT_TRUE(error_L2.maxCoeff() < 1e-7);
}

TEST(fem_pde_boundary_condition_test, dirichlet_robin_1D) {
    // exact solution
    auto solution_expr = [](SVector<1> x) -> double { return (std::cos(x[0]) + std::sin(x[0])) * x[0]; };
    auto forcing_expr = [](SVector<1> x) -> double { return 5*(2*std::sin(x[0]) - 2*std::cos(x[0]) + (std::cos(x[0]) + std::sin(x[0])) * x[0]); };

    // Robin data
    double a = 32;
    double b = 4;
    auto robin_expr = [&](SVector<1> x) -> double { return a*(std::cos(1) + std::sin(1)) + b*2*std::cos(1); };

    Mesh1D domain(0.0, 1.0, 10);
    // define the Neumann and Dirichlet boundary with a DMatrix (=0 if Dirichlet, =1 if Neumann, =2 if Robin)
    // considering the unit square,
    // we have Neumann boundary
    DMatrix<short int> boundary_matrix = DMatrix<short int>::Zero(domain.n_nodes(), 1) ; // has all zeros
    boundary_matrix(10, 0) = 2;

    SMatrix<1,1> D{{5}};

    auto L = -diffusion<FEM>(D);
    // instantiate a type-erased wrapper for this pde
    PDE<decltype(domain), decltype(L), DMatrix<double>, FEM, fem_order<1>> pde_(domain, L, boundary_matrix);
    
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
    DMatrix<double> quadrature_nodes = pde_.force_quadrature_nodes();
    DMatrix<double> f(quadrature_nodes.rows(), 1);
    for (int i = 0; i < quadrature_nodes.rows(); ++i) {
        f(i) = forcing_expr(quadrature_nodes.row(i));
    }
    pde_.set_forcing(f);
    DMatrix<double> boundary_quadrature_nodes = pde_.boundary_quadrature_nodes();
    DMatrix<double> f_robin(boundary_quadrature_nodes.rows(), 1);
    for (auto i=0; i< boundary_quadrature_nodes.rows(); ++i){
        f_robin(i) = robin_expr(boundary_quadrature_nodes.row(i));
    }
    DVector<double> robin_constants(2,1);
    robin_constants << a, b;
    pde_.set_robin_bc(f_robin, robin_constants, D(0,0));
    // init solver and solve differential problem
    pde_.init();
    pde_.solve();
    // check computed error within theoretical expectations
    DMatrix<double> error_ = solution_ex - pde_.solution();
    double error_L2 = (pde_.mass() * error_.cwiseProduct(error_)).sum();
    // std::cout << std::sqrt(error_L2) << std::endl;

    // std::ofstream file("RD_1D_P1.txt");    //it will be exported in the current build directory
    // if (file.is_open()){
    //     for(int i = 0; i < pde_.solution().rows(); ++i)
    //         file << pde_.solution()(i) << '\n';
    //     file.close();
    // } else {
    //     std::cerr << "test unable to save solution" << std::endl;
    // }

    EXPECT_TRUE(error_L2 < DOUBLE_TOLERANCE);
}

/* #include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iomanip>
TEST(fem_pde_boundary_condition_test, deSolve_1D) {
    int M = 101; //351
    DMatrix<double> times(M, 1);
    double time_max = 100; //350;
    for (int j = 0; j < M; ++j) { times(j) = time_max / (M - 1) * j; }

    // parameter for the problem
    SVector<1> b;
    b << 0.092;
    SMatrix<1,1> D{{0.039}};
    double r = 0.032;
    double ks = 0.16;
    double F0 = 0.006;
    double SO_in = 0.25;

    // forces
    auto neumann_expr = [](SVector<1> x, double t) -> double { return 0; };
    auto forcing_expr = [](SVector<1> x, double t) -> double { return 0; };
    auto robin_expr = [&](SVector<1> x, double t) -> double { 
        return F0;
        // if (t <= 100) return b[0]*0.25;
        // else if (t <= 170) return b[0]*0.5;
        // else if(t <= 270) return b[0]*1.0;
        // else return b[0]*2.0;
    };

    Mesh1D domain(0.0, 2.0, 100);

    // define the Neumann and Dirichlet boundary with a DMatrix (=0 if Dirichlet, =1 if Neumann, =2 if Robin)
    DMatrix<short int> boundary_matrix = DMatrix<short int>::Ones(domain.n_nodes(), 1);
    boundary_matrix(0, 0) = 2;

    // non linear reaction h_(u)*u
    std::function<double(SVector<1>)> h_ = [&](SVector<1> ff) -> double {return r/(ff[0] + ks);};
    // build the non-linearity object
    NonLinearReaction<1, LagrangianBasis<decltype(domain),1>::ReferenceBasis> non_linear_reaction(h_);

    auto L = dt<FEM>() - diffusion<FEM>(D) + advection<FEM>(b) + non_linear_op<FEM>(non_linear_reaction);
    // instantiate a type-erased wrapper for this pde
    PDE<decltype(domain), decltype(L), DMatrix<double>, FEM, fem_order<1>> pde_(domain, times, L, h_, boundary_matrix);
    
    // compute boundary condition and initial condition
    DMatrix<double> nodes_ = pde_.dof_coords();
    DMatrix<double> dirichlet_bc(nodes_.rows(), M);
    DMatrix<double> initial_condition(nodes_.rows(), 1);

    // set dirichlet conditions
    // for (int i = 0; i < nodes_.rows(); ++i) {
    //     dirichlet_bc.row(i) << 1.00000000, 0.59812001, 0.46472786, 0.37789535, 0.31454976, 0.26578686, 0.22706671, 0.19571037, 0.16998502, 0.14869330, 0.13096981, 0.11616213, 0.10376511, 0.09337750, 0.08467371, 0.07738627, 0.07129277, 0.06620638, 0.06196937, 0.05844820, 0.05552953, 0.05311697, 0.05112860, 0.04949480, 0.04815660, 0.04706409, 0.04617511, 0.04545416, 0.04487143, 0.04440190, 0.04402485, 0.04372294, 0.04348191, 0.04329005, 0.04313768, 0.04301700, 0.04292160, 0.04284634, 0.04278709, 0.04274048, 0.04270390, 0.04267520, 0.04265275, 0.04263518, 0.04262138, 0.04261059, 0.04260224, 0.04259577, 0.04259073, 0.04258678, 0.04258369, 0.04258127, 0.04257939, 0.04257793, 0.04257678, 0.04257588, 0.04257519, 0.04257465, 0.04257423, 0.04257390, 0.04257365, 0.04257345, 0.04257329, 0.04257318, 0.04257308, 0.04257301, 0.04257295, 0.04257291, 0.04257288, 0.04257285, 0.04257283, 0.04257281, 0.04257280, 0.04257279, 0.04257278, 0.04257278, 0.04257277, 0.04257277, 0.04257277, 0.04257277, 0.04257276, 0.04257276, 0.04257276, 0.04257276, 0.04257276, 0.04257276, 0.04257276, 0.04257276, 0.04257276, 0.04257276, 0.04257276, 0.04257276, 0.04257276, 0.04257276, 0.04257276, 0.04257276, 0.04257276, 0.04257276, 0.04257276, 0.04257276, 0.04257276;
    // }
    // pde_.set_dirichlet_bc(dirichlet_bc);

    // set initial condition
    for (int i = 0; i < nodes_.rows(); ++i) { initial_condition(i) = 1; }
    pde_.set_initial_condition(initial_condition);

    // request quadrature nodes and evaluate forcing on them
    DMatrix<double> quadrature_nodes = pde_.force_quadrature_nodes();
    DMatrix<double> f(quadrature_nodes.rows(), M);
    for (int i = 0; i < quadrature_nodes.rows(); ++i) {
        for (int j = 0; j < M; ++j) { f(i, j) = forcing_expr(quadrature_nodes.row(i), times(j)); }
    }
    pde_.set_forcing(f);

    // request boundary quadrature nodes and evaluate neumann forcing and robin forcing on them
    DMatrix<double> boundary_quadrature_nodes = pde_.boundary_quadrature_nodes();
    DMatrix<double> f_neumann(boundary_quadrature_nodes.rows(), M);
    for (auto i=0; i< boundary_quadrature_nodes.rows(); ++i){
        for (int j = 0; j < M; ++j) { f_neumann(i, j) = neumann_expr(boundary_quadrature_nodes.row(i), times(j)); }
    }
    pde_.set_neumann_bc(f_neumann, D(0,0));

    DMatrix<double> f_robin(boundary_quadrature_nodes.rows(), M);
    for (auto i=0; i< boundary_quadrature_nodes.rows(); ++i){
        for (int j = 0; j < M; ++j) { f_robin(i, j) = robin_expr(boundary_quadrature_nodes.row(i), times(j)); }
    }
    DVector<double> robin_constants(2,1);
    robin_constants << b[0], D(0,0);
    pde_.set_robin_bc(f_robin, robin_constants, D(0,0));

    // init solver and solve differential problem
    pde_.init();
    pde_.solve();

    // std::ofstream file1("CNSI_solution_deSolve_prova_preinterpolation.txt");    //it will be exported in the current build directory
    // if (file1.is_open()){
    //     for (int j = 0; j < M; ++j) {
    //         // file1 << "M = " << j << "\n";
    //         for(int i = 0; i < pde_.solution().col(j).rows(); ++i) {
    //             file1 << std::setprecision(15) << pde_.solution().col(j)(i) << '\n';
    //         }
    //         //file1 << "\n \n";
    //     }
    //     file1.close();
    // } else {
    //     std::cerr << "test unable to save solution" << std::endl;
    // }

    //interpolate the solution
    // std::ofstream file2("CNSI_solution_deSolve_prova_postinterpolation.txt");    //it will be exported in the current build directory
    // if (file2.is_open()){
    //     for (int j = 0; j < M; ++j) {
    //         // file2 << "M = " << j << "\n";
    //         for(int i = 0; i < pde_.solution().col(j).rows()-1; ++i) {
    //             auto interp_result = pde_.solution().col(j)(i) + 0.5*(pde_.solution().col(j)(i+1) - pde_.solution().col(j)(i));
    //             file2 << std::setprecision(15) << interp_result << '\n';
    //         }
    //         //file2 << "\n \n";
    //     }
    //     file2.close();
    // } else {
    //     std::cerr << "test unable to save solution" << std::endl;
    // }


    // compute L2 difference at the final time wrt. FreeFem++ solution
    DMatrix<double> freefem_solution = DMatrix<double>::Zero(nodes_.rows(), M);
    std::ifstream file("../../../R script/1D deSolve/freefem/desolve_1D_ESI_PREinterpolation_freefem.txt");
    double number;
    int j = 0;
    int i = 0;
    while (file >> number) {
        freefem_solution(i,j) = number;
        i++;
        if(i == nodes_.rows()) {i=0; j++;}
    }
    file.close();
    std::cout << freefem_solution.col(0) << std::endl;
    std::cout << '\n'<< '\n'<< '\n' << std::endl;
    std::cout << freefem_solution.col(1) << std::endl;
    std::cout << '\n'<< '\n'<< '\n' << std::endl;
    std::cout << freefem_solution.col(M-1) << std::endl;

    DMatrix<double> error_(nodes_.rows(), 1);
    DMatrix<double> error_L2 = DMatrix<double>::Zero(M, 1);
    for (int j = 0; j < M; ++j) {
        error_ = freefem_solution.col(j) - pde_.solution().col(j);
        error_L2(j, 0) = (pde_.mass() * error_.cwiseProduct(error_)).sum();
        std::cout << "t = " << j << " ErrorL2 = " << std::sqrt(error_L2(j,0)) << std::endl;
    }
    std::ofstream file2("ESI_femR_freefem_L2errors.txt");    //it will be exported in the current build directory
    if (file2.is_open()){
        for (int j = 0; j < M; ++j) {
            file2 << std::setprecision(15) << std::sqrt(error_L2(j,0)) << '\n';
        }
        file2.close();
    } else {
        std::cerr << "test unable to save solution" << std::endl;
    }

    EXPECT_TRUE(1);
}

TEST(fem_pde_boundary_condition_test, deSolve_2D) {
    // parameters for the problem
    double phi = 0.8;        // porosity
    double O2_top = 300;     // boundary oxygen concentration
    double mu = 1;           // diffusion coefficient
    double r = -0.2;         // consumption rate

    // forces
    auto dirichlet_expr = [&] (SVector<2> x) -> double { return O2_top; };
    auto neumann_expr = [](SVector<2> x) -> double { return 0; };
    auto forcing_expr = [](SVector<2> x) -> double { return 0; };

    MeshLoader<Mesh2D> domain("rectangle_grid100x30_length10x2");

    // define the Neumann and Dirichlet boundary with a DMatrix (=0 if Dirichlet, =1 if Neumann, =2 if Robin)
    DMatrix<short int> boundary_matrix = DMatrix<short int>::Ones(domain.mesh.n_nodes(), 1);
    // side y=2 Dirichlet, all the other sides Neumann
    for (int i=3030; i<3131; ++i) boundary_matrix(i, 0) = 0;

    auto L = - mu*laplacian<FEM>() - reaction<FEM>(r);
    // instantiate a type-erased wrapper for this pde
    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> pde_(domain.mesh, L, boundary_matrix);
    
    // compute boundary condition and initial condition
    DMatrix<double> nodes_ = pde_.dof_coords();
    DMatrix<double> dirichlet_bc(nodes_.rows(), 1);

    // set dirichlet conditions
    for (int i = 0; i < nodes_.rows(); ++i) {
        dirichlet_bc(i) = dirichlet_expr(nodes_.row(i));
    }
    // set dirichlet conditions
    pde_.set_dirichlet_bc(dirichlet_bc);

    // request quadrature nodes and evaluate forcing on them
    DMatrix<double> quadrature_nodes = pde_.force_quadrature_nodes();
    DMatrix<double> f(quadrature_nodes.rows(), 1);
    for (int i = 0; i < quadrature_nodes.rows(); ++i) {
        f(i) = forcing_expr(quadrature_nodes.row(i));
    }
    pde_.set_forcing(f);

    // request boundary quadrature nodes and evaluate neumann forcing and robin forcing on them
    DMatrix<double> boundary_quadrature_nodes = pde_.boundary_quadrature_nodes();
    DMatrix<double> f_neumann(boundary_quadrature_nodes.rows(), 1);
    for (auto i=0; i< boundary_quadrature_nodes.rows(); ++i){
        f_neumann(i) = neumann_expr(boundary_quadrature_nodes.row(i));
    }
    pde_.set_neumann_bc(f_neumann, mu);

    // init solver and solve differential problem
    pde_.init();
    pde_.solve();

    std::ofstream file1("solution_deSolve2D_prova_preinterpolation.txt");    //it will be exported in the current build directory
    if (file1.is_open()){
        for(int i = 0; i < pde_.solution().rows(); ++i) {
            file1 << std::setprecision(15) << pde_.solution()(i) << '\n';
        }
        file1.close();
    } else {
        std::cerr << "test unable to save solution" << std::endl;
    }

    //interpolate the solution
    // std::ofstream file2("ESI_solution_deSolve2D_prova_postinterpolation.txt");    //it will be exported in the current build directory
    // if (file2.is_open()){
    //     for (int j = 0; j < M; ++j) {
    //         // file2 << "M = " << j << "\n";
    //         for(int i = 0; i < pde_.solution().col(j).rows()-1; ++i) {
    //             auto interp_result = pde_.solution().col(j)(i) + 0.5*(pde_.solution().col(j)(i+1) - pde_.solution().col(j)(i));
    //             file2 << std::setprecision(15) << interp_result << '\n';
    //         }
    //         //file2 << "\n \n";
    //     }
    //     file2.close();
    // } else {
    //     std::cerr << "test unable to save solution" << std::endl;
    // }

    // compute L2 norm of the difference wrt. FreeFem++ solution
    // DMatrix<double> freefem_solution = DMatrix<double>::Zero(nodes_.rows(), 1);
    // std::ifstream file("../../../R script/2D deSolve/freefem_deSolve2D_preinterpolation.txt");
    // double number;
    // int j = 0;
    // while (file >> number) {
    //     freefem_solution(j) = number;
    //     j++;
    // }
    // file.close();

    // DMatrix<double> error_(nodes_.rows(), 1);
    // double error_L2 = 0;
    // error_ = freefem_solution - pde_.solution();
    // error_L2 = (pde_.mass() * error_.cwiseProduct(error_)).sum();
    // std::cout << " ErrorL2 = " << std::sqrt(error_L2) << std::endl;

    EXPECT_TRUE(1);
} */