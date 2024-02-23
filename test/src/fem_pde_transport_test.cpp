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
using fdapde::core::Integrator;
using fdapde::core::LagrangianBasis;
using fdapde::core::FEM;
using fdapde::core::MatrixConst;
using fdapde::core::MatrixPtr;
using fdapde::core::ScalarPtr;
using fdapde::core::VectorPtr;

using fdapde::core::PDE;
using fdapde::core::ScalarField;
using fdapde::core::advection;
using fdapde::core::reaction;
using fdapde::core::diffusion;
using fdapde::core::laplacian;
using fdapde::core::dt;
using fdapde::core::fem_order;
using fdapde::core::make_pde;

#include "utils/mesh_loader.h"
using fdapde::testing::MeshLoader;
#include "utils/utils.h"
using fdapde::testing::almost_equal;
using fdapde::testing::DOUBLE_TOLERANCE;

#include <iomanip>
#include <string>

// tests for Advection Dominated Elliptic Partial Differential Equations
TEST(transport_test, transportP1) {
    // define exact solution
    auto solutionExpr = [](SVector<2> x) -> double {
        return 3*sin(x[0]) + 2*x[1];
    };

    // start with a transport-dominated differential operator

    // sembra che questa dichiarazione funzioni, però poi ci sono dei problemi con la forma debole dell'operatore,
    // perché nel metodo integrate facciamo .dot(b), ma adesso non si può più fare, non esistono infatti dei test
    // nella libreria per questo caso
    /*
    auto b = [](const SVector<2>& x) -> SVector<2> {
        return SVector<2>(x[0], x[1] + x[0]*2);
    };
    */

    SVector<2> b;  b << 1., 1.;
    double mu = 1e-9;

    // non-zero forcing term
    auto forcingExpr = [mu, b](SVector<2> x) -> double {
        return 2*b[1] + 3*b[0]*cos(x[0]) + 3*mu*sin(x[0]);
        // return 2*b(x)[1] + 3*b(x)[0]*cos(x[0]) + 3*mu*sin(x[0]);
    };
    ScalarField<2> forcing(forcingExpr);   // wrap lambda expression in ScalarField object

    auto L = - mu * laplacian<FEM>() + advection<FEM>(b); //+ reaction<FEM>(c_);
    // load sample mesh for order 1 basis
    MeshLoader<Mesh2D> unit_square("unit_square_32");

    constexpr std::size_t femOrder = 1;

    // PDE<decltype(unit_square.mesh), decltype(L), ScalarField<2>, FEM, fem_order<femOrder>> pde_(unit_square.mesh, L, forcing);
    PDE<decltype(unit_square.mesh), decltype(L), ScalarField<2>, FEM, fem_order<1>> pde_(unit_square.mesh, L, forcing);

    // compute boundary condition and exact solution
    DMatrix<double> nodes_ = pde_.dof_coords();
    DMatrix<double> dirichletBC(nodes_.rows(), 1);
    DMatrix<double> solution_ex(nodes_.rows(), 1);

    // set exact sol & dirichlet conditions
    for (int i = 0; i < nodes_.rows(); ++i) {
        solution_ex(i) = solutionExpr(nodes_.row(i));
        dirichletBC(i) = solutionExpr(nodes_.row(i));
    }
    pde_.set_dirichlet_bc(dirichletBC);

    // init solver and solve differential problem
    pde_.init();
    pde_.solve();

    // check computed error
    DMatrix<double> error_ = solution_ex - pde_.solution();
    double error_L2 = (pde_.mass() * error_.cwiseProduct(error_)).sum();
    EXPECT_TRUE(error_L2 < 1e-6);

    std::cout << "error_L2 = " << std::setprecision(17) << error_L2 << std::endl;

    //save solution
    //std::string titlename = "transport_test_solution_mu_" + std::to_string(mu) + "_b_" + std::to_string(b_[0]) + "_" + std::to_string(b_[1]) + ".txt";
    std::string titlename = "transport_test_solution.txt";
    std::ofstream file(titlename);
    if (file.is_open()){
        for(int i = 0; i < pde_.solution().rows(); ++i)
            file << pde_.solution()(i) << '\n';
            file.close();
    } else {
        std::cerr << "transport test unable to save solution" << std::endl;
    }
}

/*
TEST(transport_test, transportP2_32) {
    // define exact solution
    auto solutionExpr = [](SVector<2> x) -> double {
        // return 3*x[0]*x[0] + 2*x[1]*x[1];
        return 3*sin(x[0]) + 2*x[1];
    };

    // start with a transport-dominated differential operator
    SVector<2> b;  b << 1., 1.;
    double mu = 1e-9;

    // evaluate the peclet number
    double Pe = b.norm()*(double(1)/double(32))/(2*mu); //TODO va fatto meglio
    std::cout << "Peclet number = " << Pe << std::endl;

    // non-zero forcing term
    auto forcingExpr = [mu, b](SVector<2> x) -> double {
        // return 6*x[0]*b_[0] - 10*mu + 4*x[1]*b_[1];
        return 2*b[1] + 3*b[0]*cos(x[0]) + 3*mu*sin(x[0]);
    };
    ScalarField<2> forcing(forcingExpr);   // wrap lambda expression in ScalarField object

    auto L = - mu * laplacian<FEM>() + advection<FEM>(b); //+ reaction<FEM>(c_);
    // load sample mesh for order 1 basis
    MeshLoader<Mesh2D> unit_square("unit_square_32");

    std::cout << "h = " << unit_square.mesh.element(0).measure() << std::endl;

    constexpr std::size_t femOrder = 2;

    PDE<decltype(unit_square.mesh), decltype(L), ScalarField<2>, FEM, fem_order<femOrder>> pde_(unit_square.mesh, L, forcing);

    // compute boundary condition and exact solution
    DMatrix<double> nodes_ = pde_.dof_coords();
    DMatrix<double> dirichletBC(nodes_.rows(), 1);
    DMatrix<double> solution_ex(nodes_.rows(), 1);

    // set exact sol & dirichlet conditions
    for (int i = 0; i < nodes_.rows(); ++i) {
    solution_ex(i) = solutionExpr(nodes_.row(i));
    dirichletBC(i) = solutionExpr(nodes_.row(i));
    }
    pde_.set_dirichlet_bc(dirichletBC);

    // init solver and solve differential problem
    pde_.init();
    pde_.solve();

    // check computed error
    DMatrix<double> error_ = solution_ex - pde_.solution();
    double error_L2 = (pde_.R0() * error_.cwiseProduct(error_)).sum();
    EXPECT_TRUE(error_L2 < 1e-7);

    std::cout << "error_L2 = " << std::setprecision(17) << error_L2 << std::endl;

    //save solution
    //std::string titlename = "transport_test_solution_mu_" + std::to_string(mu) + "_b_" + std::to_string(b_[0]) + "_" + std::to_string(b_[1]) + ".txt";
    std::string titlename = "transport_test_solution_P2.txt";
    std::ofstream file(titlename);
    if (file.is_open()){
    for(int i = 0; i < pde_.solution().rows(); ++i)
    file << pde_.solution()(i) << '\n';
    file.close();
    } else {
    std::cerr << "transport test unable to save solution" << std::endl;
    }
}
*/