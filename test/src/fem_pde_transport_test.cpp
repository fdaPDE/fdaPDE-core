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
using fdapde::core::PDEparameters;  // ADDED
using fdapde::core::DiscretizedMatrixField; // ADDED
using fdapde::core::DiscretizedVectorField; // ADDED

#include "utils/mesh_loader.h"
using fdapde::testing::MeshLoader;
#include "utils/utils.h"
using fdapde::testing::almost_equal;
using fdapde::testing::DOUBLE_TOLERANCE;
using fdapde::testing::read_csv;

#include <iomanip>
#include <string>

// tests for Advection Dominated Elliptic Partial Differential Equations
TEST(transport_test, TransportConsantCoefficients) {
    // define exact solution
    auto solutionExpr = [](SVector<2> x) -> double {
        return 3*sin(x[0]) + 2*x[1];
    };

    // start with a transport-dominated differential operator
    SVector<2> b;  b << 1., 1.;
    double mu = 1e-9;

    // save parameters in the PDEparameters singleton, these will be retrieved by the solver
    PDEparameters<decltype(mu), decltype(b)> &PDEparams = PDEparameters<decltype(mu), decltype(b)>::getInstance(mu, b);

    // non-zero forcing term
    auto forcingExpr = [&mu, &b](SVector<2> x) -> double {
        return 2*b[1] + 3*b[0]*cos(x[0]) + 3*mu*sin(x[0]);
        // return 2*b(x)[1] + 3*b(x)[0]*cos(x[0]) + 3*mu*sin(x[0]);
    };
    ScalarField<2> forcing(forcingExpr);   // wrap lambda expression in ScalarField object

    auto L = - mu * laplacian<FEM>() + advection<FEM>(b); //+ reaction<FEM>(c_);
    // load sample mesh for order 1 basis
    MeshLoader<Mesh2D> unit_square("unit_square_32");
    // MeshLoader<Mesh2D> unit_square("quasi_circle");

    constexpr std::size_t femOrder = 1;

    PDE< decltype(unit_square.mesh), decltype(L), ScalarField<2>, FEM, fem_order<femOrder>, decltype(mu),
            decltype(b)> pde_( unit_square.mesh, L, forcing);

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

    // std::cout << "error_L2 = " << std::setprecision(17) << error_L2 << std::endl;
}

TEST(transport_test, TransportNonConstantCoefficients) {
    // define exact solution
    auto solutionExpr = [](SVector<2> x) -> double {
        return 3*sin(x[0]) + 2*x[1];
    };

    // define domain
    // MeshLoader<Mesh2D> domain("quasi_circle");
    MeshLoader<Mesh2D> domain("unit_square_32");

    // std::cout << "domain.mesh.n_elements() = " << domain.mesh.n_elements() << std::endl;

    // define vector field containing transport data
    VectorField<2> b_callable;
    b_callable[0] = [](SVector<2> x) -> double { return std::pow(x[0], 2) + 1; };   // x^2 + 1
    b_callable[1] = [](SVector<2> x) -> double { return 2 * x[0] + x[1]; };         // 2*x + y

    size_t size_transport = domain.mesh.n_elements() * domain.mesh.element(0).coords().size();
    // std::cout << "size of discretized transport = " << size_transport << std::endl;
    DMatrix<double, Eigen::RowMajor> b_data(size_transport, 2);
    int i = 0;
    for (const auto& e : domain.mesh) {
        for (int j = 0; j < e.coords().size(); ++j) {
            SVector<2> x;
            x << e.coords()[j][0], e.coords()[j][1];
            b_data(i, 0) = b_callable[0](x);
            b_data(i, 1) = b_callable[1](x);
            // std::cout << "element = " << e.ID() << ", i = " << i << ",\t x = [" << x[0] << ", " << x[1] << "],\t b = [" << b_data(i, 0) << ", " << b_data(i, 1) << "]" << std::endl;
            //if ( ((i + 1) % 3) == 0) std::cout << std::endl;
            i++;
        }
    }
    // wrap data into a field
    DiscretizedVectorField<2,2> b_discretized(b_data);

    // coefficients
    double mu = 1; // 1e-9;

    // non-zero forcing term
    auto forcingExpr = [&mu, &b_callable](SVector<2> x) -> double {
        // return 2*b[1] + 3*b[0]*cos(x[0]) + 3*mu*sin(x[0]);
        return 2*b_callable[1](x) + 3*b_callable[0](x)*cos(x[0]) + 3*mu*sin(x[0]);
    };
    ScalarField<2> forcing(forcingExpr);   // wrap lambda expression in ScalarField object

    // save parameters in the PDEparameters singleton, these will be retrieved by the solver
    PDEparameters<decltype(mu), decltype(b_discretized)> &PDEparams = PDEparameters<decltype(mu), decltype(b_discretized)>::getInstance(mu, b_discretized);

    // define differential operator
    auto L = -mu*laplacian<FEM>() + advection<FEM>(b_discretized);

    constexpr std::size_t femOrder = 1;

    PDE< decltype(domain.mesh), decltype(L), ScalarField<2>, FEM, fem_order<femOrder>, decltype(mu),
            decltype(b_discretized)> pde_( domain.mesh, L, forcing );

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

    // std::cout << "error_L2 = " << std::setprecision(17) << error_L2 << std::endl;

    EXPECT_TRUE(1);
}