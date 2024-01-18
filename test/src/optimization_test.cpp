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
#include <fdaPDE/optimization.h>
using fdapde::core::BFGS;
using fdapde::core::GradientDescent;
using fdapde::core::Grid;
using fdapde::core::Newton;
using fdapde::core::BacktrackingLineSearch;
using fdapde::core::WolfeLineSearch;
using fdapde::core::Broyden;
using fdapde::core::ScalarField;
using fdapde::core::VectorField;

#include "utils/utils.h"
using fdapde::testing::almost_equal;
#include "utils/constants.h"
using fdapde::testing::DOUBLE_TOLERANCE;

TEST(optimization_test, grid_search) {
    // define objective function: x^2 + y^2
    ScalarField<2> f;
    f = [](SVector<2> x) -> double { return x[0] * x[0] + x[1] * x[1]; };
    // define grid of points
    std::vector<SVector<2>> grid;
    for (double x = -1; x < 1; x += 0.2) {
        for (double y = -1; y < 1; y += 0.2) { grid.push_back(SVector<2>(x, y)); }
    }
    // define optimizer
    Grid<2> opt;
    opt.optimize(f, grid);
    EXPECT_TRUE(almost_equal(opt.optimum()[0], 0.0) && almost_equal(opt.optimum()[1], 0.0));
}

TEST(optimization_test, gradient_descent_backtracking_line_search) {
    // define objective function: x*e^{-x^2 - y^2} + (x^2 + y^2)/20
    ScalarField<2> f;
    f = [](SVector<2> x) -> double {
        return x[0] * std::exp(-x[0] * x[0] - x[1] * x[1]) + (x[0] * x[0] + x[1] * x[1]) / 20;
    };
    f.set_step(1e-4);

    GradientDescent<2> opt(1000, 1e-6, 0.01);
    SVector<2> pt(-1, -1);
    opt.optimize(f, pt, BacktrackingLineSearch());

    // expected solution
    SVector<2> expected(-0.6690718221499544, 0);
    double L2_error = (opt.optimum() - expected).norm();
    EXPECT_TRUE(L2_error < 1e-6);
}

TEST(optimization_test, newton_backtracking_line_search) {
    // define objective function: x*e^{-x^2 - y^2} + (x^2 + y^2)/20
    ScalarField<2> f;
    f = [](SVector<2> x) -> double {
        return x[0] * std::exp(-x[0] * x[0] - x[1] * x[1]) + (x[0] * x[0] + x[1] * x[1]) / 20;
    };
    f.set_step(1e-4);

    Newton<2> opt(1000, 1e-6, 0.01);
    SVector<2> pt(-0.5, -0.5);
    opt.optimize(f, pt, BacktrackingLineSearch());

    // expected solution
    SVector<2> expected(-0.6690718221499544, 0);
    double L2_error = (opt.optimum() - expected).norm();
    EXPECT_TRUE(L2_error < 1e-6);
}

// test integration of Laplacian weak form for a LagrangianBasis of order 2
TEST(optimization_test, bfgs_wolfe_line_search) {
    // define objective function: x*e^{-x^2 - y^2} + (x^2 + y^2)/20
    ScalarField<2> f;
    f = [](SVector<2> x) -> double {
        return x[0] * std::exp(-x[0] * x[0] - x[1] * x[1]) + (x[0] * x[0] + x[1] * x[1]) / 20;
    };
    f.set_step(1e-4);

    // define optimizer
    BFGS<2> opt(1000, 1e-6, 0.01);
    // perform optimization
    SVector<2> pt(-1, -1);
    opt.optimize(f, pt, WolfeLineSearch());

    // expected solution
    SVector<2> expected(-0.6690718221499544, 0);
    double L2_error = (opt.optimum() - expected).norm();
    EXPECT_TRUE(L2_error < 1e-6);
}

TEST(fem_pde_test, Broyden_2D){
    // define the vector field
    VectorField<Dynamic> F(2,2);
    F[0] = [](DVector<double> x) -> double { return atan(x(0)*x(1)); };
    F[1] = [](DVector<double> x) -> double { return x(0)-5*x(1); };
    // F[0] = [](DVector<double> x) -> double { return x(0)*x(0); };
    // F[1] = [](DVector<double> x) -> double { return x(1)*x(1); };
    // Solution: x = 0., 0.

    // initial point
    DVector<double> x0(2);
    x0 << 7., 7.;

    Broyden<Dynamic> br(80, 1e-12);
    auto solution = br.solve(F, x0);
    // std::cout << "solution: \n" << solution << std::endl;

    // exact solution
    DVector<double> exact_sol(2);
    exact_sol << 0., 0.;

    EXPECT_TRUE((exact_sol - solution).norm() < 1e-5);
}

TEST(fem_pde_test, GlobalBroyden_2D){
    // define the vector field
    VectorField<Dynamic> F(2,2);
    F[0] = [](DVector<double> x) -> double { return atan(x(0)*x(1)); };
    F[1] = [](DVector<double> x) -> double { return x(0)-5*x(1); };

    // initial point
    DVector<double> x0(2);
    x0 << 10., 70.;

    Broyden<Dynamic> br(100, 1e-12);
    // auto solution = br.solveArmijo(F, x0);
    auto solution = br.solve_modified(F, x0);
    // auto solution = br.solve_modified_inv(F, x0);
    // std::cout << "solution: \n" << solution << std::endl;

    // exact solution
    DVector<double> exact_sol(2);
    exact_sol << 0., 0.;

    EXPECT_TRUE((exact_sol - solution).norm() < 1e-5);
}

// test bfgs on the norm of an operator F: Rn -> Rn
TEST(fem_pde_test, bfgs){
    VectorField<Dynamic> F(2,2);
    F[0] = [](SVector<2> x) -> double { return atan(x(0)*x(1)); };
    F[1] = [](SVector<2> x) -> double { return x(0)-x(1); };
    // F[0] = [](DVector<double> x) -> double { return x(0)*x(0); };
    // F[1] = [](DVector<double> x) -> double { return x(1)*x(1); };
    // Solution: x = 0., 0.

    ScalarField<2> f;
    f = [&](SVector<2> x) -> double {
        return F(x).squaredNorm();
    };

    // initial point
    SVector<2> x0(7.,7.);

    BFGS<2> opt(1000, 1e-15, 1);
    opt.optimize(f, x0, WolfeLineSearch());

    // expected solution
    SVector<2> expected(0., 0.);
    // std::cout << opt.optimum() << std::endl;
    // std::cout << opt.optimum().norm() << std::endl;
    double L2_error = (opt.optimum() - expected).norm();
    EXPECT_TRUE(L2_error < 1e-5);
}

// test Broyden on a grid of initial points
TEST(fem_pde_test, Broyden_2D_on_a_grid){
    // define the vector field
    VectorField<Dynamic> F(2,2);
    F[0] = [](DVector<double> x) -> double { return atan(x(0)*x(1)); };
    F[1] = [](DVector<double> x) -> double { return x(0)-5*x(1); };
    // F[0] = [](DVector<double> x) -> double { return x(0)*x(0); };
    // F[1] = [](DVector<double> x) -> double { return x(1)*x(1); };
    // Solution: x = 0., 0.

    // initial point
    DVector<double> x0(2);
    x0 << 0.01, 0.01;

    // grid of initial point
    const int n = 10; // number of initial points
    DMatrix<double> S(2,n);
    for (size_t i=0; i<2; ++i)
        for (size_t j=0; j<n; j++)
            S(i,j) = x0(i) + 0.1*i + 0.1*j;

    // solve F = 0 with Broyden
    Broyden<Dynamic> br(80, 1e-6);
    DVector<double> solution(2);

    for (size_t j=0; j<n; ++j){
        DVector<double> x1(2);
        x1 << S(0,j), S(1,j);
        solution = br.solve(F, x1);
        // solution = br.solveArmijo(F, x1);
    }
    // std::cout << "solution: \n" << solution << std::endl;

    // exact solution
    DVector<double> exact_sol(2);
    exact_sol << 0., 0.;

    EXPECT_TRUE((exact_sol - solution).norm() < 1e-5);
}