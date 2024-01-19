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
using fdapde::core::Optimizer;
using fdapde::core::BFGS;
using fdapde::core::GradientDescent;
using fdapde::core::Grid;
using fdapde::core::Newton;
using fdapde::core::BacktrackingLineSearch;
using fdapde::core::WolfeLineSearch;
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

    GradientDescent<2, BacktrackingLineSearch> opt(1000, 1e-6, 0.01);
    SVector<2> pt(-1, -1);
    opt.optimize(f, pt);

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

    Newton<2, BacktrackingLineSearch> opt(1000, 1e-6, 0.01);
    SVector<2> pt(-0.5, -0.5);
    opt.optimize(f, pt);

    // expected solution
    SVector<2> expected(-0.6690718221499544, 0);
    double L2_error = (opt.optimum() - expected).norm();
    EXPECT_TRUE(L2_error < 1e-6);
}

// test integration of Laplacian weak form for a LagrangianBasis of order 2
TEST(optimization_test, type_erased_bfgs_wolfe_line_search) {
    // define objective function: x*e^{-x^2 - y^2} + (x^2 + y^2)/20
    ScalarField<2> f;
    f = [](SVector<2> x) -> double {
        return x[0] * std::exp(-x[0] * x[0] - x[1] * x[1]) + (x[0] * x[0] + x[1] * x[1]) / 20;
    };
    f.set_step(1e-4);

    // define optimizer
    Optimizer<ScalarField<2>> opt = BFGS<2, WolfeLineSearch>(1000, 1e-6, 0.01); // use a type erasure wrapper
    // perform optimization
    SVector<2> pt(-1, -1);
    opt.optimize(f, pt);

    // expected solution
    SVector<2> expected(-0.6690718221499544, 0);
    double L2_error = (opt.optimum() - expected).norm();
    EXPECT_TRUE(L2_error < 1e-6);
}
