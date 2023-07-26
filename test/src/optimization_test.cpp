#include <cstddef>
#include <gtest/gtest-typed-test.h>
#include <gtest/gtest.h> // testing framework

#include "../../src/utils/symbols.h"
#include "../../src/optimization/bfgs.h"
#include "../../src/optimization/grid.h"
#include "../../src/optimization/gradient_descent.h"
#include "../../src/optimization/newton.h"
using fdapde::core::BFGS;
using fdapde::core::Grid;
using fdapde::core::GradientDescent;
using fdapde::core::Newton;
#include "../../src/optimization/callbacks/backtracking_line_search.h"
#include "../../src/optimization/callbacks/wolfe_line_search.h"
using fdapde::core::BacktrackingLineSearch;
using fdapde::core::WolfeLineSearch;
#include "../../src/fields/scalar_field.h"
using fdapde::core::ScalarField;
using fdapde::core::TwiceDifferentiableScalarField;
#include "../../src/fields/vector_field.h"
using fdapde::core::VectorField;

#include "utils/utils.h"
using fdapde::testing::almost_equal;
#include "utils/constants.h"
using fdapde::testing::DOUBLE_TOLERANCE;

TEST(Optimization, Grid) {
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

TEST(Optimization, GradientDescent_BacktrackingLineSearch) {
  // define objective function: x*e^{-x^2 - y^2} + (x^2 + y^2)/20
  ScalarField<2> f;
  f = [](SVector<2> x) -> double {                                                                   
    return x[0]*std::exp(-x[0]*x[0] - x[1]*x[1]) + (x[0]*x[0] + x[1]*x[1])/20;
  };                                                                                                                 
  f.set_step(1e-4);

  GradientDescent<2> opt(1000, 1e-6, 0.01);
  SVector<2> pt(-1,-1);
  opt.optimize(f, pt, BacktrackingLineSearch());

  // expected solution
  SVector<2> expected(-0.6690718221499544, 0);
  double L2_error = (opt.optimum() - expected).norm();
  EXPECT_TRUE(L2_error < 1e-6);
}

TEST(Optimization, Newton_BacktrackingLineSearch) {
  // define objective function: x*e^{-x^2 - y^2} + (x^2 + y^2)/20
  ScalarField<2> f;
  f = [](SVector<2> x) -> double {                                                                   
    return x[0]*std::exp(-x[0]*x[0] - x[1]*x[1]) + (x[0]*x[0] + x[1]*x[1])/20;
  };                                                                                                                 
  f.set_step(1e-4);
  
  Newton<2> opt(1000, 1e-6, 0.01);
  SVector<2> pt(-0.5,-0.5);
  opt.optimize(f, pt, BacktrackingLineSearch());

  // expected solution
  SVector<2> expected(-0.6690718221499544, 0);
  double L2_error = (opt.optimum() - expected).norm();
  EXPECT_TRUE(L2_error < 1e-6);
}

// test integration of Laplacian weak form for a LagrangianBasis of order 2
TEST(Optimization, BFGS_WolfeLineSearch) {
  // define objective function: x*e^{-x^2 - y^2} + (x^2 + y^2)/20
  ScalarField<2> f;
  f = [](SVector<2> x) -> double {                                                                   
    return x[0]*std::exp(-x[0]*x[0] - x[1]*x[1]) + (x[0]*x[0] + x[1]*x[1])/20;
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
