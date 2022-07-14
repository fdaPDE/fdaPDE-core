#include <iostream>
#include <functional>
#include <ostream>
#include <tuple>

#include "../utils/Symbols.h"
#include "../utils/fields/ScalarField.h"
#include "optimizers/Newton.h"
#include "extensions/Summarize.h"
#include "optimizers/GradientDescent.h"
#include "extensions/BacktrackingAdaptiveStep.h"
#include "optimizers/BFGS.h"
#include "optimizers/ExactNewton.h"

using namespace fdaPDE::core::OPT;
// load symbols
using fdaPDE::core::ScalarField;
using fdaPDE::core::DifferentiableScalarField;
using fdaPDE::core::TwiceDifferentiableScalarField;

int main() {

  std::function<double(SVector<2>)> g_newton = [](SVector<2> x) -> double {
    return 2*std::pow(x[0],2) + x[0] + 2*std::pow(x[1],2);
  };

  ScalarField<2> objective(g_newton);
  
  NewtonOptimizer<2> Optim(10000, 0.001, 0.001, 0.001);
  Optim.setStepSize(0.001);
  
  std::pair<SVector<2>,double> min = Optim.findMinimum(objective, SVector<2>(1,1), Summarize());

  // define differentiable scalar field
  std::function<SVector<2>(SVector<2>)> dg = [](SVector<2> x) -> SVector<2> {
    return SVector<2>({4*x[0] + 1, 4*x[1]});
  };
  std::function<SMatrix<2>(SVector<2>)> ddg = [](SVector<2> x) -> SMatrix<2> {
    return SMatrix<2>({{4, 0},
		       {0, 4}});
  };

  TwiceDifferentiableScalarField<2> dddfun(g_newton, dg, ddg);

  ExactNewtonOptimizer<2> ExactOptim(10000, 0.001);
  ExactOptim.setStepSize(0.001);

  std::pair<SVector<2>,double> min_Exact = ExactOptim.findMinimum(dddfun, SVector<2>(1,1),
								  Summarize());

  GradientDescentOptimizer<2> gradOptim(10000, 0.001);
  gradOptim.setStepSize(0.001);

  DifferentiableScalarField<2> ddfun(g_newton, dg);
  std::pair<SVector<2>,double> min_Grad = gradOptim.findMinimum(ddfun, SVector<2>(1,1),
								BacktrackingAdaptiveStep(1, 0.2, 0.3),
								Summarize());

  BFGSOptimizer<2> bgfsOptim(10000, 0.001);
  bgfsOptim.setStepSize(0.001);

  std::pair<SVector<2>,double> min_BFGS = bgfsOptim.findMinimum(ddfun, SVector<2>(1,1),
								BacktrackingAdaptiveStep(1, 0.2, 0.3),
								Summarize());
  
  return 0;
}
