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

#include <fdaPDE/geometric_discretization.h>
#include <fdaPDE/optimization.h>
#include <gtest/gtest.h>

namespace {
using namespace fdapde;
/// @brief fills the six independent symmetric coefficients in matrix coordinate order
SymmetricMatrix<double, 3> symmetric(double a, double b, double c, double d, double e, double f) {
    SymmetricMatrix<double, 3> result;
    result(0, 0) = a;
    result(0, 1) = b;
    result(0, 2) = c;
    result(1, 1) = d;
    result(1, 2) = e;
    result(2, 2) = f;
    return result;
}
// stable changes resolve descent hidden by a large constant and reject a rounded-equal increase
TEST(StepControl, OffsetInvariantArmijo) {
    const double offset = 1e16;
    // ordinary full-objective subtraction erases this positive change
    EXPECT_EQ(offset + .125, offset);
    const auto reject = armijo_step([](double) { return .125; }, -1., 1., .5, 1e-4, 5);
    // positive increments never satisfy sufficient decrease even when full costs would round equal
    EXPECT_FALSE(reject.accepted);
    // the configured budget bounds evaluations of an inadmissible direction
    EXPECT_EQ(reject.evaluations, 5);
    const auto accept = armijo_step([](double step) { return -step + .5 * step * step; }, -1.);
    // the exact quadratic change accepts the unit descent step independently of the constant offset
    EXPECT_TRUE(accept.accepted);
    // the difference oracle retains the known unit-step decrease
    EXPECT_DOUBLE_EQ(accept.change, -.5);
}

// invalid trials backtrack and invalid search parameters fail at the public boundary
TEST(StepControl, DomainAndDirection) {
    const auto step = armijo_step([](double h) { return h > .25 ? INFINITY : -h; }, -1.);
    // the first admissible point satisfies the strict negative Armijo bound
    EXPECT_TRUE(step.accepted);
    // two invalid points precede the quarter step
    EXPECT_EQ(step.evaluations, 3);
    // a non-descent direction cannot silently claim success
    EXPECT_THROW(armijo_step([](double) { return -1.; }, 0.), std::invalid_argument);
    // an underflowed zero bound cannot accept a zero objective increment
    EXPECT_FALSE(armijo_step([](double) { return 0.; }, -1e-300, 1e-100).accepted);
}

// the mass dual norm reports the actual controlling coefficient and its spatial unit dependence
TEST(StepControl, LumpedDensityUnits) {
    const std::vector<double> gradient {.02, -.03, .01, -.08}, masses {.001, .02}, scaled {.008, .16};
    const auto first = lumped_gradient_summary(gradient, masses, 2);
    const auto second = lumped_gradient_summary(gradient, scaled, 2);
    // the raw infinity norm is controlled by the last coefficient
    EXPECT_DOUBLE_EQ(first.raw_inf, .08);
    // the smallest mass makes the second coefficient control the density instead
    EXPECT_EQ(first.density_index, 1u);
    // dividing each coefficient by its own mass yields the specified maximum
    EXPECT_DOUBLE_EQ(first.density_inf, 30.);
    // multiplying spatial coordinates by two multiplies three-dimensional masses by eight
    EXPECT_DOUBLE_EQ(second.density_inf, first.density_inf / 8);
    // volume times density is unchanged by a uniform coordinate-unit change
    EXPECT_DOUBLE_EQ(first.volume * first.density_inf, second.volume * second.density_inf);
    // an incomplete nodal block is not a valid interleaved covector
    EXPECT_THROW(lumped_gradient_summary(gradient, masses, 3), std::invalid_argument);
    // a zero lumped mass cannot define a density
    EXPECT_THROW(lumped_gradient_summary(gradient, std::vector<double> {0, 1}, 2), std::invalid_argument);
}

// compensation retains low order contributions lost in a naive reduction
TEST(StepControl, CompensatedCancellation) {
    CompensatedSum sum;
    sum.add(1e16);
    sum.add(1);
    sum.add(-1e16);
    // the exact finite sum is one despite the large cancelling summands
    EXPECT_DOUBLE_EQ(sum.value(), 1.);
}

// the spectral increment resolves sub-ulp commuting changes without subtracting exponentials
TEST(StepControl, ExponentialTinyIncrement) {
    const auto base = symmetric(.4, 0, 0, -.2, 0, .1);
    const auto delta = symmetric(1e-18, 0, 0, -2e-18, 0, 3e-18);
    const auto change = matrix_exp_increment(base, delta);
    for (int i = 0; i < 3; ++i) {
        const double exact = std::exp(base(i, i)) * std::expm1(delta(i, i));
        // diagonal entries agree with the scalar expm1 identity below the full-value ulp
        EXPECT_NEAR(change(i, i), exact, std::abs(exact) * 2e-14);
    }
}

// changing eigenvector bases preserves the noncommuting exponential difference and its differential
TEST(StepControl, ExponentialNoncommutingIncrement) {
    const auto base = symmetric(.4, .12, -.06, -.2, .09, .1);
    const auto delta = symmetric(.03, -.02, .01, .04, .02, -.01);
    const auto change = matrix_exp_increment(base, delta);
    const auto before = matrix_exp(base), after = matrix_exp(SymmetricMatrix<double, 3>(base + delta));
    const SymmetricMatrix<double, 3> tiny(1e-17 * delta);
    const auto tiny_change = matrix_exp_increment(base, tiny);
    const auto differential = matrix_exp_frechet(base, tiny);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            // finite noncommuting increments agree with an independently evaluated resolvable difference
            EXPECT_NEAR(change(i, j), after(i, j) - before(i, j), 4e-15);
            // an infinitesimal noncommuting increment matches the independent Frechet kernel
            EXPECT_NEAR(tiny_change(i, j), differential(i, j), 1e-31);
        }
    const SymmetricMatrix<double, Dynamic> dynamic_base(base), dynamic_delta(delta);
    const auto dynamic_change = matrix_exp_increment(dynamic_base, dynamic_delta);
    // dynamic shape materialization preserves the same spectral identity
    EXPECT_NEAR(dynamic_change(1, 2), change(1, 2), 1e-15);
}

// edge differences reproduce the signed stiffness action and exactly annihilate constant fields
TEST(StepControl, P1FlatStiffness) {
    const gfe::PreparedP1LumpedLaplacianStencil stencil({
      {1., 2., 3.},
      {{0, 1, -2.}, {1, 2, -.5}}
    });
    const std::vector<double> field {.1, .4, .7}, constant(3, .37);
    // the first stiffness row is twice the first nodal difference
    EXPECT_NEAR(gfe::p1_lumped_stiffness_at(stencil, field, 0), -.6, 2e-16);
    // signed incident edges give the analytic middle row
    EXPECT_NEAR(gfe::p1_lumped_stiffness_at(stencil, field, 1), .45, 2e-16);
    for (std::size_t i = 0; i < 3; ++i) {
        // a constant coordinate vector has exactly zero stiffness residual before summation
        EXPECT_DOUBLE_EQ(gfe::p1_lumped_stiffness_at(stencil, constant, i), 0.);
    }
}
// approximate Wolfe resolves a hidden quadratic decrease using directional derivatives
TEST(StepControl, ApproximateWolfeHiddenQuadratic) {
    const double offset = 1e16;
    const auto result = approximate_wolfe_step(
      [=](double h) { return std::pair {offset + .5 * (1 - h) * (1 - h), h - 1}; }, offset + .5, -1.);
    // the unit step is the exact quadratic minimizer despite equal rounded full costs
    EXPECT_TRUE(result.accepted);
    // acceptance explicitly identifies the derivative-based numerical regime
    EXPECT_TRUE(result.approximate);
    // the endpoint interpolation recovers the analytic quadratic change
    EXPECT_DOUBLE_EQ(result.quadratic_change, -.5);
    // the exact difference would satisfy the original Armijo condition
    EXPECT_LE(result.quadratic_change, result.bound);
}

// a constant cost and unchanged directional derivative cannot pass as spurious stationarity progress
TEST(StepControl, ApproximateWolfeRejectsFlatDerivative) {
    const auto result = approximate_wolfe_step([](double) { return std::pair {1., -1.}; }, 1., -1., 1., 1.);
    // curvature must improve even if full values are indistinguishable
    EXPECT_FALSE(result.accepted);
    // reaching the step cap terminates rather than shrinking indefinitely
    EXPECT_EQ(result.evaluations, 1);
}

// resolvable energy increases cannot be excused by the approximate Wolfe derivative condition
TEST(StepControl, ApproximateWolfeRejectsObservableIncrease) {
    double evaluated_step = 0;
    const auto result = approximate_wolfe_step(
      [&](double h) {
          evaluated_step = h;
          return std::pair {1.01, 0.};
      },
      1., -1., 1., 1., 1e-4, .9, 5);
    // a positive increase far above roundoff violates both acceptance branches
    EXPECT_FALSE(result.accepted);
    // repeated failed trials respect the finite oracle budget
    EXPECT_EQ(result.evaluations, 5);
    // the reported step is the one actually evaluated last
    EXPECT_DOUBLE_EQ(result.step, evaluated_step);
}

// expansion finds derivative curvature when a tiny starting step remains too close to its initial slope
TEST(StepControl, ApproximateWolfeExpandsSmallSteps) {
    const auto result = approximate_wolfe_step(
      [](double h) { return std::pair {1e16 + .5 * (1 - h) * (1 - h), h - 1}; }, 1e16 + .5, -1., .01, 1.);
    // bounded expansion eventually reaches a sufficient change in directional slope
    EXPECT_TRUE(result.accepted);
    // powers of two from .01 first satisfy the .9 slope gate at .16
    EXPECT_NEAR(result.step, .16, 1e-16);
}

// an invalid initial trial is bracketed before a valid finite quadratic step is accepted
TEST(StepControl, ApproximateWolfeDomainAndResolvedBranch) {
    const auto result = approximate_wolfe_step(
      [](double h) {
          return h > .6 ? std::pair<double, double> {INFINITY, NAN} : std::pair {.5 * (1 - h) * (1 - h), h - 1};
      },
      .5, -1.);
    // the half step has a resolvable Armijo decrease and sufficient curvature
    EXPECT_TRUE(result.accepted);
    // the first inadmissible point does not contaminate the accepted derivative
    EXPECT_DOUBLE_EQ(result.step, .5);
    // a resolved ordinary Wolfe step is distinguished from the approximate fallback
    EXPECT_FALSE(result.approximate);
    // only one failed domain trial precedes the accepted point
    EXPECT_EQ(result.evaluations, 2);
    // a non-descent input is rejected at the public boundary
    EXPECT_THROW(approximate_wolfe_step([](double) { return std::pair {0., 0.}; }, 0., 0.), std::invalid_argument);
}

}   // namespace
