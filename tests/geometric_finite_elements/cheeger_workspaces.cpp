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
#include <gtest/gtest.h>

namespace {
using namespace fdapde;

/// @brief compares fixed and runtime coordinate storage before and after the same multistart search
template <int Count> void check_lift_storage() {
    using Runtime = manifold::internals::CheegerLift<double, 3>;
    using Fixed = manifold::internals::CheegerLift<double, 3, Count>;
    using Sym = Runtime::Sym;
    MatrixBatch<Sym> charts(Count);
    std::vector<double> weights(Count);
    for (int i = 0; i < Count; ++i) {
        Sym chart;
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c <= r; ++c) chart(r, c) = r == c ? .13 * (i + 1) * (r + 1) : .04 * std::sin(1 + i + r + c);
        charts[i] = chart;
        weights[i] = double(i + 1) / (Count * (Count + 1) / 2);
    }
    const std::optional<Sym> base = Count == 1 ? std::optional<Sym>(Sym(charts[0] * .5)) : std::nullopt;
    Runtime runtime(charts, weights, .6, base);
    Fixed fixed(charts, weights, .6, base);
    typename Runtime::Batch rotations(Count);
    for (int i = 0; i < Count; ++i) rotations[i] = rotation_exp(runtime.basis(i % 3), .03 * (i + 1));
    const auto reference = runtime.evaluate(rotations);
    const auto actual = fixed.evaluate(rotations);
    // the fixed coordinate count removes dynamic optimizer-vector storage for each active simplex size
    static_assert(Fixed::Vec::Rows == 3 * Count);
    // a static Hessian carries the same coordinate dimension in both axes
    static_assert(Fixed::Dense::Rows == 3 * Count && Fixed::Dense::Cols == 3 * Count);
    // fixed lift states keep aligned charts inline instead of allocating a vector at each candidate
    static_assert(std::is_same_v<typename Fixed::Charts, std::array<Sym, Count>>);
    // fixed lift states also retain rotation logarithms in inline storage
    static_assert(std::is_same_v<typename Fixed::Logarithms, std::array<typename Fixed::Skew, Count>>);
    // runtime states keep a growable container for arbitrary active counts
    static_assert(std::is_same_v<typename Runtime::Charts, std::vector<Sym>>);
    // identical frame arithmetic retains the exact cost before optimization
    EXPECT_EQ(actual.cost, reference.cost);
    // inline gradient storage retains every body-coordinate coefficient exactly
    EXPECT_EQ((Runtime::Vec(actual.gradient - reference.gradient).norm()), 0.);
    const auto expected_hessian = runtime.hessian(reference);
    const auto actual_hessian = fixed.hessian(actual);
    // static Hessian assembly preserves the dynamic arithmetic and coupling terms exactly
    EXPECT_EQ((Runtime::Dense(actual_hessian - expected_hessian).norm()), 0.);
    typename Fixed::Problem problem {&fixed};
    typename Fixed::Problem::Workspace workspace;
    // inner optimizer evaluations omit a rotation copy while retaining the same scalar objective
    EXPECT_EQ(problem.cost(rotations, workspace), actual.cost);
    // the workspace needs charts and logarithms for derivatives but does not own the optimizer point
    EXPECT_EQ(workspace.state->rotations.size(), 0);
    const auto workspace_gradient = problem.grad(rotations, workspace);
    // cached derivatives agree with a complete evaluation at the same nonstationary point
    EXPECT_EQ((typename Fixed::Vec(workspace_gradient - actual.gradient).norm()), 0.);
    const auto direction = actual.gradient;
    const auto workspace_action = problem.hess(rotations, direction, workspace);
    // the lazy Hessian action does not require the omitted rotation batch
    EXPECT_EQ((typename Fixed::Vec(workspace_action - actual_hessian * direction).norm()), 0.);
    const auto expected = runtime.solve();
    const auto solved = fixed.solve();
    // dispatch keeps the same deterministic local starts and chosen objective
    EXPECT_EQ(solved.cost, expected.cost);
    // preserving candidate order retains the same stationarity certificate
    EXPECT_EQ(solved.gradient.norm(), expected.gradient.norm());
    // static and runtime solvers agree on the detected competing-minimum diagnostic
    EXPECT_EQ(fixed.ambiguous(), runtime.ambiguous());
    // the selected frame must meet the unchanged inner solver tolerance
    ASSERT_TRUE(fixed.converged());
    fixed.prepare(solved);
    typename Fixed::Vec rhs(3 * Count);
    for (int i = 0; i < rhs.rows(); ++i) rhs[i] = .03 * std::cos(i + 1);
    const auto solution = fixed.solve_linear(rhs);
    // the retained static factorization satisfies the same relative residual certificate
    EXPECT_LE(fixed.residual(solution, rhs), 1e-9 * rhs.norm() + 32 * std::numeric_limits<double>::epsilon());
}

// pair, edge, triangle and tetrahedron lifts retain their solver and implicit-derivative certificates
TEST(CheegerWorkspaces, SmallActiveLiftsPreserveDynamicArithmetic) {
    check_lift_storage<1>();
    check_lift_storage<2>();
    check_lift_storage<3>();
    check_lift_storage<4>();
}

// arbitrary active node counts keep the original dynamic workspace path
TEST(CheegerWorkspaces, LargerBatchesKeepTheDynamicFallback) {
    using Lift = manifold::internals::CheegerLift<double, 3>;
    MatrixBatch<Lift::Sym> charts(5);
    for (int i = 0; i < 5; ++i) charts[i] = Matrix<double, 3, 3>({.2, 0., 0., 0., .2, 0., 0., 0., .2});
    Lift lift(charts, std::vector<double>(5, .2), .6);
    const auto solved = lift.solve();
    // equal isotropic charts admit the zero-cost identity lift at arbitrary active counts
    EXPECT_NEAR(solved.cost, 0., 1e-30);
    // five nodes retain fifteen rotation coordinates without an artificial simplex size bound
    EXPECT_EQ(solved.gradient.rows(), 15);
    // the fallback retains the same inner stationarity tolerance
    EXPECT_TRUE(lift.converged());
    // duplicate deterministic starts do not introduce a false geometric ambiguity
    EXPECT_FALSE(lift.ambiguous());
}
}   // namespace
