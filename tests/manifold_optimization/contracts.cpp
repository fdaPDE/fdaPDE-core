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

// the inherited dense debug guards erase their operands in this focused NoDebug probe
// suppress only their resulting unused diagnostics; geometry diagnostics remain enabled
#if defined(__GNUC__) || defined(__clang__)
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wunused-parameter"
#    pragma GCC diagnostic ignored "-Wunused-variable"
#endif
#include <fdaPDE/dense_linear_algebra.h>
#if defined(__GNUC__) || defined(__clang__)
#    pragma GCC diagnostic pop
#endif

#include <fdaPDE/manifold_optimization.h>
#include <gtest/gtest.h>

namespace {
using namespace fdapde;
using namespace fdapde::manifold;

// this target disables debug macros; these public contracts must still throw
template <typename Geometry> void runtime_contracts() {
    const Geometry geometry(2);
    Matrix<double, 2, 2> dense;
    dense.set_zero();
    dense(0, 0) = 1;
    dense(1, 1) = 1;
    typename Geometry::Point point(dense);
    auto u = geometry.zero_tangent(point);
    u(0, 0) = 1;
    auto bad = u;
    for (const double invalid : {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity()}) {
        bad(1, 0) = invalid;
        // projection rejects a nonfinite off-diagonal tangent coefficient with debug assertions disabled
        EXPECT_THROW(geometry.project(point, bad), std::invalid_argument);
        // norm rejects the same nonfinite tangent before metric evaluation
        EXPECT_THROW(geometry.norm(point, bad), std::invalid_argument);
        // the metric product validates the coefficients of its second tangent operand
        EXPECT_THROW(geometry.inner_product(point, u, bad), std::invalid_argument);
        // a linear combination rejects nonfinite input coefficients before arithmetic
        EXPECT_THROW(geometry.linear_combination(point, 1, bad, 0, u), std::invalid_argument);
        // a zero exponential step does not bypass validation of its tangent
        EXPECT_THROW(geometry.exponential(point, bad, 0), std::invalid_argument);
        // a zero retraction step likewise rejects an invalid tangent
        EXPECT_THROW(geometry.retract(point, bad, 0), std::invalid_argument);
        // transport rejects an invalid tangent even when both endpoints coincide
        EXPECT_THROW(geometry.transport(point, point, bad), std::invalid_argument);
        // gradient conversion rejects nonfinite ambient gradient coefficients
        EXPECT_THROW(geometry.euclidean_to_riemannian_gradient(point, bad), std::invalid_argument);
        // the first linear-combination weight must be finite
        EXPECT_THROW(geometry.linear_combination(point, invalid, u, 0, u), std::invalid_argument);
        // the second linear-combination weight must also be finite
        EXPECT_THROW(geometry.linear_combination(point, 1, u, invalid, u), std::invalid_argument);
        // exponentiation rejects a nonfinite step even with a valid tangent
        EXPECT_THROW(geometry.exponential(point, u, invalid), std::invalid_argument);
        // retraction rejects a nonfinite step even with a valid tangent
        EXPECT_THROW(geometry.retract(point, u, invalid), std::invalid_argument);
    }
    const auto snapshot = point;
    auto huge = u;
    huge(0, 0) = std::numeric_limits<double>::max();
    // an overflowing retraction reports a domain error before assignment can replace the point
    EXPECT_THROW(point = geometry.retract(point, huge, 2), std::domain_error);
    // the observed destination coefficient is unchanged after the failed assignment
    EXPECT_EQ(point(0, 0), snapshot(0, 0));
    // an overflowing weighted tangent combination reports a domain error
    EXPECT_THROW(geometry.linear_combination(point, 2, huge, 0, u), std::domain_error);
    // an unrepresentable metric product reports a domain error
    EXPECT_THROW(geometry.inner_product(point, huge, huge), std::domain_error);
    // the valid unit diagonal tangent still has norm one at the identity
    EXPECT_NEAR(geometry.norm(point, u), 1, 1e-13);

    Matrix<double, 3, 3> wrong_dense;
    wrong_dense.set_zero();
    for (int i = 0; i < 3; ++i) wrong_dense(i, i) = 1;
    const typename Geometry::Point wrong(wrong_dense);
    typename Geometry::Tangent wrong_u(3, 3);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j <= i; ++j) wrong_u(i, j) = 0;
    }
    // a geometry of order two rejects a zero-tangent request at an order-three point
    EXPECT_THROW(geometry.zero_tangent(wrong), std::invalid_argument);
    // distance rejects a target whose order differs from the geometry
    EXPECT_THROW(geometry.distance(point, wrong), std::invalid_argument);
    // logarithm validates the order of its source point
    EXPECT_THROW(geometry.logarithm(wrong, point), std::invalid_argument);
    // transport validates the order of its target point
    EXPECT_THROW(geometry.transport(point, wrong, u), std::invalid_argument);
    // exponentiation rejects an order-three tangent for this order-two geometry
    EXPECT_THROW(geometry.exponential(point, wrong_u), std::invalid_argument);
    // gradient conversion rejects the same incompatible tangent shape
    EXPECT_THROW(geometry.euclidean_to_riemannian_gradient(point, wrong_u), std::invalid_argument);

    // returning a value from temporary arguments leaves no borrowed geometry state
    auto detached = [&] {
        const typename Geometry::Point local(dense);
        return Geometry(2).transport(local, local, geometry.project(local, u));
    }();
    const auto saved = detached;
    detached(0, 0) = 9;
    // a copied transport result retains its value after the other result is modified
    EXPECT_NEAR(saved(0, 0), 1, 1e-13);
    // modifying the detached result does not write through to the source tangent
    EXPECT_EQ(u(0, 0), 1);
    auto next = geometry.exponential(point, u, 0.1);
    const auto next_copy = next;
    next.assign(dense);
    // replacing the new point leaves its previous copy equal to the analytic exponential
    EXPECT_NEAR(next_copy(0, 0), std::exp(0.1), 1e-13);
}
}   // namespace

// log-Euclidean public checks and result ownership hold with debug assertions disabled
TEST(SPDGeometryContracts, LogEuclideanChecksAndValueOwnership) {
    // exercise the shared invalid-input, failure and ownership cases with the log-Euclidean metric
    runtime_contracts<LogEuclideanSPDGeometry<double, Dynamic>>();
}
// affine-invariant public checks and result ownership hold with debug assertions disabled
TEST(SPDGeometryContracts, AffineInvariantChecksAndValueOwnership) {
    // exercise the same contracts with the affine-invariant metric
    runtime_contracts<AffineInvariantSPDGeometry<double, Dynamic>>();
}

// both metrics reject finite double steps that cannot be represented by their float scalar type
TEST(SPDGeometryContracts, RejectsUnrepresentableFloatSteps) {
    Matrix<float, 2, 2> dense;
    dense.set_zero();
    dense(0, 0) = 1;
    dense(1, 1) = 1;
    const SPDMatrix<float, 2, 2> point(dense);
    const LogEuclideanSPDGeometry<float, 2> le;
    const AffineInvariantSPDGeometry<float, 2> airm;
    const auto u = le.zero_tangent(point);
    // a finite double step too large for float is rejected by the log-Euclidean exponential
    EXPECT_THROW(le.exponential(point, u, std::numeric_limits<double>::max()), std::invalid_argument);
    // the affine-invariant retraction also rejects a step that float cannot represent
    EXPECT_THROW(airm.retract(point, u, std::numeric_limits<double>::max()), std::invalid_argument);
}
