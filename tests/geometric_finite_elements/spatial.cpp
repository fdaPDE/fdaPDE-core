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

#include <fdaPDE/execution.h>
#include <fdaPDE/geometric_finite_elements.h>
#include <fdaPDE/geometry.h>
#include <gtest/gtest.h>

#include <future>

namespace {
using namespace fdapde;
using Point = SPDMatrix<double, 2, 2>;
using Batch = MatrixBatch<SPDMatrix<double, 2, 2, Cache::Log>>;
using AIRM = manifold::AffineInvariantSPDGeometry<double, 2, Usage::BasePointMaps>;
using LE = manifold::LogEuclideanSPDGeometry<double, 2, Usage::InterpolationNodes>;
/// @brief makes noncommuting data in vertex order
Batch data() {
    Batch nodes(3);
    nodes[0] = Matrix<double, 2, 2>({2., .5, .5, 3.});
    nodes[1] = Matrix<double, 2, 2>({5., -1., -1., 4.});
    nodes[2] = Matrix<double, 2, 2>({1.5, .2, .2, 2.5});
    return nodes;
}
/// @brief recognizes the batch lifetime contract without instantiating a forbidden expression
template <typename G, typename B>
concept TemporaryBatchAccepted = requires(G geometry, const Simplex<2, 2>& cell) { geometry.interpolant(cell, B {}); };
/// @brief compares spatial materialization with the barycentric solver, exact vertices and prepared edges
void check_spatial(const auto& geometry) {
    const auto nodes = data();
    gfe::P1GeodesicLinearizationOptions options;
    options.mean.solver.gradient_tolerance = 1e-11;
    auto interpolation = geometry.interpolant(Simplex<2, 2>::Unit(), nodes.select(std::array {0, 1, 2}), options);
    const Eigen::Vector2d x(.3, .5);
    const auto expression = interpolation(x);
    const Point value(expression);
    const auto result = interpolation.result(x);
    // deferred spatial materialization must agree with the diagnostic result at the same point
    EXPECT_LT((Matrix<double, 2, 2>(value - result.value).norm()), 1e-13);
    const Point vertex(interpolation(Eigen::Vector2d(1., 0.)));
    // vertex one maps to local batch entry one without applying an iterative approximation
    EXPECT_EQ((Matrix<double, 2, 2>(vertex - nodes[1]).norm()), 0.);
    const Point edge(interpolation(Eigen::Vector2d(.4, 0.)));
    const Point curve(geometry.geodesic(nodes[0], nodes[1])(.4));
    // the two-node spatial restriction is the prepared geodesic with its barycentric parameter
    EXPECT_LT((Matrix<double, 2, 2>(edge - curve).norm()), 1e-13);
    const auto temporary = geometry.interpolant(Simplex<2, 2>::Unit(), nodes.select(std::array {0, 1, 2}), options)(x);
    const Point temporary_value(temporary);
    // an expression from a temporary interpolant retains its cell and edge factors by value
    EXPECT_LT((Matrix<double, 2, 2>(temporary_value - value).norm()), 1e-13);
    // coordinates outside the cell are rejected instead of silently extrapolating a convex mean
    EXPECT_THROW(interpolation(Eigen::Vector2d(.8, .8)), std::invalid_argument);
    // nonfinite spatial coordinates cannot yield valid barycentric weights
    EXPECT_THROW(interpolation(Eigen::Vector2d(std::numeric_limits<double>::quiet_NaN(), .2)), std::invalid_argument);
    auto a = parallel_async([&] { return Point(interpolation(x)); });
    auto b = parallel_async([&] { return Point(interpolation(x)); });
    const Point first = a.get(), second = b.get();
    parallel_join();
    // each evaluation owns its candidate workspace; immutable node caches are shared safely
    EXPECT_LT((Matrix<double, 2, 2>(first - second).norm()), 1e-13);
}
// borrowed owning batches cannot be temporaries, even when the cell and geometry are copied
TEST(P1SpatialInterpolation, RejectsTemporaryBatchAtCompileTime) {
    // constraints rule out a dangling owner while permitting temporary selection expressions
    static_assert(!TemporaryBatchAccepted<LE, Batch>);
    // the same lifetime rule applies to the iterative geometry
    static_assert(!TemporaryBatchAccepted<AIRM, Batch>);
}
// LE spatial expressions preserve nodal values, edge restrictions and deferred lifetimes
TEST(P1SpatialInterpolation, LogEuclidean) { check_spatial(LE {}); }
// AIRM spatial expressions retain per-evaluation solver storage for concurrent use
TEST(P1SpatialInterpolation, AffineInvariant) { check_spatial(AIRM {}); }
// materialization exposes convergence failure with the diagnostic fields agreed by the public contract
TEST(P1SpatialInterpolation, FailureDiagnostics) {
    const auto nodes = data();
    gfe::P1GeodesicLinearizationOptions options;
    options.mean.solver.max_iterations = 1;
    options.mean.solver.gradient_tolerance = 0;
    const auto interpolation = AIRM {}.interpolant(Simplex<2, 2>::Unit(), nodes, options);
    const Eigen::Vector2d x(.3, .5);
    // the explicit result retains the failed candidate for inspection without claiming convergence
    EXPECT_FALSE(interpolation.result(x).converged());
    try {
        const Point value(interpolation(x));
        // a failed solve must never complete ordinary SPD materialization
        FAIL() << "nonconverged interpolation was materialized";
    } catch (const std::runtime_error& failure) {
        const std::string message = failure.what();
        // the error names the reason so callers can distinguish exhausted iterations from line search failure
        EXPECT_NE(message.find("stop_reason="), std::string::npos);
        // the error includes the completed iteration count for reproducibility
        EXPECT_NE(message.find("iterations="), std::string::npos);
        // the residual reports how far the failed candidate is from stationarity
        EXPECT_NE(message.find("residual="), std::string::npos);
    }
    // edge values use the exact prepared curve even when iterative mean tolerances are unattainable
    EXPECT_NO_THROW(Point(interpolation(Eigen::Vector2d(.4, 0.))));
}
// an embedded simplex rejects off-plane points instead of interpolating their projection
TEST(P1SpatialInterpolation, EmbeddedCellAndVertexOrder) {
    const auto nodes = data();
    Eigen::Matrix<double, 3, 3> coords;
    coords << 0, 1, 0, 0, 0, 1, 2, 2, 2;
    const auto interpolation = LE {}.interpolant(Simplex<2, 3>(coords), nodes.select(std::array {2, 0, 1}));
    const Point at_origin(interpolation(Eigen::Vector3d(0, 0, 2)));
    // local entry zero follows the supplied vertex selection rather than the global batch index
    EXPECT_EQ((Matrix<double, 2, 2>(at_origin - nodes[2]).norm()), 0.);
    // barycentric projection alone would accept this point; the cell plane check must reject it
    EXPECT_THROW(interpolation(Eigen::Vector3d(.2, .2, 2.1)), std::invalid_argument);
}
}   // namespace
