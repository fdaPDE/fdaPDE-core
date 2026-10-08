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

#include <fdaPDE/finite_elements.h>
#include <fdaPDE/geometric_finite_elements.h>
#include <gtest/gtest.h>

namespace {
using namespace fdapde;
using Point = SPDMatrix<double, 2>;
using Batch = MatrixBatch<SPDMatrix<double, 2, Cache::Log>>;
using Locations = MatrixBatch<Vector<double, 2>>;
using LE = manifold::LogEuclideanSPDGeometry<double, 2, Usage::InterpolationNodes>;
using AIRM = manifold::AffineInvariantSPDGeometry<double, 2, Usage::BasePointMaps>;
/// @brief supplies two triangles whose local and global vertex orders differ
Triangulation<2, 2> domain() {
    Eigen::Matrix<double, 4, 2> vertices;
    vertices << 0, 0, 1, 0, 0, 1, 1, 1;
    Eigen::Matrix<int, 2, 3> cells;
    cells << 2, 0, 1, 1, 3, 2;
    return {vertices, cells, Eigen::Matrix<int, 4, 1>::Ones()};
}
/// @brief supplies noncommuting coefficients in global DOF order
Batch coefficients() {
    Batch batch(4);
    batch[0] = Matrix<double, 2, 2>({2., .5, .5, 3.});
    batch[1] = Matrix<double, 2, 2>({5., -1., -1., 4.});
    batch[2] = Matrix<double, 2, 2>({1.5, .2, .2, 2.5});
    batch[3] = Matrix<double, 2, 2>({6., .3, .3, 2.});
    return batch;
}
/// @brief orders repeated interior points, a shared edge and an exact vertex across two cells
Locations locations() {
    Locations points(6);
    points[0] = Vector<double, 2> {.8, .7};
    points[1] = Vector<double, 2> {.2, .3};
    points[2] = Vector<double, 2> {.5, .5};
    points[3] = Vector<double, 2> {0., 1.};
    points[4] = Vector<double, 2> {.2, .3};
    points[5] = Vector<double, 2> {.8, .7};
    return points;
}
/// @brief measures symmetric coefficient error without retaining temporary result owners
double error(const auto& lhs, const auto& rhs) { return Matrix<double, 2, 2>(lhs - rhs).norm(); }
/// @brief checks policy independence, owned locations, coefficient replacement and scalar agreement
void check_multipoint(const auto& geometry) {
    const auto mesh = domain();
    const GeometricFeSpace space(mesh, P1<1>, geometry);
    auto points = locations();
    const auto prepared = space.prepare_evaluation(points);
    // spatial preparation copies lvalue location buffers rather than borrowing mutable caller storage
    EXPECT_NE(prepared.locations().coefficients().data(), points.coefficients().data());
    // points in the same cell share one spatial geometry and local DOF record
    EXPECT_EQ(prepared.prepared_cells(), 2);
    auto moved_locations = locations();
    const auto* transferred = moved_locations.coefficients().data();
    const auto parallel_prepared = space.prepare_evaluation(std::move(moved_locations), execution_par);
    // preparation transfers native location storage from an rvalue batch
    EXPECT_EQ(parallel_prepared.locations().coefficients().data(), transferred);
    GeometricFeFunction function(space, coefficients());
    // preparing locations is independent of coefficient-dependent geodesics and caches
    EXPECT_EQ(function.prepared_cells(), 0);
    const auto sequential = prepared(function);
    if constexpr (gfe::internals::is_log_euclidean_spd_geometry<std::remove_cvref_t<decltype(geometry)>>) {
        // LE batch evaluation uses cached nodal logs without preparing unused per-cell edge curves
        EXPECT_EQ(function.prepared_cells(), 0);
    } else {
        // AIRM prepares each visited cell once and retains its vertex and edge shortcuts
        EXPECT_EQ(function.prepared_cells(), 2);
    }
    const auto parallel = prepared(function, execution_par);
    const auto parallel_then_seq = parallel_prepared(function, execution_seq);
    const auto parallel_then_par = parallel_prepared(function, execution_par);
    const auto once = function.eval_at(points);
    for (std::size_t i = 0; i < points.size(); ++i) {
        const Eigen::Vector2d x(points[i](0, 0), points[i](1, 0));
        const Point expected(function(x));
        // batch positions preserve location order and agree with the existing one-point operator
        EXPECT_LT(error(sequential[i], expected), 2e-12);
        // sequential preparation can be combined with parallel evaluation without changing values
        EXPECT_LT(error(parallel[i], sequential[i]), 1e-13);
        // parallel preparation can be combined independently with sequential evaluation
        EXPECT_LT(error(parallel_then_seq[i], sequential[i]), 1e-13);
        // using parallel policies for both phases preserves deterministic per-point computation
        EXPECT_LT(error(parallel_then_par[i], sequential[i]), 1e-13);
        // the one-shot convenience delegates to the same preparation and evaluation engine
        EXPECT_LT(error(once[i], sequential[i]), 1e-13);
    }
    // a one-hot shape value preserves the exact certified nodal matrix
    EXPECT_EQ(error(sequential[3], function.coeff()[2]), 0.);
    points[0] = Vector<double, 2> {100., 100.};
    auto copied_preparation = prepared;
    const auto independent = copied_preparation(function);
    // copying a preparation retains its own cell records, unaffected by later caller location changes
    EXPECT_LT(error(independent[0], sequential[0]), 1e-13);
    Locations one_location(1);
    one_location[0] = Vector<double, 2> {.2, .3};
    const auto one_prepared = space.prepare_evaluation(std::move(one_location));
    copied_preparation = one_prepared;
    const auto reassigned = copied_preparation(function);
    // reassignment replaces point count and spatial metadata together rather than retaining old point ids
    EXPECT_EQ(reassigned.size(), 1);
    // the reassigned plan evaluates the replacement location using its matching cell and shape weights
    EXPECT_LT(error(reassigned[0], sequential[1]), 1e-13);
    const Point constant(Matrix<double, 2, 2>({4., .2, .2, 3.}));
    auto replacement = coefficients();
    for (std::size_t i = 0; i < replacement.size(); ++i) replacement[i] = constant;
    GeometricFeFunction other_function(space, replacement);
    const auto other_values = prepared(other_function, execution_par);
    function.set_coeff(std::move(replacement));
    const auto after_update = prepared(function, execution_par);
    for (std::size_t i = 0; i < prepared.size(); ++i) {
        // the same spatial preparation can evaluate another function on the identical space
        EXPECT_LT(error(other_values[i], constant), 1e-12);
        // replacing coefficients cannot leave stale references to previous nodal caches in the preparation
        EXPECT_LT(error(after_update[i], constant), 1e-12);
    }
    // returned values own storage and survive subsequent changes to the function coefficients
    EXPECT_GT(error(sequential[0], after_update[0]), .1);
}
// LE multipoint evaluation reuses logarithms with all four preparation/evaluation policy combinations
TEST(GeometricFeEvaluation, LogEuclidean) { check_multipoint(LE {}); }
// AIRM multipoint evaluation retains the P1 interior solver and prepared edge restrictions
TEST(GeometricFeEvaluation, AffineInvariant) { check_multipoint(AIRM {}); }
/// @brief captures the expected exception text while allowing an unexpected type to fail the test
template <typename Exception, typename Call> std::string failure_message(Call call) {
    try {
        call();
    } catch (const Exception& failure) { return failure.what(); }
    return {};
}
// invalid locations and solver failures are rethrown deterministically after the parallel task group completes
TEST(GeometricFeEvaluation, ErrorsAndDiagnostics) {
    const auto mesh = domain();
    const GeometricFeSpace space(mesh, P1<1>, AIRM {});
    auto points = locations();
    points[1] = Vector<double, 2> {std::numeric_limits<double>::quiet_NaN(), 0.};
    points[2] = Vector<double, 2> {3., 3.};
    const auto sequential = failure_message<std::invalid_argument>([&] { space.prepare_evaluation(points); });
    const auto parallel =
      failure_message<std::invalid_argument>([&] { space.prepare_evaluation(points, execution_par); });
    // sequential preparation reports the first invalid location rather than returning partial data
    EXPECT_EQ(sequential, "P1 field spatial point must be finite");
    // parallel preparation rethrows the lowest-index error regardless of task completion order
    EXPECT_EQ(parallel, sequential);
    gfe::P1GeodesicLinearizationOptions options;
    options.mean.solver.max_iterations = 1;
    options.mean.solver.gradient_tolerance = 0;
    const GeometricFeFunction failing(space, coefficients(), options);
    const auto prepared = space.prepare_evaluation(locations(), execution_par);
    const auto sequential_solve = failure_message<std::runtime_error>([&] { prepared(failing); });
    const auto parallel_solve = failure_message<std::runtime_error>([&] { prepared(failing, execution_par); });
    // materialization preserves the P1 solver stop reason in the original diagnostic message
    EXPECT_NE(sequential_solve.find("stop_reason="), std::string::npos);
    // failed batch solves preserve the completed iteration count
    EXPECT_NE(sequential_solve.find("iterations="), std::string::npos);
    // failed batch solves preserve the measured stationarity residual
    EXPECT_NE(sequential_solve.find("residual="), std::string::npos);
    // parallel failure selection returns the same point's complete diagnostics as sequential evaluation
    EXPECT_EQ(parallel_solve, sequential_solve);
    Locations edge_points(2);
    edge_points[0] = Vector<double, 2> {.5, .5};
    edge_points[1] = Vector<double, 2> {0., 1.};
    const auto edges = space.prepare_evaluation(std::move(edge_points), execution_par);
    // exact edge and vertex paths do not fail even when iterative interior tolerances are unattainable
    EXPECT_NO_THROW(edges(failing, execution_par));
    // the executor remains usable after propagating worker exceptions from either phase
    EXPECT_NO_THROW(space.prepare_evaluation(locations(), execution_par));
}
// empty batches retain matrix shape and runtime identity checks reject functions from another space
TEST(GeometricFeEvaluation, EmptyAndSpaceContracts) {
    const auto mesh = domain();
    const GeometricFeSpace space(mesh, P1<1>, LE {}), other_space(mesh, P1<1>, LE {});
    const GeometricFeFunction function(space, coefficients()), other(other_space, coefficients());
    const auto empty = space.prepare_evaluation(Locations {}, execution_par);
    const auto values = empty(function, execution_par);
    // no location means no cell preparation and no fabricated output matrix
    EXPECT_TRUE(values.empty());
    // an empty SPD batch preserves its target order for downstream native consumers
    EXPECT_EQ(values.rows(), 2);
    // an empty preparation does not create any cell metadata
    EXPECT_EQ(empty.prepared_cells(), 0);
    // even empty evaluations require the same space instance, not merely identical mesh dimensions
    EXPECT_THROW(empty(other), std::invalid_argument);
    const auto prepared = space.prepare_evaluation(locations());
    // parallel evaluation applies the same space-identity validation before dispatch
    EXPECT_THROW(prepared(other, execution_par), std::invalid_argument);
    MatrixBatch<Matrix<double, Dynamic, Dynamic>> wrong_shape(2, 2, 2);
    // locations must be column vectors in the embedding dimension under the default policy
    EXPECT_THROW(space.prepare_evaluation(wrong_shape), std::invalid_argument);
    // parallel preparation rejects the same malformed location shape before starting workers
    EXPECT_THROW(space.prepare_evaluation(wrong_shape, execution_par), std::invalid_argument);
}
// preparation and evaluation can both run cooperatively from an existing executor worker
TEST(GeometricFeEvaluation, NestedExecution) {
    const auto mesh = domain();
    const GeometricFeSpace space(mesh, P1<1>, LE {});
    const GeometricFeFunction function(space, coefficients());
    auto pending = parallel_async([&] {
        const auto prepared = space.prepare_evaluation(locations(), execution_par);
        return prepared(function, execution_par);
    });
    const auto values = pending.get();
    parallel_join();
    const auto expected = function.eval_at(locations());
    // cooperative nested dispatch produces the same ordered first value as the sequential convenience
    EXPECT_LT(error(values[0], expected[0]), 1e-13);
}
// embedded location dimension and dynamic SPD order are independent in the prepared output batch
TEST(GeometricFeEvaluation, EmbeddedDynamicOrder) {
    Eigen::Matrix<double, 3, 3> vertices;
    vertices << 0, 0, 2, 1, 0, 2, 0, 1, 2;
    Eigen::Matrix<int, 1, 3> cells;
    cells << 2, 0, 1;
    const Triangulation<2, 3> mesh(vertices, cells, Eigen::Matrix<int, 3, 1>::Ones());
    const manifold::AffineInvariantSPDGeometry<double, Dynamic, Usage::BasePointMaps> geometry(3);
    const GeometricFeSpace space(mesh, P1<1>, geometry);
    MatrixBatch<SPDMatrix<double, Dynamic, Cache::Log>> coefficients(3, 3, 3);
    const GeometricFeFunction function(space, std::move(coefficients));
    MatrixBatch<Vector<double, 3>> locations(2);
    locations[0] = Vector<double, 3> {.2, .3, 2.};
    locations[1] = Vector<double, 3> {0., 1., 2.};
    const auto prepared = space.prepare_evaluation(locations, execution_par);
    const auto values = prepared(function, execution_par);
    // dynamic SPD order is taken from the target geometry rather than the number of points
    EXPECT_EQ(values.rows(), 3);
    // constant identity nodal data are reproduced on the embedded cell
    EXPECT_NEAR(values[0](0, 0), 1., 1e-13);
    locations[0] = Vector<double, 3> {.2, .3, 2.1};
    // off-surface coordinates cannot be accepted through the reference projection alone
    EXPECT_THROW(space.prepare_evaluation(locations, execution_par), std::invalid_argument);
}
}   // namespace
