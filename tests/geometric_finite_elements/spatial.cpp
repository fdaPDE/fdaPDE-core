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
using Point = SPDMatrix<double, 2>;
using Batch = MatrixBatch<SPDMatrix<double, 2, Cache::Log>>;
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
/// @brief recognizes mesh and field temporaries that could leave dangling spatial bindings
template <typename G>
concept TemporaryMeshAccepted =
  requires(G geometry, const Batch& batch) { geometry.interpolant(Triangulation<2, 2>::UnitSquare(2), batch); };
/// @brief recognizes deferred evaluation on an expiring field cache
template <typename Field>
concept TemporaryFieldAccepted = requires(Field&& field, const Eigen::Vector2d& x) { std::move(field)(x); };
/// @brief supplies a two-cell mesh whose local DOF order differs from global node order
Triangulation<2, 2> reordered_mesh() {
    Eigen::Matrix<double, 4, 2> vertices;
    vertices << 0, 0, 1, 0, 0, 1, 1, 1;
    Eigen::Matrix<int, 2, 3> cells;
    cells << 2, 0, 1, 1, 3, 2;
    return {vertices, cells, Eigen::Matrix<int, 4, 1>::Ones()};
}
/// @brief checks local ordering, lazy cell preparation, expression stability and derivative dispatch
void check_field(const auto& geometry) {
    const auto mesh = reordered_mesh();
    Batch batch(4);
    const auto first = data();
    for (int i = 0; i < 3; ++i) batch[i] = first[i];
    batch[3] = Matrix<double, 2, 2>({6., .3, .3, 2.});
    const auto field = geometry.interpolant(mesh, batch);
    // construction indexes space without preparing any cell's edge curves
    EXPECT_EQ(field.prepared_cells(), 0);
    const Eigen::Vector2d x(.2, .3), y(.8, .7);
    const auto expression = field(x);
    // one query prepares exactly its containing cell
    EXPECT_EQ(field.prepared_cells(), 1);
    const typename Triangulation<2, 2>::CellType cell(0, &mesh);
    const auto local = geometry.interpolant(cell, batch.select(std::array {2, 0, 1}));
    const Point expected(local(x));
    const Point later(field(y));
    // visiting the other cell extends the cache without eagerly preparing any unrelated entries
    EXPECT_EQ(field.prepared_cells(), 2);
    const Point value(expression);
    // an earlier deferred expression remains valid after cache growth and matches explicit local selection
    EXPECT_LT((Matrix<double, 2, 2>(value - expected).norm()), 1e-13);
    const Point vertex(field(Eigen::Vector2d(1., 1.)));
    // exact vertex evaluation identifies global node three through the second cell's local order
    EXPECT_EQ((Matrix<double, 2, 2>(vertex - batch[3]).norm()), 0.);
    const Point edge(field(Eigen::Vector2d(.5, .5)));
    const Point edge_expected(geometry.geodesic(batch[1], batch[2])(.5));
    // both incident cells restrict to the same prepared geodesic on their shared edge
    EXPECT_LT((Matrix<double, 2, 2>(edge - edge_expected).norm()), 1e-12);
    const auto derivative = field.linearization(x);
    // derivative setup reuses local ordering and the same converged value
    EXPECT_LT((Matrix<double, 2, 2>(derivative.result().value - expected).norm()), 1e-12);
    const auto direct_derivative = local.linearization(x);
    const std::array<double, 3> direction {-.3, .1, .2};
    const auto field_action = derivative.weight_jvp(direction);
    const auto local_action = direct_derivative.weight_jvp(direction);
    const auto& field_tangent = [&]() -> const auto& {
        if constexpr (requires { field_action.derivative; })
            return field_action.derivative;
        else
            return field_action;
    }();
    const auto& local_tangent = [&]() -> const auto& {
        if constexpr (requires { local_action.derivative; })
            return local_action.derivative;
        else
            return local_action;
    }();
    // a local weight direction keeps the same DOF order and borrowed data through mesh dispatch
    EXPECT_LT((Matrix<double, 2, 2>(field_tangent - local_tangent).norm()), 1e-13);
    // repeated result and derivative queries do not prepare duplicate cells
    EXPECT_EQ(field.prepared_cells(), 2);
    const auto diagnostics = field.result(y);
    // the diagnostic path must agree with deferred evaluation in the second cell
    EXPECT_LT((Matrix<double, 2, 2>(diagnostics.value - later).norm()), 1e-13);
    // an outside point must be rejected before an invalid cell index reaches the batch
    EXPECT_THROW(field(Eigen::Vector2d(2., .2)), std::invalid_argument);
    // a nonfinite point must not enter the spatial search
    EXPECT_THROW(field(Eigen::Vector2d(std::numeric_limits<double>::quiet_NaN(), .2)), std::invalid_argument);
    // a mesh temporary cannot be borrowed by an otherwise persistent field
    static_assert(!TemporaryMeshAccepted<std::remove_cvref_t<decltype(geometry)>>);
    // a field temporary cannot lend its cached interpolants to an escaping expression
    static_assert(!TemporaryFieldAccepted<std::remove_cvref_t<decltype(field)>>);
    const auto selected = geometry.interpolant(mesh, batch.select(std::array {0, 1, 2, 3}));
    const Point selected_value(selected(x));
    // a temporary selection is retained in the field, preserving its global node interpretation
    EXPECT_LT((Matrix<double, 2, 2>(selected_value - expected).norm()), 1e-13);
}
// LE uses the shared mesh wrapper while preserving the local logarithmic interpolation contract
TEST(P1FieldInterpolation, LogEuclidean) { check_field(LE {}); }
// AIRM uses the same mesh wrapper and local Karcher and derivative implementations
TEST(P1FieldInterpolation, AffineInvariant) { check_field(AIRM {}); }
// incompatible global data and empty meshes fail before indexing or preparing local interpolation
TEST(P1FieldInterpolation, InvalidBindings) {
    const auto mesh = reordered_mesh();
    const auto batch = data();
    // a three-value batch cannot represent four global vertices
    EXPECT_THROW(LE {}.interpolant(mesh, batch), std::invalid_argument);
    const Triangulation<2, 2> empty;
    // an empty mesh has no searchable cell and is rejected before tree construction
    EXPECT_THROW(LE {}.interpolant(empty, batch), std::invalid_argument);
}
// independent fields share an uncached mesh safely during simultaneous first visits and evaluations
TEST(P1FieldInterpolation, ConcurrentColdCells) {
    const auto mesh = Triangulation<2, 2>::UnitSquare(5);
    Batch batch(mesh.n_nodes());
    const Point expected(Matrix<double, 2, 2>({2., .5, .5, 3.}));
    for (int i = 0; i < mesh.n_nodes(); ++i) batch[i] = expected;
    const auto& scratch = mesh.cell(0);
    const auto saved_coordinates = scratch.nodes();
    const auto field = AIRM {}.interpolant(mesh, batch);
    const auto other = AIRM {}.interpolant(mesh, batch);
    std::vector<std::future<Point>> pending;
    for (int i = 0; i < mesh.n_cells(); ++i) {
        const Triangulation<2, 2>::CellType cell(i, &mesh);
        const Eigen::Vector2d x = cell.nodes().rowwise().mean();
        pending.push_back(parallel_async([&, x] {
            const Point value(field(x));
            const Point second(other(x));
            return Point((value + second) / 2.);
        }));
    }
    for (auto& future : pending) {
        const Point actual = future.get();
        // every independently located centroid must reproduce the constant nodal field
        EXPECT_LT((Matrix<double, 2, 2>(actual - expected).norm()), 1e-12);
    }
    parallel_join();
    // neither index construction nor queries may overwrite the mesh's externally held scratch cell
    EXPECT_EQ((scratch.nodes() - saved_coordinates).norm(), 0.);
    // distinct cell centroids prepare every cell exactly once even under concurrent cold access
    EXPECT_EQ(field.prepared_cells(), static_cast<std::size_t>(mesh.n_cells()));
}
// flat embedding axes remain searchable and off-plane points are rejected by exact simplex containment
TEST(P1FieldInterpolation, EmbeddedMesh) {
    Eigen::Matrix<double, 3, 3> vertices;
    vertices << 0, 0, 2, 1, 0, 2, 0, 1, 2;
    Eigen::Matrix<int, 1, 3> cells;
    cells << 2, 0, 1;
    const Triangulation<2, 3> mesh(vertices, cells, Eigen::Matrix<int, 3, 1>::Ones());
    const auto batch = data();
    const auto field = LE {}.interpolant(mesh, batch);
    const Point actual(field(Eigen::Vector3d(0, 0, 2)));
    // zero spatial extent in z must not turn tree normalization into a NaN query
    EXPECT_EQ((Matrix<double, 2, 2>(actual - batch[0]).norm()), 0.);
    const TreeSearch<Triangulation<2, 3>> index(&mesh);
    // the sibling all-cell locator shares the finite normalization on a flat embedding axis
    EXPECT_EQ(index.all_locate(Eigen::Vector3d(.2, .3, 2)).size(), 1);
    Eigen::MatrixXd queries(2, 3);
    queries << .2, .3, 2, .2, .3, 2.1;
    const auto located = index.locate(queries);
    // batched lookup resolves the on-surface point to its only containing cell
    EXPECT_EQ(located[0], 0);
    // batched lookup preserves the outside sentinel for an off-surface point
    EXPECT_EQ(located[1], -1);
    // a point inside the planar projection but off the surface cannot be interpolated
    EXPECT_THROW(field(Eigen::Vector3d(.2, .3, 2.1)), std::invalid_argument);
}
// the common mesh path also supports segments and tetrahedra without geometry-specific wrappers
TEST(P1FieldInterpolation, IntervalAndVolume) {
    const auto mesh = Triangulation<1, 1>::UnitInterval(2);
    Batch batch(2);
    const auto source = data();
    batch[0] = source[0];
    batch[1] = source[1];
    const auto field = LE {}.interpolant(mesh, batch);
    Eigen::Matrix<double, 1, 1> x;
    x << .4;
    const Point expected(LE {}.geodesic(batch[0], batch[1])(.4));
    const Point interval_value(field(x));
    // a one-cell interval reduces exactly to its two-node geodesic
    EXPECT_LT((Matrix<double, 2, 2>(interval_value - expected).norm()), 1e-13);
    const auto volume = Triangulation<3, 3>::UnitCube(2);
    Batch values(volume.n_nodes());
    for (int i = 0; i < volume.n_nodes(); ++i) values[i] = source[0];
    const auto volume_field = AIRM {}.interpolant(volume, values);
    const Point actual(volume_field(Eigen::Vector3d(.2, .3, .4)));
    // tetrahedral barycentric interpolation reproduces constant data just as the triangle path does
    EXPECT_LT((Matrix<double, 2, 2>(actual - source[0]).norm()), 1e-12);
}
}   // namespace
