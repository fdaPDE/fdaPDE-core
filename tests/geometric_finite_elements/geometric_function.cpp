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
#include <fdaPDE/finite_elements.h>
#include <fdaPDE/geometric_finite_elements.h>
#include <gtest/gtest.h>

namespace {
using namespace fdapde;
using Point = SPDMatrix<double, 2, 2>;
using Batch = MatrixBatch<SPDMatrix<double, 2, 2, Cache::Log>>;
using LE = manifold::LogEuclideanSPDGeometry<double, 2, Usage::InterpolationNodes>;
using AIRM = manifold::AffineInvariantSPDGeometry<double, 2, Usage::BasePointMaps>;
/// @brief supplies noncommuting global coefficients for a square split into reordered cells
Batch values() {
    Batch batch(4);
    batch[0] = Matrix<double, 2, 2>({2., .5, .5, 3.});
    batch[1] = Matrix<double, 2, 2>({5., -1., -1., 4.});
    batch[2] = Matrix<double, 2, 2>({1.5, .2, .2, 2.5});
    batch[3] = Matrix<double, 2, 2>({6., .3, .3, 2.});
    return batch;
}
/// @brief supplies connectivity whose first local vertex differs from the global ordering
Triangulation<2, 2> mesh() {
    Eigen::Matrix<double, 4, 2> vertices;
    vertices << 0, 0, 1, 0, 0, 1, 1, 1;
    Eigen::Matrix<int, 2, 3> cells;
    cells << 2, 0, 1, 1, 3, 2;
    return {vertices, cells, Eigen::Matrix<int, 4, 1>::Ones()};
}
/// @brief compares complete symmetric coefficients without retaining temporary owners
double error(const auto& lhs, const auto& rhs) { return Matrix<double, 2, 2>(lhs - rhs).norm(); }
/// @brief exercises scalar space delegation, coefficient ownership, derivatives and replacement
void check_function(const auto& geometry) {
    const auto domain = mesh();
    const GeometricFeSpace space(domain, P1<1>, geometry);
    const FeSpace scalar_space(domain, P1<1>);
    auto source = values();
    GeometricFeFunction function(space, source);
    // the geometric function borrows the exact space rather than creating another DOF enumeration
    EXPECT_EQ(&function.function_space(), &space);
    // the space retains the original spatial mesh binding
    EXPECT_EQ(&space.triangulation(), &domain);
    // each scalar nodal degree of freedom stores one full matrix
    EXPECT_EQ(space.n_dofs(), scalar_space.n_dofs());
    // geometry order describes each matrix and does not multiply scalar DOFs
    EXPECT_EQ(space.geometry().order(), 2);
    // constructing a function copies an lvalue batch into independent coefficient storage
    EXPECT_NE(function.coeff().coefficients().data(), source.coefficients().data());
    // construction leaves all coefficient-dependent cell preparation deferred
    EXPECT_EQ(function.prepared_cells(), 0);
    const Eigen::Vector2d x(.2, .3);
    const auto cell = space.dof_handler().cell(0);
    const auto ids = cell.dofs();
    const Matrix<double, 2, 1> reference {.5, .2};
    std::array<double, 3> weights;
    for (int i = 0; i < 3; ++i) {
        // local DOF order is the scalar handler's numbering, including the reordered first vertex
        EXPECT_EQ(ids[i], scalar_space.dof_handler().dofs()(0, i));
        weights[i] = space.eval_shape_value(i, reference);
        // geometric weights are evaluated through the existing scalar reference basis
        EXPECT_EQ(weights[i], scalar_space.eval_shape_value(i, reference));
    }
    const auto expected = gfe::p1_geodesic_value(geometry, source.select(ids), weights);
    const Point actual(function(x));
    // native materialization agrees with the local geometric solver using scalar DOFs and shape weights
    EXPECT_LT(error(actual, expected.value), 1e-12);
    const auto expression = function(x);
    const Point second(function(Eigen::Vector2d(.8, .7)));
    // visiting another cell must not invalidate a deferred expression already borrowing the first
    EXPECT_LT(error(Point(expression), actual), 1e-13);
    const auto linearization = function.linearization(x);
    const auto local = geometry.interpolant(cell, source.select(ids));
    const auto local_linearization = local.linearization(x);
    const std::array<double, 3> direction {-.3, .1, .2};
    const auto action = linearization.weight_jvp(direction);
    const auto expected_action = local_linearization.weight_jvp(direction);
    if constexpr (requires { action.derivative; }) {
        // AIRM derivative comparison is valid only after both local tangent solves converge
        ASSERT_TRUE(action.converged() && expected_action.converged());
        // derivative directions retain the located cell's scalar DOF order
        EXPECT_LT(error(action.derivative, expected_action.derivative), 1e-12);
    } else {
        // LE derivative directions retain the located cell's scalar DOF order
        EXPECT_LT(error(action, expected_action), 1e-12);
    }
    source[0] = Matrix<double, 2, 2>({9., 0., 0., 9.});
    // external source mutation cannot change the function's owned coefficients or its prepared cells
    EXPECT_LT(error(Point(function(x)), actual), 1e-13);
    auto replacement = source;
    const Point constant(Matrix<double, 2, 2>({4., .2, .2, 3.}));
    for (std::size_t i = 0; i < replacement.size(); ++i) replacement[i] = constant;
    const auto* transferred = replacement.coefficients().data();
    function.set_coeff(std::move(replacement));
    // replacement transfers its contiguous buffers without copying the batch
    EXPECT_EQ(function.coeff().coefficients().data(), transferred);
    // replacing coefficients drops every old prepared cell, including its old edge curves
    EXPECT_EQ(function.prepared_cells(), 0);
    // the next edge query must use the new coefficients rather than the previous prepared geodesic
    EXPECT_LT(error(Point(function(Eigen::Vector2d(.5, .5))), constant), 1e-12);
    // the new interior diagnostic value must also reproduce the replacement constant field
    EXPECT_LT(error(function.result(x).value, constant), 1e-12);
}
// the LE finite element facade uses scalar bases and owns independently replaceable matrix coefficients
TEST(GeometricFeFunction, LogEuclidean) { check_function(LE {}); }
// the AIRM facade shares the P1 solver and preserves derivative ordering across coefficient replacement
TEST(GeometricFeFunction, AffineInvariant) { check_function(AIRM {}); }
// move construction transfers storage and invalid replacements leave both coefficients and cache usable
TEST(GeometricFeFunction, OwnershipAndInvalidReplacement) {
    const auto domain = mesh();
    const GeometricFeSpace space(domain, P1<1>, LE {});
    auto batch = values();
    const auto* transferred = batch.coefficients().data();
    GeometricFeFunction function(space, std::move(batch));
    // an rvalue constructor retains the original contiguous buffer rather than deep-copying it
    EXPECT_EQ(function.coeff().coefficients().data(), transferred);
    // a successfully transferred source batch has no remaining matrices
    EXPECT_TRUE(batch.empty());
    const Eigen::Vector2d x(.2, .3);
    const auto expression = function(x);
    const Point before(expression);
    Batch wrong_count(3);
    // invalid nodal count is rejected before discarding existing prepared cells
    EXPECT_THROW(function.set_coeff(wrong_count), std::invalid_argument);
    // a failed update preserves the existing cell cache
    EXPECT_EQ(function.prepared_cells(), 1);
    // a failed update also preserves borrowed expressions obtained before the call
    EXPECT_LT(error(Point(expression), before), 1e-13);
    Batch replacement = values();
    function.set_coeff(replacement);
    // an lvalue replacement is copied so subsequent caller mutation cannot affect the function
    EXPECT_NE(function.coeff().coefficients().data(), replacement.coefficients().data());
    using DynamicBatch = MatrixBatch<SPDMatrix<double, Dynamic, Dynamic>>;
    DynamicBatch dynamic_values(4, 2, 2);
    GeometricFeFunction dynamic_function(space, std::move(dynamic_values));
    DynamicBatch wrong_shape(4, 3, 3);
    // uniform batch shape must match the target geometry before any cache invalidation
    EXPECT_THROW(dynamic_function.set_coeff(wrong_shape), std::invalid_argument);
    // construction applies the same public shape validation before indexing or cell preparation
    EXPECT_THROW(GeometricFeFunction(space, wrong_shape), std::invalid_argument);
}
/// @brief checks supported finite element tags through the public deduction interface
template <typename Fe>
concept AcceptedElement =
  requires(const Triangulation<2, 2>& mesh, Fe element) { GeometricFeSpace(mesh, element, LE {}); };
/// @brief detects a temporary spatial mesh that would leave the scalar space dangling
template <typename Mesh>
concept TemporarySpaceMesh = requires { GeometricFeSpace(Mesh {}, P1<1>, LE {}); };
/// @brief detects a temporary space that would leave a function's space binding dangling
template <typename Space>
concept TemporaryFunctionSpace =
  requires(const Triangulation<2, 2>& domain, Batch batch) { GeometricFeFunction(Space(domain, P1<1>, LE {}), batch); };
// unsupported degrees, vector bases and expiring spatial bindings fail at compile time
TEST(GeometricFeFunction, CompileTimeContracts) {
    // the agreed scalar P1 descriptor remains constructible
    static_assert(AcceptedElement<decltype(P1<1>)>);
    // quadratic shape functions are not silently routed through the P1 geometric solver
    static_assert(!AcceptedElement<decltype(P2<1>)>);
    // matrix dimensions come from geometry, so scalar vector components are rejected
    static_assert(!AcceptedElement<decltype(P1<2>)>);
    // the scalar space cannot borrow a temporary triangulation
    static_assert(!TemporarySpaceMesh<Triangulation<2, 2>>);
    using Space = GeometricFeSpace<const Triangulation<2, 2>, FeP<1, 1>, LE>;
    // a function cannot borrow a temporary space
    static_assert(!TemporaryFunctionSpace<Space>);
    // explicit const template arguments cannot circumvent the mesh lifetime constraint
    static_assert(!std::is_constructible_v<Space, Triangulation<2, 2>&&, FeP<1, 1>, LE>);
    using ConstFunction = GeometricFeFunction<const Space, typename Batch::MatrixType>;
    // explicit const template arguments cannot circumvent the function's space lifetime constraint
    static_assert(!std::is_constructible_v<ConstFunction, Space&&, Batch>);
    using Function = GeometricFeFunction<Space, typename Batch::MatrixType>;
    // the public coefficient accessor cannot bypass invalidation through mutable matrix views
    static_assert(std::same_as<decltype(std::declval<Function&>().coeff()), const Batch&>);
}
// reference shape evaluation uses local dimension on embedded triangles and still rejects off-surface points
TEST(GeometricFeFunction, EmbeddedReferenceCoordinates) {
    Eigen::Matrix<double, 3, 3> vertices;
    vertices << 0, 0, 2, 1, 0, 2, 0, 1, 2;
    Eigen::Matrix<int, 1, 3> cells;
    cells << 2, 0, 1;
    const Triangulation<2, 3> domain(vertices, cells, Eigen::Matrix<int, 3, 1>::Ones());
    const GeometricFeSpace space(domain, P1<1>, LE {});
    Batch batch(3);
    for (int i = 0; i < 3; ++i) batch[i] = Matrix<double, 2, 2>({2. + i, 0., 0., 2. + i});
    const GeometricFeFunction function(space, batch);
    const auto local = LE {}.interpolant(domain.cell(0), batch.select(std::array {2, 0, 1}));
    const Eigen::Vector3d x(.2, .3, 2.);
    // local two-dimensional reference weights give the same value as the embedded simplex engine
    EXPECT_LT(error(Point(function(x)), Point(local(x))), 1e-13);
    // off-plane points cannot reach shape evaluation through spatial projection alone
    EXPECT_THROW(function(Eigen::Vector3d(.2, .3, 2.1)), std::invalid_argument);
}
// immutable functions support concurrent lazy preparation and may be updated after workers finish
TEST(GeometricFeFunction, ConcurrentEvaluationAndUpdate) {
    const auto domain = mesh();
    const GeometricFeSpace space(domain, P1<1>, AIRM {});
    GeometricFeFunction function(space, values());
    auto first = parallel_async([&] { return Point(function(Eigen::Vector2d(.2, .3))); });
    auto second = parallel_async([&] { return Point(function(Eigen::Vector2d(.8, .7))); });
    const Point first_value = first.get(), second_value = second.get();
    parallel_join();
    // concurrent first visits prepare the two cells without losing either cache entry
    EXPECT_EQ(function.prepared_cells(), 2);
    // first-cell evaluation agrees with the subsequent sequential diagnostic result
    EXPECT_LT(error(first_value, function.result(Eigen::Vector2d(.2, .3)).value), 1e-12);
    // second-cell evaluation agrees with the subsequent sequential diagnostic result
    EXPECT_LT(error(second_value, function.result(Eigen::Vector2d(.8, .7)).value), 1e-12);
    function.set_coeff(values());
    // once workers have joined, replacement safely invalidates all prepared cells
    EXPECT_EQ(function.prepared_cells(), 0);
}
}   // namespace
