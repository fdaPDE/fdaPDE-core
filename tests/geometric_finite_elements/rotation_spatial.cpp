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
using Geometry = manifold::SOGeometry<double, 3>;
using Point = Geometry::Point;
using Batch = MatrixBatch<RotationMatrix<double, 3, 3, RotationCache::Union<RotationCache::Log, RotationCache::Schur>>>;
using Locations = MatrixBatch<Vector<double, 2>>;
/// @brief supplies two triangles with different local vertex orders
Triangulation<2, 2> mesh() {
    Eigen::Matrix<double, 4, 2> points;
    points << 0, 0, 1, 0, 0, 1, 1, 1;
    Eigen::Matrix<int, 2, 3> cells;
    cells << 2, 0, 1, 1, 3, 2;
    return {points, cells, Eigen::Matrix<int, 4, 1>::Ones()};
}
/// @brief supplies noncommuting rotations whose relative logarithms are regular
Batch nodes() {
    Batch result(4);
    const Geometry g;
    const auto identity = Point::Identity();
    for (int i = 0; i < 4; ++i) {
        Geometry::Tangent t;
        t(0, 1) = .2 * (i + 1);
        t(0, 2) = .15 * std::cos(i);
        t(1, 2) = .1 * std::sin(i);
        result[i] = g.exponential(identity, t);
    }
    return result;
}
/// @brief measures coefficient errors independently of the relative rotation logarithm
double error(const auto& a, const auto& b) { return Matrix<double, 3, 3>(a - b).norm(); }
// scalar and parallel evaluations share native rotation data, local caches and reusable spatial preparation
TEST(SOSpatialInterpolation, FunctionsPoliciesAndCacheReplacement) {
    const auto domain = mesh();
    const Geometry g;
    const GeometricFeSpace space(domain, P1<1>, g);
    auto batch = nodes();
    GeometricFeFunction function(space, batch);
    const auto field = g.interpolant(domain, batch);
    Locations locations(5);
    locations[0] = Vector<double, 2> {.8, .7};
    locations[1] = Vector<double, 2> {.2, .3};
    locations[2] = Vector<double, 2> {.5, .5};
    locations[3] = Vector<double, 2> {0, 1};
    locations[4] = Vector<double, 2> {.2, .3};
    const auto prepared = space.prepare_evaluation(locations);
    const auto parallel = space.prepare_evaluation(locations, execution_par);
    // location preparation must not allocate coefficient-dependent edge or mean caches
    EXPECT_EQ(function.prepared_cells(), 0);
    const auto seq = prepared(function), par = parallel(function, execution_par);
    const auto mixed1 = prepared(function, execution_par), mixed2 = parallel(function);
    for (int i = 0; i < 5; ++i) {
        Eigen::Vector2d x;
        x << locations[i](0, 0), locations[i](1, 0);
        const Point scalar(function(x)), direct(field(x));
        const Matrix<double, 3, 3> dense(function(x));
        // native dense materialization preserves rotation coefficients without invoking SPD certification
        EXPECT_LT(error(dense, scalar), 2e-12);
        // the legacy mesh entry point shares the same local P1 value implementation
        EXPECT_LT(error(direct, scalar), 2e-12);
        // parallel preparation and evaluation preserve point order and numerical values
        EXPECT_LT(error(seq[i], par[i]), 2e-12);
        // parallel evaluation can consume a sequential spatial plan
        EXPECT_LT(error(seq[i], mixed1[i]), 2e-12);
        // sequential evaluation can consume a parallel spatial plan
        EXPECT_LT(error(seq[i], mixed2[i]), 2e-12);
        // scalar and prepared paths select the same cell-local rotations
        EXPECT_LT(error(seq[i], scalar), 2e-12);
    }
    // repeated points share the two visited coefficient-dependent cells
    EXPECT_EQ(function.prepared_cells(), 2);
    const auto old = Point(seq[1]);
    Geometry::Tangent update;
    update(0, 1) = .07;
    for (int i = 0; i < 4; ++i) batch[i] = g.exponential(batch[i], update);
    function.set_coeff(batch);
    // replacing coefficients invalidates all prepared edge curves
    EXPECT_EQ(function.prepared_cells(), 0);
    const auto changed = prepared(function, execution_par);
    // the same spatial plan reads the replacement coefficient generation
    EXPECT_GT(error(changed[1], old), .01);
    // earlier outputs retain independent rotation storage after coefficient replacement
    EXPECT_LT(error(seq[1], old), 2e-12);
    const Eigen::Vector2d x(.2, .3);
    const auto lin = function.linearization(x);
    const std::array<double, 3> weights_direction {-.4, .1, .3};
    // the public function exposes the shared implicit rotation weight differential
    EXPECT_TRUE(lin.weight_jvp(weights_direction).converged());
    // empty location batches retain the target matrix shape and require no point work
    EXPECT_EQ(function.eval_at(Locations(0)).size(), 0);
}
// an unused ambiguous edge does not prevent exact vertex or other regular-edge interpolation
TEST(SOSpatialInterpolation, AmbiguousEdgesAndParallelErrors) {
    const auto domain = mesh();
    const Geometry g;
    const GeometricFeSpace space(domain, P1<1>, g);
    Batch batch(4);
    const auto identity = Point::Identity();
    batch[0] = identity;
    batch[1] = Matrix<double, 3, 3>({-1, 0, 0, 0, -1, 0, 0, 0, 1});
    Geometry::Tangent t;
    t(0, 1) = .4;
    batch[2] = g.exponential(identity, t);
    batch[3] = identity;
    GeometricFeFunction function(space, batch);
    const Point vertex(function(Eigen::Vector2d(0, 0)));
    // vertex reproduction does not need a logarithm for the opposite inactive edge
    EXPECT_LT(error(vertex, identity), 2e-12);
    const Point regular_edge(function(Eigen::Vector2d(0, .5)));
    // the active regular edge follows its prepared half-angle rotation
    EXPECT_LT(error(regular_edge, g.exponential(identity, t, .5)), 2e-12);
    // an active ambiguous edge requires a user-selected branch and is rejected by P1
    EXPECT_THROW((Point(function(Eigen::Vector2d(.5, 0)))), std::domain_error);
    Locations locations(2);
    locations[0] = Vector<double, 2> {0, 0};
    locations[1] = Vector<double, 2> {.5, 0};
    const auto plan = space.prepare_evaluation(locations);
    // sequential batch materialization propagates the same branch failure
    EXPECT_THROW(plan(function), std::domain_error);
    // parallel materialization rethrows the point failure after joining its workers
    EXPECT_THROW(plan(function, execution_par), std::domain_error);
    // a failed evaluation leaves all owned nodal rotation data intact
    EXPECT_LT(error(function.coeff()[1], batch[1]), 2e-12);
}
}   // namespace
