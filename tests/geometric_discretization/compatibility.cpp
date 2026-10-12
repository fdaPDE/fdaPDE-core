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

#include <fdaPDE/geometric_finite_elements.h>
#include <fdaPDE/geometric_splines.h>
#include <gtest/gtest.h>

namespace {
using namespace fdapde;
using CachePolicy = Cache::Union<Cache::Spectral, Cache::Log, Cache::LogDividedDifferences>;
using SPD = SPDMatrix<double, 3, CachePolicy>;
using LE = manifold::LogEuclideanGeometry<SPD>;
using LC = manifold::LogCholeskyGeometry<SPD>;
using AI = manifold::AffineInvariantGeometry<SPD>;
using BW = manifold::BuresWassersteinGeometry<SPD>;
using CE = manifold::CheegerLogEuclideanGeometry<SPD>;
using SO = manifold::SOGeometry<double, 3, RotationUsage::IdentityLog>;

/// @brief compares complete matrix coefficients without retaining borrowed arithmetic expressions
double error(const auto& a, const auto& b) { return Matrix<double, Dynamic, Dynamic>(a - b).norm(); }
/// @brief resolves intrinsic means and derivatives accurately enough for centered spatial differences
gfe::P1GeodesicLinearizationOptions options() {
    gfe::P1GeodesicLinearizationOptions result;
    result.mean.solver.gradient_tolerance = 1e-11;
    result.linear_solve.residual_tolerance = 1e-11;
    return result;
}
/// @brief creates close noncommuting controls while preserving the geometry's exact native point type
template <typename Geometry> MatrixBatch<typename Geometry::Point> controls(const Geometry& geometry, int count) {
    using Point = typename Geometry::Point;
    MatrixBatch<Point> result(count, geometry.order(), geometry.order());
    for (int i = 0; i < count; ++i) {
        if constexpr (SPDLike<Point>) {
            SymmetricMatrix<double, Point::Rows> logarithm;
            for (int r = 0; r < geometry.order(); ++r) {
                logarithm(r, r) = .15 * r + .03 * i;
                for (int c = 0; c < r; ++c) logarithm(r, c) = .02 * std::sin(.7 * i + r + 2 * c);
            }
            result[i] = matrix_exp(logarithm);
        } else {
            typename Geometry::Tangent direction;
            for (int r = 0; r < geometry.order(); ++r)
                for (int c = r + 1; c < geometry.order(); ++c) direction(r, c) = .03 * std::sin(.7 * i + r + 2 * c);
            result[i] = geometry.exponential(Point::Identity(geometry.order()), direction);
        }
    }
    return result;
}
/// @brief materializes the shared weighted kernel independently of any spatial basis or locator
template <typename Geometry, typename Values, typename Weights>
auto weighted_value(const Geometry& geometry, const Values& values, const Weights& weights) {
    if constexpr (gfe::internals::is_flat_spd_geometry<Geometry>)
        return gfe::p1_geodesic_value(geometry, values, weights);
    else
        return gfe::p1_geodesic_value(geometry, values, weights, options().mean);
}
/// @brief extracts flat or implicit weight derivatives after checking iterative convergence
auto derivative(const auto& result) {
    if constexpr (requires { result.derivative; }) {
        // implicit derivative actions must certify the linear solve before their tangent is used
        EXPECT_TRUE(result.converged());
        return result.derivative;
    } else {
        return result;
    }
}
/// @brief expresses a geometric tangent as an ambient matrix velocity for independent finite differences
template <typename Geometry, typename Point, typename Tangent>
auto ambient(const Geometry& geometry, const Point& point, const Tangent& direction) {
    if constexpr (RotationLike<Point>)
        return Matrix<double, Dynamic, Dynamic>(geometry.to_ambient(point, direction));
    else
        return Matrix<double, Dynamic, Dynamic>(direction);
}
/// @brief supplies intervals, branching networks, planar triangles, folded surfaces and tetrahedra
template <int D, int E> Triangulation<D, E> mesh(int flags = 0) {
    if constexpr (D == 1 && E == 1) {
        Eigen::VectorXd knots(4);
        knots << -.4, .1, .8, 1.7;
        return Triangulation<1, 1>(knots, flags);
    } else if constexpr (D == 1) {
        Eigen::Matrix<double, 4, 2> nodes;
        nodes << 0., 0., 1., 0., .3, 1., -.5, .8;
        Eigen::Matrix<int, 3, 2> cells;
        cells << 0, 1, 0, 2, 0, 3;
        return Triangulation<1, 2>(nodes, cells, Eigen::Vector4i::Ones(), flags);
    } else if constexpr (D == 2) {
        Eigen::Matrix<double, 4, E> nodes;
        nodes.setZero();
        nodes.template leftCols<2>() << 0., 0., 1., 0., 0., 1., 1., 1.;
        if constexpr (E == 3) nodes.col(2) << .2, .6, .4, 1.1;
        Eigen::Matrix<int, 2, 3> cells;
        cells << 0, 1, 2, 1, 3, 2;
        return Triangulation<D, E>(nodes, cells, Eigen::Vector4i::Ones(), flags);
    } else {
        Eigen::Matrix<double, 5, 3> nodes;
        nodes << 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 1., 1., 1.;
        Eigen::Matrix<int, 2, 4> cells;
        cells << 0, 1, 2, 3, 1, 2, 3, 4;
        return Triangulation<3, 3>(nodes, cells, Eigen::Matrix<int, 5, 1>::Ones(), flags);
    }
}
/// @brief checks geometric P1 values, spatial derivatives, endpoints, parallel plans and updates on one domain
template <int D, int E, typename Geometry> void check_fem(const Geometry& geometry, int flags) {
    SCOPED_TRACE("Triangulation<" + std::to_string(D) + "," + std::to_string(E) + ">");
    using Point = typename Geometry::Point;
    using Location = Vector<double, E>;
    const auto domain = mesh<D, E>(flags);
    const FeSpace scalar(domain, P1<1>);
    const GeometricSpace space(scalar, geometry);
    const auto nodes = controls(geometry, space.n_dofs());
    GeometricFunction field(space, nodes, options());
    MatrixBatch<Location> locations(domain.n_cells());
    for (int id = 0; id < domain.n_cells(); ++id) {
        Eigen::Matrix<double, E, 1> x = Eigen::Matrix<double, E, 1>::Zero();
        std::array<double, D + 1> weights;
        std::array<int, D + 1> dofs;
        const double total = (D + 1.) * (D + 2.) / 2.;
        for (int j = 0; j <= D; ++j) {
            weights[j] = (j + 1.) / total;
            dofs[j] = scalar.dof_handler().dofs()(id, j);
            x += weights[j] * domain.node(domain.cells()(id, j));
        }
        for (int d = 0; d < E; ++d) locations[id](d, 0) = x[d];
        const auto expected = weighted_value(geometry, nodes.select(dofs), weights);
        // the independent weighted mean must converge for the chosen close noncommuting controls
        ASSERT_TRUE(expected.converged());
        const auto actual = field.result(x);
        // single-point diagnostics certify the same mean on every supported simplex dimension
        ASSERT_TRUE(actual.converged());
        // prescribed barycentric weights reproduce the low-level mean in scalar DOF order
        EXPECT_LT(error(actual.value, expected.value), 2e-9);
        // deferred expressions retain the exact point representation and agree with diagnostics
        EXPECT_LT(error(Point(field(x)), actual.value), 2e-9);
        std::array<double, D + 1> dw {};
        dw[0] = -1.;
        dw[1] = 1.;
        const auto lin = field.linearization(x);
        const auto tangent = derivative(lin.weight_jvp(dw));
        constexpr double h = 2e-4;
        const Eigen::Matrix<double, E, 1> dx = domain.node(domain.cells()(id, 1)) - domain.node(domain.cells()(id, 0));
        const Point plus(field(Eigen::Matrix<double, E, 1>(x + h * dx)));
        const Point minus(field(Eigen::Matrix<double, E, 1>(x - h * dx)));
        // moving along a simplex edge differentiates the known barycentric direction in physical coordinates
        EXPECT_LT(error(ambient(geometry, actual.value, tangent), (plus - minus) / (2. * h)), 2e-6);
    }
    const auto Psi_seq = space.prepare_evaluation(locations);
    const auto Psi_par = space.prepare_evaluation(locations, execution_par);
    const auto seq = Psi_seq(field);
    const GeometricFunction cold_field(space, controls(geometry, space.n_dofs()), options());
    const auto par = Psi_par(cold_field, execution_par);
    const auto prepared_par = Psi_par(field);
    const auto mixed = Psi_seq(field, execution_par);
    const auto once = field.eval_at(locations, execution_par);
    // all evaluations return owning batches with the exact point and cache policy chosen by the geometry
    static_assert(std::same_as<std::remove_cvref_t<decltype(seq)>, MatrixBatch<Point>>);
    for (std::size_t i = 0; i < seq.size(); ++i) {
        // cold parallel cell preparation preserves the sequential field values and supplied point order
        EXPECT_LT(error(seq[i], par[i]), 2e-9);
        // a parallel preparation can independently choose sequential materialization
        EXPECT_LT(error(seq[i], prepared_par[i]), 2e-9);
        // a sequential preparation can independently choose parallel materialization
        EXPECT_LT(error(seq[i], mixed[i]), 2e-9);
        // one-off parallel evaluation shares the same source support and coefficient ordering
        EXPECT_LT(error(seq[i], once[i]), 2e-9);
    }
    for (int id = 0; id < domain.n_nodes(); ++id) {
        const Point value(field(domain.node(id)));
        // simplex vertices recover the corresponding geometric coefficient, including network junctions
        EXPECT_LT(error(value, nodes[id]), 2e-9);
    }
    auto replacement = controls(geometry, space.n_dofs());
    const Point constant(nodes[0]);
    for (std::size_t i = 0; i < replacement.size(); ++i) replacement[i] = constant;
    field.set_coeff(std::move(replacement));
    const auto updated = Psi_par(field, execution_par);
    for (std::size_t i = 0; i < updated.size(); ++i) {
        // retained spatial plans use new coefficients and all geometries reproduce a constant field
        EXPECT_LT(error(updated[i], constant), 2e-9);
    }
    if constexpr (D < E) {
        Eigen::Matrix<double, E, 1> off_mesh = domain.node(0);
        off_mesh[E - 1] += 10.;
        // embedded domains reject off-network and off-surface points rather than interpolating their projection
        EXPECT_THROW(field.result(off_mesh), std::invalid_argument);
    }
}
/// @brief checks spline means and spatial derivatives against the complete physical scalar basis
template <typename Geometry> void check_splines(const Geometry& geometry, int flags) {
    using Point = typename Geometry::Point;
    const auto domain = mesh<1, 1>(flags);
    for (int degree : {0, 1, 3, 5}) {
        SCOPED_TRACE("spline degree " + std::to_string(degree));
        const BsSpace scalar(domain, degree);
        const GeometricSpace space(scalar, geometry);
        const auto nodes = controls(geometry, space.n_dofs());
        GeometricFunction field(space, nodes, options());
        MatrixBatch<Vector<double, 1>> locations(5);
        const std::array<double, 5> coordinates {-.4, -.17, .1, 1.2, 1.7};
        for (std::size_t k = 0; k < coordinates.size(); ++k) locations[k] = Vector<double, 1> {coordinates[k]};
        const auto Psi = space.prepare_evaluation(locations, execution_par);
        const auto values = Psi(field, execution_par);
        for (std::size_t k = 0; k < coordinates.size(); ++k) {
            const Eigen::Matrix<double, 1, 1> x = Eigen::Matrix<double, 1, 1>::Constant(coordinates[k]);
            std::vector<double> weights(nodes.size());
            for (std::size_t i = 0; i < nodes.size(); ++i) weights[i] = scalar.physical_basis()[i](x[0]);
            const auto expected = weighted_value(geometry, nodes, weights);
            // an independently assembled full-basis mean certifies the active-support evaluation oracle
            ASSERT_TRUE(expected.converged());
            // parallel spline evaluation matches the full basis at interiors, knots and clamped endpoints
            EXPECT_LT(error(values[k], expected.value), 2e-9);
            // deferred single-point spline evaluation uses the same controls as the prepared batch
            EXPECT_LT(error(Point(field(x)), values[k]), 2e-9);
        }
        const Eigen::Matrix<double, 1, 1> x = Eigen::Matrix<double, 1, 1>::Constant(1.2);
        const auto dofs = scalar.dof_handler().active_dofs(2);
        std::vector<double> dw;
        for (int id : dofs) dw.push_back(scalar.physical_basis()[id].gradient(1)(x[0]));
        const auto lin = field.linearization(x);
        const auto tangent = derivative(lin.weight_jvp(dw));
        constexpr double h = 2e-4;
        const Point center(field(x)), plus(field(Eigen::Matrix<double, 1, 1>::Constant(x[0] + h))),
          minus(field(Eigen::Matrix<double, 1, 1>::Constant(x[0] - h)));
        // physical basis derivatives agree with centered differences within one smooth spline span
        EXPECT_LT(error(ambient(geometry, center, tangent), (plus - minus) / (2. * h)), 3e-6);
        auto constant_nodes = controls(geometry, space.n_dofs());
        const Point constant(nodes[0]);
        for (std::size_t i = 0; i < constant_nodes.size(); ++i) constant_nodes[i] = constant;
        field.set_coeff(std::move(constant_nodes));
        const auto updated = Psi(field, execution_par);
        for (std::size_t i = 0; i < updated.size(); ++i) {
            // spline support and weights survive updates while all target geometries reproduce constant controls
            EXPECT_LT(error(updated[i], constant), 2e-9);
        }
    }
}
/// @brief exercises every implemented triangulation and the supported interval spline degrees
template <typename Geometry> void check_geometry(const Geometry& geometry) {
    for (int flags : {0, cache_cells}) {
        SCOPED_TRACE("cell cache flags " + std::to_string(flags));
        check_fem<1, 1>(geometry, flags);
        check_fem<1, 2>(geometry, flags);
        check_fem<2, 2>(geometry, flags);
        check_fem<2, 3>(geometry, flags);
        check_fem<3, 3>(geometry, flags);
        check_splines(geometry, flags);
    }
}
// log-Euclidean cached SPD fields use the same scalar support on all domains and spline degrees
TEST(GeometricDiscretizationCompatibility, LogEuclidean) { check_geometry(LE {}); }
// log-Cholesky fields preserve triangular-chart interpolation across all supported scalar domains
TEST(GeometricDiscretizationCompatibility, LogCholesky) { check_geometry(LC {}); }
// affine-invariant fields retain intrinsic mean and derivative solves across dimensions and spline supports
TEST(GeometricDiscretizationCompatibility, AffineInvariant) { check_geometry(AI {}); }
// Bures-Wasserstein fields retain lift-based means and differentials on every supported scalar domain
TEST(GeometricDiscretizationCompatibility, BuresWasserstein) { check_geometry(BW {}); }
// general-order Cheeger fields preserve geometric interpolation across all domains and spline supports
TEST(GeometricDiscretizationCompatibility, Cheeger) { check_geometry(CE {}); }
// specialized planar Cheeger kernels retain the same spatial support independently of tensor order
TEST(GeometricDiscretizationCompatibility, CheegerPlanarTensors) {
    check_geometry(manifold::CheegerLogEuclideanSPDGeometry<double, 2, Usage::InterpolationNodes> {});
}
// rotation fields preserve body derivatives, native rotation outputs and intrinsic means on every domain
TEST(GeometricDiscretizationCompatibility, SpecialOrthogonal) { check_geometry(SO {}); }
}   // namespace
