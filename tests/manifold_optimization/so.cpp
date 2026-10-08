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

#include <fdaPDE/manifold_geometry.h>
#include <fdaPDE/manifold_solvers.h>
#include <gtest/gtest.h>

#include <numbers>
#include <random>

namespace {
using namespace fdapde;
using G = manifold::SOGeometry<double, Dynamic>;
using Dense = Matrix<double, Dynamic, Dynamic>;
using Cached = RotationMatrix<double, Dynamic, Dynamic, RotationCache::Union<RotationCache::Schur, RotationCache::Log>>;

// the supported optimizer contracts use body skew tangents and owning verified points
static_assert(manifold::GeodesicGeometry<G> && manifold::VectorTransportGeometry<G>);

template <typename A, typename B> void expect_matrix(const A& a, const B& b, double tolerance = 2e-11) {
    // compatible row counts ensure that the coefficient comparison checks the entire result
    ASSERT_EQ(a.rows(), b.rows());
    // compatible column counts prevent silently skipping expected coefficients
    ASSERT_EQ(a.cols(), b.cols());
    for (int i = 0; i < a.rows(); ++i)
        for (int j = 0; j < a.cols(); ++j) {
            // every coefficient agrees with the supplied independent reference
            EXPECT_NEAR(a(i, j), b(i, j), tolerance);
        }
}
Dense plane(int n, double angle) {
    Dense result(IdentityMatrix<double, Dynamic, Dynamic>(n, n));
    result(0, 0) = result(1, 1) = std::cos(angle);
    result(0, 1) = -std::sin(angle);
    result(1, 0) = std::sin(angle);
    return result;
}

// checked rotations reject reflections and nonfinite or nonorthogonal coefficients regardless of debug mode
TEST(SORotation, InvariantsAndAtomicViewReplacement) {
    const Dense reflection(Matrix<double, 2, 2>({1, 0, 0, -1}));
    // a reflection is orthogonal but violates the determinant plus-one invariant
    EXPECT_THROW((Cached(reflection)), std::domain_error);
    const Dense scaled(Matrix<double, 2, 2>({2, 0, 0, 1}));
    // scaling a column violates orthogonality before cache construction
    EXPECT_THROW((Cached(scaled)), std::domain_error);
    Dense nonfinite(plane(2, .2));
    nonfinite(0, 0) = std::numeric_limits<double>::quiet_NaN();
    // nonfinite coefficients cannot become verified rotations
    EXPECT_THROW((Cached(nonfinite)), std::domain_error);
    Cached q(plane(2, .4));
    auto view = q.view();
    typename Cached::ConstView read_only(view);
    const Cached saved(q);
    // a failed view update must not replace either coefficients or the cached branch
    EXPECT_THROW(view = reflection, std::domain_error);
    // the preserved coefficients agree with the saved checked value
    expect_matrix(q, saved);
    // the preserved logarithm cache still represents the saved rotation angle
    EXPECT_NEAR(q.cache().distance(), std::sqrt(2.) * .4, 1e-12);
    view = plane(2, .8);
    // a read-only alias observes the whole-value update through the mutable view
    expect_matrix(read_only, plane(2, .8));
    // updated cache quantities use the new coefficients
    EXPECT_NEAR(q.cache().distance(), std::sqrt(2.) * .8, 1e-12);
    q = plane(2, .3);
    // same-shape owner replacement preserves existing value views and their cache bindings
    EXPECT_NEAR(read_only.cache().distance(), std::sqrt(2.) * .3, 1e-12);
    const auto inverse = q.inv();
    const auto identity = q * inverse;
    // inverse and composition agree with the analytic identity rotation
    expect_matrix(identity, Cached::Identity(2));
}

// ordinary logs reject the cut locus while explicit branches and distances remain available for every policy
TEST(SORotation, CutLocusAndNumericalProximity) {
    const Dense cut(Matrix<double, 2, 2>({-1, 0, 0, -1}));
    const RotationMatrix<double, 2, 2> none(cut);
    const RotationMatrix<double, 2, 2, RotationCache::Schur> schur(cut);
    const RotationMatrix<double, 2, 2, RotationCache::Log> log(cut);
    const Cached both(cut);
    // selecting a logarithm cache does not reject a valid cut-locus rotation or hide its ambiguity
    EXPECT_EQ(log.cache().diagnostics().status, RotationLogStatus::Ambiguous);
    // the Schur-only cached distance follows the full-Frobenius angle normalization
    EXPECT_NEAR(rotation_distance_identity(schur), std::sqrt(2.) * std::numbers::pi, 1e-12);
    // uncached distance also remains defined at the cut locus
    EXPECT_NEAR(rotation_distance_identity(none), rotation_distance_identity(schur), 1e-12);
    // the ordinary uncached logarithm must not silently choose a nonunique branch
    EXPECT_THROW(rotation_log(none), std::domain_error);
    // the logarithm cache stores diagnostics instead of an arbitrary cut-locus logarithm
    EXPECT_THROW(rotation_log(log), std::domain_error);
    G geometry(2);
    const auto first = G::Point::Identity(2);
    const auto selected = geometry.minimum_logarithm(first, both);
    const auto curve = geometry.geodesic(first, both, selected);
    // a prepared explicit branch reaches the supplied cut-locus endpoint
    expect_matrix(G::Point(curve(1)), cut);
    // the same prepared branch has constant speed and the correct midpoint distance
    EXPECT_NEAR(geometry.distance(first, G::Point(curve(.5))), std::numbers::pi / std::sqrt(2.), 1e-11);
    auto relabeled = selected;
    relabeled.diagnostics = {};
    // endpoint-derived diagnostics preserve cut-locus ambiguity even if the supplied branch metadata is changed
    EXPECT_EQ(geometry.geodesic(first, both, relabeled).diagnostics().status, RotationLogStatus::Ambiguous);
    // implicit geodesic preparation cannot silently choose an ambiguous branch
    EXPECT_THROW(geometry.geodesic(first, both), std::domain_error);
    const Cached near(plane(2, std::numbers::pi - 1e-8));
    // a resolved nearby rotation is numerically close to the cut locus without being classified as nonunique
    EXPECT_EQ(near.cache().diagnostics().status, RotationLogStatus::NearCut);
    // ordinary logarithms remain available on a resolved nearby branch
    EXPECT_NO_THROW(rotation_log(near));
    // derivatives explicitly require a numerically regular branch
    EXPECT_THROW(geometry.rotation_penalty_gradient(near), std::domain_error);
    const Cached unresolved(plane(2, std::numbers::pi - 1e-15));
    // sub-resolution proximity is distinguished from the exact negative identity's mathematical ambiguity
    EXPECT_EQ(unresolved.cache().diagnostics().status, RotationLogStatus::Unresolved);
}

// mixed rotation planes in general dimensions retain distance and reconstruction near zero and pi
TEST(SORotation, GeneralOrderConjugatedPlanes) {
    std::mt19937 random(42);
    std::uniform_real_distribution<double> sample(-1, 1);
    for (int n : {2, 3, 4, 6, 9})
        for (int trial = 0; trial < 8; ++trial) {
            G geometry(n);
            const auto identity = G::Point::Identity(n);
            auto omega = geometry.zero_tangent(identity);
            for (int i = 0; i < n; ++i)
                for (int j = i + 1; j < n; ++j) omega(i, j) = sample(random);
            const auto basis = rotation_exp(omega);
            Dense blocks(IdentityMatrix<double, Dynamic, Dynamic>(n, n));
            double distance = 0;
            for (int k = 0; k + 1 < n; k += 2) {
                double angle = sample(random) * std::numbers::pi;
                if (trial % 4 == 0) angle *= 1e-5;
                if (trial % 4 == 1) angle = std::numbers::pi - std::abs(angle) * 1e-8;
                if (trial % 4 == 2) angle = (k ? 1 : 0) * std::numbers::pi;
                blocks(k, k) = blocks(k + 1, k + 1) = std::cos(angle);
                blocks(k, k + 1) = -std::sin(angle);
                blocks(k + 1, k) = std::sin(angle);
                distance = std::hypot(distance, std::sqrt(2.) * angle);
            }
            const Dense expected(basis * blocks * basis.transpose());
            const Cached rotation(expected);
            // orthogonal conjugation preserves the analytic sum of squared principal plane angles
            EXPECT_NEAR(rotation_distance_identity(rotation), distance, 2e-10);
            const auto branch = minimum_rotation_log(rotation);
            // exponentiating an explicitly selected minimum logarithm reconstructs the known conjugated blocks
            expect_matrix(rotation_exp(branch.tangent), expected, 2e-10);
        }
    const RotationMatrix<float, 3, 3, RotationCache::Log> single(Matrix<float, 3, 3>({0, -1, 0, 1, 0, 0, 0, 0, 1}));
    // fixed single-precision rotations use the same general path and angle normalization
    EXPECT_NEAR(rotation_distance_identity(single), std::sqrt(2.) * std::numbers::pi / 2, 2e-5);
    const auto trivial = RotationMatrix<double, 1, 1, RotationCache::Log>::Identity();
    // the trivial SO(1) group has zero distance and a valid empty skew logarithm
    EXPECT_DOUBLE_EQ(rotation_distance_identity(trivial), 0);
}

// body-coordinate gradients agree with finite differences using the full Frobenius metric without a half factor
TEST(SOGeometry, GradientsMetricTransportAndPreparedCurves) {
    G geometry(3);
    const G::Point point(plane(3, .4));
    auto direction = geometry.zero_tangent(point);
    direction(0, 1) = .3;
    direction(0, 2) = -.2;
    direction(1, 2) = .15;
    const Dense ambient(Matrix<double, 3, 3>({1, 2, -1, .3, 4, .7, 1, -.5, 2}));
    const auto gradient = geometry.euclidean_to_riemannian_gradient(point, ambient);
    auto objective = [&](const auto& q) {
        double value = 0;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) value += ambient(i, j) * q(i, j);
        return value;
    };
    const double h = 1e-6;
    const double difference =
      (objective(geometry.exponential(point, direction, h)) - objective(geometry.exponential(point, direction, -h))) /
      (2 * h);
    // the metric pairing of the converted gradient equals the central directional derivative
    EXPECT_NEAR(geometry.inner_product(point, gradient, direction), difference, 2e-9);
    // off-diagonal tangent coefficients are counted twice in the full Frobenius metric
    EXPECT_NEAR(geometry.inner_product(point, direction, direction), 2 * (.3 * .3 + .2 * .2 + .15 * .15), 1e-13);
    const auto penalty_gradient = geometry.rotation_penalty_gradient(point);
    const double penalty_difference = (geometry.rotation_penalty(geometry.exponential(point, direction, h)) -
                                       geometry.rotation_penalty(geometry.exponential(point, direction, -h))) /
                                      (2 * h);
    // the identity-penalty logarithm uses the same gradient normalization
    EXPECT_NEAR(geometry.inner_product(point, penalty_gradient, direction), penalty_difference, 2e-9);
    const auto last = geometry.exponential(point, direction);
    const auto curve = geometry.geodesic(point, last);
    for (double t : {-.3, 0., .5, 1., 1.3}) {
        // prepared interpolation and extrapolation agree with the independently evaluated exponential path
        expect_matrix(G::Point(curve(t)), geometry.exponential(point, direction, t));
    }
    const auto transported = geometry.transport(point, last, direction);
    // parallel transport preserves the bi-invariant metric norm
    EXPECT_NEAR(geometry.norm(last, transported), geometry.norm(point, direction), 1e-12);
    const auto owned = G(3).geodesic(G::Point::Identity(3), G::Point(plane(3, .5)))(.4);
    // temporary geometry, endpoints and prepared curve may expire before expression materialization
    expect_matrix(G::Point(owned), plane(3, .2));
}

// batch views keep rotation coefficients and selected caches coherent across updates and independent copies
TEST(SORotation, BatchCachesAndSelections) {
    // dynamic rotation batches cannot construct nonsquare rows of supposedly verified rotations
    EXPECT_THROW((MatrixBatch<Cached>(1, 2, 3)), std::invalid_argument);
    MatrixBatch<Cached> points(3, 3, 3);
    points[1] = plane(3, .6);
    auto view = points[1];
    typename Cached::ConstView read(view);
    const auto mapped = points.select(std::vector<int> {1, 1}).map([](const auto& q) { return q; });
    MatrixBatch<Cached> selected(mapped);
    // selection and map preserve each independently materialized rotation and its cached distance
    EXPECT_NEAR(selected[0].cache().distance(), std::sqrt(2.) * .6, 1e-12);
    const auto saved = points;
    view = plane(3, .9);
    // an existing const view observes jointly replaced coefficients and cache
    EXPECT_NEAR(read.cache().distance(), std::sqrt(2.) * .9, 1e-12);
    // deep batch copies retain the original independent cache contents
    EXPECT_NEAR(saved[1].cache().distance(), std::sqrt(2.) * .6, 1e-12);
    MatrixBatch<RotationMatrix<double, 1, 1, RotationCache::Log>> trivial(2);
    // zero-dimensional SO(1) caches also initialize correctly inside aggregate batch storage
    EXPECT_DOUBLE_EQ(trivial[0].cache().distance(), 0);
}
// fixed geometries share the dynamic implementation and enforce shape, finite parameters and branch consistency
TEST(SOGeometry, FixedOrderAndInvalidInputs) {
    const manifold::SOGeometry<double, 2, RotationUsage::IdentityLog> geometry;
    using Fixed = decltype(geometry)::Point;
    const auto first = Fixed::Identity();
    const Fixed last(plane(2, .6));
    const auto curve = geometry.geodesic(first, last);
    // a fixed prepared midpoint matches the independent planar rotation formula
    expect_matrix(Fixed(curve(.5)), plane(2, .3));
    const auto invalid = curve(std::numeric_limits<double>::quiet_NaN());
    // parameter validation is deferred until complete expression materialization
    EXPECT_THROW((Fixed(invalid)), std::invalid_argument);
    const auto wrong = RotationMatrix<double, Dynamic, Dynamic>::Identity(3);
    // geometry methods reject a verified point having the wrong runtime order
    EXPECT_THROW(geometry.distance(first, wrong), std::invalid_argument);
    auto branch = geometry.minimum_logarithm(first, last);
    branch.tangent(0, 1) = 0;
    // explicit branches are checked against the supplied endpoint before preparation
    EXPECT_THROW(geometry.geodesic(first, last, branch), std::invalid_argument);
    const manifold::SOGeometry<float, 3> single;
    const auto single_identity = decltype(single)::Point::Identity();
    auto omega = single.zero_tangent(single_identity);
    omega(0, 1) = -.2f;
    const auto result = single.exponential(single_identity, omega);
    // fixed float geometry exponentials follow the same planar formula within float accuracy
    EXPECT_NEAR(result(0, 1), -std::sin(.2f), 2e-6);
}

/// @brief supplies a coupled ambient quadratic with an independently known attainable rotation minimum
template <typename Geometry> struct CoupledRotations {
    using Point = typename manifold::ProductGeometry<Geometry>::Point;
    using Ambient = MatrixBatch<Matrix<typename Geometry::Scalar, Geometry::Point::Rows, Geometry::Point::Cols>>;
    /// @brief provides candidate-local storage without shared objective intermediates
    struct Workspace { };
    Point target;
    /// @brief penalizes both factor residuals and their difference so the objective is not separable
    double cost(const Point& point, Workspace&) const {
        double value = 0;
        for (std::size_t i = 0; i < point.size(); ++i) value += .5 * (point[i] - target[i]).squared_norm();
        const auto difference = (point[0] - target[0]) - (point[1] - target[1]);
        return value + .5 * difference.squared_norm();
    }
    /// @brief differentiates the shared residual difference in both ambient factor matrices
    Ambient egrad(const Point& point, Workspace&) const {
        Ambient result(point.size(), point.rows(), point.cols());
        for (std::size_t i = 0; i < point.size(); ++i) result[i] = point[i] - target[i];
        const typename Ambient::MatrixType difference(result[0] - result[1]);
        result[0] += difference;
        result[1] -= difference;
        return result;
    }
    /// @brief applies every diagonal and cross-factor block of the constant ambient Hessian
    Ambient ehess(const Point&, const Ambient& direction, Workspace&) const {
        Ambient result(direction);
        const typename Ambient::MatrixType difference(direction[0] - direction[1]);
        result[0] += difference;
        result[1] -= difference;
        return result;
    }
};

/// @brief exercises product tangent operations with nonidentity rotations and borrowed skew entries
template <typename Geometry> void check_product_operations(const Geometry& factor) {
    using Product = manifold::ProductGeometry<Geometry>;
    // fixed and dynamic SO products retain geodesic and transport contracts with owning skew batches
    static_assert(manifold::GeodesicGeometry<Product> && manifold::VectorTransportGeometry<Product>);
    const Product geometry(factor, 2);
    typename Product::Point points(2, factor.order(), factor.order());
    points[0] = plane(factor.order(), .4);
    points[1] = plane(factor.order(), -.25);
    auto u = geometry.zero_tangent(points), v = geometry.zero_tangent(points);
    for (std::size_t i = 0; i < points.size(); ++i) {
        u[i](0, 1) = .2 + .1 * i;
        u[i](0, 2) = -.15;
        u[i](1, 2) = .07;
        v[i](0, 1) = -.1;
        v[i](0, 2) = .3;
        v[i](1, 2) = .15 - .1 * i;
    }
    double pairing = 0, squared_norm = 0;
    for (std::size_t k = 0; k < points.size(); ++k)
        for (int i = 0; i < factor.order(); ++i)
            for (int j = 0; j < factor.order(); ++j) {
                pairing += u[k](i, j) * v[k](i, j);
                squared_norm += u[k](i, j) * u[k](i, j);
            }
    // the product counts only independent strict-upper coefficients of both skew tangents
    EXPECT_EQ(geometry.dimension(), 2 * factor.dimension());
    // an independent complete-matrix contraction counts both signed skew triangles
    EXPECT_NEAR(geometry.inner_product(points, u, v), pairing, 1e-14);
    // the product norm agrees with the independently accumulated full Frobenius norm
    EXPECT_NEAR(geometry.norm(points, u), std::sqrt(squared_norm), 1e-14);
    const auto projected = geometry.project(points, u), combined = geometry.linear_combination(points, 2., u, -.5, v);
    const auto ambient = geometry.to_ambient(points, u);
    const auto endpoint = geometry.exponential(points, u, .3), retracted = geometry.retract(points, u, .3);
    const auto logarithm = geometry.logarithm(points, endpoint), transported = geometry.transport(points, endpoint, v);
    for (std::size_t i = 0; i < points.size(); ++i) {
        // projecting a borrowed skew body tangent at a nonidentity rotation must preserve its coordinates
        expect_matrix(projected[i], u[i]);
        // typed skew arithmetic keeps the body convention instead of selecting the ambient projection overload
        expect_matrix(factor.project(points[i], 2. * u[i] - u[i]), u[i]);
        // both source skew views participate in the prescribed factorwise linear combination
        expect_matrix(combined[i], 2. * u[i] - .5 * v[i]);
        // body velocity becomes an ambient matrix by multiplying by its own base rotation
        expect_matrix(ambient[i], points[i] * u[i]);
        // a full dense velocity still takes the ambient projection route and recovers its body coordinates
        expect_matrix(factor.project(points[i], ambient[i]), u[i]);
        // the product exponential uses the same finite geodesic parameter at every factor
        expect_matrix(endpoint[i], factor.exponential(points[i], typename Geometry::Tangent(u[i]), .3));
        // exact exponential retraction preserves each verified product rotation
        expect_matrix(retracted[i], endpoint[i]);
        // a short exponential stays on the unique logarithm branch with the original body direction
        expect_matrix(logarithm[i], .3 * u[i]);
        // transporting a borrowed skew direction agrees with its independent owning factor invocation
        expect_matrix(transported[i], factor.transport(points[i], endpoint[i], typename Geometry::Tangent(v[i])));
    }
    // the product geodesic distance equals the combined speed times the short common parameter
    EXPECT_NEAR(geometry.distance(points, endpoint), .3 * std::sqrt(squared_norm), 2e-12);
    // factorwise parallel transport preserves the full product metric norm
    EXPECT_NEAR(geometry.norm(endpoint, transported), geometry.norm(points, v), 2e-12);
    const auto zero = geometry.zero_tangent(points);
    // freshly allocated packed skew rows represent the additive identity at every factor
    EXPECT_DOUBLE_EQ(geometry.norm(points, zero), 0.);
}

/// @brief validates cross-factor ambient derivatives and solves the same coupled rotation objective
template <typename Geometry> void check_product_optimization(const Geometry& factor) {
    using Product = manifold::ProductGeometry<Geometry>;
    const Product geometry(factor, 2);
    typename Product::Point initial(2, factor.order(), factor.order()), target(initial);
    target[0] = plane(factor.order(), .35);
    target[1] = plane(factor.order(), -.2);
    initial[0] = plane(factor.order(), -.1);
    initial[1] = plane(factor.order(), .15);
    CoupledRotations<Geometry> problem {target};
    typename CoupledRotations<Geometry>::Workspace workspace;
    auto u = geometry.zero_tangent(initial), v = geometry.zero_tangent(initial);
    u[0](0, 1) = .3;
    u[0](0, 2) = -.1;
    v[1](0, 1) = -.2;
    v[1](0, 2) = .12;
    const auto egrad = problem.egrad(initial, workspace);
    const auto grad = geometry.euclidean_to_riemannian_gradient(initial, egrad);
    const auto ambient_u = geometry.to_ambient(initial, u), ambient_v = geometry.to_ambient(initial, v);
    const auto hessian =
      geometry.euclidean_to_riemannian_hessian(initial, egrad, problem.ehess(initial, ambient_u, workspace), u);
    double mixed = 0;
    for (int i = 0; i < factor.order(); ++i)
        for (int j = 0; j < factor.order(); ++j) mixed -= ambient_u[0](i, j) * ambient_v[1](i, j);
    // disjoint factor directions isolate the exact negative cross block of the coupled ambient residual
    EXPECT_NEAR(geometry.inner_product(initial, v, hessian), mixed, 2e-13);
    constexpr double step = 1e-6;
    const auto plus = geometry.exponential(initial, u, step), minus = geometry.exponential(initial, u, -step);
    const double finite = (problem.cost(plus, workspace) - problem.cost(minus, workspace)) / (2 * step);
    // independent geodesic perturbations check the converted coupled gradient in body coordinates
    EXPECT_NEAR(geometry.inner_product(initial, grad, u), finite, 2e-9);
    manifold::TrustRegionOptions options;
    options.gradient_tolerance = 1e-9;
    const auto result = manifold::RiemannianTrustRegion(options).optimize(problem, geometry, initial);
    // the complete ambient Hessian route converges to the coupled objective's attainable zero residual
    ASSERT_TRUE(result.converged());
    // no positive residual remains at the independently prescribed rotation target
    EXPECT_LT(result.cost, 1e-16);
    for (std::size_t i = 0; i < target.size(); ++i) {
        // each optimized native rotation matches its factor of the exact coupled minimizer
        expect_matrix(result.point[i], target[i], 2e-8);
    }
    manifold::SteepestDescentOptions descent_options;
    descent_options.gradient_tolerance = 1e-8;
    const auto descent = manifold::RiemannianSteepestDescent(descent_options).optimize(problem, geometry, initial);
    // first-order optimization accepts the same native skew-batch gradient without a Hessian callback
    ASSERT_TRUE(descent.converged());
    // Armijo updates reach the same zero-residual solution as the second-order solver
    EXPECT_LT(descent.cost, 1e-15);
}

// fixed and dynamic products preserve every first-order, geodesic and transport operation on skew views
TEST(SOGeometry, NativeProductOperations) {
    // fixed SO(3) factors retain the selected point-log cache and typed borrowed tangent semantics
    check_product_operations(manifold::SOGeometry<double, 3, RotationUsage::IdentityLog> {});
    // runtime SO(3) factors reproduce the same metric and map oracles through dynamic skew rows
    check_product_operations(G(3));
}

// joint optimization retains cross-factor ambient Hessian terms and body-coordinate tangent conventions
TEST(SOGeometry, CoupledProductOptimization) {
    // fixed packed tangents preserve the coupled gradient, cross Hessian block and exact optimizer target
    check_product_optimization(manifold::SOGeometry<double, 3> {});
    // dynamic tangent storage uses the same full ambient derivative and joint solver contracts
    check_product_optimization(G(3));
}

/// @brief checks the zero-dimensional product without indexing its empty packed tangent storage
template <typename Geometry> void check_trivial_product(const Geometry& factor) {
    const manifold::ProductGeometry geometry(factor, 2);
    using Product = decltype(geometry);
    const typename Product::Point points(2, 1, 1);
    const auto zero = geometry.zero_tangent(points);
    // the product of trivial rotation groups has no independent tangent coordinates
    EXPECT_EQ(geometry.dimension(), 0u);
    // empty packed tangents retain the zero metric norm without dereferencing coefficient storage
    EXPECT_DOUBLE_EQ(geometry.norm(points, zero), 0.);
    const auto retracted = geometry.retract(points, zero, 1.);
    // zero-dimensional retraction preserves the only possible rotation in each factor
    EXPECT_DOUBLE_EQ(geometry.distance(points, retracted), 0.);
    CoupledRotations<Geometry> problem {points};
    const auto result = manifold::RiemannianTrustRegion().optimize(problem, geometry, points);
    // the zero tangent gradient certifies the trivial objective before any subproblem solve
    EXPECT_TRUE(result.converged());
    // a singleton manifold requires no trust-region iterations
    EXPECT_EQ(result.iterations, 0u);
}

// SO(1) products keep verified identity points with valid zero-length tangent rows
TEST(SOGeometry, TrivialProduct) {
    // fixed SO(1) verifies zero-dimensional arithmetic and optimizer termination without stored coefficients
    check_trivial_product(manifold::SOGeometry<double, 1> {});
    // runtime SO(1) retains the same valid empty tangent representation and immediate convergence
    check_trivial_product(G(1));
}

}   // namespace
