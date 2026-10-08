// SPDX-License-Identifier: GPL-3.0-or-later
#include <fdaPDE/manifold_optimization.h>
#include <gtest/gtest.h>

namespace {
using namespace fdapde;
using namespace fdapde::manifold;
using Sym = SymmetricMatrix<double, 2>;
using Batch = MatrixBatch<Sym>;
using Policy = Cache::Union<
  Cache::Log, Cache::Spectral, Cache::LogDividedDifferences, Cache::Sqrt, Cache::InverseSqrt, Cache::Cholesky,
  Cache::LogCholesky>;
using SPD = SPDMatrix<double, 2, Policy>;
using DynamicSPD = SPDMatrix<double, Dynamic, Policy>;

/// @brief supplies a coupled batch quadratic whose exact minimizer is its target batch
template <typename MatrixType> struct BatchQuadratic {
    /// @brief satisfies the solver workspace contract without caching matrix data
    struct Workspace { };
    using Point = MatrixBatch<MatrixType>;
    Point target;
    std::size_t evaluations = 0;
    /// @brief includes a residual difference coupling both matrices in one objective
    double cost(const Point& point, Workspace&) {
        ++evaluations;
        double value = 0;
        for (std::size_t i = 0; i < point.size(); ++i) value += .5 * (point[i] - target[i]).squared_norm();
        const MatrixType difference((point[0] - target[0]) - (point[1] - target[1]));
        return value + .5 * difference.squared_norm();
    }
    /// @brief returns the full ambient batch gradient including the cross-matrix residual
    Point egrad(const Point& point, Workspace&) {
        ++evaluations;
        Point result(point.size(), point.rows(), point.cols());
        for (std::size_t i = 0; i < point.size(); ++i) result[i] = point[i] - target[i];
        const MatrixType difference(result[0] - result[1]);
        result[0] += difference;
        result[1] -= difference;
        return result;
    }
    /// @brief applies the ambient Hessian while retaining its cross-matrix contributions
    Point ehess(const Point&, const Point& direction, Workspace&) {
        ++evaluations;
        Point result(direction);
        const MatrixType difference(direction[0] - direction[1]);
        result[0] += difference;
        result[1] -= difference;
        return result;
    }
};

/// @brief creates unequal target matrices with nonzero off-diagonal coordinates
Batch target_batch() {
    Batch target(2);
    target[0] = Sym(Vector<double, 3> {.2, .3, .4});
    target[1] = Sym(Vector<double, 3> {.5, -.6, .7});
    return target;
}

/// @brief checks the native result type and the independent closed-form coupled minimizer
template <typename Result> void check_solution(const Result& solution, const Batch& target) {
    // the optimizer result owns the original native symmetric batch type
    static_assert(std::same_as<decltype(solution.point), Batch>);
    // convergence must be certified for the entire coupled gradient
    ASSERT_TRUE(solution.converged());
    // one returned matrix is retained for each input factor
    ASSERT_EQ(solution.point.size(), target.size());
    for (std::size_t i = 0; i < target.size(); ++i) {
        // the exact minimizer makes both individual and cross-matrix residuals vanish
        EXPECT_LT((solution.point[i] - target[i]).norm(), 1e-9);
    }
}

/// @brief checks all optimizer entry points reject a mismatched product before objective work
template <typename Geometry, typename Point>
void check_invalid_initial(const Geometry& geometry, const Point& initial) {
    using Problem = BatchQuadratic<typename Point::MatrixType>;
    Problem problem {initial};
    // trust-region validation rejects the incompatible batch before constructing candidate evaluations
    EXPECT_THROW(RiemannianTrustRegion().optimize(problem, geometry, initial), std::invalid_argument);
    // no cost, gradient or Hessian runs on a batch outside the configured product
    EXPECT_EQ(problem.evaluations, 0);
    // descent validates the same geometry-to-batch count and shape relationship
    EXPECT_THROW(RiemannianSteepestDescent().optimize(problem, geometry, initial), std::invalid_argument);
    // the default descent context does not trigger objective work after rejection
    EXPECT_EQ(problem.evaluations, 0);
    EvaluationContext<Point, typename Problem::Workspace> context;
    // the retained-context overload rejects the mismatch at the same public boundary
    EXPECT_THROW(RiemannianSteepestDescent().optimize(problem, geometry, initial, context), std::invalid_argument);
    // supplying a context does not bypass product validation
    EXPECT_EQ(problem.evaluations, 0);
}

/// @brief checks exact cached owner types and zero steps in a fixed or dynamic SPD product
template <typename NativePoint, typename Geometry>
void check_spd_product(const Geometry& geometry, std::size_t count, int order) {
    using Symmetric = SymmetricMatrix<typename NativePoint::Scalar, NativePoint::Rows>;
    // native metric aliases preserve the complete requested owner including its cache policy
    static_assert(std::same_as<typename Geometry::Point, NativePoint>);
    // one matrix uses its ambient native symmetric tangent type
    static_assert(std::same_as<typename Geometry::Tangent, Symmetric>);
    const ProductGeometry product(geometry, count);
    using Product = std::remove_cvref_t<decltype(product)>;
    // product points own a batch of the exact cached native SPD owner
    static_assert(std::same_as<typename Product::Point, MatrixBatch<NativePoint>>);
    // product tangents own one native symmetric direction per SPD factor
    static_assert(std::same_as<typename Product::Tangent, MatrixBatch<Symmetric>>);
    // constructing a product does not alter the single-matrix tangent dimension
    EXPECT_EQ(geometry.dimension(), std::size_t(order) * (order + 1) / 2);
    // the product dimension counts all independent symmetric coordinates across its factors
    EXPECT_EQ(product.dimension(), count * geometry.dimension());
    const MatrixBatch<NativePoint> points(count, order, order);
    const auto zero = product.zero_tangent(points);
    // tangent allocation retains exactly the configured number of factors
    EXPECT_EQ(zero.size(), count);
    // tangent allocation retains the fixed or dynamic matrix row extent
    EXPECT_EQ(zero.rows(), order);
    // tangent allocation retains the fixed or dynamic matrix column extent
    EXPECT_EQ(zero.cols(), order);
    const auto retracted = product.retract(points, zero, 1.);
    // joint retraction preserves the exact cached native batch owner type
    static_assert(std::same_as<std::remove_cvref_t<decltype(retracted)>, MatrixBatch<NativePoint>>);
    // joint retraction returns one checked SPD matrix per configured factor
    ASSERT_EQ(retracted.size(), count);
    for (std::size_t i = 0; i < count; ++i) {
        // a zero tangent leaves each certified identity point unchanged
        EXPECT_LT((retracted[i] - points[i]).norm(), 1e-12);
    }
}

/// @brief checks invalid product cardinalities without allocating any matrix batch
template <typename Geometry> void check_invalid_counts(const Geometry& geometry) {
    const std::size_t empty = 0, overflow = std::numeric_limits<std::size_t>::max();
    // an empty product has no admissible optimization factors
    EXPECT_THROW(ProductGeometry(geometry, empty), std::invalid_argument);
    // multiplying the maximum count by the nontrivial local dimension must fail before allocation
    EXPECT_THROW(ProductGeometry(geometry, overflow), std::length_error);
}

// an explicit Euclidean product lets both solvers minimize a coupled native batch in one solve
TEST(ProductGeometry, CoupledBatchUsesOneTrustRegionAndDescentSolve) {
    const Batch initial(2), target = target_batch();
    const EuclideanGeometry<Sym> element;
    const ProductGeometry geometry(element, initial.size());
    // the underlying metric continues to describe one native symmetric matrix
    static_assert(std::same_as<EuclideanGeometry<Sym>::Point, Sym>);
    // a single two-by-two symmetric matrix retains three independent coordinates
    EXPECT_EQ(element.dimension(), 3);
    // the explicit two-factor product has six independent coordinates
    EXPECT_EQ(geometry.dimension(), 6);
    BatchQuadratic<Sym> problem {target};
    // trust-region optimization returns the exact coupled minimizer in native batch storage
    check_solution(RiemannianTrustRegion().optimize(problem, geometry, initial), target);
    SteepestDescentOptions options;
    options.gradient_tolerance = 1e-10;
    // descent solves the same cross-matrix objective using its joint product gradient
    check_solution(RiemannianSteepestDescent(options).optimize(problem, geometry, initial), target);
}

// externally supplied descent contexts retain evaluations with batch gradients after the joint solve
TEST(ProductGeometry, BatchDescentRetainsSuppliedContext) {
    const Batch initial(2), target = target_batch();
    const ProductGeometry geometry(EuclideanGeometry<Sym> {}, initial.size());
    BatchQuadratic<Sym> problem {target};
    EvaluationContext<Batch, BatchQuadratic<Sym>::Workspace> context;
    SteepestDescentOptions options;
    options.gradient_tolerance = 1e-10;
    const auto solution = RiemannianSteepestDescent(options).optimize(problem, geometry, initial, context);
    // supplying an external batch context preserves the same exact minimizer and native result type
    check_solution(solution, target);
    // the supplied current slot retains the accepted candidate cost
    ASSERT_TRUE(context.current().cost().has_value());
    // the retained cost agrees exactly with the final optimizer result
    EXPECT_DOUBLE_EQ(*context.current().cost(), solution.cost);
    // the supplied slot retains a native batch gradient for subsequent evaluation
    ASSERT_TRUE(context.current().gradient().has_value());
    // the retained gradient has one symmetric direction per configured field factor
    EXPECT_EQ(context.current().gradient()->size(), initial.size());
}

// product count and matrix shape errors are rejected before any coupled objective evaluation
TEST(ProductGeometry, BatchMismatchRejectsBeforeEvaluation) {
    // a three-factor product rejects a two-matrix initial batch in every optimizer entry point
    check_invalid_initial(ProductGeometry(EuclideanGeometry<Sym> {}, 3), Batch(2));
    using DynamicSym = SymmetricMatrix<double, Dynamic>;
    // a product of order-three matrices rejects an order-two initial batch before cost evaluation
    check_invalid_initial(ProductGeometry(EuclideanGeometry<DynamicSym>(3, 3), 2), MatrixBatch<DynamicSym>(2, 2, 2));
}

// all five native SPD metrics produce products of the exact requested cached owner type
TEST(ProductGeometry, NativeSPDTypesAndZeroSteps) {
    // log-Euclidean products preserve full-cache SPD owners and their zero tangent identity
    check_spd_product<SPD>(LogEuclideanGeometry<SPD> {}, 3, 2);
    // affine-invariant products preserve owners without deriving a replacement cache policy
    check_spd_product<SPD>(AffineInvariantGeometry<SPD> {}, 3, 2);
    // Bures-Wasserstein zero retraction retains the declared native batch type
    check_spd_product<SPD>(BuresWassersteinGeometry<SPD> {}, 3, 2);
    // log-Cholesky zero retraction retains the same cache-bearing identity owners
    check_spd_product<SPD>(LogCholeskyGeometry<SPD> {}, 3, 2);
    // Cheeger products preserve native point types with an explicitly selected scalar epsilon
    check_spd_product<SPD>(CheegerLogEuclideanGeometry<SPD>(.75), 3, 2);
}

// runtime matrix extents are inferred from the element metric independently of the factor count
TEST(ProductGeometry, DynamicExtents) {
    using DynamicSym = SymmetricMatrix<double, Dynamic>;
    const EuclideanGeometry<DynamicSym> element(3, 3);
    const ProductGeometry product(element, 4);
    // a three-by-three symmetric element has six independent coordinates
    EXPECT_EQ(element.dimension(), 6);
    // four factors contribute the sum of their local dimensions
    EXPECT_EQ(product.dimension(), 24);
    const auto zero = product.zero_tangent(MatrixBatch<DynamicSym>(4, 3, 3));
    // a dynamically shaped tangent retains the explicit factor count
    EXPECT_EQ(zero.size(), 4);
    // the Euclidean metric supplies the common runtime row extent
    EXPECT_EQ(zero.rows(), 3);
    // the Euclidean metric supplies the common runtime column extent
    EXPECT_EQ(zero.cols(), 3);
    // dynamic log-Euclidean products retain order three and the exact native cache policy
    check_spd_product<DynamicSPD>(LogEuclideanGeometry<DynamicSPD>(3), 4, 3);
    // dynamic affine-invariant products retain both runtime extents
    check_spd_product<DynamicSPD>(AffineInvariantGeometry<DynamicSPD>(3), 4, 3);
    // dynamic Bures-Wasserstein zero steps retain native order-three identity batches
    check_spd_product<DynamicSPD>(BuresWassersteinGeometry<DynamicSPD>(3), 4, 3);
    // dynamic log-Cholesky products keep the four-factor count outside their order-three chart
    check_spd_product<DynamicSPD>(LogCholeskyGeometry<DynamicSPD>(3), 4, 3);
    // dynamic Cheeger products preserve the native SPD type with explicit order and epsilon
    check_spd_product<DynamicSPD>(CheegerLogEuclideanGeometry<DynamicSPD>(3, .75), 4, 3);
}

// Cheeger epsilon belongs to the element metric and is preserved when copied into a product
TEST(ProductGeometry, CheegerParameterIsPreserved) {
    const CheegerLogEuclideanGeometry<SPD> element(.75);
    const ProductGeometry product(element, 2);
    const SPD point(Sym(Vector<double, 3> {1., 0., 2.}));
    const Sym direction(Vector<double, 3> {0., .3, 0.});
    MatrixBatch<SPD> points(2);
    Batch tangents(2);
    for (std::size_t i = 0; i < points.size(); ++i) {
        points[i] = point;
        tangents[i] = direction;
    }
    // construction leaves the scalar epsilon independent of the number of product factors
    EXPECT_DOUBLE_EQ(element.epsilon(), .75);
    // anisotropic off-diagonal directions reproduce twice the metric with the requested epsilon
    EXPECT_NEAR(
      product.inner_product(points, tangents, tangents), 2 * element.inner_product(point, direction, direction), 1e-12);
    const auto retracted = product.retract(points, tangents, 0.);
    for (std::size_t i = 0; i < points.size(); ++i) {
        // a zero step keeps each cached anisotropic SPD point unchanged
        EXPECT_LT((retracted[i] - points[i]).norm(), 1e-12);
    }
}

// product counts fail at construction when empty or too large to multiply by the local tangent dimension
TEST(ProductGeometry, RejectsEmptyAndOverflowCounts) {
    // Euclidean products reject zero factors and size multiplication overflow
    check_invalid_counts(EuclideanGeometry<Sym> {});
    // log-Euclidean products enforce the same count bounds before matrix allocation
    check_invalid_counts(LogEuclideanGeometry<SPD> {});
    // affine-invariant products enforce the same count bounds
    check_invalid_counts(AffineInvariantGeometry<SPD> {});
    // Bures-Wasserstein products enforce the same count bounds
    check_invalid_counts(BuresWassersteinGeometry<SPD> {});
    // log-Cholesky products enforce the same count bounds
    check_invalid_counts(LogCholeskyGeometry<SPD> {});
    // Cheeger products validate counts independently of their explicit epsilon parameter
    check_invalid_counts(CheegerLogEuclideanGeometry<SPD>(.75));
}
}   // namespace
