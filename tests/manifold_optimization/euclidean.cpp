// SPDX-License-Identifier: GPL-3.0-or-later
#include <fdaPDE/manifold_optimization.h>
#include <gtest/gtest.h>

namespace {
using namespace fdapde;
using namespace fdapde::manifold;
using Sym = SymmetricMatrix<double, 2>;
using Batch = MatrixBatch<Sym>;

/// @brief supplies a coupled quadratic whose Frobenius gradient and Hessian have an exact known solution
struct Quadratic {
    /// @brief satisfies evaluation-local workspace ownership without retaining candidate data
    struct Workspace { };
    Batch target;
    /// @brief sums full Frobenius residuals, including both mirrored off-diagonal entries
    double cost(const Batch& point, Workspace&) {
        double value = 0;
        for (std::size_t i = 0; i < point.size(); ++i) value += 0.5 * (point[i] - target[i]).squared_norm();
        const Sym difference((point[0] - target[0]) - (point[1] - target[1]));
        return value + 0.5 * difference.squared_norm();
    }
    /// @brief returns full ambient residuals with the coupled difference contribution
    Batch egrad(const Batch& point, Workspace&) {
        Batch result(point.size());
        for (std::size_t i = 0; i < point.size(); ++i) result[i] = point[i] - target[i];
        const Sym difference(result[0] - result[1]);
        result[0] += difference;
        result[1] -= difference;
        return result;
    }
    /// @brief applies the coupled ambient Hessian to a native symmetric batch direction
    Batch ehess(const Batch&, const Batch& direction, Workspace&) {
        Batch result(direction);
        const Sym difference(direction[0] - direction[1]);
        result[0] += difference;
        result[1] -= difference;
        return result;
    }
};

// vector geometry retains Euclidean arithmetic and stable norms without an Eigen dependency
TEST(EuclideanGeometry, NativeVectorsAndStableNorms) {
    using Vec = Vector<double, Dynamic>;
    const EuclideanGeometry<Vec> geometry(3);
    const Vec point(Vector<double, 3> {0., 0., 0.}), u(Vector<double, 3> {1., 2., 3.}),
      v(Vector<double, 3> {4., 5., 6.});
    // three scalar vector coefficients give dimension three
    EXPECT_EQ(geometry.dimension(), 3);
    // the inner product agrees with the independently expanded scalar sum
    EXPECT_DOUBLE_EQ(geometry.inner_product(point, u, v), 32.);
    const Vec step = geometry.retract(point, u, 2.);
    // affine steps preserve all native vector entries
    EXPECT_DOUBLE_EQ((step - 2. * u).norm(), 0.);
    const Vec zero = geometry.zero_tangent(point);
    // the allocated direction has the right extent and additive-identity value
    EXPECT_EQ(zero.size(), 3);
    // every coefficient of the zero direction vanishes
    EXPECT_DOUBLE_EQ(zero.norm(), 0.);
    const Vec large(Vector<double, 3> {1e200, 1e200, 0.});
    // the norm avoids overflow from squaring representable large coefficients
    EXPECT_DOUBLE_EQ(geometry.norm(point, large), std::hypot(1e200, 1e200));
    const EuclideanGeometry<Vector<float, 3>> fixed;
    // fixed native vector extents determine the geometry without runtime dimensions
    EXPECT_EQ(fixed.dimension(), 3);
}

// symmetric geometry uses full Frobenius contractions rather than the norm of packed coordinates
TEST(EuclideanGeometry, SymmetricOffDiagonalMetric) {
    const EuclideanGeometry<Sym> geometry;
    const Sym point, direction(Vector<double, 3> {0., 3., 0.});
    // a symmetric two-by-two matrix has three independent entries
    EXPECT_EQ(geometry.dimension(), 3);
    // the shared coefficient appears twice in the full Frobenius inner product
    EXPECT_DOUBLE_EQ(geometry.inner_product(point, direction, direction), 18.);
    // the metric norm includes both mirrored entries and agrees with the full matrix norm
    EXPECT_DOUBLE_EQ(geometry.norm(point, direction), direction.norm());
    const Sym step = geometry.retract(point, direction, 0.5);
    // affine retraction produces a native symmetric owner with the expected mirrored entry
    EXPECT_DOUBLE_EQ(step(0, 1), 1.5);
    // symmetry survives the affine update without reconstructing a packed vector
    EXPECT_DOUBLE_EQ(step(1, 0), step(0, 1));
    const auto spd = matrix_exp(step);
    // exponentiating the symmetric optimum produces a checked SPD with positive diagonal entries
    EXPECT_GT(spd(0, 0), 0.);
    // the two-by-two determinant is a second independent positive-definiteness check
    EXPECT_GT(spd(0, 0) * spd(1, 1) - spd(0, 1) * spd(1, 0), 0.);
}

// cached symmetric points retain ordinary Frobenius geometry and refresh spectra after affine retraction
TEST(EuclideanGeometry, CachedSymmetricRetraction) {
    using CachedSym = SymmetricMatrix<double, 2, Cache::Spectral>;
    const EuclideanGeometry<CachedSym> geometry;
    const CachedSym point(Vector<double, 3> {2., 0., 3.});
    const CachedSym direction(Vector<double, 3> {1., 2., -1.});
    point.cache();
    // the full Frobenius contraction counts the off-diagonal coefficient twice under either cache policy
    EXPECT_DOUBLE_EQ(geometry.inner_product(point, direction, direction), 10.);
    const auto step = geometry.retract(point, direction, .5);
    // affine retraction retains the selected public symmetric owner and its spectral policy
    static_assert(std::same_as<std::remove_cvref_t<decltype(step)>, CachedSym>);
    const Matrix<double, 2, 2> expected({2.5, 1., 1., 2.5});
    // the full returned matrix agrees with independently expanded affine coefficients
    EXPECT_DOUBLE_EQ((step - expected).norm(), 0.);
    const auto values = step.cache().eigenvalues();
    // fresh factors represent eigenvalues 1.5 and 3.5 regardless of their solver ordering
    EXPECT_NEAR(values[0] * values[1], 5.25, 1e-13);
}

// the product geometry preserves native batch shapes and solves every matrix in one trust-region call
TEST(EuclideanGeometry, JointSymmetricTrustRegion) {
    Batch initial(2), target(2);
    target[0] = Sym(Vector<double, 3> {.2, .3, .4});
    target[1] = Sym(Vector<double, 3> {.5, -.6, .7});
    const ProductGeometry geometry(EuclideanGeometry<Sym> {}, 2);
    Quadratic problem {target};
    const auto solution = RiemannianTrustRegion().optimize(problem, geometry, initial);
    // two symmetric two-by-two matrices contribute six independent coordinates
    EXPECT_EQ(geometry.dimension(), 6);
    // the outer solver certifies convergence in the product Frobenius metric
    ASSERT_TRUE(solution.converged());
    // the joint optimum retains one native symmetric matrix per initial batch element
    ASSERT_EQ(solution.point.size(), 2);
    for (std::size_t i = 0; i < target.size(); ++i) {
        // each full matrix agrees with the independent closed-form quadratic minimizer
        EXPECT_LT((solution.point[i] - target[i]).norm(), 1e-10);
    }
    const Batch combined = geometry.linear_combination(initial, 2., target, -1., target);
    // batch affine arithmetic preserves the same full Frobenius norm as its target oracle
    EXPECT_DOUBLE_EQ(geometry.norm(initial, combined), geometry.norm(initial, target));
}

// invalid dimensions and mismatched solver operands fail before coefficient contractions
TEST(EuclideanGeometry, RejectsInvalidShapes) {
    using Vec = Vector<double, Dynamic>;
    // a runtime vector space cannot have zero independent coordinates
    EXPECT_THROW(EuclideanGeometry<Vec>(0), std::invalid_argument);
    // runtime dimensions cannot contradict a fixed native matrix extent
    EXPECT_THROW(EuclideanGeometry<Sym>(3, 3), std::invalid_argument);
    // a dynamic symmetric coordinate space must remain square
    EXPECT_THROW((EuclideanGeometry<SymmetricMatrix<double, Dynamic>>(2, 3)), std::invalid_argument);
    // an empty product space cannot be passed to the optimization solver
    EXPECT_THROW(ProductGeometry(EuclideanGeometry<Sym> {}, 0), std::invalid_argument);
    const EuclideanGeometry<Vec> geometry(3);
    const Vec point(Vector<double, 3> {0., 0., 0.}), wrong(Vector<double, 2> {1., 2.});
    // debug operand checks reject a direction from a different vector space
    EXPECT_THROW(geometry.norm(point, wrong), std::invalid_argument);
    const ProductGeometry batch_geometry(EuclideanGeometry<Sym> {}, 2);
    const Batch batch(2), short_batch(1);
    // batch count mismatch is rejected before a local element is accessed
    EXPECT_THROW(batch_geometry.norm(batch, short_batch), std::invalid_argument);
}
}   // namespace
