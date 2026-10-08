// SPDX-License-Identifier: GPL-3.0-or-later
#include <fdaPDE/manifold_optimization.h>
#include <gtest/gtest.h>

namespace {
using namespace fdapde;
using namespace fdapde::manifold;

/// @brief isolates derivative dispatch with a constant nonidentity scalar metric
struct ScaledChart {
    using Point = double;
    using Tangent = double;
    /// @brief counts the single scalar coordinate
    std::size_t dimension() const { return 1; }
    /// @brief scales the ambient inner product by four
    double inner_product(double, double u, double v) const { return 4 * u * v; }
    /// @brief measures the scaled tangent length
    double norm(double, double u) const { return 2 * std::abs(u); }
    /// @brief retains every scalar direction
    double project(double, double u) const { return u; }
    /// @brief creates the additive identity
    double zero_tangent(double) const { return 0; }
    /// @brief combines tangent directions without changing coordinates
    double linear_combination(double, double a, double u, double b, double v) const { return a * u + b * v; }
    /// @brief follows the flat scalar chart
    double retract(double x, double u, double step) const { return x + step * u; }
    /// @brief applies the inverse constant metric to the ambient gradient
    double euclidean_to_riemannian_gradient(double, double u) const { return u / 4; }
    /// @brief applies the inverse constant metric to the ambient Hessian action
    double euclidean_to_riemannian_hessian(double, double, double h, double) const { return h / 4; }
};

/// @brief supplies independent optional intrinsic derivatives and counts every dispatch route
template <bool IntrinsicGradient = false, bool IntrinsicHessian = false> struct Quadratic {
    /// @brief reserves candidate-local storage for an objective without intermediates
    struct Workspace { };
    int grad_calls = 0, hess_calls = 0, egrad_calls = 0, ehess_calls = 0;
    /// @brief evaluates a convex quadratic with its known minimum at two
    double cost(double x, Workspace&) { return .5 * (x - 2) * (x - 2); }
    /// @brief provides the exact intrinsic gradient of the scaled chart
    double grad(double x, Workspace&)
        requires IntrinsicGradient
    {
        ++grad_calls;
        return (x - 2) / 4;
    }
    /// @brief provides the exact intrinsic Hessian action of the scaled chart
    double hess(double, double v, Workspace&)
        requires IntrinsicHessian
    {
        ++hess_calls;
        return v / 4;
    }
    /// @brief provides the ambient gradient before metric conversion
    double egrad(double x, Workspace&) {
        ++egrad_calls;
        return x - 2;
    }
    /// @brief provides the ambient Hessian action before metric conversion
    double ehess(double, double v, Workspace&) {
        ++ehess_calls;
        return v;
    }
};

/// @brief checks every intrinsic and ambient combination against an independent quadratic optimum
template <bool Gradient, bool Hessian> void check_dispatch() {
    Quadratic<Gradient, Hessian> objective;
    const auto result = RiemannianTrustRegion().optimize(objective, ScaledChart {}, -3.);
    // the scaled chart still reaches the unique closed-form ambient minimum
    ASSERT_TRUE(result.converged());
    // the optimizer preserves the objective when choosing a derivative representation
    EXPECT_NEAR(result.point, 2., 1e-12);
    // intrinsic gradients take precedence whenever their method is supplied
    EXPECT_EQ(objective.grad_calls > 0, Gradient);
    // intrinsic Hessians take precedence independently of gradient representation
    EXPECT_EQ(objective.hess_calls > 0, Hessian);
    // ambient Hessian actions are requested exactly when the intrinsic action is absent
    EXPECT_EQ(objective.ehess_calls > 0, !Hessian);
    // fully intrinsic problems never evaluate unused ambient gradients
    EXPECT_EQ(objective.egrad_calls > 0, !Gradient || !Hessian);
}

// short intrinsic method names override ambient derivatives, including either mixed route
TEST(AmbientObjective, DerivativePrecedenceAndMixedRoutes) {
    // the fully ambient route requires both conversions
    check_dispatch<false, false>();
    // an intrinsic gradient still supplies its ambient counterpart to Hessian conversion
    check_dispatch<true, false>();
    // an intrinsic Hessian does not require ambient second-order information
    check_dispatch<false, true>();
    // both intrinsic derivatives suppress every ambient derivative evaluation
    check_dispatch<true, true>();
}

/// @brief supplies intrinsic derivatives while exposing an unusable ambient callback
struct IntrinsicQuadratic {
    /// @brief reserves candidate-local storage for intrinsic callbacks
    struct Workspace { };
    /// @brief evaluates the same quadratic without an ambient derivative contract
    double cost(double x, Workspace&) { return .5 * (x - 2) * (x - 2); }
    /// @brief supplies the intrinsic gradient in the scaled chart
    double grad(double x, Workspace&) { return (x - 2) / 4; }
    /// @brief supplies the intrinsic Hessian action in the scaled chart
    double hess(double, double v, Workspace&) { return v / 4; }
    /// @brief detects accidental use of an invalid ambient gradient
    void egrad(double, Workspace&) {
        // a void ambient callback must be ignored when a valid intrinsic gradient is available
        ADD_FAILURE() << "intrinsic gradient must take precedence";
    }
};

/// @brief supplies an intrinsic gradient and ambient Hessian without its required ambient gradient
struct MissingAmbientGradient : Quadratic<true, false> {
    /// @brief makes the inherited ambient gradient unavailable to Hessian conversion
    double egrad(double, Workspace&) = delete;
};

// ambient Hessian conversion requires its gradient independently of intrinsic first-order derivatives
TEST(AmbientObjective, AmbientHessianRequiresAmbientGradient) {
    // an intrinsic gradient remains sufficient for first-order optimization without an ambient counterpart
    static_assert(FirstOrderProblem<MissingAmbientGradient, ScaledChart>);
    // an ambient Hessian action needs egrad even when grad already supplies the first derivative
    static_assert(!SecondOrderProblem<MissingAmbientGradient, ScaledChart>);
}

/// @brief exposes a combined gradient to test explicit intrinsic precedence and direct cache helpers
template <bool Explicit> struct CombinedQuadratic : Quadratic<Explicit, true> {
    /// @brief returns the joint intrinsic result, with an intentionally distinct unused gradient
    std::pair<double, double> cost_gradient(double x, typename Quadratic<Explicit, true>::Workspace& workspace) {
        return {this->cost(x, workspace), Explicit ? 99. : (x - 2) / 4};
    }
};

// combined-only objectives remain usable and explicit intrinsic gradients fill direct helper caches
TEST(AmbientObjective, CombinedIntrinsicGradient) {
    CombinedQuadratic<false> combined;
    const auto solution = RiemannianTrustRegion().optimize(combined, ScaledChart {}, -3.);
    // the joint intrinsic gradient is sufficient even without a separate intrinsic callback
    ASSERT_TRUE(solution.converged());
    // combined-only evaluation reaches the known minimum without ambient conversion
    EXPECT_EQ(combined.egrad_calls, 0);
    CombinedQuadratic<true> explicit_gradient;
    evaluation_context_t<CombinedQuadratic<true>, ScaledChart> context;
    evaluate_cost_gradient(explicit_gradient, ScaledChart {}, -3., context.current());
    // the direct joint helper retains its contract of filling both cached results
    ASSERT_TRUE(context.current().gradient());
    // an explicit intrinsic callback takes precedence over the distinct combined gradient
    EXPECT_DOUBLE_EQ(*context.current().gradient(), -1.25);
}

// intrinsic callbacks remain valid when the unused ambient derivative has an invalid return type
TEST(AmbientObjective, IntrinsicMethodsIgnoreInvalidAmbientDerivative) {
    IntrinsicQuadratic objective;
    const auto result = RiemannianTrustRegion().optimize(objective, ScaledChart {}, -3.);
    // valid intrinsic methods retain the exact quadratic convergence certificate
    ASSERT_TRUE(result.converged());
    // ignored ambient methods do not change the intrinsic solution
    EXPECT_NEAR(result.point, 2., 1e-12);
}

// gradient and Hessian conversions share one ambient gradient within each candidate generation
TEST(AmbientObjective, CachePromotionAndInvalidation) {
    Quadratic<> objective;
    ScaledChart geometry;
    evaluation_context_t<Quadratic<>, ScaledChart> context;
    const auto& gradient = evaluate_gradient(objective, geometry, -3., context.current());
    SteihaugTruncatedCG subproblem;
    for (int i = 0; i < 2; ++i)
        subproblem.solve(
          objective, geometry, -3., gradient, .1, context.current().workspace(),
          &*context.current().euclidean_gradient());
    // repeated subproblems at one point reuse the ambient gradient cached by first-order evaluation
    EXPECT_EQ(objective.egrad_calls, 1);
    evaluate_gradient(objective, geometry, 1., context.trial());
    context.promote_trial();
    evaluate_gradient(objective, geometry, 1., context.current());
    // accepted trial promotion carries both gradient representations without reevaluation
    EXPECT_EQ(objective.egrad_calls, 2);
    context.reset_trial();
    evaluate_gradient(objective, geometry, 0., context.trial());
    context.reset_trial();
    // a rejected trial clears its own ambient cache
    EXPECT_FALSE(context.trial().euclidean_gradient());
    // rejecting a trial preserves the accepted point's ambient cache and gradient
    EXPECT_DOUBLE_EQ(*context.current().euclidean_gradient(), -1.);
    context.reset_current();
    evaluate_gradient(objective, geometry, 0., context.current());
    // binding the current slot to a different point forces a fresh ambient gradient
    EXPECT_EQ(objective.egrad_calls, 4);
}

// first-order solvers use the same ambient conversion contract as trust regions
TEST(AmbientObjective, SteepestDescentAmbientGradient) {
    Quadratic<> objective;
    SteepestDescentOptions options;
    options.gradient_tolerance = 1e-9;
    const auto result = RiemannianSteepestDescent(options).optimize(objective, ScaledChart {}, -3.);
    // Armijo descent terminates at the known convex minimum using the converted gradient
    ASSERT_TRUE(result.converged());
    // the metric changes step lengths while preserving the optimum
    EXPECT_NEAR(result.point, 2., 1e-7);
    // a first-order algorithm does not request second-order ambient information
    EXPECT_EQ(objective.ehess_calls, 0);
}

using Sym = SymmetricMatrix<double, 2>;
using SPD = SPDMatrix<double, 2>;
using SymBatch = MatrixBatch<Sym>;
using SPDBatch = MatrixBatch<SPD>;

/// @brief supplies a native ambient quadratic for dense and packed symmetric Euclidean points
template <typename Point> struct NativeResidual {
    /// @brief reserves candidate-local storage for the native residual
    struct Workspace { };
    Point target;
    /// @brief evaluates the full Frobenius residual including both mirrored entries
    double cost(const Point& point, Workspace&) { return .5 * (point - target).squared_norm(); }
    /// @brief returns an owning ambient residual
    Point egrad(const Point& point, Workspace&) { return Point(point - target); }
    /// @brief applies the constant ambient Hessian
    Point ehess(const Point&, const Point& direction, Workspace&) { return direction; }
};

// native dense vectors and packed symmetric matrices share the constant-metric ambient path
TEST(AmbientObjective, NativeEuclideanPoints) {
    NativeResidual<Sym> symmetric {Sym(Vector<double, 3> {.3, .4, -.2})};
    const auto symmetric_result = RiemannianTrustRegion().optimize(symmetric, EuclideanGeometry<Sym> {}, Sym {});
    // the full Frobenius metric returns the known residual minimum including its off-diagonal coefficient
    ASSERT_TRUE(symmetric_result.converged());
    // packed storage does not halve or double the native ambient optimum
    EXPECT_LT((symmetric_result.point - symmetric.target).norm(), 1e-12);
    using Vec = Vector<double, 3>;
    NativeResidual<Vec> vector {
      Vec {1., -.2, .3}
    };
    const auto vector_result = RiemannianTrustRegion().optimize(vector, EuclideanGeometry<Vec> {}, Vec {});
    // dense vector owners reach the same analytic Euclidean quadratic minimum
    ASSERT_TRUE(vector_result.converged());
    // the ambient action remains the identity for ordinary native vectors
    EXPECT_LT((vector_result.point - vector.target).norm(), 1e-12);
}

/// @brief couples both SPD nodes through the sum of their ambient residuals
struct CoupledSPD {
    /// @brief reserves candidate-local storage for a coupled polynomial objective
    struct Workspace { };
    const SPDBatch& target;
    /// @brief combines positive residual energies with a nonseparable sum penalty
    double cost(const SPDBatch& point, Workspace&) {
        const Sym r0(point[0] - target[0]), r1(point[1] - target[1]);
        const Sym sum(r0 + r1);
        return .5 * (r0.squared_norm() + r1.squared_norm()) + .25 * sum.squared_norm();
    }
    /// @brief includes both nodes in each ambient gradient component
    SymBatch egrad(const SPDBatch& point, Workspace&) {
        const Sym r0(point[0] - target[0]), r1(point[1] - target[1]);
        const Sym sum(r0 + r1);
        SymBatch result(2);
        result[0] = r0 + .5 * sum;
        result[1] = r1 + .5 * sum;
        return result;
    }
    /// @brief retains mixed-node Hessian blocks before elementwise metric conversion
    SymBatch ehess(const SPDBatch&, const SymBatch& direction, Workspace&) {
        const Sym sum(direction[0] + direction[1]);
        SymBatch result(2);
        result[0] = direction[0] + .5 * sum;
        result[1] = direction[1] + .5 * sum;
        return result;
    }
};

// product conversion retains cross-node Hessian terms and reaches the common coupled SPD minimum
TEST(AmbientObjective, CoupledSPDProduct) {
    SPDBatch target(2);
    target[0] = SPD(Vector<double, 3> {1.8, .3, 1.2});
    target[1] = SPD(Vector<double, 3> {1.1, -.2, 2.});
    CoupledSPD objective {target};
    CoupledSPD::Workspace workspace;
    SymBatch direction(2);
    direction[0] = Sym(Vector<double, 3> {.2, .1, -.3});
    const auto ambient_hessian = objective.ehess(target, direction, workspace);
    const Sym expected(.5 * direction[0]);
    // a direction at the first node produces the analytic mixed Hessian component at the second
    EXPECT_LT((ambient_hessian[1] - expected).norm(), 1e-14);
    const ProductGeometry geometry(LogEuclideanGeometry<SPD> {}, 2);
    const auto result = RiemannianTrustRegion().optimize(objective, geometry, SPDBatch(2));
    // the joint positive quadratic has its unique feasible SPD minimum at the given target
    ASSERT_TRUE(result.converged());
    // neither node is solved independently or loses its coupling during conversion
    EXPECT_LT((result.point[0] - target[0]).norm() + (result.point[1] - target[1]).norm(), 1e-7);
}
}   // namespace
