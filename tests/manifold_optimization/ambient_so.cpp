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

#include <fdaPDE/manifold_optimization.h>
#include <gtest/gtest.h>

namespace {
using namespace fdapde;
using Geometry = manifold::SOGeometry<double, 3>;
using Dense = Matrix<double, 3, 3>;
using Skew = Geometry::Tangent;

/// @brief supplies ambient derivatives whose full-matrix owners differ from SO body tangents
struct RotationObjective {
    /// @brief records the ambient direction delivered to the Euclidean Hessian callback
    struct Workspace {
        Dense received;
    };
    Dense target;
    /// @brief evaluates the ambient Frobenius residual on a verified rotation
    double cost(const Geometry::Point& point, Workspace&) {
        const Dense residual(point - target);
        return .5 * residual.squared_norm();
    }
    /// @brief returns the full-matrix Euclidean residual without a tangent projection
    Dense egrad(const Geometry::Point& point, Workspace&) { return Dense(point - target); }
    /// @brief retains the supplied ambient velocity and applies the ambient identity Hessian
    Dense ehess(const Geometry::Point&, const Dense& direction, Workspace& workspace) {
        workspace.received = direction;
        return direction;
    }
};

/// @brief computes the ambient gradient of an anisotropic quadratic loss
template <RotationLike Q> Dense ambient_gradient(const Q& q, const Dense& a, const Dense& b) {
    const Dense residual(a * q - b);
    return Dense(a.transpose() * residual);
}

/// @brief compares every entry of native matrix results with the independent oracle
template <typename A, typename B> void expect_matrix(const A& a, const B& b, double tolerance) {
    // matching row counts ensure that the oracle covers every result row
    ASSERT_EQ(a.rows(), b.rows());
    // matching column counts ensure that the oracle covers every result column
    ASSERT_EQ(a.cols(), b.cols());
    for (int i = 0; i < a.rows(); ++i)
        for (int j = 0; j < a.cols(); ++j) {
            // each full-matrix coefficient agrees with the independently constructed reference
            EXPECT_NEAR(a(i, j), b(i, j), tolerance);
        }
}

// the converted Hessian differentiates a transported gradient and includes the nonzero curvature correction
TEST(SOAmbientDerivatives, CovariantHessianMatchesTransportedGradient) {
    const Geometry geometry;
    Skew base, direction;
    base(0, 1) = .4;
    base(0, 2) = -.23;
    base(1, 2) = .31;
    direction(0, 1) = .19;
    direction(0, 2) = -.37;
    direction(1, 2) = .14;
    const Geometry::Point point(rotation_exp(base));
    const Dense a({1.2, -.4, .3, .2, .7, -.1, -.5, .4, 1.1});
    const Dense b({.2, .8, -.6, -.3, .5, .4, .7, -.2, .1});
    const Dense gradient = ambient_gradient(point, a, b);
    const Dense velocity(point * direction), action(a.transpose() * a * velocity);
    const auto converted = geometry.euclidean_to_riemannian_hessian(point, gradient, action, direction);
    constexpr double step = 2e-5;
    const auto plus_delta = rotation_exp(direction, step), minus_delta = rotation_exp(direction, -step);
    const Geometry::Point plus(point * plus_delta), minus(point * minus_delta);
    const Dense plus_gradient = ambient_gradient(plus, a, b), minus_gradient = ambient_gradient(minus, a, b);
    const auto plus_body = geometry.euclidean_to_riemannian_gradient(plus, plus_gradient);
    const auto minus_body = geometry.euclidean_to_riemannian_gradient(minus, minus_gradient);
    const auto plus_half = rotation_exp(direction, step / 2), minus_half = rotation_exp(direction, -step / 2);
    const Dense plus_back(plus_half * plus_body * plus_half.transpose());
    const Dense minus_back(minus_half * minus_body * minus_half.transpose());
    const Dense difference((plus_back - minus_back) / (2 * step));
    // exact half-angle conjugation provides a transport oracle independent of the new conversion
    expect_matrix(converted, difference, 3e-10);
    const auto uncorrected = geometry.euclidean_to_riemannian_gradient(point, action);
    const Dense missing_connection(converted - uncorrected);
    // this anisotropic loss makes the embedded connection term observably nonzero
    EXPECT_GT(missing_connection.norm(), .1);
    const auto ambient_direction = geometry.to_ambient(point, direction);
    // the objective Hessian direction is the ambient velocity Q Omega rather than the body skew matrix
    expect_matrix(ambient_direction, velocity, 2e-15);
}

// cached owners and borrowed batch views preserve the same ambient-to-body Hessian conversion
TEST(SOAmbientDerivatives, CachedOwnersAndViewsAgree) {
    using Cached = RotationMatrix<double, 3, 3, RotationCache::Union<RotationCache::Schur, RotationCache::Log>>;
    const Geometry geometry;
    Skew base, direction;
    base(0, 1) = .29;
    base(1, 2) = -.18;
    direction(0, 2) = .43;
    const Cached point(rotation_exp(base));
    MatrixBatch<Cached> batch(1);
    batch[0] = point;
    const Dense gradient({.4, -.7, .1, .2, .9, -.3, -.6, .5, 1.2});
    const Dense action({-.1, .3, .5, .8, -.2, .6, .4, .7, -.9});
    const double saved = point.cache().distance();
    const auto owner = geometry.euclidean_to_riemannian_hessian(point, gradient, action, direction);
    const auto view = geometry.euclidean_to_riemannian_hessian(batch[0], gradient, action, direction);
    // Hessian conversion depends on coefficients and accepts the exact native rotation cache owner
    expect_matrix(view, owner, 2e-15);
    // read-only conversion leaves the checked logarithm cache unchanged
    EXPECT_DOUBLE_EQ(point.cache().distance(), saved);
}

// dynamic geometries reject incompatible ambient Hessian inputs before matrix products access coefficients
TEST(SOAmbientDerivatives, RejectsAmbientShapeMismatch) {
    const manifold::SOGeometry<double, Dynamic> geometry(3);
    const auto point = decltype(geometry)::Point::Identity(3);
    const auto direction = geometry.zero_tangent(point);
    const Matrix<double, Dynamic, Dynamic> valid(3, 3), invalid(4, 4);
    // an incompatible Euclidean gradient is rejected by the geometry's public shape contract
    EXPECT_THROW(geometry.euclidean_to_riemannian_hessian(point, invalid, valid, direction), std::invalid_argument);
    // an incompatible Euclidean Hessian action is rejected by the same public shape contract
    EXPECT_THROW(geometry.euclidean_to_riemannian_hessian(point, valid, invalid, direction), std::invalid_argument);
}

// the generic dispatcher converts full-matrix derivatives and passes an ambient velocity to the objective
TEST(SOAmbientDerivatives, DispatcherSeparatesAmbientAndBodyRepresentations) {
    const Geometry geometry;
    Skew base, direction;
    base(0, 1) = .41;
    base(1, 2) = -.27;
    direction(0, 2) = .38;
    direction(1, 2) = .19;
    const Geometry::Point point(rotation_exp(base));
    RotationObjective objective {Dense({.4, -.7, .1, .2, .9, -.3, -.6, .5, 1.2})};
    manifold::evaluation_context_t<RotationObjective, Geometry> context;
    const auto& gradient = manifold::evaluate_gradient(objective, geometry, point, context.current());
    const Dense ambient_gradient(point - objective.target), velocity(point * direction);
    const auto expected_gradient = geometry.euclidean_to_riemannian_gradient(point, ambient_gradient);
    // dispatch converts the full ambient residual to the required owning skew tangent
    expect_matrix(gradient, expected_gradient, 2e-15);
    const auto hessian = manifold::evaluate_hessian_vector(
      objective, geometry, point, direction, context.current().workspace(), &*context.current().euclidean_gradient());
    const auto expected = geometry.euclidean_to_riemannian_hessian(point, ambient_gradient, velocity, direction);
    // dispatch uses the geometry's connection-aware Hessian conversion with the cached full-matrix gradient
    expect_matrix(hessian, expected, 2e-15);
    // the objective receives Q Omega, including its generally nonskew ambient coefficients
    expect_matrix(context.current().workspace().received, velocity, 2e-15);
}

}   // namespace
