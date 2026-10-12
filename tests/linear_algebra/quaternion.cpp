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

#include <fdaPDE/dense_linear_algebra.h>
#include <gtest/gtest.h>

namespace {
using namespace fdapde;
using Quaternion = UnitQuaternion<double>;
using Rotation = RotationMatrix<double, 3>;
using Tangent = SkewSymmetricMatrix<double, 3>;
using FullCache = RotationCache::Union<RotationCache::Log, RotationCache::Schur>;

/// @brief builds a tangent from its geometric rotation vector using the cross-product sign convention
Tangent tangent(const Vector<double, 3>& axis, double angle) {
    Tangent result;
    result(0, 1) = -angle * axis[2];
    result(0, 2) = angle * axis[1];
    result(1, 2) = -angle * axis[0];
    return result;
}
/// @brief compares all matrix coefficients without invoking a rotation logarithm
double error(const auto& a, const auto& b) { return Matrix<double, Dynamic, Dynamic>(a - b).norm(); }
/// @brief embeds SO(3) into SO(4) to exercise the independent general-order Schur implementation
auto general_log(const Rotation& source) {
    Matrix<double, 4, 4> embedded {IdentityMatrix<double, 4, 4>()};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) embedded(i, j) = source(i, j);
    const auto full = rotation_log(RotationMatrix<double, 4>(embedded));
    Tangent result;
    for (int i = 0; i < 3; ++i)
        for (int j = i + 1; j < 3; ++j) result(i, j) = full(i, j);
    return result;
}

// quaternion normalization handles nonzero scales near overflow and underflow without changing the orientation
TEST(UnitQuaternion, SafeNormalizationAndInputValidation) {
    const Quaternion ordinary(1., -2., 3., -4.);
    const double huge = std::numeric_limits<double>::max() / 4;
    const double tiny = std::numeric_limits<double>::min();
    const Quaternion large(huge, -2 * huge, 3 * huge, -4 * huge);
    const Quaternion small(tiny, -2 * tiny, 3 * tiny, -4 * tiny);
    // scale normalization preserves every rotation coefficient for very large Hamilton inputs
    EXPECT_LT(error(ordinary.rotation(), large.rotation()), 3e-15);
    // scale normalization preserves every rotation coefficient for very small Hamilton inputs
    EXPECT_LT(error(ordinary.rotation(), small.rotation()), 3e-15);
    double norm = 0;
    for (double coefficient : ordinary.coefficients()) norm = std::hypot(norm, coefficient);
    // the scalar-first coefficients have Euclidean unit norm after construction
    EXPECT_NEAR(norm, 1., 3e-16);
    // opposite unit quaternion signs represent exactly the same spatial orientation
    EXPECT_LT(error(Quaternion(-1., 2., -3., 4.).rotation(), ordinary.rotation()), 3e-15);
    // default construction agrees with the independently constructed identity rotation
    EXPECT_LT(error(Quaternion().rotation(), Rotation::Identity()), 1e-15);
    // a zero quaternion has no orientation and is rejected at the public boundary
    EXPECT_THROW((Quaternion(0., 0., 0., 0.)), std::domain_error);
    // NaN coefficients cannot enter normalized orientation storage
    EXPECT_THROW((Quaternion(std::numeric_limits<double>::quiet_NaN(), 0., 0., 0.)), std::invalid_argument);
    // infinite coefficients cannot enter normalized orientation storage
    EXPECT_THROW((Quaternion(1., std::numeric_limits<double>::infinity(), 0., 0.)), std::invalid_argument);
}

// quaternion composition and inversion agree with noncommuting rotation matrix operations
TEST(UnitQuaternion, ReusableCompositionAndConversions) {
    const Vector<double, 3> first {.4, -.1, .7}, second {-.2, .5, .3};
    const auto a = Quaternion::from_rotation_vector(first), b = Quaternion::from_rotation_vector(second);
    const auto a_rotation = a.rotation(), b_rotation = b.rotation();
    // Hamilton multiplication uses the same application order as native matrix composition
    EXPECT_LT(error((a * b).rotation(), a_rotation * b_rotation), 2e-15);
    // quaternion inversion agrees with the transposed orientation matrix
    EXPECT_LT(error(a.inverse().rotation(), a_rotation.transpose()), 2e-15);
    // composing an orientation with its inverse restores the identity
    EXPECT_LT(error((a * a.inverse()).rotation(), Rotation::Identity()), 2e-15);
    const auto cached = a.rotation<FullCache>();
    // conversion from a verified cached rotation preserves its independent coefficients
    EXPECT_LT(error(Quaternion::from_rotation(cached).rotation(), cached), 2e-15);
    const RotationMatrix<double, Dynamic> runtime(cached);
    // runtime order-three rotations use the same reusable quaternion conversion
    EXPECT_LT(error(Quaternion::from_rotation(runtime).rotation(), cached), 2e-15);
    // a verified rotation of another order cannot represent a three-dimensional quaternion
    EXPECT_THROW(Quaternion::from_rotation(RotationMatrix<double, 2>::Identity()), std::invalid_argument);
    const Vector<double, Dynamic> short_vector(2);
    // axis-angle conversion rejects vectors whose shape is not three by one
    EXPECT_THROW(Quaternion::from_rotation_vector(short_vector), std::invalid_argument);
}

// specialized exponentials match the independent general Taylor algorithm and analytic axial rotations
TEST(SO3Kernels, ExponentialMatchesGeneralAndAnalyticOracles) {
    const std::array<Vector<double, 3>, 4> axes {
      Vector<double, 3> {1,      0,       0     },
        Vector<double, 3> {0,      1,       0     },
        Vector<double, 3> {0,      0,       1     },
      Vector<double, 3> {2. / 3, -1. / 3, 2. / 3}
    };
    for (const auto& axis : axes)
        for (double angle :
             {0., 1e-14, 1e-8, .3, 1.9, std::numbers::pi - 1e-10, std::numbers::pi + 1e-6, 12., 10000.}) {
            const auto body = tangent(axis, angle);
            const auto specialized = rotation_exp(body);
            const auto general = internals::skew_exponential_general(body);
            // scaling and squaring provides an independent exponential oracle including multiple revolutions
            EXPECT_LT(error(specialized, general), angle > 100 ? 3e-11 : 3e-14) << angle;
            const SkewSymmetricMatrix<double, Dynamic> dynamic_body(body);
            // runtime order-three dispatch agrees coefficientwise with the fixed-size kernel
            EXPECT_LT(error(rotation_exp(dynamic_body), specialized), 2e-15) << angle;
            // reversing the step agrees with transposition for every tested angle
            EXPECT_LT(error(rotation_exp(body, -1), specialized.transpose()), 2e-15) << angle;
        }
    Tangent axial;
    axial(0, 1) = -.7;
    const Matrix<double, 3, 3> analytic({std::cos(.7), -std::sin(.7), 0, std::sin(.7), std::cos(.7), 0, 0, 0, 1});
    // the independent planar sine-cosine formula fixes the geometric sign convention
    EXPECT_LT(error(rotation_exp(axial), analytic), 2e-15);
}

// principal logarithms preserve tiny angles and agree with the independent general Schur decomposition
TEST(SO3Kernels, LogarithmMatchesGeneralAndPrincipalAngles) {
    const std::array<Vector<double, 3>, 4> axes {
      Vector<double, 3> {1,      0,       0     },
        Vector<double, 3> {0,      1,       0     },
        Vector<double, 3> {0,      0,       1     },
      Vector<double, 3> {2. / 3, -1. / 3, 2. / 3}
    };
    for (const auto& axis : axes)
        for (double angle : {0., 1e-10, 1e-6, .3, 1.9, std::numbers::pi - 1e-5, std::numbers::pi + 1e-5, 6.2}) {
            const auto rotation = rotation_exp(tangent(axis, angle));
            const auto actual = rotation_log(rotation);
            const auto expected = tangent(axis, std::remainder(angle, 2 * std::numbers::pi));
            // scalar principal-angle reduction gives the exact branch for the known rotation axis
            EXPECT_LT(error(actual, expected), 3e-15) << angle;
            // embedding in order four forces the original Schur path as an independent logarithm oracle
            EXPECT_LT(error(actual, general_log(rotation)), 8e-11) << angle;
            // exponentiating the principal logarithm reconstructs all original rotation coefficients
            EXPECT_LT(error(rotation_exp(actual), rotation), 4e-15) << angle;
            const RotationMatrix<double, Dynamic> dynamic_rotation(rotation);
            // runtime and fixed order-three logarithms select the same principal tangent
            EXPECT_LT(error(rotation_log(dynamic_rotation), actual), 2e-15) << angle;
        }
}

// deterministic axis sweeps check all quaternion pivots and cancellation limits without a spectral oracle
TEST(SO3Kernels, AxisSweepAndSmallAngleAccuracy) {
    for (int sample = 0; sample < 1000; ++sample) {
        Vector<double, 3> axis {std::sin(.73 * sample + .2), std::cos(.37 * sample + .5), std::sin(.19 * sample + 1.)};
        axis /= axis.norm();
        for (double angle : {1e-16, 1e-12, .001, 2., std::numbers::pi - 1e-9, std::numbers::pi + 1e-9}) {
            const auto rotation = rotation_exp(tangent(axis, angle));
            const auto principal = rotation_log(rotation);
            const auto expected = tangent(axis, std::remainder(angle, 2 * std::numbers::pi));
            // the known axis and principal-angle reduction independently certify every pivot branch
            EXPECT_LT(error(principal, expected), 8e-15) << sample << " " << angle;
            // principal logarithms preserve relative accuracy when the full angle is tiny
            if (angle < 1e-10) EXPECT_LT(error(principal, expected), 2e-15 * angle) << sample << " " << angle;
        }
    }
}

// the SO(3) specialization preserves exact, near and unresolved cut-locus diagnostics
TEST(SO3Kernels, CutLocusDiagnosticsAndDeterministicMinimum) {
    const Vector<double, 3> axis {0, 0, 1};
    const auto ordinary = rotation_exp(tangent(axis, .2));
    // an ordinary principal rotation has a derivative-regular logarithm
    EXPECT_EQ(minimum_rotation_log(ordinary).diagnostics.status, RotationLogStatus::Regular);
    const auto near = rotation_exp(tangent(axis, std::numbers::pi - 1e-9));
    const auto near_log = minimum_rotation_log(near);
    // the resolved near-pi branch remains unique but is explicitly marked derivative-sensitive
    EXPECT_EQ(near_log.diagnostics.status, RotationLogStatus::NearCut);
    // the stored cut gap follows the analytical angle difference rather than a cosine cancellation
    EXPECT_NEAR(near_log.diagnostics.cut_gap, 1e-9, 1e-15);
    // the ordinary value logarithm remains available for a resolved near-cut branch
    EXPECT_NO_THROW(rotation_log(near));
    const auto unresolved = rotation_exp(tangent(axis, std::numbers::pi - 1e-15));
    // a nonzero sine below the branch resolution threshold must not be certified unique
    EXPECT_EQ(minimum_rotation_log(unresolved).diagnostics.status, RotationLogStatus::Unresolved);
    // ordinary logarithms reject a numerically unresolved branch
    EXPECT_THROW(rotation_log(unresolved), std::domain_error);
    const Rotation exact(Matrix<double, 3, 3>({-1, 0, 0, 0, -1, 0, 0, 0, 1}));
    const auto selected = minimum_rotation_log(exact);
    // a symmetric exact half-turn retains its explicit nonunique status
    EXPECT_EQ(selected.diagnostics.status, RotationLogStatus::Ambiguous);
    // repeated explicit minimum selections use the same deterministic half-turn axis
    EXPECT_EQ(error(selected.tangent, minimum_rotation_log(exact).tangent), 0.);
    // the explicitly selected tangent still reconstructs the exact half-turn
    EXPECT_LT(error(rotation_exp(selected.tangent), exact), 1e-15);
    // the ordinary logarithm does not silently select the explicit minimum branch
    EXPECT_THROW(rotation_log(exact), std::domain_error);
    // distance remains defined at the cut locus with the full-Frobenius normalization
    EXPECT_NEAR(rotation_distance_identity(exact), std::sqrt(2.) * std::numbers::pi, 2e-15);
}

// selected logarithm and Schur caches preserve the fast-path results through batch replacement
TEST(SO3Kernels, CachePoliciesAndReplacement) {
    const auto value = rotation_exp(tangent(Vector<double, 3> {2. / 3, -1. / 3, 2. / 3}, .9));
    const auto logarithm = rotation_log(value);
    const RotationMatrix<double, 3, RotationCache::Log> log_cached(value);
    const RotationMatrix<double, 3, RotationCache::Schur> schur_cached(value);
    const RotationMatrix<double, 3, FullCache> full_cached(value);
    // a logarithm-only cache retains the same unique tangent as the uncached kernel
    EXPECT_LT(error(rotation_log(log_cached), logarithm), 2e-15);
    // a Schur-only cache reconstructs the same unique tangent from its stored rotation plane
    EXPECT_LT(error(rotation_log(schur_cached), logarithm), 2e-15);
    // the combined cache publishes matching logarithm coefficients and diagnostics
    EXPECT_LT(error(rotation_log(full_cached), logarithm), 2e-15);
    const RotationMatrix<double, Dynamic, FullCache> dynamic_cached(value);
    // runtime cached storage shares the fixed-size SO(3) plane representation
    EXPECT_LT(error(rotation_log(dynamic_cached), logarithm), 2e-15);
    // cached distances agree with the known full-Frobenius angular distance
    EXPECT_NEAR(rotation_distance_identity(full_cached), std::sqrt(2.) * .9, 1e-15);
    MatrixBatch<RotationMatrix<double, 3, FullCache>> batch(1);
    batch[0] = value;
    const RotationMatrix<double, 3, FullCache> old(batch[0]);
    batch[0] = rotation_exp(tangent(Vector<double, 3> {1, 0, 0}, -.4));
    // replacing a batch slot refreshes its logarithm together with its matrix coefficients
    EXPECT_LT(error(rotation_log(batch[0]), tangent(Vector<double, 3> {1, 0, 0}, -.4)), 1e-15);
    // an earlier owner retains independent coefficients and a valid previous logarithm
    EXPECT_LT(error(rotation_log(old), logarithm), 2e-15);
    const Rotation exact(Matrix<double, 3, 3>({-1, 0, 0, 0, -1, 0, 0, 0, 1}));
    batch[0] = exact;
    // ambiguous cache replacement retains branch diagnostics rather than a stale regular logarithm
    EXPECT_EQ(batch[0].cache().diagnostics().status, RotationLogStatus::Ambiguous);
    // a cached ambiguous logarithm rejects implicit branch selection after replacement
    EXPECT_THROW(rotation_log(batch[0]), std::domain_error);
    // explicit minimum selection remains available through a combined ambiguous cache
    EXPECT_LT(error(rotation_exp(minimum_rotation_log(batch[0]).tangent), exact), 1e-15);
}

// fast exponentials preserve finite-input checks and instantiate natively for single precision
TEST(SO3Kernels, InputValidationAndFloatInstantiation) {
    Tangent invalid;
    invalid(0, 1) = std::numeric_limits<double>::quiet_NaN();
    // a nonfinite tangent cannot publish a verified rotation
    EXPECT_THROW(rotation_exp(invalid), std::domain_error);
    invalid(0, 1) = std::numeric_limits<double>::infinity();
    // an infinite tangent cannot publish a verified rotation
    EXPECT_THROW(rotation_exp(invalid), std::domain_error);
    invalid(0, 1) = .1;
    // a nonfinite step is rejected before evaluating quaternion coefficients
    EXPECT_THROW(rotation_exp(invalid, std::numeric_limits<double>::quiet_NaN()), std::invalid_argument);
    invalid(0, 1) = std::numeric_limits<double>::max();
    // overflow of finite tangent coefficients retains the existing norm-overflow rejection
    EXPECT_THROW(rotation_exp(invalid, 2.), std::domain_error);
    const Vector<float, 3> vector {.1f, -.2f, .3f};
    const auto quaternion = UnitQuaternion<float>::from_rotation_vector(vector);
    SkewSymmetricMatrix<float, 3> body;
    body(0, 1) = -.3f;
    body(0, 2) = -.2f;
    body(1, 2) = -.1f;
    // the native float quaternion exponential agrees with the native float matrix API
    EXPECT_LT(error(quaternion.rotation(), rotation_exp(body)), 3e-7);
    // the float logarithm retains the original body tangent within scalar precision
    EXPECT_LT(error(rotation_log(rotation_exp(body)), body), 3e-7);
}
}   // namespace
