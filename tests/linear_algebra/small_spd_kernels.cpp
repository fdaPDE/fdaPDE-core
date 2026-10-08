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

#include <array>
#include <cmath>
#include <limits>
#include <type_traits>

namespace {

using namespace fdapde;

template <typename Scalar, int Order> using fixed_matrix = Matrix<Scalar, Order, Order>;
template <typename Scalar, int Order> using fixed_symmetric = SymmetricMatrix<Scalar, Order>;
template <typename Scalar, int Order> using fixed_spd = SPDMatrix<Scalar, Order>;
template <typename Scalar> using dynamic_matrix = Matrix<Scalar, Dynamic, Dynamic>;
template <typename Scalar> using dynamic_symmetric = SymmetricMatrix<Scalar, Dynamic>;
template <typename Scalar> using dynamic_spd = SPDMatrix<Scalar, Dynamic>;

/// @brief creates a fixed matrix from row-major test coefficients
template <typename Scalar, int Order>
fixed_matrix<Scalar, Order> make_matrix(const std::array<Scalar, Order * Order>& coefficients) {
    fixed_matrix<Scalar, Order> result;
    for (int row = 0; row < Order; ++row) {
        for (int col = 0; col < Order; ++col) result(row, col) = coefficients[row * Order + col];
    }
    return result;
}

/// @brief measures coefficient error relative to the expected matrix scale
template <typename Actual, typename Expected> double relative_error(const Actual& actual, const Expected& expected) {
    const dynamic_matrix<double> actual_dense(actual);
    const dynamic_matrix<double> expected_dense(expected);
    return (actual_dense - expected_dense).norm() / expected_dense.norm();
}

/// @brief checks eigenpair reconstruction, residual and orthogonality identities
template <typename Symmetric, typename Decomposition>
void expect_evd_residuals(const Symmetric& source, const Decomposition& evd, double tolerance) {
    const dynamic_matrix<double> dense_source(source);
    const dynamic_matrix<double> vectors(evd.eigenvectors());
    dynamic_matrix<double> diagonal(source.rows(), source.cols());
    dynamic_matrix<double> identity(source.rows(), source.cols());
    diagonal.set_zero();
    identity.set_zero();
    for (int index = 0; index < source.rows(); ++index) {
        diagonal(index, index) = evd.eigenvalues()[index];
        identity(index, index) = 1.0;
    }
    // eigenpairs reconstruct the original matrix at a scale-relative tolerance
    EXPECT_LT((vectors * diagonal * vectors.transpose() - dense_source).norm() / dense_source.norm(), tolerance);
    // the returned eigenvector columns form an orthonormal basis
    EXPECT_LT((vectors.transpose() * vectors - identity).norm(), tolerance);
    // each residual is bounded relative to the source matrix scale
    EXPECT_LT((dense_source * vectors - vectors * diagonal).norm() / dense_source.norm(), tolerance);
}

/// @brief compares fixed small-matrix spectral operations with the dynamic generic path
template <typename Scalar, int Order>
void compare_fixed_and_dynamic_paths(const fixed_matrix<Scalar, Order>& dense, double tolerance) {
    const auto fixed_view = dense.template as_symmetric<Lower>();
    const fixed_symmetric<Scalar, Order> fixed_symmetric_value(fixed_view);
    const dynamic_matrix<Scalar> dynamic_dense(dense);
    const auto dynamic_view = dynamic_dense.template as_symmetric<Lower>();
    const dynamic_symmetric<Scalar> dynamic_symmetric_value(dynamic_view);
    const EVD fixed_evd(fixed_symmetric_value);
    const EVD dynamic_evd(dynamic_symmetric_value);
    // the fixed specialization reports a completed factorization
    EXPECT_TRUE(fixed_evd.computed());
    // the dynamic generic path reports a completed factorization
    EXPECT_TRUE(dynamic_evd.computed());
    // the fixed specialization satisfies eigenpair and orthogonality identities
    expect_evd_residuals(fixed_symmetric_value, fixed_evd, tolerance);
    // the dynamic generic oracle satisfies the same eigenpair and orthogonality identities
    expect_evd_residuals(dynamic_symmetric_value, dynamic_evd, tolerance);

    const fixed_spd<Scalar, Order> fixed_point(dense);
    // the generic oracle owns the same coefficients with runtime extents
    const dynamic_spd<Scalar> dynamic_point {dynamic_matrix<Scalar>(dense)};
    const auto fixed_log = matrix_log(fixed_point);
    const auto dynamic_log = matrix_log(dynamic_point);
    const auto fixed_exp = matrix_exp(fixed_log);
    const auto dynamic_exp = matrix_exp(dynamic_log);
    // fixed logarithms agree with the dynamic EVD path coefficientwise
    EXPECT_LT(relative_error(fixed_log, dynamic_log), tolerance);
    // fixed exponentials agree with the dynamic generic path coefficientwise
    EXPECT_LT(relative_error(fixed_exp, dynamic_exp), tolerance * 4.0);
    // exponentiating the fixed logarithm recovers the SPD input at relative scale
    EXPECT_LT(relative_error(fixed_exp, dense), tolerance * 4.0);

    const auto fixed_root = matrix_sqrt(fixed_point);
    const auto dynamic_root = matrix_sqrt(dynamic_point);
    // fixed principal roots agree with the dynamic EVD path
    EXPECT_LT(relative_error(fixed_root, dynamic_root), tolerance * 4.0);
    // the fixed principal root squares back to its input
    EXPECT_LT(relative_error(fixed_root * fixed_root, dense), tolerance * 8.0);
}

/// @brief checks deterministic spectra and numerical boundary cases for one scalar and order
template <typename Scalar, int Order> void check_spectrum_cases() {
    const double tolerance = std::is_same_v<Scalar, float> ? 2.0e-5 : 2.0e-12;
    // diagonal spectra exercise the zero-pivot path against an independent dynamic EVD
    compare_fixed_and_dynamic_paths(
      make_matrix<Scalar, Order>([] {
          std::array<Scalar, Order * Order> values {};
          for (int index = 0; index < Order; ++index) values[index * Order + index] = Scalar(index + 2);
          return values;
      }()),
      tolerance);

    if constexpr (Order == 2) {
        // a rotated spectrum gives the fixed two-dimensional kernel a nonzero Jacobi pivot
        compare_fixed_and_dynamic_paths(
          make_matrix<Scalar, 2>({Scalar(4), Scalar(1), Scalar(1), Scalar(3)}), tolerance);
        // equal eigenvalues exercise exact repetition without requiring a unique eigenbasis
        compare_fixed_and_dynamic_paths(
          make_matrix<Scalar, 2>({Scalar(2), Scalar(0), Scalar(0), Scalar(2)}), tolerance);
    } else {
        // this rotated positive spectrum activates all three Jacobi pivot positions
        compare_fixed_and_dynamic_paths(
          make_matrix<Scalar, 3>(
            {Scalar(4), Scalar(1), Scalar(0.5), Scalar(1), Scalar(3), Scalar(-0.25), Scalar(0.5), Scalar(-0.25),
             Scalar(2)}),
          tolerance);
        // a repeated eigenspace checks reconstruction without relying on eigenvector ordering
        compare_fixed_and_dynamic_paths(
          make_matrix<Scalar, 3>(
            {Scalar(2), Scalar(0), Scalar(0), Scalar(0), Scalar(2), Scalar(0), Scalar(0), Scalar(0), Scalar(5)}),
          tolerance);
    }

    const Scalar gap = std::is_same_v<Scalar, float> ? Scalar(1.0e-3) : Scalar(1.0e-6);
    fixed_matrix<Scalar, Order> near_repeated;
    near_repeated.set_zero();
    near_repeated(0, 0) = Scalar(2);
    near_repeated(1, 1) = Scalar(2) + gap;
    near_repeated(0, 1) = near_repeated(1, 0) = gap / Scalar(2);
    for (int index = 2; index < Order; ++index) near_repeated(index, index) = Scalar(index + 2);
    // a small but resolvable eigengap exercises rotations near a repeated eigenspace
    compare_fixed_and_dynamic_paths(near_repeated, tolerance);

    fixed_matrix<Scalar, Order> barely_off_diagonal;
    barely_off_diagonal.set_zero();
    for (int index = 0; index < Order; ++index) barely_off_diagonal(index, index) = Scalar(index + 2);
    barely_off_diagonal(0, 1) = barely_off_diagonal(1, 0) =
      std::numeric_limits<Scalar>::epsilon() * static_cast<Scalar>(Order) / Scalar(2);
    // pivots at or below dimension times epsilon are safely omitted within the residual tolerance
    compare_fixed_and_dynamic_paths(barely_off_diagonal, tolerance);

    const Scalar scale = std::is_same_v<Scalar, float> ? Scalar(1.0e15) : Scalar(1.0e150);
    const Scalar inverse_scale = Scalar(1) / scale;
    // a well-conditioned rotated shape supports uniform extreme-scale checks
    const fixed_matrix<Scalar, Order> shape = make_matrix<Scalar, Order>([] {
        std::array<Scalar, Order * Order> values {};
        for (int index = 0; index < Order; ++index) values[index * Order + index] = Scalar(index + 2);
        if constexpr (Order == 2) {
            values[1] = values[2] = Scalar(0.25);
        } else {
            values[1] = values[3] = Scalar(0.25);
            values[2] = values[6] = Scalar(-0.125);
            values[5] = values[7] = Scalar(0.125);
        }
        return values;
    }());
    const fixed_matrix<Scalar, Order> large = shape * scale;
    const fixed_matrix<Scalar, Order> tiny = shape * inverse_scale;
    // uniform large scaling keeps the normalized EVD and spectral operations finite
    compare_fixed_and_dynamic_paths(large, tolerance * 16.0);
    // uniform tiny scaling keeps the normalized EVD and spectral operations finite
    compare_fixed_and_dynamic_paths(tiny, tolerance * 16.0);

    fixed_matrix<Scalar, Order> accepted;
    accepted.set_zero();
    for (int index = 0; index < Order - 1; ++index) accepted(index, index) = Scalar(1);
    accepted(Order - 1, Order - 1) = Scalar(512 * Order) * std::numeric_limits<Scalar>::epsilon();
    // eigenvalues above the documented relative positivity threshold are accepted by fixed storage
    EXPECT_NO_THROW((fixed_spd<Scalar, Order>(accepted)));
    // the same accepted spectrum passes checked dynamic SPD construction
    EXPECT_NO_THROW((dynamic_spd<Scalar>(dynamic_matrix<Scalar>(accepted))));
    accepted(Order - 1, Order - 1) = Scalar(16 * Order) * std::numeric_limits<Scalar>::epsilon();
    // eigenvalues below the relative positivity threshold are rejected by fixed storage
    EXPECT_THROW((fixed_spd<Scalar, Order>(accepted)), std::domain_error);
    // the same rejected spectrum fails checked dynamic SPD construction
    EXPECT_THROW((dynamic_spd<Scalar>(dynamic_matrix<Scalar>(accepted))), std::domain_error);
}

/// @brief compares public logarithm and exponential Frechet actions with central differences
template <int Order> void check_frechet_finite_differences() {
    using Scalar = double;
    const double tolerance = 2.0e-8;
    const fixed_matrix<Scalar, Order> point_matrix = [] {
        // the rotated SPD point keeps the finite-difference perturbations well inside the domain
        if constexpr (Order == 2)
            return make_matrix<double, 2>({4.0, 1.0, 1.0, 3.0});
        else
            return make_matrix<double, 3>({4.0, 1.0, 0.5, 1.0, 3.0, -0.25, 0.5, -0.25, 2.0});
    }();
    fixed_symmetric<Scalar, Order> direction;
    for (int row = 0; row < Order; ++row) {
        for (int col = 0; col <= row; ++col) direction(row, col) = row == col ? 0.25 : -0.125;
    }
    const fixed_spd<Scalar, Order> point(point_matrix);
    constexpr double step = 1.0e-6;
    const fixed_spd<Scalar, Order> plus(point_matrix + step * direction);
    const fixed_spd<Scalar, Order> minus(point_matrix - step * direction);
    const auto plus_log = matrix_log(plus);
    const auto minus_log = matrix_log(minus);
    const auto log_difference = (plus_log - minus_log) / (2.0 * step);
    const auto log_derivative = matrix_log_frechet(point, direction);
    // the logarithm Frechet action agrees with the central-difference public API oracle
    EXPECT_LT(relative_error(log_derivative, log_difference), tolerance);

    fixed_symmetric<Scalar, Order> chart;
    for (int row = 0; row < Order; ++row) {
        for (int col = 0; col <= row; ++col) chart(row, col) = row == col ? 0.2 * (row + 1) : 0.05;
    }
    const auto plus_exp = matrix_exp(chart + step * direction);
    const auto minus_exp = matrix_exp(chart - step * direction);
    const auto exp_difference = (plus_exp - minus_exp) / (2.0 * step);
    const auto exp_derivative = matrix_exp_frechet(chart, direction);
    // the exponential Frechet action agrees with the central-difference public API oracle
    EXPECT_LT(relative_error(exp_derivative, exp_difference), tolerance);
}

/// @brief verifies failed fixed EVD recomputation invalidates previously computed factors
void check_invalid_fixed_evd_recompute() {
    using symmetric = fixed_symmetric<double, 3>;
    EVD<symmetric> decomposition;
    // a rotated positive matrix exercises all fixed three-dimensional Jacobi pivots
    const fixed_matrix<double, 3> valid_dense =
      make_matrix<double, 3>({4.0, 1.0, 0.5, 1.0, 3.0, -0.25, 0.5, -0.25, 2.0});
    const symmetric valid(valid_dense.template as_symmetric<Lower>());
    // an empty fixed decomposition exposes no completed factors
    EXPECT_FALSE(decomposition.computed());
    decomposition.compute(valid);
    // the small fixed path publishes completed factors for valid input
    EXPECT_TRUE(decomposition.computed());

    symmetric nonfinite = valid;
    nonfinite(1, 0) = std::numeric_limits<double>::quiet_NaN();
    // a nonfinite coefficient is rejected before fixed factors can be published
    EXPECT_THROW(decomposition.compute(nonfinite), std::invalid_argument);
    // failed recomputation clears factors left by the preceding valid input
    EXPECT_FALSE(decomposition.computed());

    decomposition.compute(valid);
    // valid input can recompute factors after an earlier validation failure
    EXPECT_TRUE(decomposition.computed());
    nonfinite(1, 0) = std::numeric_limits<double>::infinity();
    // an infinite coefficient is rejected by the always-active finite-input check
    EXPECT_THROW(decomposition.compute(nonfinite), std::invalid_argument);
    // infinite-input rejection also clears the previously computed factors
    EXPECT_FALSE(decomposition.computed());
}

// fixed two- and three-dimensional float and double paths match dynamic spectral oracles across deterministic edge
// cases
TEST(linear_algebra, small_spd_kernels_match_dynamic_spectral_oracles) {
    // double two-dimensional cases cover rotations, repeated values, threshold pivots and extreme scales
    check_spectrum_cases<double, 2>();
    // double three-dimensional cases cover every Jacobi pivot and the same numerical boundaries
    check_spectrum_cases<double, 3>();
    // float two-dimensional cases use scalar-appropriate residual tolerances
    check_spectrum_cases<float, 2>();
    // float three-dimensional cases use scalar-appropriate residual tolerances
    check_spectrum_cases<float, 3>();
    // invalid fixed input cannot leave stale factors from an earlier small-kernel computation
    check_invalid_fixed_evd_recompute();
}

// fixed Frechet differentials match central differences for two- and three-dimensional public transforms
TEST(linear_algebra, small_spd_frechet_matches_central_differences) {
    // the fixed two-dimensional logarithm and exponential derivatives match independent central differences
    check_frechet_finite_differences<2>();
    // the fixed three-dimensional logarithm and exponential derivatives match independent central differences
    check_frechet_finite_differences<3>();
}

}   // namespace
