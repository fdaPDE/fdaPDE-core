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

#include <fdaPDE/linear_algebra.h>
#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <type_traits>
#include <utility>

namespace {

namespace native = fdapde::linalg;

using fixed_matrix = native::Matrix<double, 3, 3>;
using dynamic_matrix = native::Matrix<double, fdapde::Dynamic, fdapde::Dynamic>;
using fixed_symmetric = native::SymmetricMatrix<double, 3, 3>;
using dynamic_symmetric = native::SymmetricMatrix<double, fdapde::Dynamic, fdapde::Dynamic>;
using fixed_spd = native::SPDMatrix<double, 3, 3>;
using dynamic_spd = native::SPDMatrix<double, fdapde::Dynamic, fdapde::Dynamic>;

template <typename T>
concept permits_rvalue_derived = requires(T value) { std::move(value).derived(); };
template <typename T>
concept permits_rvalue_rep = requires(T value) { std::move(value).rep(); };
template <typename T>
concept permits_rvalue_data = requires(T value) { std::move(value).data(); };
template <typename T>
concept permits_rvalue_copy_assignment = requires(T lhs, const T rhs) { std::move(lhs) = rhs; };
template <typename T>
concept permits_rvalue_checked_assignment =
  requires(T value, const fixed_matrix matrix) { std::move(value).assign(matrix, native::checked); };
template <typename T>
concept permits_coefficient_write = requires(T value) { value(0, 0) = 1.0; };
template <typename T>
concept permits_compound_addition = requires(T value) { value += value; };

static_assert(fixed_spd::ReadOnly == 1);
static_assert(native::is_spd_matrix_v<fixed_spd>);
static_assert(native::is_spd_matrix_v<const fixed_spd&>);
static_assert(native::is_symmetric_matrix_v<fixed_spd>);
static_assert(!std::is_default_constructible_v<fixed_spd>);
static_assert(std::is_copy_constructible_v<fixed_spd>);
static_assert(std::is_copy_assignable_v<fixed_spd>);
static_assert(std::is_same_v<decltype(std::declval<const fixed_spd&>().data()), const double*>);
static_assert(std::is_const_v<std::remove_reference_t<decltype(std::declval<const fixed_spd&>().rep())>>);
static_assert(!permits_rvalue_derived<fixed_spd>);
static_assert(!permits_rvalue_rep<fixed_spd>);
static_assert(!permits_rvalue_data<fixed_spd>);
static_assert(!permits_rvalue_copy_assignment<fixed_spd>);
static_assert(!permits_rvalue_checked_assignment<fixed_spd>);
static_assert(!permits_coefficient_write<fixed_spd>);
static_assert(!permits_compound_addition<fixed_spd>);

fixed_matrix reference_spd(double scale = 1.0) {
    return fixed_matrix(
      {scale * 53.0 / 9.0, scale * -26.0 / 9.0, scale * 4.0 / 9.0, scale * -26.0 / 9.0, scale * 44.0 / 9.0,
       scale * -22.0 / 9.0, scale * 4.0 / 9.0, scale * -22.0 / 9.0, scale * 29.0 / 9.0});
}

fixed_symmetric reference_direction() {
    fixed_symmetric direction;
    direction(0, 0) = 0.5;
    direction(1, 0) = -0.25;
    direction(1, 1) = 1.0;
    direction(2, 0) = 0.125;
    direction(2, 1) = 0.375;
    direction(2, 2) = -0.75;
    return direction;
}

fixed_symmetric reference_second_direction() {
    fixed_symmetric direction;
    direction(0, 0) = -0.75;
    direction(1, 0) = 0.5;
    direction(1, 1) = 0.25;
    direction(2, 0) = -0.375;
    direction(2, 1) = 0.125;
    direction(2, 2) = 1.0;
    return direction;
}

template <typename Symmetric> void set_symmetric_zero(Symmetric& matrix) {
    for (int i = 0; i < matrix.rows(); ++i) {
        for (int j = 0; j <= i; ++j) matrix(i, j) = 0.0;
    }
}

template <typename Actual, typename Expected> double relative_error(const Actual& actual, const Expected& expected) {
    const dynamic_matrix actual_dense(actual);
    const dynamic_matrix expected_dense(expected);
    return (actual_dense - expected_dense).norm() / expected_dense.norm();
}

template <typename SPDType, typename MatrixType> void expect_spectral_oracles(const MatrixType& dense) {
    const SPDType point(dense, native::checked);
    const auto logarithm = native::matrix_log(point);
    const auto restored = native::matrix_exp(logarithm);
    EXPECT_TRUE((native::is_spd_matrix_v<decltype(restored)>));
    EXPECT_LT(relative_error(restored, dense), 1.0e-11);
    EXPECT_NO_THROW((SPDType(restored, native::checked)));

    const auto root = native::matrix_sqrt(point);
    const dynamic_matrix root_dense(root);
    EXPECT_LT(relative_error(root_dense * root_dense, dense), 1.0e-11);
    EXPECT_NO_THROW((SPDType(root, native::checked)));

    const auto inverse_root = native::matrix_inverse_sqrt(point);
    const dynamic_matrix inverse_root_dense(inverse_root);
    const dynamic_matrix dense_dynamic(dense);
    const dynamic_matrix identity(native::IdentityMatrix<double, fdapde::Dynamic, fdapde::Dynamic>(3, 3));
    EXPECT_LT((inverse_root_dense * dense_dynamic * inverse_root_dense - identity).norm(), 1.0e-11);
    EXPECT_NO_THROW((SPDType(inverse_root, native::checked)));
}

}   // namespace

TEST(NativeSPD, CheckedStorageIsMetricNeutralAndReadOnly) {
    const fixed_matrix dense = reference_spd();
    const fixed_spd point(dense, native::checked);

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) EXPECT_DOUBLE_EQ(point(i, j), dense(i, j));
    }
}

TEST(NativeSPD, SupportsFixedAndDynamicThreeByThreeSpectralAlgebra) {
    const fixed_matrix fixed = reference_spd();
    const dynamic_matrix dynamic(fixed);
    expect_spectral_oracles<fixed_spd>(fixed);
    expect_spectral_oracles<dynamic_spd>(dynamic);
}

TEST(NativeSPD, SpectralAlgebraIsScaleSafe) {
    for (const double scale : {1.0e-150, 1.0e150}) {
        const fixed_matrix fixed = reference_spd(scale);
        const dynamic_matrix dynamic(fixed);
        expect_spectral_oracles<fixed_spd>(fixed);
        expect_spectral_oracles<dynamic_spd>(dynamic);
    }
}

TEST(NativeSPD, CopyAndCheckedAssignmentPreserveTheInvariant) {
    const fixed_matrix initial = reference_spd();
    fixed_spd point(initial, native::checked);
    fixed_spd copied(point);
    fixed_spd assigned(reference_spd(2.0), native::checked);
    assigned = point;

    EXPECT_LT(relative_error(copied, initial), 1.0e-15);
    EXPECT_LT(relative_error(assigned, initial), 1.0e-15);

    fixed_matrix invalid = initial;
    invalid(0, 0) = -100.0;
    EXPECT_THROW(point.assign(invalid, native::checked), std::domain_error);
    EXPECT_LT(relative_error(point, initial), 1.0e-15);

    const dynamic_matrix two_by_two(native::Matrix<double, 2, 2>({4.0, 1.0, 1.0, 3.0}));
    dynamic_spd dynamic_point(two_by_two, native::checked);
    dynamic_point.assign(dynamic_matrix(initial), native::checked);
    EXPECT_EQ(dynamic_point.rows(), 3);
    EXPECT_LT(relative_error(dynamic_point, initial), 1.0e-15);

    dynamic_spd dynamic_copy(dynamic_point);
    dynamic_spd dynamic_assigned(two_by_two, native::checked);
    dynamic_assigned = dynamic_copy;
    EXPECT_EQ(dynamic_assigned.rows(), 3);
    EXPECT_LT(relative_error(dynamic_assigned, initial), 1.0e-15);
}

TEST(NativeSPD, CheckedConstructionRejectsInvalidInputsAndOversizeShapes) {
    dynamic_matrix empty(0, 0);
    dynamic_matrix nonsquare(2, 3);
    nonsquare.set_zero();
    fixed_matrix asymmetric = reference_spd();
    asymmetric(0, 1) += 1.0;
    fixed_matrix indefinite = reference_spd();
    indefinite(0, 0) = -100.0;
    fixed_matrix nonfinite = reference_spd();
    nonfinite(0, 0) = std::numeric_limits<double>::infinity();

    EXPECT_THROW((dynamic_spd(empty, native::checked)), std::invalid_argument);
    EXPECT_THROW((dynamic_spd(nonsquare, native::checked)), std::invalid_argument);
    EXPECT_THROW(
      (fixed_spd(native::Matrix<double, 2, 2>({2.0, 0.0, 0.0, 2.0}), native::checked)), std::invalid_argument);
    EXPECT_THROW((fixed_spd(asymmetric, native::checked)), std::invalid_argument);
    EXPECT_THROW((fixed_spd(indefinite, native::checked)), std::domain_error);
    EXPECT_THROW((fixed_spd(nonfinite, native::checked)), std::invalid_argument);

    for (const double scale : {1.0e-150, 1.0e150}) {
        const fixed_matrix scaled_indefinite({scale, 2.0 * scale, 0.0, 2.0 * scale, scale, 0.0, 0.0, 0.0, scale});
        EXPECT_THROW((fixed_spd(scaled_indefinite, native::checked)), std::domain_error);
    }

    const native::IdentityMatrix<double, fdapde::Dynamic, fdapde::Dynamic> too_large(46341, 46341);
    EXPECT_THROW((dynamic_spd(too_large, native::checked)), std::length_error);
    const auto too_large_symmetric = too_large.template as_symmetric<native::Lower>();
    EXPECT_THROW(native::matrix_exp(too_large_symmetric), std::length_error);
}

TEST(NativeSPD, CheckedSymmetryToleranceRemainsFiniteNearTheScalarLimit) {
    native::Matrix<double, 4, 4> asymmetric;
    asymmetric.set_zero();
    for (int i = 0; i < 4; ++i) asymmetric(i, i) = 1.0e308;
    asymmetric(0, 1) = 1.0e307;

    EXPECT_THROW((native::SPDMatrix<double, 4, 4>(asymmetric, native::checked)), std::invalid_argument);
}

TEST(NativeSPD, NumericalPositiveDefinitenessUsesARelativeThreshold) {
    fixed_matrix accepted;
    accepted.set_zero();
    accepted(0, 0) = 1.0;
    accepted(1, 1) = 1.0;
    accepted(2, 2) = 1.0e-13;
    EXPECT_NO_THROW((fixed_spd(accepted, native::checked)));

    fixed_matrix rejected = accepted;
    rejected(2, 2) = 1.0e-14;
    EXPECT_THROW((fixed_spd(rejected, native::checked)), std::domain_error);
}

TEST(NativeSPD, SpectralResultsNeverPublishAnInvalidPointAtTheNumericalBoundary) {
    const fixed_matrix dense(
      {4.006195626983144e-21, -7.9529560182295794e-22, -3.966472048506717e-21, -7.9529560182295794e-22,
       7.4540966370188263e-21, -2.6426962474257787e-21, -3.966472048506717e-21, -2.6426962474257787e-21,
       5.5397077359984564e-21});
    const fixed_spd point(dense, native::checked);
    const fixed_symmetric logarithm(native::matrix_log(point));

    try {
        const auto restored = native::matrix_exp(logarithm);
        EXPECT_NO_THROW((fixed_spd(restored, native::checked)));
    } catch (const std::domain_error&) { SUCCEED(); }
}

TEST(NativeSPD, FrechetDifferentialsHandleRepeatedAndCloseSpectra) {
    fixed_matrix two_identity;
    two_identity.set_zero();
    for (int i = 0; i < 3; ++i) two_identity(i, i) = 2.0;
    const fixed_spd point(two_identity, native::checked);
    const fixed_symmetric direction = reference_direction();
    const fixed_symmetric log_derivative(native::matrix_log_frechet(point, direction));
    const fixed_symmetric expected_log_derivative(direction / 2.0);
    EXPECT_LT((log_derivative - expected_log_derivative).norm(), 1.0e-12);

    fixed_symmetric log_two_identity;
    set_symmetric_zero(log_two_identity);
    for (int i = 0; i < 3; ++i) log_two_identity(i, i) = std::log(2.0);
    const fixed_symmetric exp_derivative(native::matrix_exp_frechet(log_two_identity, direction));
    const fixed_symmetric expected_exp_derivative(2.0 * direction);
    EXPECT_LT((exp_derivative - expected_exp_derivative).norm(), 1.0e-12);

    constexpr double gap = 1.0e-12;
    fixed_matrix close;
    close.set_zero();
    close(0, 0) = 2.0;
    close(1, 1) = 2.0 + gap;
    close(2, 2) = 5.0;
    const fixed_spd close_point(close, native::checked);
    fixed_symmetric off_diagonal;
    set_symmetric_zero(off_diagonal);
    off_diagonal(1, 0) = 1.0;
    const auto derivative = native::matrix_log_frechet(close_point, off_diagonal);
    EXPECT_NEAR(derivative(0, 1), std::log1p(gap / 2.0) / gap, 1.0e-12);

    fixed_symmetric close_log;
    set_symmetric_zero(close_log);
    close_log(0, 0) = 0.2;
    close_log(1, 1) = 0.2 + gap;
    close_log(2, 2) = 0.7;
    const auto exp_close_derivative = native::matrix_exp_frechet(close_log, off_diagonal);
    EXPECT_NEAR(exp_close_derivative(0, 1), std::exp(0.2) * std::expm1(gap) / gap, 1.0e-12);
}

TEST(NativeSPD, FrechetDifferentialsAreInverseAndMatchFiniteDifferences) {
    const fixed_matrix dense = reference_spd();
    const fixed_spd point(dense, native::checked);
    const fixed_symmetric direction = reference_direction();
    const auto logarithm = native::matrix_log(point);
    const auto log_direction = native::matrix_log_frechet(point, direction);
    const auto recovered_direction = native::matrix_exp_frechet(logarithm, log_direction);
    EXPECT_LT(relative_error(recovered_direction, direction), 1.0e-10);

    const dynamic_matrix dynamic_dense(dense);
    const dynamic_spd dynamic_point(dynamic_dense, native::checked);
    dynamic_symmetric dynamic_direction(3, 3);
    dynamic_direction = direction;
    const dynamic_symmetric dynamic_logarithm(native::matrix_log(dynamic_point));
    const dynamic_symmetric dynamic_log_direction(native::matrix_log_frechet(dynamic_point, dynamic_direction));
    const dynamic_symmetric dynamic_recovered_direction(
      native::matrix_exp_frechet(dynamic_logarithm, dynamic_log_direction));
    EXPECT_LT(relative_error(dynamic_recovered_direction, dynamic_direction), 1.0e-10);

    constexpr double step = 1.0e-6;
    const fixed_spd plus(dense + step * direction, native::checked);
    const fixed_spd minus(dense - step * direction, native::checked);
    const fixed_symmetric plus_log(native::matrix_log(plus));
    const fixed_symmetric minus_log(native::matrix_log(minus));
    const fixed_symmetric finite_difference((plus_log - minus_log) / (2.0 * step));
    EXPECT_LT(relative_error(native::matrix_log_frechet(point, direction), finite_difference), 5.0e-8);
}

TEST(NativeSPD, LogSecondFrechetIsSymmetricAndBilinearInItsDirections) {
    const fixed_spd point(reference_spd(), native::checked);
    const fixed_symmetric first = reference_direction();
    const fixed_symmetric second = reference_second_direction();
    const fixed_symmetric third(first - 0.25 * second);

    const fixed_symmetric first_second(native::matrix_log_second_frechet(point, first, second));
    const fixed_symmetric second_first(native::matrix_log_second_frechet(point, second, first));
    EXPECT_LT(relative_error(first_second, second_first), 1.0e-12);

    const fixed_symmetric linear_first(native::matrix_log_second_frechet(point, first + 0.375 * third, second));
    const fixed_symmetric third_second(native::matrix_log_second_frechet(point, third, second));
    const fixed_symmetric expected_first(first_second + third_second * 0.375);
    EXPECT_LT(relative_error(linear_first, expected_first), 2.0e-12);

    const fixed_symmetric linear_second(native::matrix_log_second_frechet(point, first, second - 0.625 * third));
    const fixed_symmetric first_third(native::matrix_log_second_frechet(point, first, third));
    const fixed_symmetric expected_second(first_second - first_third * 0.625);
    EXPECT_LT(relative_error(linear_second, expected_second), 2.0e-12);
}

TEST(NativeSPD, LogSecondFrechetHandlesScalarRepeatedAndCloseSpectra) {
    const fixed_symmetric first = reference_direction();
    const fixed_symmetric second = reference_second_direction();
    fixed_matrix two_identity;
    two_identity.set_zero();
    for (int i = 0; i < 3; ++i) two_identity(i, i) = 2.0;
    const fixed_spd scalar_point(two_identity, native::checked);
    const fixed_matrix first_dense(first);
    const fixed_matrix second_dense(second);
    const fixed_matrix anticommutator(first_dense * second_dense + second_dense * first_dense);
    const fixed_matrix scalar_expected_dense(anticommutator * -0.125);
    const fixed_symmetric scalar_expected(scalar_expected_dense.template as_symmetric<native::Lower>());
    EXPECT_LT(relative_error(native::matrix_log_second_frechet(scalar_point, first, second), scalar_expected), 2.0e-12);

    fixed_matrix repeated;
    repeated.set_zero();
    repeated(0, 0) = 2.0;
    repeated(1, 1) = 2.0;
    repeated(2, 2) = 5.0;
    fixed_symmetric first_chain;
    fixed_symmetric second_chain;
    set_symmetric_zero(first_chain);
    set_symmetric_zero(second_chain);
    first_chain(1, 0) = 1.0;
    second_chain(2, 1) = 1.0;
    const auto repeated_result =
      native::matrix_log_second_frechet(fixed_spd(repeated, native::checked), first_chain, second_chain);
    EXPECT_NEAR(repeated_result(2, 0), (std::log(2.5) - 1.5) / 9.0, 2.0e-14);

    constexpr double gap = 1.0e-12;
    fixed_matrix close;
    close.set_zero();
    close(0, 0) = 2.0;
    close(1, 1) = 2.0 + gap;
    close(2, 2) = 2.0 + 2.0 * gap;
    const auto close_result = native::matrix_log_second_frechet(fixed_spd(close, native::checked), first, second);
    EXPECT_LT(relative_error(close_result, scalar_expected), 3.0e-12);
}

TEST(NativeSPD, LogSecondFrechetRespectsScaleAndSupportsDynamicStorage) {
    const fixed_spd point(reference_spd(), native::checked);
    const fixed_symmetric first = reference_direction();
    const fixed_symmetric second = reference_second_direction();
    const fixed_symmetric expected(native::matrix_log_second_frechet(point, first, second));
    for (const double scale : {1.0e-150, 1.0e150}) {
        const fixed_spd scaled_point(reference_spd(scale), native::checked);
        const fixed_symmetric scaled_first(scale * first);
        const fixed_symmetric scaled_second(scale * second);
        EXPECT_LT(
          relative_error(native::matrix_log_second_frechet(scaled_point, scaled_first, scaled_second), expected),
          2.0e-10);
    }

    const dynamic_spd dynamic_point(dynamic_matrix(reference_spd()), native::checked);
    dynamic_symmetric dynamic_first(3, 3);
    dynamic_symmetric dynamic_second(3, 3);
    dynamic_first = first;
    dynamic_second = second;
    const dynamic_symmetric dynamic_result(
      native::matrix_log_second_frechet(dynamic_point, dynamic_first, dynamic_second));
    static_assert(std::is_same_v<decltype(dynamic_result), const dynamic_symmetric>);
    EXPECT_LT(relative_error(dynamic_result, expected), 2.0e-12);
}

TEST(NativeSPD, LogSecondFrechetMatchesTheFiniteDifferenceOfTheFirstFrechetDerivative) {
    const fixed_matrix dense = reference_spd();
    const fixed_spd point(dense, native::checked);
    const fixed_symmetric first = reference_direction();
    const fixed_symmetric second = reference_second_direction();
    constexpr double step = 2.0e-5;
    const fixed_spd plus(dense + step * second, native::checked);
    const fixed_spd minus(dense - step * second, native::checked);
    const fixed_symmetric plus_derivative(native::matrix_log_frechet(plus, first));
    const fixed_symmetric minus_derivative(native::matrix_log_frechet(minus, first));
    const fixed_symmetric finite_difference((plus_derivative - minus_derivative) / (2.0 * step));
    EXPECT_LT(relative_error(native::matrix_log_second_frechet(point, first, second), finite_difference), 2.0e-9);
}

TEST(NativeSPD, ConstSymmetricViewsProduceOwningNonconstResults) {
    const double packed_log_two[] = {std::log(2.0), 0.0, std::log(2.0), 0.0, 0.0, std::log(2.0)};
    const native::SymmetricMatrixView<const double, 3, 3> logarithm(packed_log_two);
    const auto point = native::matrix_exp(logarithm);

    static_assert(std::is_same_v<typename decltype(point)::Scalar, double>);
    EXPECT_TRUE((native::is_spd_matrix_v<decltype(point)>));
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) EXPECT_NEAR(point(i, j), i == j ? 2.0 : 0.0, 1.0e-12);
    }
}

TEST(NativeSPD, SpectralOperationsRejectInvalidNumericsAndDirections) {
    fixed_matrix indefinite = reference_spd();
    indefinite(0, 0) = -100.0;
    const fixed_spd unchecked_indefinite(indefinite, native::unchecked);
    EXPECT_THROW(native::matrix_log(unchecked_indefinite), std::domain_error);
    EXPECT_THROW(native::matrix_sqrt(unchecked_indefinite), std::domain_error);
    EXPECT_THROW(native::matrix_inverse_sqrt(unchecked_indefinite), std::domain_error);

    fixed_symmetric overflow;
    set_symmetric_zero(overflow);
    for (int i = 0; i < 3; ++i) overflow(i, i) = 1000.0;
    EXPECT_THROW(native::matrix_exp(overflow), std::domain_error);

    fixed_symmetric underflow;
    set_symmetric_zero(underflow);
    underflow(1, 1) = -1000.0;
    underflow(2, 2) = -1000.0;
    EXPECT_THROW(native::matrix_exp(underflow), std::domain_error);

    fixed_symmetric nonfinite;
    set_symmetric_zero(nonfinite);
    nonfinite(0, 0) = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(native::matrix_exp(nonfinite), std::invalid_argument);

    const fixed_spd point(reference_spd(), native::checked);
    native::SymmetricMatrix<double, fdapde::Dynamic, fdapde::Dynamic> wrong_direction(2, 2);
    set_symmetric_zero(wrong_direction);
    EXPECT_THROW(native::matrix_log_frechet(point, wrong_direction), std::invalid_argument);
    EXPECT_THROW(
      native::matrix_log_second_frechet(point, wrong_direction, reference_direction()), std::invalid_argument);
    EXPECT_THROW(
      native::matrix_log_second_frechet(point, reference_direction(), wrong_direction), std::invalid_argument);

    fixed_symmetric nonfinite_direction = reference_direction();
    nonfinite_direction(2, 1) = std::numeric_limits<double>::infinity();
    EXPECT_THROW(native::matrix_exp_frechet(native::matrix_log(point), nonfinite_direction), std::invalid_argument);
    EXPECT_THROW(
      native::matrix_log_second_frechet(point, nonfinite_direction, reference_direction()), std::invalid_argument);
}

TEST(NativeSPD, SpectralOperationsOwnResultsFromSafeTemporaries) {
    const fixed_matrix dense = reference_spd();
    const auto logarithm = native::matrix_log(fixed_spd(dense, native::checked));
    const auto restored = native::matrix_exp(fixed_symmetric(logarithm));
    EXPECT_LT(relative_error(restored, dense), 1.0e-11);
}
