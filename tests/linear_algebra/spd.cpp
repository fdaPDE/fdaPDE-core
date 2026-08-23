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

using namespace fdapde;

using fixed_matrix = Matrix<double, 3, 3>;
using dynamic_matrix = Matrix<double, Dynamic, Dynamic>;
using fixed_symmetric = SymmetricMatrix<double, 3, 3>;
using dynamic_symmetric = SymmetricMatrix<double, Dynamic, Dynamic>;
using fixed_spd = SPDMatrix<double, 3, 3>;
using dynamic_spd = SPDMatrix<double, Dynamic, Dynamic>;

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
  requires(T value, const fixed_matrix matrix) { std::move(value).assign(matrix, checked); };
template <typename T>
concept permits_public_unchecked_construction = requires(const fixed_matrix matrix) { T(matrix, unchecked); };
template <typename T>
concept permits_coefficient_write = requires(T value) { value(0, 0) = 1.0; };
template <typename T>
concept permits_compound_addition = requires(T value) { value += value; };

static_assert(fixed_spd::ReadOnly == 1);
static_assert(is_spd_matrix_v<fixed_spd>);
static_assert(is_spd_matrix_v<const fixed_spd&>);
static_assert(is_symmetric_matrix_v<fixed_spd>);
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
static_assert(!permits_public_unchecked_construction<fixed_spd>);
static_assert(!permits_coefficient_write<fixed_spd>);
static_assert(!permits_compound_addition<fixed_spd>);

struct oversized_symmetric_expression : SymmetricMatrixExpr<oversized_symmetric_expression> {
    using Scalar = double;
    static constexpr int Rows = Dynamic;
    static constexpr int Cols = Dynamic;

    explicit oversized_symmetric_expression(bool& coefficient_accessed) :
        coefficient_accessed_(&coefficient_accessed) { }

    double operator()(int, int) const {
        *coefficient_accessed_ = true;
        return 0.0;
    }
    constexpr int rows() const { return 46341; }
    constexpr int cols() const { return 46341; }
   private:
    bool* coefficient_accessed_;
};

struct invalid_spd_expression : SPDMatrixExpr<invalid_spd_expression> {
    using Scalar = double;
    static constexpr int Rows = 3;
    static constexpr int Cols = 3;

    explicit invalid_spd_expression(const fixed_matrix& data) : data_(data) { }

    constexpr double operator()(int i, int j) const { return data_(i, j); }
    constexpr int rows() const { return Rows; }
    constexpr int cols() const { return Cols; }
   private:
    fixed_matrix data_;
};

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
    const SPDType point(dense, checked);
    const auto logarithm = matrix_log(point);
    const auto restored = matrix_exp(logarithm);
    EXPECT_TRUE((is_spd_matrix_v<decltype(restored)>));
    EXPECT_LT(relative_error(restored, dense), 1.0e-11);
    EXPECT_NO_THROW((SPDType(restored, checked)));

    const auto root = matrix_sqrt(point);
    const dynamic_matrix root_dense(root);
    EXPECT_LT(relative_error(root_dense * root_dense, dense), 1.0e-11);
    EXPECT_NO_THROW((SPDType(root, checked)));

    const auto inverse_root = matrix_inverse_sqrt(point);
    const dynamic_matrix inverse_root_dense(inverse_root);
    const dynamic_matrix dense_dynamic(dense);
    const dynamic_matrix identity(IdentityMatrix<double, Dynamic, Dynamic>(3, 3));
    EXPECT_LT((inverse_root_dense * dense_dynamic * inverse_root_dense - identity).norm(), 1.0e-11);
    EXPECT_NO_THROW((SPDType(inverse_root, checked)));
}

TEST(linear_algebra, spd_checked_storage_is_metric_neutral_and_read_only) {
    const fixed_matrix dense = reference_spd();
    const fixed_spd point(dense, checked);

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) EXPECT_DOUBLE_EQ(point(i, j), dense(i, j));
    }
}

TEST(linear_algebra, spd_fixed_and_dynamic_spectral_algebra) {
    const fixed_matrix fixed = reference_spd();
    const dynamic_matrix dynamic(fixed);
    expect_spectral_oracles<fixed_spd>(fixed);
    expect_spectral_oracles<dynamic_spd>(dynamic);
}

TEST(linear_algebra, spd_spectral_algebra_is_scale_safe) {
    for (const double scale : {1.0e-150, 1.0e150}) {
        const fixed_matrix fixed = reference_spd(scale);
        const dynamic_matrix dynamic(fixed);
        expect_spectral_oracles<fixed_spd>(fixed);
        expect_spectral_oracles<dynamic_spd>(dynamic);
    }
}

TEST(linear_algebra, spd_copy_and_checked_assignment_preserve_the_invariant) {
    const fixed_matrix initial = reference_spd();
    fixed_spd point(initial, checked);
    fixed_spd copied(point);
    fixed_spd assigned(reference_spd(2.0), checked);
    assigned = point;

    EXPECT_LT(relative_error(copied, initial), 1.0e-15);
    EXPECT_LT(relative_error(assigned, initial), 1.0e-15);

    fixed_matrix invalid = initial;
    invalid(0, 0) = -100.0;
    EXPECT_THROW(point.assign(invalid, checked), std::domain_error);
    EXPECT_LT(relative_error(point, initial), 1.0e-15);

    const dynamic_matrix two_by_two(Matrix<double, 2, 2>({4.0, 1.0, 1.0, 3.0}));
    dynamic_spd dynamic_point(two_by_two, checked);
    dynamic_point.assign(dynamic_matrix(initial), checked);
    EXPECT_EQ(dynamic_point.rows(), 3);
    EXPECT_LT(relative_error(dynamic_point, initial), 1.0e-15);

    dynamic_spd dynamic_copy(dynamic_point);
    dynamic_spd dynamic_assigned(two_by_two, checked);
    dynamic_assigned = dynamic_copy;
    EXPECT_EQ(dynamic_assigned.rows(), 3);
    EXPECT_LT(relative_error(dynamic_assigned, initial), 1.0e-15);
}

TEST(linear_algebra, spd_checked_construction_rejects_invalid_inputs_and_oversize_shapes) {
    dynamic_matrix empty(0, 0);
    dynamic_matrix nonsquare(2, 3);
    nonsquare.set_zero();
    fixed_matrix asymmetric = reference_spd();
    asymmetric(0, 1) += 1.0;
    fixed_matrix indefinite = reference_spd();
    indefinite(0, 0) = -100.0;
    fixed_matrix nonfinite = reference_spd();
    nonfinite(0, 0) = std::numeric_limits<double>::infinity();

    EXPECT_THROW((dynamic_spd(empty, checked)), std::invalid_argument);
    EXPECT_THROW((dynamic_spd(nonsquare, checked)), std::invalid_argument);
    EXPECT_THROW((fixed_spd(Matrix<double, 2, 2>({2.0, 0.0, 0.0, 2.0}), checked)), std::invalid_argument);
    EXPECT_THROW((fixed_spd(asymmetric, checked)), std::invalid_argument);
    EXPECT_THROW((fixed_spd(indefinite, checked)), std::domain_error);
    EXPECT_THROW((fixed_spd(nonfinite, checked)), std::invalid_argument);

    for (const double scale : {1.0e-150, 1.0e150}) {
        const fixed_matrix scaled_indefinite({scale, 2.0 * scale, 0.0, 2.0 * scale, scale, 0.0, 0.0, 0.0, scale});
        EXPECT_THROW((fixed_spd(scaled_indefinite, checked)), std::domain_error);
    }

    bool coefficient_accessed = false;
    const oversized_symmetric_expression too_large(coefficient_accessed);
    EXPECT_THROW((dynamic_spd(too_large, checked)), std::length_error);
    EXPECT_FALSE(coefficient_accessed);
    EXPECT_THROW(matrix_exp(too_large), std::length_error);
    EXPECT_FALSE(coefficient_accessed);
}

TEST(linear_algebra, spd_checked_symmetry_tolerance_remains_finite_near_the_scalar_limit) {
    Matrix<double, 4, 4> asymmetric;
    asymmetric.set_zero();
    for (int i = 0; i < 4; ++i) asymmetric(i, i) = 1.0e308;
    asymmetric(0, 1) = 1.0e307;

    EXPECT_THROW((SPDMatrix<double, 4, 4>(asymmetric, checked)), std::invalid_argument);
}

TEST(linear_algebra, spd_numerical_positive_definiteness_uses_a_relative_threshold) {
    fixed_matrix accepted;
    accepted.set_zero();
    accepted(0, 0) = 1.0;
    accepted(1, 1) = 1.0;
    accepted(2, 2) = 1.0e-13;
    EXPECT_NO_THROW((fixed_spd(accepted, checked)));

    fixed_matrix rejected = accepted;
    rejected(2, 2) = 1.0e-14;
    EXPECT_THROW((fixed_spd(rejected, checked)), std::domain_error);
}

TEST(linear_algebra, spd_spectral_results_preserve_the_numerical_invariant) {
    const fixed_matrix dense(
      {4.006195626983144e-21, -7.9529560182295794e-22, -3.966472048506717e-21, -7.9529560182295794e-22,
       7.4540966370188263e-21, -2.6426962474257787e-21, -3.966472048506717e-21, -2.6426962474257787e-21,
       5.5397077359984564e-21});
    const fixed_spd point(dense, checked);
    const fixed_symmetric logarithm(matrix_log(point));

    try {
        const auto restored = matrix_exp(logarithm);
        EXPECT_NO_THROW((fixed_spd(restored, checked)));
    } catch (const std::domain_error&) { SUCCEED(); }
}

TEST(linear_algebra, spd_frechet_differentials_handle_repeated_and_close_spectra) {
    fixed_matrix two_identity;
    two_identity.set_zero();
    for (int i = 0; i < 3; ++i) two_identity(i, i) = 2.0;
    const fixed_spd point(two_identity, checked);
    const fixed_symmetric direction = reference_direction();
    const fixed_symmetric log_derivative(matrix_log_frechet(point, direction));
    const fixed_symmetric expected_log_derivative(direction / 2.0);
    EXPECT_LT((log_derivative - expected_log_derivative).norm(), 1.0e-12);

    fixed_symmetric log_two_identity;
    set_symmetric_zero(log_two_identity);
    for (int i = 0; i < 3; ++i) log_two_identity(i, i) = std::log(2.0);
    const fixed_symmetric exp_derivative(matrix_exp_frechet(log_two_identity, direction));
    const fixed_symmetric expected_exp_derivative(2.0 * direction);
    EXPECT_LT((exp_derivative - expected_exp_derivative).norm(), 1.0e-12);

    constexpr double gap = 1.0e-12;
    fixed_matrix close;
    close.set_zero();
    close(0, 0) = 2.0;
    close(1, 1) = 2.0 + gap;
    close(2, 2) = 5.0;
    const fixed_spd close_point(close, checked);
    fixed_symmetric off_diagonal;
    set_symmetric_zero(off_diagonal);
    off_diagonal(1, 0) = 1.0;
    const auto derivative = matrix_log_frechet(close_point, off_diagonal);
    EXPECT_NEAR(derivative(0, 1), std::log1p(gap / 2.0) / gap, 1.0e-12);

    fixed_symmetric close_log;
    set_symmetric_zero(close_log);
    close_log(0, 0) = 0.2;
    close_log(1, 1) = 0.2 + gap;
    close_log(2, 2) = 0.7;
    const auto exp_close_derivative = matrix_exp_frechet(close_log, off_diagonal);
    EXPECT_NEAR(exp_close_derivative(0, 1), std::exp(0.2) * std::expm1(gap) / gap, 1.0e-12);
}

TEST(linear_algebra, spd_frechet_differentials_are_inverse_and_match_finite_differences) {
    const fixed_matrix dense = reference_spd();
    const fixed_spd point(dense, checked);
    const fixed_symmetric direction = reference_direction();
    const auto logarithm = matrix_log(point);
    const auto log_direction = matrix_log_frechet(point, direction);
    const auto recovered_direction = matrix_exp_frechet(logarithm, log_direction);
    EXPECT_LT(relative_error(recovered_direction, direction), 1.0e-10);

    const dynamic_matrix dynamic_dense(dense);
    const dynamic_spd dynamic_point(dynamic_dense, checked);
    dynamic_symmetric dynamic_direction(3, 3);
    dynamic_direction = direction;
    const dynamic_symmetric dynamic_logarithm(matrix_log(dynamic_point));
    const dynamic_symmetric dynamic_log_direction(matrix_log_frechet(dynamic_point, dynamic_direction));
    const dynamic_symmetric dynamic_recovered_direction(matrix_exp_frechet(dynamic_logarithm, dynamic_log_direction));
    EXPECT_LT(relative_error(dynamic_recovered_direction, dynamic_direction), 1.0e-10);

    constexpr double step = 1.0e-6;
    const fixed_spd plus(dense + step * direction, checked);
    const fixed_spd minus(dense - step * direction, checked);
    const fixed_symmetric plus_log(matrix_log(plus));
    const fixed_symmetric minus_log(matrix_log(minus));
    const fixed_symmetric finite_difference((plus_log - minus_log) / (2.0 * step));
    EXPECT_LT(relative_error(matrix_log_frechet(point, direction), finite_difference), 5.0e-8);
}

TEST(linear_algebra, spd_const_symmetric_views_produce_owning_nonconst_results) {
    const double packed_log_two[] = {std::log(2.0), 0.0, std::log(2.0), 0.0, 0.0, std::log(2.0)};
    const SymmetricMatrixView<const double, 3, 3> logarithm(packed_log_two);
    const auto point = matrix_exp(logarithm);

    static_assert(std::is_same_v<typename decltype(point)::Scalar, double>);
    EXPECT_TRUE((is_spd_matrix_v<decltype(point)>));
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) EXPECT_NEAR(point(i, j), i == j ? 2.0 : 0.0, 1.0e-12);
    }
}

TEST(linear_algebra, spd_spectral_operations_reject_invalid_numerics_and_directions) {
    fixed_matrix indefinite = reference_spd();
    indefinite(0, 0) = -100.0;
    const invalid_spd_expression unchecked_indefinite(indefinite);
    EXPECT_THROW(matrix_log(unchecked_indefinite), std::domain_error);
    EXPECT_THROW(matrix_sqrt(unchecked_indefinite), std::domain_error);
    EXPECT_THROW(matrix_inverse_sqrt(unchecked_indefinite), std::domain_error);

    fixed_symmetric overflow;
    set_symmetric_zero(overflow);
    for (int i = 0; i < 3; ++i) overflow(i, i) = 1000.0;
    EXPECT_THROW(matrix_exp(overflow), std::domain_error);

    fixed_symmetric underflow;
    set_symmetric_zero(underflow);
    underflow(1, 1) = -1000.0;
    underflow(2, 2) = -1000.0;
    EXPECT_THROW(matrix_exp(underflow), std::domain_error);

    fixed_symmetric nonfinite;
    set_symmetric_zero(nonfinite);
    nonfinite(0, 0) = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(matrix_exp(nonfinite), std::invalid_argument);

    const fixed_spd point(reference_spd(), checked);
    SymmetricMatrix<double, Dynamic, Dynamic> wrong_direction(2, 2);
    set_symmetric_zero(wrong_direction);
    EXPECT_THROW(matrix_log_frechet(point, wrong_direction), std::invalid_argument);

    fixed_symmetric nonfinite_direction = reference_direction();
    nonfinite_direction(2, 1) = std::numeric_limits<double>::infinity();
    EXPECT_THROW(matrix_exp_frechet(matrix_log(point), nonfinite_direction), std::invalid_argument);
}

TEST(linear_algebra, spd_spectral_operations_own_results_from_safe_temporaries) {
    const fixed_matrix dense = reference_spd();
    const auto logarithm = matrix_log(fixed_spd(dense, checked));
    const auto restored = matrix_exp(fixed_symmetric(logarithm));
    EXPECT_LT(relative_error(restored, dense), 1.0e-11);
}

}   // namespace
