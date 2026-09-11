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

template <typename Decomposition>
concept permits_rvalue_lu_factors = requires(Decomposition decomposition) {
    std::move(decomposition).P();
    std::move(decomposition).L();
    std::move(decomposition).U();
};

using fixed_lu = PartialPivLU<Matrix<double, 3, 3>>;
using const_view_lu = PartialPivLU<MatrixView<const double, 3, 3>>;
// the LU factors cannot be borrowed from a temporary decomposition
static_assert(!permits_rvalue_lu_factors<fixed_lu>);
// a const input view produces owned double-valued factors
static_assert(std::is_same_v<typename const_view_lu::Scalar, double>);
constexpr Matrix<int, 2, 2> integral_matrix({1, 2, 3, 4});
// the integral two-by-two determinant remains a constant expression
static_assert(integral_matrix.determinant() == -2);

template <typename Actual, typename Expected>
void expect_matrix_near(const Actual& actual, const Expected& expected, double tolerance = 1.0e-12) {
    // matrix comparison requires matching row counts before coefficient access
    ASSERT_EQ(actual.rows(), expected.rows());
    // matrix comparison requires matching column counts before coefficient access
    ASSERT_EQ(actual.cols(), expected.cols());
    for (int row = 0; row < actual.rows(); ++row) {
        for (int col = 0; col < actual.cols(); ++col) {
            // each computed coefficient matches its independently assembled reference within absolute tolerance
            EXPECT_NEAR(static_cast<double>(actual(row, col)), static_cast<double>(expected(row, col)), tolerance);
        }
    }
}

template <int StorageOrder> void check_partial_piv_lu_contracts() {
    using matrix_type = Matrix<double, 3, 3, StorageOrder>;
    using vector_type = Matrix<double, 3, 1, StorageOrder>;
    const matrix_type matrix({0.0, 2.0, 1.0, 1.0, -2.0, -3.0, 2.0, 3.0, 1.0});
    const vector_type expected({2.0, -1.0, 3.0});
    const vector_type rhs(matrix * expected);

    const PartialPivLU factorization(matrix);
    // a nonsingular matrix produces usable LU factors
    EXPECT_EQ(factorization.info(), 0);
    // all three independent directions are retained by rank detection
    EXPECT_EQ(factorization.rank(), 3);
    // the pivoted determinant equals the independently calculated value minus seven
    EXPECT_DOUBLE_EQ(factorization.determinant(), -7.0);
    expect_matrix_near(factorization.solve(rhs), expected);
    const Matrix<double, 3, 2, StorageOrder> expected_multiple({1.0, -1.0, 2.0, 0.5, -3.0, 4.0});
    const Matrix<double, 3, 2, StorageOrder> rhs_multiple(matrix * expected_multiple);
    expect_matrix_near(factorization.solve(rhs_multiple), expected_multiple);
    expect_matrix_near(
      Matrix<double, 3, 3>(factorization.P() * matrix), Matrix<double, 3, 3>(factorization.L() * factorization.U()));

    const MatrixView<const double, 3, 3, StorageOrder> const_view(matrix.data());
    const PartialPivLU const_view_factorization(const_view);
    // factoring a const view produces the same determinant
    EXPECT_DOUBLE_EQ(const_view_factorization.determinant(), -7.0);
    expect_matrix_near(const_view_factorization.solve(rhs), expected);

    const auto retained_factorization = [] {
        const matrix_type source({0.0, 2.0, 1.0, 1.0, -2.0, -3.0, 2.0, 3.0, 1.0});
        const matrix_type zero = matrix_type::Zero();
        return PartialPivLU(source + zero);
    }();
    expect_matrix_near(retained_factorization.solve(vector_type({1.0, -5.0, 4.0})), expected);

    const matrix_type singular({1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0});
    const PartialPivLU singular_factorization(singular);
    // the first unusable pivot in the singular example is the second one
    EXPECT_EQ(singular_factorization.info(), 2);
    // rank detection retains two independent directions despite singularity
    EXPECT_EQ(singular_factorization.rank(), 2);
    // the singular example has algebraic determinant zero
    EXPECT_DOUBLE_EQ(singular_factorization.determinant(), 0.0);
    expect_matrix_near(
      Matrix<double, 3, 3>(singular_factorization.P() * singular),
      Matrix<double, 3, 3>(singular_factorization.L() * singular_factorization.U()));
    // solving with a singular factorization raises a domain error
    EXPECT_THROW(static_cast<void>(singular_factorization.solve(rhs)), std::domain_error);

    const matrix_type zero = matrix_type::Zero();
    const PartialPivLU zero_factorization(zero);
    // the zero matrix reports its first pivot as unusable
    EXPECT_EQ(zero_factorization.info(), 1);
    // the zero matrix has rank zero
    EXPECT_EQ(zero_factorization.rank(), 0);
    // the zero matrix has determinant zero
    EXPECT_DOUBLE_EQ(zero_factorization.determinant(), 0.0);
    const Matrix<double, 3, 3> zero_reconstruction(zero_factorization.L() * zero_factorization.U());
    expect_matrix_near(Matrix<double, 3, 3>(zero_factorization.P() * zero), zero_reconstruction);
    for (int row = 0; row < 3; ++row) {
        // reconstructing the zero matrix from its factors does not produce NaN or infinity
        for (int col = 0; col < 3; ++col) EXPECT_TRUE(std::isfinite(zero_reconstruction(row, col)));
    }

    Matrix<double, Dynamic, Dynamic, StorageOrder> rectangular(2, 3);
    Matrix<double, Dynamic, Dynamic, StorageOrder> empty(0, 0);
    // the LU rejects a rectangular input matrix
    EXPECT_THROW(static_cast<void>(PartialPivLU(rectangular)), std::invalid_argument);
    // the LU rejects an empty input matrix
    EXPECT_THROW(static_cast<void>(PartialPivLU(empty)), std::invalid_argument);

    matrix_type nonfinite = matrix;
    nonfinite(1, 1) = std::numeric_limits<double>::quiet_NaN();
    // the LU rejects a NaN input coefficient
    EXPECT_THROW(static_cast<void>(PartialPivLU(nonfinite)), std::invalid_argument);
    nonfinite(1, 1) = std::numeric_limits<double>::infinity();
    // the LU rejects an infinite input coefficient
    EXPECT_THROW(static_cast<void>(PartialPivLU(nonfinite)), std::invalid_argument);

    Matrix<double, Dynamic, 1, StorageOrder> wrong_rows(2);
    // the right-hand side must have one row for each matrix row
    EXPECT_THROW(static_cast<void>(factorization.solve(wrong_rows)), std::invalid_argument);
    Matrix<double, Dynamic, Dynamic, StorageOrder> no_columns(3, 0);
    // solving requires at least one right-hand-side column
    EXPECT_THROW(static_cast<void>(factorization.solve(no_columns)), std::invalid_argument);
    PartialPivLU<Matrix<double, Dynamic, Dynamic, StorageOrder>> unavailable;
    Matrix<double, Dynamic, 1, StorageOrder> zero_row_rhs(0);
    // an uncomputed factorization cannot solve a system
    EXPECT_THROW(static_cast<void>(unavailable.solve(zero_row_rhs)), std::domain_error);
    // an uncomputed factorization has no available determinant
    EXPECT_THROW(static_cast<void>(unavailable.determinant()), std::domain_error);
}

template <int StorageOrder> void check_partial_piv_lu_scale_and_determinant() {
    const Matrix<double, 2, 2, StorageOrder> large({1.0e308, 1.0e308, -1.0e308, 1.0e308});
    const Matrix<double, 2, 1, StorageOrder> large_rhs({1.0e308, 0.0});
    const PartialPivLU large_factorization(large);
    // normalization keeps the large-scale system numerically usable
    ASSERT_EQ(large_factorization.info(), 0);
    expect_matrix_near(large_factorization.solve(large_rhs), Matrix<double, 2, 1, StorageOrder>({0.5, 0.5}));

    const Matrix<double, 2, 2, StorageOrder> small_pivot({1.0, 0.0, 0.0, 1.0e-20});
    const PartialPivLU small_pivot_factorization(small_pivot);
    // the small second pivot falls below the numerical-rank threshold
    EXPECT_EQ(small_pivot_factorization.info(), 2);
    // the small-pivot matrix is treated as numerically rank one
    EXPECT_EQ(small_pivot_factorization.rank(), 1);
    // the algebraic determinant retains the small pivot instead of being forced to zero
    EXPECT_DOUBLE_EQ(small_pivot_factorization.determinant(), 1.0e-20);
    // the matrix determinant agrees with the factorization's nonzero algebraic determinant
    EXPECT_DOUBLE_EQ(small_pivot.determinant(), 1.0e-20);

    const Matrix<double, 4, 4, StorageOrder> matrix(
      {0.0, 3.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 5.0});
    // the signed permutation and diagonal factors produce determinant minus one hundred twenty
    EXPECT_DOUBLE_EQ(matrix.determinant(), -120.0);
    const Matrix<double, 4, 4, StorageOrder> inverse(matrix.inverse());
    Matrix<double, 4, 4, StorageOrder> identity;
    identity.set_zero();
    for (int i = 0; i < 4; ++i) identity(i, i) = 1.0;
    expect_matrix_near(Matrix<double, 4, 4, StorageOrder>(matrix * inverse), identity);

    const Matrix<double, 4, 4, StorageOrder> shifted(matrix + identity);
    const Matrix<double, 4, 4, StorageOrder> expression_inverse((matrix + identity).inverse());
    expect_matrix_near(Matrix<double, 4, 4, StorageOrder>(shifted * expression_inverse), identity);
    // determinant evaluation of a lazy sum agrees with its materialized matrix
    EXPECT_DOUBLE_EQ((matrix + identity).determinant(), shifted.determinant());

    Matrix<double, Dynamic, Dynamic, StorageOrder> dynamic(matrix);
    // dynamic storage produces the same determinant as fixed storage
    EXPECT_DOUBLE_EQ(dynamic.determinant(), -120.0);
    const Matrix<double, Dynamic, Dynamic, StorageOrder> dynamic_inverse(dynamic.inverse());
    expect_matrix_near(Matrix<double, Dynamic, Dynamic, StorageOrder>(dynamic * dynamic_inverse), identity);

    const Matrix<double, 4, 4, StorageOrder> mixed_exponents(
      {1.0e300, 0.0, 0.0, 0.0, 0.0, 1.0e-100, 0.0, 0.0, 0.0, 0.0, 1.0e-100, 0.0, 0.0, 0.0, 0.0, 1.0e-100});
    // opposing binary exponents cancel without losing the unit determinant
    EXPECT_DOUBLE_EQ(mixed_exponents.determinant(), 1.0);

    const double small = std::ldexp(1.0, -40);
    const double coefficient = std::ldexp(1.0, -20);
    const double large_value = std::ldexp(1.0, 40);
    const double larger_value = std::ldexp(1.0, 80);
    const Matrix<double, 3, 3, StorageOrder> complete_pivot_case(
      {small, 0.0, coefficient, 1.0, large_value, larger_value, small, 0.0, 0.0});
    // pivoting preserves the determinant sign in the tiny-coefficient example
    EXPECT_DOUBLE_EQ(complete_pivot_case.determinant(), -coefficient);

    constexpr int wilkinson_size = 130;
    Matrix<float, Dynamic, Dynamic, StorageOrder> wilkinson(wilkinson_size, wilkinson_size);
    wilkinson.set_zero();
    for (int row = 0; row < wilkinson_size; ++row) {
        wilkinson(row, row) = 0.5F;
        wilkinson(row, wilkinson_size - 1) = 0.5F;
        for (int col = 0; col < row; ++col) wilkinson(row, col) = -0.5F;
    }
    const PartialPivLU wilkinson_factorization(wilkinson);
    // nonfinite factor growth marks the LU factors unavailable
    EXPECT_EQ(wilkinson_factorization.info(), -1);
    // the algebraic determinant remains available even when LU factor growth overflows
    EXPECT_FLOAT_EQ(wilkinson_factorization.determinant(), 0.5F);
    // the matrix determinant uses the same independent determinant calculation
    EXPECT_FLOAT_EQ(wilkinson.determinant(), 0.5F);

    Matrix<double, Dynamic, Dynamic, StorageOrder> singular(4, 4);
    singular.set_zero();
    // a singular matrix cannot produce an inverse
    EXPECT_THROW(static_cast<void>(singular.inverse()), std::domain_error);
}

// checks pivoted LU solves, singular-state reporting and determinants across extreme coefficient scales
TEST(linear_algebra, partial_piv_lu) {
    check_partial_piv_lu_contracts<RowMajor>();
    check_partial_piv_lu_contracts<ColMajor>();
    check_partial_piv_lu_scale_and_determinant<RowMajor>();
    check_partial_piv_lu_scale_and_determinant<ColMajor>();
}

}   // namespace
