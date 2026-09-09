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
// checks at compile time: !permits_rvalue_lu_factors<fixed_lu>
static_assert(!permits_rvalue_lu_factors<fixed_lu>);
// checks at compile time: std::is_same_v<typename const_view_lu::Scalar, double>
static_assert(std::is_same_v<typename const_view_lu::Scalar, double>);
constexpr Matrix<int, 2, 2> integral_matrix({1, 2, 3, 4});
// checks at compile time: integral_matrix.determinant() == -2
static_assert(integral_matrix.determinant() == -2);

template <typename Actual, typename Expected>
void expect_matrix_near(const Actual& actual, const Expected& expected, double tolerance = 1.0e-12) {
    // compares actual.rows(), expected.rows() using eq semantics
    ASSERT_EQ(actual.rows(), expected.rows());
    // compares actual.cols(), expected.cols() using eq semantics
    ASSERT_EQ(actual.cols(), expected.cols());
    for (int row = 0; row < actual.rows(); ++row) {
        for (int col = 0; col < actual.cols(); ++col) {
            // compares the computed and expected values within the stated absolute tolerance
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
    // compares factorization.info(), 0 using eq semantics
    EXPECT_EQ(factorization.info(), 0);
    // compares factorization.rank(), 3 using eq semantics
    EXPECT_EQ(factorization.rank(), 3);
    // compares factorization.determinant(), -7.0 using double_eq semantics
    EXPECT_DOUBLE_EQ(factorization.determinant(), -7.0);
    expect_matrix_near(factorization.solve(rhs), expected);
    const Matrix<double, 3, 2, StorageOrder> expected_multiple({1.0, -1.0, 2.0, 0.5, -3.0, 4.0});
    const Matrix<double, 3, 2, StorageOrder> rhs_multiple(matrix * expected_multiple);
    expect_matrix_near(factorization.solve(rhs_multiple), expected_multiple);
    expect_matrix_near(
      Matrix<double, 3, 3>(factorization.P() * matrix), Matrix<double, 3, 3>(factorization.L() * factorization.U()));

    const MatrixView<const double, 3, 3, StorageOrder> const_view(matrix.data());
    const PartialPivLU const_view_factorization(const_view);
    // compares const_view_factorization.determinant(), -7.0 using double_eq semantics
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
    // compares singular_factorization.info(), 2 using eq semantics
    EXPECT_EQ(singular_factorization.info(), 2);
    // compares singular_factorization.rank(), 2 using eq semantics
    EXPECT_EQ(singular_factorization.rank(), 2);
    // compares singular_factorization.determinant(), 0.0 using double_eq semantics
    EXPECT_DOUBLE_EQ(singular_factorization.determinant(), 0.0);
    expect_matrix_near(
      Matrix<double, 3, 3>(singular_factorization.P() * singular),
      Matrix<double, 3, 3>(singular_factorization.L() * singular_factorization.U()));
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(singular_factorization.solve(rhs)), std::domain_error);

    const matrix_type zero = matrix_type::Zero();
    const PartialPivLU zero_factorization(zero);
    // compares zero_factorization.info(), 1 using eq semantics
    EXPECT_EQ(zero_factorization.info(), 1);
    // compares zero_factorization.rank(), 0 using eq semantics
    EXPECT_EQ(zero_factorization.rank(), 0);
    // compares zero_factorization.determinant(), 0.0 using double_eq semantics
    EXPECT_DOUBLE_EQ(zero_factorization.determinant(), 0.0);
    const Matrix<double, 3, 3> zero_reconstruction(zero_factorization.L() * zero_factorization.U());
    expect_matrix_near(Matrix<double, 3, 3>(zero_factorization.P() * zero), zero_reconstruction);
    for (int row = 0; row < 3; ++row) {
        // checks std::isfinite(zero_reconstruction(row, col))
        for (int col = 0; col < 3; ++col) EXPECT_TRUE(std::isfinite(zero_reconstruction(row, col)));
    }

    Matrix<double, Dynamic, Dynamic, StorageOrder> rectangular(2, 3);
    Matrix<double, Dynamic, Dynamic, StorageOrder> empty(0, 0);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(PartialPivLU(rectangular)), std::invalid_argument);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(PartialPivLU(empty)), std::invalid_argument);

    matrix_type nonfinite = matrix;
    nonfinite(1, 1) = std::numeric_limits<double>::quiet_NaN();
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(PartialPivLU(nonfinite)), std::invalid_argument);
    nonfinite(1, 1) = std::numeric_limits<double>::infinity();
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(PartialPivLU(nonfinite)), std::invalid_argument);

    Matrix<double, Dynamic, 1, StorageOrder> wrong_rows(2);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(factorization.solve(wrong_rows)), std::invalid_argument);
    Matrix<double, Dynamic, Dynamic, StorageOrder> no_columns(3, 0);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(factorization.solve(no_columns)), std::invalid_argument);
    PartialPivLU<Matrix<double, Dynamic, Dynamic, StorageOrder>> unavailable;
    Matrix<double, Dynamic, 1, StorageOrder> zero_row_rhs(0);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(unavailable.solve(zero_row_rhs)), std::domain_error);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(unavailable.determinant()), std::domain_error);
}

template <int StorageOrder> void check_partial_piv_lu_scale_and_determinant() {
    const Matrix<double, 2, 2, StorageOrder> large({1.0e308, 1.0e308, -1.0e308, 1.0e308});
    const Matrix<double, 2, 1, StorageOrder> large_rhs({1.0e308, 0.0});
    const PartialPivLU large_factorization(large);
    // compares large_factorization.info(), 0 using eq semantics
    ASSERT_EQ(large_factorization.info(), 0);
    expect_matrix_near(large_factorization.solve(large_rhs), Matrix<double, 2, 1, StorageOrder>({0.5, 0.5}));

    const Matrix<double, 2, 2, StorageOrder> small_pivot({1.0, 0.0, 0.0, 1.0e-20});
    const PartialPivLU small_pivot_factorization(small_pivot);
    // compares small_pivot_factorization.info(), 2 using eq semantics
    EXPECT_EQ(small_pivot_factorization.info(), 2);
    // compares small_pivot_factorization.rank(), 1 using eq semantics
    EXPECT_EQ(small_pivot_factorization.rank(), 1);
    // compares small_pivot_factorization.determinant(), 1.0e-20 using double_eq semantics
    EXPECT_DOUBLE_EQ(small_pivot_factorization.determinant(), 1.0e-20);
    // compares small_pivot.determinant(), 1.0e-20 using double_eq semantics
    EXPECT_DOUBLE_EQ(small_pivot.determinant(), 1.0e-20);

    const Matrix<double, 4, 4, StorageOrder> matrix(
      {0.0, 3.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 5.0});
    // compares matrix.determinant(), -120.0 using double_eq semantics
    EXPECT_DOUBLE_EQ(matrix.determinant(), -120.0);
    const Matrix<double, 4, 4, StorageOrder> inverse(matrix.inverse());
    Matrix<double, 4, 4, StorageOrder> identity;
    identity.set_zero();
    for (int i = 0; i < 4; ++i) identity(i, i) = 1.0;
    expect_matrix_near(Matrix<double, 4, 4, StorageOrder>(matrix * inverse), identity);

    const Matrix<double, 4, 4, StorageOrder> shifted(matrix + identity);
    const Matrix<double, 4, 4, StorageOrder> expression_inverse((matrix + identity).inverse());
    expect_matrix_near(Matrix<double, 4, 4, StorageOrder>(shifted * expression_inverse), identity);
    // compares (matrix + identity).determinant(), shifted.determinant() using double_eq semantics
    EXPECT_DOUBLE_EQ((matrix + identity).determinant(), shifted.determinant());

    Matrix<double, Dynamic, Dynamic, StorageOrder> dynamic(matrix);
    // compares dynamic.determinant(), -120.0 using double_eq semantics
    EXPECT_DOUBLE_EQ(dynamic.determinant(), -120.0);
    const Matrix<double, Dynamic, Dynamic, StorageOrder> dynamic_inverse(dynamic.inverse());
    expect_matrix_near(Matrix<double, Dynamic, Dynamic, StorageOrder>(dynamic * dynamic_inverse), identity);

    const Matrix<double, 4, 4, StorageOrder> mixed_exponents(
      {1.0e300, 0.0, 0.0, 0.0, 0.0, 1.0e-100, 0.0, 0.0, 0.0, 0.0, 1.0e-100, 0.0, 0.0, 0.0, 0.0, 1.0e-100});
    // compares mixed_exponents.determinant(), 1.0 using double_eq semantics
    EXPECT_DOUBLE_EQ(mixed_exponents.determinant(), 1.0);

    const double small = std::ldexp(1.0, -40);
    const double coefficient = std::ldexp(1.0, -20);
    const double large_value = std::ldexp(1.0, 40);
    const double larger_value = std::ldexp(1.0, 80);
    const Matrix<double, 3, 3, StorageOrder> complete_pivot_case(
      {small, 0.0, coefficient, 1.0, large_value, larger_value, small, 0.0, 0.0});
    // compares complete_pivot_case.determinant(), -coefficient using double_eq semantics
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
    // compares wilkinson_factorization.info(), -1 using eq semantics
    EXPECT_EQ(wilkinson_factorization.info(), -1);
    // compares wilkinson_factorization.determinant(), 0.5F using float_eq semantics
    EXPECT_FLOAT_EQ(wilkinson_factorization.determinant(), 0.5F);
    // compares wilkinson.determinant(), 0.5F using float_eq semantics
    EXPECT_FLOAT_EQ(wilkinson.determinant(), 0.5F);

    Matrix<double, Dynamic, Dynamic, StorageOrder> singular(4, 4);
    singular.set_zero();
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(singular.inverse()), std::domain_error);
}

// verifies partial piv lu through the public algebra API
TEST(linear_algebra, partial_piv_lu) {
    check_partial_piv_lu_contracts<RowMajor>();
    check_partial_piv_lu_contracts<ColMajor>();
    check_partial_piv_lu_scale_and_determinant<RowMajor>();
    check_partial_piv_lu_scale_and_determinant<ColMajor>();
}

}   // namespace
