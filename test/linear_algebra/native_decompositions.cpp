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

#include <algorithm>
#include <cmath>
#include <type_traits>
#include <utility>

namespace {

namespace native = fdapde::linalg;

template <typename Decomposition>
concept permits_rvalue_qr_factors = requires(Decomposition decomposition) {
    std::move(decomposition).Q();
    std::move(decomposition).R();
};
template <typename Decomposition>
concept permits_rvalue_evd_factors = requires(Decomposition decomposition) {
    std::move(decomposition).eigenvalues();
    std::move(decomposition).eigenvectors();
};
template <typename Decomposition>
concept permits_rvalue_lu_factors = requires(Decomposition decomposition) {
    std::move(decomposition).P();
    std::move(decomposition).L();
    std::move(decomposition).U();
};

using fixed_lu = native::PartialPivLU<native::Matrix<double, 3, 3>>;
using fixed_qr = native::HouseholderQR<double, 3, 2>;
using fixed_symmetric = native::SymmetricMatrix<double, 3, 3>;
using fixed_evd = decltype(std::declval<const fixed_symmetric&>().evd());
static_assert(!permits_rvalue_qr_factors<fixed_qr>);
static_assert(!permits_rvalue_evd_factors<fixed_evd>);
static_assert(!permits_rvalue_lu_factors<fixed_lu>);

template <typename Actual, typename Expected>
void expect_matrix_near(const Actual& actual, const Expected& expected, double tolerance = 1.0e-12) {
    ASSERT_EQ(actual.rows(), expected.rows());
    ASSERT_EQ(actual.cols(), expected.cols());
    for (int i = 0; i < actual.rows(); ++i) {
        for (int j = 0; j < actual.cols(); ++j) {
            EXPECT_NEAR(static_cast<double>(actual(i, j)), static_cast<double>(expected(i, j)), tolerance);
        }
    }
}

template <typename Actual, typename Expected>
void expect_relative_matrix_near(const Actual& actual, const Expected& expected, double tolerance) {
    using matrix_type = native::Matrix<double, fdapde::Dynamic, fdapde::Dynamic>;
    const matrix_type actual_dense(actual);
    const matrix_type expected_dense(expected);
    const double scale = expected_dense.norm();
    const double denominator = scale > 0.0 ? scale : 1.0;
    EXPECT_LT((actual_dense - expected_dense).norm() / denominator, tolerance);
}

template <typename SymmetricMatrix> void expect_valid_evd(const SymmetricMatrix& matrix, double tolerance = 1.0e-10) {
    const auto decomposition = matrix.evd();
    ASSERT_TRUE(decomposition.computed());
    const auto eigenvectors_view = decomposition.eigenvectors();
    const auto& eigenvalues = decomposition.eigenvalues();
    using Scalar = typename SymmetricMatrix::Scalar;
    constexpr int Rows = SymmetricMatrix::Rows;
    constexpr int Cols = SymmetricMatrix::Cols;
    native::Matrix<Scalar, Rows, Cols> eigenvectors(eigenvectors_view);
    native::Matrix<Scalar, Rows, Cols> diagonal;
    native::Matrix<Scalar, Rows, Cols> identity;
    if constexpr (Rows == fdapde::Dynamic || Cols == fdapde::Dynamic) {
        diagonal.resize(matrix.rows(), matrix.cols());
        identity.resize(matrix.rows(), matrix.cols());
    }
    diagonal.set_zero();
    identity.set_zero();
    for (int i = 0; i < matrix.rows(); ++i) {
        diagonal(i, i) = eigenvalues[i];
        identity(i, i) = Scalar(1);
    }
    const native::Matrix<Scalar, Rows, Cols> dense(matrix);
    const double scale = dense.norm();
    const double denominator = scale > 0.0 ? scale : 1.0;
    EXPECT_LT((dense * eigenvectors - eigenvectors * diagonal).norm() / denominator, tolerance);
    EXPECT_LT((eigenvectors.transpose() * eigenvectors - identity).norm(), tolerance);
    EXPECT_LT((eigenvectors * diagonal * eigenvectors.transpose() - dense).norm() / denominator, tolerance);
    for (const auto eigenvalue : eigenvalues) EXPECT_TRUE(std::isfinite(eigenvalue));
}

TEST(NativePartialPivLU, PivotingSolvesAndReconstructs) {
    const native::Matrix<double, 3, 3> matrix({0.0, 2.0, 1.0, 1.0, -2.0, -3.0, 2.0, 3.0, 1.0});
    const native::Vector<double, 3> expected({2.0, -1.0, 3.0});
    const native::Vector<double, 3> rhs(matrix * expected);

    const native::PartialPivLU factorization(matrix);
    EXPECT_EQ(factorization.info(), 0);
    EXPECT_EQ(factorization.rank(), 3);
    EXPECT_DOUBLE_EQ(factorization.determinant(), -7.0);
    expect_matrix_near(factorization.solve(rhs), expected);
    expect_matrix_near(
      native::Matrix<double, 3, 3>(factorization.P() * matrix),
      native::Matrix<double, 3, 3>(factorization.L() * factorization.U()));

    const native::Matrix<double, 3, 2> expected_multiple({1.0, -1.0, 2.0, 0.5, -3.0, 4.0});
    const native::Matrix<double, 3, 2> rhs_multiple(matrix * expected_multiple);
    expect_matrix_near(factorization.solve(rhs_multiple), expected_multiple);
}

TEST(NativePartialPivLU, ReportsSingularAndZeroMatrices) {
    const native::Matrix<double, 3, 3> singular({1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0});
    const native::PartialPivLU singular_factorization(singular);
    EXPECT_EQ(singular_factorization.info(), 2);
    EXPECT_EQ(singular_factorization.rank(), 2);
    EXPECT_DOUBLE_EQ(singular_factorization.determinant(), 0.0);
    expect_matrix_near(
      native::Matrix<double, 3, 3>(singular_factorization.P() * singular),
      native::Matrix<double, 3, 3>(singular_factorization.L() * singular_factorization.U()));

    const native::Matrix<double, 3, 3> zero(native::Matrix<double, 3, 3>::Zero());
    const native::PartialPivLU zero_factorization(zero);
    EXPECT_EQ(zero_factorization.info(), 1);
    EXPECT_EQ(zero_factorization.rank(), 0);
    EXPECT_DOUBLE_EQ(zero_factorization.determinant(), 0.0);
}

TEST(NativePartialPivLU, PowersGenericInverseAndDeterminant) {
    const native::Matrix<int, 2, 2> integer_matrix({1, 2, 3, 4});
    EXPECT_EQ(integer_matrix.determinant(), -2);

    const native::Matrix<double, 4, 4> matrix(
      {0.0, 3.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 5.0});
    EXPECT_DOUBLE_EQ(matrix.determinant(), -120.0);
    const native::Matrix<double, 4, 4> inverse(matrix.inverse());
    const native::Matrix<double, 4, 4> identity = native::IdentityMatrix<double, 4, 4>();
    expect_matrix_near(native::Matrix<double, 4, 4>(matrix * inverse), identity);

    native::Matrix<double, fdapde::Dynamic, fdapde::Dynamic> dynamic(matrix);
    EXPECT_DOUBLE_EQ(dynamic.determinant(), -120.0);
    const native::Matrix<double, fdapde::Dynamic, fdapde::Dynamic> dynamic_inverse(dynamic.inverse());
    expect_matrix_near(native::Matrix<double, fdapde::Dynamic, fdapde::Dynamic>(dynamic * dynamic_inverse), identity);

    const native::Matrix<double, 4, 4> expression_inverse((matrix + identity).inverse());
    expect_matrix_near(
      native::Matrix<double, 4, 4>((matrix + identity) * expression_inverse), native::Matrix<double, 4, 4>(identity));
}

TEST(NativePartialPivLU, SolvesFiniteSystemsWithoutEliminationOverflow) {
    const native::Matrix<double, 2, 2> matrix({1.0e308, 1.0e308, -1.0e308, 1.0e308});
    const native::Vector<double, 2> rhs({1.0e308, 0.0});
    const native::PartialPivLU factorization(matrix);
    ASSERT_EQ(factorization.info(), 0);
    EXPECT_EQ(factorization.rank(), 2);
    expect_matrix_near(factorization.solve(rhs), native::Vector<double, 2>({0.5, 0.5}), 1.0e-12);

    const native::Matrix<double, 2, 2> singular({1.0e308, 1.0e308, 1.0e308, 1.0e308});
    EXPECT_DOUBLE_EQ(singular.determinant(), 0.0);

    const native::Matrix<double, 2, 2> small_pivot({1.0, 0.0, 0.0, 1.0e-20});
    const native::PartialPivLU small_pivot_factorization(small_pivot);
    EXPECT_GT(small_pivot_factorization.info(), 0);
    EXPECT_DOUBLE_EQ(small_pivot.determinant(), 1.0e-20);
    EXPECT_DOUBLE_EQ(small_pivot_factorization.determinant(), 1.0e-20);

    const native::Matrix<double, 2, 2> mixed_exponents({1.0e-200, 0.0, 0.0, 1.0e200});
    const native::PartialPivLU mixed_exponents_factorization(mixed_exponents);
    EXPECT_DOUBLE_EQ(mixed_exponents.determinant(), 1.0);
    EXPECT_DOUBLE_EQ(mixed_exponents_factorization.determinant(), 1.0);

    const native::Matrix<double, 3, 3> mixed_three({1.0e300, 0.0, 0.0, 0.0, 1.0e-200, 0.0, 0.0, 0.0, 1.0e-200});
    EXPECT_DOUBLE_EQ(mixed_three.determinant(), 1.0e-100);

    const native::Matrix<double, 4, 4> mixed_four(
      {1.0e300, 0.0, 0.0, 0.0, 0.0, 1.0e-100, 0.0, 0.0, 0.0, 0.0, 1.0e-100, 0.0, 0.0, 0.0, 0.0, 1.0e-100});
    EXPECT_DOUBLE_EQ(mixed_four.determinant(), 1.0);

    const double small = std::ldexp(1.0, -40);
    const double coefficient = std::ldexp(1.0, -20);
    const double large = std::ldexp(1.0, 40);
    const double larger = std::ldexp(1.0, 80);
    const native::Matrix<double, 3, 3> complete_pivot_case(
      {small, 0.0, coefficient, 1.0, large, larger, small, 0.0, 0.0});
    EXPECT_DOUBLE_EQ(complete_pivot_case.determinant(), -coefficient);

    constexpr int wilkinson_size = 130;
    native::Matrix<float, fdapde::Dynamic, fdapde::Dynamic> wilkinson(wilkinson_size, wilkinson_size);
    wilkinson.set_zero();
    for (int row = 0; row < wilkinson_size; ++row) {
        wilkinson(row, row) = 0.5F;
        wilkinson(row, wilkinson_size - 1) = 0.5F;
        for (int col = 0; col < row; ++col) wilkinson(row, col) = -0.5F;
    }
    const native::PartialPivLU wilkinson_factorization(wilkinson);
    EXPECT_EQ(wilkinson_factorization.info(), -1);
    EXPECT_FLOAT_EQ(wilkinson_factorization.determinant(), 0.5F);
    EXPECT_FLOAT_EQ(wilkinson.determinant(), 0.5F);
}

TEST(NativeHouseholderQR, ReconstructsTallAndWideMatrices) {
    const native::Matrix<double, 4, 3> tall({1.0, 2.0, -1.0, 2.0, 0.0, 3.0, -1.0, 4.0, 2.0, 3.0, -2.0, 1.0});
    const native::HouseholderQR tall_qr(tall);
    ASSERT_TRUE(tall_qr.computed());
    EXPECT_EQ(tall_qr.rank(), 3);
    expect_relative_matrix_near(native::Matrix<double, 4, 3>(tall_qr.Q() * tall_qr.R()), tall, 1.0e-12);
    const native::Matrix<double, 4, 4> tall_identity = native::IdentityMatrix<double, 4, 4>();
    expect_matrix_near(native::Matrix<double, 4, 4>(tall_qr.Q().transpose() * tall_qr.Q()), tall_identity, 1.0e-12);

    native::Matrix<double, fdapde::Dynamic, fdapde::Dynamic> wide(2, 3);
    wide(0, 0) = 1.0;
    wide(0, 1) = -2.0;
    wide(0, 2) = 3.0;
    wide(1, 0) = 4.0;
    wide(1, 1) = 1.0;
    wide(1, 2) = -1.0;
    const native::HouseholderQR wide_qr(wide);
    ASSERT_TRUE(wide_qr.computed());
    EXPECT_EQ(wide_qr.rank(), 2);
    expect_relative_matrix_near(
      native::Matrix<double, fdapde::Dynamic, fdapde::Dynamic>(wide_qr.Q() * wide_qr.R()), wide, 1.0e-12);
    const native::Matrix<double, 2, 2> wide_identity = native::IdentityMatrix<double, 2, 2>();
    expect_matrix_near(
      native::Matrix<double, fdapde::Dynamic, fdapde::Dynamic>(wide_qr.Q().transpose() * wide_qr.Q()), wide_identity,
      1.0e-12);
}

TEST(NativeHouseholderQR, IsScaleStableAndDetectsRankDeficiency) {
    for (const double scale : {1.0e-150, 1.0e150}) {
        const native::Matrix<double, 3, 2> matrix({scale, 2.0 * scale, -3.0 * scale, scale, 2.0 * scale, -scale});
        const native::HouseholderQR qr(matrix);
        ASSERT_TRUE(qr.computed());
        EXPECT_EQ(qr.rank(), 2);
        expect_relative_matrix_near(native::Matrix<double, 3, 2>(qr.Q() * qr.R()), matrix, 1.0e-12);
    }

    const native::Matrix<double, 2, 2> mixed_scale({1.0, 0.0, 0.0, 1.0e-160});
    const native::HouseholderQR mixed_scale_qr(mixed_scale);
    ASSERT_TRUE(mixed_scale_qr.computed());
    EXPECT_EQ(mixed_scale_qr.rank(), 1);
    expect_relative_matrix_near(
      native::Matrix<double, 2, 2>(mixed_scale_qr.Q() * mixed_scale_qr.R()), mixed_scale, 1.0e-12);
    const native::Matrix<double, 2, 2> mixed_scale_identity = native::IdentityMatrix<double, 2, 2>();
    expect_matrix_near(
      native::Matrix<double, 2, 2>(mixed_scale_qr.Q().transpose() * mixed_scale_qr.Q()), mixed_scale_identity, 1.0e-12);

    const native::Matrix<double, 4, 3> rank_deficient({1.0, 2.0, 3.0, 2.0, 4.0, 1.0, 3.0, 6.0, -1.0, 4.0, 8.0, 2.0});
    const native::HouseholderQR qr(rank_deficient);
    EXPECT_EQ(qr.rank(), 2);
    expect_relative_matrix_near(native::Matrix<double, 4, 3>(qr.Q() * qr.R()), rank_deficient, 1.0e-12);

    const native::Matrix<double, 3, 3> leading_zero_column({0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0});
    const native::HouseholderQR shifted_qr(leading_zero_column);
    EXPECT_EQ(shifted_qr.rank(), 2);
    expect_relative_matrix_near(
      native::Matrix<double, 3, 3>(shifted_qr.Q() * shifted_qr.R()), leading_zero_column, 1.0e-12);
}

TEST(NativeEVD, TwoByTwoAndRepeatedDynamicSpectrum) {
    const native::SymmetricMatrix<double, 2, 2> two_by_two({2.0, 1.0, 3.0});
    const auto decomposition = two_by_two.evd();
    const double minimum = std::min(decomposition.eigenvalues()[0], decomposition.eigenvalues()[1]);
    const double maximum = std::max(decomposition.eigenvalues()[0], decomposition.eigenvalues()[1]);
    EXPECT_NEAR(minimum, (5.0 - std::sqrt(5.0)) / 2.0, 1.0e-12);
    EXPECT_NEAR(maximum, (5.0 + std::sqrt(5.0)) / 2.0, 1.0e-12);
    expect_valid_evd(two_by_two);

    native::SymmetricMatrix<double, fdapde::Dynamic, fdapde::Dynamic> repeated(4, 4);
    for (int row = 0; row < 4; ++row) {
        for (int col = row; col < 4; ++col) repeated(row, col) = row == col ? 2.75 : 0.75;
    }
    const auto repeated_decomposition = repeated.evd();
    int repeated_twos = 0;
    int repeated_fives = 0;
    for (const double eigenvalue : repeated_decomposition.eigenvalues()) {
        repeated_twos += std::abs(eigenvalue - 2.0) < 1.0e-10;
        repeated_fives += std::abs(eigenvalue - 5.0) < 1.0e-10;
    }
    EXPECT_EQ(repeated_twos, 3);
    EXPECT_EQ(repeated_fives, 1);
    expect_valid_evd(repeated);
}

TEST(NativeEVD, PreservesScaleAndHandlesZero) {
    for (const double scale : {1.0e-20, 1.0e150}) {
        const native::SymmetricMatrix<double, 2, 2> matrix({2.0 * scale, scale, 2.0 * scale});
        const auto decomposition = matrix.evd();
        const double minimum = std::min(decomposition.eigenvalues()[0], decomposition.eigenvalues()[1]);
        const double maximum = std::max(decomposition.eigenvalues()[0], decomposition.eigenvalues()[1]);
        EXPECT_NEAR(minimum / scale, 1.0, 1.0e-12);
        EXPECT_NEAR(maximum / scale, 3.0, 1.0e-12);
        expect_valid_evd(matrix);
    }

    const native::SymmetricMatrix<double, 3, 3> zero({0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
    const auto decomposition = zero.evd();
    EXPECT_EQ(decomposition.eigenvalues(), (native::Vector<double, 3>({0.0, 0.0, 0.0})));
    const native::Matrix<double, 3, 3> identity = native::IdentityMatrix<double, 3, 3>();
    expect_matrix_near(native::Matrix<double, 3, 3>(decomposition.eigenvectors()), identity);
}

}   // namespace
