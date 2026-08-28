#include <fdaPDE/linear_algebra.h>
#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace {
using fdapde::Dynamic;
using fdapde::Matrix;
using fdapde::SparseMatrix;

template <typename ActualType, typename ExpectedType>
void expect_near(const ActualType& actual, const ExpectedType& expected) {
    ASSERT_EQ(actual.rows(), expected.rows());
    ASSERT_EQ(actual.cols(), expected.cols());
    for (int row = 0; row < actual.rows(); ++row) {
        for (int col = 0; col < actual.cols(); ++col) { EXPECT_NEAR(actual(row, col), expected(row, col), 1.0e-10); }
    }
}

TEST(linear_algebra, spectral_matrix_functions_preserve_native_dense_contracts) {
    const Matrix<double, 2, 2> source({2.0, 1.0, 1.0, 2.0});
    const auto logarithm = fdapde::logm(source);
    const auto exponential = fdapde::expm(logarithm);
    const auto squared = fdapde::powm(source, 2);
    const auto root = fdapde::sqrtm(source);
    static_assert(std::is_same_v<std::remove_cvref_t<decltype(logarithm)>, Matrix<double, 2, 2>>);
    static_assert(std::is_same_v<std::remove_cvref_t<decltype(squared)>, Matrix<double, 2, 2>>);

    expect_near(exponential, source);
    expect_near(squared, Matrix<double, 2, 2>({5.0, 4.0, 4.0, 5.0}));
    expect_near(root * root, source);
    Matrix<double, 2, 2> identity;
    identity.set_zero();
    identity(0, 0) = identity(1, 1) = 1.0;
    expect_near(fdapde::powm(source, 0), identity);

    const Matrix<double, 2, 2> diagonal({4.0, 0.0, 0.0, 9.0});
    expect_near(fdapde::powm(diagonal, -1), Matrix<double, 2, 2>({0.25, 0.0, 0.0, 1.0 / 9.0}));

    const Matrix<double, Dynamic, Dynamic> dynamic(source);
    const auto dynamic_root = fdapde::sqrtm(dynamic);
    static_assert(std::is_same_v<std::remove_cvref_t<decltype(dynamic_root)>, Matrix<double, Dynamic, Dynamic>>);
    expect_near(dynamic_root * dynamic_root, dynamic);

    const Matrix<double, 3, 3> partial_source({4.0, 0.0, 0.0, 0.0, 9.0, 0.0, 0.0, 0.0, 16.0});
    const Matrix<double, Dynamic, 3> partial_dynamic(partial_source);
    const auto partial_root = fdapde::sqrtm(partial_dynamic);
    static_assert(std::is_same_v<std::remove_cvref_t<decltype(partial_root)>, Matrix<double, Dynamic, 3>>);
    expect_near(partial_root * partial_root, partial_dynamic);
}

TEST(linear_algebra, spectral_matrix_functions_keep_runtime_contracts_in_release) {
    const Matrix<double, 2, 3> rectangular(2, 3);
    Matrix<double, Dynamic, Dynamic> empty(0, 2);
    const Matrix<double, 2, 2> asymmetric({2.0, 1.0, 0.0, 2.0});
    const Matrix<double, 2, 2> nonpositive({0.0, 0.0, 0.0, 1.0});
    const Matrix<double, 2, 2> negative({1.0, 0.0, 0.0, -1.0});
    const Matrix<double, 2, 2> singular({1.0, 0.0, 0.0, 0.0});
    const Matrix<double, 2, 2> overflowing({1000.0, 0.0, 0.0, 1000.0});
    Matrix<double, 2, 2> nonfinite({1.0, 0.0, 0.0, 1.0});
    nonfinite(0, 1) = std::numeric_limits<double>::quiet_NaN();

    EXPECT_THROW(static_cast<void>(fdapde::logm(rectangular)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(fdapde::expm(empty)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(fdapde::powm(asymmetric, 2)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(fdapde::sqrtm(nonfinite)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(fdapde::logm(nonpositive)), std::domain_error);
    EXPECT_THROW(static_cast<void>(fdapde::sqrtm(negative)), std::domain_error);
    EXPECT_THROW(static_cast<void>(fdapde::powm(singular, -1)), std::domain_error);
    EXPECT_THROW(static_cast<void>(fdapde::expm(overflowing)), std::domain_error);
}

TEST(linear_algebra, native_dense_and_sparse_emptiness_is_shape_based) {
    const Matrix<double, 2, 2> fixed = Matrix<double, 2, 2>::Zero();
    const Matrix<double, Dynamic, Dynamic> empty_rows(0, 3);
    const Matrix<double, Dynamic, Dynamic> empty_cols(3, 0);
    const SparseMatrix<double> sparse_default;
    const SparseMatrix<double> sparse_empty_rows(0, 3, {});
    const SparseMatrix<double> sparse_empty_cols(3, 0, {});
    const SparseMatrix<double> sparse_zero(2, 2, {});

    EXPECT_FALSE(fdapde::is_empty(fixed));
    EXPECT_TRUE(fdapde::is_empty(empty_rows));
    EXPECT_TRUE(fdapde::is_empty(empty_cols));
    EXPECT_TRUE(fdapde::is_empty(sparse_default));
    EXPECT_TRUE(fdapde::is_empty(sparse_empty_rows));
    EXPECT_TRUE(fdapde::is_empty(sparse_empty_cols));
    EXPECT_FALSE(fdapde::is_empty(sparse_zero));
}

}   // namespace
