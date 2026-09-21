#include <fdaPDE/dense_linear_algebra.h>
#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace {
using fdapde::Dynamic;
using fdapde::Matrix;

template <typename ActualType, typename ExpectedType>
// compares matrix shapes before checking each coefficient against the supplied oracle
void expect_near(const ActualType& actual, const ExpectedType& expected) {
    // the result row count agrees with the oracle before coefficient access
    ASSERT_EQ(actual.rows(), expected.rows());
    // the result column count agrees with the oracle before coefficient access
    ASSERT_EQ(actual.cols(), expected.cols());
    for (int row = 0; row < actual.rows(); ++row) {
        for (int col = 0; col < actual.cols(); ++col) {
            // each coefficient agrees with the independent expected value within absolute tolerance
            EXPECT_NEAR(actual(row, col), expected(row, col), 1.0e-10);
        }
    }
}

// native spectral functions preserve fixed and dynamic shapes and satisfy explicit powers and reconstruction identities
TEST(linear_algebra, spectral_matrix_functions_preserve_native_dense_contracts) {
    const Matrix<double, 2, 2> source({2.0, 1.0, 1.0, 2.0});
    const auto logarithm = fdapde::logm(source);
    const auto exponential = fdapde::expm(logarithm);
    const auto squared = fdapde::powm(source, 2);
    const auto root = fdapde::sqrtm(source);
    // logarithm preserves the fixed shape and materializes a double matrix
    static_assert(std::is_same_v<std::remove_cvref_t<decltype(logarithm)>, Matrix<double, 2, 2>>);
    // integer powers preserve the fixed shape and materialize a double matrix
    static_assert(std::is_same_v<std::remove_cvref_t<decltype(squared)>, Matrix<double, 2, 2>>);

    // exponentiating the logarithm recovers the original symmetric matrix
    expect_near(exponential, source);
    // squaring agrees with the explicitly calculated coefficients
    expect_near(squared, Matrix<double, 2, 2>({5.0, 4.0, 4.0, 5.0}));
    // multiplying the principal root by itself recovers the input
    expect_near(root * root, source);
    Matrix<double, 2, 2> identity;
    identity.set_zero();
    identity(0, 0) = identity(1, 1) = 1.0;
    // exponent zero returns the identity for this nonsingular input
    expect_near(fdapde::powm(source, 0), identity);

    const Matrix<double, 2, 2> diagonal({4.0, 0.0, 0.0, 9.0});
    // negative exponent reciprocates the known diagonal eigenvalues
    expect_near(fdapde::powm(diagonal, -1), Matrix<double, 2, 2>({0.25, 0.0, 0.0, 1.0 / 9.0}));

    const Matrix<double, Dynamic, Dynamic> dynamic(source);
    const auto dynamic_root = fdapde::sqrtm(dynamic);
    // a fully dynamic input produces a fully dynamic owned result
    static_assert(std::is_same_v<std::remove_cvref_t<decltype(dynamic_root)>, Matrix<double, Dynamic, Dynamic>>);
    // the dynamic principal root satisfies the same square identity as the fixed result
    expect_near(dynamic_root * dynamic_root, dynamic);

    const Matrix<double, 3, 3> partial_source({4.0, 0.0, 0.0, 0.0, 9.0, 0.0, 0.0, 0.0, 16.0});
    const Matrix<double, Dynamic, 3> partial_dynamic(partial_source);
    const auto partial_root = fdapde::sqrtm(partial_dynamic);
    // the result retains the compile-time column count of partially dynamic input
    static_assert(std::is_same_v<std::remove_cvref_t<decltype(partial_root)>, Matrix<double, Dynamic, 3>>);
    // the partially dynamic principal root reconstructs the diagonal input
    expect_near(partial_root * partial_root, partial_dynamic);
}

// native spectral functions reject invalid shapes, asymmetry, nonfinite coefficients and unsupported spectral domains
TEST(linear_algebra, spectral_matrix_functions_reject_invalid_inputs_and_results) {
    const Matrix<double, 2, 3> rectangular(2, 3);
    Matrix<double, Dynamic, Dynamic> empty(0, 2);
    const Matrix<double, 2, 2> asymmetric({2.0, 1.0, 0.0, 2.0});
    const Matrix<double, 2, 2> nonpositive({0.0, 0.0, 0.0, 1.0});
    const Matrix<double, 2, 2> negative({1.0, 0.0, 0.0, -1.0});
    const Matrix<double, 2, 2> singular({1.0, 0.0, 0.0, 0.0});
    const Matrix<double, 2, 2> overflowing({1000.0, 0.0, 0.0, 1000.0});
    Matrix<double, 2, 2> nonfinite({1.0, 0.0, 0.0, 1.0});
    nonfinite(0, 1) = std::numeric_limits<double>::quiet_NaN();

    // rectangular input is rejected even when its shape is known at compile time
    EXPECT_THROW(static_cast<void>(fdapde::logm(rectangular)), std::invalid_argument);
    // an empty axis is rejected before eigendecomposition
    EXPECT_THROW(static_cast<void>(fdapde::expm(empty)), std::invalid_argument);
    // integer powers reject asymmetric input instead of selecting one triangle
    EXPECT_THROW(static_cast<void>(fdapde::powm(asymmetric, 2)), std::invalid_argument);
    // NaN in the upper triangle is rejected before a symmetric view is formed
    EXPECT_THROW(static_cast<void>(fdapde::sqrtm(nonfinite)), std::invalid_argument);
    // logarithm rejects a zero eigenvalue
    EXPECT_THROW(static_cast<void>(fdapde::logm(nonpositive)), std::domain_error);
    // square root rejects a genuinely negative eigenvalue
    EXPECT_THROW(static_cast<void>(fdapde::sqrtm(negative)), std::domain_error);
    // negative powers reject a singular matrix
    EXPECT_THROW(static_cast<void>(fdapde::powm(singular, -1)), std::domain_error);
    // exponentiation reports overflow instead of returning nonfinite coefficients
    EXPECT_THROW(static_cast<void>(fdapde::expm(overflowing)), std::domain_error);
}

// closed-form eigenvectors give coefficient oracles independently of the spectral implementation
TEST(linear_algebra, spectral_functions_match_closed_form_coefficients) {
    const Matrix<int, 2, 2> source({2, 1, 1, 2});
    const double half_log_three = std::log(3.0) / 2.0;
    const double root_diagonal = (std::sqrt(3.0) + 1.0) / 2.0;
    const double root_off_diagonal = (std::sqrt(3.0) - 1.0) / 2.0;
    // eigenvectors (1,1) and (1,-1) give diagonal log(3)/2 and equal off-diagonal entries
    expect_near(
      fdapde::logm(source), Matrix<double, 2, 2>({half_log_three, half_log_three, half_log_three, half_log_three}));
    // the principal square root has eigenvalues sqrt(3) and one in the same eigenvector basis
    expect_near(
      fdapde::sqrtm(source),
      Matrix<double, 2, 2>({root_diagonal, root_off_diagonal, root_off_diagonal, root_diagonal}));
    const Matrix<double, 2, 2> diagonal({4.0, 0.0, 0.0, 9.0});
    // a nonidentity root detects accidental truncation of a fractional exponent to zero
    expect_near(fdapde::sqrtm(diagonal), Matrix<double, 2, 2>({2.0, 0.0, 0.0, 3.0}));
    const Matrix<double, 2, 2> symmetric({0.0, 1.0, 1.0, 0.0});
    // eigenvalues plus and minus one yield cosh on the diagonal and sinh off the diagonal
    expect_near(
      fdapde::expm(symmetric), Matrix<double, 2, 2>({std::cosh(1.0), std::sinh(1.0), std::sinh(1.0), std::cosh(1.0)}));
}

// semidefinite roots accept zero eigenvalues and small roundoff but retain a meaningful negativity threshold
TEST(linear_algebra, spectral_square_root_distinguishes_semidefinite_input_from_indefiniteness) {
    const Matrix<double, 2, 2> semidefinite({0.0, 0.0, 0.0, 4.0});
    // the zero eigenvalue stays zero while the positive eigenvalue has its principal root
    expect_near(fdapde::sqrtm(semidefinite), Matrix<double, 2, 2>({0.0, 0.0, 0.0, 2.0}));
    const Matrix<double, 2, 2> roundoff({-1.0e-15, 0.0, 0.0, 4.0});
    // negativity within the relative tolerance is clamped to zero
    expect_near(fdapde::sqrtm(roundoff), Matrix<double, 2, 2>({0.0, 0.0, 0.0, 2.0}));
    const Matrix<double, 2, 2> indefinite({-1.0e-10, 0.0, 0.0, 4.0});
    // a negative eigenvalue beyond roundoff is a domain error rather than an implicit projection
    EXPECT_THROW(fdapde::sqrtm(indefinite), std::domain_error);
}

// native spectral functions materialize temporary expressions and honor column-major coefficient access
TEST(linear_algebra, spectral_results_own_temporary_expression_values) {
    const auto root = [] {
        const Matrix<double, 2, 2, fdapde::ColMajor> source({2.0, 1.0, 1.0, 2.0});
        return fdapde::sqrtm(source * 2.0);
    }();
    // the returned root remains usable after both the expression and its column-major owner have died
    expect_near(root * root, Matrix<double, 2, 2>({4.0, 2.0, 2.0, 4.0}));
}

}   // namespace
