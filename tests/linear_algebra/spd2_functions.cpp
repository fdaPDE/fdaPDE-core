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

#include <cmath>
#include <limits>

namespace {
using namespace fdapde;

/// @brief constructs an analytical spectrum in a prescribed planar eigenframe
SymmetricMatrix<double, 2> rotated(double first, double second, double angle) {
    const double c = std::cos(angle), s = std::sin(angle);
    SymmetricMatrix<double, 2> value;
    value(0, 0) = c * c * first + s * s * second;
    value(1, 1) = s * s * first + c * c * second;
    value(1, 0) = c * s * (first - second);
    return value;
}

/// @brief checks every matrix coefficient against its independently prescribed spectral transform
template <typename Actual, typename Expected>
void expect_relative(const Actual& actual, const Expected& expected, double scale_floor = 1e-300) {
    double error = 0, scale = 0;
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j) {
            error = std::hypot(error, actual(i, j) - expected(i, j));
            scale = std::hypot(scale, expected(i, j));
        }
    // the coefficient residual is bounded relative to the analytical transform rather than its eigenvectors
    EXPECT_LE(error, 3e-12 * std::max(scale, scale_floor));
}

// scalar spectral kernels agree with independent rotated spectra at repeated values and extreme uniform scales
TEST(SPD2Functions, AnalyticalSpectraAndUniformScales) {
    for (double angle : {0., .27, 1.2})
        for (double gap : {0., 1e-12, .2, 5.})
            for (double scale : {1e-280, 1., 1e280}) {
                const double first = scale * (1 + gap), second = scale;
                const SPDMatrix<double, 2> point(rotated(first, second, angle));
                // logarithms retain the prescribed spectrum after rotation and uniform scaling
                expect_relative(matrix_log(point), rotated(std::log(first), std::log(second), angle), 1.);
                // principal square roots match the independently transformed eigenvalues
                expect_relative(matrix_sqrt(point), rotated(std::sqrt(first), std::sqrt(second), angle));
                // inverse roots retain the same eigenframe with reciprocal scalar roots
                expect_relative(matrix_inv_sqrt(point), rotated(1 / std::sqrt(first), 1 / std::sqrt(second), angle));
                const auto chart = rotated(std::log(first), std::log(second), angle);
                // exponentiating the independent chart reconstructs its prescribed positive spectrum
                expect_relative(matrix_exp(chart), rotated(first, second, angle));
            }
}

// a sub-ulp eigengap must retain the exponential derivative even when the rounded eigenvalues coincide
TEST(SPD2Functions, ExponentialSubUlpGap) {
    SymmetricMatrix<double, 2> value;
    value(0, 0) = value(1, 1) = 1.;
    value(1, 0) = 1e-18;
    const auto result = matrix_exp(value);
    // the off-diagonal sinh expansion has leading term exp(1) times the input coefficient
    EXPECT_NEAR(result(1, 0) / 1e-18, std::exp(1.), 1e-14);
}

// nonnegative spectral weights preserve small diagonal entries of conditioned and nearly axis-aligned exponentials
TEST(SPD2Functions, ExponentialConditionedDiagonalAndNearAxis) {
    for (double angle : {0., 1e-7, -1e-6, 1.5707963267948966 - 1e-7}) {
        const double minor = 1e-12, chart_minor = std::log(minor);
        const auto result = matrix_exp(rotated(0., chart_minor, angle));
        const auto expected = rotated(1., std::exp(chart_minor), angle);
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j <= i; ++j) {
                // each coefficient retains relative accuracy, including the minor diagonal hidden by a normwise check
                EXPECT_LE(std::abs(result(i, j) - expected(i, j)), 1e-12 * std::abs(expected(i, j)));
            }
    }
    for (float angle : {0.f, 1e-3f, -1e-3f}) {
        const float c = std::cos(angle), s = std::sin(angle), chart_minor = -10.f;
        SymmetricMatrix<float, 2> chart;
        chart(0, 0) = s * s * chart_minor;
        chart(1, 1) = c * c * chart_minor;
        chart(1, 0) = -c * s * chart_minor;
        const auto result = matrix_exp(chart);
        const float positive = std::exp(chart_minor);
        const float expected[] {c * c + s * s * positive, c * s * (1 - positive), s * s + c * c * positive};
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j <= i; ++j) {
                const float reference = expected[i * (i + 1) / 2 + j];
                // the native float transform retains each coefficient within scalar rounding accuracy
                EXPECT_LE(
                  std::abs(result(i, j) - reference), 32 * std::numeric_limits<float>::epsilon() * std::abs(reference));
            }
    }
}

// repeated subnormal spectra remain positive without first rounding both half-sums toward zero
TEST(SPD2Functions, ExponentialRepeatedSubnormalValues) {
    for (double value : {-744., -745.}) {
        SymmetricMatrix<double, 2> chart;
        chart(0, 0) = chart(1, 1) = value;
        chart(1, 0) = 0.;
        const auto result = matrix_exp(chart);
        for (int i = 0; i < 2; ++i) {
            // a scalar chart has exactly the independently evaluated scalar exponential on its diagonal
            EXPECT_EQ(result(i, i), std::exp(value));
        }
        // the scalar chart has no off-diagonal exponential component
        EXPECT_EQ(result(1, 0), 0.);
    }
    for (float value : {-100.f, -103.f}) {
        SymmetricMatrix<float, 2> chart;
        chart(0, 0) = chart(1, 1) = value;
        chart(1, 0) = 0.f;
        const auto result = matrix_exp(chart);
        for (int i = 0; i < 2; ++i) {
            // native float subnormals match the scalar exponential without a separately rounded half-sum
            EXPECT_EQ(result(i, i), std::exp(value));
        }
        // the repeated native float spectrum reconstructs a diagonal matrix
        EXPECT_EQ(result(1, 0), 0.f);
    }
}

// diagonal roots retain the small component without cancellation in the half-sum reconstruction
TEST(SPD2Functions, ConditionedDiagonalRoots) {
    const SPDMatrix<double, 2> point(rotated(1., 1e-12, 0.));
    const auto root = matrix_sqrt(point), inverse = matrix_inv_sqrt(point);
    // the principal root of the minor eigenvalue is exactly its scalar square root
    EXPECT_NEAR(root(1, 1), 1e-6, 1e-20);
    // the inverse root of the minor eigenvalue is its scalar reciprocal square root
    EXPECT_NEAR(inverse(1, 1), 1e6, 1e-8);
}

// output caches and selected input spectra remain coherent with the same mathematical transform
TEST(SPD2Functions, SelectedCachesRemainCoherent) {
    using Policy = Cache::Union<Cache::Log, Cache::Sqrt, Cache::InverseSqrt, Cache::Spectral>;
    const auto chart = rotated(.7, -.3, .4);
    const auto point = matrix_exp<Policy>(chart);
    // a selected logarithm cache preserves the input chart
    expect_relative(matrix_log(point), chart);
    const auto unretained = matrix_exp(chart);
    // cached and direct square roots represent the same principal root
    expect_relative(matrix_sqrt(point), matrix_sqrt(unretained));
    // cached and direct inverse roots represent the same reciprocal transform
    expect_relative(matrix_inv_sqrt(point), matrix_inv_sqrt(unretained));
}

// exact binary64 inputs are checked against independently computed 80-digit Decimal matrix functions
TEST(SPD2Functions, ConditionedRotatedHighPrecisionReferences) {
    {
        SymmetricMatrix<double, 2> value;
        value(0, 0) = 0.3996764710676651;
        value(1, 0) = 0.4898317971936516;
        value(1, 1) = 0.6003235289333348;
        const SPDMatrix<double, 2> point(value);
        SymmetricMatrix<double, 2> expected_log;
        expected_log(0, 0) = -16.587574290468588;
        expected_log(1, 0) = 13.534570834219721;
        expected_log(1, 1) = -11.043483782422069;
        // the log agrees with the high-precision transform of the rounded input matrix
        expect_relative(matrix_log(point), expected_log, 1.);
        SymmetricMatrix<double, 2> expected_sqrt;
        expected_sqrt(0, 0) = 0.39967707137950076;
        expected_sqrt(1, 0) = 0.4898313073713955;
        expected_sqrt(1, 1) = 0.6003239286020209;
        // the sqrt agrees with the high-precision transform of the rounded input matrix
        expect_relative(matrix_sqrt(point), expected_sqrt);
        SymmetricMatrix<double, 2> expected_inverse_sqrt;
        expected_inverse_sqrt(0, 0) = 600335.0217788541;
        expected_inverse_sqrt(1, 0) = -489840.35879355646;
        expected_inverse_sqrt(1, 1) = 399684.45687292813;
        // the inverse sqrt agrees with the high-precision transform of the rounded input matrix
        expect_relative(matrix_inv_sqrt(point), expected_inverse_sqrt);
    }
    {
        SymmetricMatrix<double, 2> value;
        value(0, 0) = 5.2461025473888045e-201;
        value(1, 0) = -4.993939680865319e-201;
        value(1, 1) = 4.753897452621195e-201;
        const SPDMatrix<double, 2> point(value);
        SymmetricMatrix<double, 2> expected_log;
        expected_log(0, 0) = -473.65253034370295;
        expected_log(1, 0) = -13.798773319198622;
        expected_log(1, 1) = -475.01254407299894;
        // the log agrees with the high-precision transform of the rounded input matrix
        expect_relative(matrix_log(point), expected_log, 1.);
        SymmetricMatrix<double, 2> expected_sqrt;
        expected_sqrt(0, 0) = 5.246107301243227e-101;
        expected_sqrt(1, 0) = -4.993934686970841e-101;
        expected_sqrt(1, 1) = 4.753902698676257e-101;
        // the sqrt agrees with the high-precision transform of the rounded input matrix
        expect_relative(matrix_sqrt(point), expected_sqrt);
        SymmetricMatrix<double, 2> expected_inverse_sqrt;
        expected_inverse_sqrt(0, 0) = 4.753940975246541e+105;
        expected_inverse_sqrt(1, 0) = 4.99397489618507e+105;
        expected_inverse_sqrt(1, 1) = 5.246149540852988e+105;
        // the inverse sqrt agrees with the high-precision transform of the rounded input matrix
        expect_relative(matrix_inv_sqrt(point), expected_inverse_sqrt);
    }
}
}   // namespace
