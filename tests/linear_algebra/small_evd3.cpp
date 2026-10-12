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
/// @brief retains exact binary64 input coefficients and independently computed eighty-digit reference values
struct GoldenCase {
    std::array<double, 9> matrix;
    std::array<double, 3> values;
    std::array<double, 9> logarithm;
};
const std::array golden {
  GoldenCase {
              {1.3591616216487361e+200, -1.3838516085330096e+200, 4.330815965690849e+199, -1.3838516085330096e+200,
     1.4098393155887576e+200, -4.714150031979887e+199, 4.330815965690849e+199, -4.714150031979887e+199,
     1.230999062763507e+200},
              {1.0002181503480936e+188, 1.0000000000000001e+200, 3.0000000000000002e+200},
              {447.0536901978275, -14.2951190351582, -0.13937748801707314, -14.2951190351582, 447.3316863935414,
     -0.6327526911944482, -0.13937748801707314, -0.6327526911944482, 460.6334885043548}                    },
  GoldenCase {
              {1.9088787226976736, 0.08767370216524675, -1.274189470574059, 0.08767370216524675, 0.8339858408404105,
     -0.6394408353212535, -1.274189470574059, -0.6394408353212535, 1.2571354364629166},
              {9.999522472175166e-13, 1.0000000000000002, 3.0000000000000004},
              {-5.27411489398202, -6.312001855986233, -9.786698541880673, -6.312001855986233, -7.097788771235022,
     -10.361589937122588, -9.786698541880673, -10.361589937122588, -14.16055291596608}                     },
  GoldenCase {
              {0.03243097553374951, -0.1443437138880442, 0.2301954157861202, -0.1443437138880442, 1.0377955590534536,
     -0.3087927917036066, 0.2301954157861202, -0.3087927917036066, 2.9297734654137972},
              {9.999970349236574e-13, 1.0, 3.0000000000000004},
              {-27.1180916787673, -3.25517727755224, 1.8806523308521612, -3.25517727755224, -0.35861275153211336,
     0.03992581102343716, 1.8806523308521612, 0.03992581102343716, 0.944292637958235}                      },
  GoldenCase {
              {1.0, 4.940916064069407e-16, -1.6835500695047716e-15, 4.940916064069407e-16, 1.0000000000000124,
     6.458547299132897e-15, -1.6835500695047716e-15, 6.458547299132897e-15, 1.0000000000000275},
              {0.9999999999999998, 1.0000000000000102, 1.00000000000003},
              {-1.5392336760256327e-30, 4.940916064069431e-16, -1.68355006950475e-15, 4.940916064069431e-16,
     1.2434497875801655e-14, 6.458547299132769e-15, -1.68355006950475e-15, 6.458547299132769e-15,
     2.753353101070348e-14}                                                                                },
  GoldenCase {
              {2.6475907880462404, 1.0095744795919575, -1.8394772878252017, 1.0095744795919575, 1.6141940230372223,
     -1.1872655133984644, -1.8394772878252017, -1.1872655133984644, 2.2382151889165356},
              {0.49999999999999944, 0.9999999999999997, 4.999999999999999},
              {0.5476701034034112, 0.31239033479755396, -0.9261826331751944, 0.31239033479755396, 0.17075196036078666,
     -0.6291616014226176, -0.9261826331751944, -0.6291616014226176, 0.19786866810995563}                   },
  GoldenCase {
              {1e-200, 0.0, 0.0, 0.0, 1e-200, 0.0, 0.0, 0.0, 3e-200},
              {1e-200, 1e-200, 3e-200},
              {-460.51701859880916, 0.0, 0.0, 0.0, -460.51701859880916, 0.0, 0.0, 0.0, -459.41840631014105}},
  GoldenCase {
              {1.0001037758268895e+200, -5.141922971200433e+195, 1.0186383731722831e+198, -5.141922971200433e+195,
     1.0000254773906952e+200, -5.047187004727717e+197, 1.0186383731722831e+198, -5.047187004727717e+197,
     1.9998707467834154e+200},
              {1e+200, 1.0000000000010002e+200, 2e+200},
              {460.51709053073097, -3.564109410610321e-05, 0.007060663163745351, -3.564109410610321e-05, 460.517036258391,
     -0.003498443442084216, 0.007060663163745351, -0.003498443442084216, 461.21007618786643}               },
  GoldenCase {
              {1.0053226808084872, -0.06096376521535084, 0.03972114308860671, -0.06096376521535084, 1.9962635313005295,
     0.002434508553293577, 0.03972114308860671, 0.002434508553293577, 1.9984137878909938},
              {1.0000000000000002, 2.0, 2.00000000000001},
              {0.003689401195423478, -0.04225686197533889, 0.02753259834048582, -0.04225686197533889, 0.6905572578156568,
     0.0016874727397645695, 0.02753259834048582, 0.0016874727397645695, 0.6920477021088156}                },
  GoldenCase {
              {1.0000000000000003e-200, -7.976187008953857e-216, 1.5069572130998995e-215, -7.976187008953857e-216,
     1.0000000000000101e-200, -4.8848104398619915e-216, 1.5069572130998995e-215, -4.8848104398619915e-216,
     1.00000000000003e-200},
              {1.0000000000000001e-200, 1.0000000000000101e-200, 1.0000000000000302e-200},
              {-460.51701859880916, -7.976187008953812e-16, 1.5069572130998765e-15, -7.976187008953812e-16, -460.5170185988091,
     -4.884810439861887e-16, 1.5069572130998765e-15, -4.884810439861887e-16, -460.5170185988091}           },
  GoldenCase {
              {0.9999999999999999, 1.6191808682419709e-19, -1.6049886891684536e-17, 1.6191808682419709e-19, 1.0,
     2.1763278136883484e-17, -1.6049886891684536e-17, 2.1763278136883484e-17, 1.0},
              {0.9999999999999999, 1.0, 1.0},
              {-1.1102230246251565e-16, 1.6191808682419728e-19, -1.6049886891684536e-17, 1.6191808682419728e-19,
     -2.368332463650957e-34, 2.1763278136883484e-17, -1.6049886891684536e-17, 2.1763278136883484e-17,
     -3.656195722496089e-34}                                                                               },
};
/// @brief checks normalized eigen-equation and orthogonality residuals without overflowing extreme matrix scales
void check_residual(const auto& matrix, const auto& decomposition, double tolerance) {
    const auto& q = decomposition.eigenvectors();
    const auto& values = decomposition.eigenvalues();
    double scale = 0, residual = 0, gram = 0;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) scale = std::max(scale, std::abs(double(matrix(i, j))));
    if (scale == 0) scale = 1;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            double image = 0, pairing = 0;
            for (int k = 0; k < 3; ++k) {
                image += matrix(i, k) / scale * q(k, j);
                pairing += double(q(k, i)) * q(k, j);
            }
            residual = std::max(residual, std::abs(image - values[j] / scale * q(i, j)));
            gram = std::max(gram, std::abs(pairing - double(i == j)));
        }
    // the eigen-equation is measured against independently retained input coefficients at their original scale
    EXPECT_LT(residual, tolerance);
    // the three output columns form a complete orthonormal basis rather than only individually normalized vectors
    EXPECT_LT(gram, tolerance);
}

// ill-conditioned and repeated spectra are checked against Decimal eigenpairs for the exact stored binary64 matrices
TEST(SmallEVD3, IndependentHighPrecisionEigenvaluesAndLogarithms) {
    for (const auto& example : golden) {
        SymmetricMatrix<double, 3> matrix;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j <= i; ++j) matrix(i, j) = example.matrix[3 * i + j];
        const EVD decomposition(matrix);
        std::array<double, 3> actual;
        for (int i = 0; i < 3; ++i) actual[i] = decomposition.eigenvalues()[i];
        std::sort(actual.begin(), actual.end());
        for (int i = 0; i < 3; ++i) {
            // the reference diagonalizes the exact input at eighty digits, including its smallest
            // cancellation-sensitive root
            EXPECT_LT(std::abs(actual[i] - example.values[i]) / example.values[i], 3e-14);
        }
        check_residual(matrix, decomposition, 5e-15);
        const auto logarithm = matrix_log(SPDMatrix<double, 3>(matrix));
        double error = 0, norm = 0;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) {
                error = std::hypot(error, logarithm(i, j) - example.logarithm[3 * i + j]);
                norm = std::hypot(norm, example.logarithm[3 * i + j]);
            }
        // an independent high-precision matrix logarithm checks eigenvectors and eigenvalues together
        EXPECT_LT(error / std::max(1., norm), 1e-13);
        const SymmetricMatrix<double, Dynamic> runtime(matrix);
        const EVD dynamic(runtime);
        std::array<double, 3> dynamic_values;
        for (int i = 0; i < 3; ++i) dynamic_values[i] = dynamic.eigenvalues()[i];
        std::sort(dynamic_values.begin(), dynamic_values.end());
        for (int i = 0; i < 3; ++i) {
            // runtime order-three dispatch retains the same accurate eigenvalue specialization
            EXPECT_EQ(dynamic_values[i], actual[i]);
        }
    }
}

// the certified kernel handles indefinite, singular, zero and diagonal matrices across extreme finite scales
TEST(SmallEVD3, GeneralSymmetricAndExtremeScaleContracts) {
    const std::array<std::array<double, 9>, 5> inputs {
      {{0, 2, 0, 2, 0, 0, 0, 0, -3},
       {1, 2, 3, 2, 4, 6, 3, 6, 9},
       {0, 1, 1, 1, 0, 1, 1, 1, 0},
       {0, 0, 0, 0, 0, 0, 0, 0, 0},
       {-2, 0, 0, 0, 1e-12, 0, 0, 0, 3}}
    };
    const std::array<std::array<double, 3>, 5> expected {
      {{-3, -2, 2}, {0, 0, 14}, {-1, -1, 2}, {0, 0, 0}, {-2, 1e-12, 3}}
    };
    for (double scale : {1e-280, 1., 1e280})
        for (std::size_t index = 0; index < inputs.size(); ++index) {
            SymmetricMatrix<double, 3> matrix;
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j <= i; ++j) matrix(i, j) = scale * inputs[index][3 * i + j];
            const EVD decomposition(matrix);
            std::array<double, 3> actual;
            for (int i = 0; i < 3; ++i) actual[i] = decomposition.eigenvalues()[i] / scale;
            std::sort(actual.begin(), actual.end());
            for (int i = 0; i < 3; ++i) {
                // exact integer spectra supply independent signed and null eigenvalue oracles at every scale
                EXPECT_NEAR(actual[i], expected[index][i], 1e-14);
            }
            check_residual(matrix, decomposition, 5e-15);
        }
}

// float and two-dimensional runtime paths preserve their established scalar precision and matrix shapes
TEST(SmallEVD3, FloatAndRuntimeTwoDimensionalDispatch) {
    SymmetricMatrix<float, 3> matrix;
    matrix(0, 0) = 2;
    matrix(1, 1) = 3;
    matrix(2, 2) = 4;
    matrix(1, 0) = .3f;
    matrix(2, 0) = -.2f;
    matrix(2, 1) = .1f;
    const EVD decomposition(matrix);
    check_residual(matrix, decomposition, 2e-6);
    const SymmetricMatrix<float, Dynamic> runtime(matrix);
    const EVD dynamic(runtime);
    for (int i = 0; i < 3; ++i) {
        // the float runtime specialization preserves the established fixed-order Jacobi values
        EXPECT_EQ(dynamic.eigenvalues()[i], decomposition.eigenvalues()[i]);
    }
    SymmetricMatrix<double, Dynamic> pair(2, 2);
    pair(0, 0) = pair(1, 1) = 2;
    pair(1, 0) = 1;
    const EVD two(pair);
    std::array<double, 2> values {two.eigenvalues()[0], two.eigenvalues()[1]};
    std::sort(values.begin(), values.end());
    // the runtime two-by-two kernel recovers the independently known lower eigenvalue
    EXPECT_NEAR(values[0], 1., 1e-15);
    // the runtime two-by-two kernel recovers the independently known upper eigenvalue
    EXPECT_NEAR(values[1], 3., 1e-15);
}

// fixed and runtime order-two caches preserve tiny eigenvalues against exact-input eighty-digit references
TEST(SmallEVD2, CompensatedEigenvaluesAndCachedLogarithms) {
    /// @brief retains exact two-dimensional input coefficients and eighty-digit eigenvalue and logarithm references
    struct Example {
        std::array<double, 3> matrix;
        std::array<double, 2> values;
        std::array<double, 3> logarithm;
    };
    const std::array examples {
      Example {
               {2.5386492326389995e-200, -8.425301206351125e-201, 1.4613507673610008e-200},
               {1e-200, 3e-200},
               {-459.67182912134564, -0.4628069720513797, -460.2635957876045}         },
      Example {
               {2.826942293075604, -0.5622867986382583, 1.1730577069243955},
               {0.9999999999999999, 2.9999999999999996},
               {1.003550626930177, -0.30886759336992076, 0.0950616617379325}          },
      Example {
               {8.95245735585577e+199, -3.062365237872701e+199, 1.0475426441542285e+199},
               {1.0000153825498231e+188, 9.999999999999998e+199},
               {457.6225529181631, -8.461623144580724, 435.7804785459582}             },
      Example {
               {9.687910308653712e-201, -1.738820559464125e-201, 3.1208969135628743e-202},
               {1.0000015461255172e-212, 1e-200},
               {-461.37935423572236, -4.804538490697921, -487.28570253170017}         },
      Example {
               {9.750585325018833e+199, -1.5594675596068915e+199, 2.494146759811648e+198},
               {9.999998830987682e+189, 9.999999999999998e+199},
               {459.94272008325044, -3.5908067743480823, 438.06546606752613}          },
      Example {
               {1.0000000000000193, 1.3779854526851576e-13, 1.0000000000009808},
               {1.0, 1.000000000001},
               {1.9317880628468043e-14, 1.3779854526844685e-13, 9.807710199533728e-13}},
    };
    for (const auto& example : examples) {
        SymmetricMatrix<double, 2> matrix;
        matrix(0, 0) = example.matrix[0];
        matrix(1, 0) = example.matrix[1];
        matrix(1, 1) = example.matrix[2];
        const EVD decomposition(matrix);
        std::array<double, 2> values {decomposition.eigenvalues()[0], decomposition.eigenvalues()[1]};
        std::sort(values.begin(), values.end());
        for (int i = 0; i < 2; ++i) {
            // the exact stored binary64 matrix supplies the reference, preserving conditioning effects from input
            // rounding
            EXPECT_LT(std::abs(values[i] - example.values[i]) / example.values[i], 3e-14);
        }
        const SPDMatrix<double, 2, Cache::Spectral> cached(matrix);
        const auto logarithm = matrix_log(cached);
        const SPDMatrix<double, Dynamic, Cache::Spectral> runtime(cached);
        const auto dynamic_logarithm = matrix_log(runtime);
        double error = 0, norm = 0;
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j) {
                const double expected = example.logarithm[i == j ? 2 * i : 1];
                error = std::hypot(error, logarithm(i, j) - expected);
                norm = std::hypot(norm, expected);
                // runtime spectral caches use the same native order-two kernel and preserve its accurate logarithm
                EXPECT_NEAR(dynamic_logarithm(i, j), logarithm(i, j), 1e-13);
            }
        // cached logarithms agree with independently evaluated matrix functions despite extreme scale and conditioning
        EXPECT_LT(error / std::max(1., norm), 1e-13);
    }
}
}   // namespace
