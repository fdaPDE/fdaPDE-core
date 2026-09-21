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

#include <Eigen/Dense>
#include <random>

namespace {
using native_matrix = fdapde::Matrix<double, fdapde::Dynamic, fdapde::Dynamic>;

// creates repeatable non-diagonal inputs while preserving the historical test dimensions
Eigen::MatrixXd sampled_matrix(int rows, int cols) {
    std::mt19937 engine(1729);
    std::uniform_real_distribution<double> sample(-1., 1.);
    Eigen::MatrixXd matrix(rows, cols);
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) matrix(row, col) = sample(engine);
    }
    return matrix;
}

// copies the same coefficients into the native owner used by the production algorithms
native_matrix native_copy(const Eigen::MatrixXd& source) {
    native_matrix result(source.rows(), source.cols());
    for (int row = 0; row < result.rows(); ++row) {
        for (int col = 0; col < result.cols(); ++col) result(row, col) = source(row, col);
    }
    return result;
}

// compares all three requested singular values with an independent full Jacobi SVD
void check_svd(const Eigen::MatrixXd& source) {
    const native_matrix matrix = native_copy(source);
    const fdapde::RSI<native_matrix> rsi(matrix, 3, 1.e-8, 150, 42);
    const fdapde::RBKI<native_matrix> rbki(matrix, 3, 1.e-8, 150, 42);
    const Eigen::JacobiSVD<Eigen::MatrixXd> oracle(source);
    double rsi_error = 0.;
    double rbki_error = 0.;
    for (int i = 0; i < 3; ++i) {
        rsi_error = std::hypot(rsi_error, rsi.singular_values()[i] - oracle.singularValues()[i]);
        rbki_error = std::hypot(rbki_error, rbki.singular_values()[i] - oracle.singularValues()[i]);
    }
    // the native subspace iteration satisfies the historical Euclidean spectral-error tolerance
    EXPECT_LT(rsi_error, 1.e-3);
    // the native Krylov iteration satisfies the same independent spectral-error oracle
    EXPECT_LT(rbki_error, 1.e-3);
}

// compares the leading PSD eigenvalues against an independent full Jacobi SVD
void check_psd(const Eigen::MatrixXd& generator) {
    const Eigen::MatrixXd source = generator * generator.transpose();
    const native_matrix matrix = native_copy(source);
    const fdapde::NysRSI<native_matrix> rsi(matrix, 3, 1.e-8, 150, 42);
    const fdapde::NysRBKI<native_matrix> rbki(matrix, 3, 1.e-8, 150, 42);
    const Eigen::JacobiSVD<Eigen::MatrixXd> oracle(source);
    double rsi_error = 0.;
    double rbki_error = 0.;
    for (int i = 0; i < 3; ++i) {
        rsi_error = std::hypot(rsi_error, rsi.eigenvalues()[i] - oracle.singularValues()[i]);
        rbki_error = std::hypot(rbki_error, rbki.eigenvalues()[i] - oracle.singularValues()[i]);
    }
    // the subspace Nyström approximation satisfies the historical PSD spectral-error tolerance
    EXPECT_LT(rsi_error, 1.e-4);
    // the Krylov Nyström approximation matches the same independent PSD spectrum
    EXPECT_LT(rbki_error, 1.e-4);
}

// the historical square case compares rank-three approximations of a 20-by-20 sampled matrix
TEST(randomized_eigen, square_spectrum) {
    // the full Eigen SVD supplies the leading-value oracle for both native methods
    check_svd(sampled_matrix(20, 20));
}

// the historical rectangular case exercises both orientations of the same sampled matrix
TEST(randomized_eigen, rectangular_spectrum) {
    const auto source = sampled_matrix(10, 20);
    // the wide input is compared with its full Eigen SVD
    check_svd(source);
    // transposition preserves the spectrum and exercises the native tall-matrix path
    check_svd(source.transpose());
}

// a square generator yields the historical full-rank PSD spectrum comparison
TEST(randomized_eigen, full_rank_psd) {
    // the independent SVD of the Gram matrix supplies both Nyström spectral oracles
    check_psd(sampled_matrix(20, 20));
}

// a tall generator yields a rank-deficient PSD matrix with the historical dimensions
TEST(randomized_eigen, deficient_psd) {
    // a 40-by-20 generator leaves twenty zero modes in the 40-by-40 Gram matrix
    check_psd(sampled_matrix(40, 20));
}
}   // namespace
