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

#ifndef __FDAPDE_LINALG_RBKI_H__
#define __FDAPDE_LINALG_RBKI_H__

#include <cmath>
#include <memory>
#include <random>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "header_check.h"
#include "randomized_svd.h"

namespace fdapde {

// Randomized block Krylov iteration for a compact rank-revealing SVD.
// The stopping tolerance is absolute, matching the direct API introduced in
// b4b43f8 and officially exposed by 53b5e92.
template <typename MatrixType_>
    requires(internals::matrix_expression<MatrixType_>)
class RBKI {
   public:
    using MatrixType = std::remove_cvref_t<MatrixType_>;
    using Scalar = std::remove_cv_t<typename MatrixType::Scalar>;
    using FactorType = Matrix<Scalar, Dynamic, Dynamic, MatrixType::StorageOrder>;
    using SingularValuesType = Vector<Scalar, Dynamic>;
    fdapde_static_assert(std::is_floating_point_v<Scalar>, RBKI_REQUIRES_FLOATING_POINT_SCALARS);

    RBKI() : seed_(resolve_seed_(random_seed)) { }
    RBKI(const RBKI&) = default;
    RBKI& operator=(const RBKI&) = default;
    RBKI(const MatrixType& matrix, int rank) : RBKI() { compute(matrix, rank); }
    RBKI(const MatrixType& matrix, int rank, Scalar tolerance, int max_iterations, int seed = random_seed) :
        RBKI(tolerance, max_iterations, seed) {
        compute(matrix, rank);
    }
    RBKI(Scalar tolerance, int max_iterations, int seed = random_seed) :
        tolerance_(tolerance), max_iterations_(max_iterations), seed_(resolve_seed_(seed)) {
        validate_configuration_();
    }

    void compute(const MatrixType& matrix, int rank) {
        const int minimum_dimension = fdapde::min(matrix.rows(), matrix.cols());
        compute(matrix, rank, minimum_dimension <= 100 ? 1 : 10);
    }

    void compute(const MatrixType& matrix, int rank, int block_size) {
        validate_configuration_();
        const int rows = matrix.rows();
        const int cols = matrix.cols();
        const int minimum_dimension = fdapde::min(rows, cols);
        if (rows <= 0 || cols <= 0) { throw std::invalid_argument("RBKI requires a nonempty matrix"); }
        (void)internals::checked_matrix_size(rows, cols);
        if (rank <= 0 || rank > minimum_dimension) {
            throw std::invalid_argument("RBKI rank must be positive and no larger than either matrix dimension");
        }
        if (block_size <= 0 || block_size > minimum_dimension) {
            throw std::invalid_argument("RBKI block size must be positive and fit both matrix dimensions");
        }

        Scalar scale = Scalar(0);
        for (int row = 0; row < rows; ++row) {
            for (int col = 0; col < cols; ++col) {
                const Scalar value = static_cast<Scalar>(matrix(row, col));
                if (!std::isfinite(value)) { throw std::invalid_argument("RBKI requires finite matrix coefficients"); }
                scale = fdapde::max(scale, fdapde::abs(value));
            }
        }

        const Scalar normalization = scale == Scalar(0) ? Scalar(1) : scale;
        const bool transposed = rows > cols;
        FactorType source(transposed ? cols : rows, transposed ? rows : cols);
        for (int row = 0; row < rows; ++row) {
            for (int col = 0; col < cols; ++col) {
                const Scalar value = static_cast<Scalar>(matrix(row, col)) / normalization;
                if (transposed) {
                    source(col, row) = value;
                } else {
                    source(row, col) = value;
                }
            }
        }

        std::mt19937 engine(seed_);
        std::normal_distribution<Scalar> normal(Scalar(0), Scalar(1));
        FactorType omega(source.cols(), block_size);
        for (int row = 0; row < omega.rows(); ++row) {
            for (int col = 0; col < omega.cols(); ++col) {
                omega(row, col) = normal(engine);
                if (!std::isfinite(omega(row, col))) {
                    throw std::domain_error("RBKI Gaussian sampling produced a nonfinite coefficient");
                }
            }
        }

        FactorType range = Ops::orthonormalize(Ops::multiply(source, omega));
        FactorType last_corange = Ops::transpose_multiply(source, range);
        typename Ops::Result candidate = Ops::compact_state(source, range, fdapde::min(rank, range.cols()));
        Scalar residual = Ops::residual(source, candidate);
        const Scalar normalized_tolerance = tolerance_ / normalization;

        for (int expansion = 0;
             residual > normalized_tolerance && expansion < max_iterations_ && range.cols() < minimum_dimension;
             ++expansion) {
            const int added_columns = fdapde::min(block_size, minimum_dimension - range.cols());
            const FactorType raw_full = Ops::multiply(source, last_corange);
            FactorType raw(raw_full.rows(), added_columns);
            for (int row = 0; row < raw.rows(); ++row) {
                for (int col = 0; col < raw.cols(); ++col) raw(row, col) = raw_full(row, col);
            }
            FactorType next = Ops::orthonormalize(raw, range);
            range = Ops::append_columns(range, next);
            last_corange = Ops::transpose_multiply(source, next);
            candidate = Ops::compact_state(source, range, fdapde::min(rank, range.cols()));
            residual = Ops::residual(source, candidate);
        }

        for (int i = 0; i < candidate.values.rows(); ++i) {
            candidate.values[i] *= normalization;
            if (!std::isfinite(candidate.values[i])) {
                throw std::domain_error("RBKI singular values exceed the supported scalar range");
            }
        }
        if (transposed) {
            publish_(std::move(candidate.right), std::move(candidate.left), std::move(candidate.values));
        } else {
            publish_(std::move(candidate.left), std::move(candidate.right), std::move(candidate.values));
        }
    }

    const FactorType& matrixU() const& { return state_->left; }
    void matrixU() const&& = delete;
    const FactorType& matrixV() const& { return state_->right; }
    void matrixV() const&& = delete;
    const SingularValuesType& singularValues() const& { return state_->values; }
    void singularValues() const&& = delete;
    int rank() const { return state_->values.rows(); }
   private:
    using Ops = internals::randomized_svd_ops<FactorType>;

    struct State {
        FactorType left;
        FactorType right;
        SingularValuesType values;

        State() = default;
        State(FactorType&& left_, FactorType&& right_, SingularValuesType&& values_) :
            left(std::move(left_)), right(std::move(right_)), values(std::move(values_)) { }
    };

    void publish_(FactorType&& left, FactorType&& right, SingularValuesType&& values) {
        auto replacement = std::make_shared<const State>(std::move(left), std::move(right), std::move(values));
        state_.swap(replacement);
    }

    void validate_configuration_() const {
        if (!std::isfinite(tolerance_) || tolerance_ < Scalar(0)) {
            throw std::invalid_argument("RBKI tolerance must be finite and nonnegative");
        }
        if (max_iterations_ <= 0) { throw std::invalid_argument("RBKI maximum iterations must be positive"); }
    }

    static unsigned int resolve_seed_(int seed) {
        return seed == random_seed ? std::random_device {}() : static_cast<unsigned int>(seed);
    }

    std::shared_ptr<const State> state_ = std::make_shared<State>();
    Scalar tolerance_ = Scalar(1.0e-5);
    int max_iterations_ = 50;
    unsigned int seed_;
};

// Randomized block Krylov iteration with a Nyström approximation for a
// symmetric positive-semidefinite matrix; Tropp and Webber (2023), Algorithm
// 5.8. The initial Gaussian block is part of the Krylov basis.
template <typename MatrixType_>
    requires(internals::matrix_expression<MatrixType_>)
class NysRBKI {
   public:
    using MatrixType = std::remove_cvref_t<MatrixType_>;
    using Scalar = std::remove_cv_t<typename MatrixType::Scalar>;
    using FactorType = Matrix<Scalar, Dynamic, Dynamic, MatrixType::StorageOrder>;
    using EigenValuesType = Vector<Scalar, Dynamic>;
    fdapde_static_assert(std::is_floating_point_v<Scalar>, NYS_RBKI_REQUIRES_FLOATING_POINT_SCALARS);
    fdapde_static_assert(
      MatrixType::Rows == Dynamic || MatrixType::Cols == Dynamic || MatrixType::Rows == MatrixType::Cols,
      NYS_RBKI_REQUIRES_A_SQUARE_MATRIX);

    NysRBKI() : seed_(resolve_seed_(random_seed)) { }
    NysRBKI(const NysRBKI&) = default;
    NysRBKI& operator=(const NysRBKI&) = default;
    NysRBKI(const MatrixType& matrix, int rank) : NysRBKI() { compute(matrix, rank); }
    NysRBKI(const MatrixType& matrix, int rank, Scalar tolerance, int max_iterations, int seed = random_seed) :
        NysRBKI(tolerance, max_iterations, seed) {
        compute(matrix, rank);
    }
    NysRBKI(Scalar tolerance, int max_iterations, int seed = random_seed) :
        tolerance_(tolerance), max_iterations_(max_iterations), seed_(resolve_seed_(seed)) {
        validate_configuration_();
    }

    void compute(const MatrixType& matrix, int rank) { compute(matrix, rank, matrix.rows() <= 100 ? 1 : 10); }

    void compute(const MatrixType& matrix, int rank, int block_size) {
        validate_configuration_();
        const int rows = matrix.rows();
        const int cols = matrix.cols();
        if (rows <= 0 || rows != cols) { throw std::invalid_argument("NysRBKI requires a nonempty square matrix"); }
        (void)internals::checked_matrix_size(rows, cols);
        if (rank <= 0 || rank > rows) {
            throw std::invalid_argument("NysRBKI rank must be positive and no larger than the matrix dimension");
        }
        if (block_size <= 0 || block_size > rows) {
            throw std::invalid_argument("NysRBKI block size must be positive and fit the matrix dimension");
        }

        FactorType source(rows, cols);
        Scalar scale = Scalar(0);
        for (int row = 0; row < rows; ++row) {
            for (int col = 0; col < cols; ++col) {
                const Scalar value = static_cast<Scalar>(matrix(row, col));
                if (!std::isfinite(value)) {
                    throw std::invalid_argument("NysRBKI requires finite matrix coefficients");
                }
                source(row, col) = value;
                scale = fdapde::max(scale, fdapde::abs(value));
            }
        }
        const Scalar normalization = scale == Scalar(0) ? Scalar(1) : scale;
        if (scale != Scalar(0)) {
            for (int row = 0; row < rows; ++row) {
                for (int col = 0; col < cols; ++col) source(row, col) /= normalization;
            }
        }

        const Scalar roundoff = Scalar(64) * std::numeric_limits<Scalar>::epsilon() * static_cast<Scalar>(rows);
        // ponytail: exact PSD certification is cubic and defeats this low-rank routine. Match RpChol's cheap
        // necessary-condition screen and reject any projected-factorization breakdown below.
        for (int row = 0; row < rows; ++row) {
            if (source(row, row) < -roundoff) {
                throw std::domain_error("NysRBKI requires a positive-semidefinite matrix");
            }
            for (int col = row + 1; col < cols; ++col) {
                if (fdapde::abs(source(row, col) - source(col, row)) > roundoff) {
                    throw std::invalid_argument("NysRBKI requires a symmetric matrix");
                }
                const Scalar diagonal_product = source(row, row) * source(col, col);
                const Scalar coefficient_square = source(row, col) * source(row, col);
                if (coefficient_square > diagonal_product + roundoff) {
                    throw std::domain_error("NysRBKI requires a positive-semidefinite matrix");
                }
            }
        }

        const int initial_rank = fdapde::min(rank, block_size);
        if (scale == Scalar(0)) {
            FactorType vectors(rows, initial_rank);
            vectors.set_zero();
            for (int i = 0; i < initial_rank; ++i) vectors(i, i) = Scalar(1);
            EigenValuesType values(initial_rank);
            for (int i = 0; i < initial_rank; ++i) values[i] = Scalar(0);
            publish_(std::move(vectors), std::move(values));
            return;
        }

        Scalar trace = Scalar(0);
        for (int i = 0; i < rows; ++i) trace += fdapde::max(Scalar(0), source(i, i));
        const Scalar shift = trace * std::numeric_limits<Scalar>::epsilon();
        if (!(shift > Scalar(0)) || !std::isfinite(shift)) {
            throw std::domain_error("NysRBKI could not construct a finite stabilization shift");
        }

        std::mt19937 engine(seed_);
        std::normal_distribution<Scalar> normal(Scalar(0), Scalar(1));
        FactorType omega(rows, block_size);
        for (int row = 0; row < omega.rows(); ++row) {
            for (int col = 0; col < omega.cols(); ++col) {
                omega(row, col) = normal(engine);
                if (!std::isfinite(omega(row, col))) {
                    throw std::domain_error("NysRBKI Gaussian sampling produced a nonfinite coefficient");
                }
            }
        }

        FactorType basis = Ops::orthonormalize(omega);
        FactorType last_basis(basis);
        FactorType last_product = Ops::multiply(source, last_basis);
        FactorType product(last_product);
        State candidate = Ops::nystrom_state(basis, product, shift, initial_rank, true);
        Scalar residual = Ops::eigen_residual(source, candidate);
        const Scalar normalized_tolerance = tolerance_ / normalization;

        for (int expansion = 0; residual > normalized_tolerance && expansion < max_iterations_ && basis.cols() < rows;
             ++expansion) {
            const int added_columns = fdapde::min(block_size, rows - basis.cols());
            FactorType raw(rows, added_columns);
            for (int row = 0; row < raw.rows(); ++row) {
                for (int col = 0; col < raw.cols(); ++col) {
                    raw(row, col) = last_product(row, col) + shift * last_basis(row, col);
                    if (!std::isfinite(raw(row, col))) {
                        throw std::domain_error("NysRBKI shifted Krylov block contains a nonfinite coefficient");
                    }
                }
            }
            FactorType next = Ops::orthonormalize(raw, basis);
            basis = Ops::append_columns(basis, next);
            last_basis = std::move(next);
            last_product = Ops::multiply(source, last_basis);
            product = Ops::append_columns(product, last_product);
            candidate = Ops::nystrom_state(basis, product, shift, fdapde::min(rank, basis.cols()), true);
            residual = Ops::eigen_residual(source, candidate);
        }

        for (int i = 0; i < candidate.values.rows(); ++i) {
            candidate.values[i] *= normalization;
            if (!std::isfinite(candidate.values[i])) {
                throw std::domain_error("NysRBKI eigenvalues exceed the supported scalar range");
            }
        }
        publish_(std::move(candidate.vectors), std::move(candidate.values));
    }

    const FactorType& matrixU() const& { return state_->vectors; }
    void matrixU() const&& = delete;
    const EigenValuesType& eigenValues() const& { return state_->values; }
    void eigenValues() const&& = delete;
    int rank() const { return state_->values.rows(); }
   private:
    using Ops = internals::randomized_svd_ops<FactorType>;
    using State = typename Ops::NystromResult;

    void publish_(FactorType&& vectors, EigenValuesType&& values) {
        auto replacement = std::make_shared<const State>(std::move(vectors), std::move(values));
        state_.swap(replacement);
    }

    void validate_configuration_() const {
        if (!std::isfinite(tolerance_) || tolerance_ < Scalar(0)) {
            throw std::invalid_argument("NysRBKI tolerance must be finite and nonnegative");
        }
        if (max_iterations_ <= 0) { throw std::invalid_argument("NysRBKI maximum iterations must be positive"); }
    }

    static unsigned int resolve_seed_(int seed) {
        return seed == random_seed ? std::random_device {}() : static_cast<unsigned int>(seed);
    }

    std::shared_ptr<const State> state_ = std::make_shared<State>();
    Scalar tolerance_ = Scalar(1.0e-5);
    int max_iterations_ = 50;
    unsigned int seed_;
};

}   // namespace fdapde

#endif   // __FDAPDE_LINALG_RBKI_H__
