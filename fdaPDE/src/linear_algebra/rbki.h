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

/// @brief approximates leading singular triplets with randomized block Krylov iteration
template <typename MatrixType_>
    requires(internals::matrix_expression<MatrixType_>)
class RBKI {
   public:
    using MatrixType = std::remove_cvref_t<MatrixType_>;
    using Scalar = std::remove_cv_t<typename MatrixType::Scalar>;
    using FactorType = Matrix<Scalar, Dynamic, Dynamic, MatrixType::StorageOrder>;
    using SingularValuesType = Vector<Scalar, Dynamic>;
    fdapde_static_assert(std::is_floating_point_v<Scalar>, RBKI_REQUIRES_FLOATING_POINT_SCALARS);

    /// @brief constructs an empty approximation with default parameters and a resolved seed
    RBKI() : seed_(resolve_seed_(random_seed)) { }
    /// @brief copies the immutable published result and sampling configuration
    RBKI(const RBKI&) = default;
    /// @brief copies the immutable published result and sampling configuration
    RBKI& operator=(const RBKI&) = default;
    /// @brief constructs and computes the requested approximation
    RBKI(const MatrixType& matrix, int rank) : RBKI() { compute(matrix, rank); }
    /// @brief constructs and computes the requested approximation
    RBKI(const MatrixType& matrix, int rank, Scalar tolerance, int max_iterations, int seed = random_seed) :
        RBKI(tolerance, max_iterations, seed) {
        compute(matrix, rank);
    }
    /// @brief validates the stopping parameters and resolves the sampling seed
    RBKI(Scalar tolerance, int max_iterations, int seed = random_seed) :
        tolerance_(tolerance), max_iterations_(max_iterations), seed_(resolve_seed_(seed)) {
        validate_configuration_();
    }

    /// @brief computes a replacement approximation without changing published results on failure
    void compute(const MatrixType& matrix, int rank) {
        const int minimum_dimension = fdapde::min(matrix.rows(), matrix.cols());
        compute(matrix, rank, minimum_dimension <= 100 ? 1 : 10);
    }

    /// @brief computes a replacement approximation without changing published results on failure
    void compute(const MatrixType& matrix, int rank, int block_size) {
        validate_configuration_();
        const int rows = matrix.rows();
        const int cols = matrix.cols();
        const int minimum_dimension = fdapde::min(rows, cols);
        fdapde_strong_assert(!(rows <= 0 || cols <= 0), std::invalid_argument, "RBKI requires a nonempty matrix");
        (void)internals::checked_matrix_size(rows, cols);
        fdapde_strong_assert(
          !(rank <= 0 || rank > minimum_dimension), std::invalid_argument,
          "RBKI rank must be positive and no larger than either matrix dimension");
        fdapde_strong_assert(
          !(block_size <= 0 || block_size > minimum_dimension), std::invalid_argument,
          "RBKI block size must be positive and fit both matrix dimensions");

        Scalar scale = Scalar(0);
        for (int row = 0; row < rows; ++row) {
            for (int col = 0; col < cols; ++col) {
                const Scalar value = static_cast<Scalar>(matrix(row, col));
                fdapde_strong_assert(
                  std::isfinite(value), std::invalid_argument, "RBKI requires finite matrix coefficients");
                scale = fdapde::max(scale, fdapde::abs(value));
            }
        }

        // normalize before projection so products remain in the input scalar range
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

        // restart the resolved seed so repeated computations use the same sampled range
        std::mt19937 engine(seed_);
        std::normal_distribution<Scalar> normal(Scalar(0), Scalar(1));
        FactorType omega(source.cols(), block_size);
        for (int row = 0; row < omega.rows(); ++row) {
            for (int col = 0; col < omega.cols(); ++col) {
                omega(row, col) = normal(engine);
                fdapde_strong_assert(
                  std::isfinite(omega(row, col)), std::domain_error,
                  "RBKI Gaussian sampling produced a nonfinite coefficient");
            }
        }

        FactorType range = Ops::orthonormalize(Ops::multiply(source, omega));
        FactorType last_corange = Ops::transpose_multiply(source, range);
        typename Ops::Result candidate = Ops::compact_state(source, range, fdapde::min(rank, range.cols()));
        Scalar residual = Ops::residual(source, candidate);
        const Scalar normalized_tolerance = tolerance_ / normalization;

        // enlarge the orthogonal basis until the absolute residual or the iteration cap stops expansion
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

        // restore spectral values to the original scale before publishing all factors atomically
        for (int i = 0; i < candidate.values.rows(); ++i) {
            candidate.values[i] *= normalization;
            fdapde_strong_assert(
              std::isfinite(candidate.values[i]), std::domain_error,
              "RBKI singular values exceed the supported scalar range");
        }
        if (transposed) {
            publish_(std::move(candidate.right), std::move(candidate.left), std::move(candidate.values));
        } else {
            publish_(std::move(candidate.left), std::move(candidate.right), std::move(candidate.values));
        }
    }

    /// @brief borrows the left singular vectors from a live approximation
    const FactorType& left_vectors() const& { return state_->left; }
    /// @brief rejects borrowing result storage from a temporary approximation
    void left_vectors() const&& = delete;
    /// @brief borrows the right singular vectors from a live approximation
    const FactorType& right_vectors() const& { return state_->right; }
    /// @brief rejects borrowing result storage from a temporary approximation
    void right_vectors() const&& = delete;
    /// @brief borrows descending singular values from a live approximation
    const SingularValuesType& singular_values() const& { return state_->values; }
    /// @brief rejects borrowing result storage from a temporary approximation
    void singular_values() const&& = delete;
    /// @brief returns the number of published spectral components
    int rank() const { return state_->values.rows(); }
   private:
    using Ops = internals::randomized_svd_ops<FactorType>;

    /// @brief owns a complete set of factors and associated spectral values
    struct State {
        FactorType left;
        FactorType right;
        SingularValuesType values;

        /// @brief constructs empty factor storage
        State() = default;
        /// @brief takes ownership of completed factors and spectral values
        State(FactorType&& left_, FactorType&& right_, SingularValuesType&& values_) :
            left(std::move(left_)), right(std::move(right_)), values(std::move(values_)) { }
    };

    /// @brief publishes completed factors together after all numerical checks succeed
    void publish_(FactorType&& left, FactorType&& right, SingularValuesType&& values) {
        auto replacement = std::make_shared<const State>(std::move(left), std::move(right), std::move(values));
        state_.swap(replacement);
    }

    /// @brief checks finite nonnegative tolerance and a positive iteration cap
    void validate_configuration_() const {
        fdapde_strong_assert(
          !(!std::isfinite(tolerance_) || tolerance_ < Scalar(0)), std::invalid_argument,
          "RBKI tolerance must be finite and nonnegative");
        fdapde_strong_assert(
          !(max_iterations_ <= 0), std::invalid_argument, "RBKI maximum iterations must be positive");
    }

    /// @brief resolves one nondeterministic seed or preserves the supplied fixed seed
    static unsigned int resolve_seed_(int seed) {
        return seed == random_seed ? std::random_device {}() : static_cast<unsigned int>(seed);
    }

    std::shared_ptr<const State> state_ = std::make_shared<State>();
    Scalar tolerance_ = Scalar(1.0e-5);
    int max_iterations_ = 50;
    unsigned int seed_;
};

/// @brief approximates leading PSD eigenpairs with a stabilized randomized Nyström range
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

    /// @brief constructs an empty approximation with default parameters and a resolved seed
    NysRBKI() : seed_(resolve_seed_(random_seed)) { }
    /// @brief copies the immutable published result and sampling configuration
    NysRBKI(const NysRBKI&) = default;
    /// @brief copies the immutable published result and sampling configuration
    NysRBKI& operator=(const NysRBKI&) = default;
    /// @brief constructs and computes the requested approximation
    NysRBKI(const MatrixType& matrix, int rank) : NysRBKI() { compute(matrix, rank); }
    /// @brief constructs and computes the requested approximation
    NysRBKI(const MatrixType& matrix, int rank, Scalar tolerance, int max_iterations, int seed = random_seed) :
        NysRBKI(tolerance, max_iterations, seed) {
        compute(matrix, rank);
    }
    /// @brief validates the stopping parameters and resolves the sampling seed
    NysRBKI(Scalar tolerance, int max_iterations, int seed = random_seed) :
        tolerance_(tolerance), max_iterations_(max_iterations), seed_(resolve_seed_(seed)) {
        validate_configuration_();
    }

    /// @brief computes a replacement approximation without changing published results on failure
    void compute(const MatrixType& matrix, int rank) { compute(matrix, rank, matrix.rows() <= 100 ? 1 : 10); }

    /// @brief computes a replacement approximation without changing published results on failure
    void compute(const MatrixType& matrix, int rank, int block_size) {
        validate_configuration_();
        const int rows = matrix.rows();
        const int cols = matrix.cols();
        fdapde_strong_assert(
          !(rows <= 0 || rows != cols), std::invalid_argument, "NysRBKI requires a nonempty square matrix");
        (void)internals::checked_matrix_size(rows, cols);
        fdapde_strong_assert(
          !(rank <= 0 || rank > rows), std::invalid_argument,
          "NysRBKI rank must be positive and no larger than the matrix dimension");
        fdapde_strong_assert(
          !(block_size <= 0 || block_size > rows), std::invalid_argument,
          "NysRBKI block size must be positive and fit the matrix dimension");

        FactorType source(rows, cols);
        Scalar scale = Scalar(0);
        for (int row = 0; row < rows; ++row) {
            for (int col = 0; col < cols; ++col) {
                const Scalar value = static_cast<Scalar>(matrix(row, col));
                fdapde_strong_assert(
                  std::isfinite(value), std::invalid_argument, "NysRBKI requires finite matrix coefficients");
                source(row, col) = value;
                scale = fdapde::max(scale, fdapde::abs(value));
            }
        }
        // normalize before projection so products remain in the input scalar range
        const Scalar normalization = scale == Scalar(0) ? Scalar(1) : scale;
        if (scale != Scalar(0)) {
            for (int row = 0; row < rows; ++row) {
                for (int col = 0; col < cols; ++col) source(row, col) /= normalization;
            }
        }

        const Scalar roundoff = Scalar(64) * std::numeric_limits<Scalar>::epsilon() * static_cast<Scalar>(rows);
        // ponytail: necessary PSD checks are not a certificate; use full factorization when certification is needed
        for (int row = 0; row < rows; ++row) {
            fdapde_strong_assert(
              !(source(row, row) < -roundoff), std::domain_error, "NysRBKI requires a positive-semidefinite matrix");
            for (int col = row + 1; col < cols; ++col) {
                fdapde_strong_assert(
                  !(fdapde::abs(source(row, col) - source(col, row)) > roundoff), std::invalid_argument,
                  "NysRBKI requires a symmetric matrix");
                const Scalar diagonal_product = source(row, row) * source(col, col);
                const Scalar coefficient_square = source(row, col) * source(row, col);
                fdapde_strong_assert(
                  !(coefficient_square > diagonal_product + roundoff), std::domain_error,
                  "NysRBKI requires a positive-semidefinite matrix");
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
        fdapde_strong_assert(
          shift > Scalar(0) && std::isfinite(shift), std::domain_error,
          "NysRBKI could not construct a finite stabilization shift");

        // restart the resolved seed so repeated computations use the same sampled range
        std::mt19937 engine(seed_);
        std::normal_distribution<Scalar> normal(Scalar(0), Scalar(1));
        FactorType omega(rows, block_size);
        for (int row = 0; row < omega.rows(); ++row) {
            for (int col = 0; col < omega.cols(); ++col) {
                omega(row, col) = normal(engine);
                fdapde_strong_assert(
                  std::isfinite(omega(row, col)), std::domain_error,
                  "NysRBKI Gaussian sampling produced a nonfinite coefficient");
            }
        }

        FactorType basis = Ops::orthonormalize(omega);
        FactorType last_basis(basis);
        FactorType last_product = Ops::multiply(source, last_basis);
        FactorType product(last_product);
        State candidate = Ops::nystrom_state(basis, product, shift, initial_rank, true);
        Scalar residual = Ops::eigen_residual(source, candidate);
        const Scalar normalized_tolerance = tolerance_ / normalization;

        // enlarge the orthogonal basis until the absolute residual or the iteration cap stops expansion
        for (int expansion = 0; residual > normalized_tolerance && expansion < max_iterations_ && basis.cols() < rows;
             ++expansion) {
            const int added_columns = fdapde::min(block_size, rows - basis.cols());
            FactorType raw(rows, added_columns);
            for (int row = 0; row < raw.rows(); ++row) {
                for (int col = 0; col < raw.cols(); ++col) {
                    raw(row, col) = last_product(row, col) + shift * last_basis(row, col);
                    fdapde_strong_assert(
                      std::isfinite(raw(row, col)), std::domain_error,
                      "NysRBKI shifted Krylov block contains a nonfinite coefficient");
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

        // restore spectral values to the original scale before publishing all factors atomically
        for (int i = 0; i < candidate.values.rows(); ++i) {
            candidate.values[i] *= normalization;
            fdapde_strong_assert(
              std::isfinite(candidate.values[i]), std::domain_error,
              "NysRBKI eigenvalues exceed the supported scalar range");
        }
        publish_(std::move(candidate.vectors), std::move(candidate.values));
    }

    /// @brief borrows the leading Nyström eigenvectors from a live approximation
    const FactorType& eigenvectors() const& { return state_->vectors; }
    /// @brief rejects borrowing result storage from a temporary approximation
    void eigenvectors() const&& = delete;
    /// @brief borrows descending nonnegative Nyström eigenvalues from a live approximation
    const EigenValuesType& eigenvalues() const& { return state_->values; }
    /// @brief rejects borrowing result storage from a temporary approximation
    void eigenvalues() const&& = delete;
    /// @brief returns the number of published spectral components
    int rank() const { return state_->values.rows(); }
   private:
    using Ops = internals::randomized_svd_ops<FactorType>;
    using State = typename Ops::NystromResult;

    /// @brief publishes completed factors together after all numerical checks succeed
    void publish_(FactorType&& vectors, EigenValuesType&& values) {
        auto replacement = std::make_shared<const State>(std::move(vectors), std::move(values));
        state_.swap(replacement);
    }

    /// @brief checks finite nonnegative tolerance and a positive iteration cap
    void validate_configuration_() const {
        fdapde_strong_assert(
          !(!std::isfinite(tolerance_) || tolerance_ < Scalar(0)), std::invalid_argument,
          "NysRBKI tolerance must be finite and nonnegative");
        fdapde_strong_assert(
          !(max_iterations_ <= 0), std::invalid_argument, "NysRBKI maximum iterations must be positive");
    }

    /// @brief resolves one nondeterministic seed or preserves the supplied fixed seed
    static unsigned int resolve_seed_(int seed) {
        return seed == random_seed ? std::random_device {}() : static_cast<unsigned int>(seed);
    }

    std::shared_ptr<const State> state_ = std::make_shared<State>();
    Scalar tolerance_ = Scalar(1.0e-5);
    int max_iterations_ = 50;
    unsigned int seed_;
};

template <internals::matrix_expression MatrixType>
/// @brief constructs and computes the requested approximation
RBKI(const MatrixType&, int) -> RBKI<std::remove_cvref_t<MatrixType>>;

template <internals::matrix_expression MatrixType>
/// @brief constructs and computes the requested approximation
RBKI(const MatrixType&, int, typename std::remove_cvref_t<MatrixType>::Scalar, int)
  -> RBKI<std::remove_cvref_t<MatrixType>>;

template <internals::matrix_expression MatrixType>
/// @brief constructs and computes the requested approximation
RBKI(const MatrixType&, int, typename std::remove_cvref_t<MatrixType>::Scalar, int, int)
  -> RBKI<std::remove_cvref_t<MatrixType>>;

template <internals::matrix_expression MatrixType>
/// @brief constructs and computes the requested approximation
NysRBKI(const MatrixType&, int) -> NysRBKI<std::remove_cvref_t<MatrixType>>;

template <internals::matrix_expression MatrixType>
/// @brief constructs and computes the requested approximation
NysRBKI(const MatrixType&, int, typename std::remove_cvref_t<MatrixType>::Scalar, int)
  -> NysRBKI<std::remove_cvref_t<MatrixType>>;

template <internals::matrix_expression MatrixType>
/// @brief constructs and computes the requested approximation
NysRBKI(const MatrixType&, int, typename std::remove_cvref_t<MatrixType>::Scalar, int, int)
  -> NysRBKI<std::remove_cvref_t<MatrixType>>;

}   // namespace fdapde

#endif   // __FDAPDE_LINALG_RBKI_H__
