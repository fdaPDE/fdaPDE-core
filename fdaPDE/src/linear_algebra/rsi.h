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

#ifndef __FDAPDE_LINALG_RSI_H__
#define __FDAPDE_LINALG_RSI_H__

#include <cmath>
#include <cstdint>
#include <memory>
#include <random>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "header_check.h"
#include "randomized_svd.h"

namespace fdapde {

// Randomized subspace iteration for a compact rank-revealing SVD. The
// stopping tolerance is absolute, matching the direct RSI API introduced in
// b4b43f8 and officially exposed by 53b5e92.
template <typename MatrixType_>
    requires(internals::matrix_expression<MatrixType_>)
class RSI {
   public:
    using MatrixType = std::remove_cvref_t<MatrixType_>;
    using Scalar = std::remove_cv_t<typename MatrixType::Scalar>;
    using FactorType = Matrix<Scalar, Dynamic, Dynamic, MatrixType::StorageOrder>;
    using SingularValuesType = Vector<Scalar, Dynamic>;
    fdapde_static_assert(std::is_floating_point_v<Scalar>, RSI_REQUIRES_FLOATING_POINT_SCALARS);

    RSI() : seed_(resolve_seed_(random_seed)) { }
    RSI(const RSI&) = default;
    RSI& operator=(const RSI&) = default;
    RSI(const MatrixType& matrix, int rank) : RSI() { compute(matrix, rank); }
    RSI(const MatrixType& matrix, int rank, Scalar tolerance, int max_iterations, int seed = random_seed) :
        RSI(tolerance, max_iterations, seed) {
        compute(matrix, rank);
    }
    RSI(Scalar tolerance, int max_iterations, int seed = random_seed) :
        tolerance_(tolerance), max_iterations_(max_iterations), seed_(resolve_seed_(seed)) {
        validate_configuration_();
    }

    void compute(const MatrixType& matrix, int rank) {
        const int minimum_dimension = fdapde::min(matrix.rows(), matrix.cols());
        if (rank <= 0 || rank > minimum_dimension) {
            throw std::invalid_argument("RSI rank must be positive and no larger than either matrix dimension");
        }
        const int block_size = rank > minimum_dimension - rank ? minimum_dimension : 2 * rank;
        compute(matrix, rank, block_size);
    }

    void compute(const MatrixType& matrix, int rank, int block_size) {
        validate_configuration_();
        const int rows = matrix.rows();
        const int cols = matrix.cols();
        const int minimum_dimension = fdapde::min(rows, cols);
        if (rows <= 0 || cols <= 0) { throw std::invalid_argument("RSI requires a nonempty matrix"); }
        (void)internals::checked_matrix_size(rows, cols);
        if (rank <= 0 || rank > minimum_dimension) {
            throw std::invalid_argument("RSI rank must be positive and no larger than either matrix dimension");
        }
        if (block_size < rank || block_size > minimum_dimension) {
            throw std::invalid_argument(
              "RSI block size must contain the requested rank and fit both matrix dimensions");
        }

        Scalar scale = Scalar(0);
        for (int row = 0; row < rows; ++row) {
            for (int col = 0; col < cols; ++col) {
                const Scalar value = static_cast<Scalar>(matrix(row, col));
                if (!std::isfinite(value)) { throw std::invalid_argument("RSI requires finite matrix coefficients"); }
                scale = fdapde::max(scale, fdapde::abs(value));
            }
        }

        const Scalar normalization = scale == Scalar(0) ? Scalar(1) : scale;
        FactorType source(rows, cols);
        for (int row = 0; row < rows; ++row) {
            for (int col = 0; col < cols; ++col) {
                source(row, col) = static_cast<Scalar>(matrix(row, col)) / normalization;
            }
        }

        std::mt19937 engine(seed_);
        std::normal_distribution<Scalar> normal(Scalar(0), Scalar(1));
        FactorType omega(cols, block_size);
        for (int row = 0; row < cols; ++row) {
            for (int col = 0; col < block_size; ++col) {
                omega(row, col) = normal(engine);
                if (!std::isfinite(omega(row, col))) {
                    throw std::domain_error("RSI Gaussian sampling produced a nonfinite coefficient");
                }
            }
        }

        FactorType range = orthonormalize_(multiply_(source, omega));
        if (range.cols() == 0) { throw std::domain_error("RSI could not construct a nonzero sampled range"); }
        State candidate = compact_state_(source, range, rank);
        Scalar residual = residual_(source, candidate);
        const Scalar normalized_tolerance = tolerance_ / normalization;

        for (int iteration = 0; residual > normalized_tolerance && iteration < max_iterations_; ++iteration) {
            FactorType corange = orthonormalize_(transpose_multiply_(source, range));
            if (corange.cols() == 0) { throw std::domain_error("RSI subspace iteration lost its sampled range"); }
            range = orthonormalize_(multiply_(source, corange));
            if (range.cols() == 0) { throw std::domain_error("RSI subspace iteration lost its sampled range"); }
            candidate = compact_state_(source, range, rank);
            residual = residual_(source, candidate);
        }

        for (int i = 0; i < candidate.values.rows(); ++i) {
            candidate.values[i] *= normalization;
            if (!std::isfinite(candidate.values[i])) {
                throw std::domain_error("RSI singular values exceed the supported scalar range");
            }
        }
        publish_(std::move(candidate.left), std::move(candidate.right), std::move(candidate.values));
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
    using CompactSvd = typename Ops::Result;

    struct State {
        FactorType left;
        FactorType right;
        SingularValuesType values;

        State() = default;
        State(FactorType&& left_, FactorType&& right_, SingularValuesType&& values_) :
            left(std::move(left_)), right(std::move(right_)), values(std::move(values_)) { }
    };

    static FactorType multiply_(const FactorType& lhs, const FactorType& rhs) { return Ops::multiply(lhs, rhs); }

    static FactorType transpose_multiply_(const FactorType& lhs, const FactorType& rhs) {
        return Ops::transpose_multiply(lhs, rhs);
    }

    static FactorType orthonormalize_(const FactorType& input) { return Ops::orthonormalize(input); }

    static State compact_state_(const FactorType& source, const FactorType& range, int requested_rank) {
        CompactSvd compact = Ops::compact_state(source, range, requested_rank);
        return {std::move(compact.left), std::move(compact.right), std::move(compact.values)};
    }

    static Scalar residual_(const FactorType& source, const State& state) { return Ops::residual(source, state); }

    void publish_(FactorType&& left, FactorType&& right, SingularValuesType&& values) {
        auto replacement = std::make_shared<const State>(std::move(left), std::move(right), std::move(values));
        state_.swap(replacement);
    }

    void validate_configuration_() const {
        if (!std::isfinite(tolerance_) || tolerance_ < Scalar(0)) {
            throw std::invalid_argument("RSI tolerance must be finite and nonnegative");
        }
        if (max_iterations_ <= 0) { throw std::invalid_argument("RSI maximum iterations must be positive"); }
    }

    static unsigned int resolve_seed_(int seed) {
        return seed == random_seed ? std::random_device {}() : static_cast<unsigned int>(seed);
    }

    std::shared_ptr<const State> state_ = std::make_shared<State>();
    Scalar tolerance_ = Scalar(1.0e-5);
    int max_iterations_ = 50;
    unsigned int seed_;
};

// Randomized subspace iteration with a Nyström approximation for a symmetric
// positive-semidefinite matrix; Tropp and Webber (2023), Algorithm 5.7.
template <typename MatrixType_>
    requires(internals::matrix_expression<MatrixType_>)
class NysRSI {
   public:
    using MatrixType = std::remove_cvref_t<MatrixType_>;
    using Scalar = std::remove_cv_t<typename MatrixType::Scalar>;
    using FactorType = Matrix<Scalar, Dynamic, Dynamic, MatrixType::StorageOrder>;
    using EigenValuesType = Vector<Scalar, Dynamic>;
    fdapde_static_assert(std::is_floating_point_v<Scalar>, NYS_RSI_REQUIRES_FLOATING_POINT_SCALARS);
    fdapde_static_assert(
      MatrixType::Rows == Dynamic || MatrixType::Cols == Dynamic || MatrixType::Rows == MatrixType::Cols,
      NYS_RSI_REQUIRES_A_SQUARE_MATRIX);

    NysRSI() : seed_(resolve_seed_(random_seed)) { }
    NysRSI(const NysRSI&) = default;
    NysRSI& operator=(const NysRSI&) = default;
    NysRSI(const MatrixType& matrix, int rank) : NysRSI() { compute(matrix, rank); }
    NysRSI(const MatrixType& matrix, int rank, Scalar tolerance, int max_iterations, int seed = random_seed) :
        NysRSI(tolerance, max_iterations, seed) {
        compute(matrix, rank);
    }
    NysRSI(Scalar tolerance, int max_iterations, int seed = random_seed) :
        tolerance_(tolerance), max_iterations_(max_iterations), seed_(resolve_seed_(seed)) {
        validate_configuration_();
    }

    void compute(const MatrixType& matrix, int rank) {
        const int dimension = matrix.rows();
        const std::int64_t doubled_rank = std::int64_t(rank) * 2;
        const int block_size = static_cast<int>(fdapde::min<std::int64_t>(doubled_rank, dimension));
        compute(matrix, rank, block_size);
    }

    void compute(const MatrixType& matrix, int rank, int block_size) {
        validate_configuration_();
        const int rows = matrix.rows();
        const int cols = matrix.cols();
        if (rows <= 0 || rows != cols) { throw std::invalid_argument("NysRSI requires a nonempty square matrix"); }
        (void)internals::checked_matrix_size(rows, cols);
        if (rank <= 0 || rank > rows) {
            throw std::invalid_argument("NysRSI rank must be positive and no larger than the matrix dimension");
        }
        if (block_size < rank || block_size > rows) {
            throw std::invalid_argument(
              "NysRSI block size must contain the requested rank and fit the matrix dimension");
        }

        FactorType source(rows, cols);
        Scalar scale = Scalar(0);
        for (int row = 0; row < rows; ++row) {
            for (int col = 0; col < cols; ++col) {
                const Scalar value = static_cast<Scalar>(matrix(row, col));
                if (!std::isfinite(value)) {
                    throw std::invalid_argument("NysRSI requires finite matrix coefficients");
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
                throw std::domain_error("NysRSI requires a positive-semidefinite matrix");
            }
            for (int col = row + 1; col < cols; ++col) {
                if (fdapde::abs(source(row, col) - source(col, row)) > roundoff) {
                    throw std::invalid_argument("NysRSI requires a symmetric matrix");
                }
                const Scalar diagonal_product = source(row, row) * source(col, col);
                const Scalar coefficient_square = source(row, col) * source(row, col);
                if (coefficient_square > diagonal_product + roundoff) {
                    throw std::domain_error("NysRSI requires a positive-semidefinite matrix");
                }
            }
        }

        if (scale == Scalar(0)) {
            FactorType vectors(rows, rank);
            vectors.set_zero();
            for (int i = 0; i < rank; ++i) vectors(i, i) = Scalar(1);
            EigenValuesType values(rank);
            for (int i = 0; i < rank; ++i) values[i] = Scalar(0);
            publish_(std::move(vectors), std::move(values));
            return;
        }

        Scalar trace = Scalar(0);
        for (int i = 0; i < rows; ++i) trace += fdapde::max(Scalar(0), source(i, i));
        const Scalar shift = trace * std::numeric_limits<Scalar>::epsilon();
        if (!(shift > Scalar(0)) || !std::isfinite(shift)) {
            throw std::domain_error("NysRSI could not construct a finite stabilization shift");
        }

        std::mt19937 engine(seed_);
        std::normal_distribution<Scalar> normal(Scalar(0), Scalar(1));
        FactorType range_input(rows, block_size);
        for (int row = 0; row < rows; ++row) {
            for (int col = 0; col < block_size; ++col) {
                range_input(row, col) = normal(engine);
                if (!std::isfinite(range_input(row, col))) {
                    throw std::domain_error("NysRSI Gaussian sampling produced a nonfinite coefficient");
                }
            }
        }

        State candidate;
        const Scalar normalized_tolerance = tolerance_ / normalization;
        for (int iteration = 0; iteration < max_iterations_; ++iteration) {
            const FactorType basis = Ops::orthonormalize(range_input);
            FactorType product = Ops::multiply(source, basis);
            candidate = nystrom_state_(basis, product, shift, rank);
            if (residual_(source, candidate) <= normalized_tolerance) break;
            range_input = std::move(product);
        }

        for (int i = 0; i < candidate.values.rows(); ++i) {
            candidate.values[i] *= normalization;
            if (!std::isfinite(candidate.values[i])) {
                throw std::domain_error("NysRSI eigenvalues exceed the supported scalar range");
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

    struct State {
        FactorType vectors;
        EigenValuesType values;

        State() = default;
        State(FactorType&& vectors_, EigenValuesType&& values_) :
            vectors(std::move(vectors_)), values(std::move(values_)) { }
    };

    static State nystrom_state_(const FactorType& basis, const FactorType& product, Scalar shift, int rank) {
        FactorType shifted(product);
        for (int row = 0; row < shifted.rows(); ++row) {
            for (int col = 0; col < shifted.cols(); ++col) {
                shifted(row, col) += shift * basis(row, col);
                if (!std::isfinite(shifted(row, col))) {
                    throw std::domain_error("NysRSI shifted range contains a nonfinite coefficient");
                }
            }
        }

        FactorType gram(basis.cols(), basis.cols());
        gram.set_zero();
        Scalar gram_scale = Scalar(0);
        for (int row = 0; row < gram.rows(); ++row) {
            for (int col = 0; col <= row; ++col) {
                Scalar lower = Scalar(0);
                Scalar upper = Scalar(0);
                for (int k = 0; k < basis.rows(); ++k) {
                    lower += basis(k, row) * shifted(k, col);
                    upper += basis(k, col) * shifted(k, row);
                }
                const Scalar value = Scalar(0.5) * (lower + upper);
                if (!std::isfinite(value)) {
                    throw std::domain_error("NysRSI stabilized Gram matrix contains a nonfinite coefficient");
                }
                gram(row, col) = gram(col, row) = value;
                gram_scale = fdapde::max(gram_scale, fdapde::abs(value));
            }
        }

        const auto symmetric = gram.template as_symmetric<Lower>();
        const EVD decomposition(symmetric);
        const auto eigenvectors = decomposition.eigenvectors();
        const auto& eigenvalues = decomposition.eigenvalues();
        const Scalar spectral_roundoff = Scalar(64) * std::numeric_limits<Scalar>::epsilon() *
                                         static_cast<Scalar>(gram.rows()) * fdapde::max(Scalar(1), gram_scale);
        FactorType inverse_sqrt(gram.rows(), gram.cols());
        inverse_sqrt.set_zero();
        for (int component = 0; component < gram.rows(); ++component) {
            const Scalar eigenvalue = eigenvalues[component];
            if (!std::isfinite(eigenvalue) || eigenvalue < -spectral_roundoff) {
                throw std::domain_error("NysRSI requires a positive-semidefinite matrix");
            }
            const Scalar stabilized = fdapde::max(shift, eigenvalue);
            const Scalar inverse_root = Scalar(1) / std::sqrt(stabilized);
            if (!std::isfinite(inverse_root)) {
                throw std::domain_error("NysRSI stabilized Gram inverse is not representable");
            }
            for (int row = 0; row < inverse_sqrt.rows(); ++row) {
                for (int col = 0; col < inverse_sqrt.cols(); ++col) {
                    inverse_sqrt(row, col) +=
                      eigenvectors(row, component) * inverse_root * eigenvectors(col, component);
                }
            }
        }

        const FactorType factor = Ops::multiply(shifted, inverse_sqrt);
        FactorType factor_transpose(factor.cols(), factor.rows());
        for (int row = 0; row < factor.rows(); ++row) {
            for (int col = 0; col < factor.cols(); ++col) factor_transpose(col, row) = factor(row, col);
        }
        typename Ops::Result compact = Ops::compact_svd(factor_transpose, rank);
        EigenValuesType values(rank);
        const Scalar value_roundoff =
          Scalar(128) * std::numeric_limits<Scalar>::epsilon() * static_cast<Scalar>(factor.rows());
        for (int i = 0; i < rank; ++i) {
            const Scalar square = compact.values[i] * compact.values[i];
            const Scalar value = square - shift;
            const Scalar tolerance = value_roundoff * fdapde::max(Scalar(1), fdapde::max(square, shift));
            if (!std::isfinite(value) || value < -tolerance) {
                throw std::domain_error("NysRSI produced a non-positive-semidefinite approximation");
            }
            values[i] = value > tolerance ? value : Scalar(0);
        }
        return {std::move(compact.right), std::move(values)};
    }

    static Scalar residual_(const FactorType& source, const State& state) {
        Scalar maximum = Scalar(0);
        for (int col = 0; col < state.values.rows(); ++col) {
            Scalar norm = Scalar(0);
            for (int row = 0; row < source.rows(); ++row) {
                Scalar value = Scalar(0);
                for (int inner = 0; inner < source.cols(); ++inner) {
                    value += source(row, inner) * state.vectors(inner, col);
                }
                value -= state.vectors(row, col) * state.values[col];
                if (!std::isfinite(value)) {
                    throw std::domain_error("NysRSI eigen-residual contains a nonfinite coefficient");
                }
                norm = internals::scale_safe_hypot(norm, value);
            }
            maximum = fdapde::max(maximum, norm);
        }
        return std::sqrt(Scalar(2)) * maximum;
    }

    void publish_(FactorType&& vectors, EigenValuesType&& values) {
        auto replacement = std::make_shared<const State>(std::move(vectors), std::move(values));
        state_.swap(replacement);
    }

    void validate_configuration_() const {
        if (!std::isfinite(tolerance_) || tolerance_ < Scalar(0)) {
            throw std::invalid_argument("NysRSI tolerance must be finite and nonnegative");
        }
        if (max_iterations_ <= 0) { throw std::invalid_argument("NysRSI maximum iterations must be positive"); }
    }

    static unsigned int resolve_seed_(int seed) {
        return seed == random_seed ? std::random_device {}() : static_cast<unsigned int>(seed);
    }

    std::shared_ptr<const State> state_ = std::make_shared<State>();
    Scalar tolerance_ = Scalar(1.0e-5);
    int max_iterations_ = 50;
    unsigned int seed_;
};

// TODO(P4-M): NysRBKI remains preserved in the dormant Eigen archive until its native
// positive-semidefinite compact-decomposition slice is dependency-closed.

}   // namespace fdapde

#endif   // __FDAPDE_LINALG_RSI_H__
