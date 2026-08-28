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

// TODO(P4-M): NysRSI and NysRBKI remain preserved in the dormant Eigen archive until their own native
// compact-decomposition slices are dependency-closed.

}   // namespace fdapde

#endif   // __FDAPDE_LINALG_RSI_H__
