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

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "header_check.h"

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
    struct CompactSvd {
        FactorType left;
        FactorType right;
        SingularValuesType values;
    };

    struct State {
        FactorType left;
        FactorType right;
        SingularValuesType values;

        State() = default;
        State(FactorType&& left_, FactorType&& right_, SingularValuesType&& values_) :
            left(std::move(left_)), right(std::move(right_)), values(std::move(values_)) { }
    };

    static FactorType multiply_(const FactorType& lhs, const FactorType& rhs) {
        if (lhs.cols() != rhs.rows()) { throw std::logic_error("RSI internal matrix product has incompatible shapes"); }
        FactorType result(lhs.rows(), rhs.cols());
        for (int row = 0; row < result.rows(); ++row) {
            for (int col = 0; col < result.cols(); ++col) {
                Scalar value = Scalar(0);
                for (int k = 0; k < lhs.cols(); ++k) value += lhs(row, k) * rhs(k, col);
                if (!std::isfinite(value)) { throw std::domain_error("RSI matrix product produced a nonfinite value"); }
                result(row, col) = value;
            }
        }
        return result;
    }

    static FactorType transpose_multiply_(const FactorType& lhs, const FactorType& rhs) {
        if (lhs.rows() != rhs.rows()) {
            throw std::logic_error("RSI internal transpose product has incompatible shapes");
        }
        FactorType result(lhs.cols(), rhs.cols());
        for (int row = 0; row < result.rows(); ++row) {
            for (int col = 0; col < result.cols(); ++col) {
                Scalar value = Scalar(0);
                for (int k = 0; k < lhs.rows(); ++k) value += lhs(k, row) * rhs(k, col);
                if (!std::isfinite(value)) {
                    throw std::domain_error("RSI transpose product produced a nonfinite value");
                }
                result(row, col) = value;
            }
        }
        return result;
    }

    // ponytail: keep this thin two-pass orthonormalization private until a
    // second randomized algorithm needs a shared compact-basis primitive.
    static FactorType orthonormalize_(const FactorType& input) {
        FactorType basis(input.rows(), input.cols());
        basis.set_zero();
        int accepted = 0;
        const Scalar tolerance = Scalar(64) * std::numeric_limits<Scalar>::epsilon() *
                                 std::sqrt(Scalar(fdapde::max(input.rows(), input.cols())));
        std::vector<Scalar> column(static_cast<std::size_t>(input.rows()));

        auto append_column = [&]() {
            for (int pass = 0; pass < 2; ++pass) {
                for (int previous = 0; previous < accepted; ++previous) {
                    Scalar dot = Scalar(0);
                    for (int row = 0; row < input.rows(); ++row) {
                        dot += basis(row, previous) * column[static_cast<std::size_t>(row)];
                    }
                    for (int row = 0; row < input.rows(); ++row) {
                        column[static_cast<std::size_t>(row)] -= dot * basis(row, previous);
                    }
                }
            }

            Scalar norm = Scalar(0);
            for (const Scalar value : column) norm = internals::scale_safe_hypot(norm, value);
            if (!(norm > tolerance) || !std::isfinite(norm)) return false;
            for (int row = 0; row < input.rows(); ++row) {
                basis(row, accepted) = column[static_cast<std::size_t>(row)] / norm;
            }
            ++accepted;
            return true;
        };

        for (int source_col = 0; source_col < input.cols(); ++source_col) {
            Scalar scale = Scalar(0);
            for (int row = 0; row < input.rows(); ++row) {
                scale = fdapde::max(scale, fdapde::abs(input(row, source_col)));
            }
            if (scale == Scalar(0)) continue;
            for (int row = 0; row < input.rows(); ++row) {
                column[static_cast<std::size_t>(row)] = input(row, source_col) / scale;
            }

            (void)append_column();
        }

        // Householder QR exposes a full thin basis even when the sampled
        // range is deficient. Complete it deterministically to preserve the
        // direct RSI requested-rank contract, including exact zero modes.
        for (int coordinate = 0; accepted < input.cols() && coordinate < input.rows(); ++coordinate) {
            std::fill(column.begin(), column.end(), Scalar(0));
            column[static_cast<std::size_t>(coordinate)] = Scalar(1);
            (void)append_column();
        }
        if (accepted != input.cols()) {
            throw std::domain_error("RSI could not complete its sampled orthonormal basis");
        }

        FactorType compact(input.rows(), accepted);
        for (int row = 0; row < compact.rows(); ++row) {
            for (int col = 0; col < compact.cols(); ++col) compact(row, col) = basis(row, col);
        }
        return compact;
    }

    // ponytail: a private Gram/EVD compact SVD is enough for RSI. Promote it
    // only when RBKI provides the second native consumer.
    static CompactSvd compact_svd_(const FactorType& core, int requested_rank) {
        FactorType gram(core.rows(), core.rows());
        gram.set_zero();
        for (int row = 0; row < gram.rows(); ++row) {
            for (int col = 0; col <= row; ++col) {
                Scalar value = Scalar(0);
                for (int k = 0; k < core.cols(); ++k) value += core(row, k) * core(col, k);
                if (!std::isfinite(value)) { throw std::domain_error("RSI compact Gram matrix is not finite"); }
                gram(row, col) = gram(col, row) = value;
            }
        }

        const auto symmetric = gram.template as_symmetric<Lower>();
        const EVD decomposition(symmetric);
        const auto eigenvectors = decomposition.eigenvectors();
        const auto& eigenvalues = decomposition.eigenvalues();
        Scalar spectral_scale = Scalar(0);
        for (int i = 0; i < eigenvalues.rows(); ++i) {
            spectral_scale = fdapde::max(spectral_scale, fdapde::abs(eigenvalues[i]));
        }
        const Scalar negative_tolerance = Scalar(64) * std::numeric_limits<Scalar>::epsilon() *
                                          Scalar(fdapde::max(gram.rows(), core.cols())) *
                                          fdapde::max(Scalar(1), spectral_scale);
        for (int i = 0; i < eigenvalues.rows(); ++i) {
            if (eigenvalues[i] < -negative_tolerance) {
                throw std::domain_error("RSI compact Gram matrix is not positive semidefinite");
            }
        }

        std::vector<Scalar> recovered_values(static_cast<std::size_t>(gram.rows()));
        std::vector<int> order(static_cast<std::size_t>(gram.rows()));
        std::iota(order.begin(), order.end(), 0);
        for (int eigen_col = 0; eigen_col < gram.rows(); ++eigen_col) {
            Scalar norm = Scalar(0);
            for (int row = 0; row < core.cols(); ++row) {
                Scalar value = Scalar(0);
                for (int k = 0; k < core.rows(); ++k) value += core(k, row) * eigenvectors(k, eigen_col);
                norm = internals::scale_safe_hypot(norm, value);
            }
            recovered_values[static_cast<std::size_t>(eigen_col)] = norm;
        }
        std::sort(order.begin(), order.end(), [&](int lhs, int rhs) {
            return recovered_values[static_cast<std::size_t>(lhs)] > recovered_values[static_cast<std::size_t>(rhs)];
        });

        FactorType left(core.rows(), requested_rank);
        FactorType right(core.cols(), requested_rank);
        right.set_zero();
        SingularValuesType values(requested_rank);
        std::vector<Scalar> right_column(static_cast<std::size_t>(core.cols()));
        const Scalar completion_tolerance = Scalar(64) * std::numeric_limits<Scalar>::epsilon() *
                                            std::sqrt(Scalar(fdapde::max(core.rows(), core.cols())));
        for (int col = 0; col < requested_rank; ++col) {
            const int source_col = order[static_cast<std::size_t>(col)];
            const Scalar singular_value = recovered_values[static_cast<std::size_t>(source_col)];
            values[col] = singular_value;
            for (int row = 0; row < left.rows(); ++row) left(row, col) = eigenvectors(row, source_col);

            if (singular_value > Scalar(0)) {
                for (int row = 0; row < right.rows(); ++row) {
                    Scalar value = Scalar(0);
                    for (int k = 0; k < core.rows(); ++k) value += core(k, row) * left(k, col);
                    value /= singular_value;
                    if (!std::isfinite(value)) {
                        throw std::domain_error("RSI compact singular vectors contain a nonfinite coefficient");
                    }
                    right(row, col) = value;
                }
                continue;
            }

            bool completed = false;
            for (int coordinate = 0; !completed && coordinate < right.rows(); ++coordinate) {
                std::fill(right_column.begin(), right_column.end(), Scalar(0));
                right_column[static_cast<std::size_t>(coordinate)] = Scalar(1);
                for (int pass = 0; pass < 2; ++pass) {
                    for (int previous = 0; previous < col; ++previous) {
                        Scalar dot = Scalar(0);
                        for (int row = 0; row < right.rows(); ++row) {
                            dot += right(row, previous) * right_column[static_cast<std::size_t>(row)];
                        }
                        for (int row = 0; row < right.rows(); ++row) {
                            right_column[static_cast<std::size_t>(row)] -= dot * right(row, previous);
                        }
                    }
                }
                Scalar norm = Scalar(0);
                for (const Scalar value : right_column) norm = internals::scale_safe_hypot(norm, value);
                if (!(norm > completion_tolerance) || !std::isfinite(norm)) continue;
                for (int row = 0; row < right.rows(); ++row) {
                    right(row, col) = right_column[static_cast<std::size_t>(row)] / norm;
                }
                completed = true;
            }
            if (!completed) { throw std::domain_error("RSI could not complete its right singular-vector basis"); }
        }
        return {std::move(left), std::move(right), std::move(values)};
    }

    static State compact_state_(const FactorType& source, const FactorType& range, int requested_rank) {
        const FactorType core = transpose_multiply_(range, source);
        CompactSvd compact = compact_svd_(core, requested_rank);
        FactorType left = multiply_(range, compact.left);
        return {std::move(left), std::move(compact.right), std::move(compact.values)};
    }

    static Scalar residual_(const FactorType& source, const State& state) {
        Scalar maximum = Scalar(0);
        for (int col = 0; col < state.values.rows(); ++col) {
            Scalar norm = Scalar(0);
            for (int row = 0; row < source.rows(); ++row) {
                Scalar value = Scalar(0);
                for (int k = 0; k < source.cols(); ++k) value += source(row, k) * state.right(k, col);
                value -= state.left(row, col) * state.values[col];
                if (!std::isfinite(value)) { throw std::domain_error("RSI residual contains a nonfinite coefficient"); }
                norm = internals::scale_safe_hypot(norm, value);
            }
            maximum = fdapde::max(maximum, norm);
        }
        return maximum;
    }

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

// TODO(P4-M): RBKI, NysRSI, and NysRBKI remain preserved in the dormant Eigen archive until their own native
// compact-decomposition slices are dependency-closed.

}   // namespace fdapde

#endif   // __FDAPDE_LINALG_RSI_H__
