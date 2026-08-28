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

#ifndef __FDAPDE_LINALG_RANDOMIZED_SVD_H__
#define __FDAPDE_LINALG_RANDOMIZED_SVD_H__

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "header_check.h"

namespace fdapde::internals {

template <typename FactorType_> struct randomized_svd_ops {
    using FactorType = std::remove_cvref_t<FactorType_>;
    using Scalar = std::remove_cv_t<typename FactorType::Scalar>;
    using SingularValuesType = Vector<Scalar, Dynamic>;

    struct Result {
        FactorType left;
        FactorType right;
        SingularValuesType values;
    };

    struct NystromResult {
        FactorType vectors;
        SingularValuesType values;

        NystromResult() = default;
        NystromResult(FactorType&& vectors_, SingularValuesType&& values_) :
            vectors(std::move(vectors_)), values(std::move(values_)) { }
    };

    static FactorType multiply(const FactorType& lhs, const FactorType& rhs) {
        if (lhs.cols() != rhs.rows()) {
            throw std::logic_error("randomized SVD internal matrix product has incompatible shapes");
        }
        FactorType result(lhs.rows(), rhs.cols());
        for (int row = 0; row < result.rows(); ++row) {
            for (int col = 0; col < result.cols(); ++col) {
                Scalar value = Scalar(0);
                for (int k = 0; k < lhs.cols(); ++k) value += lhs(row, k) * rhs(k, col);
                if (!std::isfinite(value)) {
                    throw std::domain_error("randomized SVD matrix product produced a nonfinite value");
                }
                result(row, col) = value;
            }
        }
        return result;
    }

    static FactorType transpose_multiply(const FactorType& lhs, const FactorType& rhs) {
        if (lhs.rows() != rhs.rows()) {
            throw std::logic_error("randomized SVD internal transpose product has incompatible shapes");
        }
        FactorType result(lhs.cols(), rhs.cols());
        for (int row = 0; row < result.rows(); ++row) {
            for (int col = 0; col < result.cols(); ++col) {
                Scalar value = Scalar(0);
                for (int k = 0; k < lhs.rows(); ++k) value += lhs(k, row) * rhs(k, col);
                if (!std::isfinite(value)) {
                    throw std::domain_error("randomized SVD transpose product produced a nonfinite value");
                }
                result(row, col) = value;
            }
        }
        return result;
    }

    static FactorType append_columns(const FactorType& lhs, const FactorType& rhs) {
        if (lhs.rows() != rhs.rows()) {
            throw std::logic_error("randomized SVD basis blocks have incompatible row counts");
        }
        if (rhs.cols() > std::numeric_limits<int>::max() - lhs.cols()) {
            throw std::length_error("randomized SVD basis width exceeds the supported range");
        }
        const int columns = lhs.cols() + rhs.cols();
        (void)checked_matrix_size(lhs.rows(), columns);
        FactorType result(lhs.rows(), columns);
        for (int row = 0; row < result.rows(); ++row) {
            for (int col = 0; col < lhs.cols(); ++col) result(row, col) = lhs(row, col);
            for (int col = 0; col < rhs.cols(); ++col) result(row, lhs.cols() + col) = rhs(row, col);
        }
        return result;
    }

    static FactorType orthonormalize(const FactorType& input) {
        return orthonormalize(input, FactorType(input.rows(), 0));
    }

    static FactorType orthonormalize(const FactorType& input, const FactorType& existing) {
        if (input.rows() != existing.rows()) {
            throw std::logic_error("randomized SVD basis extension has incompatible row counts");
        }
        if (input.cols() > input.rows() - existing.cols()) {
            throw std::logic_error("randomized SVD basis extension exceeds the ambient dimension");
        }

        FactorType basis(input.rows(), input.cols());
        basis.set_zero();
        int accepted = 0;
        const Scalar tolerance = Scalar(64) * std::numeric_limits<Scalar>::epsilon() *
                                 std::sqrt(Scalar(fdapde::max(input.rows(), input.cols() + existing.cols())));
        std::vector<Scalar> column(static_cast<std::size_t>(input.rows()));

        auto append_column = [&]() {
            for (int pass = 0; pass < 2; ++pass) {
                for (int previous = 0; previous < existing.cols(); ++previous) {
                    Scalar dot = Scalar(0);
                    for (int row = 0; row < input.rows(); ++row) {
                        dot += existing(row, previous) * column[static_cast<std::size_t>(row)];
                    }
                    for (int row = 0; row < input.rows(); ++row) {
                        column[static_cast<std::size_t>(row)] -= dot * existing(row, previous);
                    }
                }
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
            for (const Scalar value : column) norm = scale_safe_hypot(norm, value);
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

        for (int coordinate = 0; accepted < input.cols() && coordinate < input.rows(); ++coordinate) {
            std::fill(column.begin(), column.end(), Scalar(0));
            column[static_cast<std::size_t>(coordinate)] = Scalar(1);
            (void)append_column();
        }
        if (accepted != input.cols()) {
            throw std::domain_error("randomized SVD could not complete its sampled orthonormal basis");
        }
        return basis;
    }

    static Result compact_svd(const FactorType& core, int requested_rank) {
        if (requested_rank <= 0 || requested_rank > core.rows()) {
            throw std::logic_error("randomized SVD compact rank is incompatible with its basis");
        }

        FactorType gram(core.rows(), core.rows());
        gram.set_zero();
        for (int row = 0; row < gram.rows(); ++row) {
            for (int col = 0; col <= row; ++col) {
                Scalar value = Scalar(0);
                for (int k = 0; k < core.cols(); ++k) value += core(row, k) * core(col, k);
                if (!std::isfinite(value)) {
                    throw std::domain_error("randomized SVD compact Gram matrix is not finite");
                }
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
                throw std::domain_error("randomized SVD compact Gram matrix is not positive semidefinite");
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
                norm = scale_safe_hypot(norm, value);
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
                        throw std::domain_error(
                          "randomized SVD compact singular vectors contain a nonfinite coefficient");
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
                for (const Scalar value : right_column) norm = scale_safe_hypot(norm, value);
                if (!(norm > completion_tolerance) || !std::isfinite(norm)) continue;
                for (int row = 0; row < right.rows(); ++row) {
                    right(row, col) = right_column[static_cast<std::size_t>(row)] / norm;
                }
                completed = true;
            }
            if (!completed) {
                throw std::domain_error("randomized SVD could not complete its right singular-vector basis");
            }
        }
        return {std::move(left), std::move(right), std::move(values)};
    }

    static Result compact_state(const FactorType& source, const FactorType& range, int requested_rank) {
        const FactorType core = transpose_multiply(range, source);
        Result compact = compact_svd(core, requested_rank);
        FactorType left = multiply(range, compact.left);
        return {std::move(left), std::move(compact.right), std::move(compact.values)};
    }

    static NystromResult nystrom_state(
      const FactorType& basis, const FactorType& product, Scalar shift, int requested_rank, bool rank_revealing) {
        FactorType shifted(product);
        for (int row = 0; row < shifted.rows(); ++row) {
            for (int col = 0; col < shifted.cols(); ++col) {
                shifted(row, col) += shift * basis(row, col);
                if (!std::isfinite(shifted(row, col))) {
                    throw std::domain_error("randomized Nystrom shifted range contains a nonfinite coefficient");
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
                    throw std::domain_error(
                      "randomized Nystrom stabilized Gram matrix contains a nonfinite coefficient");
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
                throw std::domain_error("randomized Nystrom approximation requires a positive-semidefinite matrix");
            }
            const Scalar stabilized = fdapde::max(shift, eigenvalue);
            const Scalar inverse_root =
              rank_revealing && eigenvalue <= spectral_roundoff ? Scalar(0) : Scalar(1) / std::sqrt(stabilized);
            if (!std::isfinite(inverse_root)) {
                throw std::domain_error("randomized Nystrom stabilized Gram inverse is not representable");
            }
            for (int row = 0; row < inverse_sqrt.rows(); ++row) {
                for (int col = 0; col < inverse_sqrt.cols(); ++col) {
                    inverse_sqrt(row, col) +=
                      eigenvectors(row, component) * inverse_root * eigenvectors(col, component);
                }
            }
        }

        const FactorType factor = multiply(shifted, inverse_sqrt);
        FactorType factor_transpose(factor.cols(), factor.rows());
        for (int row = 0; row < factor.rows(); ++row) {
            for (int col = 0; col < factor.cols(); ++col) factor_transpose(col, row) = factor(row, col);
        }
        Result compact = compact_svd(factor_transpose, requested_rank);
        SingularValuesType values(requested_rank);
        const Scalar value_roundoff =
          Scalar(128) * std::numeric_limits<Scalar>::epsilon() * static_cast<Scalar>(factor.rows());
        int resolved_rank = 0;
        for (int i = 0; i < requested_rank; ++i) {
            const Scalar square = compact.values[i] * compact.values[i];
            const Scalar value = square - shift;
            const Scalar tolerance = value_roundoff * fdapde::max(Scalar(1), fdapde::max(square, shift));
            if (!std::isfinite(value) || value < -tolerance) {
                throw std::domain_error("randomized Nystrom produced a non-positive-semidefinite approximation");
            }
            values[i] = value > tolerance ? value : Scalar(0);
            if (values[i] > Scalar(0)) resolved_rank = i + 1;
        }
        if (rank_revealing && resolved_rank < requested_rank) {
            FactorType resolved(compact.right.rows(), resolved_rank);
            for (int row = 0; row < resolved.rows(); ++row) {
                for (int col = 0; col < resolved.cols(); ++col) resolved(row, col) = compact.right(row, col);
            }
            FactorType missing(compact.right.rows(), requested_rank - resolved_rank);
            missing.set_zero();
            const FactorType completion = orthonormalize(missing, resolved);
            for (int row = 0; row < completion.rows(); ++row) {
                for (int col = 0; col < completion.cols(); ++col) {
                    compact.right(row, resolved_rank + col) = completion(row, col);
                }
            }
        }
        return {std::move(compact.right), std::move(values)};
    }

    static Scalar eigen_residual(const FactorType& source, const NystromResult& result) {
        Scalar maximum = Scalar(0);
        for (int col = 0; col < result.values.rows(); ++col) {
            Scalar norm = Scalar(0);
            for (int row = 0; row < source.rows(); ++row) {
                Scalar value = Scalar(0);
                for (int inner = 0; inner < source.cols(); ++inner) {
                    value += source(row, inner) * result.vectors(inner, col);
                }
                value -= result.vectors(row, col) * result.values[col];
                if (!std::isfinite(value)) {
                    throw std::domain_error("randomized Nystrom eigen-residual contains a nonfinite coefficient");
                }
                norm = scale_safe_hypot(norm, value);
            }
            maximum = fdapde::max(maximum, norm);
        }
        return std::sqrt(Scalar(2)) * maximum;
    }

    template <typename ResultType> static Scalar residual(const FactorType& source, const ResultType& result) {
        Scalar maximum = Scalar(0);
        for (int col = 0; col < result.values.rows(); ++col) {
            Scalar norm = Scalar(0);
            for (int row = 0; row < source.rows(); ++row) {
                Scalar value = Scalar(0);
                for (int k = 0; k < source.cols(); ++k) value += source(row, k) * result.right(k, col);
                value -= result.left(row, col) * result.values[col];
                if (!std::isfinite(value)) {
                    throw std::domain_error("randomized SVD residual contains a nonfinite coefficient");
                }
                norm = scale_safe_hypot(norm, value);
            }
            maximum = fdapde::max(maximum, norm);
        }
        return maximum;
    }
};

}   // namespace fdapde::internals

#endif   // __FDAPDE_LINALG_RANDOMIZED_SVD_H__
