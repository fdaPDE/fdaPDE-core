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

#ifndef __FDAPDE_LINALG_KRONECKER_H__
#define __FDAPDE_LINALG_KRONECKER_H__

#include "header_check.h"

namespace fdapde {

/// @brief builds an owning CSR Kronecker product with checked dimensions and integral arithmetic
template <typename LhsScalar, typename RhsScalar>
auto kron(const SparseMatrix<LhsScalar>& lhs, const SparseMatrix<RhsScalar>& rhs) {
    using ResultScalar =
      std::common_type_t<typename SparseMatrix<LhsScalar>::Scalar, typename SparseMatrix<RhsScalar>::Scalar>;
    using Result = SparseMatrix<ResultScalar>;

    const int result_rows = internals::checked_matrix_size(lhs.rows(), rhs.rows());
    const int result_cols = internals::checked_matrix_size(lhs.cols(), rhs.cols());
    const auto count_nonzero_entries = [](const auto& matrix) {
        std::size_t count = 0;
        for (int row = 0; row < matrix.rows(); ++row) {
            for (const auto entry : matrix.row(row)) {
                if (entry.value() != typename std::remove_cvref_t<decltype(matrix)>::Scalar {}) ++count;
            }
        }
        return count;
    };
    const std::size_t lhs_entries = count_nonzero_entries(lhs);
    const std::size_t rhs_entries = count_nonzero_entries(rhs);
    fdapde_strong_assert(
      lhs_entries == 0 || rhs_entries <= static_cast<std::size_t>(std::numeric_limits<int>::max()) / lhs_entries,
      std::length_error, "sparse Kronecker product exceeds the supported int range");

    std::vector<typename Result::triplet_type> triplets;
    triplets.reserve(lhs_entries * rhs_entries);
    for (int lhs_row = 0; lhs_row < lhs.rows(); ++lhs_row) {
        for (const auto lhs_entry : lhs.row(lhs_row)) {
            if (lhs_entry.value() == typename SparseMatrix<LhsScalar>::Scalar {}) continue;
            for (int rhs_row = 0; rhs_row < rhs.rows(); ++rhs_row) {
                for (const auto rhs_entry : rhs.row(rhs_row)) {
                    const ResultScalar a = static_cast<ResultScalar>(lhs_entry.value());
                    const ResultScalar b = static_cast<ResultScalar>(rhs_entry.value());
                    if constexpr (std::is_integral_v<ResultScalar>) {
                        // test the product range before multiplication can overflow
                        constexpr ResultScalar max = std::numeric_limits<ResultScalar>::max();
                        bool fits = true;
                        if (a != 0 && b != 0) {
                            if constexpr (std::is_signed_v<ResultScalar>) {
                                constexpr ResultScalar min = std::numeric_limits<ResultScalar>::min();
                                fits =
                                  a > 0 ? (b > 0 ? a <= max / b : b >= min / a) : (b > 0 ? a >= min / b : b >= max / a);
                            } else {
                                fits = a <= max / b;
                            }
                        }
                        fdapde_strong_assert(
                          fits, std::overflow_error, "sparse Kronecker coefficient exceeds the scalar range");
                    }
                    const ResultScalar value = a * b;
                    if (value == ResultScalar {}) continue;
                    triplets.emplace_back(
                      lhs_row * rhs.rows() + rhs_row, lhs_entry.column() * rhs.cols() + rhs_entry.column(), value);
                }
            }
        }
    }
    return Result(result_rows, result_cols, triplets);
}

}   // namespace fdapde

#endif   // __FDAPDE_LINALG_KRONECKER_H__
