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

#include <fdaPDE/linear_algebra.h>
#include <gtest/gtest.h>

#include <array>
#include <vector>

namespace fdapde {
namespace {

template <int StorageOrder> void check_owner_behavior() {
    using fixed_matrix = Matrix<int, 2, 3, StorageOrder>;
    using dynamic_column = Matrix<int, Dynamic, 1, StorageOrder>;

    constexpr int fixed_input[6] {1, 2, 3, 4, 5, 6};
    constexpr fixed_matrix matrix(fixed_input);
    static_assert(matrix.rows() == 2);
    static_assert(matrix.cols() == 3);
    static_assert(matrix(1, 2) == 6);

    const std::array<int, 6> expected_storage = StorageOrder == RowMajor
      ? std::array<int, 6> {1, 2, 3, 4, 5, 6}
      : std::array<int, 6> {1, 4, 2, 5, 3, 6};
    for (int i = 0; i < matrix.size(); ++i) { EXPECT_EQ(matrix.data()[i], expected_storage[i]); }

    const std::vector<int> input {1, 2, 3, 4, 5, 6};
    const fixed_matrix from_vector(input);
    EXPECT_EQ(from_vector, matrix);

    Matrix<int, Dynamic, 3, StorageOrder> dynamic_rows(2, 3);
    EXPECT_EQ(dynamic_rows.end() - dynamic_rows.begin(), 6);
    dynamic_rows.resize(4, 3);
    EXPECT_EQ(dynamic_rows.rows(), 4);
    EXPECT_EQ(dynamic_rows.cols(), 3);
    EXPECT_EQ(dynamic_rows.end() - dynamic_rows.begin(), 12);

    Matrix<int, 2, Dynamic, StorageOrder> dynamic_cols(2, 3);
    EXPECT_EQ(dynamic_cols.end() - dynamic_cols.begin(), 6);
    dynamic_cols.resize(2, 4);
    EXPECT_EQ(dynamic_cols.rows(), 2);
    EXPECT_EQ(dynamic_cols.cols(), 4);
    EXPECT_EQ(dynamic_cols.end() - dynamic_cols.begin(), 8);

    dynamic_column column(std::vector<int> {1, 2, 3});
    EXPECT_EQ(column.rows(), 3);
    EXPECT_EQ(column.cols(), 1);
    column = {4, 5};
    EXPECT_EQ(column.rows(), 2);
    EXPECT_EQ(column.cols(), 1);
    EXPECT_EQ(column[0], 4);
    EXPECT_EQ(column[1], 5);

    const Matrix<int, 1, 3, StorageOrder> row({7, 8, 9});
    const Matrix<int, 3, 1, StorageOrder> fixed_column_from_row(row);
    EXPECT_EQ(fixed_column_from_row[0], 7);
    EXPECT_EQ(fixed_column_from_row[1], 8);
    EXPECT_EQ(fixed_column_from_row[2], 9);

    const dynamic_column column_from_row(row);
    EXPECT_EQ(column_from_row.rows(), 3);
    EXPECT_EQ(column_from_row.cols(), 1);
    EXPECT_EQ(column_from_row[0], 7);
    EXPECT_EQ(column_from_row[1], 8);
    EXPECT_EQ(column_from_row[2], 9);

    const Matrix<int, 1, 1, StorageOrder> scalar_vector(11);
    const dynamic_column column_from_scalar(scalar_vector);
    EXPECT_EQ(column_from_scalar.rows(), 1);
    EXPECT_EQ(column_from_scalar[0], 11);

    constexpr int OtherStorageOrder = StorageOrder == RowMajor ? ColMajor : RowMajor;
    const Matrix<int, 2, 3, OtherStorageOrder> other_order(matrix);
    EXPECT_EQ(other_order(0, 1), 2);
    EXPECT_EQ(other_order(1, 2), 6);
}

}   // namespace

TEST(NativeDenseMatrix, OwnerShapeStorageAndVectorCopy) {
    check_owner_behavior<RowMajor>();
    check_owner_behavior<ColMajor>();

    const auto ones = [](int, int) { return 1; };
    ProceduralMatrix<decltype(ones), 2, Dynamic> procedural(2, 3, ones);
    procedural.resize(2, 4);
    EXPECT_EQ(procedural.rows(), 2);
    EXPECT_EQ(procedural.cols(), 4);
    EXPECT_EQ(procedural(1, 3), 1);
}

}   // namespace fdapde
