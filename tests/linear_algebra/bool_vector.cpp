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

using namespace fdapde;

TEST(linear_algebra, dynamic_boolean_vector_from_std_vector) {
    std::vector<bool> data(130, false);
    data[1] = true;
    data[65] = true;
    data[129] = true;

    const Vector<bool, Dynamic> packed(data);
    EXPECT_EQ(packed.size(), 130);
    EXPECT_EQ(packed.count(), 3);
    EXPECT_TRUE(packed[1]);
    EXPECT_TRUE(packed[65]);
    EXPECT_TRUE(packed[129]);
    Vector<bool, Dynamic> packed_copy(data);
    EXPECT_EQ(packed, packed_copy);
    packed_copy[129] = false;
    EXPECT_NE(packed, packed_copy);
}

TEST(linear_algebra, dynamic_column_vector_from_row_expression) {
    const Matrix<int, 1, 3> row({1, 2, 3});
    const Matrix<int, Dynamic, 1> column = row;

    EXPECT_EQ(column.rows(), 3);
    EXPECT_EQ(column.cols(), 1);
    EXPECT_EQ(column[0], 1);
    EXPECT_EQ(column[1], 2);
    EXPECT_EQ(column[2], 3);
}
