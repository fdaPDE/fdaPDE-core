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

#include <gtest/gtest.h>   // testing framework

#include "../../fdaPDE/linear_algebra/binary_matrix.h"
using fdapde::Dynamic;
using fdapde::core::BinaryMatrix;
using fdapde::core::BinaryVector;

// replaced by tests/linear_algebra/historical_boolean.cpp: HistoricalBoolean.static_sized_matrix

// replaced by tests/linear_algebra/historical_boolean.cpp: HistoricalBoolean.dynamic_sized_matrix

// replaced by tests/linear_algebra/historical_boolean.cpp: HistoricalBoolean.binary_vector

// replaced by tests/linear_algebra/historical_boolean.cpp: HistoricalBoolean.block_operations

// replaced by tests/linear_algebra/historical_boolean.cpp: HistoricalBoolean.binary_expresssions

// replaced by tests/linear_algebra/historical_boolean.cpp: HistoricalBoolean.visitors

// replaced by tests/linear_algebra/historical_boolean.cpp: HistoricalBoolean.block_repeat

TEST(binary_matrix_test, eigen_assignment_and_construct) {
    DMatrix<int> e1(5, 5);
    e1.setZero();
    e1(1, 2) = 4;
    e1(2, 3) = 5;
    e1(4, 4) = 6;
    e1(3, 4) = 7;

    BinaryMatrix<Dynamic> m1(e1);
    EXPECT_TRUE(m1.rows() == 5);
    EXPECT_TRUE(m1.cols() == 5);
    EXPECT_TRUE(m1.count() == 4);
    EXPECT_TRUE(m1(1, 2) && m1(2, 3) && m1(4, 4) && m1(3, 4));

    BinaryMatrix<Dynamic> m2;
    m2 = e1;
    EXPECT_TRUE(m2.rows() == 5);
    EXPECT_TRUE(m2.cols() == 5);
    EXPECT_TRUE(m2.count() == 4);
    EXPECT_TRUE(m2(1, 2) && m2(2, 3) && m2(4, 4) && m2(3, 4));
}

// replaced by tests/linear_algebra/historical_boolean.cpp: HistoricalBoolean.reshaped
