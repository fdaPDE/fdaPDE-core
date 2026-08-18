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

TEST(linear_algebra, boolean) {
    fdapde::Matrix<bool, 2, 3> fixed({true, false, true, false, true, false});
    EXPECT_EQ(fixed.rows(), 2);
    EXPECT_EQ(fixed.cols(), 3);
    EXPECT_TRUE(fixed(0, 0));
    EXPECT_FALSE(fixed(0, 1));

    fixed(1, 0).set();
    fixed(0, 2).clear();
    EXPECT_TRUE(fixed(1, 0));
    EXPECT_FALSE(fixed(0, 2));

    using dynamic_bool_matrix = fdapde::Matrix<bool, fdapde::Dynamic, fdapde::Dynamic>;
    constexpr int pack_size = static_cast<int>(dynamic_bool_matrix::PackSize);
    dynamic_bool_matrix dynamic(1, pack_size + 1);
    dynamic(0, pack_size - 1) = true;
    dynamic(0, pack_size) = true;

    dynamic_bool_matrix copy = dynamic;
    EXPECT_TRUE(copy(0, pack_size - 1));
    EXPECT_TRUE(copy(0, pack_size));
    copy(0, pack_size - 1).clear();
    EXPECT_FALSE(copy(0, pack_size - 1));
    EXPECT_TRUE(copy(0, pack_size));
    EXPECT_TRUE(dynamic(0, pack_size - 1));
}

// Current regression adapted from 86ff6d12:tests/linear_algebra/bool.cpp.
// Stable source: a2a9c88:test/src/binary_matrix_test.cpp.
// Stable declarations (9): static_sized_matrix, dynamic_sized_matrix, binary_vector, block_operations,
// binary_expresssions, visitors, block_repeat, eigen_assignment_and_construct, and reshaped.
// TODO(P4-B): cover exact pack sizing, resize, views, aliasing, Boolean vectors, row/column/block access,
// expressions and reductions, repeat, reshape, and select.
// Replace the historical Eigen assignment/construct assertion with native numeric-matrix conversion; do not
// restore an implicit Eigen bridge.
