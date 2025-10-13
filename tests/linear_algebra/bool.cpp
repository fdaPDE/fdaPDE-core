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
#include <gtest/gtest.h>   // testing framework
using namespace fdapde;

TEST(linear_algebra, boolean) {
    // static sized
    {
        static constexpr Matrix<bool, 2, 2> B1;
        // check dimensions
        static_assert(B1.rows() == 2);
        static_assert(B1.cols() == 2);
        static_assert(B1.size() == 4);
        // query all false
        static_assert([]() {
            for (int i = 0; i < B1.rows(); ++i) {
                for (int j = 0; j < B1.cols(); ++j) {
                    if (B1(i, j)) return false;
                }
            }
            return true;
        }());

        Matrix<bool, 3, 3> B2;
        // non-const access via set
        B2(1, 1).set();
        EXPECT_EQ(B2(1, 1), true);
        B2(1, 1).clear();
        EXPECT_EQ(B2(1, 1), false);
        // non-const access via operator=
        B2(2, 2) = 1;
        EXPECT_EQ(B2(2, 2), true);

        Matrix<bool, 3, 3> B3({1, 1, 1, 0, 0, 0, 1, 1, 0});
        B2 = B3;
        EXPECT_EQ(B2, B3);

        constexpr Vector<bool, 2> v1({1, 0});
        static_assert(v1.rows() == 2);
        static_assert(v1.cols() == 1);
        static_assert(v1.size() == 2);
        static_assert(v1[0] == 1 && v1[1] == 0);   // access by subscript

	// bitwise negation
        auto B4 = ~B2;
        for (int i = 0; i < B4.rows(); ++i) {
            for (int j = 0; j < B4.cols(); ++j) {
                if (B2(i, j) == false) {
                    EXPECT_EQ(B4(i, j), true);
                } else {
                    EXPECT_EQ(B4(i, j), false);
                }
            }
        }

	// bitwise arithmetic
        Matrix<bool, 3, 3> B5({1, 0, 0, 0, 0, 0, 0, 0, 0});
        auto B6 = B5 & B3;
        EXPECT_EQ(B6, B5);
        auto B7 = B5 | B3;
        EXPECT_EQ(B7, B3);
        Matrix<bool, 3, 3> B8({0, 1, 1, 0, 0, 0, 1, 1, 0});
        auto B9 = B5 ^ B7;
        EXPECT_EQ(B9, B8);
	// assign expression to matrix
	Matrix<bool, 3, 3> B10 = B9;
	EXPECT_EQ(B10, B9);
    }

    // dynamic sized
    {
        Matrix<bool, Dynamic, Dynamic> B1(10, 10);
        // check dimensions
        EXPECT_EQ(B1.rows(), 10);
        EXPECT_EQ(B1.cols(), 10);
        EXPECT_EQ(B1.size(), 100);
        // 100 bits must fit on 2 64bit-wide slots
        EXPECT_EQ(B1.bitpacks(), (1 + 100 / (sizeof(Matrix<bool, Dynamic, Dynamic>::bitpack_t) * 8)));
        // matrix is zero initialized
        for (int i = 0; i < B1.rows(); ++i) {
            for (int j = 0; j < B1.cols(); ++j) { EXPECT_EQ(B1(i, j), false); }
        }

	// value-initialization
        Matrix<bool, Dynamic, Dynamic> B2(10, 10, true);
        // matrix is 1 initialized
        for (int i = 0; i < B2.rows(); ++i) {
            for (int j = 0; j < B2.cols(); ++j) { EXPECT_EQ(B2(i, j), true); }
        }
	
        // non-const access
        B1(4, 4) = 1;
        EXPECT_EQ(B1(4, 4), true);

        // create empty, assign
        Matrix<bool, Dynamic, Dynamic> B3;
        B3 = B2;
        EXPECT_EQ(B3.rows(), B2.cols());
        EXPECT_EQ(B3.cols(), B2.cols());
        EXPECT_EQ(B3.size(), B2.size());
        EXPECT_EQ(B3, B2);

	// clear
	B3.clear();
        for (int i = 0; i < B3.rows(); ++i) {
            for (int j = 0; j < B3.cols(); ++j) { EXPECT_EQ(B3(i, j), false); }
        }
    }
}
