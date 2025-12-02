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

TEST(linear_algebra, symmetric) {
    // static-sized
    {
        Matrix<double, 3, 3> M({1, 2, 3, 4, 5, 6, 7, 8, 9});

        auto s1 = M.as_symmetric<Lower>();
        Matrix<double, 3, 3> s1_({1, 4, 7, 4, 5, 8, 7, 8, 9});
        EXPECT_EQ(s1, s1_);	
        auto s2 = M.as_symmetric<Upper>();
        Matrix<double, 3, 3> s2_({1, 2, 3, 2, 5, 6, 3, 6, 9});
        EXPECT_EQ(s2, s2_);

        // test storage order
        SymmetricMatrix<double, 3, 3, RowMajor> M1({1, 2, 3, 4, 5, 6});
        Matrix<double, 3, 3> M1_({1, 2, 4, 2, 3, 5, 4, 5, 6});
        EXPECT_EQ(M1, M1_);
        SymmetricMatrix<double, 3, 3, ColMajor> M2({1, 2, 3, 4, 5, 6});
        Matrix<double, 3, 3> M2_({1, 2, 3, 2, 4, 5, 3, 5, 6});
        EXPECT_EQ(M2, M2_);

        // symmetric arithmetic
        auto s3 = M1 + M1;
        Matrix<double, 3, 3> s3_({2, 4, 8, 4, 6, 10, 8, 10, 12});
        EXPECT_EQ(s3, s3_);
        auto s4 = 3 * M1;
        Matrix<double, 3, 3> s4_({3, 6, 12, 6, 9, 15, 12, 15, 18});
        EXPECT_EQ(s4, s4_);
	SymmetricMatrix<double, 3, 3> M3 = M1;
        M3.cwise() += 2;
        Matrix<double, 3, 3> s5_({3, 4, 6, 4, 5, 7, 6, 7, 8});
        EXPECT_EQ(M3, s5_);
	M3.diagonal().cwise() = 5;	
        Matrix<double, 3, 3> s6_({5, 4, 6, 4, 5, 7, 6, 7, 5});
        EXPECT_EQ(M3, s6_);
        auto blk = M2.block<2, 2>(0, 0);
        Matrix<double, 2, 2> s7_({1, 2, 2, 4});
        EXPECT_TRUE(blk == s7_);

	// const-access
        EXPECT_EQ(M2(1, 0), 2);
        EXPECT_EQ(M2(0, 1), 2);
        EXPECT_EQ(M2(2, 2), 6);

        // non-const access
        SymmetricMatrix<double, 3, 3> M4 = M2;   // rhs of different StorageOrder
        M4(2, 0) = 8;
        EXPECT_EQ(M4(2, 0), 8);
        EXPECT_EQ(M4(0, 2), 8);   // symmetric invariance preserved
    }

    // dynamic sized
    // {
    // }
}
