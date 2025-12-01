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

TEST(linear_algebra, triangular) {
    // static-sized
    {
        Matrix<double, 3, 3> M({1, 2, 3, 4, 5, 6, 7, 8, 9});

        auto tb1 = M.triangular_block<Lower>();
        Matrix<double, 3, 3> tb1_({1, 0, 0, 4, 5, 0, 7, 8, 9});
        EXPECT_EQ(tb1, tb1_);
        auto tb2 = M.triangular_block<Upper>();
        Matrix<double, 3, 3> tb2_({1, 2, 3, 0, 5, 6, 0, 0, 9});
        EXPECT_EQ(tb2, tb2_);
	// coeffwise on triangular blocks
	tb1.cwise() += 1;
	Matrix<double, 3, 3> tb3_({2, 0, 0, 5, 6, 0, 8, 9, 10});
        EXPECT_EQ(tb1, tb3_);
	tb1.cwise() -= 1;
	
        // test storage order
        LowerTriangularMatrix<double, 3, 3, RowMajor> M1({1, 2, 3, 4, 5, 6});
        Matrix<double, 3, 3> M1_({1, 0, 0, 2, 3, 0, 4, 5, 6});
        EXPECT_EQ(M1, M1_);
        LowerTriangularMatrix<double, 3, 3, ColMajor> M2({1, 2, 3, 4, 5, 6});
        Matrix<double, 3, 3> M2_({1, 0, 0, 2, 4, 0, 3, 5, 6});
        EXPECT_EQ(M2, M2_);

        // forward substitution solver
        Vector<double, 3> ls1 = (tb1 + tb1).solve(Vector<double, 3>({1, 1, 1}));
        Vector<double, 3> ls1_({0.5, -0.3, -1.2 / 18});
        EXPECT_TRUE(almost_equal(ls1, ls1_));
        // backward substitution solver
        Vector<double, 3> ls2 = (tb2 + tb2).solve(Vector<double, 3>({1, 1, 1}));
        Vector<double, 3> ls2_({1. / 2 * (1 - 4. / 10 * (1 - 12. / 18) - 6. / 18), 1. / 10 * (1 - 12. / 18), 1. / 18});
        EXPECT_TRUE(almost_equal(ls2, ls2_));

        // lower-triangular / lower-triangular product (different storage order)
        auto e1 = M1 * M2;
        Matrix<double, 3, 3> e1_({1, 0, 0, 8, 12, 0, 32, 50, 36});
        EXPECT_EQ(e1, e1_);
        UpperTriangularMatrix<double, 3, 3> M3({1, 2, 3, 4, 5, 6});
        // lower-triangular / upper-triangular product
        auto e2 = M1 * M3;
        Matrix<double, 3, 3> e2_({1, 0, 0, 0, 12, 0, 0, 0, 36});
        EXPECT_EQ(e2, e2_);
        // upper-triangular / upper-triangular product
        UpperTriangularMatrix<double, 3, 3> M4 = M3;
        auto e3 = M3 * M4;
        Matrix<double, 3, 3> e3_({1, 10, 31, 0, 16, 50, 0, 0, 36});
        EXPECT_EQ(e3, e3_);

	// copy assignment
        UpperTriangularMatrix<double, 3, 3> M5;
        M5 = M3;
        EXPECT_EQ(M5, M3);

        // triangular arithmetic
        auto e4 = 5 * M1;
        Matrix<double, 3, 3> e4_({5, 0, 0, 10, 15, 0, 20, 25, 30});
        EXPECT_EQ(e4, e4_);
        auto e5 = M1 / 2.0;
        Matrix<double, 3, 3> e5_({1. / 2, 0, 0, 1, 3. / 2, 0, 2, 5. / 2, 3});
        EXPECT_EQ(e5, e5_);
        auto e6 = M1 + M1;
        Matrix<double, 3, 3> e6_({2, 0, 0, 4, 6, 0, 8, 10, 12});
        EXPECT_EQ(e6, e6_);
        auto e7 = M1 + M2;   // different storage order	
        Matrix<double, 3, 3> e7_({2, 0, 0, 4, 7, 0, 7, 10, 12});
        EXPECT_EQ(e7, e7_);
        auto e8 = M1 - M1;
        Matrix<double, 3, 3> e8_({0, 0, 0, 0, 0, 0, 0, 0, 0});
        EXPECT_EQ(e8, e8_);
        auto e9 = tb1 + tb1;
        Matrix<double, 3, 3> e9_ = (M + M).triangular_block<Lower>();
        EXPECT_EQ(e9, e9_);
        auto e10 = tb1 * tb1;
        Matrix<double, 3, 3> e10_({1, 0, 0, 24, 25, 0, 102, 112, 81});
        EXPECT_EQ(e10, e10_);

	// const access
        EXPECT_EQ(M1(1, 0), 2);   // Lower - RowMajor
        EXPECT_EQ(M2(2, 0), 3);   // Lower - ColMajor
        EXPECT_EQ(M1(0, 2), 0);   // off-triangular returns 0
        // non-const access
        M1(1, 0) = 5;
        EXPECT_EQ(M1(1, 0), 5);
        M1(0, 2) = 5;   // off-triangular accesses are absorbed
        EXPECT_EQ(M1(0, 2), 0);

	// coeffwise on owning triangular matrix
        M1.cwise() = 1;
        Matrix<double, 3, 3> e11_({1, 0, 0, 1, 1, 0, 1, 1, 1});
	EXPECT_EQ(M1, e11_);
    }

    // dynamic sized
    {
        LowerTriangularMatrix<double, Dynamic, Dynamic> M1(5, 5);
        EXPECT_EQ(M1.rows(), 5);
        EXPECT_EQ(M1.cols(), 5);
        EXPECT_EQ(M1.size(), 25);

        // check is zero initialized
        for (int i = 0; i < M1.rows(); ++i) {
            for (int j = 0; j < M1.cols(); ++j) {
                double m1_ = M1(i, j);
                EXPECT_EQ(m1_, 0);
            }
        }
    }
}
