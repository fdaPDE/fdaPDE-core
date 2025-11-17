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
    }

    // dynamic sized
    // {
    // }
}
