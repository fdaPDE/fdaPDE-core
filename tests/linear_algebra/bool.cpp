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

        // block accessors
        static constexpr Matrix<bool, 3, 3> B11({1, 1, 0, 0, 1, 1, 0, 1, 0});
        static_assert(B11.block<2, 2>(0, 0) == Matrix<bool, 2, 2>({1, 1, 0, 1}));   // static-sized block
        static_assert(B11.block(0, 0, 2, 2) == Matrix<bool, 2, 2>({1, 1, 0, 1}));   // dynamic-sized block
        static_assert(B11.col(0) == Matrix<bool, 3, 1>({1, 0, 0}));
        static_assert(B11.row(1) == Matrix<bool, 1, 3>({0, 1, 1}));
        static_assert(B11.top_rows(2) == Matrix<bool, 2, 3>({1, 1, 0, 0, 1, 1}));
        static_assert(B11.bottom_rows(2) == Matrix<bool, 2, 3>({0, 1, 1, 0, 1, 0}));
        static_assert(B11.left_cols(2) == Matrix<bool, 3, 2>({1, 1, 0, 1, 0, 1}));
        static_assert(B11.right_cols(2) == Matrix<bool, 3, 2>({1, 0, 1, 1, 1, 0}));

	// non-const block accessors
        Matrix<bool, 9, 9> B12;   // this should span more than one bitpack on 64-bit machines
        B12.block<7, 7>(1, 1).set();
        for (int i = 0; i < 9; ++i) {
            for (int j = 0; j < 9; ++j) {
                if ((i >= 1 && i <= 7) && (j >= 1 && j <= 7)) {
                    EXPECT_EQ(B12(i, j), 1);
                } else {
                    EXPECT_EQ(B12(i, j), 0);
                }
            }
        }
        B12.block<7, 7>(1, 1).clear();
        for (int i = 0; i < 9; ++i) {
            for (int j = 0; j < 9; ++j) { EXPECT_EQ(B12(i, j), 0); }
        }
        // block assignment
        Matrix<bool, 3, 3> B13({1, 0, 0, 0, 1, 0, 0, 0, 1});
        Matrix<bool, 3, 3> B14({1, 1, 0, 1, 1, 0, 0, 1, 1});
        auto B15 = B12.block<3, 3>(1, 1);
        B15 = B13 & B14;
        EXPECT_EQ(B15, B13);
        B15 = B13 | B14;
        EXPECT_EQ(B15, B14);
        Matrix<bool, 3, 3> B16({0, 1, 0, 1, 0, 0, 0, 1, 0});
        B15 = B13 ^ B14;
        EXPECT_EQ(B15, B16);

        // compound algebra
        Matrix<bool, 3, 3> B17({1, 1, 0, 1, 1, 0, 0, 1, 1});
        B17 &= B13;
        EXPECT_EQ(B17, B13);
        B17 |= B14;
        EXPECT_EQ(B17, B14);
        B17 ^= B13;
        EXPECT_EQ(B17, B16);

        // use blocks in expressions
        auto B18 = B14.block<2, 2>(0, 0);   // spans a single bitpack
        Matrix<bool, 2, 2> B19({1, 0, 0, 1});
        Matrix<bool, 2, 2> B20 = B18 & B19;
        EXPECT_EQ(B20, B19);
        auto B21 = B12.block(1, 1, 7, 7);   // spans over multiple bitpacks
        Matrix<bool, Dynamic, Dynamic> B22 = B21 & B21;
        EXPECT_EQ(B22, B21);
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

        // boolean reductions
        Matrix<bool, Dynamic, Dynamic> B4(30, 30);
        EXPECT_FALSE(B4.any());
        B4(29, 29) = 1;
        EXPECT_TRUE(B4.any());
        B4.set();
        EXPECT_TRUE(B4.all());
        B4(15, 14) = 0;
        EXPECT_FALSE(B4.all());
        EXPECT_EQ(B4.count(), 30 * 30 - 1);

    }

    // ternary selection
    Matrix<double, 3, 3> M1({1, 2, 3, 4, 5, 6, 7, 8, 9});
    Matrix<double, 3, 3> M2 = 10 * M1;
    Matrix<bool, 3, 3> B1({1, 1, 1, 0, 1, 0, 1, 1, 1});
    auto s = B1.select(M1, M2);
    Matrix<double, 3, 3> M4({1, 2, 3, 40, 5, 60, 7, 8, 9});
    EXPECT_EQ(s, M4);

    // reshaping
    Matrix<bool, 4, 4> B5;
    B5.top_rows(2).set();
    auto B6 = B5.reshape(2, 8);
    
    Matrix<bool, 1, 8> v;
    v.set();
    EXPECT_EQ(B6.row(0), v);    

    // view
    int x = 3;
    MatrixView<bool, 2, 2> view(&x);
    Matrix<bool, 2, 2> B7({1, 1, 0, 0});
    EXPECT_EQ(view, B7);
    view.set();
    EXPECT_EQ(x, 15); // x = 0b1111
    Matrix<bool, 2, 2> B8({1, 1, 1, 1});
    EXPECT_EQ(view, B8);
    view.clear();
    Matrix<bool, 2, 2> B9({0, 0, 0, 0});
    EXPECT_EQ(x, 0);
    EXPECT_EQ(view, B9);

    std::vector<int> y(6);
    MatrixView<bool, Dynamic, Dynamic> view2(y.data(), 10, 10);
    y[0] = 2;
    y[1] = 2;
    EXPECT_EQ(view2(0, 1), true);
    EXPECT_EQ(view2(3, 3), true);
    view2.set();
    Matrix<bool, Dynamic, Dynamic> B10(10, 10);
    B10.set();
    EXPECT_EQ(view2, B10);
    auto B11 = view2 ^ view2;
    B10.clear();
    EXPECT_EQ(B11, B10);


    // static constructors
    Matrix<bool, Dynamic, Dynamic> B12 = Matrix<bool, Dynamic, Dynamic>::Ones(20, 20);
    // TODO
}
