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

TEST(linear_algebra, block) {
    // static-sized
    {
        static constexpr Matrix<double, 3, 3> A({1, 2, 3, 4, 5, 6, 7, 8, 9});
        // static sized block
        constexpr auto b1 = A.block<2, 2>(0, 0);
        static_assert(b1.rows() == 2);
        static_assert(b1.cols() == 2);
        static_assert(b1.size() == 4);
        static_assert(b1 == Matrix<double, 2, 2>({1, 2, 4, 5}));
        // dynamic sized block of static-sized matrix
        constexpr auto b2 = A.block(0, 0, 2, 2);
        static_assert(b2.rows() == 2);
        static_assert(b2.cols() == 2);
        static_assert(b2.size() == 4);
        static_assert(b2 == Matrix<double, 2, 2>({1, 2, 4, 5}));
        // column block
        constexpr auto b3 = A.col(0);
        static_assert(b3.rows() == 3);
        static_assert(b3.cols() == 1);
        static_assert(b3.size() == 3);
        static_assert(b3 == Matrix<double, 3, 1>({1, 4, 7}));
        // row block
        constexpr auto b4 = A.row(0);
        static_assert(b4.rows() == 1);
        static_assert(b4.cols() == 3);
        static_assert(b4.size() == 3);
        static_assert(b4 == Matrix<double, 1, 3>({1, 2, 3}));

        // special block accessors
        constexpr auto b5 = A.top_rows<2>();
        static_assert(b5.rows() == 2);
        static_assert(b5.cols() == 3);
        static_assert(b5.size() == 6);
        static_assert(b5 == Matrix<double, 2, 3>({1, 2, 3, 4, 5, 6}));
        // colwise block reduction
        static_assert([b5]() {
            constexpr auto e = b5.colwise().prod();
            return e == Matrix<double, 1, 3>({4, 10, 18});
        }());

        constexpr auto b6 = A.top_rows(2);
        static_assert(b6.rows() == 2);
        static_assert(b6.cols() == 3);
        static_assert(b6.size() == 6);
        static_assert(b6 == Matrix<double, 2, 3>({1, 2, 3, 4, 5, 6}));
        // block redux
        static_assert(b6.squared_norm() == 91);
        static_assert(b6.inf_norm() == 6);

        constexpr auto b7 = A.bottom_rows<2>();
        static_assert(b7.rows() == 2);
        static_assert(b7.cols() == 3);
        static_assert(b7.size() == 6);
        static_assert(b7 == Matrix<double, 2, 3>({4, 5, 6, 7, 8, 9}));
        // block coeff-wise
        static_assert([b7]() {
            constexpr auto e = b7.cwise().exp();

            constexpr double r1 = 54.598150033144239078;
            constexpr double r2 = 148.41315910257660342;
            constexpr double r3 = 403.42879349273512260;
            constexpr double r4 = 1096.6331584284585992;
            constexpr double r5 = 2980.9579870417282747;
            constexpr double r6 = 8103.0839275753840077;
            return almost_equal(e, Matrix<double, 2, 3>({r1, r2, r3, r4, r5, r6}));
        }());

        constexpr auto b8 = A.bottom_rows(2);
        static_assert(b8.rows() == 2);
        static_assert(b8.cols() == 3);
        static_assert(b8.size() == 6);
        static_assert(b8 == Matrix<double, 2, 3>({4, 5, 6, 7, 8, 9}));
        // block-transpose
        static_assert(b8.transpose() == Matrix<double, 3, 2>({4, 7, 5, 8, 6, 9}));

        constexpr auto b9 = A.left_cols<2>();
        static_assert(b9.rows() == 3);
        static_assert(b9.cols() == 2);
        static_assert(b9.size() == 6);
        static_assert(b9 == Matrix<double, 3, 2>({1, 2, 4, 5, 7, 8}));
        // column block of block
        static_assert([b9]() {
            constexpr auto e = b9.col(0);
            return e.sum() == 12;
        }());

        constexpr auto b10 = A.left_cols(2);
        static_assert(b10.rows() == 3);
        static_assert(b10.cols() == 2);
        static_assert(b10.size() == 6);
        static_assert(b10 == Matrix<double, 3, 2>({1, 2, 4, 5, 7, 8}));
        // block of block-expression
        static_assert([b1, b10]() {
            constexpr auto e = 2 * b1 + b10.transpose() * b10;
            return e.block<1, 2>(0, 0) == Matrix<double, 1, 2>({68, 82});
        }());

        constexpr auto b11 = A.right_cols<2>();
        static_assert(b11.rows() == 3);
        static_assert(b11.cols() == 2);
        static_assert(b11.size() == 6);
        static_assert(b11 == Matrix<double, 3, 2>({2, 3, 5, 6, 8, 9}));
        // rowwise block redux
        static_assert([b11]() {
            constexpr auto e = b11.rowwise().sum();
            return e == Matrix<double, 3, 1>({5, 11, 17});
        }());
        // block expression
        static_assert((2 * b10 + b11) == Matrix<double, 3, 2>({4, 7, 13, 16, 22, 25}));

        constexpr auto b12 = A.right_cols(2);
        static_assert(b12.rows() == 3);
        static_assert(b12.cols() == 2);
        static_assert(b12.size() == 6);
        static_assert(b11 == Matrix<double, 3, 2>({2, 3, 5, 6, 8, 9}));
        // row block of block
        static_assert([b12]() {
            constexpr auto e = b12.row(1);
            return almost_equal(e.mean(), 11. / 2);
        }());
    }

    // dynamic sized
    {
        Matrix<double, Dynamic, Dynamic> A(8, 10);
        auto b = A.block<5, 5>(1, 1);
        EXPECT_EQ(b.rows(), 5);
        EXPECT_EQ(b.cols(), 5);
        EXPECT_EQ(b.size(), 25);

        auto check_block_eq = [](const auto& mtx, auto value) {
            for (int i = 0; i < mtx.rows(); ++i) {
                for (int j = 0; j < mtx.cols(); ++j) {
                    if ((i >= 1 && i < 6) && (j >= 1 && j < 6)) {
                        EXPECT_EQ(mtx(i, j), value);
                    } else {
                        EXPECT_EQ(mtx(i, j), 0);
                    }
                }
            }
        };
        // assignment
        b = Matrix<double, 5, 5>::Ones();
        check_block_eq(A, 1);
        // compound block arithmetic
        Matrix<double, 5, 5> B = 4 * Matrix<double, 5, 5>::Ones();
        b += B;
        check_block_eq(A, 5);
        b *= 5;
        check_block_eq(A, 25);
        b /= 5;
        check_block_eq(A, 5);
        b -= B;
        check_block_eq(A, 1);
        b *= B;
        check_block_eq(A, 20);
    }
}
