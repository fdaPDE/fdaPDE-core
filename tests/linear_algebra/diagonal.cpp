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

TEST(linear_algebra, diagonal) {
    // static-sized
    {
        static constexpr Matrix<double, 2, 2> M1({1, 2, 3, 4});
        static constexpr Vector<double, 2> v({1, 1});

        // diagonal expressions are vector expressions
        static_assert(M1.diagonal().rows() == 2);
        static_assert(M1.diagonal().cols() == 1);
        static_assert(M1.diagonal().size() == 2);
        static_assert(M1.diagonal() == Vector<double, 2>({1, 4}));
        static_assert((M1.diagonal() + v) == Vector<double, 2>({2, 5}));

        Matrix<double, 3, 3> M2({1, 2, 3, 4, 5, 6, 7, 8, 9});
        Vector<double, 3> w({3, 3, 3});

        auto check_diag_eq = [](const auto& mtx, auto value) {
            for (int i = 0; i < mtx.rows(); ++i) {
                for (int j = 0; j < mtx.cols(); ++j) {
                    if (i == j) { EXPECT_EQ(mtx(i, j), value); }
                }
            }
        };

        // diagonal assign
        M2.diagonal() = Vector<double, 3>::Ones();
        check_diag_eq(M2, 1);
        M2.diagonal() += w;
        check_diag_eq(M2, 4);
        M2.diagonal() *= 10;
        check_diag_eq(M2, 40);
        M2.diagonal() /= 10;
        check_diag_eq(M2, 4);
        M2.diagonal() -= w;
        check_diag_eq(M2, 1);

        // convert vector expression to diagonal expression
	Matrix<double, 3, 3> M3({1, 2, 3, 4, 5, 6, 7, 8, 9});
        auto e = (w + M3.diagonal()).as_diagonal();
	EXPECT_EQ(e.rows(), 3);
	EXPECT_EQ(e.cols(), 3);
	EXPECT_EQ(e.size(), 9);
        Matrix<double, 3, 3> M4({4, 0, 0, 0, 8, 0, 0, 0, 12});
        EXPECT_EQ(e, M4);

	// diagonal expression API
        EXPECT_EQ(e.determinant(), 384);
        Matrix<double, 3, 3> invM4({1. / 4, 0, 0, 0, 1. / 8, 0, 0, 0, 1. / 12});
        EXPECT_EQ(e.inverse(), invM4);
        Vector<double, 3> x1 = e.solve(Vector<double, 3>({2, 2, 2}));
        Vector<double, 3> x2 = Vector<double, 3>({1. / 2, 1. / 4, 1. / 6});
	EXPECT_EQ(x1, x2);
    }
}
