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

TEST(linear_algebra, arithmetic) {
    // constexpr arithmetic
    constexpr Matrix<double, 2, 2> A({1, 2, 3, 4});
    constexpr Matrix<double, 2, 2> B({1, 2, 3, 4});

    static_assert((A + B)(0, 0) == 2);
    static_assert((A - B)(0, 0) == 0);
    static_assert((A * B)(0, 0) == 7);
    static_assert((2.0 * A)(0, 0) == 2);
    static_assert(((A + B) / 2.0)(0, 0) == 1);
}

TEST(linear_algebra, cwise) {
    static constexpr Matrix<double, 2, 2> C({-1, 2, 3, -4});   // need static address for constexpr expressions
    static_assert([]() {
        constexpr auto e = C.cwise().abs();
        constexpr Matrix<double, 2, 2> r({1, 2, 3, 4});
        return e == r;
    }());
    static_assert([]() {
        constexpr auto e = C.cwise().pow(3);
        constexpr Matrix<double, 2, 2> r({-1, 8, 27, -64});
        return e == r;
    }());
    static_assert([]() {
        constexpr auto e = C.cwise().pow2();
        constexpr Matrix<double, 2, 2> r({1, 4, 9, 16});
        return e == r;
    }());
    static_assert([]() {
        constexpr auto e = C.cwise().abs().sqrt();

        constexpr double r1 = 1;
        constexpr double r2 = 1.414213562373095;
        constexpr double r3 = 1.732050807568877;
        constexpr double r4 = 2;
        constexpr Matrix<double, 2, 2> r({r1, r2, r3, r4});
        return almost_equal(e, r);
    }());
    static_assert([]() {
        constexpr auto e = C.cwise().inv();
        constexpr Matrix<double, 2, 2> r({-1, 1. / 2, 1. / 3, -1. / 4});
        return almost_equal(e, r);
    }());
    static_assert([]() {
        constexpr auto e = C.cwise().exp();

        constexpr double r1 = 0.36787944117144;
        constexpr double r2 = 7.38905609893065;
        constexpr double r3 = 20.0855369231876;
        constexpr double r4 = 0.01831563888873;
        constexpr Matrix<double, 2, 2> r({r1, r2, r3, r4});
        return almost_equal(e, r);
    }());
    static_assert([]() {
        constexpr auto e = C.cwise().abs().log();

        constexpr double r1 = 0;
        constexpr double r2 = 0.69314718055994;
        constexpr double r3 = 1.09861228866811;
        constexpr double r4 = 1.38629436111989;
        constexpr Matrix<double, 2, 2> r({r1, r2, r3, r4});
        return almost_equal(e, r);
    }());


    // Matrix<double, 4, 4> AA;
    // auto ee = AA.cwise() + 5;
    
}

TEST(linear_algebra, redux) {
    static constexpr Matrix<double, 2, 2> C({2, 1.5, 1, 0.2});   // need static address for constexpr expressions
    static_assert(almost_equal((3 * C).squared_norm(), 65.61));
    static_assert(almost_equal((C + C).norm(), 5.4));
    static_assert(almost_equal(C.inf_norm(), 2.0));
    static_assert(almost_equal(C.sum(), 4.7));
    static_assert(almost_equal(C.prod(), 0.6));
    static_assert(almost_equal(C.mean(), 4.7 / 4));
    static_assert(C.max() == 2);
    static_assert(C.min() == 0.2);
}

TEST(linear_algebra, vectorwise) {
    static constexpr Matrix<double, 4, 3> A = Matrix<double, 4, 3>::Ones();
    // colwise
    constexpr auto r = A.colwise();
    static_assert(r.rows() == 1);
    static_assert(r.cols() == A.cols());
    static_assert(r.size() == A.cols());
    // reductions
    static_assert([r]() {
        constexpr auto e = r.sum();
        return e == Matrix<double, 1, 3>({4, 4, 4});
    }());
    static_assert([r]() {
        constexpr auto e = r.prod();
        return e == Matrix<double, 1, 3>({1, 1, 1});
    }());
    static_assert([r]() {
        constexpr auto e = r.mean();
        return e == Matrix<double, 1, 3>({1, 1, 1});
    }());
    static_assert([r]() {
        constexpr auto e = r.squared_norm();
        return e == Matrix<double, 1, 3>({4, 4, 4});
    }());
    static_assert([r]() {
        constexpr auto e = r.norm();
        return e == Matrix<double, 1, 3>({2, 2, 2});
    }());
    static_assert([r]() {
        constexpr auto e = r.inf_norm();
        return e == Matrix<double, 1, 3>({1, 1, 1});
    }());

    // rowwise
    constexpr auto c = A.rowwise();
    static_assert(c.rows() == A.rows());
    static_assert(c.cols() == 1);
    static_assert(c.size() == A.rows());
    // reductions
    static_assert([c]() {
        constexpr auto e = c.sum();
        return e == Matrix<double, 4, 1>({3, 3, 3, 3});
    }());
    static_assert([c]() {
        constexpr auto e = c.prod();
        return e == Matrix<double, 4, 1>({1, 1, 1, 1});
    }());
    static_assert([c]() {
        constexpr auto e = c.mean();
        return e == Matrix<double, 4, 1>({1, 1, 1, 1});
    }());
    static_assert([c]() {
        constexpr auto e = c.squared_norm();
        return e == Matrix<double, 4, 1>({3, 3, 3, 3});
    }());
    static_assert([c]() {
        constexpr auto e = c.norm();
        double s = fdapde::sqrt(3.0);
        return e == Matrix<double, 4, 1>({s, s, s, s});
    }());
    static_assert([c]() {
        constexpr auto e = c.inf_norm();
        return e == Matrix<double, 4, 1>({1, 1, 1, 1});
    }());
}

TEST(linear_algebra, transpose) {
    // static-sized
    {
        static constexpr Matrix<double, 3, 3> A({1, 2, 3, 4, 5, 6, 7, 8, 9});
        static_assert(A.transpose() == Matrix<double, 3, 3>({1, 4, 7, 2, 5, 8, 3, 6, 9}));
    }

    // dynamic-sized
    {
        Matrix<double, Dynamic, Dynamic> A(10, 8);
        A(2, 3) = 4;
        A(7, 7) = 1;
        A(7, 1) = 3;

        auto At = A.transpose();
        EXPECT_EQ(At.rows(), A.cols());
        EXPECT_EQ(At.cols(), A.rows());
        EXPECT_EQ(At.size(), A.size());
        EXPECT_EQ(At(3, 2), 4);
        EXPECT_EQ(At(7, 7), 1);
        EXPECT_EQ(At(1, 7), 3);
    }
}

TEST(linear_algebra, inverse) {
    // static sized
    {
        // 1 x 1 inverse
        constexpr Matrix<double, 1, 1> A1(4.0);
        constexpr auto invA1 = A1.inverse();
        static_assert(invA1 == Matrix<double, 1, 1>(1. / 4));
        // 2 x 2 inverse
        constexpr Matrix<double, 2, 2> A2({4.0, 7.0, 2.0, 6.0});
        constexpr auto invA2 = A2.inverse();
        static_assert(almost_equal(invA2 * A2, Matrix<double, 2, 2>({1, 0, 0, 1})));
        // 3 x 3 inverse
        constexpr Matrix<double, 3, 3> A3({1.0, 2.0, 3.0, 0.0, 1.0, 4.0, 5.0, 6.0, 0.0});
        constexpr auto invA3 = A3.inverse();
        static_assert(almost_equal(invA3 * A3, Matrix<double, 3, 3>({1, 0, 0, 0, 1, 0, 0, 0, 1})));
    }

    // TODO: dynamic case and trigger LU factorization
}
