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
        constexpr auto e = C.cwise_abs();
        constexpr Matrix<double, 2, 2> r({1, 2, 3, 4});
        return e == r;
    }());
    static_assert([]() {
        constexpr auto e = C.cwise_pow(3);
        constexpr Matrix<double, 2, 2> r({-1, 8, 27, -64});
        return e == r;
    }());
    static_assert([]() {
        constexpr auto e = C.cwise_pow2();
        constexpr Matrix<double, 2, 2> r({1, 4, 9, 16});
        return e == r;
    }());
    static_assert([]() {
        constexpr auto e = C.cwise_abs().cwise_sqrt();
	
        constexpr double r1 = 1;
        constexpr double r2 = 1.414213562373095;
        constexpr double r3 = 1.732050807568877;
        constexpr double r4 = 2;
        constexpr Matrix<double, 2, 2> r({r1, r2, r3, r4});
        return almost_equal(e, r);
    }());
    static_assert([]() {
        constexpr auto e = C.cwise_inv();
        constexpr Matrix<double, 2, 2> r({-1, 1./2, 1./3, -1./4});
        return almost_equal(e, r);
    }());    
    static_assert([]() {
        constexpr auto e = C.cwise_exp();
	
        constexpr double r1 = 0.36787944117144;
        constexpr double r2 = 7.38905609893065;
        constexpr double r3 = 20.0855369231876;
        constexpr double r4 = 0.01831563888873;
        constexpr Matrix<double, 2, 2> r({r1, r2, r3, r4});
        return almost_equal(e, r);
    }());
    static_assert([]() {
        constexpr auto e = C.cwise_abs().cwise_log();

        constexpr double r1 = 0;
        constexpr double r2 = 0.69314718055994;
        constexpr double r3 = 1.09861228866811;
        constexpr double r4 = 1.38629436111989;	
        constexpr Matrix<double, 2, 2> r({r1, r2, r3, r4});
        return almost_equal(e, r);
    }());
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
