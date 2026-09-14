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

#include <fdaPDE/dense_linear_algebra.h>
#include <gtest/gtest.h>   // testing framework
using namespace fdapde;

// exercise constant-evaluated dense arithmetic and mixed scalar promotion
TEST(linear_algebra, arithmetic) {
    // constexpr arithmetic
    constexpr Matrix<double, 2, 2> A({1, 2, 3, 4});
    constexpr Matrix<double, 2, 2> B({1, 2, 3, 4});

    // constant evaluation adds the two leading coefficients
    static_assert((A + B)(0, 0) == 2);
    // constant evaluation subtracts equal leading coefficients to zero
    static_assert((A - B)(0, 0) == 0);
    // constant evaluation uses a row-column dot product for matrix multiplication
    static_assert((A * B)(0, 0) == 7);
    // constant evaluation scales the leading coefficient by two
    static_assert((2.0 * A)(0, 0) == 2);
    // dividing a summed expression by two recovers the original leading coefficient
    static_assert(((A + B) / 2.0)(0, 0) == 1);

    constexpr Matrix<int, 1, 1> C(1);
    constexpr Matrix<double, 1, 1> D(0.5);
    using Mixed = decltype(C + D);
    // mixed integer and double addition promotes its scalar type to double
    static_assert(std::is_same_v<typename Mixed::Scalar, double>);
    // mixed scalar addition preserves the fractional part of the leading coefficient
    static_assert((C + D)(0, 0) == 1.5);
}

// exercise constant-evaluated coefficientwise functions against explicit numeric references
TEST(linear_algebra, cwise) {
    static constexpr Matrix<double, 2, 2> C({-1, 2, 3, -4});   // need static address for constexpr expressions
    // coefficientwise absolute value removes each input sign
    static_assert([]() {
        constexpr auto e = C.cwise().abs();
        constexpr Matrix<double, 2, 2> r({1, 2, 3, 4});
        return e.mwise() == r;
    }());
    // coefficientwise cubing preserves odd-power signs
    static_assert([]() {
        constexpr auto e = C.cwise().pow(3);
        constexpr Matrix<double, 2, 2> r({-1, 8, 27, -64});
        return e.mwise() == r;
    }());
    // coefficientwise squaring produces the four expected nonnegative values
    static_assert([]() {
        constexpr auto e = C.cwise().pow2();
        constexpr Matrix<double, 2, 2> r({1, 4, 9, 16});
        return e.mwise() == r;
    }());
    // square roots of absolute coefficients agree with the explicit reference matrix
    static_assert([]() {
        constexpr auto e = C.cwise().abs().sqrt();

        constexpr double r1 = 1;
        constexpr double r2 = 1.414213562373095;
        constexpr double r3 = 1.732050807568877;
        constexpr double r4 = 2;
        constexpr Matrix<double, 2, 2> r({r1, r2, r3, r4});
        return almost_equal(e.mwise(), r);
    }());
    // square roots retain relative accuracy for coefficients much smaller than one
    static_assert([]() {
        constexpr Matrix<double, 1, 2> values({1.0e-16, 4.0e-16});
        constexpr Matrix<double, 1, 2> roots = values.cwise().sqrt();
        return almost_equal(roots[0] / 1.0e-8, 1.0) && almost_equal(roots[1] / 2.0e-8, 1.0);
    }());
    // coefficientwise inversion agrees with the explicit reciprocal matrix
    static_assert([]() {
        constexpr auto e = C.cwise().inv();
        constexpr Matrix<double, 2, 2> r({-1, 1. / 2, 1. / 3, -1. / 4});
        return almost_equal(e.mwise(), r);
    }());
    // coefficientwise exponential agrees with the four reference exponential values
    static_assert([]() {
        constexpr auto e = C.cwise().exp();

        constexpr double r1 = 0.36787944117144;
        constexpr double r2 = 7.38905609893065;
        constexpr double r3 = 20.0855369231876;
        constexpr double r4 = 0.01831563888873;
        constexpr Matrix<double, 2, 2> r({r1, r2, r3, r4});
        return almost_equal(e.mwise(), r);
    }());
    // logarithms of absolute coefficients agree with the explicit logarithm matrix
    static_assert([]() {
        constexpr auto e = C.cwise().abs().log();

        constexpr double r1 = 0;
        constexpr double r2 = 0.69314718055994;
        constexpr double r3 = 1.09861228866811;
        constexpr double r4 = 1.38629436111989;
        constexpr Matrix<double, 2, 2> r({r1, r2, r3, r4});
        return almost_equal(e.mwise(), r);
    }());

    Matrix<double, 4, 4> AA;
    AA(0, 1) = 4;
    AA(0, 2) = 2;
    AA(1, 1) = 4;

    AA.cwise() += 2;

    Matrix<double, 4, 4> BB;
    BB(0, 1) = AA(0, 1);

    Matrix<double, Dynamic, Dynamic> CC = BB.cwise();

    AA.cwise() *= BB.cwise();
}

// exercise scalar reductions and stable Euclidean norms during constant evaluation
TEST(linear_algebra, redux) {
    static constexpr Matrix<double, 2, 2> C({2, 1.5, 1, 0.2});   // need static address for constexpr expressions
    // scaling by three multiplies the squared norm by nine
    static_assert(almost_equal((3 * C).squared_norm(), 65.61));
    // summing a vector with itself doubles its Euclidean norm
    static_assert(almost_equal((C + C).norm(), 5.4));
    // the infinity norm equals the largest absolute coefficient
    static_assert(almost_equal(C.inf_norm(), 2.0));
    // the sum includes all four coefficients
    static_assert(almost_equal(C.sum(), 4.7));
    // the product includes all four coefficients
    static_assert(almost_equal(C.prod(), 0.6));
    // the mean divides the sum by the four-coefficient count
    static_assert(almost_equal(C.mean(), 4.7 / 4));
    // the maximum reduction returns the largest coefficient
    static_assert(C.max() == 2);
    // the minimum reduction returns the smallest coefficient
    static_assert(C.min() == 0.2);
    // scaled norm evaluation avoids intermediate underflow and overflow, and reports true result overflow
    static_assert([]() {
        constexpr Matrix<double, 1, 2> underflowing_square({3.0e-200, 4.0e-200});
        constexpr Matrix<double, 1, 2> overflowing_square({3.0e200, 4.0e200});
        constexpr Matrix<double, 1, 2> overflow_boundary({1.3313981757491274e308, 1.2079236336552887e308});
        return almost_equal(underflowing_square.norm() / 5.0e-200, 1.0) &&
               almost_equal(overflowing_square.norm() / 5.0e200, 1.0) &&
               overflow_boundary.norm() == std::numeric_limits<double>::infinity();
    }());
}

// exercise rowwise and columnwise reductions, including stable norms at extreme scales
TEST(linear_algebra, vectorwise) {
    static constexpr Matrix<double, 4, 3> A = Matrix<double, 4, 3>::Ones();
    // colwise
    constexpr auto r = A.colwise();
    // columnwise reduction produces a single result row
    static_assert(r.rows() == 1);
    // columnwise reduction produces one result for each source column
    static_assert(r.cols() == A.cols());
    // the columnwise result size equals the source's column count
    static_assert(r.size() == A.cols());
    // reductions
    // each column of four ones sums to four
    static_assert([r]() {
        constexpr auto e = r.sum();
        return e == Matrix<double, 1, 3>({4, 4, 4});
    }());
    // each column of ones has product one
    static_assert([r]() {
        constexpr auto e = r.prod();
        return e == Matrix<double, 1, 3>({1, 1, 1});
    }());
    // each column of ones has mean one
    static_assert([r]() {
        constexpr auto e = r.mean();
        return e == Matrix<double, 1, 3>({1, 1, 1});
    }());
    // each column of four ones has squared norm four
    static_assert([r]() {
        constexpr auto e = r.squared_norm();
        return e == Matrix<double, 1, 3>({4, 4, 4});
    }());
    // each column of four ones has Euclidean norm two
    static_assert([r]() {
        constexpr auto e = r.norm();
        return almost_equal(e, Matrix<double, 1, 3>({2, 2, 2}));
    }());
    // each column of ones has infinity norm one
    static_assert([r]() {
        constexpr auto e = r.inf_norm();
        return e == Matrix<double, 1, 3>({1, 1, 1});
    }());

    // rowwise
    constexpr auto c = A.rowwise();
    // rowwise reduction produces one result for each source row
    static_assert(c.rows() == A.rows());
    // rowwise reduction produces a single result column
    static_assert(c.cols() == 1);
    // the rowwise result size equals the source's row count
    static_assert(c.size() == A.rows());
    // reductions
    // each row of three ones sums to three
    static_assert([c]() {
        constexpr auto e = c.sum();
        return e == Matrix<double, 4, 1>({3, 3, 3, 3});
    }());
    // each row of ones has product one
    static_assert([c]() {
        constexpr auto e = c.prod();
        return e == Matrix<double, 4, 1>({1, 1, 1, 1});
    }());
    // each row of ones has mean one
    static_assert([c]() {
        constexpr auto e = c.mean();
        return e == Matrix<double, 4, 1>({1, 1, 1, 1});
    }());
    // each row of three ones has squared norm three
    static_assert([c]() {
        constexpr auto e = c.squared_norm();
        return e == Matrix<double, 4, 1>({3, 3, 3, 3});
    }());
    // each row of three ones has Euclidean norm sqrt(3)
    static_assert([c]() {
        constexpr auto e = c.norm();
        double s = fdapde::sqrt(3.0);
        return almost_equal(e, Matrix<double, 4, 1>({s, s, s, s}));
    }());
    // rowwise norms retain relative accuracy at both tiny and huge coefficient scales
    static_assert([]() {
        constexpr Matrix<double, 2, 2> values({3.0e-200, 4.0e-200, 3.0e200, 4.0e200});
        constexpr Matrix<double, 2, 1> norms = values.rowwise().norm();
        return almost_equal(norms[0] / 5.0e-200, 1.0) && almost_equal(norms[1] / 5.0e200, 1.0);
    }());
    // each row of ones has infinity norm one
    static_assert([c]() {
        constexpr auto e = c.inf_norm();
        return e == Matrix<double, 4, 1>({1, 1, 1, 1});
    }());
}

// exercise transpose shape and coefficient mapping for fixed and dynamic matrices
TEST(linear_algebra, transpose) {
    // static-sized
    {
        static constexpr Matrix<double, 3, 3> A({1, 2, 3, 4, 5, 6, 7, 8, 9});
        // constant evaluation swaps every row and column in the explicit 3-by-3 reference
        static_assert(A.transpose() == Matrix<double, 3, 3>({1, 4, 7, 2, 5, 8, 3, 6, 9}));

        // test static sized rows/cols for non square matrices
    }

    // dynamic-sized
    {
        Matrix<double, Dynamic, Dynamic> A(10, 8);
        A(2, 3) = 4;
        A(7, 7) = 1;
        A(7, 1) = 3;

        auto At = A.transpose();
        // transposition exchanges the source column count into the result row count
        EXPECT_EQ(At.rows(), A.cols());
        // transposition exchanges the source row count into the result column count
        EXPECT_EQ(At.cols(), A.rows());
        // transposition preserves the total coefficient count
        EXPECT_EQ(At.size(), A.size());
        // the transposed coordinate reads the source coefficient from the exchanged indices
        EXPECT_EQ(At(3, 2), 4);
        // transposition preserves a coefficient on the main diagonal
        EXPECT_EQ(At(7, 7), 1);
        // the transposed off-diagonal coordinate reads the corresponding source entry
        EXPECT_EQ(At(1, 7), 3);
    }
}

// exercise constant-evaluated inverses for matrices of sizes one through three
TEST(linear_algebra, inverse) {
    // static sized
    {
        // 1 x 1 inverse
        constexpr Matrix<double, 1, 1> A1(4.0);
        constexpr auto invA1 = A1.inverse();
        // a singleton inverse equals the reciprocal of its sole coefficient
        static_assert(invA1 == Matrix<double, 1, 1>(1. / 4));
        // 2 x 2 inverse
        constexpr Matrix<double, 2, 2> A2({4.0, 7.0, 2.0, 6.0});
        constexpr auto invA2 = A2.inverse();
        // the computed 2-by-2 inverse multiplies its input to the identity within tolerance
        static_assert(almost_equal(invA2 * A2, Matrix<double, 2, 2>({1, 0, 0, 1})));
        // 3 x 3 inverse
        constexpr Matrix<double, 3, 3> A3({1.0, 2.0, 3.0, 0.0, 1.0, 4.0, 5.0, 6.0, 0.0});
        constexpr auto invA3 = A3.inverse();
        // the computed 3-by-3 inverse multiplies its input to the identity within tolerance
        static_assert(almost_equal(invA3 * A3, Matrix<double, 3, 3>({1, 0, 0, 0, 1, 0, 0, 0, 1})));
    }

    // todo: dynamic case and trigger LU factorization
}
