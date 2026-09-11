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

// constructs fixed and dynamic matrices, checks initialization and verifies independent copies
TEST(linear_algebra, matrix) {
    // static-sized
    {
        // construct empty
        constexpr Matrix<double, 2, 2> M0;
        // fixed storage exposes its two compile-time rows
        static_assert(M0.rows() == 2);
        // fixed storage exposes its two compile-time columns
        static_assert(M0.cols() == 2);
        // a two-by-two matrix contains four coefficients
        static_assert(M0.size() == 4);

        // construct from C-array
        constexpr Matrix<double, 2, 3> M1({1, 2, 3, 4, 5, 6});
        // the C-array constructor retains the declared two rows
        static_assert(M1.rows() == 2);
        // the C-array constructor retains the declared three columns
        static_assert(M1.cols() == 3);
        // the input array fills all six matrix coefficients
        static_assert(M1.size() == 6);
        // successive C-array values occupy matrix coordinates in row order
        static_assert([M1]() {
            bool v = true;
            int k = 1;
            for (int i = 0; i < M1.rows(); ++i) {
                for (int j = 0; j < M1.cols(); ++j) {
                    if (M1(i, j) != k++) { v = false; }
                }
            }
            return v;
        }());

        // construct empty and assign
        Matrix<double, 2, 3> M2;
        M2 = M1;
        // copy construction preserves the source row count
        EXPECT_EQ(M2.rows(), M1.rows());
        // copy construction preserves the source column count
        EXPECT_EQ(M2.cols(), M1.cols());
        // copy construction preserves the source coefficient count
        EXPECT_EQ(M2.size(), M1.size());
        // copy construction preserves every source coefficient
        EXPECT_EQ(M2, M1);

        // copy-construct
        Matrix<double, 2, 3> M3 = M1;
        // copy assignment preserves the source row count
        EXPECT_EQ(M3.rows(), M1.rows());
        // copy assignment preserves the source column count
        EXPECT_EQ(M3.cols(), M1.cols());
        // copy assignment preserves the source coefficient count
        EXPECT_EQ(M3.size(), M1.size());
        // copy assignment preserves every source coefficient
        EXPECT_EQ(M3, M1);

        // value-construct
        constexpr Matrix<int, 3, 3> M4(6);
        // scalar-filled fixed storage retains three rows
        static_assert(M4.rows() == 3);
        // scalar-filled fixed storage retains three columns
        static_assert(M4.cols() == 3);
        // scalar-filled fixed storage contains nine coefficients
        static_assert(M4.size() == 9);
        // the scalar constructor fills every coefficient with six
        static_assert([M4]() {
            bool v = true;
            for (int i = 0; i < M4.rows(); ++i) {
                for (int j = 0; j < M4.cols(); ++j) { v &= M4(i, j) == 6 ? true : false; }
            }
            return v;
        }());

        // static-construct
        constexpr Matrix<int, 4, 4> M6 = Matrix<int, 4, 4>::Zero();
        // the zero factory produces only zero coefficients at compile time
        static_assert(M6 == Matrix<int, 4, 4>::Zero());
        constexpr Matrix<int, 4, 4> M7 = Matrix<int, 4, 4>::Ones();
        // the ones factory produces only unit coefficients at compile time
        static_assert(M7 == Matrix<int, 4, 4>::Ones());

        // const-access
        // const coefficient access returns the first C-array value
        EXPECT_EQ(M1(0, 0), 1);
        // const coefficient access is also valid during constant evaluation
        static_assert(M1(0, 0) == 1);
        // non-const access
        M2(1, 1) = 10;
        // writable coefficient access stores the replacement value ten
        EXPECT_EQ(M2(1, 1), 10);
    }

    // dynamic-sized
    {
        // construct empty
        Matrix<double, Dynamic, Dynamic> M0;
        // a default dynamic matrix has no rows
        EXPECT_EQ(M0.rows(), 0);
        // a default dynamic matrix has no columns
        EXPECT_EQ(M0.cols(), 0);
        // a default dynamic matrix allocates no coefficients
        EXPECT_EQ(M0.size(), 0);

        // construct empty and resize
        Matrix<double, Dynamic, Dynamic> M1;
        M1.resize(3, 3);   // allocate memory
        // the runtime constructor uses the requested three rows
        EXPECT_EQ(M1.rows(), 3);
        // the runtime constructor uses the requested three columns
        EXPECT_EQ(M1.cols(), 3);
        // the runtime shape allocates nine coefficients
        EXPECT_EQ(M1.size(), 9);
        for (int i = 0; i < M1.rows(); ++i) {
            // dimension-only construction value-initializes each double to zero
            for (int j = 0; j < M1.cols(); ++j) { EXPECT_EQ(M1(i, j), 0); }
        }

        // construct with sizes
        Matrix<double, Dynamic, Dynamic> M2(5, 5);
        // resizing changes the row count to five
        EXPECT_EQ(M2.rows(), 5);
        // resizing changes the column count to five
        EXPECT_EQ(M2.cols(), 5);
        // resizing allocates twenty-five coefficients
        EXPECT_EQ(M2.size(), 25);
        for (int i = 0; i < M2.rows(); ++i) {
            // all coefficients are zero after growing the initially empty matrix
            for (int j = 0; j < M2.cols(); ++j) { EXPECT_EQ(M2(i, j), 0); }
        }

        // value-construct
        Matrix<double, Dynamic, Dynamic> M3(5, 5, 1.0);
        // scalar-filled dynamic storage uses five rows
        EXPECT_EQ(M3.rows(), 5);
        // scalar-filled dynamic storage uses five columns
        EXPECT_EQ(M3.cols(), 5);
        // scalar-filled dynamic storage contains twenty-five coefficients
        EXPECT_EQ(M3.size(), 25);
        for (int i = 0; i < M3.rows(); ++i) {
            // the dynamic scalar constructor fills every coefficient with one
            for (int j = 0; j < M3.cols(); ++j) { EXPECT_EQ(M3(i, j), 1.0); }
        }

        // static-construct
        Matrix<int, Dynamic, Dynamic> M4 = Matrix<int, Dynamic, Dynamic>::Zero(10, 10);
        for (int i = 0; i < M4.rows(); ++i) {
            // the dynamic zero factory initializes every coordinate to zero
            for (int j = 0; j < M4.cols(); ++j) { EXPECT_EQ(M4(i, j), 0); }
        }
        Matrix<int, Dynamic, Dynamic> M5 = Matrix<int, Dynamic, Dynamic>::Ones(10, 10);
        for (int i = 0; i < M5.rows(); ++i) {
            // the dynamic ones factory initializes every coordinate to one
            for (int j = 0; j < M5.cols(); ++j) { EXPECT_EQ(M5(i, j), 1); }
        }

        // assignement
        M1 = M2;   // dynamic-sized to dynamic-sized
        // assignment resizes the destination to the source row count
        EXPECT_EQ(M1.rows(), M2.rows());
        // assignment resizes the destination to the source column count
        EXPECT_EQ(M1.cols(), M2.cols());
        // assignment resizes the destination to the source coefficient count
        EXPECT_EQ(M1.size(), M2.size());
        // assignment copies all coefficients after resizing
        EXPECT_EQ(M1, M2);

        constexpr Matrix<double, 2, 3> M6({1, 2, 3, 4, 5, 6});
        M1 = M6;   // static-sized to dynamic-sized
        // copy construction preserves dynamic row count
        EXPECT_EQ(M1.rows(), M6.rows());
        // copy construction preserves dynamic column count
        EXPECT_EQ(M1.cols(), M6.cols());
        // copy construction preserves dynamic coefficient count
        EXPECT_EQ(M1.size(), M6.size());
        // the dynamic copy equals the complete source matrix
        EXPECT_EQ(M1, M6);

        // const access
        // coefficient access reads the previously assigned value five
        EXPECT_EQ(M1(1, 1), 5);
        // non-const access
        M1(1, 1) = 4;
        // coefficient assignment replaces five with four
        EXPECT_EQ(M1(1, 1), 4);
    }
}

// checks vector initialization, coordinate constructors, copies and physical-order iteration
TEST(linear_algebra, vector) {
    // static-sized
    {
        // construct empty
        constexpr Vector<double, 3> v0;
        // fixed vector storage has three rows
        static_assert(v0.rows() == 3);
        // fixed vectors retain a single column
        static_assert(v0.cols() == 1);
        // fixed vector storage contains three coefficients
        static_assert(v0.size() == 3);

        // construct from C-array
        constexpr Vector<double, 6> v1({1, 2, 3, 4, 5, 6});
        // the C-array vector retains its six rows
        static_assert(v1.rows() == 6);
        // the C-array vector remains column-shaped
        static_assert(v1.cols() == 1);
        // all six C-array entries fit the fixed vector
        static_assert(v1.size() == 6);
        // vector indexing preserves the six input values in order
        static_assert([v1]() {
            bool v = true;
            for (int i = 0; i < v1.size(); ++i) v &= v1[i] == (i + 1) ? true : false;
            return v;
        }());

        // point constructors
        constexpr Vector<double, 1> p1(1);
        // the one-coordinate constructor stores its sole component
        static_assert(p1[0] == 1);
        constexpr Vector<double, 2> p2(1, 2);
        // the two-coordinate constructor preserves both components
        static_assert(p2[0] == 1 && p2[1] == 2);
        constexpr Vector<double, 3> p3(1, 2, 3);
        // the three-coordinate constructor preserves all three components
        static_assert(p3[0] == 1 && p3[1] == 2 && p3[2] == 3);

        // construct empty and assign
        Vector<double, 6> v2;
        v2 = v1;
        // vector assignment preserves the source length
        EXPECT_EQ(v2.rows(), v1.rows());
        // vector assignment preserves the coefficient count
        EXPECT_EQ(v2.size(), v1.size());
        // vector assignment copies every coefficient
        EXPECT_EQ(v2, v1);

        // copy-construct
        Vector<double, 6> v3 = v2;
        // vector copy construction preserves its length
        EXPECT_EQ(v3.rows(), v2.rows());
        // vector copy construction preserves its coefficient count
        EXPECT_EQ(v3.size(), v2.size());
        // vector copy construction preserves every coefficient
        EXPECT_EQ(v3, v2);

        // value-initialize
        Vector<double, 6> v4(2.0);
        // the fixed scalar constructor fills every component with two
        for (int i = 0; i < v4.size(); ++i) { EXPECT_EQ(v4[i], 2.0); }

        // const access
        // const vector access returns the first input value
        EXPECT_EQ(v1[0], 1);
        // const vector access remains usable in constant evaluation
        static_assert(v1[0] == 1);
        // non-const access
        v3[0] = 10;
        // mutable vector indexing writes the replacement value ten
        EXPECT_EQ(v3[0], 10);

        // range-for
        int i = 0;
        // iteration follows the same coefficient order as vector indexing
        for (const auto& value : v3) { EXPECT_EQ(value, v3[i++]); }
        for (auto& value : v3) { value = 5; }
        // mutable iteration writes five through every coefficient reference
        for (const auto& value : v3) { EXPECT_EQ(value, 5); }
        // the standard accumulator can traverse the fixed vector at compile time
        static_assert(std::accumulate(v1.begin(), v1.end(), 0) == 21);   // constexpr begin/end
    }

    // dynamic-sized
    {
        // construct empty
        Vector<double, Dynamic> v0;
        // a default dynamic vector has zero rows
        EXPECT_EQ(v0.rows(), 0);
        // an empty dynamic vector still has one column
        EXPECT_EQ(v0.cols(), 1);
        // a default dynamic vector contains no coefficients
        EXPECT_EQ(v0.size(), 0);

        // construct empty and resize
        Vector<double, Dynamic> v1;
        v1.resize(10);   // allocate memory
        // the runtime vector constructor creates ten rows
        EXPECT_EQ(v1.rows(), 10);
        // the runtime vector constructor creates ten coefficients
        EXPECT_EQ(v1.size(), 10);
        // dimension-only vector construction initializes every component to zero
        for (int i = 0; i < v1.size(); ++i) { EXPECT_EQ(v1[i], 0); }

        // construct with sizes
        Vector<double, Dynamic> v2(5);
        // resizing creates the requested five rows
        EXPECT_EQ(v2.rows(), 5);
        // resizing creates five vector coefficients
        EXPECT_EQ(v2.size(), 5);
        // the resized initially empty vector has zero-valued components
        for (int i = 0; i < v2.size(); ++i) { EXPECT_EQ(v2[i], 0); }

        // value-construct
        Vector<double, Dynamic> v3(10, 5.0);
        // the filled vector constructor creates ten rows
        EXPECT_EQ(v3.rows(), 10);
        // the filled vector constructor allocates ten components
        EXPECT_EQ(v3.size(), 10);
        // the supplied scalar fills each component with five
        for (int i = 0; i < v3.size(); ++i) { EXPECT_EQ(v3[i], 5.0); }

        // static-construct
        Vector<int, Dynamic> v5 = Vector<int, Dynamic>::Zero(5);
        // the dynamic vector zero factory clears every component
        for (int i = 0; i < v5.size(); ++i) { EXPECT_EQ(v5[i], 0); }
        Vector<int, Dynamic> v6 = Vector<int, Dynamic>::Ones(5);
        // the dynamic vector ones factory sets every component to one
        for (int i = 0; i < v6.size(); ++i) { EXPECT_EQ(v6[i], 1); }
        Vector<double, Dynamic> v7 = Vector<double, Dynamic>::LinSpaced(10, 0, 1);
        // the equally spaced factory includes the requested evenly spaced values from zero to one
        for (double i = 0; i < v7.size(); ++i) { EXPECT_EQ(v7[i], i * (1. / 9)); }
        // const access
        // const dynamic indexing reads the initial zero
        EXPECT_EQ(v1[0], 0);
        // non-const access
        v3[0] = 10;
        // mutable dynamic indexing stores the replacement value ten
        EXPECT_EQ(v3[0], 10);

        // range-for
        // const iteration visits the vector's zero-initialized components
        for (const auto& value : v5) { EXPECT_EQ(value, 0); }
        for (auto& value : v5) { value = 5; }
        // mutable iteration stores five in each dynamic component
        for (const auto& value : v5) { EXPECT_EQ(value, 5); }
    }
}
