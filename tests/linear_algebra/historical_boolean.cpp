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
using fdapde::Dynamic;
template <int Rows, int Cols = Rows> using BinaryMatrix = fdapde::Matrix<bool, Rows, Cols>;
template <int Rows> using BinaryVector = fdapde::Vector<bool, Rows>;

// exercise fixed Boolean dimensions, zero initialization and isolated bit writes
TEST(HistoricalBoolean, static_sized_matrix) {
    // build a static-sized binary matrix
    BinaryMatrix<5, 3> m;
    // check dimensionalities
    // a fixed Boolean matrix retains its five rows
    EXPECT_TRUE(m.rows() == 5);
    // a fixed Boolean matrix retains its three columns
    EXPECT_TRUE(m.cols() == 3);
    // the fixed shape contains fifteen logical bits
    EXPECT_TRUE(m.size() == 15);
    // check all is set to zero
    for (int i = 0; i < m.rows(); ++i) {
        // default construction clears every logical coefficient
        for (int j = 0; j < m.cols(); ++j) { EXPECT_TRUE(m(i, j) == false); }
    }
    // set a coefficient to true and check that it is the only one set to true
    m.set(3, 1);
    for (int i = 0; i < m.rows(); ++i) {
        for (int j = 0; j < m.cols(); ++j) {
            if (i == 3 && j == 1) {
                // setting one coordinate makes that coordinate true
                EXPECT_TRUE(m(i, j) == true);
            } else {
                // setting one coordinate leaves every other coordinate false
                EXPECT_TRUE(m(i, j) == false);
            }
        }
    }
    // set back to false, and check all is false
    m.clear(3, 1);
    for (int i = 0; i < m.rows(); ++i) {
        // clearing the only set coordinate restores an all-false matrix
        for (int j = 0; j < m.cols(); ++j) { EXPECT_TRUE(m(i, j) == false); }
    }
}

// exercise dynamic Boolean allocation, bit writes across words and clearing on resize
TEST(HistoricalBoolean, dynamic_sized_matrix) {
    // build a dynamic-sized binary matrix, large enought to span multiple bitpacks
    BinaryMatrix<Dynamic> m(5, 100);
    // check dimensionalities
    // a dynamic Boolean matrix adopts the requested five rows
    EXPECT_TRUE(m.rows() == 5);
    // a dynamic Boolean matrix adopts the requested hundred columns
    EXPECT_TRUE(m.cols() == 100);
    // the dynamic shape contains five hundred logical bits
    EXPECT_TRUE(m.size() == 500);
    // check all is set to zero
    for (int i = 0; i < m.rows(); ++i) {
        // allocation clears every logical bit across all storage words
        for (int j = 0; j < m.cols(); ++j) { EXPECT_TRUE(m(i, j) == false); }
    }
    // set a coefficient to true and check that it is the only one set to true
    m.set(3, 47);
    for (int i = 0; i < m.rows(); ++i) {
        for (int j = 0; j < m.cols(); ++j) {
            if (i == 3 && j == 47) {
                // the addressed coordinate becomes true after set
                EXPECT_TRUE(m(i, j) == true);
            } else {
                // setting a coordinate in a later storage word leaves all other bits false
                EXPECT_TRUE(m(i, j) == false);
            }
        }
    }
    // set back to false, and check all is false
    m.clear(3, 47);
    for (int i = 0; i < m.rows(); ++i) {
        // clearing the addressed coordinate restores every bit to false
        for (int j = 0; j < m.cols(); ++j) { EXPECT_TRUE(m(i, j) == false); }
    }
    // resize matrix and check dimensionalities
    m.set(0, 0);
    m.resize(20, 20);
    // resize adopts the requested twenty rows
    EXPECT_TRUE(m.rows() == 20);
    // resize adopts the requested twenty columns
    EXPECT_TRUE(m.cols() == 20);
    // the resized shape contains four hundred logical bits
    EXPECT_TRUE(m.size() == 400);
    // resizing a matrix should destruct previous memory and set all to 0
    for (int i = 0; i < m.rows(); ++i) {
        // shape-changing resize clears the previously set bit and all new coefficients
        for (int j = 0; j < m.cols(); ++j) { EXPECT_TRUE(m(i, j) == false); }
    }
}

// exercise fixed and dynamic Boolean vectors through vector and matrix indexing
TEST(HistoricalBoolean, binary_vector) {
    // build a static sized binary vector
    BinaryVector<5> v;
    // check dimensionalities
    // a fixed column vector retains its five rows
    EXPECT_TRUE(v.rows() == 5);
    // a Boolean column vector has one column
    EXPECT_TRUE(v.cols() == 1);
    // the fixed vector exposes five logical bits
    EXPECT_TRUE(v.size() == 5);
    // test vector interface
    v.set(1);
    // vector indexing observes the bit set at index one
    EXPECT_TRUE(v[1] == true);
    // vector and matrix indexing address the same logical bit
    EXPECT_TRUE(v[1] == v(1, 0));   // matrix-like interface still works
    v.clear(1);
    // clearing the only set bit restores every vector coefficient to false
    for (int i = 0; i < v.size(); ++i) { EXPECT_TRUE(v[i] == false); }

    // dynamic-sized vector
    BinaryVector<Dynamic> s;
    s.resize(100);
    // check dimensionalities
    // resizing a dynamic vector adopts the requested length as its row count
    EXPECT_TRUE(s.rows() == 100);
    // resizing a dynamic vector preserves its single column
    EXPECT_TRUE(s.cols() == 1);
    // the resized vector contains one hundred logical bits
    EXPECT_TRUE(s.size() == 100);

    s.set(10);
    s.set(70);
    for (int i = 0; i < s.size(); ++i) {
        if (i == 10 || i == 70) {
            // both explicitly set indices remain true across the storage-word boundary
            EXPECT_TRUE(s[i] == true);
        } else {
            // all indices other than the two explicitly set positions remain false
            EXPECT_TRUE(s[i] == false);
        }
    }
}

// exercise Boolean rows, columns and blocks with owner copies of the selected coefficients
TEST(HistoricalBoolean, block_operations) {
    // build a dynamic-sized binary matrix, large enought to span multiple bitpacks
    BinaryMatrix<Dynamic> m(5, 100);
    m.set(3, 40);
    m.set(4, 60);

    // extract a row
    auto r = m.row(3);
    // check dimensionalities
    // a row view has one row
    EXPECT_TRUE(r.rows() == 1);
    // a row view spans all hundred columns
    EXPECT_TRUE(r.cols() == 100);
    for (int i = 0; i < r.size(); ++i) {
        if (i == 40) {
            // the extracted row retains the source bit at column forty
            EXPECT_TRUE(r(0, i) == true);
        } else {
            // the extracted row has no other set bits
            EXPECT_TRUE(r(0, i) == false);
        }
    }
    // assign row to vector
    BinaryVector<Dynamic> v1 = r.reshape(r.size(), 1);
    // reshaping the row into an owning vector preserves its set bit
    EXPECT_TRUE(v1[40] == true);

    // extract a column
    auto c = m.col(60);
    // check dimensionalities
    // a column view spans all five rows
    EXPECT_TRUE(c.rows() == 5);
    // a column view has one column
    EXPECT_TRUE(c.cols() == 1);
    for (int i = 0; i < c.size(); ++i) {
        if (i == 4) {
            // the extracted column retains the source bit in the final row
            EXPECT_TRUE(c(i, 0) == true);
        } else {
            // the extracted column has no other set bits
            EXPECT_TRUE(c(i, 0) == false);
        }
    }
    // assign column to vector
    BinaryVector<Dynamic> v2 = c;
    // copying the column into a vector preserves the final set bit
    EXPECT_TRUE(v2[4] == true);

    // extract a generic block
    auto block = m.block(2, 40, 3, 30);
    // assign to binarymatrix
    BinaryMatrix<Dynamic> bm = block;
    // check dimensionalities
    // copying a block adopts its three rows
    EXPECT_TRUE(bm.rows() == 3);
    // copying a block adopts its thirty columns
    EXPECT_TRUE(bm.cols() == 30);
    // the copied block contains ninety logical bits
    EXPECT_TRUE(bm.size() == 90);
    // the two source bits move to the expected block-relative coordinates
    EXPECT_TRUE(bm(1, 0) == true && bm(2, 20) == true);

    // static sized block
    auto static_block = m.block<3, 30>(2, 40);
    // static and runtime block extents select the same coefficients
    EXPECT_TRUE(block == static_block);
}

// exercise Boolean identities, lazy bitwise expressions and operations on expression blocks
TEST(HistoricalBoolean, binary_expresssions) {
    // define two binary matrices (dynamic-sized)
    BinaryMatrix<Dynamic> m1(4, 5);
    m1.set(3, 3);
    BinaryMatrix<Dynamic> m2(4, 5);
    m2.set(2, 2);
    m2.set(3, 3);
    // test some expressions
    // a mask OR its complement produces an all-true matrix
    EXPECT_TRUE((m1 | ~m1) == BinaryMatrix<Dynamic>(BinaryMatrix<Dynamic>::Ones(4, 5)));
    // a mask AND its complement produces an all-false matrix
    EXPECT_TRUE((m1 & ~m1) == BinaryMatrix<Dynamic>(4, 5));
    auto e1 = m1 | m2;
    // union retains both the shared bit and the bit present only in the second mask
    EXPECT_TRUE(e1(3, 3) && e1(2, 2));
    auto e2 = m1 & m2;
    // intersection retains the bit shared by both masks
    EXPECT_TRUE(e2(3, 3));
    auto e3 = m1 ^ m2;
    // exclusive OR retains the bit present only in the second mask
    EXPECT_TRUE(e3(2, 2));
    auto e4 = ((m1 ^ m2) | e2);
    // combining exclusive and shared bits reconstructs the second mask
    EXPECT_TRUE(e4 == m2);

    // block expressions
    // union and intersection have the same empty first row
    EXPECT_TRUE(e1.row(0) == e2.row(0));

    BinaryMatrix<Dynamic> I = BinaryMatrix<Dynamic>::Ones(2, 2);
    // boolean AND with an all-true block leaves the selected source block unchanged
    EXPECT_TRUE((m1.block(2, 3, 2, 2) & I) == m1.block(2, 3, 2, 2));
}

// exercise all, any and count at the first, middle and last storage-word positions
TEST(HistoricalBoolean, visitors) {
    // define a matrix of all ones
    BinaryMatrix<Dynamic> m1 = BinaryMatrix<Dynamic>::Ones(150, 4);
    // all() must return true
    // all recognizes a matrix filled with true bits
    EXPECT_TRUE(m1.all());
    // count includes every logical bit of the all-true matrix
    EXPECT_TRUE(m1.count() == m1.size());
    // test for zero in different bitpack positions (first, middle, last)
    m1.clear(0, 0);
    // all detects a cleared bit at the first coordinate
    EXPECT_FALSE(m1.all());
    // count decreases by one after clearing the first coordinate
    EXPECT_TRUE(m1.count() == (m1.size() - 1));
    m1.set(0, 0);
    m1.clear(100, 2);
    // all detects a cleared bit in the middle of the matrix
    EXPECT_FALSE(m1.all());
    m1.set(100, 2);
    m1.clear(149, 3);
    // all detects a cleared bit at the final coordinate
    EXPECT_FALSE(m1.all());
    // test with a vector
    BinaryVector<Dynamic> v1 = BinaryVector<Dynamic>::Ones(500);
    // all recognizes an all-true multiword vector
    EXPECT_TRUE(v1.all());
    // count includes every logical bit of the all-true vector
    EXPECT_TRUE(v1.count() == v1.size());
    v1.clear(0, 0);
    // all detects the cleared first vector bit
    EXPECT_FALSE(v1.all());
    // count decreases by one after clearing the first vector bit
    EXPECT_TRUE(v1.count() == (v1.size() - 1));
    v1.clear(200, 0);
    // count decreases by two after clearing a second vector bit
    EXPECT_TRUE(v1.count() == (v1.size() - 2));

    BinaryVector<Dynamic> v2(500);
    // v2 is a vector of 0, any() must return false
    // any is false for a zero-initialized multiword vector
    EXPECT_FALSE(v2.any());
    // count is zero for a zero-initialized multiword vector
    EXPECT_TRUE(v2.count() == 0);
    // test for one in different bitpack posistions (first, middle, last)
    v2.set(0);
    // any detects a set bit at the beginning of the vector
    EXPECT_TRUE(v2.any());
    v2.clear(0);
    v2.set(300);
    // any detects a set bit in the middle of the vector
    EXPECT_TRUE(v2.any());
    v2.clear(300);
    v2.set(499);
    // any detects a set bit at the end of the vector
    EXPECT_TRUE(v2.any());

    // static sized
    BinaryVector<3> v3;
    for (int i = 0; i < 3; ++i) v3.set(i);
    // all ignores unused padding bits in a three-bit fixed vector
    EXPECT_TRUE(v3.all());
    v3.clear(1);
    // all detects a cleared bit in the three-bit fixed vector
    EXPECT_FALSE(v3.all());
    for (int i = 0; i < 3; ++i) v3.clear(i);
    // any is false after every fixed-vector bit is cleared
    EXPECT_FALSE(v3.any());
    // dynamic sized (one bitpack only)
    BinaryVector<Dynamic> v4(3);
    for (int i = 0; i < 3; ++i) v4.set(i);
    // all ignores unused padding bits in a three-bit dynamic vector
    EXPECT_TRUE(v4.all());
    for (int i = 0; i < 3; ++i) v4.clear(i);
    // any is false after every dynamic-vector bit is cleared
    EXPECT_FALSE(v4.any());
}

// exercise two-dimensional tiling and repeated columns of a Boolean vector
TEST(HistoricalBoolean, block_repeat) {
    BinaryMatrix<Dynamic> m1 = BinaryMatrix<Dynamic>::Ones(3, 4);
    m1.row(1).clear();
    m1.set(1, 1);
    BinaryMatrix<Dynamic> m2 = m1.repeat(2, 4);
    // repeating three rows twice produces six rows
    EXPECT_TRUE(m2.rows() == 6);
    // repeating four columns four times produces sixteen columns
    EXPECT_TRUE(m2.cols() == 16);
    // check equality
    BinaryMatrix<Dynamic> res = BinaryMatrix<Dynamic>::Ones(6, 16);
    res.row(1).clear();
    res.row(4).clear();
    res.set(1, 1);
    res.set(1, 5);
    res.set(1, 9);
    res.set(1, 13);
    res.set(4, 1);
    res.set(4, 5);
    res.set(4, 9);
    res.set(4, 13);
    // the repeated matrix matches the explicitly tiled pattern
    EXPECT_TRUE(m2 == res);

    BinaryVector<Dynamic> v1(10);
    v1.set(4);
    BinaryMatrix<Dynamic> res2(10, 10);
    res2.row(4).set();
    // repeating the vector into ten columns creates a full row at its set index
    EXPECT_TRUE(v1.repeat(1, 10) == res2);
}

// exercise Boolean reshape order and composition of reshape with repeat
TEST(HistoricalBoolean, reshaped) {
    BinaryMatrix<Dynamic> m1(5, 20);
    m1.set(3, 15);
    m1.set(4, 19);
    BinaryMatrix<Dynamic> m2 = m1.reshape(4, 25);
    // reshape adopts the requested four rows
    EXPECT_TRUE(m2.rows() == 4);
    // reshape adopts the requested twenty-five columns
    EXPECT_TRUE(m2.cols() == 25);
    // reshape preserves the two set bits
    EXPECT_TRUE(m2.count() == 2);
    // reshape preserves the total number of logical coefficients
    EXPECT_TRUE(m2.size() == m1.size());
    // check correctly reshaped
    for (int i = 0; i < m2.rows(); ++i) {
        for (int j = 0; j < m2.cols(); ++j) {
            // the source bit at linear index seventy-five maps to row three, column zero
            if (i == 3 && j == 0) { EXPECT_TRUE(m2(i, j) == true); }
            // the final source bit maps to the final reshaped coordinate
            if (i == 3 && j == 24) { EXPECT_TRUE(m2(i, j) == true); }
        }
    }
    // reshape and then repeat
    BinaryMatrix<Dynamic> m3 = m1.reshape(100, 1).repeat(1, 10);
    BinaryMatrix<Dynamic> m4(100, 10);
    m4.row(75).set();
    m4.row(99).set();
    // reshaping to a vector and repeating it reproduces the two explicit set rows
    EXPECT_TRUE(m3 == m4);
}
