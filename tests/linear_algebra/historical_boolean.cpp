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

// verifies static sized matrix through the public algebra API
TEST(HistoricalBoolean, static_sized_matrix) {
    // build a static-sized binary matrix
    BinaryMatrix<5, 3> m;
    // check dimensionalities
    // checks m.rows() == 5
    EXPECT_TRUE(m.rows() == 5);
    // checks m.cols() == 3
    EXPECT_TRUE(m.cols() == 3);
    // checks m.size() == 15
    EXPECT_TRUE(m.size() == 15);
    // check all is set to zero
    for (int i = 0; i < m.rows(); ++i) {
        // checks m(i, j) == false
        for (int j = 0; j < m.cols(); ++j) { EXPECT_TRUE(m(i, j) == false); }
    }
    // set a coefficient to true and check that it is the only one set to true
    m.set(3, 1);
    for (int i = 0; i < m.rows(); ++i) {
        for (int j = 0; j < m.cols(); ++j) {
            if (i == 3 && j == 1) {
                // checks m(i, j) == true
                EXPECT_TRUE(m(i, j) == true);
            } else {
                // checks m(i, j) == false
                EXPECT_TRUE(m(i, j) == false);
            }
        }
    }
    // set back to false, and check all is false
    m.clear(3, 1);
    for (int i = 0; i < m.rows(); ++i) {
        // checks m(i, j) == false
        for (int j = 0; j < m.cols(); ++j) { EXPECT_TRUE(m(i, j) == false); }
    }
}

// verifies dynamic sized matrix through the public algebra API
TEST(HistoricalBoolean, dynamic_sized_matrix) {
    // build a dynamic-sized binary matrix, large enought to span multiple bitpacks
    BinaryMatrix<Dynamic> m(5, 100);
    // check dimensionalities
    // checks m.rows() == 5
    EXPECT_TRUE(m.rows() == 5);
    // checks m.cols() == 100
    EXPECT_TRUE(m.cols() == 100);
    // checks m.size() == 500
    EXPECT_TRUE(m.size() == 500);
    // check all is set to zero
    for (int i = 0; i < m.rows(); ++i) {
        // checks m(i, j) == false
        for (int j = 0; j < m.cols(); ++j) { EXPECT_TRUE(m(i, j) == false); }
    }
    // set a coefficient to true and check that it is the only one set to true
    m.set(3, 47);
    for (int i = 0; i < m.rows(); ++i) {
        for (int j = 0; j < m.cols(); ++j) {
            if (i == 3 && j == 47) {
                // checks m(i, j) == true
                EXPECT_TRUE(m(i, j) == true);
            } else {
                // checks m(i, j) == false
                EXPECT_TRUE(m(i, j) == false);
            }
        }
    }
    // set back to false, and check all is false
    m.clear(3, 47);
    for (int i = 0; i < m.rows(); ++i) {
        // checks m(i, j) == false
        for (int j = 0; j < m.cols(); ++j) { EXPECT_TRUE(m(i, j) == false); }
    }
    // resize matrix and check dimensionalities
    m.set(0, 0);
    m.resize(20, 20);
    // checks m.rows() == 20
    EXPECT_TRUE(m.rows() == 20);
    // checks m.cols() == 20
    EXPECT_TRUE(m.cols() == 20);
    // checks m.size() == 400
    EXPECT_TRUE(m.size() == 400);
    // resizing a matrix should destruct previous memory and set all to 0
    for (int i = 0; i < m.rows(); ++i) {
        // checks m(i, j) == false
        for (int j = 0; j < m.cols(); ++j) { EXPECT_TRUE(m(i, j) == false); }
    }
}

// verifies binary vector through the public algebra API
TEST(HistoricalBoolean, binary_vector) {
    // build a static sized binary vector
    BinaryVector<5> v;
    // check dimensionalities
    // checks v.rows() == 5
    EXPECT_TRUE(v.rows() == 5);
    // checks v.cols() == 1
    EXPECT_TRUE(v.cols() == 1);
    // checks v.size() == 5
    EXPECT_TRUE(v.size() == 5);
    // test vector interface
    v.set(1);
    // checks v[1] == true
    EXPECT_TRUE(v[1] == true);
    // checks v[1] == v(1, 0)
    EXPECT_TRUE(v[1] == v(1, 0));   // matrix-like interface still works
    v.clear(1);
    // checks v[i] == false
    for (int i = 0; i < v.size(); ++i) { EXPECT_TRUE(v[i] == false); }

    // dynamic-sized vector
    BinaryVector<Dynamic> s;
    s.resize(100);
    // check dimensionalities
    // checks s.rows() == 100
    EXPECT_TRUE(s.rows() == 100);
    // checks s.cols() == 1
    EXPECT_TRUE(s.cols() == 1);
    // checks s.size() == 100
    EXPECT_TRUE(s.size() == 100);

    s.set(10);
    s.set(70);
    for (int i = 0; i < s.size(); ++i) {
        if (i == 10 || i == 70) {
            // checks s[i] == true
            EXPECT_TRUE(s[i] == true);
        } else {
            // checks s[i] == false
            EXPECT_TRUE(s[i] == false);
        }
    }
}

// verifies block operations through the public algebra API
TEST(HistoricalBoolean, block_operations) {
    // build a dynamic-sized binary matrix, large enought to span multiple bitpacks
    BinaryMatrix<Dynamic> m(5, 100);
    m.set(3, 40);
    m.set(4, 60);

    // extract a row
    auto r = m.row(3);
    // check dimensionalities
    // checks r.rows() == 1
    EXPECT_TRUE(r.rows() == 1);
    // checks r.cols() == 100
    EXPECT_TRUE(r.cols() == 100);
    for (int i = 0; i < r.size(); ++i) {
        if (i == 40) {
            // checks r(0, i) == true
            EXPECT_TRUE(r(0, i) == true);
        } else {
            // checks r(0, i) == false
            EXPECT_TRUE(r(0, i) == false);
        }
    }
    // assign row to vector
    BinaryVector<Dynamic> v1 = r.reshape(r.size(), 1);
    // checks v1[40] == true
    EXPECT_TRUE(v1[40] == true);

    // extract a column
    auto c = m.col(60);
    // check dimensionalities
    // checks c.rows() == 5
    EXPECT_TRUE(c.rows() == 5);
    // checks c.cols() == 1
    EXPECT_TRUE(c.cols() == 1);
    for (int i = 0; i < c.size(); ++i) {
        if (i == 4) {
            // checks c(i, 0) == true
            EXPECT_TRUE(c(i, 0) == true);
        } else {
            // checks c(i, 0) == false
            EXPECT_TRUE(c(i, 0) == false);
        }
    }
    // assign column to vector
    BinaryVector<Dynamic> v2 = c;
    // checks v2[4] == true
    EXPECT_TRUE(v2[4] == true);

    // extract a generic block
    auto block = m.block(2, 40, 3, 30);
    // assign to binarymatrix
    BinaryMatrix<Dynamic> bm = block;
    // check dimensionalities
    // checks bm.rows() == 3
    EXPECT_TRUE(bm.rows() == 3);
    // checks bm.cols() == 30
    EXPECT_TRUE(bm.cols() == 30);
    // checks bm.size() == 90
    EXPECT_TRUE(bm.size() == 90);
    // checks bm(1, 0) == true && bm(2, 20) == true
    EXPECT_TRUE(bm(1, 0) == true && bm(2, 20) == true);

    // static sized block
    auto static_block = m.block<3, 30>(2, 40);
    // checks block == static_block
    EXPECT_TRUE(block == static_block);
}

// verifies binary expresssions through the public algebra API
TEST(HistoricalBoolean, binary_expresssions) {
    // define two binary matrices (dynamic-sized)
    BinaryMatrix<Dynamic> m1(4, 5);
    m1.set(3, 3);
    BinaryMatrix<Dynamic> m2(4, 5);
    m2.set(2, 2);
    m2.set(3, 3);
    // test some expressions
    // checks (m1 | ~m1) == BinaryMatrix<Dynamic>(BinaryMatrix<Dynamic>::Ones(4, 5))
    EXPECT_TRUE((m1 | ~m1) == BinaryMatrix<Dynamic>(BinaryMatrix<Dynamic>::Ones(4, 5)));
    // checks (m1 & ~m1) == BinaryMatrix<Dynamic>(4, 5)
    EXPECT_TRUE((m1 & ~m1) == BinaryMatrix<Dynamic>(4, 5));
    auto e1 = m1 | m2;
    // checks e1(3, 3) && e1(2, 2)
    EXPECT_TRUE(e1(3, 3) && e1(2, 2));
    auto e2 = m1 & m2;
    // checks e2(3, 3)
    EXPECT_TRUE(e2(3, 3));
    auto e3 = m1 ^ m2;
    // checks e3(2, 2)
    EXPECT_TRUE(e3(2, 2));
    auto e4 = ((m1 ^ m2) | e2);
    // checks e4 == m2
    EXPECT_TRUE(e4 == m2);

    // block expressions
    // checks e1.row(0) == e2.row(0)
    EXPECT_TRUE(e1.row(0) == e2.row(0));

    BinaryMatrix<Dynamic> I = BinaryMatrix<Dynamic>::Ones(2, 2);
    // checks (m1.block(2, 3, 2, 2) & I) == m1.block(2, 3, 2, 2)
    EXPECT_TRUE((m1.block(2, 3, 2, 2) & I) == m1.block(2, 3, 2, 2));
}

// verifies visitors through the public algebra API
TEST(HistoricalBoolean, visitors) {
    // define a matrix of all ones
    BinaryMatrix<Dynamic> m1 = BinaryMatrix<Dynamic>::Ones(150, 4);
    // all() must return true
    // checks m1.all()
    EXPECT_TRUE(m1.all());
    // checks m1.count() == m1.size()
    EXPECT_TRUE(m1.count() == m1.size());
    // test for zero in different bitpack positions (first, middle, last)
    m1.clear(0, 0);
    // checks m1.all()
    EXPECT_FALSE(m1.all());
    // checks m1.count() == (m1.size() - 1)
    EXPECT_TRUE(m1.count() == (m1.size() - 1));
    m1.set(0, 0);
    m1.clear(100, 2);
    // checks m1.all()
    EXPECT_FALSE(m1.all());
    m1.set(100, 2);
    m1.clear(149, 3);
    // checks m1.all()
    EXPECT_FALSE(m1.all());
    // test with a vector
    BinaryVector<Dynamic> v1 = BinaryVector<Dynamic>::Ones(500);
    // checks v1.all()
    EXPECT_TRUE(v1.all());
    // checks v1.count() == v1.size()
    EXPECT_TRUE(v1.count() == v1.size());
    v1.clear(0, 0);
    // checks v1.all()
    EXPECT_FALSE(v1.all());
    // checks v1.count() == (v1.size() - 1)
    EXPECT_TRUE(v1.count() == (v1.size() - 1));
    v1.clear(200, 0);
    // checks v1.count() == (v1.size() - 2)
    EXPECT_TRUE(v1.count() == (v1.size() - 2));

    BinaryVector<Dynamic> v2(500);
    // v2 is a vector of 0, any() must return false
    // checks v2.any()
    EXPECT_FALSE(v2.any());
    // checks v2.count() == 0
    EXPECT_TRUE(v2.count() == 0);
    // test for one in different bitpack posistions (first, middle, last)
    v2.set(0);
    // checks v2.any()
    EXPECT_TRUE(v2.any());
    v2.clear(0);
    v2.set(300);
    // checks v2.any()
    EXPECT_TRUE(v2.any());
    v2.clear(300);
    v2.set(499);
    // checks v2.any()
    EXPECT_TRUE(v2.any());

    // static sized
    BinaryVector<3> v3;
    for (int i = 0; i < 3; ++i) v3.set(i);
    // checks v3.all()
    EXPECT_TRUE(v3.all());
    v3.clear(1);
    // checks v3.all()
    EXPECT_FALSE(v3.all());
    for (int i = 0; i < 3; ++i) v3.clear(i);
    // checks v3.any()
    EXPECT_FALSE(v3.any());
    // dynamic sized (one bitpack only)
    BinaryVector<Dynamic> v4(3);
    for (int i = 0; i < 3; ++i) v4.set(i);
    // checks v4.all()
    EXPECT_TRUE(v4.all());
    for (int i = 0; i < 3; ++i) v4.clear(i);
    // checks v4.any()
    EXPECT_FALSE(v4.any());
}

// verifies block repeat through the public algebra API
TEST(HistoricalBoolean, block_repeat) {
    BinaryMatrix<Dynamic> m1 = BinaryMatrix<Dynamic>::Ones(3, 4);
    m1.row(1).clear();
    m1.set(1, 1);
    BinaryMatrix<Dynamic> m2 = m1.repeat(2, 4);
    // checks m2.rows() == 6
    EXPECT_TRUE(m2.rows() == 6);
    // checks m2.cols() == 16
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
    // checks m2 == res
    EXPECT_TRUE(m2 == res);

    BinaryVector<Dynamic> v1(10);
    v1.set(4);
    BinaryMatrix<Dynamic> res2(10, 10);
    res2.row(4).set();
    // checks v1.repeat(1, 10) == res2
    EXPECT_TRUE(v1.repeat(1, 10) == res2);
}

// verifies reshaped through the public algebra API
TEST(HistoricalBoolean, reshaped) {
    BinaryMatrix<Dynamic> m1(5, 20);
    m1.set(3, 15);
    m1.set(4, 19);
    BinaryMatrix<Dynamic> m2 = m1.reshape(4, 25);
    // checks m2.rows() == 4
    EXPECT_TRUE(m2.rows() == 4);
    // checks m2.cols() == 25
    EXPECT_TRUE(m2.cols() == 25);
    // checks m2.count() == 2
    EXPECT_TRUE(m2.count() == 2);
    // checks m2.size() == m1.size()
    EXPECT_TRUE(m2.size() == m1.size());
    // check correctly reshaped
    for (int i = 0; i < m2.rows(); ++i) {
        for (int j = 0; j < m2.cols(); ++j) {
            // checks m2(i, j) == true
            if (i == 3 && j == 0) { EXPECT_TRUE(m2(i, j) == true); }
            // checks m2(i, j) == true
            if (i == 3 && j == 24) { EXPECT_TRUE(m2(i, j) == true); }
        }
    }
    // reshape and then repeat
    BinaryMatrix<Dynamic> m3 = m1.reshape(100, 1).repeat(1, 10);
    BinaryMatrix<Dynamic> m4(100, 10);
    m4.row(75).set();
    m4.row(99).set();
    // checks m3 == m4
    EXPECT_TRUE(m3 == m4);
}
