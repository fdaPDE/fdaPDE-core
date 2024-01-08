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
#include <fdaPDE/utils.h>
#include <gtest/gtest.h>   // testing framework
using fdapde::Dynamic;
using fdapde::core::BinaryMatrix;
using fdapde::core::BinaryVector;

#include <bitset>

// test definition of spline basis
TEST(binary_matrix_test, static_sized_matrix) {
    // build a static-sized binary matrix
    BinaryMatrix<5, 3> m;
    // check dimensionalities
    EXPECT_TRUE(m.rows() == 5);
    EXPECT_TRUE(m.cols() == 3);
    EXPECT_TRUE(m.size() == 15);
    // check all is set to zero
    for (std::size_t i = 0; i < m.rows(); ++i) {
        for (std::size_t j = 0; j < m.cols(); ++j) { EXPECT_TRUE(m(i, j) == false); }
    }
    // set a coefficient to true and check that it is the only one set to true
    m.set(3, 1);
    for (std::size_t i = 0; i < m.rows(); ++i) {
        for (std::size_t j = 0; j < m.cols(); ++j) {
            if (i == 3 && j == 1) {
                EXPECT_TRUE(m(i, j) == true);
            } else {
                EXPECT_TRUE(m(i, j) == false);
            }
        }
    }
    // set back to false, and check all is false
    m.clear(3, 1);
    for (std::size_t i = 0; i < m.rows(); ++i) {
        for (std::size_t j = 0; j < m.cols(); ++j) { EXPECT_TRUE(m(i, j) == false); }
    }
}

TEST(binary_matrix_test, dynamic_sized_matrix) {
    // build a dynamic-sized binary matrix, large enought to span multiple bitpacks
    BinaryMatrix<Dynamic> m(5, 100);
    // check dimensionalities
    EXPECT_TRUE(m.rows() == 5);
    EXPECT_TRUE(m.cols() == 100);
    EXPECT_TRUE(m.size() == 500);
    // check all is set to zero
    for (std::size_t i = 0; i < m.rows(); ++i) {
        for (std::size_t j = 0; j < m.cols(); ++j) { EXPECT_TRUE(m(i, j) == false); }
    }
    // set a coefficient to true and check that it is the only one set to true
    m.set(3, 47);
    for (std::size_t i = 0; i < m.rows(); ++i) {
        for (std::size_t j = 0; j < m.cols(); ++j) {
            if (i == 3 && j == 47) {
                EXPECT_TRUE(m(i, j) == true);
            } else {
                EXPECT_TRUE(m(i, j) == false);
            }
        }
    }
    // set back to false, and check all is false
    m.clear(3, 47);
    for (std::size_t i = 0; i < m.rows(); ++i) {
        for (std::size_t j = 0; j < m.cols(); ++j) { EXPECT_TRUE(m(i, j) == false); }
    }
    // resize matrix and check dimensionalities
    m.set(0, 0);
    m.resize(20, 20);
    EXPECT_TRUE(m.rows() == 20);
    EXPECT_TRUE(m.cols() == 20);
    EXPECT_TRUE(m.size() == 400);
    // resizing a matrix should destruct previous memory and set all to 0
    for (std::size_t i = 0; i < m.rows(); ++i) {
        for (std::size_t j = 0; j < m.cols(); ++j) { EXPECT_TRUE(m(i, j) == false); }
    }
}

TEST(binary_matrix_test, binary_vector) {
    // build a static sized binary vector
    BinaryVector<5> v;
    // check dimensionalities
    EXPECT_TRUE(v.rows() == 5);
    EXPECT_TRUE(v.cols() == 1);
    EXPECT_TRUE(v.size() == 5);
    // test vector interface
    v.set(1);
    EXPECT_TRUE(v[1] == true);
    EXPECT_TRUE(v[1] == v(1, 0));   // matrix-like interface still works
    v.clear(1);
    for (std::size_t i = 0; i < v.size(); ++i) { EXPECT_TRUE(v[i] == false); }

    // dynamic-sized vector
    BinaryVector<Dynamic> s;
    s.resize(100);
    // check dimensionalities
    EXPECT_TRUE(s.rows() == 100);
    EXPECT_TRUE(s.cols() == 1);
    EXPECT_TRUE(s.size() == 100);

    s.set(10);
    s.set(70);
    for (std::size_t i = 0; i < s.size(); ++i) {
        if (i == 10 || i == 70) {
            EXPECT_TRUE(s[i] == true);
        } else {
            EXPECT_TRUE(s[i] == false);
        }
    }
}

TEST(binary_matrix_test, block_operations) {
    // build a dynamic-sized binary matrix, large enought to span multiple bitpacks
    BinaryMatrix<Dynamic> m(5, 100);
    m.set(3, 40);
    m.set(4, 60);

    // extract a row
    auto r = m.row(3);
    // check dimensionalities
    EXPECT_TRUE(r.rows() == 1);
    EXPECT_TRUE(r.cols() == 100);
    for (std::size_t i = 0; i < r.size(); ++i) {
        if (i == 40) {
            EXPECT_TRUE(r(0, i) == true);
        } else {
            EXPECT_TRUE(r(0, i) == false);
        }
    }
    // assign row to vector
    BinaryVector<Dynamic> v1 = r;
    EXPECT_TRUE(v1[40] == true);

    // extract a column
    auto c = m.col(60);
    // check dimensionalities
    EXPECT_TRUE(c.rows() == 5);
    EXPECT_TRUE(c.cols() == 1);
    for (std::size_t i = 0; i < c.size(); ++i) {
        if (i == 4) {
            EXPECT_TRUE(c(i, 0) == true);
        } else {
            EXPECT_TRUE(c(i, 0) == false);
        }
    }
    // assign column to vector
    BinaryVector<Dynamic> v2 = c;
    EXPECT_TRUE(v2[4] == true);

    // extract a generic block
    auto block = m.block(2, 40, 3, 30);
    // assign to binarymatrix
    BinaryMatrix<Dynamic> bm = block;
    // check dimensionalities
    EXPECT_TRUE(bm.rows() == 3);
    EXPECT_TRUE(bm.cols() == 30);
    EXPECT_TRUE(bm.size() == 90);
    EXPECT_TRUE(bm(1, 0) == true && bm(2, 20) == true);

    // static sized block
    auto static_block = m.block<3, 30>(2, 40);
    EXPECT_TRUE(block == static_block);
}

TEST(binary_matrix_test, binary_expresssions) {
    // define two binary matrices (dynamic-sized)
    BinaryMatrix<Dynamic> m1(4, 5);
    m1.set(3, 3);
    BinaryMatrix<Dynamic> m2(4, 5);
    m2.set(2, 2);
    m2.set(3, 3);
    // test some expressions
    EXPECT_TRUE((m1 | ~m1) == BinaryMatrix<Dynamic>::Ones(4, 5));
    EXPECT_TRUE((m1 & ~m1) == BinaryMatrix<Dynamic>(4, 5));
    auto e1 = m1 | m2;
    EXPECT_TRUE(e1(3, 3) && e1(2, 2));
    auto e2 = m1 & m2;
    EXPECT_TRUE(e2(3, 3));
    auto e3 = m1 ^ m2;
    EXPECT_TRUE(e3(2, 2));
    auto e4 = ((m1 ^ m2) | e2);
    EXPECT_TRUE(e4 == m2);

    // block expressions
    EXPECT_TRUE(e1.row(0) == e2.row(0));

    BinaryMatrix<Dynamic> I = BinaryMatrix<Dynamic>::Ones(2, 2);
    EXPECT_TRUE((m1.block(2, 3, 2, 2) & I) == m1.block(2, 3, 2, 2));
}

TEST(binary_matrix_test, visitors) {
    // define a matrix of all ones
    BinaryMatrix<Dynamic> m1 = BinaryMatrix<Dynamic>::Ones(150, 4);
    // all() must return true
    EXPECT_TRUE(m1.all());
    EXPECT_TRUE(m1.count() == m1.size());
    // test for zero in different bitpack positions (first, middle, last)
    m1.clear(0, 0);
    EXPECT_FALSE(m1.all());
    EXPECT_TRUE(m1.count() == (m1.size() - 1));
    m1.set(0, 0);
    m1.clear(100, 2);
    EXPECT_FALSE(m1.all());
    m1.set(100, 2);
    m1.clear(149, 3);
    EXPECT_FALSE(m1.all());
    // test with a vector
    BinaryVector<Dynamic> v1 = BinaryVector<Dynamic>::Ones(500);
    EXPECT_TRUE(v1.all());
    EXPECT_TRUE(v1.count() == v1.size());
    v1.clear(0, 0);
    EXPECT_FALSE(v1.all());
    EXPECT_TRUE(v1.count() == (v1.size() - 1));
    v1.clear(200, 0);
    EXPECT_TRUE(v1.count() == (v1.size() - 2));
    
    BinaryVector<Dynamic> v2(500);
    // v2 is a vector of 0, any() must return false
    EXPECT_FALSE(v2.any());
    EXPECT_TRUE(v2.count() == 0);
    // test for one in different bitpack posistions (first, middle, last)
    v2.set(0);
    EXPECT_TRUE(v2.any());
    v2.clear(0);
    v2.set(300);
    EXPECT_TRUE(v2.any());
    v2.clear(300);
    v2.set(499);
    EXPECT_TRUE(v2.any());
}

TEST(binary_matrix_test, block_repeat) {
    BinaryMatrix<Dynamic> m1 = BinaryMatrix<Dynamic>::Ones(3, 4);
    m1.row(1).clear();
    m1.set(1, 1);
    BinaryMatrix<Dynamic> m2 = m1.blk_repeat(2, 4);
    EXPECT_TRUE(m2.rows() == 6);
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
    EXPECT_TRUE(m2 == res);

    BinaryVector<Dynamic> v1(10);
    v1.set(4);
    BinaryMatrix<Dynamic> res2(10,10);
    res2.row(4).set();
    EXPECT_TRUE(v1.blk_repeat(1, 10) == res2);
}
