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

#include <gtest/gtest.h>   // testing framework
#include <cstddef>

#include <fdaPDE/linear_algebra.h>
using fdapde::core::Kronecker;

#include "utils/utils.h"
using fdapde::testing::almost_equal;

TEST(kronecker_product_test, dense_dense_square) {
    // define dense operands (square matrices)
    DMatrix<double> m1;
    m1.resize(2, 2);
    m1 << 1, 2, 3, 4;
    DMatrix<double> n1 = DMatrix<double>::Identity(2, 2);
    // evaluate kronecker product n1 \kron m1
    DMatrix<double> kron1 = Kronecker(n1, m1);
    // expected result
    DMatrix<double> expected1;
    expected1.resize(4, 4);
    expected1 <<
      1, 2, 0, 0,
      3, 4, 0, 0,
      0, 0, 1, 2,
      0, 0, 3, 4;

    // check equality
    for (std::size_t i = 0; i < 4; ++i) {
        for (std::size_t j = 0; j < 4; ++j) { EXPECT_TRUE(almost_equal(kron1(i, j), expected1(i, j))); }
    }
}

TEST(kronecker_product_test, dense_dense_rectangular) {
    // define dense operands
    DMatrix<double> m1;
    m1.resize(2, 3);
    m1 << 1, 2, 0, 4, 0, 1;
    DMatrix<double> n1;
    n1.resize(2, 2);
    n1 << 1, 1, 0, 1;
    // evaluate kronecker product m1 \kron n1
    DMatrix<double> kron1 = Kronecker(m1, n1);
    // expected result
    DMatrix<double> expected1;
    expected1.resize(4, 6);
    expected1 <<
      1, 1, 2, 2, 0, 0,
      0, 1, 0, 2, 0, 0,
      4, 4, 0, 0, 1, 1,
      0, 4, 0, 0, 0, 1;

    // check equality
    for (std::size_t i = 0; i < 4; ++i) {
        for (std::size_t j = 0; j < 6; ++j) { EXPECT_TRUE(almost_equal(kron1(i, j), expected1(i, j))); }
    }
}

TEST(kronecker_product_test, sparse_sparse_square) {
    std::vector<Eigen::Triplet<double>> coeff;
    coeff.push_back(Eigen::Triplet<double>(0, 0, 2.0));
    coeff.push_back(Eigen::Triplet<double>(1, 1, 3.0));
    coeff.push_back(Eigen::Triplet<double>(1, 0, 1.0));
    coeff.push_back(Eigen::Triplet<double>(2, 2, 1.0));
    SpMatrix<double> m1(3, 3);
    m1.setFromTriplets(coeff.begin(), coeff.end());
    m1.makeCompressed();
    SpMatrix<double> n1(3, 3);
    n1.setIdentity();

    // evaluate kronecker product n1 \kron m1
    SpMatrix<double> kron1 = Kronecker(n1, m1);
    // expected result
    DMatrix<double> expected1;
    expected1.resize(9, 9);
    expected1 <<
      2, 0, 0, 0, 0, 0, 0, 0, 0,
      1, 3, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 1, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 2, 0, 0, 0, 0, 0,
      0, 0, 0, 1, 3, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 1, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 2, 0, 0,
      0, 0, 0, 0, 0, 0, 1, 3, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 1;

    // check equality
    for (int k = 0; k < kron1.outerSize(); ++k) {
        for (SpMatrix<double>::InnerIterator it(kron1, k); it; ++it) {
            EXPECT_TRUE(almost_equal(expected1(it.row(), it.col()), it.value()));
        }
    }
}

TEST(kronecker_product_test, sparse_sparse_rectangular) {
    std::vector<Eigen::Triplet<double>> coeff;
    coeff.push_back(Eigen::Triplet<double>(0, 0, 2.0));
    coeff.push_back(Eigen::Triplet<double>(1, 1, 3.0));
    coeff.push_back(Eigen::Triplet<double>(1, 0, 1.0));
    coeff.push_back(Eigen::Triplet<double>(1, 2, 1.0));
    SpMatrix<double> m1(2, 3);
    m1.setFromTriplets(coeff.begin(), coeff.end());
    m1.makeCompressed();

    std::vector<Eigen::Triplet<double>> coeff2;
    coeff2.push_back(Eigen::Triplet<double>(0, 0, 5.0));
    coeff2.push_back(Eigen::Triplet<double>(0, 1, 6.0));
    coeff2.push_back(Eigen::Triplet<double>(1, 1, 1.0));
    SpMatrix<double> n1(3, 2);
    n1.setFromTriplets(coeff2.begin(), coeff2.end());
    n1.makeCompressed();

    // evaluate kronecker product m1 \kron n1
    SpMatrix<double> kron1 = Kronecker(m1, n1);
    // expected result
    DMatrix<double> expected1;
    expected1.resize(6, 6);
    expected1 <<
      10, 12,  0,  0, 0, 0,
      0,   2,  0,  0, 0, 0,
      0,   0,  0,  0, 0, 0,
      5,   6, 15, 18, 5, 6,
      0,   1,  0,  3, 0, 1,
      0,   0,  0,  0, 0, 0;

    // check equality
    for (int k = 0; k < kron1.outerSize(); ++k) {
        for (SpMatrix<double>::InnerIterator it(kron1, k); it; ++it) {
            EXPECT_TRUE(almost_equal(expected1(it.row(), it.col()), it.value()));
        }
    }
}
