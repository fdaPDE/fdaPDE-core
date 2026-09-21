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

#include <fdaPDE/sparse_linear_algebra.h>
#include <gtest/gtest.h>

#include <Eigen/SparseCore>
#include <unsupported/Eigen/KroneckerProduct>

// independent Eigen tensor products agree across empty, rectangular and duplicate-containing sparse inputs
TEST(linear_algebra, kronecker_matches_eigen) {
    for (int rows : {0, 1, 3, 7}) {
        for (int cols : {0, 1, 4}) {
            std::vector<fdapde::Triplet<double>> native;
            std::vector<Eigen::Triplet<double>> eigen;
            for (int i = 0; i < rows; ++i) {
                for (int j = 0; j < cols; ++j) {
                    if ((i + j) % 3 != 0) continue;
                    native.emplace_back(i, j, i - j + 2.);
                    native.emplace_back(i, j, -1.);
                    eigen.emplace_back(i, j, i - j + 2.);
                    eigen.emplace_back(i, j, -1.);
                }
            }
            const fdapde::SparseMatrix<double> a(rows, cols, native);
            const fdapde::SparseMatrix<double> b(
              2, 3,
              {
                {0, 1, -2.},
                {1, 0, 3. },
                {1, 2, 1. }
            });
            Eigen::SparseMatrix<double, Eigen::RowMajor> ea(rows, cols), eb(2, 3);
            ea.setFromTriplets(eigen.begin(), eigen.end());
            ea.prune(0.);
            eb.insert(0, 1) = -2.;
            eb.insert(1, 0) = 3.;
            eb.insert(1, 2) = 1.;
            const auto actual = fdapde::kron(a, b);
            Eigen::SparseMatrix<double, Eigen::RowMajor> expected = Eigen::kroneckerProduct(ea, eb);
            expected.prune(0.);
            // the independent Eigen expression agrees on the multiplied row count
            ASSERT_EQ(actual.rows(), expected.rows());
            // the independent Eigen expression agrees on the multiplied column count
            ASSERT_EQ(actual.cols(), expected.cols());
            // pruning exact zeros produces the same number of stored products as Eigen
            EXPECT_EQ(actual.non_zeros(), expected.nonZeros());
            for (int i = 0; i < actual.rows(); ++i) {
                for (int j = 0; j < actual.cols(); ++j) {
                    // coefficient lookup checks both structural zeros and nonzero tensor entries
                    EXPECT_DOUBLE_EQ(actual.coeff(i, j), expected.coeff(i, j));
                }
            }
        }
    }
}
