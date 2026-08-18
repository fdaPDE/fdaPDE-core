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
#include <gtest/gtest.h>

namespace fdapde {

TEST(LinearAlgebraRuntimeContracts, DynamicSquareOperationsRemainAvailable) {
    Matrix<double, Dynamic, Dynamic> matrix = Matrix<double, 2, 2>({2, 1, 1, 3});
    const Matrix<double, 2, 2> zero = Matrix<double, 2, 2>::Zero();
    const Vector<double, 2> expected_diagonal({2, 3});

    EXPECT_EQ(matrix.symm_part(), matrix);
    EXPECT_EQ(matrix.skew_part(), zero);
    EXPECT_TRUE(almost_equal(matrix.inverse() * matrix, Matrix<double, 2, 2>({1, 0, 0, 1})));
    EXPECT_EQ(matrix.determinant(), 5);
    EXPECT_EQ(matrix.diagonal(), expected_diagonal);
}

TEST(LinearAlgebraRuntimeContracts, SquareOperationsRejectRectangularMatrices) {
    Matrix<double, Dynamic, Dynamic> matrix(2, 3);

    EXPECT_THROW((void)matrix.symm_part(), std::invalid_argument);
    EXPECT_THROW((void)matrix.skew_part(), std::invalid_argument);
    EXPECT_THROW((void)matrix.inverse(), std::invalid_argument);
    EXPECT_THROW((void)matrix.determinant(), std::invalid_argument);
}

TEST(LinearAlgebraRuntimeContracts, DenseDiagonalViewChecksShapeAndIndexes) {
    Matrix<double, Dynamic, Dynamic> rectangular(2, 3);
    EXPECT_THROW((void)rectangular.diagonal(), std::invalid_argument);

    Matrix<double, 2, 2> matrix({1, 2, 3, 4});
    auto diagonal = matrix.diagonal();
    const auto& const_diagonal = diagonal;

    EXPECT_THROW((void)diagonal(-1, 0), std::out_of_range);
    EXPECT_THROW((void)diagonal(2, 0), std::out_of_range);
    EXPECT_THROW((void)diagonal(0, -1), std::out_of_range);
    EXPECT_THROW((void)diagonal(0, 1), std::out_of_range);
    EXPECT_THROW((void)diagonal[-1], std::out_of_range);
    EXPECT_THROW((void)diagonal[2], std::out_of_range);
    EXPECT_THROW((void)const_diagonal(-1, 0), std::out_of_range);
    EXPECT_THROW((void)const_diagonal(2, 0), std::out_of_range);
    EXPECT_THROW((void)const_diagonal(0, -1), std::out_of_range);
    EXPECT_THROW((void)const_diagonal(0, 1), std::out_of_range);
    EXPECT_THROW((void)const_diagonal[-1], std::out_of_range);
    EXPECT_THROW((void)const_diagonal[2], std::out_of_range);
}

}   // namespace fdapde
