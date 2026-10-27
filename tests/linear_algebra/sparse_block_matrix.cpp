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

namespace {
using fdapde::SparseBlockMatrix;

// scalar zeros must inherit both block-axis extents before sparse evaluation
TEST(SparseBlockMatrix, ScalarZerosPreserveDiagonalBlockDimensions) {
    Eigen::Matrix2d first, second;
    first << 1., 2., 3., 4.;
    second << 5., 6., 7., 8.;
    const SparseBlockMatrix<double, 2, 2> blocks(first, 0., 0., second);
    // the two block rows contribute two rows each
    ASSERT_EQ(blocks.rows(), 4);
    // the two block columns contribute two columns each, including scalar placeholders
    ASSERT_EQ(blocks.cols(), 4);
    Eigen::Matrix4d expected = Eigen::Matrix4d::Zero();
    expected.topLeftCorner<2, 2>() = first;
    expected.bottomRightCorner<2, 2>() = second;
    const Eigen::SparseMatrix<double> assembled(blocks);
    // sparse iteration must reproduce both diagonal blocks and their inferred zero blocks
    EXPECT_EQ((Eigen::MatrixXd(assembled) - expected).norm(), 0.);
}

// rectangular block shapes distinguish row extents from column extents during zero-block allocation
TEST(SparseBlockMatrix, ScalarZerosRetainRectangularBlockShapes) {
    const Eigen::MatrixXd first = Eigen::MatrixXd::Constant(2, 3, 2.);
    const Eigen::MatrixXd second = Eigen::MatrixXd::Constant(4, 1, 3.);
    SparseBlockMatrix<double, 2, 2> blocks(first, 0., 0., second);
    // the upper right zero block inherits the first row's height
    ASSERT_EQ(blocks.block(0, 1).rows(), 2);
    // the upper right zero block inherits the second column's width
    ASSERT_EQ(blocks.block(0, 1).cols(), 1);
    // the lower left zero block inherits the second row's height
    ASSERT_EQ(blocks.block(1, 0).rows(), 4);
    // the lower left zero block inherits the first column's width
    ASSERT_EQ(blocks.block(1, 0).cols(), 3);
    blocks.coeffRef(1, 3) = 7.;
    blocks.coeffRef(5, 2) = 8.;
    Eigen::MatrixXd expected = Eigen::MatrixXd::Zero(6, 4);
    expected.topLeftCorner(2, 3) = first;
    expected.bottomRightCorner(4, 1) = second;
    expected(1, 3) = 7.;
    expected(5, 2) = 8.;
    const Eigen::SparseMatrix<double> assembled(blocks);
    // global writes into both inferred zero blocks reach their independently known logical coordinates
    EXPECT_EQ((Eigen::MatrixXd(assembled) - expected).norm(), 0.);
}

// unconstrained scalar-only rows and columns fall back to unit extents without reading past the block layout
TEST(SparseBlockMatrix, AllScalarLayoutUsesUnitExtents) {
    const SparseBlockMatrix<double, 3, 2> blocks(0., 0., 0., 0., 0., 0.);
    // three scalar-only block rows each infer a unit height
    ASSERT_EQ(blocks.rows(), 3);
    // two scalar-only block columns each infer a unit width
    ASSERT_EQ(blocks.cols(), 2);
    const Eigen::SparseMatrix<double> assembled(blocks);
    // every inferred unit block is empty sparse storage
    EXPECT_EQ(assembled.nonZeros(), 0);
}

// all non-scalar blocks within a column must agree on width, regardless of which block establishes that width
TEST(SparseBlockMatrix, RejectsInconsistentBlockColumnWidths) {
    const Eigen::MatrixXd upper = Eigen::MatrixXd::Ones(2, 3);
    const Eigen::MatrixXd lower = Eigen::MatrixXd::Ones(4, 2);
    // equal widths within each row cannot hide the conflicting widths within the first column
    EXPECT_THROW((SparseBlockMatrix<double, 2, 2>(upper, 0., lower, 0.)), std::invalid_argument);
}
}   // namespace
