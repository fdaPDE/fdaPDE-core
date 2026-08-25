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

// clang-format off
#include <fdaPDE/linear_algebra.h>
#include <Eigen/SparseCore>
#include <fdaPDE/src/linear_algebra/eigen/eigen_helper.h>
#include <gtest/gtest.h>
// clang-format on

#include <vector>

namespace {

using native_triplet = fdapde::Triplet<double>;
using eigen_sparse = Eigen::SparseMatrix<double, Eigen::RowMajor, int>;

std::vector<native_triplet> make_grid_triplets(int subdivisions) {
    const int nodes_per_side = subdivisions + 1;
    std::vector<native_triplet> triplets;
    triplets.reserve(static_cast<std::size_t>(18 * subdivisions * subdivisions));
    const auto append_triangle = [&triplets](int first, int second, int third) {
        const int nodes[3] {first, second, third};
        for (int row = 0; row < 3; ++row) {
            for (int col = 0; col < 3; ++col) {
                triplets.emplace_back(nodes[row], nodes[col], row == col ? 2.0 : -0.5);
            }
        }
    };
    for (int row = 0; row < subdivisions; ++row) {
        for (int col = 0; col < subdivisions; ++col) {
            const int lower_left = row * nodes_per_side + col;
            const int lower_right = lower_left + 1;
            const int upper_left = lower_left + nodes_per_side;
            const int upper_right = upper_left + 1;
            append_triangle(lower_left, lower_right, upper_right);
            append_triangle(lower_left, upper_right, upper_left);
        }
    }
    return triplets;
}

void expect_same_compression(int rows, int cols, const std::vector<native_triplet>& triplets) {
    const fdapde::SparseMatrix<double> native(rows, cols, triplets);

    std::vector<Eigen::Triplet<double, int>> eigen_triplets;
    eigen_triplets.reserve(triplets.size());
    for (const auto& triplet : triplets) { eigen_triplets.emplace_back(triplet.row(), triplet.col(), triplet.value()); }
    eigen_sparse oracle(rows, cols);
    oracle.setFromTriplets(eigen_triplets.begin(), eigen_triplets.end());
    oracle.prune(0.0);
    oracle.makeCompressed();

    ASSERT_EQ(native.rows(), oracle.rows());
    ASSERT_EQ(native.cols(), oracle.cols());
    ASSERT_EQ(native.non_zeros(), oracle.nonZeros());
    for (int row = 0; row < rows; ++row) {
        auto native_it = native.row(row).begin();
        const auto native_end = native.row(row).end();
        for (eigen_sparse::InnerIterator oracle_it(oracle, row); oracle_it; ++oracle_it) {
            ASSERT_NE(native_it, native_end);
            EXPECT_EQ((*native_it).column(), oracle_it.col());
            EXPECT_DOUBLE_EQ((*native_it).value(), oracle_it.value());
            ++native_it;
        }
        EXPECT_EQ(native_it, native_end);
    }
}

TEST(NativeSparseOracle, MatchesEigenTripletCompression) {
    expect_same_compression(
      3, 4,
      std::vector<native_triplet> {
        {2, 3, 4.0 },
        {0, 1, 2.0 },
        {2, 1, -1.0},
        {0, 1, 3.0 },
        {2, 1, 1.0 },
        {0, 3, 0.0 }
    });

    const int subdivisions = 32;
    const int nodes = (subdivisions + 1) * (subdivisions + 1);
    expect_same_compression(nodes, nodes, make_grid_triplets(subdivisions));
}

}   // namespace
