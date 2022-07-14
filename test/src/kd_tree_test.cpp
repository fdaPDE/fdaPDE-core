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

#include <fdaPDE/geometry.h>
#include <fdaPDE/utils.h>
#include <gtest/gtest.h>   // testing framework

#include "utils/utils.h"
using fdapde::testing::almost_equal;

DMatrix<double> kdtree_test_sample_dataset() {
    // construct a simple dataset in R^2
    DMatrix<double> point_set(6, 2);
    point_set(0, 0) = 2;
    point_set(0, 1) = 3;
    point_set(1, 0) = 5;
    point_set(1, 1) = 4;
    point_set(2, 0) = 9;
    point_set(2, 1) = 6;
    point_set(3, 0) = 4;
    point_set(3, 1) = 7;
    point_set(4, 0) = 8;
    point_set(4, 1) = 1;
    point_set(5, 0) = 7;
    point_set(5, 1) = 2;
    return point_set;
}

TEST(kd_tree_test, construction) {
    // construct a simple dataset in R^2
    DMatrix<double> point_set = kdtree_test_sample_dataset();
    // build kd-tree for point_set
    fdapde::core::KDTree<2> tree(point_set);
    // check tree topology is correct
    std::vector<int> expected_dfs_order = {5, 1, 0, 3, 2, 4};
    std::size_t i = 0;
    for (auto v : tree) {
        EXPECT_TRUE(v == expected_dfs_order[i]);
        i++;
    }
}

TEST(kd_tree_test, nearest_neighbor_search) {
    // construct a simple dataset in R^2
    DMatrix<double> point_set = kdtree_test_sample_dataset();
    // build kd-tree for point_set
    fdapde::core::KDTree<2> tree(point_set);
    // find nearest point
    auto it = tree.nn_search(SVector<2>(9, 2));
    EXPECT_TRUE(*it == 4);
    EXPECT_TRUE((point_set.row(*it).transpose() - SVector<2>(9, 2)).norm() == std::sqrt(2));
}

TEST(kd_tree_test, range_search) {
    // construct a simple dataset in R^2
    DMatrix<double> point_set = kdtree_test_sample_dataset();
    // build kd-tree for point_set
    fdapde::core::KDTree<2> tree(point_set);
    // find points lying inside the query rectangle
    auto set = tree.range_search({SVector<2>(3, 2), SVector<2> (8,6)});
    EXPECT_TRUE(set.size() == 2);
    EXPECT_TRUE(set.find(1) != set.end());
    EXPECT_TRUE(set.find(5) != set.end());
}

