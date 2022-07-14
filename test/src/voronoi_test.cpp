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

// using fdapde::core::circumcenter;
// using fdapde::core::Voronoi;

// #include "utils/mesh_loader.h"
#include "utils/utils.h"
using fdapde::testing::almost_equal;
// using fdapde::testing::MeshLoader;

// TEST(voronoi_test, circumcenter) {
//   // define vertices of 2D test triangle
//   SVector<2> p1(0.5, 0.5);
//   SVector<2> p2(0.4, 0.3);
//   SVector<2> p3(1.0, 1.0);
//   // compute circumcenter
//   SVector<2> c = circumcenter(p1, p2, p3);
//   EXPECT_TRUE(almost_equal(c[0], 1.75));
//   EXPECT_TRUE(almost_equal(c[1], -0.25));
// }

// TEST(voronoi_test, delanoy_dual) {
//   // load mesh
//   MeshLoader<Mesh2D> domain("unit_square");
//   Voronoi<2,2> v(domain.mesh);  
// }

TEST(voronoi_test, kd_tree) {
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

TEST(voronoi_test, nearest_neighbor_search) {
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

    // build kd-tree for point_set
    fdapde::core::KDTree<2> tree(point_set);
    // find nearest point
    auto it = tree.nn_search(SVector<2>(9, 2));
    EXPECT_TRUE(*it == 4);
    EXPECT_TRUE((point_set.row(*it).transpose() - SVector<2>(9, 2)).norm() == std::sqrt(2));
}

TEST(voronoi_test, range_search) {
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

    // build kd-tree for point_set
    fdapde::core::KDTree<2> tree(point_set);
    // find nearest point
    auto it = tree.range_search({SVector<2>(3, 2), SVector<2> (8,6)});

    for (auto ll = it.begin(); ll != it.end(); ++ll) { std::cout << *ll << std::endl; }
}
