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
#include <limits>

#include <fdaPDE/utils.h>
#include <fdaPDE/linear_algebra.h>
using fdapde::core::VectorSpace;

#include "utils/constants.h"
using fdapde::testing::DOUBLE_TOLERANCE;

TEST(vector_space_test, project_over_1d_space) {
    // define a vector space by defining its direction
    SVector<2> i(1, 1);
    VectorSpace<1, 2> vs({i});   // a line embedded in a 2D plane
    // its orthonormal basis is maden by the direction \tilde i = (1/sqrt(2), 1/sqrt(2))

    // take a 2D point and project it into vs
    SVector<2> p(4, 3.5);
    EXPECT_TRUE((vs.project_into(p) - SVector<2>(3.75, 3.75)).norm() < DOUBLE_TOLERANCE);
    EXPECT_TRUE((vs.project_onto(p) - SVector<1>(5.30330085889910643301)).norm() < DOUBLE_TOLERANCE);
}

TEST(vector_space_test, project_over_2d_space) {
    // define a vector space by defining its direction
    SVector<3> i(1, 1, 0), j(0, 1, 1);
    VectorSpace<2, 3> vs({i, j});   // a 2D plane embedded in a 3D space
    // orthonormal basis of vs is maden by the set {\tilde i, \tilde j} = {(1/sqrt(2), 1/sqrt(2), 0), (-1/sqrt(6),
    // 1/sqrt(6), 2/sqrt(6))}

    // take a 3D point and project it into vs
    SVector<3> p(7.1, 3.4, 2);
    EXPECT_TRUE((vs.project_into(p) - SVector<3>(5.20, 5.3, 0.1)).norm() < DOUBLE_TOLERANCE);
    EXPECT_TRUE(
      (vs.project_onto(p) - SVector<2>(7.42462120245874900621, 0.12247448713915890491)).norm() < DOUBLE_TOLERANCE);
}

TEST(vector_space_test, l2_distance_from_1d_space) {
    // define a vector space by defining its direction
    SVector<2> i(1, 2);
    VectorSpace<1, 2> vs({i});   // a line embedded in a 2D plane

    // the distance between a point in the space and the space itself should be zero
    EXPECT_NEAR(vs.distance(vs({1})), 0, DOUBLE_TOLERANCE);
    // the distance between a point projected into the space and the space itself should be zero
    SVector<2> p(7, 9);
    EXPECT_NEAR(vs.distance(vs.project_into(p)), 0, DOUBLE_TOLERANCE);
    // expect .distance() to compute correct distance between q and its projection
    EXPECT_NEAR(vs.distance(p), (p - vs.project_into(p)).squaredNorm(), DOUBLE_TOLERANCE);
}

TEST(vector_space_test, l2_distance_from_2d_space) {
    // define a vector space by defining its direction
    SVector<3> i(1, 2, 10), j(7, 7, 5);
    VectorSpace<2, 3> vs({i, j});   // a place embedded in a 3D space

    // the distance between a point in the space and the space itself should be zero
    EXPECT_NEAR(vs.distance(vs({1})), 0, DOUBLE_TOLERANCE);
    // the distance between a point projected into the space and the space itself should be zero
    SVector<3> p(3, 9, 2);
    EXPECT_NEAR(vs.distance(vs.project_into(p)), 0, DOUBLE_TOLERANCE);
    // expect .distance() to compute correct distance between q and its projection
    EXPECT_NEAR(vs.distance(p), (p - vs.project_into(p)).squaredNorm(), DOUBLE_TOLERANCE);
}

TEST(vector_space_test, affine_space) {
    // define a vector space by defining its direction
    SVector<3> i(1, 1, 0), j(0, 1, 1);
    SVector<3> offset(0, 0, 5);
    VectorSpace<2, 3> vs({i, j}, offset);   // a plane embedded in a 3D space passing by (0,0,5)
    // orthonormal basis of vs is maden by the set
    // {\tilde i, \tilde j} = {(1/sqrt(2), 1/sqrt(2), 0), (-1/sqrt(6), 1/sqrt(6), 2/sqrt(6))}

    // take a 3D point and project it into vs (we expect the same results of project2D corrected by (0,0,5))
    SVector<3> p(7.1, 3.4, 2);
    EXPECT_TRUE(
      (vs.project_into(p) - SVector<3>(6.86666666666666666667, 3.63333333333333333333, -3.23333333333333333334) -
       offset).norm() < DOUBLE_TOLERANCE);
    EXPECT_TRUE(
      (vs.project_onto(p) - SVector<2>(7.42462120245874900621, -3.96000841749947125875)).norm() < DOUBLE_TOLERANCE);

    // the distance between a point in the space and the space itself should be zero
    EXPECT_NEAR(vs.distance(vs({1, 2})), 0, DOUBLE_TOLERANCE);
    // the distance between a point projected into the space and the space itself should be zero
    SVector<3> q(3, 9, 2);
    EXPECT_NEAR(vs.distance(vs.project_into(q)), 0, DOUBLE_TOLERANCE);
    // expect .distance() to compute correct distance between q and its projection
    EXPECT_NEAR(vs.distance(q), (q - vs.project_into(q)).squaredNorm(), DOUBLE_TOLERANCE);
}
