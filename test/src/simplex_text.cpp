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

#include <fdaPDE/utils.h>
#include <fdaPDE/geometry.h>
using fdapde::core::Simplex;

#include "utils/utils.h"
using fdapde::testing::almost_equal;

TEST(simplex_test, triangle_2d) {
  SMatrix<2, 3> coords;
  coords.col(0) = SVector<2>(0.0, 0.0);
  coords.col(1) = SVector<2>(0.5, 0.0);
  coords.col(2) = SVector<2>(0.0, 0.8);
  Simplex<2, 2> t(coords);

  EXPECT_TRUE(almost_equal(t.measure(), (0.5 * 0.8) / 2));
  EXPECT_TRUE(almost_equal(t.circumcenter(), SVector<2>(0.25, 0.4)));
  EXPECT_TRUE(almost_equal(t.circumradius(), std::sqrt(0.25 * 0.25 + 0.4 * 0.4)));
  EXPECT_TRUE(almost_equal(t.barycenter(), SVector<2>(0.166666666666667, 0.2666666666666667)));
  EXPECT_TRUE(t.bounding_box().first == SVector<2>(0, 0) && t.bounding_box().second == SVector<2>(0.5, 0.8));
  EXPECT_TRUE(t.contains(SVector<2>(0.25, 1.25)) == 0);   // point falls outside
  EXPECT_TRUE(t.contains(SVector<2>(0.25, 0.25)) == 1);   // point falls inside
  EXPECT_TRUE(t.contains(SVector<2>(0.25, 0.00)) == 2);   // point falls on edge
  EXPECT_TRUE(t.contains(SVector<2>(0.00, 0.00)) == 3);   // point falls on vertex

  // compute perimeter of triangle by face iterator
  EXPECT_TRUE(almost_equal(
    std::accumulate(
      t.face_begin(), t.face_end(), 0.0, [](double perimeter, auto& f) { return perimeter + f.measure(); }),
    0.5 + 0.8 + std::sqrt(0.5 * 0.5 + 0.8 * 0.8)));
  // extract face
  Simplex<2, 2>::FaceType f = t.face(2);
  EXPECT_TRUE(f[0] == SVector<2>(0.5, 0) && f[1] == SVector<2>(0, 0.8));
  EXPECT_TRUE(almost_equal(f.measure(), std::sqrt(0.5 * 0.5 + 0.8 * 0.8)));
  EXPECT_TRUE(almost_equal(f.circumcenter(), f.barycenter()));
}

TEST(simplex_test, triangle_3d) {
  SMatrix<3, 3> coords;
  coords.col(0) = SVector<3>(0.0, 0.0, 0.0);
  coords.col(1) = SVector<3>(0.5, 0.2, 0.0);
  coords.col(2) = SVector<3>(0.0, 0.8, 0.6);
  Simplex<2, 3> t(coords);
  
  EXPECT_TRUE(almost_equal(t.measure(), 0.25709920264364883));
  EXPECT_TRUE(almost_equal(
    t.circumcenter(), t[0] + ((t[2] - t[0]).squaredNorm() * ((t[1] - t[0]).cross(t[2] - t[0])).cross(t[1] - t[0]) +
                              (t[1] - t[0]).squaredNorm() * ((t[2] - t[0]).cross(t[1] - t[0])).cross(t[2] - t[0])) /
                               (2 * ((t[1] - t[0]).cross(t[2] - t[0])).squaredNorm())));
  EXPECT_TRUE(almost_equal(t.circumradius(), t.circumcenter().norm()));   // distance from zero
  EXPECT_TRUE(almost_equal(t.barycenter(), SVector<3>(0.1666666666666667, 0.3333333333333333, 0.2000000000000000)));
  EXPECT_TRUE(t.bounding_box().first == SVector<3>(0, 0, 0) && t.bounding_box().second == SVector<3>(0.5, 0.8, 0.6));
  EXPECT_TRUE(t.contains(SVector<3>(0.00, 1.00, 1.00)) == 0);   // point falls outside
  EXPECT_TRUE(t.contains(SVector<3>(0.25, 0.30, 0.15)) == 1);   // point falls inside
  EXPECT_TRUE(t.contains(SVector<3>(0.00, 0.40, 0.30)) == 2);   // point falls on edge
  EXPECT_TRUE(t.contains(SVector<3>(0.50, 0.20, 0.00)) == 3);   // point falls on vertex
  EXPECT_TRUE(almost_equal(t.normal(), ((t[1] - t[0]).cross(t[2] - t[0])).normalized()));  
  
  // compute perimeter of triangle by face iterator
  EXPECT_TRUE(almost_equal(
    std::accumulate(
      t.face_begin(), t.face_end(), 0.0, [](double perimeter, auto& f) { return perimeter + f.measure(); }),
    2.523402260893061));
  // extract face
  Simplex<2, 3>::FaceType f = t.face(0);
  EXPECT_TRUE(f[0] == SVector<3>(0.0, 0.0, 0.0) && f[1] == SVector<3>(0.5, 0.2, 0.0));
  EXPECT_TRUE(almost_equal(f.measure(), std::sqrt(0.5 * 0.5 + 0.2 * 0.2)));
  EXPECT_TRUE(almost_equal(f.circumcenter(), f.barycenter()));
}

TEST(simplex_test, tetrahedron_3d) {
  SMatrix<3, 4> coords;
  coords.col(0) = SVector<3>(0.0, 0.0, 0.0);
  coords.col(1) = SVector<3>(0.4, 0.2, 0.0);
  coords.col(2) = SVector<3>(0.0, 0.8, 0.6);
  coords.col(3) = SVector<3>(0.4, 0.6, 0.8);
  Simplex<3, 3> t(coords);

  EXPECT_TRUE(almost_equal(t.measure(), 0.0266666666666666));
  EXPECT_TRUE(almost_equal(t.circumcenter(), SVector<3>(0.11, 0.28, 0.46)));
  EXPECT_TRUE(almost_equal(t.circumradius(), std::sqrt(0.11 * 0.11 + 0.28 * 0.28 + 0.46 * 0.46)));
  EXPECT_TRUE(almost_equal(t.barycenter(), SVector<3>(0.2, 0.4, 0.35)));
  EXPECT_TRUE(t.bounding_box().first == SVector<3>(0, 0, 0) && t.bounding_box().second == SVector<3>(0.4, 0.8, 0.8));
  EXPECT_TRUE(t.contains(SVector<3>(0.25, 1.25, 0.25)) == 0);   // point falls outside
  EXPECT_TRUE(t.contains(t.barycenter()) == 1);                 // point falls inside
  EXPECT_TRUE(t.contains(t.face(0).barycenter()) == 2);         // point falls on face
  EXPECT_TRUE(t.contains(SVector<3>(0.40, 0.60, 0.80)) == 3);   // point falls on vertex

  // compute surface area of tetrahedron by face iterator
  EXPECT_TRUE(almost_equal(
    std::accumulate(
      t.face_begin(), t.face_end(), 0.0, [](double perimeter, auto& f) { return perimeter + f.measure(); }),
    0.86430301));

  // extract face
  Simplex<3, 3>::FaceType f = t.face(1);
  EXPECT_TRUE(
    f[0] == SVector<3>(0.0, 0.0, 0.0) && f[1] == SVector<3>(0.4, 0.2, 0.0) && f[2] == SVector<3>(0.4, 0.6, 0.8));
  EXPECT_TRUE(almost_equal(f.measure(), 0.19595918));
  EXPECT_TRUE(f.normal() == ((f[1] - f[0]).cross(f[2] - f[0])).normalized());
}
