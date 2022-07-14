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

#ifndef __FDAPDE_GEOMETRIC_PRIMITIVES_H__
#define __FDAPDE_GEOMETRIC_PRIMITIVES_H__

#include "header_check.h"

namespace fdapde {
namespace internals {

// a collection of functions to perform common computational geometric tasks, see, e.g.
//  (1) O’Rourke, J. (1998). Computational geometry in C. Cambridge University Press.

// 2D geometry
  
// signed area of 2D triangle given its vetices (lemma 1.3.1 of (1)). area is negative if a, b, c form a clockwise path
template <typename PointT>
    requires(internals::is_subscriptable<PointT, int>)
constexpr double signed_measure_2d_tri(const PointT& a, const PointT& b, const PointT& c) {
    return 0.5 * ((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]));
}
// unsigned area od 2D triangle given its vertices
template <typename PointT>
    requires(internals::is_subscriptable<PointT, int>)
constexpr double measure_2d_tri(const PointT& a, const PointT& b, const PointT& c) {
    return std::abs(signed_measure_2d_tri(a, b, c));
}
// finds whether 2D points a, b, and c are sorted clockwise or counterclockwise
template <typename PointT> constexpr bool are_2d_clockwise_sorted(const PointT& a, const PointT& b, const PointT& c) {
    return signed_measure_2d_tri(a, b, c) < 0;
}
template <typename PointT>
constexpr bool are_2d_counterclockwise_sorted(const PointT& a, const PointT& b, const PointT& c) {
    return signed_measure_2d_tri(a, b, c) > 0;
}
// area of 2D polygon given counterclockwise sorted vertices v_0, v_1, \ldots, v_{n - 1} (lemma 1.3.3 of (1))
template <typename PointList>
    requires(internals::is_eigen_dense_xpr_v<PointList> || internals::is_subscriptable<PointList, int>)
constexpr double signed_measure_2d_polygon(const PointList& points) {
    double area = 0;
    if constexpr (internals::is_eigen_dense_xpr_v<PointList>) {
        fdapde_assert(points.rows() > 0 && points.cols() == 2);
        int n_points = points.rows();
        for (int i = 0; i < n_points - 1; ++i) {
            area += (points(i, 0) + points(i + 1, 0)) * (points(i + 1, 1) - points(i, 1));
        }
        area += (points(n_points - 1, 0) + points(0, 0)) * (points(0, 1) - points(n_points - 1, 1));
    } else if (internals::is_subscriptable<PointList, int>) {
        // assume points to be a RowMajor expansion of the polygon nodes' coordinates
        fdapde_assert(points.size() % 2 == 0);
        int n_points = points.size() / 2;
        for (int i = 0; i < n_points - 1; ++i) {
            area += (points[i] + points[i + 2]) * (points[i + 3] - points[i + 1]);
        }
        area += (points[n_points - 2] + points[0]) * (points[1] - points[n_points - 1]);
    }
    return area;
}
template <typename PointList> constexpr bool are_2d_counterclockwise_sorted(const PointList& points) {
    return signed_measure_2d_polygon(points) > 0;
}
template <typename PointList> constexpr bool are_2d_clockwise_sorted(const PointList& points) {
    return signed_measure_2d_polygon(points) < 0;
}

// 2D point-line orientation test

// finds whether a 2D point p is on the positive side (left), negative side (right) or is collinear to the 2D directed
// line identified by points (a, b)
enum Orientation { LEFT = 0, RIGHT = 1, COLLINEAR = 2 };
template <typename PointT>
    requires(internals::is_subscriptable<PointT, int>)
constexpr Orientation orientation(const PointT& p, const PointT& a, const PointT& b) {
    double signed_measure = signed_measure_2d_tri(a, b, p);
    if (signed_measure > machine_epsilon) return Orientation::LEFT;
    if (signed_measure < machine_epsilon) return Orientation::RIGHT;
    return Orientation::COLLINEAR;
}
// finds whether the triplet of 2D points {a, b, c} are collinear
template <typename PointT>
    requires(internals::is_subscriptable<PointT, int>)
constexpr bool collinear(const PointT& a, const PointT& b, const PointT& c) {
    return almost_equal(signed_measure_2d_tri(a, b, c), 0.0);
}
// find whether the pair of 2D segments {{a, b}, {b, c}} form a convex turn
template <typename point_t> bool convex_turn(const point_t& a, const point_t& b, const point_t& c) {
    return orientation(a, b, c) == Orientation::RIGHT;    // convex turn \iff a is left to or on {b, c}
}
// find whether the pair of 2D segments {{a, b}, {b, c}} form a reflex turn
template <typename point_t> bool reflex_turn(const point_t& a, const point_t& b, const point_t& c) {
    return !convex_turn(a, b, c);   // reflex turn \iff not convex turn
}

// test whether a point a belongs to the 2D segment identified by points b and c
template <typename PointT>
    requires(internals::is_subscriptable<PointT, int>)
constexpr bool contains(const PointT& a, const PointT& b, const PointT& c) {
    if (!collinear(a, b, c)) return false;
    // if bc is not vertical, check x coordinates, otherwise check y coordinates
    if (b[0] != c[0]) {
        return ((b[0] <= a[0]) && (a[0] <= c[0])) || ((b[0] >= a[0]) && (a[0] >= c[0]));
    } else {
        return ((b[1] <= a[1]) && (a[1] <= c[1])) || ((b[1] >= a[1]) && (a[1] >= c[1]));
    }
}
  
// 2D segment-segment intersection test
template <typename PointT>
    requires(internals::is_subscriptable<PointT, int>)
constexpr bool intersect(const PointT& a, const PointT& b, const PointT& c, const PointT& d) {
    // check proper intersection (points {c, d} lies on opposite sides of ab and points {a, b} on opposite sides of cd)
    if (
      (orientation(c, a, b) == Orientation::LEFT ^ orientation(d, a, b) == Orientation::LEFT) &&
      (orientation(a, c, d) == Orientation::LEFT ^ orientation(b, c, d) == Orientation::LEFT)) {
        return true;
    }
    // check if an endpoint of a segment lies on the other segment
    if (contains(c, a, b) || contains(d, a, b) || contains(a, c, d) || contains(b, c, d)) { return true; }
    return false;
}

template <typename PointT>
    requires(internals::is_subscriptable<PointT, int>)
constexpr bool point_in_2d_tri(const PointT& a, const PointT& t1, const PointT& t2, const PointT& t3) {
    return (t3[0] - a[0]) * (t1[1] - a[1]) >= (t1[0] - a[0]) * (t3[1] - a[1]) &&
           (t1[0] - a[0]) * (t2[1] - a[1]) >= (t2[0] - a[0]) * (t1[1] - a[1]) &&
           (t2[0] - a[0]) * (t3[1] - a[1]) >= (t3[0] - a[0]) * (t2[1] - a[1]);
}

  // then we can detect if a diagonal is fully contained in a polygon

  // 3D geometry

  // signed volume of 3D tetrahedron given its vertices (formula 1.15 of (1))
template <typename PointT>
    requires(internals::is_subscriptable<PointT, int>)
constexpr double signed_measure_3d_tet(const PointT& a, const PointT& b, const PointT& c, const PointT& d) {
    double a1d1 = a[1] - d[1], b2d2 = b[2] - d[2], a2d2 = a[2] - d[2], b1d1 = b[1] - d[1], a0d0 = a[0] - d[0],
           b0d0 = b[0] - d[0];
    return ((a1d1 * b2d2 - a2d2 * b1d1) * (c[0] - d[0]) + (a2d2 * b0d0 - a0d0 * b2d2) * (c[1] - d[1]) +
            (a0d0 * b1d1 - a1d1 * b0d0) * (c[2] - d[2])) /
           6;
}
template <typename PointT>
    requires(internals::is_subscriptable<PointT, int>)
constexpr double measure_3d_tet(const PointT& a, const PointT& b, const PointT& c, const PointT& d) {
    return std::abs(signed_3d_tet_measure(a, b, c, d));
}
  // volume of polyhedron (pag 23 of (1))

}   // namespace internals
}   // namespace fdapde

#endif // __FDAPDE_GEOMETRIC_PRIMITIVES_H__
