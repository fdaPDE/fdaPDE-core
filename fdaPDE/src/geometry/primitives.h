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
template <typename PolygonT>
    requires(internals::is_matrix_like_v<PolygonT>)
constexpr double signed_measure_2d_polygon(const PolygonT& points) {
    double area = 0;
    fdapde_assert(points.rows() > 0 && points.cols() == 2);
    int n_points = points.rows();
    for (int i = 0; i < n_points - 1; ++i) {
        area += (points(i, 0) + points(i + 1, 0)) * (points(i + 1, 1) - points(i, 1));
    }
    area += (points(n_points - 1, 0) + points(0, 0)) * (points(0, 1) - points(n_points - 1, 1));
    return area;
}
template <typename PolygonT> constexpr bool are_2d_counterclockwise_sorted(const PolygonT& points) {
    return signed_measure_2d_polygon(points) > 0;
}
template <typename PolygonT> constexpr bool are_2d_clockwise_sorted(const PolygonT& points) {
    return signed_measure_2d_polygon(points) < 0;
}

// 2D point-line orientation test

// finds whether point p is on the positive side (left), negative side (right) or is collinear to the directed line
// identified by points (a, b)
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

// test whether point a belongs to the 2D segment identified by points b and c
template <typename PointT>
    requires(internals::is_subscriptable<PointT, int>)
constexpr bool contains(const PointT& a, const PointT& b, const PointT& c) { // -------------------- rename in point_in_2d_segment
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
constexpr bool segment_proper_intersect_2d_segment(const PointT& a, const PointT& b, const PointT& c, const PointT& d) {
    // check proper intersection (points {c, d} lies on opposite sides of ab and points {a, b} on opposite sides of cd)
    if (
      (orientation(c, a, b) == Orientation::LEFT ^ orientation(d, a, b) == Orientation::LEFT) &&
      (orientation(a, c, d) == Orientation::LEFT ^ orientation(b, c, d) == Orientation::LEFT)) {
        return true;
    }
    return false;
}

template <typename PointT>
    requires(internals::is_subscriptable<PointT, int>)
constexpr bool intersect(const PointT& a, const PointT& b, const PointT& c, const PointT& d) {
    // check proper intersection
    if (segment_proper_intersect_2d_segment(a, b, c, d)) { return true; }
    // check if an endpoint of a segment lies on the other segment
    if (contains(c, a, b) || contains(d, a, b) || contains(a, c, d) || contains(b, c, d)) {
        return true; }
    return false;
}

// 2D point in triangle test
template <typename PointT>
    requires(internals::is_subscriptable<PointT, int>)
constexpr bool point_in_2d_tri(const PointT& a, const PointT& t1, const PointT& t2, const PointT& t3) {
    return (t3[0] - a[0]) * (t1[1] - a[1]) >= (t1[0] - a[0]) * (t3[1] - a[1]) &&
           (t1[0] - a[0]) * (t2[1] - a[1]) >= (t2[0] - a[0]) * (t1[1] - a[1]) &&
           (t2[0] - a[0]) * (t3[1] - a[1]) >= (t3[0] - a[0]) * (t2[1] - a[1]);
}

// 2D point in polygon (closed simple chain of points) test. ray-casting algorithm
template <typename PolygonT, typename PointT>
    requires(internals::is_matrix_like_v<PolygonT> && internals::is_vector_like_v<PointT>)
constexpr bool point_in_2d_polygon(const PolygonT& points, const PointT& p) {
    bool inside = false;
    fdapde_assert(points.rows() > 0 && points.cols() == 2);
    int n = points.rows();
    for (int i = 0, j = n - 1; i < n; j = i++) {
        if (
          ((points(i, 1) > p[1]) != (points(j, 1) > p[1])) &&
          (p[0] <
           (points(j, 0) - points(i, 0)) * (p[1] - points(i, 1)) / (points(j, 1) - points(i, 1)) + points(i, 0))) {
            inside = !inside;
        }
    }
    return inside;
}

// 2D polygon in polygon test
template <typename InnerPolygonT, typename OuterPolygonT>
    requires(internals::is_matrix_like_v<InnerPolygonT> && internals::is_matrix_like_v<OuterPolygonT>)
constexpr bool polygon_in_2d_polygon(const InnerPolygonT& P, const OuterPolygonT& Q) {
    fdapde_assert(P.rows() > 0 && P.cols() == 2 && Q.rows() > 0 && Q.cols() == 2);
    // check if all points of P are inside Q
    for (int i = 0, n = P.rows(); i < n; ++i) {
        if (!point_in_2d_polygon(Q, P.row(i))) { return false; }
    }
    // check if all edges of P do not properly intersect any edge of Q
    for (int i = 0, n = P.rows() - 1; i < n; ++i) {
        int h = i + 1 % n;
        for (int j = 0, m = Q.rows() - 1; j < m; ++j) {
            if (segment_proper_intersect_2d_segment(P.row(i), P.row(h), Q.row(j), Q.row(j + 1 % m))) { return false; }
        }
    }
    return true;
}

// checks if point D is inside the circumcircle of the triangle (A, B, C) (Delaunay criterion)
// template <typename PointT>
//     requires(internals::is_subscriptable<PointT, int>)
// constexpr bool in_circle(const PointT& A, const PointT& B, const PointT& C, const PointT& D) {
//     double Ax = A[0] - D[0], Ay = A[1] - D[1];
//     double Bx = B[0] - D[0], By = B[1] - D[1];
//     double Cx = C[0] - D[0], Cy = C[1] - D[1];

//     double det = Ax * (By * (Cx * Cx + Cy * Cy) - Cy * (Bx * Bx + By * By)) -
//                  Ay * (Bx * (Cx * Cx + Cy * Cy) - Cx * (Bx * Bx + By * By)) +
//                  (Ax * Ax + Ay * Ay) * (Bx * Cy - By * Cx);

//     return det > 0;  // D is inside the circumcircle if determinant is positive
// }

// // computes circumcenter of triangle given its 2D coordinates
// template <typename PointT>
// requires(internals::is_subscriptable<PointT, int>)
// constexpr PointT circumcenter(const PointT& A, const PointT& B, const PointT& C) {
//     double x1 = A[0], y1 = A[1];
//     double x2 = B[0], y2 = B[1];
//     double x3 = C[0], y3 = C[1];

//     double D = 2.0 * (x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2));
    
//     double x1sq = x1 * x1 + y1 * y1;
//     double x2sq = x2 * x2 + y2 * y2;
//     double x3sq = x3 * x3 + y3 * y3;

//     double Ux = (x1sq*(y2 - y3) + x2sq*(y3 - y1) + x3sq*(y1 - y2)) / D;
//     double Uy = (x1sq*(x3 - x2) + x2sq*(x1 - x3) + x3sq*(x2 - x1)) / D;

//     return PointT(Ux, Uy);
// }

// // detects if p is inside circle of diameter ab
// template <typename PointT>
//     requires(internals::is_subscriptable<PointT, int>)
// constexpr bool is_encroached(const PointT& p, const PointT& a, const PointT& b) {
//     PointT m = 0.5 * (a + b);  // midpoint
//     double radius_sq = 0.25 * (a - b).squaredNorm();
//     double dist_sq = (p - m).squaredNorm();
//     return dist_sq < radius_sq - machine_epsilon; 
// }

// // computes the angle between two segments that share vertex p in 2D (counterclockwise)
// // the angle is in degrees
// template <typename PointT>
//     requires(internals::is_subscriptable<PointT, int>)
// constexpr double angle_between(const PointT& a, const PointT& p, const PointT& b) {
//     PointT v1 = a - p;
//     PointT v2 = b - p;
//     double dot = v1.dot(v2);
//     double norm1 = std::sqrt(v1.squaredNorm());
//     double norm2 = std::sqrt(v2.squaredNorm());

//     double cos_theta = dot / (norm1 * norm2);
//     cos_theta = std::fmax(-1.0, std::fmin(1.0, cos_theta));  

//     double angle_rad = std::acos(cos_theta);
//     // 2D vector product to dtermine orientation
//     double cross = v1[0] * v2[1] - v1[1] * v2[0];
//     // if cross > 0: angle is clockwise, so we need to subtract from 2 * pi since boundary is counterclockwise oriented
//     if (cross > 0)
//         angle_rad = 2 * M_PI - angle_rad;

//     return angle_rad * 180.0 / M_PI;
// }

// // checks if the angle between two segments that share vertex p in 2D is acute
// template <typename PointT>
//     requires(internals::is_subscriptable<PointT, int>)
// constexpr bool is_angle_acute(const PointT& a, const PointT& p, const PointT& b) {
//     return angle_between(a, p, b) < 90.0 - machine_epsilon;
// }

// // calculates segment ab's length (2D)
// template <typename PointT>
//     requires(internals::is_subscriptable<PointT, int>)
// constexpr double segment_length(const PointT& a, const PointT& b) {
//     const double dx = a[0] - b[0];
//     const double dy = a[1] - b[1];
//     return std::sqrt(dx * dx + dy * dy);
// }


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
