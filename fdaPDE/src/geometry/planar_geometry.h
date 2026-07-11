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

#ifndef __FDAPDE_GEOMETRY_PLANAR_GEOMETRY_H__
#define __FDAPDE_GEOMETRY_PLANAR_GEOMETRY_H__

#include <stdexcept>

#include "header_check.h"

namespace fdapde {
namespace internals {
namespace planar {

struct point_t {
    double x;
    double y;
};

enum class predicate_sign : int {
    negative = -1,
    zero = 0,
    positive = 1
};

enum class point_location : int {
    outside = -1,
    boundary = 0,
    inside = 1
};

inline int normalization_shift(double max_coordinate) {
    return max_coordinate == 0.0 ? 0 : -std::ilogb(max_coordinate);
}

template <typename... Points> inline int normalization_shift(const point_t& first, const Points&... rest) {
    double max_coordinate = std::max(std::abs(first.x), std::abs(first.y));
    const auto include = [&](const point_t& point) {
        max_coordinate = std::max({max_coordinate, std::abs(point.x), std::abs(point.y)});
    };
    (include(rest), ...);
    return normalization_shift(max_coordinate);
}

inline point_t normalize(const point_t& point, int shift) {
    return {std::scalbn(point.x, shift), std::scalbn(point.y, shift)};
}

inline bool less(const point_t& a, const point_t& b) { return a.x < b.x || (a.x == b.x && a.y < b.y); }
inline bool equal(const point_t& a, const point_t& b) { return a.x == b.x && a.y == b.y; }

// certified fast filter from Jonathan Shewchuk's public-domain robust predicates; power-of-two normalization avoids
// overflow and underflow across uniformly scaled inputs, while unresolved signs are classified as degenerate
inline predicate_sign orient2d(const point_t& a, const point_t& b, const point_t& c) {
    const int shift = normalization_shift(a, b, c);
    const point_t normalized_a = normalize(a, shift);
    const point_t normalized_b = normalize(b, shift);
    const point_t normalized_c = normalize(c, shift);
    const double acx = normalized_a.x - normalized_c.x;
    const double bcx = normalized_b.x - normalized_c.x;
    const double acy = normalized_a.y - normalized_c.y;
    const double bcy = normalized_b.y - normalized_c.y;
    const double det_left = acx * bcy;
    const double det_right = acy * bcx;
    const double det = det_left - det_right;

    double det_sum;
    if (det_left > 0.0) {
        if (det_right <= 0.0) return predicate_sign::positive;
        det_sum = det_left + det_right;
    } else if (det_left < 0.0) {
        if (det_right >= 0.0) return predicate_sign::negative;
        det_sum = -det_left - det_right;
    } else {
        if (det_right < 0.0) return predicate_sign::positive;
        if (det_right > 0.0) return predicate_sign::negative;
        return predicate_sign::zero;
    }

    constexpr double epsilon = std::numeric_limits<double>::epsilon();
    const double error_bound = (3.0 + 16.0 * epsilon) * epsilon * det_sum;
    if (det > error_bound) return predicate_sign::positive;
    if (det < -error_bound) return predicate_sign::negative;
    return predicate_sign::zero;
}

inline bool between(double a, double b, double x) { return x >= std::min(a, b) && x <= std::max(a, b); }

inline bool on_segment(const point_t& a, const point_t& b, const point_t& p) {
    return orient2d(a, b, p) == predicate_sign::zero && between(a.x, b.x, p.x) && between(a.y, b.y, p.y);
}

inline bool segments_intersect(const point_t& a, const point_t& b, const point_t& c, const point_t& d) {
    if (
      std::max(a.x, b.x) < std::min(c.x, d.x) || std::max(c.x, d.x) < std::min(a.x, b.x) ||
      std::max(a.y, b.y) < std::min(c.y, d.y) || std::max(c.y, d.y) < std::min(a.y, b.y)) {
        return false;
    }
    const predicate_sign o1 = orient2d(a, b, c);
    const predicate_sign o2 = orient2d(a, b, d);
    const predicate_sign o3 = orient2d(c, d, a);
    const predicate_sign o4 = orient2d(c, d, b);
    if (o1 == predicate_sign::zero && between(a.x, b.x, c.x) && between(a.y, b.y, c.y)) return true;
    if (o2 == predicate_sign::zero && between(a.x, b.x, d.x) && between(a.y, b.y, d.y)) return true;
    if (o3 == predicate_sign::zero && between(c.x, d.x, a.x) && between(c.y, d.y, a.y)) return true;
    if (o4 == predicate_sign::zero && between(c.x, d.x, b.x) && between(c.y, d.y, b.y)) return true;
    return o1 != o2 && o3 != o4;
}

// normalized shoelace accumulation is used only for a reliable orientation sign
inline predicate_sign ring_orientation(const std::vector<point_t>& points, const std::vector<int>& ring) {
    double max_coordinate = 0.0;
    for (int id : ring) { max_coordinate = std::max({max_coordinate, std::abs(points[id].x), std::abs(points[id].y)}); }
    const int shift = normalization_shift(max_coordinate);
    const point_t origin = normalize(points[ring.front()], shift);
    double sum = 0.0;
    double compensation = 0.0;
    double magnitude = 0.0;
    for (int i = 0, n = ring.size(); i < n; ++i) {
        const point_t a = normalize(points[ring[i]], shift);
        const point_t b = normalize(points[ring[(i + 1) % n]], shift);
        const double term = (a.x - origin.x) * (b.y - origin.y) - (a.y - origin.y) * (b.x - origin.x);
        const double corrected = term - compensation;
        const double updated = sum + corrected;
        compensation = (updated - sum) - corrected;
        sum = updated;
        magnitude += std::abs(term);
    }
    const double error_bound = 16.0 * std::numeric_limits<double>::epsilon() * magnitude;
    if (std::abs(sum) <= error_bound) {
        throw std::invalid_argument("Polygon ring has numerically ambiguous signed area.");
    }
    return sum > 0.0 ? predicate_sign::positive : predicate_sign::negative;
}

inline std::vector<point_t> read_points(const Matrix<double, Dynamic, Dynamic>& matrix, const char* name) {
    if (matrix.cols() != 2) { throw std::invalid_argument(std::string(name) + " must have exactly two columns."); }
    std::vector<point_t> points(matrix.rows());
    for (int i = 0; i < matrix.rows(); ++i) {
        const double x = matrix(i, 0);
        const double y = matrix(i, 1);
        if (!std::isfinite(x) || !std::isfinite(y)) {
            throw std::invalid_argument(std::string(name) + " must contain only finite coordinates.");
        }
        points[i] = {x, y};
    }
    return points;
}

inline void validate_unique(const std::vector<point_t>& points) {
    std::vector<int> order(points.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) { return less(points[a], points[b]); });
    for (std::size_t i = 1; i < order.size(); ++i) {
        if (equal(points[order[i - 1]], points[order[i]])) {
            throw std::invalid_argument("Planar geometry contains duplicate points.");
        }
    }
}

inline point_location
locate_in_polygon(const std::vector<point_t>& points, const std::vector<int>& ring, const point_t& p) {
    int winding = 0;
    for (int i = 0, n = ring.size(); i < n; ++i) {
        const point_t& a = points[ring[i]];
        const point_t& b = points[ring[(i + 1) % n]];
        const predicate_sign sign = orient2d(a, b, p);
        if (sign == predicate_sign::zero && between(a.x, b.x, p.x) && between(a.y, b.y, p.y)) {
            return point_location::boundary;
        }
        if (a.y <= p.y && b.y > p.y && sign == predicate_sign::positive) ++winding;
        if (a.y > p.y && b.y <= p.y && sign == predicate_sign::negative) --winding;
    }
    return winding == 0 ? point_location::outside : point_location::inside;
}

inline void validate_boundary(const std::vector<point_t>& points) {
    if (points.size() < 3) { throw std::invalid_argument("Polygon ring needs at least three points."); }
    validate_unique(points);
    if (equal(points.front(), points.back())) { throw std::invalid_argument("Polygon ring must be unclosed."); }
    const int n = points.size();
    for (int i = 0; i < n; ++i) {
        if (orient2d(points[(i + n - 1) % n], points[i], points[(i + 1) % n]) == predicate_sign::zero) {
            throw std::invalid_argument("Polygon ring contains consecutive collinear points.");
        }
    }
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            if (j == i + 1 || (i == 0 && j == n - 1)) continue;
            if (segments_intersect(points[i], points[(i + 1) % n], points[j], points[(j + 1) % n])) {
                throw std::invalid_argument("Polygon ring must be simple.");
            }
        }
    }
}

inline bool
rings_intersect(const std::vector<point_t>& points, const std::vector<int>& first, const std::vector<int>& second) {
    for (int i = 0; i < int(first.size()); ++i) {
        for (int j = 0; j < int(second.size()); ++j) {
            if (segments_intersect(
                  points[first[i]], points[first[(i + 1) % first.size()]], points[second[j]],
                  points[second[(j + 1) % second.size()]])) {
                return true;
            }
        }
    }
    return false;
}

struct validated_domain_t {
    std::vector<point_t> points;
    std::vector<int> outer;
    std::vector<std::vector<int>> holes;
};

inline std::vector<int>
append_ring(std::vector<point_t>& points, const Matrix<double, Dynamic, Dynamic>& matrix, const char* name) {
    std::vector<point_t> local = read_points(matrix, name);
    validate_boundary(local);
    const int begin = points.size();
    points.insert(points.end(), local.begin(), local.end());
    std::vector<int> ring(local.size());
    std::iota(ring.begin(), ring.end(), begin);
    return ring;
}

inline validated_domain_t validate_planar_domain(const PlanarDomain& domain) {
    validated_domain_t result;
    result.outer = append_ring(result.points, domain.outer, "Planar-domain outer boundary");
    result.holes.reserve(domain.holes.size());
    for (const auto& hole : domain.holes) {
        result.holes.push_back(append_ring(result.points, hole, "Planar-domain hole boundary"));
    }
    validate_unique(result.points);

    for (const auto& hole : result.holes) {
        if (rings_intersect(result.points, result.outer, hole)) {
            throw std::invalid_argument("Planar-domain holes must not touch or intersect the outer boundary.");
        }
        for (int node : hole) {
            if (locate_in_polygon(result.points, result.outer, result.points[node]) != point_location::inside) {
                throw std::invalid_argument("Planar-domain holes must lie strictly inside the outer boundary.");
            }
        }
    }
    for (int i = 0; i < int(result.holes.size()); ++i) {
        for (int j = i + 1; j < int(result.holes.size()); ++j) {
            if (rings_intersect(result.points, result.holes[i], result.holes[j])) {
                throw std::invalid_argument("Planar-domain holes must be mutually disjoint and non-touching.");
            }
            if (
              locate_in_polygon(result.points, result.holes[i], result.points[result.holes[j][0]]) !=
                point_location::outside ||
              locate_in_polygon(result.points, result.holes[j], result.points[result.holes[i][0]]) !=
                point_location::outside) {
                throw std::invalid_argument("Planar-domain holes must not be nested.");
            }
        }
    }
    return result;
}

}   // namespace planar
}   // namespace internals
}   // namespace fdapde

#endif   // __FDAPDE_GEOMETRY_PLANAR_GEOMETRY_H__
