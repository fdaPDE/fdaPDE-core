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

#ifndef __FDAPDE_GEOMETRY_DELAUNAY_H__
#define __FDAPDE_GEOMETRY_DELAUNAY_H__

#include <stdexcept>

#include "header_check.h"

namespace fdapde {

/** @brief Optional centroid refinement bounded by both cell area and insertion count. */
struct DelaunayRefinement {
    double max_area;
    int max_insertions = 1000;
};

namespace internals {
namespace delaunay_2d {

// algorithmic source: Emilia Farina, Emiliaa02/fdaPDE-core stable@ca8fa3e7c03806fc92048e56f1a1dd2ffd88b927
// this is an independent rewrite without the fork's DCEL, JSON, random-site, hole, or Voronoi dependencies

struct point_t {
    double x;
    double y;
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

using cell_t = std::array<int, 3>;
using edge_t = std::array<int, 2>;

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

inline edge_t edge(int a, int b) { return a < b ? edge_t {a, b} : edge_t {b, a}; }

inline bool less(const point_t& a, const point_t& b) { return a.x < b.x || (a.x == b.x && a.y < b.y); }
inline bool equal(const point_t& a, const point_t& b) { return a.x == b.x && a.y == b.y; }

// certified fast filter from Jonathan Shewchuk's public-domain robust predicates; power-of-two normalization avoids
// overflow and underflow across uniformly scaled inputs. Unresolved signs are classified as degenerate.
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

inline predicate_sign incircle(const point_t& a, const point_t& b, const point_t& c, const point_t& d) {
    const predicate_sign orientation = orient2d(a, b, c);
    if (orientation == predicate_sign::zero) {
        throw std::runtime_error("Delaunay triangulation contains a degenerate cell.");
    }

    const int shift = normalization_shift(a, b, c, d);
    const point_t normalized_a = normalize(a, shift);
    const point_t normalized_b = normalize(b, shift);
    const point_t normalized_c = normalize(c, shift);
    const point_t normalized_d = normalize(d, shift);
    const double adx = normalized_a.x - normalized_d.x;
    const double ady = normalized_a.y - normalized_d.y;
    const double bdx = normalized_b.x - normalized_d.x;
    const double bdy = normalized_b.y - normalized_d.y;
    const double cdx = normalized_c.x - normalized_d.x;
    const double cdy = normalized_c.y - normalized_d.y;
    const double abdet = adx * bdy - bdx * ady;
    const double bcdet = bdx * cdy - cdx * bdy;
    const double cadet = cdx * ady - adx * cdy;
    const double alift = adx * adx + ady * ady;
    const double blift = bdx * bdx + bdy * bdy;
    const double clift = cdx * cdx + cdy * cdy;
    double det = alift * bcdet + blift * cadet + clift * abdet;
    if (orientation == predicate_sign::negative) det = -det;

    const double permanent = (std::abs(bdx * cdy) + std::abs(cdx * bdy)) * alift +
                             (std::abs(cdx * ady) + std::abs(adx * cdy)) * blift +
                             (std::abs(adx * bdy) + std::abs(bdx * ady)) * clift;
    constexpr double epsilon = std::numeric_limits<double>::epsilon();
    const double error_bound = (10.0 + 96.0 * epsilon) * epsilon * permanent;
    if (det > error_bound) return predicate_sign::positive;
    if (det < -error_bound) return predicate_sign::negative;
    // an unresolved incircle sign keeps the current edge, the stable cocircular policy
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

inline double signed_area(const std::vector<point_t>& points, const std::vector<int>& ring) {
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
        throw std::invalid_argument("Delaunay boundary has numerically ambiguous signed area.");
    }
    return 0.5 * sum;
}

inline double cell_area(const std::vector<point_t>& points, const cell_t& cell) {
    const int shift = normalization_shift(points[cell[0]], points[cell[1]], points[cell[2]]);
    const point_t a = normalize(points[cell[0]], shift);
    const point_t b = normalize(points[cell[1]], shift);
    const point_t c = normalize(points[cell[2]], shift);
    const double normalized_area = 0.5 * std::abs((a.x - c.x) * (b.y - c.y) - (a.y - c.y) * (b.x - c.x));
    return std::scalbn(normalized_area, -2 * shift);
}

inline cell_t make_cell(int a, int b, int c, const std::vector<point_t>& points) {
    const predicate_sign sign = orient2d(points[a], points[b], points[c]);
    if (sign == predicate_sign::zero) {
        throw std::runtime_error(
          "Delaunay triangulation attempted to create a zero-area or numerically ambiguous cell.");
    }
    return sign == predicate_sign::positive ? cell_t {a, b, c} : cell_t {a, c, b};
}

inline cell_t canonical_cell(cell_t cell) {
    const int min_pos = int(std::min_element(cell.begin(), cell.end()) - cell.begin());
    if (min_pos == 1) return {cell[1], cell[2], cell[0]};
    if (min_pos == 2) return {cell[2], cell[0], cell[1]};
    return cell;
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
            throw std::invalid_argument("Delaunay input contains duplicate points.");
        }
    }
}

inline std::vector<int> convex_hull(const std::vector<point_t>& points) {
    std::vector<int> order(points.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) { return less(points[a], points[b]); });

    std::vector<int> lower;
    for (int id : order) {
        while (lower.size() >= 2 && orient2d(points[lower[lower.size() - 2]], points[lower.back()], points[id]) !=
                                      predicate_sign::positive) {
            lower.pop_back();
        }
        lower.push_back(id);
    }
    std::vector<int> upper;
    for (auto it = order.rbegin(); it != order.rend(); ++it) {
        const int id = *it;
        while (upper.size() >= 2 && orient2d(points[upper[upper.size() - 2]], points[upper.back()], points[id]) !=
                                      predicate_sign::positive) {
            upper.pop_back();
        }
        upper.push_back(id);
    }
    lower.pop_back();
    upper.pop_back();
    lower.insert(lower.end(), upper.begin(), upper.end());
    if (lower.size() < 3) {
        throw std::invalid_argument("Delaunay input points are collinear or numerically ambiguous.");
    }
    return lower;
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

inline bool in_ccw_triangle(const std::vector<point_t>& points, const cell_t& cell, const point_t& p) {
    for (int i = 0; i < 3; ++i) {
        if (orient2d(points[cell[i]], points[cell[(i + 1) % 3]], p) == predicate_sign::negative) return false;
    }
    return true;
}

inline void validate_boundary(const std::vector<point_t>& points) {
    if (points.size() < 3) { throw std::invalid_argument("Delaunay boundary needs at least three points."); }
    validate_unique(points);
    if (equal(points.front(), points.back())) {
        throw std::invalid_argument("Delaunay boundary must be an unclosed ring.");
    }
    const int n = points.size();
    for (int i = 0; i < n; ++i) {
        if (orient2d(points[(i + n - 1) % n], points[i], points[(i + 1) % n]) == predicate_sign::zero) {
            throw std::invalid_argument("Delaunay boundary contains consecutive collinear points.");
        }
    }
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            if (j == i + 1 || (i == 0 && j == n - 1)) continue;
            if (segments_intersect(points[i], points[(i + 1) % n], points[j], points[(j + 1) % n])) {
                throw std::invalid_argument("Delaunay boundary must be a simple ring.");
            }
        }
    }
}

inline std::vector<cell_t> ear_clip(const std::vector<point_t>& points, std::vector<int> ring) {
    if (signed_area(points, ring) < 0.0) std::reverse(ring.begin(), ring.end());
    const auto first =
      std::min_element(ring.begin(), ring.end(), [&](int a, int b) { return less(points[a], points[b]); });
    std::rotate(ring.begin(), first, ring.end());

    std::vector<cell_t> cells;
    cells.reserve(ring.size() - 2);
    while (ring.size() > 3) {
        bool found = false;
        for (int i = 0, n = ring.size(); i < n; ++i) {
            const int prev = ring[(i + n - 1) % n];
            const int curr = ring[i];
            const int next = ring[(i + 1) % n];
            if (orient2d(points[prev], points[curr], points[next]) != predicate_sign::positive) continue;
            const cell_t candidate {prev, curr, next};
            bool contains = false;
            for (int id : ring) {
                if (id != prev && id != curr && id != next && in_ccw_triangle(points, candidate, points[id])) {
                    contains = true;
                    break;
                }
            }
            if (contains) continue;
            cells.push_back(candidate);
            ring.erase(ring.begin() + i);
            found = true;
            break;
        }
        if (!found) { throw std::runtime_error("Delaunay ear clipping could not find a valid ear."); }
    }
    cells.push_back(make_cell(ring[0], ring[1], ring[2], points));
    return cells;
}

inline std::vector<cell_t> ear_clip(const std::vector<point_t>& points, int boundary_size) {
    std::vector<int> ring(boundary_size);
    std::iota(ring.begin(), ring.end(), 0);
    return ear_clip(points, std::move(ring));
}

struct edge_info {
    int first = -1;
    int second = -1;
};

inline std::map<edge_t, edge_info> edge_adjacency(const std::vector<cell_t>& cells) {
    std::map<edge_t, edge_info> adjacency;
    for (int i = 0; i < int(cells.size()); ++i) {
        for (int j = 0; j < 3; ++j) {
            edge_info& info = adjacency[edge(cells[i][j], cells[i][(j + 1) % 3])];
            if (info.first == -1) {
                info.first = i;
            } else if (info.second == -1) {
                info.second = i;
            } else {
                throw std::runtime_error("Delaunay triangulation contains a non-manifold edge.");
            }
        }
    }
    return adjacency;
}

inline int opposite(const cell_t& cell, const edge_t& edge_) {
    for (int id : cell) {
        if (id != edge_[0] && id != edge_[1]) return id;
    }
    throw std::runtime_error("Delaunay edge has no opposite cell vertex.");
}

inline bool contains(const cell_t& cell, int node) { return cell[0] == node || cell[1] == node || cell[2] == node; }

inline bool properly_crosses(const edge_t& first, const edge_t& second, const std::vector<point_t>& points) {
    if (first[0] == second[0] || first[0] == second[1] || first[1] == second[0] || first[1] == second[1]) {
        return false;
    }
    const point_t& a = points[first[0]];
    const point_t& b = points[first[1]];
    const point_t& c = points[second[0]];
    const point_t& d = points[second[1]];
    if (
      std::max(a.x, b.x) < std::min(c.x, d.x) || std::max(c.x, d.x) < std::min(a.x, b.x) ||
      std::max(a.y, b.y) < std::min(c.y, d.y) || std::max(c.y, d.y) < std::min(a.y, b.y)) {
        return false;
    }
    const predicate_sign side_c = orient2d(a, b, c);
    const predicate_sign side_d = orient2d(a, b, d);
    const predicate_sign side_a = orient2d(c, d, a);
    const predicate_sign side_b = orient2d(c, d, b);
    if (
      side_c == predicate_sign::zero || side_d == predicate_sign::zero || side_a == predicate_sign::zero ||
      side_b == predicate_sign::zero) {
        if (on_segment(a, b, c) || on_segment(a, b, d) || on_segment(c, d, a) || on_segment(c, d, b)) {
            throw std::runtime_error("Delaunay constraint recovery encountered a nonendpoint segment touch.");
        }
        return false;
    }
    return side_c != side_d && side_a != side_b;
}

inline std::vector<int> trace_cavity_chain(
  int start, int finish, int first, const std::map<int, std::set<int>>& neighbors, std::size_t edge_count) {
    std::vector<int> chain {start};
    int previous = start;
    int current = first;
    while (true) {
        chain.push_back(current);
        if (current == finish) return chain;
        const auto found = neighbors.find(current);
        if (found == neighbors.end() || found->second.size() != 2) {
            throw std::runtime_error("Delaunay constraint cavity boundary is not a simple ring.");
        }
        if (!found->second.contains(previous)) {
            throw std::runtime_error("Delaunay constraint cavity boundary lost its preceding vertex.");
        }
        auto next = found->second.begin();
        if (*next == previous) ++next;
        previous = current;
        current = *next;
        if (chain.size() > edge_count) {
            throw std::runtime_error("Delaunay constraint cavity boundary did not reach its endpoint.");
        }
    }
}

inline void recover_constraint(
  const edge_t& constraint, std::vector<cell_t>& cells, const std::vector<point_t>& points,
  const std::set<edge_t>& fixed_constraints) {
    auto adjacency = edge_adjacency(cells);
    if (adjacency.contains(constraint)) return;

    int current = -1;
    for (int i = 0; i < int(cells.size()); ++i) {
        if (!contains(cells[i], constraint[0])) continue;
        std::array<int, 2> opposite_nodes {};
        int count = 0;
        for (int node : cells[i]) {
            if (node != constraint[0]) opposite_nodes[count++] = node;
        }
        const edge_t opposite_edge = edge(opposite_nodes[0], opposite_nodes[1]);
        if (!properly_crosses(constraint, opposite_edge, points)) continue;
        if (current != -1) { throw std::runtime_error("Delaunay constraint recovery found multiple starting cells."); }
        current = i;
    }
    if (current == -1) { throw std::runtime_error("Delaunay constraint recovery could not find a starting cell."); }

    std::set<int> strip;
    edge_t entry {-1, -1};
    while (true) {
        if (!strip.insert(current).second) {
            throw std::runtime_error("Delaunay constraint recovery revisited a triangle.");
        }
        if (contains(cells[current], constraint[1])) break;
        std::optional<edge_t> exit;
        for (int i = 0; i < 3; ++i) {
            const edge_t candidate = edge(cells[current][i], cells[current][(i + 1) % 3]);
            if (candidate == entry || !properly_crosses(constraint, candidate, points)) continue;
            if (exit.has_value()) {
                throw std::runtime_error("Delaunay constraint recovery found an ambiguous triangle exit.");
            }
            exit = candidate;
        }
        if (!exit.has_value()) { throw std::runtime_error("Delaunay constraint recovery found no triangle exit."); }
        const auto found = adjacency.find(*exit);
        if (found == adjacency.end() || found->second.second == -1) {
            throw std::runtime_error("Delaunay constraint recovery left the triangulated domain.");
        }
        current = found->second.first == current ? found->second.second : found->second.first;
        entry = *exit;
    }

    std::map<edge_t, int> cavity_edge_counts;
    for (int cell : strip) {
        for (int i = 0; i < 3; ++i) ++cavity_edge_counts[edge(cells[cell][i], cells[cell][(i + 1) % 3])];
    }
    std::map<int, std::set<int>> cavity_neighbors;
    std::size_t boundary_edges = 0;
    for (const auto& [edge_, count] : cavity_edge_counts) {
        if (count != 1) continue;
        cavity_neighbors[edge_[0]].insert(edge_[1]);
        cavity_neighbors[edge_[1]].insert(edge_[0]);
        ++boundary_edges;
    }
    const auto start = cavity_neighbors.find(constraint[0]);
    if (start == cavity_neighbors.end() || start->second.size() != 2) {
        throw std::runtime_error("Delaunay constraint cavity does not contain its first endpoint.");
    }
    const auto first_neighbor = start->second.begin();
    const std::vector<int> first_chain =
      trace_cavity_chain(constraint[0], constraint[1], *first_neighbor, cavity_neighbors, boundary_edges);
    const std::vector<int> second_chain =
      trace_cavity_chain(constraint[0], constraint[1], *std::next(first_neighbor), cavity_neighbors, boundary_edges);
    if (
      first_chain.size() < 3 || second_chain.size() < 3 ||
      first_chain.size() + second_chain.size() != boundary_edges + 2) {
        throw std::runtime_error("Delaunay constraint cavity produced invalid boundary chains.");
    }

    std::vector<cell_t> updated;
    updated.reserve(cells.size() - strip.size() + boundary_edges - 2);
    for (int i = 0; i < int(cells.size()); ++i) {
        if (!strip.contains(i)) updated.push_back(cells[i]);
    }
    const auto first_cells = ear_clip(points, first_chain);
    const auto second_cells = ear_clip(points, second_chain);
    updated.insert(updated.end(), first_cells.begin(), first_cells.end());
    updated.insert(updated.end(), second_cells.begin(), second_cells.end());
    cells = std::move(updated);

    adjacency = edge_adjacency(cells);
    if (!adjacency.contains(constraint)) {
        throw std::runtime_error("Delaunay constraint recovery did not produce the requested edge.");
    }
    for (const edge_t& fixed : fixed_constraints) {
        if (!adjacency.contains(fixed)) {
            throw std::runtime_error("Delaunay constraint recovery removed an earlier constraint.");
        }
    }
}

inline bool should_flip(
  const edge_t& edge_, const edge_info& info, const std::vector<point_t>& points, const std::vector<cell_t>& cells) {
    if (info.second == -1) return false;
    const int c = opposite(cells[info.first], edge_);
    const int d = opposite(cells[info.second], edge_);
    const predicate_sign side_c = orient2d(points[c], points[d], points[edge_[0]]);
    const predicate_sign side_d = orient2d(points[c], points[d], points[edge_[1]]);
    if (side_c == predicate_sign::zero || side_d == predicate_sign::zero || side_c == side_d) return false;
    return incircle(points[edge_[0]], points[edge_[1]], points[c], points[d]) == predicate_sign::positive;
}

inline void
legalize(std::vector<cell_t>& cells, const std::vector<point_t>& points, const std::set<edge_t>& constraints = {}) {
    // ponytail: rebuild adjacency after a flip; use a local edge queue if profiling shows this dominates
    const std::size_t max_flips = std::max<std::size_t>(128, 32 * cells.size() * cells.size());
    std::size_t flips = 0;
    while (true) {
        const auto adjacency = edge_adjacency(cells);
        bool changed = false;
        for (const auto& [edge_, info] : adjacency) {
            if (constraints.contains(edge_)) continue;
            if (!should_flip(edge_, info, points, cells)) continue;
            const int c = opposite(cells[info.first], edge_);
            const int d = opposite(cells[info.second], edge_);
            cells[info.first] = make_cell(c, d, edge_[0], points);
            cells[info.second] = make_cell(d, c, edge_[1], points);
            if (++flips > max_flips) { throw std::runtime_error("Delaunay edge legalization did not converge."); }
            changed = true;
            break;
        }
        if (!changed) return;
    }
}

inline void insert_node(
  int node, bool allow_boundary_split, std::vector<point_t>& points, std::vector<cell_t>& cells,
  const std::set<edge_t>& constraints = {}) {
    const point_t& p = points[node];
    const auto adjacency = edge_adjacency(cells);
    for (const auto& [edge_, info] : adjacency) {
        if (!on_segment(points[edge_[0]], points[edge_[1]], p)) continue;
        if (info.second == -1 && !allow_boundary_split) {
            throw std::invalid_argument("Delaunay interior point lies on the domain boundary.");
        }
        std::vector<cell_t> updated;
        updated.reserve(cells.size() + (info.second == -1 ? 1 : 2));
        for (int i = 0; i < int(cells.size()); ++i) {
            if (i != info.first && i != info.second) updated.push_back(cells[i]);
        }
        const int first_opposite = opposite(cells[info.first], edge_);
        updated.push_back(make_cell(edge_[0], node, first_opposite, points));
        updated.push_back(make_cell(node, edge_[1], first_opposite, points));
        if (info.second != -1) {
            const int second_opposite = opposite(cells[info.second], edge_);
            updated.push_back(make_cell(edge_[1], node, second_opposite, points));
            updated.push_back(make_cell(node, edge_[0], second_opposite, points));
        }
        cells = std::move(updated);
        legalize(cells, points, constraints);
        return;
    }

    for (int i = 0; i < int(cells.size()); ++i) {
        if (!in_ccw_triangle(points, cells[i], p)) continue;
        const cell_t old = cells[i];
        cells[i] = make_cell(old[0], old[1], node, points);
        cells.push_back(make_cell(old[1], old[2], node, points));
        cells.push_back(make_cell(old[2], old[0], node, points));
        legalize(cells, points, constraints);
        return;
    }
    throw std::runtime_error("Delaunay point insertion could not locate its containing cell.");
}

inline void refine(
  std::vector<point_t>& points, std::vector<cell_t>& cells, const std::optional<DelaunayRefinement>& refinement,
  const std::set<edge_t>& constraints = {}) {
    if (!refinement.has_value()) return;
    if (!std::isfinite(refinement->max_area) || refinement->max_area <= 0.0) {
        throw std::invalid_argument("Delaunay refinement max_area must be finite and positive.");
    }
    if (refinement->max_insertions < 0) {
        throw std::invalid_argument("Delaunay refinement max_insertions cannot be negative.");
    }

    int insertions = 0;
    while (true) {
        int worst = -1;
        double worst_area = refinement->max_area;
        cell_t worst_key {};
        for (int i = 0; i < int(cells.size()); ++i) {
            const double area = cell_area(points, cells[i]);
            const cell_t key = canonical_cell(cells[i]);
            if (area > worst_area || (area == worst_area && worst != -1 && key < worst_key)) {
                worst = i;
                worst_area = area;
                worst_key = key;
            }
        }
        if (worst == -1) return;
        if (insertions == refinement->max_insertions) {
            std::ostringstream message;
            message << "Delaunay refinement reached max_insertions=" << refinement->max_insertions
                    << " with maximum cell area " << worst_area << ".";
            throw std::runtime_error(message.str());
        }
        const cell_t cell = cells[worst];
        const int shift = normalization_shift(points[cell[0]], points[cell[1]], points[cell[2]]);
        const point_t a = normalize(points[cell[0]], shift);
        const point_t b = normalize(points[cell[1]], shift);
        const point_t c = normalize(points[cell[2]], shift);
        point_t centroid {std::scalbn((a.x + b.x + c.x) / 3.0, -shift), std::scalbn((a.y + b.y + c.y) / 3.0, -shift)};
        if (
          !std::isfinite(centroid.x) || !std::isfinite(centroid.y) ||
          std::any_of(points.begin(), points.end(), [&](const point_t& point) { return equal(point, centroid); })) {
            throw std::runtime_error("Delaunay refinement cannot represent another distinct finite centroid.");
        }
        points.push_back(centroid);
        insert_node(points.size() - 1, false, points, cells, constraints);
        ++insertions;
    }
}

inline Triangulation<2, 2> make_triangulation(
  const std::vector<point_t>& points, std::vector<cell_t> cells, const std::vector<edge_t>& required_boundary = {}) {
    for (cell_t& cell : cells) cell = canonical_cell(cell);
    std::sort(cells.begin(), cells.end());
    if (std::adjacent_find(cells.begin(), cells.end()) != cells.end()) {
        throw std::runtime_error("Delaunay triangulation contains duplicate cells.");
    }
    const auto adjacency = edge_adjacency(cells);
    for (const edge_t& edge_ : required_boundary) {
        const auto it = adjacency.find(edge_);
        if (it == adjacency.end() || it->second.second != -1) {
            throw std::runtime_error("Delaunay triangulation did not preserve a constrained boundary edge.");
        }
    }
    if (!required_boundary.empty()) {
        const std::size_t boundary_edges = std::count_if(
          adjacency.begin(), adjacency.end(), [](const auto& entry) { return entry.second.second == -1; });
        if (boundary_edges != required_boundary.size()) {
            throw std::runtime_error("Delaunay triangulation contains an unexpected boundary edge.");
        }
    }

    Matrix<double, Dynamic, Dynamic> nodes(points.size(), 2);
    Matrix<int, Dynamic, Dynamic> cell_matrix(cells.size(), 3);
    Matrix<int, Dynamic, Dynamic> boundary(points.size(), 1);
    for (int i = 0; i < int(points.size()); ++i) {
        nodes(i, 0) = points[i].x;
        nodes(i, 1) = points[i].y;
        boundary(i, 0) = 0;
    }
    for (int i = 0; i < int(cells.size()); ++i) {
        for (int j = 0; j < 3; ++j) cell_matrix(i, j) = cells[i][j];
    }
    for (const auto& [edge_, info] : adjacency) {
        if (info.second == -1) {
            boundary(edge_[0], 0) = 1;
            boundary(edge_[1], 0) = 1;
        }
    }
    return Triangulation<2, 2>(nodes, cell_matrix, boundary);
}

inline std::vector<int> sorted_ids(const std::vector<point_t>& points, std::vector<int> ids) {
    std::sort(ids.begin(), ids.end(), [&](int a, int b) { return less(points[a], points[b]); });
    return ids;
}

inline std::vector<edge_t> ring_edges(const std::vector<int>& ring) {
    std::vector<edge_t> result;
    result.reserve(ring.size());
    for (int i = 0; i < int(ring.size()); ++i) result.push_back(edge(ring[i], ring[(i + 1) % ring.size()]));
    return result;
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

struct domain_input_t {
    std::vector<point_t> points;
    std::vector<int> outer;
    std::vector<std::vector<int>> holes;
    std::vector<edge_t> boundary_edges;
    std::vector<edge_t> hole_edges;
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

inline domain_input_t read_domain(const PlanarDomain& domain, const Matrix<double, Dynamic, Dynamic>& interior_matrix) {
    domain_input_t result;
    result.outer = append_ring(result.points, domain.outer, "Delaunay outer boundary");
    result.holes.reserve(domain.holes.size());
    for (const auto& hole : domain.holes) {
        result.holes.push_back(append_ring(result.points, hole, "Delaunay hole boundary"));
    }
    validate_unique(result.points);

    for (const auto& hole : result.holes) {
        if (rings_intersect(result.points, result.outer, hole)) {
            throw std::invalid_argument("Delaunay hole boundary must not touch or intersect the outer boundary.");
        }
        for (int node : hole) {
            if (locate_in_polygon(result.points, result.outer, result.points[node]) != point_location::inside) {
                throw std::invalid_argument("Delaunay holes must lie strictly inside the outer boundary.");
            }
        }
    }
    for (int i = 0; i < int(result.holes.size()); ++i) {
        for (int j = i + 1; j < int(result.holes.size()); ++j) {
            if (rings_intersect(result.points, result.holes[i], result.holes[j])) {
                throw std::invalid_argument("Delaunay holes must be mutually disjoint and non-touching.");
            }
            if (
              locate_in_polygon(result.points, result.holes[i], result.points[result.holes[j][0]]) !=
                point_location::outside ||
              locate_in_polygon(result.points, result.holes[j], result.points[result.holes[i][0]]) !=
                point_location::outside) {
                throw std::invalid_argument("Delaunay holes must not be nested.");
            }
        }
    }

    std::vector<point_t> interior = read_points(interior_matrix, "Delaunay interior points");
    for (const point_t& point : interior) {
        if (locate_in_polygon(result.points, result.outer, point) != point_location::inside) {
            throw std::invalid_argument("Delaunay interior points must lie strictly inside the outer boundary.");
        }
        for (const auto& hole : result.holes) {
            if (locate_in_polygon(result.points, hole, point) != point_location::outside) {
                throw std::invalid_argument("Delaunay interior points must lie strictly outside every hole.");
            }
        }
    }
    result.points.insert(result.points.end(), interior.begin(), interior.end());
    validate_unique(result.points);

    result.boundary_edges = ring_edges(result.outer);
    for (const auto& hole : result.holes) {
        std::vector<edge_t> edges = ring_edges(hole);
        result.hole_edges.insert(result.hole_edges.end(), edges.begin(), edges.end());
        result.boundary_edges.insert(result.boundary_edges.end(), edges.begin(), edges.end());
    }
    const auto geometrically_less = [&](const edge_t& lhs, const edge_t& rhs) {
        point_t lhs_first = result.points[lhs[0]];
        point_t lhs_second = result.points[lhs[1]];
        point_t rhs_first = result.points[rhs[0]];
        point_t rhs_second = result.points[rhs[1]];
        if (less(lhs_second, lhs_first)) std::swap(lhs_first, lhs_second);
        if (less(rhs_second, rhs_first)) std::swap(rhs_first, rhs_second);
        return less(lhs_first, rhs_first) || (equal(lhs_first, rhs_first) && less(lhs_second, rhs_second));
    };
    std::sort(result.hole_edges.begin(), result.hole_edges.end(), geometrically_less);
    return result;
}

inline void remove_hole_interiors(
  std::vector<cell_t>& cells, const std::vector<point_t>& points, std::vector<std::vector<int>> holes,
  const std::set<edge_t>& constraints) {
    const auto adjacency = edge_adjacency(cells);
    std::queue<int> pending;
    std::vector<bool> excluded(cells.size(), false);
    for (auto& hole : holes) {
        if (signed_area(points, hole) > 0.0) std::reverse(hole.begin(), hole.end());
        for (int i = 0; i < int(hole.size()); ++i) {
            const int a = hole[i];
            const int b = hole[(i + 1) % hole.size()];
            const auto found = adjacency.find(edge(a, b));
            if (found == adjacency.end() || found->second.second == -1) {
                throw std::runtime_error("Delaunay hole boundary is not an internal constrained edge.");
            }
            int right = -1;
            for (const int cell : {found->second.first, found->second.second}) {
                const int other = opposite(cells[cell], edge(a, b));
                const predicate_sign side = orient2d(points[a], points[b], points[other]);
                if (side == predicate_sign::zero) {
                    throw std::runtime_error("Delaunay hole boundary has ambiguous adjacent geometry.");
                }
                if (side == predicate_sign::negative) {
                    if (right != -1) {
                        throw std::runtime_error("Delaunay hole boundary has two triangles on its interior side.");
                    }
                    right = cell;
                }
            }
            if (right == -1) { throw std::runtime_error("Delaunay hole boundary has no interior-side triangle."); }
            pending.push(right);
        }
    }

    while (!pending.empty()) {
        const int cell = pending.front();
        pending.pop();
        if (excluded[cell]) continue;
        excluded[cell] = true;
        for (int i = 0; i < 3; ++i) {
            const edge_t edge_ = edge(cells[cell][i], cells[cell][(i + 1) % 3]);
            if (constraints.contains(edge_)) continue;
            const edge_info& info = adjacency.at(edge_);
            const int next = info.first == cell ? info.second : info.first;
            if (next != -1 && !excluded[next]) pending.push(next);
        }
    }

    std::vector<cell_t> retained;
    retained.reserve(cells.size());
    for (int i = 0; i < int(cells.size()); ++i) {
        if (!excluded[i]) retained.push_back(cells[i]);
    }
    if (retained.empty()) { throw std::runtime_error("Delaunay hole removal discarded the entire domain."); }
    cells = std::move(retained);
}

}   // namespace delaunay_2d
}   // namespace internals

/**
 * @brief Triangulate a finite, unique planar point set over its convex hull.
 *
 * Input node order is preserved. Cells are counter-clockwise and returned in deterministic index order. Predicates
 * are power-of-two normalized and use certified floating-point filters; configurations whose sign remains ambiguous
 * at machine precision are treated as degenerate.
 */
inline Triangulation<2, 2> delaunay(
  const Matrix<double, Dynamic, Dynamic>& point_matrix, std::optional<DelaunayRefinement> refinement = std::nullopt) {
    using namespace internals::delaunay_2d;
    std::vector<point_t> points = read_points(point_matrix, "Delaunay points");
    if (points.size() < 3) { throw std::invalid_argument("Delaunay triangulation needs at least three points."); }
    validate_unique(points);
    const std::vector<int> hull = convex_hull(points);

    std::vector<cell_t> cells;
    cells.reserve(2 * points.size());
    for (int i = 1; i + 1 < int(hull.size()); ++i) {
        cells.push_back(make_cell(hull[0], hull[i], hull[i + 1], points));
    }
    legalize(cells, points);

    std::vector<bool> on_hull(points.size(), false);
    for (int id : hull) on_hull[id] = true;
    std::vector<int> remaining;
    for (int i = 0; i < int(points.size()); ++i) {
        if (!on_hull[i]) remaining.push_back(i);
    }
    for (int id : sorted_ids(points, std::move(remaining))) insert_node(id, true, points, cells);
    refine(points, cells, refinement);
    return make_triangulation(points, std::move(cells));
}

/**
 * @brief Triangulate a simple polygonal ring with optional strictly interior sites.
 *
 * The boundary is an unclosed clockwise or counter-clockwise ring with no consecutive collinear vertices. Its edges
 * are preserved as constraints. Holes, internal constraint segments, and points on the boundary are not supported.
 */
inline Triangulation<2, 2> constrained_delaunay(
  const Matrix<double, Dynamic, Dynamic>& boundary_matrix, const Matrix<double, Dynamic, Dynamic>& interior_matrix,
  std::optional<DelaunayRefinement> refinement = std::nullopt) {
    using namespace internals::delaunay_2d;
    std::vector<point_t> points = read_points(boundary_matrix, "Delaunay boundary");
    const int boundary_size = points.size();
    validate_boundary(points);
    std::vector<point_t> interior = read_points(interior_matrix, "Delaunay interior points");
    points.insert(points.end(), interior.begin(), interior.end());
    validate_unique(points);

    std::vector<int> boundary_ring(boundary_size);
    std::iota(boundary_ring.begin(), boundary_ring.end(), 0);
    for (int i = boundary_size; i < int(points.size()); ++i) {
        if (locate_in_polygon(points, boundary_ring, points[i]) != point_location::inside) {
            throw std::invalid_argument("Delaunay interior points must lie strictly inside the boundary.");
        }
    }

    std::vector<cell_t> cells = ear_clip(points, boundary_size);
    legalize(cells, points);
    std::vector<int> interior_ids(points.size() - boundary_size);
    std::iota(interior_ids.begin(), interior_ids.end(), boundary_size);
    for (int id : sorted_ids(points, std::move(interior_ids))) insert_node(id, false, points, cells);
    refine(points, cells, refinement);

    std::vector<edge_t> boundary_edges;
    boundary_edges.reserve(boundary_size);
    for (int i = 0; i < boundary_size; ++i) boundary_edges.push_back(edge(i, (i + 1) % boundary_size));
    return make_triangulation(points, std::move(cells), boundary_edges);
}

inline Triangulation<2, 2> constrained_delaunay(
  const Matrix<double, Dynamic, Dynamic>& boundary, std::optional<DelaunayRefinement> refinement = std::nullopt) {
    return constrained_delaunay(boundary, Matrix<double, Dynamic, Dynamic>(0, 2), refinement);
}

/**
 * @brief Triangulate one simple outer ring with disjoint simple hole rings and optional strictly interior sites.
 *
 * Ring orientation and cyclic starting vertices carry no semantics. Every supplied ring edge is preserved as a
 * topological boundary edge. Node order is the outer ring, each hole in supplied order, the interior sites, and any
 * refinement nodes. Holes must lie strictly inside the outer ring and must be mutually disjoint and non-nested.
 */
inline Triangulation<2, 2> constrained_delaunay(
  const PlanarDomain& domain, const Matrix<double, Dynamic, Dynamic>& interior_matrix,
  std::optional<DelaunayRefinement> refinement = std::nullopt) {
    if (domain.holes.empty()) return constrained_delaunay(domain.outer, interior_matrix, refinement);
    using namespace internals::delaunay_2d;
    domain_input_t input = read_domain(domain, interior_matrix);

    std::vector<cell_t> cells = ear_clip(input.points, input.outer);
    legalize(cells, input.points);
    std::vector<int> inserted_ids(input.points.size() - input.outer.size());
    std::iota(inserted_ids.begin(), inserted_ids.end(), input.outer.size());
    for (int id : sorted_ids(input.points, std::move(inserted_ids))) { insert_node(id, false, input.points, cells); }

    std::set<edge_t> constraints(input.boundary_edges.begin(), input.boundary_edges.begin() + input.outer.size());
    for (const edge_t& edge_ : input.hole_edges) {
        recover_constraint(edge_, cells, input.points, constraints);
        constraints.insert(edge_);
    }
    remove_hole_interiors(cells, input.points, input.holes, constraints);
    legalize(cells, input.points, constraints);
    refine(input.points, cells, refinement, constraints);
    return make_triangulation(input.points, std::move(cells), input.boundary_edges);
}

inline Triangulation<2, 2>
constrained_delaunay(const PlanarDomain& domain, std::optional<DelaunayRefinement> refinement = std::nullopt) {
    return constrained_delaunay(domain, Matrix<double, Dynamic, Dynamic>(0, 2), refinement);
}

}   // namespace fdapde

#endif   // __FDAPDE_GEOMETRY_DELAUNAY_H__
