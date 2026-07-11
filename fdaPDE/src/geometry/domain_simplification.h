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

#ifndef __FDAPDE_GEOMETRY_DOMAIN_SIMPLIFICATION_H__
#define __FDAPDE_GEOMETRY_DOMAIN_SIMPLIFICATION_H__

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "header_check.h"

namespace fdapde {
namespace internals {
namespace domain_simplification {

using planar::less;
using planar::locate_in_polygon;
using planar::normalization_shift;
using planar::normalize;
using planar::orient2d;
using planar::point_location;
using planar::point_t;
using planar::predicate_sign;
using planar::read_points;
using planar::segments_intersect;
using planar::signed_area;
using planar::validate_planar_domain;

inline Matrix<double, Dynamic, Dynamic>
canonical_ring(const std::vector<point_t>& points, std::vector<int> ring, bool counter_clockwise) {
    if ((signed_area(points, ring) > 0.0) != counter_clockwise) std::reverse(ring.begin(), ring.end());
    const auto first =
      std::min_element(ring.begin(), ring.end(), [&](int a, int b) { return less(points[a], points[b]); });
    std::rotate(ring.begin(), first, ring.end());
    Matrix<double, Dynamic, Dynamic> result(ring.size(), 2);
    for (int i = 0; i < int(ring.size()); ++i) {
        result(i, 0) = points[ring[i]].x;
        result(i, 1) = points[ring[i]].y;
    }
    return result;
}

inline bool matrix_less(const Matrix<double, Dynamic, Dynamic>& first, const Matrix<double, Dynamic, Dynamic>& second) {
    const int common = std::min(first.rows(), second.rows());
    for (int i = 0; i < common; ++i) {
        if (first(i, 0) != second(i, 0)) return first(i, 0) < second(i, 0);
        if (first(i, 1) != second(i, 1)) return first(i, 1) < second(i, 1);
    }
    return first.rows() < second.rows();
}

inline PlanarDomain canonical_domain(const PlanarDomain& domain) {
    planar::validated_domain_t validated = validate_planar_domain(domain);
    PlanarDomain result {.outer = canonical_ring(validated.points, validated.outer, true), .holes = {}};
    result.holes.reserve(validated.holes.size());
    for (auto& hole : validated.holes) {
        result.holes.push_back(canonical_ring(validated.points, std::move(hole), false));
    }
    std::sort(result.holes.begin(), result.holes.end(), matrix_less);
    return result;
}

struct ring_state_t {
    std::vector<point_t> original;
    std::vector<int> previous;
    std::vector<int> next;
    std::vector<unsigned> generation;
    std::vector<bool> active;
    int size;
    bool outer;
};

inline ring_state_t make_state(const Matrix<double, Dynamic, Dynamic>& ring, bool outer) {
    ring_state_t result;
    result.original = read_points(ring, "Polygon simplification ring");
    result.previous.resize(ring.rows());
    result.next.resize(ring.rows());
    result.generation.assign(ring.rows(), 0);
    result.active.assign(ring.rows(), true);
    result.size = ring.rows();
    result.outer = outer;
    for (int i = 0; i < ring.rows(); ++i) {
        result.previous[i] = (i + ring.rows() - 1) % ring.rows();
        result.next[i] = (i + 1) % ring.rows();
    }
    return result;
}

inline std::vector<int> active_ring(const ring_state_t& ring, int skipped = -1) {
    std::vector<int> result;
    result.reserve(ring.size - (skipped == -1 ? 0 : 1));
    int first = -1;
    for (int i = 0; i < int(ring.active.size()); ++i) {
        if (ring.active[i] && i != skipped) {
            first = i;
            break;
        }
    }
    if (first == -1) return result;
    int current = first;
    do {
        if (current != skipped) result.push_back(current);
        current = ring.next[current];
        if (current == skipped) current = ring.next[current];
    } while (current != first);
    return result;
}

inline int first_active(const ring_state_t& ring) {
    for (int i = 0; i < int(ring.active.size()); ++i) {
        if (ring.active[i]) return i;
    }
    throw std::runtime_error("Polygon simplification lost an entire ring.");
}

inline double point_segment_distance(const point_t& point, const point_t& first, const point_t& second) {
    const int shift = normalization_shift(point, first, second);
    const point_t p = normalize(point, shift);
    const point_t a = normalize(first, shift);
    const point_t b = normalize(second, shift);
    const double dx = b.x - a.x;
    const double dy = b.y - a.y;
    const double length_squared = dx * dx + dy * dy;
    if (length_squared == 0.0) return std::numeric_limits<double>::infinity();
    const double projection = std::clamp(((p.x - a.x) * dx + (p.y - a.y) * dy) / length_squared, 0.0, 1.0);
    const double distance = std::hypot(p.x - (a.x + projection * dx), p.y - (a.y + projection * dy));
    return std::scalbn(distance, -shift);
}

inline double shortcut_error(const ring_state_t& ring, int vertex) {
    const int previous = ring.previous[vertex];
    const int next = ring.next[vertex];
    double error = 0.0;
    int current = previous;
    do {
        error =
          std::max(error, point_segment_distance(ring.original[current], ring.original[previous], ring.original[next]));
        current = (current + 1) % ring.original.size();
    } while (current != (next + 1) % ring.original.size());
    return error;
}

struct candidate_t {
    double error;
    int ring;
    int vertex;
    unsigned generation;

    auto key() const { return std::tuple {error, ring, vertex, generation}; }
};

struct candidate_greater {
    bool operator()(const candidate_t& first, const candidate_t& second) const { return first.key() > second.key(); }
};

using candidate_queue_t = std::priority_queue<candidate_t, std::vector<candidate_t>, candidate_greater>;

inline void push_candidate(
  const std::vector<ring_state_t>& rings, int ring_id, int vertex, double maximum_deviation,
  candidate_queue_t& candidates) {
    const ring_state_t& ring = rings[ring_id];
    if (ring.size <= 3 || !ring.active[vertex]) return;
    const double error = shortcut_error(ring, vertex);
    if (error > maximum_deviation) return;
    candidates.push({error, ring_id, vertex, ring.generation[vertex]});
}

inline point_location locate(const ring_state_t& ring, const point_t& point, int skipped = -1) {
    return locate_in_polygon(ring.original, active_ring(ring, skipped), point);
}

inline bool shortcut_intersects(
  const std::vector<ring_state_t>& rings, int ring_id, int vertex, const point_t& first, const point_t& second) {
    for (int other_id = 0; other_id < int(rings.size()); ++other_id) {
        const ring_state_t& other = rings[other_id];
        const std::vector<int> ids = active_ring(other);
        for (int i = 0; i < int(ids.size()); ++i) {
            const int a = ids[i];
            const int b = ids[(i + 1) % ids.size()];
            if (
              other_id == ring_id && (a == vertex || b == vertex || a == rings[ring_id].previous[vertex] ||
                                      b == rings[ring_id].previous[vertex] || a == rings[ring_id].next[vertex] ||
                                      b == rings[ring_id].next[vertex])) {
                continue;
            }
            if (segments_intersect(first, second, other.original[a], other.original[b])) return true;
        }
    }
    return false;
}

inline bool preserves_topology(const std::vector<ring_state_t>& rings, int ring_id, int vertex) {
    const ring_state_t& ring = rings[ring_id];
    const int previous = ring.previous[vertex];
    const int next = ring.next[vertex];
    const int before_previous = ring.previous[previous];
    const int after_next = ring.next[next];
    if (
      orient2d(ring.original[before_previous], ring.original[previous], ring.original[next]) == predicate_sign::zero ||
      orient2d(ring.original[previous], ring.original[next], ring.original[after_next]) == predicate_sign::zero) {
        return false;
    }
    if (shortcut_intersects(rings, ring_id, vertex, ring.original[previous], ring.original[next])) return false;

    try {
        const double area = signed_area(ring.original, active_ring(ring, vertex));
        if ((area > 0.0) != ring.outer) return false;
    } catch (const std::invalid_argument&) { return false; }

    if (ring.outer) {
        for (int hole = 1; hole < int(rings.size()); ++hole) {
            const point_t& point = rings[hole].original[first_active(rings[hole])];
            if (locate(ring, point, vertex) != point_location::inside) return false;
        }
    } else {
        const point_t& point = ring.original[next];
        if (locate(rings[0], point) != point_location::inside) return false;
        for (int other = 1; other < int(rings.size()); ++other) {
            if (other == ring_id) continue;
            const point_t& other_point = rings[other].original[first_active(rings[other])];
            if (
              locate(ring, other_point, vertex) != point_location::outside ||
              locate(rings[other], point) != point_location::outside) {
                return false;
            }
        }
    }
    return true;
}

inline Matrix<double, Dynamic, Dynamic> matrix(const ring_state_t& ring) {
    const std::vector<int> ids = active_ring(ring);
    Matrix<double, Dynamic, Dynamic> result(ids.size(), 2);
    for (int i = 0; i < int(ids.size()); ++i) {
        result(i, 0) = ring.original[ids[i]].x;
        result(i, 1) = ring.original[ids[i]].y;
    }
    return result;
}

inline PlanarDomain simplify(const PlanarDomain& input, double maximum_deviation) {
    const PlanarDomain domain = canonical_domain(input);
    std::vector<ring_state_t> rings;
    rings.reserve(domain.holes.size() + 1);
    rings.push_back(make_state(domain.outer, true));
    for (const auto& hole : domain.holes) rings.push_back(make_state(hole, false));

    candidate_queue_t candidates;
    for (int ring = 0; ring < int(rings.size()); ++ring) {
        for (int vertex = 0; vertex < int(rings[ring].active.size()); ++vertex) {
            push_candidate(rings, ring, vertex, maximum_deviation, candidates);
        }
    }
    while (!candidates.empty()) {
        const candidate_t candidate = candidates.top();
        candidates.pop();
        ring_state_t& ring = rings[candidate.ring];
        if (
          !ring.active[candidate.vertex] || ring.generation[candidate.vertex] != candidate.generation ||
          ring.size <= 3) {
            continue;
        }
        if (!preserves_topology(rings, candidate.ring, candidate.vertex)) continue;

        const int previous = ring.previous[candidate.vertex];
        const int next = ring.next[candidate.vertex];
        ring.active[candidate.vertex] = false;
        ring.next[previous] = next;
        ring.previous[next] = previous;
        --ring.size;
        ++ring.generation[previous];
        ++ring.generation[next];
        push_candidate(rings, candidate.ring, previous, maximum_deviation, candidates);
        push_candidate(rings, candidate.ring, next, maximum_deviation, candidates);
    }

    PlanarDomain result {.outer = matrix(rings.front()), .holes = {}};
    result.holes.reserve(rings.size() - 1);
    for (int i = 1; i < int(rings.size()); ++i) result.holes.push_back(matrix(rings[i]));
    return canonical_domain(result);
}

}   // namespace domain_simplification
}   // namespace internals

/**
 * @brief Simplify a planar domain without changing its ring topology.
 *
 * Every output edge replaces one contiguous input-ring arc whose vertices are at most `maximum_deviation` from the
 * edge. The result has a canonical counter-clockwise outer ring; holes are canonical clockwise rings sorted
 * lexicographically. Shortcuts that would create a touch/intersection, lose strict containment, nest holes, reverse
 * orientation, or make a ring degenerate are retained instead. Simplification may therefore stop before every
 * geometrically eligible vertex is removed. This one-sided boundary-error bound does not promise an exact vertex
 * count, area preservation, or retention of observations inside the input domain.
 */
inline PlanarDomain simplify_polygon_domain(const PlanarDomain& domain, double maximum_deviation) {
    if (!std::isfinite(maximum_deviation) || maximum_deviation < 0.0) {
        throw std::invalid_argument("Polygon simplification maximum_deviation must be finite and nonnegative.");
    }
    return internals::domain_simplification::simplify(domain, maximum_deviation);
}

}   // namespace fdapde

#endif   // __FDAPDE_GEOMETRY_DOMAIN_SIMPLIFICATION_H__
