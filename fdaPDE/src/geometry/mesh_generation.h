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

#ifndef __FDAPDE_GEOMETRY_MESH_GENERATION_H__
#define __FDAPDE_GEOMETRY_MESH_GENERATION_H__

#include <cstdint>
#include <stdexcept>

#include "header_check.h"

namespace fdapde {
namespace internals {
namespace lattice_boundary {

// functional source: Pietro Donelli, donellipietro/mesh_generator main@70591db6498cdfa16b5d0afe19542c334dc9b195
// this implementation replaces the prototype's sp/sf/rmapshaper union with integer cell topology

using cell_t = std::array<std::int64_t, 2>;
using vertex_t = std::array<std::int64_t, 2>;
using directed_edge_t = std::pair<vertex_t, vertex_t>;

template <typename Neighbors>
inline std::set<cell_t> largest_component(const std::set<cell_t>& occupied, Neighbors&& neighbors) {
    std::set<cell_t> unvisited = occupied;
    std::set<cell_t> best;
    while (!unvisited.empty()) {
        const cell_t first = *unvisited.begin();
        std::queue<cell_t> queue;
        std::set<cell_t> component;
        queue.push(first);
        unvisited.erase(first);
        while (!queue.empty()) {
            const cell_t cell = queue.front();
            queue.pop();
            component.insert(cell);
            for (const cell_t& next : neighbors(cell)) {
                auto it = unvisited.find(next);
                if (it != unvisited.end()) {
                    queue.push(next);
                    unvisited.erase(it);
                }
            }
        }
        if (
          component.size() > best.size() || (component.size() == best.size() && !component.empty() &&
                                             (best.empty() || *component.begin() < *best.begin()))) {
            best = std::move(component);
        }
    }
    return best;
}

inline int turn_rank(const vertex_t& previous, const vertex_t& current, const vertex_t& next) {
    const std::int64_t dx1 = current[0] - previous[0];
    const std::int64_t dy1 = current[1] - previous[1];
    const std::int64_t dx2 = next[0] - current[0];
    const std::int64_t dy2 = next[1] - current[1];
    const std::int64_t cross = dx1 * dy2 - dy1 * dx2;
    const std::int64_t dot = dx1 * dx2 + dy1 * dy2;
    if (cross < 0) return 0;               // right
    if (cross == 0 && dot > 0) return 1;   // straight
    if (cross > 0) return 2;               // left
    return 3;                              // reverse
}

inline std::vector<vertex_t> compress_ring(std::vector<vertex_t> ring) {
    bool changed = true;
    while (changed && ring.size() > 3) {
        changed = false;
        for (int i = 0, n = ring.size(); i < n; ++i) {
            const vertex_t& previous = ring[(i + n - 1) % n];
            const vertex_t& current = ring[i];
            const vertex_t& next = ring[(i + 1) % n];
            const std::int64_t dx1 = current[0] - previous[0];
            const std::int64_t dy1 = current[1] - previous[1];
            const std::int64_t dx2 = next[0] - current[0];
            const std::int64_t dy2 = next[1] - current[1];
            if (dx1 * dy2 == dy1 * dx2 && dx1 * dx2 + dy1 * dy2 > 0) {
                ring.erase(ring.begin() + i);
                changed = true;
                break;
            }
        }
    }
    return ring;
}

inline long double twice_signed_area(const std::vector<vertex_t>& ring) {
    long double area = 0.0L;
    for (int i = 0, n = ring.size(); i < n; ++i) {
        const vertex_t& a = ring[i];
        const vertex_t& b = ring[(i + 1) % n];
        area += static_cast<long double>(a[0]) * b[1] - static_cast<long double>(a[1]) * b[0];
    }
    return area;
}

inline std::vector<std::vector<vertex_t>> trace_rings(const std::set<directed_edge_t>& edges) {
    std::map<vertex_t, std::set<vertex_t>> outgoing;
    for (const auto& [from, to] : edges) outgoing[from].insert(to);

    std::vector<std::vector<vertex_t>> rings;
    std::size_t remaining = edges.size();
    while (remaining > 0) {
        auto start_it =
          std::find_if(outgoing.begin(), outgoing.end(), [](const auto& entry) { return !entry.second.empty(); });
        if (start_it == outgoing.end()) { throw std::runtime_error("Lattice boundary has an open ring."); }
        const vertex_t start = start_it->first;
        vertex_t previous = start;
        vertex_t current = *start_it->second.begin();
        start_it->second.erase(start_it->second.begin());
        --remaining;

        std::vector<vertex_t> ring {start};
        std::size_t steps = 1;
        while (current != start) {
            ring.push_back(current);
            auto next_it = outgoing.find(current);
            if (next_it == outgoing.end() || next_it->second.empty()) {
                throw std::runtime_error("Lattice boundary has an open ring.");
            }
            auto selected = next_it->second.begin();
            int selected_rank = turn_rank(previous, current, *selected);
            for (auto it = std::next(selected); it != next_it->second.end(); ++it) {
                const int rank = turn_rank(previous, current, *it);
                if (rank < selected_rank || (rank == selected_rank && *it < *selected)) {
                    selected = it;
                    selected_rank = rank;
                }
            }
            const vertex_t next = *selected;
            next_it->second.erase(selected);
            --remaining;
            previous = current;
            current = next;
            if (++steps > edges.size()) { throw std::runtime_error("Lattice boundary tracing did not close."); }
        }
        ring = compress_ring(std::move(ring));
        if (ring.size() < 3) { throw std::runtime_error("Lattice boundary contains a degenerate ring."); }
        rings.push_back(std::move(ring));
    }
    return rings;
}

inline std::vector<vertex_t> outer_ring(const std::set<directed_edge_t>& edges, const char* lattice_name) {
    const auto rings = trace_rings(edges);
    const std::vector<vertex_t>* outer = nullptr;
    for (const auto& ring : rings) {
        if (twice_signed_area(ring) <= 0.0L) continue;
        if (outer != nullptr) {
            throw std::runtime_error(std::string(lattice_name) + " lattice component has multiple outer rings.");
        }
        outer = std::addressof(ring);
    }
    if (outer == nullptr) {
        throw std::runtime_error(std::string(lattice_name) + " lattice component has no outer ring.");
    }
    std::vector<vertex_t> canonical = *outer;
    std::rotate(canonical.begin(), std::min_element(canonical.begin(), canonical.end()), canonical.end());
    return canonical;
}

}   // namespace lattice_boundary

namespace square_lattice {

using lattice_boundary::cell_t;
using lattice_boundary::directed_edge_t;
using lattice_boundary::vertex_t;

inline cell_t neighbor(const cell_t& cell, int dx, int dy) { return {cell[0] + dx, cell[1] + dy}; }

inline std::int64_t cell_index(double coordinate, double origin, double spacing) {
    const long double normalized = (static_cast<long double>(coordinate) - static_cast<long double>(origin)) / spacing;
    const long double lower = std::floor(normalized);
    const long double index = lower + ((normalized - lower) >= 0.5L ? 1.0L : 0.0L);
    constexpr std::int64_t limit = std::numeric_limits<std::int64_t>::max() / 4;
    const long double floating_limit = static_cast<long double>(limit);
    if (!std::isfinite(index) || index < -floating_limit || index > floating_limit) {
        throw std::invalid_argument("Square lattice coordinate exceeds the supported index range.");
    }
    return static_cast<std::int64_t>(index);
}

inline std::set<cell_t> largest_component(const std::set<cell_t>& occupied) {
    return lattice_boundary::largest_component(occupied, [](const cell_t& cell) {
        return std::array<cell_t, 4> {
          neighbor(cell, -1, 0), neighbor(cell, 0, -1), neighbor(cell, 1, 0), neighbor(cell, 0, 1)};
    });
}

inline std::set<directed_edge_t> boundary_edges(const std::set<cell_t>& cells) {
    std::set<directed_edge_t> edges;
    for (const cell_t& cell : cells) {
        const std::int64_t x = 2 * cell[0];
        const std::int64_t y = 2 * cell[1];
        const vertex_t bottom_left {x - 1, y - 1};
        const vertex_t bottom_right {x + 1, y - 1};
        const vertex_t top_right {x + 1, y + 1};
        const vertex_t top_left {x - 1, y + 1};
        if (!cells.contains(neighbor(cell, 0, -1))) edges.emplace(bottom_left, bottom_right);
        if (!cells.contains(neighbor(cell, 1, 0))) edges.emplace(bottom_right, top_right);
        if (!cells.contains(neighbor(cell, 0, 1))) edges.emplace(top_right, top_left);
        if (!cells.contains(neighbor(cell, -1, 0))) edges.emplace(top_left, bottom_left);
    }
    return edges;
}

}   // namespace square_lattice

namespace hexagonal_lattice {

using lattice_boundary::cell_t;
using lattice_boundary::directed_edge_t;
using lattice_boundary::vertex_t;

inline cell_t neighbor(const cell_t& cell, int dq, int dr) { return {cell[0] + dq, cell[1] + dr}; }

inline void validate_coordinate(double x, double y, double anchor_x, double anchor_y, double spacing) {
    constexpr long double sqrt_three = 1.73205080756887729352744634150587237L;
    constexpr std::int64_t limit = std::numeric_limits<std::int64_t>::max() / 16;
    constexpr long double floating_limit = static_cast<long double>(limit);
    const long double dx = (static_cast<long double>(x) - anchor_x) / spacing;
    const long double dy = (static_cast<long double>(y) - anchor_y) / spacing;
    const long double q = dx - dy / sqrt_three;
    const long double r = 2.0L * dy / sqrt_three;
    if (!std::isfinite(q) || !std::isfinite(r) || std::abs(q) > floating_limit || std::abs(r) > floating_limit) {
        throw std::invalid_argument("Hexagonal lattice coordinate exceeds the supported index range.");
    }
}

inline cell_t nearest_cell(double x, double y, double anchor_x, double anchor_y, double spacing) {
    constexpr long double sqrt_three = 1.73205080756887729352744634150587237L;
    const long double dx = (static_cast<long double>(x) - anchor_x) / spacing;
    const long double dy = (static_cast<long double>(y) - anchor_y) / spacing;
    const std::int64_t base_r = static_cast<std::int64_t>(std::floor(2.0L * dy / sqrt_three));
    cell_t best {};
    long double best_distance = std::numeric_limits<long double>::infinity();
    for (std::int64_t r = base_r - 1; r <= base_r + 2; ++r) {
        const long double q_coordinate = dx - 0.5L * r;
        const std::int64_t base_q = static_cast<std::int64_t>(std::floor(q_coordinate));
        for (std::int64_t q = base_q; q <= base_q + 1; ++q) {
            const long double delta_x = dx - (q + 0.5L * r);
            const long double delta_y = dy - 0.5L * sqrt_three * r;
            const long double distance = delta_x * delta_x + delta_y * delta_y;
            const cell_t candidate {q, r};
            if (distance < best_distance || (distance == best_distance && candidate < best)) {
                best = candidate;
                best_distance = distance;
            }
        }
    }
    return best;
}

inline std::set<cell_t> largest_component(const std::set<cell_t>& occupied) {
    return lattice_boundary::largest_component(occupied, [](const cell_t& cell) {
        return std::array<cell_t, 6> {neighbor(cell, 1, 0),  neighbor(cell, 0, 1),  neighbor(cell, -1, 1),
                                      neighbor(cell, -1, 0), neighbor(cell, 0, -1), neighbor(cell, 1, -1)};
    });
}

inline std::set<directed_edge_t> boundary_edges(const std::set<cell_t>& cells) {
    std::set<directed_edge_t> edges;
    for (const cell_t& cell : cells) {
        const std::int64_t x = 2 * cell[0] + cell[1];
        const std::int64_t y = 3 * cell[1];
        const std::array<vertex_t, 6> vertices {
          vertex_t {x + 1, y + 1},
           vertex_t {x,     y + 2},
           vertex_t {x - 1, y + 1},
          vertex_t {x - 1, y - 1},
           vertex_t {x,     y - 2},
           vertex_t {x + 1, y - 1}
        };
        const std::array<cell_t, 6> neighbors {neighbor(cell, 0, 1),  neighbor(cell, -1, 1), neighbor(cell, -1, 0),
                                               neighbor(cell, 0, -1), neighbor(cell, 1, -1), neighbor(cell, 1, 0)};
        for (int i = 0; i < 6; ++i) {
            if (!cells.contains(neighbors[i])) edges.emplace(vertices[i], vertices[(i + 1) % 6]);
        }
    }
    return edges;
}

template <typename ExecutionPolicy>
inline Matrix<double, Dynamic, Dynamic> boundary(
  ExecutionPolicy, const Matrix<double, Dynamic, Dynamic>& points, double spacing,
  std::optional<Vector<double, 2>> anchor) {
    if (points.rows() == 0 || points.cols() != 2) {
        throw std::invalid_argument("Hexagonal lattice points must be a nonempty matrix with two columns.");
    }
    if (!std::isfinite(spacing) || spacing <= 0.0) {
        throw std::invalid_argument("Hexagonal lattice spacing must be finite and positive.");
    }

    double anchor_x = points(0, 0);
    double anchor_y = points(0, 1);
    if (anchor.has_value()) {
        anchor_x = (*anchor)[0];
        anchor_y = (*anchor)[1];
        if (!std::isfinite(anchor_x) || !std::isfinite(anchor_y)) {
            throw std::invalid_argument("Hexagonal lattice anchor must contain finite coordinates.");
        }
    }
    for (int i = 0; i < points.rows(); ++i) {
        if (!std::isfinite(points(i, 0)) || !std::isfinite(points(i, 1))) {
            throw std::invalid_argument("Hexagonal lattice points must contain only finite coordinates.");
        }
        if (!anchor.has_value()) {
            anchor_x = std::min(anchor_x, points(i, 0));
            anchor_y = std::max(anchor_y, points(i, 1));
        }
    }
    for (int i = 0; i < points.rows(); ++i) {
        validate_coordinate(points(i, 0), points(i, 1), anchor_x, anchor_y, spacing);
    }

    std::vector<cell_t> assignments(points.rows());
    const auto classify = [&](int i) {
        assignments[i] = nearest_cell(points(i, 0), points(i, 1), anchor_x, anchor_y, spacing);
    };
    if constexpr (std::is_same_v<ExecutionPolicy, execution_par_t>) {
        parallel_for(0, points.rows(), classify);
    } else {
        for (int i = 0; i < points.rows(); ++i) classify(i);
    }

    const std::set<cell_t> occupied(assignments.begin(), assignments.end());
    const std::set<cell_t> component = largest_component(occupied);
    const std::vector<vertex_t> ring = lattice_boundary::outer_ring(boundary_edges(component), "Hexagonal");
    Matrix<double, Dynamic, Dynamic> result(ring.size(), 2);
    std::set<std::array<double, 2>> represented_vertices;
    constexpr long double sqrt_three = 1.73205080756887729352744634150587237L;
    for (int i = 0; i < int(ring.size()); ++i) {
        result(i, 0) = anchor_x + 0.5L * spacing * ring[i][0];
        result(i, 1) = anchor_y + 0.5L / sqrt_three * spacing * ring[i][1];
        if (!std::isfinite(result(i, 0)) || !std::isfinite(result(i, 1))) {
            throw std::invalid_argument("Hexagonal lattice boundary exceeds the finite coordinate range.");
        }
        if (!represented_vertices.insert({result(i, 0), result(i, 1)}).second) {
            throw std::invalid_argument(
              "Hexagonal lattice spacing is too small to represent distinct boundary vertices at this anchor.");
        }
    }
    return result;
}

}   // namespace hexagonal_lattice
}   // namespace internals

/**
 * @brief Build the outer boundary of the largest 4-connected component of occupied square cells.
 *
 * Each point occupies its nearest lattice cell; midpoint ties select the cell with the greater integer index and
 * duplicate points have no effect. The optional anchor is the center of cell (0, 0). Without an explicit anchor,
 * `(min(points.col(0)), max(points.col(1)))` is used. Equal-size components are resolved by their lexicographically
 * smallest cell. Interior holes are filled, and the result is an unclosed, counter-clockwise ring beginning at its
 * lexicographically smallest vertex.
 */
inline Matrix<double, Dynamic, Dynamic> square_lattice_boundary(
  const Matrix<double, Dynamic, Dynamic>& points, double spacing,
  std::optional<Vector<double, 2>> anchor = std::nullopt) {
    using namespace internals::square_lattice;
    if (points.rows() == 0 || points.cols() != 2) {
        throw std::invalid_argument("Square lattice points must be a nonempty matrix with two columns.");
    }
    if (!std::isfinite(spacing) || spacing <= 0.0) {
        throw std::invalid_argument("Square lattice spacing must be finite and positive.");
    }

    double anchor_x;
    double anchor_y;
    if (anchor.has_value()) {
        anchor_x = (*anchor)[0];
        anchor_y = (*anchor)[1];
        if (!std::isfinite(anchor_x) || !std::isfinite(anchor_y)) {
            throw std::invalid_argument("Square lattice anchor must contain finite coordinates.");
        }
    } else {
        anchor_x = points(0, 0);
        anchor_y = points(0, 1);
        for (int i = 0; i < points.rows(); ++i) {
            if (!std::isfinite(points(i, 0)) || !std::isfinite(points(i, 1))) {
                throw std::invalid_argument("Square lattice points must contain only finite coordinates.");
            }
            anchor_x = std::min(anchor_x, points(i, 0));
            anchor_y = std::max(anchor_y, points(i, 1));
        }
    }

    std::set<cell_t> occupied;
    for (int i = 0; i < points.rows(); ++i) {
        if (!std::isfinite(points(i, 0)) || !std::isfinite(points(i, 1))) {
            throw std::invalid_argument("Square lattice points must contain only finite coordinates.");
        }
        occupied.insert({cell_index(points(i, 0), anchor_x, spacing), cell_index(points(i, 1), anchor_y, spacing)});
    }

    const std::set<cell_t> component = largest_component(occupied);
    const std::vector<vertex_t> canonical =
      internals::lattice_boundary::outer_ring(boundary_edges(component), "Square");
    Matrix<double, Dynamic, Dynamic> boundary(canonical.size(), 2);
    std::set<std::array<double, 2>> represented_vertices;
    for (int i = 0; i < int(canonical.size()); ++i) {
        boundary(i, 0) = anchor_x + 0.5 * spacing * canonical[i][0];
        boundary(i, 1) = anchor_y + 0.5 * spacing * canonical[i][1];
        if (!std::isfinite(boundary(i, 0)) || !std::isfinite(boundary(i, 1))) {
            throw std::invalid_argument("Square lattice boundary exceeds the finite coordinate range.");
        }
        if (!represented_vertices.insert({boundary(i, 0), boundary(i, 1)}).second) {
            throw std::invalid_argument(
              "Square lattice spacing is too small to represent distinct boundary vertices at this anchor.");
        }
    }
    return boundary;
}

/**
 * @brief Build the outer boundary of the largest 6-connected component of occupied pointy hexagonal cells.
 *
 * `spacing` is the horizontal distance between neighboring cell centers. The optional anchor is the center of axial
 * cell (0, 0); without one, `(min(points.col(0)), max(points.col(1)))` is used. Each point occupies its nearest cell,
 * with exactly equidistant cells resolved lexicographically. Duplicate points have no effect. Interior holes are
 * filled, and the result is an unclosed, counter-clockwise ring beginning at its lexicographically smallest vertex.
 * The default overload executes sequentially; pass `execution_par` to classify points in parallel before the serial,
 * deterministic topology stage.
 */
inline Matrix<double, Dynamic, Dynamic> hexagonal_lattice_boundary(
  execution_seq_t policy, const Matrix<double, Dynamic, Dynamic>& points, double spacing,
  std::optional<Vector<double, 2>> anchor = std::nullopt) {
    return internals::hexagonal_lattice::boundary(policy, points, spacing, anchor);
}

inline Matrix<double, Dynamic, Dynamic> hexagonal_lattice_boundary(
  execution_par_t policy, const Matrix<double, Dynamic, Dynamic>& points, double spacing,
  std::optional<Vector<double, 2>> anchor = std::nullopt) {
    return internals::hexagonal_lattice::boundary(policy, points, spacing, anchor);
}

inline Matrix<double, Dynamic, Dynamic> hexagonal_lattice_boundary(
  const Matrix<double, Dynamic, Dynamic>& points, double spacing,
  std::optional<Vector<double, 2>> anchor = std::nullopt) {
    return hexagonal_lattice_boundary(execution_seq, points, spacing, anchor);
}

}   // namespace fdapde

#endif   // __FDAPDE_GEOMETRY_MESH_GENERATION_H__
