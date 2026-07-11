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
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>

namespace simplification_testing {

using fdapde::Dynamic;
using fdapde::Matrix;

Matrix<double, Dynamic, Dynamic> points(std::initializer_list<std::array<double, 2>> coordinates) {
    Matrix<double, Dynamic, Dynamic> result(coordinates.size(), 2);
    int i = 0;
    for (const auto& point : coordinates) {
        result(i, 0) = point[0];
        result(i, 1) = point[1];
        ++i;
    }
    return result;
}

double distance_to_segment(double x, double y, double ax, double ay, double bx, double by) {
    const double dx = bx - ax;
    const double dy = by - ay;
    const double projection = std::clamp(((x - ax) * dx + (y - ay) * dy) / (dx * dx + dy * dy), 0.0, 1.0);
    return std::hypot(x - (ax + projection * dx), y - (ay + projection * dy));
}

int vertex_index(const Matrix<double, Dynamic, Dynamic>& ring, double x, double y) {
    for (int i = 0; i < ring.rows(); ++i) {
        if (ring(i, 0) == x && ring(i, 1) == y) return i;
    }
    return -1;
}

double maximum_retained_arc_error(
  const Matrix<double, Dynamic, Dynamic>& original, const Matrix<double, Dynamic, Dynamic>& simplified) {
    double maximum = 0.0;
    for (int edge = 0; edge < simplified.rows(); ++edge) {
        const int next_edge = (edge + 1) % simplified.rows();
        const int first = vertex_index(original, simplified(edge, 0), simplified(edge, 1));
        const int last = vertex_index(original, simplified(next_edge, 0), simplified(next_edge, 1));
        if (first == -1 || last == -1) return std::numeric_limits<double>::infinity();
        int current = first;
        while (true) {
            maximum = std::max(
              maximum, distance_to_segment(
                         original(current, 0), original(current, 1), simplified(edge, 0), simplified(edge, 1),
                         simplified(next_edge, 0), simplified(next_edge, 1)));
            if (current == last) break;
            current = (current + 1) % original.rows();
        }
    }
    return maximum;
}

}   // namespace simplification_testing

TEST(domain_simplification, respects_deviation_and_canonicalizes_equivalent_inputs) {
    using namespace simplification_testing;
    const auto ring = points({
      {0, 0  },
      {4, 0  },
      {4, 4  },
      {2, 4.2},
      {0, 4  }
    });
    const auto unchanged = fdapde::simplify_polygon_domain(fdapde::PlanarDomain {.outer = ring}, 0.19);
    EXPECT_EQ(unchanged.outer.rows(), 5);

    const auto simplified = fdapde::simplify_polygon_domain(fdapde::PlanarDomain {.outer = ring}, 0.21);
    const auto expected = points({
      {0, 0},
      {4, 0},
      {4, 4},
      {0, 4}
    });
    EXPECT_EQ(simplified.outer, expected);
    EXPECT_TRUE(simplified.holes.empty());
    EXPECT_NEAR(fdapde::constrained_delaunay(simplified).measure(), 16.0, 1e-14);

    const auto rotated = points({
      {4, 4  },
      {2, 4.2},
      {0, 4  },
      {0, 0  },
      {4, 0  }
    });
    const auto reversed = points({
      {0, 4  },
      {2, 4.2},
      {4, 4  },
      {4, 0  },
      {0, 0  }
    });
    EXPECT_EQ(fdapde::simplify_polygon_domain(fdapde::PlanarDomain {.outer = rotated}, 0.21).outer, simplified.outer);
    EXPECT_EQ(fdapde::simplify_polygon_domain(fdapde::PlanarDomain {.outer = reversed}, 0.21).outer, simplified.outer);

    for (const double scale : {1e-150, 1e150}) {
        auto scaled = ring;
        for (int i = 0; i < scaled.rows(); ++i) {
            scaled(i, 0) *= scale;
            scaled(i, 1) *= scale;
        }
        EXPECT_EQ(
          fdapde::simplify_polygon_domain(fdapde::PlanarDomain {.outer = scaled}, 0.21 * scale).outer.rows(), 4);
    }
}

TEST(domain_simplification, retains_shortcuts_that_would_exclude_a_hole) {
    using namespace simplification_testing;
    const fdapde::PlanarDomain domain {
      .outer = points({{0, 0}, {4, 0}, {4, 4}, {2, 4.2}, {0, 4}}
        ),
      .holes = {points({{1.9, 4.05}, {2.0, 4.15}, {2.1, 4.05}})}
    };
    const auto simplified = fdapde::simplify_polygon_domain(domain, 0.21);
    EXPECT_EQ(simplified.outer.rows(), 5);
    ASSERT_EQ(simplified.holes.size(), 1);
    EXPECT_EQ(simplified.holes[0].rows(), 3);
    const auto mesh = fdapde::constrained_delaunay(simplified);
    EXPECT_EQ(mesh.n_boundary_edges(), 8);
    EXPECT_GT(mesh.measure(), 16.0);
}

TEST(domain_simplification, retains_shortcuts_that_would_self_intersect) {
    using namespace simplification_testing;
    const fdapde::PlanarDomain domain {
      .outer = points({{0, 0}, {6, 0}, {6, 6}, {0, 6}, {0, 4}, {4, 4}, {4, 2}, {0, 2}}
        )
    };
    const auto canonical = fdapde::internals::domain_simplification::canonical_domain(domain);
    std::vector<fdapde::internals::domain_simplification::ring_state_t> rings {
      fdapde::internals::domain_simplification::make_state(canonical.outer, true)};
    const int crossing_vertex = vertex_index(canonical.outer, 6.0, 0.0);
    ASSERT_NE(crossing_vertex, -1);
    EXPECT_FALSE(fdapde::internals::domain_simplification::preserves_topology(rings, 0, crossing_vertex));

    const auto simplified = fdapde::simplify_polygon_domain(domain, 5.0);
    const auto mesh = fdapde::constrained_delaunay(simplified);
    EXPECT_EQ(mesh.n_boundary_edges(), simplified.outer.rows());
    EXPECT_GT(mesh.measure(), 0.0);
}

TEST(domain_simplification, bounds_each_retained_arc_and_composes_with_meshing) {
    using namespace simplification_testing;
    constexpr int vertices = 16;
    Matrix<double, Dynamic, Dynamic> ring(vertices, 2);
    for (int i = 0; i < vertices; ++i) {
        const double angle = 2.0 * std::numbers::pi * i / vertices;
        ring(i, 0) = 5.0 * std::cos(angle);
        ring(i, 1) = 5.0 * std::sin(angle);
    }
    constexpr double tolerance = 0.5;
    const auto simplified = fdapde::simplify_polygon_domain(fdapde::PlanarDomain {.outer = ring}, tolerance);
    EXPECT_LT(simplified.outer.rows(), ring.rows());
    EXPECT_GE(simplified.outer.rows(), 3);
    EXPECT_LE(maximum_retained_arc_error(ring, simplified.outer), tolerance + 1e-13);
    const auto mesh = fdapde::constrained_delaunay(simplified);
    EXPECT_EQ(mesh.n_boundary_edges(), simplified.outer.rows());
    EXPECT_GT(mesh.measure(), 0.0);
    const auto locations = points({
      {0,  0 },
      {10, 10}
    });
    const auto cells = mesh.locate(locations);
    EXPECT_NE(cells[0], -1);
    EXPECT_EQ(cells[1], -1);
}

TEST(domain_simplification, preserves_multiple_holes_and_is_repeatable) {
    using namespace simplification_testing;
    const auto outer = points({
      {0,  0},
      {10, 0},
      {10, 8},
      {0,  8}
    });
    const auto first_hole = points({
      {1, 1},
      {1, 3},
      {3, 3},
      {3, 1}
    });
    const auto second_hole = points({
      {6, 2},
      {6, 5},
      {9, 5},
      {9, 2}
    });
    const auto first = fdapde::simplify_polygon_domain(
      fdapde::PlanarDomain {
        .outer = outer, .holes = {second_hole, first_hole}
    },
      2.0);
    const auto second = fdapde::simplify_polygon_domain(
      fdapde::PlanarDomain {
        .outer = outer, .holes = {first_hole, second_hole}
    },
      2.0);
    EXPECT_EQ(first.outer, second.outer);
    EXPECT_EQ(first.holes, second.holes);
    ASSERT_EQ(first.holes.size(), 2);
    EXPECT_EQ(first.holes[0].rows(), 3);
    EXPECT_EQ(first.holes[1].rows(), 4);
    EXPECT_LE(maximum_retained_arc_error(first_hole, first.holes[0]), 2.0 + 1e-14);
    EXPECT_LE(maximum_retained_arc_error(second_hole, first.holes[1]), 2.0 + 1e-14);
    const auto mesh = fdapde::constrained_delaunay(first);
    EXPECT_EQ(mesh.n_boundary_edges(), first.outer.rows() + first.holes[0].rows() + first.holes[1].rows());
    EXPECT_EQ(mesh.n_nodes() - mesh.n_edges() + mesh.n_cells(), -1);
}

TEST(domain_simplification, rejects_invalid_domains_and_tolerances) {
    using namespace simplification_testing;
    const fdapde::PlanarDomain square {
      .outer = points({{0, 0}, {4, 0}, {4, 4}, {0, 4}}
        )
    };
    EXPECT_THROW(fdapde::simplify_polygon_domain(square, -1.0), std::invalid_argument);
    EXPECT_THROW(
      fdapde::simplify_polygon_domain(square, std::numeric_limits<double>::quiet_NaN()), std::invalid_argument);
    EXPECT_THROW(
      fdapde::simplify_polygon_domain(
        fdapde::PlanarDomain {
          .outer = square.outer, .holes = {points({{0, 1}, {1, 1}, {1, 2}})}
    },
        0.1),
      std::invalid_argument);
}
