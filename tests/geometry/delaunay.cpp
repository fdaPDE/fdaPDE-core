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

#include <cstdint>

namespace delaunay_testing {

using fdapde::Dynamic;
using fdapde::Matrix;
using mesh_t = fdapde::Triangulation<2, 2>;

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

double orient(const mesh_t& mesh, int a, int b, int c) {
    return (mesh.nodes()(a, 0) - mesh.nodes()(c, 0)) * (mesh.nodes()(b, 1) - mesh.nodes()(c, 1)) -
           (mesh.nodes()(a, 1) - mesh.nodes()(c, 1)) * (mesh.nodes()(b, 0) - mesh.nodes()(c, 0));
}

int opposite(const mesh_t& mesh, int cell, int a, int b) {
    for (int j = 0; j < 3; ++j) {
        const int node = mesh.cells()(cell, j);
        if (node != a && node != b) return node;
    }
    return -1;
}

void expect_valid_topology(const mesh_t& mesh) {
    std::vector<bool> expected_boundary(mesh.n_nodes(), false);
    for (int i = 0; i < mesh.n_cells(); ++i) {
        EXPECT_GT(orient(mesh, mesh.cells()(i, 0), mesh.cells()(i, 1), mesh.cells()(i, 2)), 0.0);
    }
    for (int i = 0; i < mesh.n_edges(); ++i) {
        if (mesh.is_edge_on_boundary(i)) {
            expected_boundary[mesh.edges()(i, 0)] = true;
            expected_boundary[mesh.edges()(i, 1)] = true;
        }
    }
    for (int i = 0; i < mesh.n_nodes(); ++i) { EXPECT_EQ(mesh.is_node_on_boundary(i), expected_boundary[i]); }
}

void expect_locally_delaunay(const mesh_t& mesh) {
    const auto adjacent = mesh.edge_to_cells();
    for (int i = 0; i < mesh.n_edges(); ++i) {
        const int first = adjacent(i, 0);
        const int second = adjacent(i, 1);
        if (second == -1) continue;
        const int a = mesh.edges()(i, 0);
        const int b = mesh.edges()(i, 1);
        const int c = opposite(mesh, first, a, b);
        const int d = opposite(mesh, second, a, b);
        ASSERT_NE(c, -1);
        ASSERT_NE(d, -1);
        const double adx = mesh.nodes()(a, 0) - mesh.nodes()(d, 0);
        const double ady = mesh.nodes()(a, 1) - mesh.nodes()(d, 1);
        const double bdx = mesh.nodes()(b, 0) - mesh.nodes()(d, 0);
        const double bdy = mesh.nodes()(b, 1) - mesh.nodes()(d, 1);
        const double cdx = mesh.nodes()(c, 0) - mesh.nodes()(d, 0);
        const double cdy = mesh.nodes()(c, 1) - mesh.nodes()(d, 1);
        double determinant = (adx * adx + ady * ady) * (bdx * cdy - cdx * bdy) +
                             (bdx * bdx + bdy * bdy) * (cdx * ady - adx * cdy) +
                             (cdx * cdx + cdy * cdy) * (adx * bdy - bdx * ady);
        if (orient(mesh, a, b, c) < 0.0) determinant = -determinant;
        EXPECT_LE(determinant, 1e-12);
    }
}

bool has_edge(const mesh_t& mesh, int a, int b) {
    if (a > b) std::swap(a, b);
    for (int i = 0; i < mesh.n_edges(); ++i) {
        if (mesh.edges()(i, 0) == a && mesh.edges()(i, 1) == b) return true;
    }
    return false;
}

int node_at(const mesh_t& mesh, double x, double y) {
    for (int i = 0; i < mesh.n_nodes(); ++i) {
        if (mesh.nodes()(i, 0) == x && mesh.nodes()(i, 1) == y) return i;
    }
    return -1;
}

bool has_geometric_edge(const mesh_t& mesh, double ax, double ay, double bx, double by) {
    const int a = node_at(mesh, ax, ay);
    const int b = node_at(mesh, bx, by);
    return a != -1 && b != -1 && has_edge(mesh, a, b);
}

Matrix<double, Dynamic, Dynamic> deterministic_points(int size) {
    Matrix<double, Dynamic, Dynamic> result(size, 2);
    std::uint64_t state = 0x9e3779b97f4a7c15ULL;
    for (int i = 0; i < size; ++i) {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        result(i, 0) = static_cast<double>(state >> 11) * 0x1.0p-53;
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        result(i, 1) = static_cast<double>(state >> 11) * 0x1.0p-53;
    }
    return result;
}

}   // namespace delaunay_testing

TEST(delaunay, triangulates_point_sets) {
    using namespace delaunay_testing;
    const auto input = points({
      {0, 0},
      {2, 0},
      {2, 2},
      {0, 2},
      {1, 0},
      {1, 1}
    });
    const mesh_t mesh = fdapde::delaunay(input);

    EXPECT_EQ(mesh.n_nodes(), 6);
    EXPECT_EQ(mesh.n_cells(), 5);
    EXPECT_EQ(mesh.n_boundary_edges(), 5);
    EXPECT_NEAR(mesh.measure(), 4.0, 1e-14);
    EXPECT_TRUE(has_edge(mesh, 0, 4));
    EXPECT_TRUE(has_edge(mesh, 1, 4));
    EXPECT_TRUE(mesh.is_node_on_boundary(4));
    EXPECT_FALSE(mesh.is_node_on_boundary(5));
    expect_valid_topology(mesh);
    expect_locally_delaunay(mesh);
}

TEST(delaunay, is_repeatable_for_cocircular_points) {
    using namespace delaunay_testing;
    const auto input = points({
      {0, 0},
      {1, 0},
      {1, 1},
      {0, 1}
    });
    const mesh_t first = fdapde::delaunay(input);
    const mesh_t second = fdapde::delaunay(input);

    EXPECT_EQ(first.nodes(), second.nodes());
    EXPECT_EQ(first.cells(), second.cells());
    EXPECT_EQ(first.boundary_nodes(), second.boundary_nodes());
    EXPECT_EQ(first.n_cells(), 2);
    EXPECT_NEAR(first.measure(), 1.0, 1e-14);
    expect_valid_topology(first);
    expect_locally_delaunay(first);
}

TEST(delaunay, normalizes_uniformly_scaled_predicates) {
    using namespace delaunay_testing;
    for (const double scale : {1.0, 1e100, 1e-100}) {
        const auto input = points({
          {0,         0        },
          {2 * scale, 0        },
          {2 * scale, 2 * scale},
          {0,         scale    }
        });
        const mesh_t mesh = fdapde::delaunay(input);

        EXPECT_TRUE(has_edge(mesh, 1, 3));
        EXPECT_FALSE(has_edge(mesh, 0, 2));
        expect_valid_topology(mesh);
    }
}

TEST(delaunay, preserves_invariants_for_a_general_position_cloud) {
    using namespace delaunay_testing;
    const auto input = deterministic_points(64);
    const mesh_t first = fdapde::delaunay(input);
    const mesh_t second = fdapde::delaunay(input);

    EXPECT_EQ(first.nodes(), second.nodes());
    EXPECT_EQ(first.cells(), second.cells());
    EXPECT_EQ(first.n_nodes() - first.n_edges() + first.n_cells(), 1);
    std::vector<bool> used(first.n_nodes(), false);
    for (int i = 0; i < first.n_cells(); ++i) {
        for (int j = 0; j < 3; ++j) used[first.cells()(i, j)] = true;
    }
    EXPECT_TRUE(std::all_of(used.begin(), used.end(), [](bool value) { return value; }));
    expect_valid_topology(first);
    expect_locally_delaunay(first);
}

TEST(delaunay, preserves_a_nonconvex_boundary_and_interior_sites) {
    using namespace delaunay_testing;
    const auto boundary = points({
      {0, 0},
      {2, 0},
      {2, 1},
      {1, 1},
      {1, 2},
      {0, 2}
    });
    const auto sites = points({
      {0.5, 0.5}
    });
    const mesh_t mesh = fdapde::constrained_delaunay(boundary, sites);

    EXPECT_EQ(mesh.n_nodes(), 7);
    EXPECT_EQ(mesh.n_cells(), 6);
    EXPECT_EQ(mesh.n_boundary_edges(), 6);
    EXPECT_NEAR(mesh.measure(), 3.0, 1e-14);
    for (int i = 0; i < boundary.rows(); ++i) EXPECT_TRUE(has_edge(mesh, i, (i + 1) % boundary.rows()));
    EXPECT_FALSE(mesh.is_node_on_boundary(6));
    expect_valid_topology(mesh);
    expect_locally_delaunay(mesh);
}

TEST(delaunay, accepts_clockwise_boundaries_and_sites_on_interior_edges) {
    using namespace delaunay_testing;
    const auto clockwise = points({
      {0, 2},
      {1, 2},
      {1, 1},
      {2, 1},
      {2, 0},
      {0, 0}
    });
    const mesh_t clockwise_mesh = fdapde::constrained_delaunay(clockwise);
    EXPECT_EQ(clockwise_mesh.n_cells(), 4);
    EXPECT_NEAR(clockwise_mesh.measure(), 3.0, 1e-14);

    const auto square = points({
      {0, 0},
      {1, 0},
      {1, 1},
      {0, 1}
    });
    const mesh_t split_mesh = fdapde::constrained_delaunay(
      square, points({
                {0.5, 0.5}
    }));
    EXPECT_EQ(split_mesh.n_nodes(), 5);
    EXPECT_EQ(split_mesh.n_cells(), 4);
    EXPECT_NEAR(split_mesh.measure(), 1.0, 1e-14);
    expect_valid_topology(split_mesh);
    expect_locally_delaunay(split_mesh);
}

TEST(delaunay, canonicalizes_equivalent_constrained_rings) {
    using namespace delaunay_testing;
    const std::array rings {
      points({{0, 0}, {1, 0}, {1, 1}, {0, 1}}
      ), points({{1, 0}, {1, 1}, {0, 1}, {0, 0}}
      ),
      points({{0, 0}, {0, 1}, {1, 1}, {1, 0}}
      )
    };

    for (const auto& ring : rings) {
        const mesh_t mesh = fdapde::constrained_delaunay(ring);
        EXPECT_TRUE(has_geometric_edge(mesh, 0, 1, 1, 0));
        EXPECT_FALSE(has_geometric_edge(mesh, 0, 0, 1, 1));
        expect_valid_topology(mesh);
    }
}

TEST(delaunay, enforces_bounded_max_area_refinement) {
    using namespace delaunay_testing;
    const auto square = points({
      {0, 0},
      {1, 0},
      {1, 1},
      {0, 1}
    });
    const mesh_t refined =
      fdapde::constrained_delaunay(square, fdapde::DelaunayRefinement {.max_area = 0.2, .max_insertions = 20});
    double max_area = 0.0;
    for (auto it = refined.cells_begin(); it != refined.cells_end(); ++it) {
        max_area = std::max(max_area, it->measure());
    }
    EXPECT_LE(max_area, 0.2);
    EXPECT_GT(refined.n_nodes(), square.rows());
    EXPECT_NEAR(refined.measure(), 1.0, 1e-14);

    EXPECT_THROW(
      fdapde::constrained_delaunay(square, fdapde::DelaunayRefinement {.max_area = 0.01, .max_insertions = 1}),
      std::runtime_error);
    EXPECT_NO_THROW(
      fdapde::constrained_delaunay(square, fdapde::DelaunayRefinement {.max_area = 1.0, .max_insertions = 0}));
    EXPECT_THROW(
      fdapde::constrained_delaunay(square, fdapde::DelaunayRefinement {.max_area = 0.0, .max_insertions = 1}),
      std::invalid_argument);
    EXPECT_THROW(
      fdapde::constrained_delaunay(square, fdapde::DelaunayRefinement {.max_area = 1.0, .max_insertions = -1}),
      std::invalid_argument);
}

TEST(delaunay, rejects_invalid_geometry) {
    using namespace delaunay_testing;
    EXPECT_THROW(
      fdapde::delaunay(points({
        {0, 0},
        {1, 0}
    })),
      std::invalid_argument);
    EXPECT_THROW(
      fdapde::delaunay(points({
        {0, 0},
        {1, 0},
        {2, 0}
    })),
      std::invalid_argument);
    EXPECT_THROW(
      fdapde::delaunay(points({
        {0, 0},
        {1, 0},
        {0, 1},
        {1, 0}
    })),
      std::invalid_argument);

    auto nonfinite = points({
      {0, 0},
      {1, 0},
      {0, 1}
    });
    nonfinite(1, 0) = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(fdapde::delaunay(nonfinite), std::invalid_argument);

    const auto bow_tie = points({
      {0, 0},
      {1, 1},
      {0, 1},
      {1, 0}
    });
    EXPECT_THROW(fdapde::constrained_delaunay(bow_tie), std::invalid_argument);
    const auto collinear_boundary = points({
      {0, 0},
      {1, 0},
      {2, 0},
      {2, 1},
      {0, 1}
    });
    EXPECT_THROW(fdapde::constrained_delaunay(collinear_boundary), std::invalid_argument);
    const auto closed = points({
      {0, 0},
      {1, 0},
      {1, 1},
      {0, 1},
      {0, 0}
    });
    EXPECT_THROW(fdapde::constrained_delaunay(closed), std::invalid_argument);

    const auto square = points({
      {0, 0},
      {1, 0},
      {1, 1},
      {0, 1}
    });
    EXPECT_THROW(
      fdapde::constrained_delaunay(
        square, points({
                  {2, 2}
    })),
      std::invalid_argument);
    EXPECT_THROW(
      fdapde::constrained_delaunay(
        square, points({
                  {0.5, 0}
    })),
      std::invalid_argument);
}
