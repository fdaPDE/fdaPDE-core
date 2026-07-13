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

int missing_hole_edges_before_recovery(
  const fdapde::PlanarDomain& domain, const Matrix<double, Dynamic, Dynamic>& sites) {
    using namespace fdapde::internals::delaunay_2d;
    domain_input_t input = read_domain(domain, sites);
    std::vector<cell_t> cells = ear_clip(input.points, input.outer);
    legalize(cells, input.points);
    std::vector<int> inserted_ids(input.points.size() - input.outer.size());
    std::iota(inserted_ids.begin(), inserted_ids.end(), input.outer.size());
    for (int id : sorted_ids(input.points, std::move(inserted_ids))) { insert_node(id, false, input.points, cells); }
    const auto adjacency = edge_adjacency(cells);
    return std::count_if(input.hole_edges.begin(), input.hole_edges.end(), [&](const edge_t& edge_) {
        return !adjacency.contains(edge_);
    });
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
    for (int i = 0; i < input.rows(); ++i) {
        EXPECT_DOUBLE_EQ(mesh.nodes()(i, 0), input(i, 0));
        EXPECT_DOUBLE_EQ(mesh.nodes()(i, 1), input(i, 1));
    }
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

TEST(delaunay, triangulates_planar_domains_with_holes) {
    using namespace delaunay_testing;
    const fdapde::PlanarDomain annulus {
      .outer = points({{0, 0}, {4, 0}, {4, 4}, {0, 4}}
        ), .holes = {points({{1, 1}, {3, 1}, {3, 3}, {1, 3}})}
    };
    const mesh_t mesh = fdapde::constrained_delaunay(annulus);

    EXPECT_EQ(mesh.n_nodes(), 8);
    EXPECT_EQ(mesh.n_cells(), 8);
    EXPECT_EQ(mesh.n_boundary_edges(), 8);
    EXPECT_EQ(mesh.n_nodes() - mesh.n_edges() + mesh.n_cells(), 0);
    EXPECT_NEAR(mesh.measure(), 12.0, 1e-14);
    for (int i = 0; i < 4; ++i) {
        EXPECT_TRUE(has_edge(mesh, i, (i + 1) % 4));
        EXPECT_TRUE(has_edge(mesh, 4 + i, 4 + (i + 1) % 4));
    }
    expect_valid_topology(mesh);
    expect_locally_delaunay(mesh);
}

TEST(delaunay, triangulates_multiple_and_concave_holes_with_sites) {
    using namespace delaunay_testing;
    const fdapde::PlanarDomain two_holes {
      .outer = points({{0, 0}, {10, 0}, {10, 8}, {0, 8}}
        ),
      .holes = {points({{1, 1}, {3, 1}, {3, 3}, {1, 3}}), points({{6, 2}, {9, 2}, {7, 4}})}
    };
    const mesh_t multiple = fdapde::constrained_delaunay(two_holes);
    EXPECT_EQ(multiple.n_nodes(), 11);
    EXPECT_EQ(multiple.n_cells(), 13);
    EXPECT_EQ(multiple.n_boundary_edges(), 11);
    EXPECT_EQ(multiple.n_nodes() - multiple.n_edges() + multiple.n_cells(), -1);
    EXPECT_NEAR(multiple.measure(), 73.0, 1e-13);
    expect_valid_topology(multiple);
    expect_locally_delaunay(multiple);

    const fdapde::PlanarDomain concave_hole {
      .outer = points({{0, 0}, {10, 0}, {10, 10}, {0, 10}}
        ),
      .holes = {points({{2, 2}, {6, 2}, {6, 3}, {3, 3}, {3, 6}, {2, 6}})}
    };
    const auto sites = points({
      {5, 5},
      {8, 8}
    });
    const mesh_t concave = fdapde::constrained_delaunay(concave_hole, sites);
    EXPECT_EQ(concave.n_nodes(), 12);
    EXPECT_EQ(concave.n_cells(), 14);
    EXPECT_EQ(concave.n_boundary_edges(), 10);
    EXPECT_NEAR(concave.measure(), 93.0, 1e-13);
    expect_valid_topology(concave);
    expect_locally_delaunay(concave);
}

TEST(delaunay, triangulates_reference_fork_domains) {
    using namespace delaunay_testing;

    // domains adapted from francesca1606/Luca-Francesca stable@fbd20e1
    const auto star = points({
      {0.0,    100.0 },
      {-22.45, 30.90 },
      {-95.11, 30.90 },
      {-36.33, -11.80},
      {-58.78, -80.90},
      {0.0,    -38.20},
      {58.78,  -80.90},
      {36.33,  -11.80},
      {95.11,  30.90 },
      {22.45,  30.90 }
    });

    const mesh_t star_mesh = fdapde::constrained_delaunay(star);
    EXPECT_EQ(star_mesh.n_boundary_edges(), star.rows());
    EXPECT_NEAR(star_mesh.measure(), 11225.978, 1e-9);
    expect_valid_topology(star_mesh);
    expect_locally_delaunay(star_mesh);

    const fdapde::PlanarDomain letter_a {
      .outer = points(
        {{0.0, 0.0}, {7.5, 0.0}, {10.0, 10.0}, {20.0, 10.0}, {22.5, 0.0}, {30.0, 0.0}, {22.5, 30.0}, {7.5, 30.0}}
          ),
      .holes = {points({{11.5, 16.0}, {18.5, 16.0}, {17.0, 24.0}, {13.0, 24.0}})}
    };

    const mesh_t letter_a_mesh = fdapde::constrained_delaunay(letter_a);
    EXPECT_EQ(letter_a_mesh.n_boundary_edges(), 12);
    EXPECT_NEAR(letter_a_mesh.measure(), 506.0, 1e-12);
    expect_valid_topology(letter_a_mesh);
    expect_locally_delaunay(letter_a_mesh);
}

TEST(delaunay, recovers_missing_hole_constraints) {
    using namespace delaunay_testing;
    const fdapde::PlanarDomain domain {
      .outer = points({{0, 0}, {10, 0}, {10, 10}, {0, 10}}
        ), .holes = {points({{4, 4}, {6, 4}, {5, 6}})}
    };
    const auto sites = points({
      {5, 3.9},
      {3, 4  }
    });
    EXPECT_EQ(missing_hole_edges_before_recovery(domain, sites), 1);
    const mesh_t mesh = fdapde::constrained_delaunay(domain, sites);
    EXPECT_TRUE(has_edge(mesh, 4, 5));
    EXPECT_EQ(mesh.n_nodes(), 9);
    EXPECT_EQ(mesh.n_cells(), 11);
    EXPECT_EQ(mesh.n_boundary_edges(), 7);
    EXPECT_EQ(mesh.n_nodes() - mesh.n_edges() + mesh.n_cells(), 0);
    EXPECT_NEAR(mesh.measure(), 98.0, 1e-13);
    expect_valid_topology(mesh);
    expect_locally_delaunay(mesh);
}

TEST(delaunay, refines_planar_domains_without_filling_holes) {
    using namespace delaunay_testing;
    const fdapde::PlanarDomain annulus {
      .outer = points({{0, 0}, {4, 0}, {4, 4}, {0, 4}}
        ), .holes = {points({{1, 1}, {3, 1}, {3, 3}, {1, 3}})}
    };
    const mesh_t refined =
      fdapde::constrained_delaunay(annulus, fdapde::DelaunayRefinement {.max_area = 0.5, .max_insertions = 200});
    double largest_cell = 0.0;
    for (auto cell = refined.cells_begin(); cell != refined.cells_end(); ++cell) {
        largest_cell = std::max(largest_cell, cell->measure());
    }
    EXPECT_LE(largest_cell, 0.5);
    EXPECT_EQ(refined.n_boundary_edges(), 8);
    EXPECT_EQ(refined.n_nodes() - refined.n_edges() + refined.n_cells(), 0);
    EXPECT_NEAR(refined.measure(), 12.0, 1e-13);
    expect_valid_topology(refined);
    expect_locally_delaunay(refined);
}

TEST(delaunay, planar_domains_are_repeatable_and_delegate_without_holes) {
    using namespace delaunay_testing;
    const auto outer = points({
      {0, 0},
      {4, 0},
      {4, 4},
      {0, 4}
    });
    const fdapde::PlanarDomain annulus {
      .outer = outer, .holes = {points({{1, 1}, {3, 1}, {3, 3}, {1, 3}})}
    };
    const mesh_t first = fdapde::constrained_delaunay(annulus);
    const mesh_t second = fdapde::constrained_delaunay(annulus);
    EXPECT_EQ(first.nodes(), second.nodes());
    EXPECT_EQ(first.cells(), second.cells());
    EXPECT_EQ(first.boundary_nodes(), second.boundary_nodes());

    const mesh_t direct = fdapde::constrained_delaunay(outer);
    const mesh_t delegated = fdapde::constrained_delaunay(fdapde::PlanarDomain {.outer = outer, .holes = {}});
    EXPECT_EQ(direct.nodes(), delegated.nodes());
    EXPECT_EQ(direct.cells(), delegated.cells());
    EXPECT_EQ(direct.boundary_nodes(), delegated.boundary_nodes());
}

TEST(delaunay, planar_domains_accept_orientation_changes_and_uniform_scaling) {
    using namespace delaunay_testing;
    for (const double scale : {1e-100, 1.0, 1e100}) {
        const fdapde::PlanarDomain domain {
          .outer = points({{0, 4 * scale}, {4 * scale, 4 * scale}, {4 * scale, 0}, {0, 0}}
            ),
          .holes = {
                           points({{3 * scale, 1 * scale}, {1 * scale, 1 * scale}, {1 * scale, 3 * scale}, {3 * scale, 3 * scale}})}
        };
        const mesh_t mesh = fdapde::constrained_delaunay(domain);
        EXPECT_EQ(mesh.n_cells(), 8);
        EXPECT_EQ(mesh.n_boundary_edges(), 8);
        EXPECT_NEAR(mesh.measure() / (scale * scale), 12.0, 1e-12);
        expect_valid_topology(mesh);
        if (scale == 1.0) expect_locally_delaunay(mesh);
    }
}

TEST(delaunay, rejects_invalid_planar_domains) {
    using namespace delaunay_testing;
    const auto outer = points({
      {0,  0 },
      {10, 0 },
      {10, 10},
      {0,  10}
    });
    const auto hole = points({
      {2, 2},
      {5, 2},
      {5, 5},
      {2, 5}
    });
    const auto rejects = [&](std::vector<Matrix<double, Dynamic, Dynamic>> holes) {
        EXPECT_THROW(
          fdapde::constrained_delaunay(fdapde::PlanarDomain {.outer = outer, .holes = std::move(holes)}),
          std::invalid_argument);
    };
    rejects({
      points({{11, 1}, {12, 1}, {11, 2}}
      )
    });
    rejects({
      points({{0, 1}, {2, 1}, {1, 2}}
      )
    });
    rejects({
      points({{8, 1}, {11, 1}, {8, 3}}
      )
    });
    rejects({
      hole, points({{4, 4}, {7, 4}, {7, 7}, {4, 7}}
       )
    });
    rejects({
      hole, points({{3, 3}, {4, 3}, {4, 4}, {3, 4}}
       )
    });
    rejects({
      points({{2, 2}, {5, 5}, {2, 5}, {5, 2}}
      )
    });

    const fdapde::PlanarDomain domain {.outer = outer, .holes = {hole}};
    EXPECT_THROW(
      fdapde::constrained_delaunay(
        domain, points({
                  {3, 3}
    })),
      std::invalid_argument);
    EXPECT_THROW(
      fdapde::constrained_delaunay(
        domain, points({
                  {2, 3}
    })),
      std::invalid_argument);
    EXPECT_THROW(
      fdapde::constrained_delaunay(
        domain, points({
                  {11, 3}
    })),
      std::invalid_argument);
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
    const mesh_t repeated =
      fdapde::constrained_delaunay(square, fdapde::DelaunayRefinement {.max_area = 0.2, .max_insertions = 20});
    EXPECT_EQ(refined.nodes(), repeated.nodes());
    EXPECT_EQ(refined.cells(), repeated.cells());
    EXPECT_EQ(refined.boundary_nodes(), repeated.boundary_nodes());

    const mesh_t refined_points =
      fdapde::delaunay(square, fdapde::DelaunayRefinement {.max_area = 0.2, .max_insertions = 20});
    double point_set_max_area = 0.0;
    for (auto it = refined_points.cells_begin(); it != refined_points.cells_end(); ++it) {
        point_set_max_area = std::max(point_set_max_area, it->measure());
    }
    EXPECT_LE(point_set_max_area, 0.2);
    EXPECT_GT(refined_points.n_nodes(), square.rows());
    EXPECT_NEAR(refined_points.measure(), 1.0, 1e-14);

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
