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

namespace dcel_testing {

using fdapde::Dynamic;
using fdapde::Matrix;

Matrix<double, Dynamic, Dynamic> points(std::initializer_list<std::array<double, 2>> coordinates) {
    Matrix<double, Dynamic, Dynamic> result(coordinates.size(), 2);
    int row = 0;
    for (const auto& point : coordinates) {
        result(row, 0) = point[0];
        result(row, 1) = point[1];
        ++row;
    }
    return result;
}

void expect_valid_links(const fdapde::DCEL<2, 2>& dcel) {
    for (int id = 0; id < dcel.n_halfedges(); ++id) {
        const auto& halfedge = dcel.halfedge(id);
        EXPECT_EQ(dcel.halfedge(halfedge.twin()).twin(), id);
        EXPECT_EQ(dcel.halfedge(halfedge.next()).previous(), id);
        EXPECT_EQ(dcel.halfedge(halfedge.previous()).next(), id);
        EXPECT_EQ(dcel.destination(id), dcel.halfedge(halfedge.next()).origin());
    }
}

}   // namespace dcel_testing

TEST(dcel, round_trips_triangulations_with_constant_time_links) {
    const auto source = fdapde::Triangulation<2, 2>::UnitSquare(3);
    const auto dcel = fdapde::to_dcel(source);

    EXPECT_EQ(dcel.n_nodes(), 9);
    EXPECT_EQ(dcel.n_cells(), 8);
    EXPECT_EQ(dcel.n_edges(), 16);
    EXPECT_EQ(dcel.n_boundary_edges(), 8);
    dcel_testing::expect_valid_links(dcel);

    const auto copy = dcel;
    EXPECT_NE(&copy.node(0), &dcel.node(0));
    const auto bridged = fdapde::to_triangulation(copy);
    EXPECT_EQ(bridged.nodes(), source.nodes());
    EXPECT_EQ(bridged.cells(), source.cells());
    EXPECT_EQ(bridged.boundary_nodes(), source.boundary_nodes());
    EXPECT_EQ(bridged.n_edges(), source.n_edges());
    EXPECT_NEAR(bridged.measure(), source.measure(), 1e-14);
}

TEST(dcel, builds_constrained_domains_and_bridges_on_demand) {
    using namespace dcel_testing;
    const auto outer = points({
      {0,  0 },
      {10, 0 },
      {10, 10},
      {0,  10}
    });
    const fdapde::PlanarDomain domain {
      .outer = outer, .holes = {points({{2, 2}, {6, 2}, {6, 3}, {3, 3}, {3, 6}, {2, 6}})}
    };
    const auto dcel = fdapde::constrained_delaunay_dcel(
      domain, points({
                {5, 5},
                {8, 8}
    }));

    EXPECT_EQ(dcel.n_nodes(), 12);
    EXPECT_EQ(dcel.n_cells(), 14);
    EXPECT_EQ(dcel.n_boundary_edges(), 10);
    EXPECT_EQ(
      std::count_if(dcel.halfedges_begin(), dcel.halfedges_end(), [](const auto& edge) { return edge.is_segment(); }),
      20);
    expect_valid_links(dcel);

    const auto mesh = dcel.triangulation();
    EXPECT_EQ(mesh.n_nodes(), 12);
    EXPECT_EQ(mesh.n_cells(), 14);
    EXPECT_EQ(mesh.n_boundary_edges(), 10);
    EXPECT_NEAR(mesh.measure(), 93.0, 1e-13);
}

TEST(dcel, rejects_non_manifold_triangle_soups) {
    using namespace dcel_testing;
    const auto nodes = points({
      {0, 0 },
      {1, 0 },
      {0, 1 },
      {0, -1},
      {0, 2 }
    });
    Matrix<int, Dynamic, Dynamic> cells(3, 3);
    cells.row(0) = {0, 1, 2};
    cells.row(1) = {1, 0, 3};
    cells.row(2) = {0, 1, 4};
    EXPECT_THROW(fdapde::DCEL<2, 2>::from_triangles(nodes, cells), std::invalid_argument);
}

TEST(dcel, normalizes_uniformly_scaled_cells) {
    using namespace dcel_testing;
    for (double scale : {1e-200, 1e200}) {
        const auto nodes = points({
          {0,     0        },
          {scale, scale    },
          {scale, 2 * scale}
        });
        Matrix<int, Dynamic, Dynamic> cells(1, 3);
        cells.row(0) = {0, 2, 1};

        const auto dcel = fdapde::DCEL<2, 2>::from_triangles(nodes, cells);
        EXPECT_EQ(dcel.cell_nodes(0), (std::array<int, 3> {0, 1, 2}));
        expect_valid_links(dcel);
    }
}
