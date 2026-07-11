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

using namespace fdapde;

TEST(geometry, simplex_contains_and_mesh_locate) {
    using simplex_t = Simplex<2, 2>;
    const Matrix<double, 2, 3> coordinates({0.0, 1.0, 0.0, 0.0, 0.0, 1.0});
    const simplex_t triangle(coordinates);
    EXPECT_EQ(triangle.contains(Vector<double, 2>(0.0, 0.0)), simplex_t::ON_VERTEX);
    EXPECT_EQ(triangle.contains(Vector<double, 2>(0.5, 0.0)), simplex_t::ON_FACE);
    EXPECT_EQ(triangle.contains(Vector<double, 2>(0.2, 0.2)), simplex_t::INSIDE);
    EXPECT_EQ(triangle.contains(Vector<double, 2>(-2.0 * machine_epsilon, 0.2)), simplex_t::OUTSIDE);

    const auto mesh = Triangulation<2, 2>::UnitSquare(2);
    Matrix<double, Dynamic, Dynamic> locations(3, 2);
    locations(0, 0) = 0.0;
    locations(0, 1) = 0.0;
    locations(1, 0) = 0.5;
    locations(1, 1) = 0.5;
    locations(2, 0) = 2.0;
    locations(2, 1) = 2.0;
    const auto cells = mesh.locate(locations);
    EXPECT_NE(cells[0], -1);
    EXPECT_NE(cells[1], -1);
    EXPECT_EQ(cells[2], -1);
}

TEST(geometry, structured_triangulation_iterators) {
    auto mesh = Triangulation<2, 2>::UnitSquare(3);
    static_assert(std::bidirectional_iterator<decltype(mesh.cells_begin())>);
    static_assert(std::bidirectional_iterator<decltype(mesh.boundary_begin())>);

    EXPECT_EQ(mesh.n_nodes(), 9);
    EXPECT_EQ(mesh.n_cells(), 8);
    EXPECT_EQ(mesh.n_edges(), 16);
    EXPECT_EQ(mesh.n_boundary_edges(), 8);
    EXPECT_DOUBLE_EQ(mesh.measure(), 1.0);

    int cells = 0;
    for (auto it = mesh.cells_begin(); it != mesh.cells_end(); it++) { ++cells; }
    EXPECT_EQ(cells, mesh.n_cells());

    auto first_cell = mesh.cells_begin();
    const auto original_cell = first_cell++;
    EXPECT_EQ(original_cell->id(), 0);
    EXPECT_EQ(first_cell->id(), 1);

    auto last_cell = mesh.cells_end();
    --last_cell;
    EXPECT_EQ(last_cell->id(), mesh.n_cells() - 1);

    mesh.mark_cells(7, [](const auto& cell) { return cell.id() % 2 == 0; });
    int marked_cells = 0;
    for (auto it = mesh.cells_begin(7); it != mesh.cells_end(7); ++it) {
        EXPECT_EQ(it->id() % 2, 0);
        ++marked_cells;
    }
    EXPECT_EQ(marked_cells, 4);

    int edges = 0;
    for (auto it = mesh.edges_begin(); it != mesh.edges_end(); ++it) { ++edges; }
    EXPECT_EQ(edges, mesh.n_edges());

    auto last_edge = mesh.edges_end();
    --last_edge;
    EXPECT_EQ(last_edge->id(), mesh.n_edges() - 1);

    int boundary_edges = 0;
    for (auto it = mesh.boundary_begin(); it != mesh.boundary_end(); ++it) { ++boundary_edges; }
    EXPECT_EQ(boundary_edges, mesh.n_boundary_edges());

    const int selected_edge = mesh.boundary_begin()->id();
    mesh.mark_boundary(7, [&](const auto& edge) { return edge.id() == selected_edge; });
    EXPECT_EQ(std::distance(mesh.boundary_begin(7), mesh.boundary_end(7)), 1);
    EXPECT_EQ(std::distance(mesh.boundary_begin(0), mesh.boundary_end(0)), 0);

    int boundary_nodes = 0;
    for (auto it = mesh.boundary_nodes_begin(); it != mesh.boundary_nodes_end(); ++it) { ++boundary_nodes; }
    EXPECT_EQ(boundary_nodes, 8);
}

TEST(geometry, volumetric_triangulation_iterators) {
    auto mesh = Triangulation<3, 3>::UnitCube(4);

    int faces = 0;
    for (auto it = mesh.faces_begin(); it != mesh.faces_end(); ++it) { ++faces; }
    EXPECT_EQ(faces, mesh.n_faces());

    auto last_face = mesh.faces_end();
    --last_face;
    EXPECT_EQ(last_face->id(), mesh.n_faces() - 1);

    int boundary_faces = 0;
    for (auto it = mesh.boundary_begin(); it != mesh.boundary_end(); ++it) { ++boundary_faces; }
    EXPECT_EQ(boundary_faces, mesh.n_boundary_faces());

    Vector<bool, Dynamic> mask(mesh.n_faces());
    const int selected_face = mesh.boundary_begin()->id();
    mask[selected_face] = true;
    mesh.mark_boundary(mask);
    EXPECT_EQ(std::distance(mesh.boundary_begin(1), mesh.boundary_end(1)), 1);
}
