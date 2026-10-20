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

#include <fdaPDE/geometric_finite_elements.h>
#include <gtest/gtest.h>

namespace {
using namespace fdapde;

// planar vector P1 retains geometric numbering, component offsets and boundary flags
TEST(FiniteElementEnumeration, PlanarVectorP1PreservesNumbering) {
    const auto mesh = Triangulation<2, 2>::UnitSquare(6);
    const FeSpace space(mesh, P1<2>);
    const auto& handler = space.dof_handler();
    // two components allocate two copies of the geometric node degrees of freedom
    EXPECT_EQ(space.n_dofs(), 2 * mesh.n_nodes());
    // the first component inherits the exact mesh connectivity
    EXPECT_TRUE(handler.dofs().leftCols(3) == mesh.cells());
    // the second component uses the first component's numbering plus its node count
    EXPECT_TRUE((handler.dofs().rightCols(3).array() == mesh.cells().array() + mesh.n_nodes()).all());
    // the independent geometric node coordinates identify the unique physical dofs
    EXPECT_LT((handler.dofs_coords() - mesh.nodes()).norm(), 1e-13);
    // boundary flags replicate for both components without changing membership
    EXPECT_EQ(handler.n_boundary_dofs(), 2 * mesh.n_boundary_nodes());
}

// a refined tetrahedral P1 space preserves nodal coordinates and the extracted boundary set
TEST(FiniteElementEnumeration, TetrahedralP1PreservesGeometryAndSurface) {
    const auto mesh = Triangulation<3, 3>::UnitCube(12);
    const FeSpace space(mesh, P1<1>);
    const auto& handler = space.dof_handler();
    // a continuous scalar P1 field assigns one dof to each original mesh node
    EXPECT_EQ(space.n_dofs(), mesh.n_nodes());
    // direct connectivity equality detects any change in local vertex ordering
    EXPECT_TRUE(handler.dofs() == mesh.cells());
    // mapping reference dofs through their cells reproduces the physical mesh nodes
    EXPECT_LT((handler.dofs_coords() - mesh.nodes()).norm(), 1e-12);
    const auto surface = handler.surface_dofs();
    // the extracted surface contains every geometric boundary node exactly once
    EXPECT_EQ(surface.size(), static_cast<std::size_t>(mesh.n_boundary_nodes()));
    for (int node : surface) {
        // geometric boundary membership is the independent oracle for each returned surface dof
        EXPECT_TRUE(mesh.is_node_on_boundary(node));
    }
}

// tetrahedral P2 retains shared edge degrees of freedom and their physical midpoint coordinates
TEST(FiniteElementEnumeration, TetrahedralP2PreservesSharedEdges) {
    const auto mesh = Triangulation<3, 3>::UnitCube(3);
    const FeSpace space(mesh, P2<1>);
    const auto& handler = space.dof_handler();
    // the P2 space adds one shared dof for each geometric edge to the original node dofs
    EXPECT_EQ(space.n_dofs(), mesh.n_nodes() + mesh.n_edges());
    const auto coordinates = handler.dofs_coords();
    for (int cell = 0; cell < mesh.n_cells(); ++cell) {
        const auto element = mesh.cell(cell);
        int local = 0;
        for (auto edge = element.edges_begin(); edge != element.edges_end(); ++edge, ++local) {
            const auto nodes = edge->node_ids();
            const Eigen::RowVector3d midpoint = .5 * (mesh.nodes().row(nodes[0]) + mesh.nodes().row(nodes[1]));
            const int dof = handler.dofs()(cell, 4 + local);
            // each local edge dof maps to the geometric midpoint regardless of global enumeration order
            EXPECT_LT((coordinates.row(dof) - midpoint).norm(), 1e-13);
        }
    }
    // geometric boundary nodes and edges independently count the P2 boundary dofs
    EXPECT_EQ(handler.n_boundary_dofs(), mesh.n_boundary_nodes() + mesh.n_boundary_edges());
}
}   // namespace
