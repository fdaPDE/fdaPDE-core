#include <fdaPDE/finite_elements.h>
#include <gtest/gtest.h>

namespace fdapde {

TEST(dg_dof_handler, discontinuous_lagrange_numbering) {
    auto mesh = Triangulation<2, 2>::UnitSquare(2);
    FeSpace cg(mesh, P1<1>);
    FeSpace dg(mesh, DG<1, 1>);
    FeSpace dg2(mesh, DG<2, 1>);

    EXPECT_EQ(cg.n_dofs(), 4);
    EXPECT_EQ(dg.n_dofs(), 6);
    EXPECT_EQ(dg2.n_dofs(), 12);
    EXPECT_TRUE(cg.dof_handler().dof_sharing());
    EXPECT_FALSE(dg.dof_handler().dof_sharing());
    EXPECT_EQ(cg.sobolev_regularity(), 1);
    EXPECT_EQ(dg.sobolev_regularity(), 0);

    auto first = dg.dof_handler().active_dofs(0);
    auto second = dg.dof_handler().active_dofs(1);
    for (int i = 0; i < first.size(); ++i) {
        for (int j = 0; j < second.size(); ++j) { EXPECT_NE(first[i], second[j]); }
    }
}

TEST(dg_dof_handler, preserves_cell_local_coordinates) {
    auto mesh = Triangulation<2, 2>::UnitSquare(2);
    FeSpace dg(mesh, DG<1, 1>);
    auto coordinates = dg.dof_handler().dofs_coords();

    EXPECT_EQ(coordinates.rows(), 6);
    int repeated_diagonal_vertices = 0;
    for (int i = 0; i < coordinates.rows(); ++i) {
        for (int j = i + 1; j < coordinates.rows(); ++j) {
            if ((coordinates.row(i) - coordinates.row(j)).norm() < 1e-14) ++repeated_diagonal_vertices;
        }
    }
    EXPECT_EQ(repeated_diagonal_vertices, 2);
}

TEST(dg_dof_handler, maps_boundary_metadata_to_cell_local_dofs) {
    auto mesh = Triangulation<2, 2>::UnitSquare(2);
    mesh.mark_boundary(7);
    FeSpace cg(mesh, P1<1>);
    FeSpace dg(mesh, DG<1, 1>);
    FeSpace dg2(mesh, DG<2, 1>);

    EXPECT_EQ(cg.dof_handler().n_boundary_dofs(), 4);
    EXPECT_EQ(dg.dof_handler().n_boundary_dofs(), 6);
    EXPECT_EQ(dg2.dof_handler().n_boundary_dofs(), 10);
    for (int dof = 0; dof < dg.n_dofs(); ++dof) {
        EXPECT_TRUE(dg.dof_handler().is_dof_on_boundary(dof));
        EXPECT_EQ(dg.dof_handler().dof_marker(dof), 7);
    }
}

TEST(dg_dof_handler, dof_informed_edges_use_the_requested_cell_trace) {
    auto mesh = Triangulation<2, 2>::UnitSquare(2);
    FeSpace linear_space(mesh, DG<1, 1>);
    FeSpace quadratic_space(mesh, DG<2, 1>);
    FeSpace cubic_space(mesh, DG<3, 1>);

    auto check_edges = [&mesh](const auto& space) {
        const int n_edge_dofs = space.dof_handler().n_dofs_per_edge();
        for (int cell_id = 0; cell_id < mesh.n_cells(); ++cell_id) {
            auto cell = space.dof_handler().cell(cell_id);
            auto active_dofs = space.dof_handler().active_dofs(cell_id);
            for (int local_edge = 0; local_edge < mesh.n_edges_per_cell; ++local_edge) {
                auto edge_dofs = cell.edge(local_edge).dofs();
                ASSERT_EQ(edge_dofs.size(), 2 + n_edge_dofs);
                EXPECT_EQ(edge_dofs[0], active_dofs[mesh.edge_pattern(local_edge, 0)]);
                EXPECT_EQ(edge_dofs[1], active_dofs[mesh.edge_pattern(local_edge, 1)]);
                for (int k = 0; k < n_edge_dofs; ++k) {
                    EXPECT_EQ(edge_dofs[2 + k], active_dofs[3 + n_edge_dofs * local_edge + k]);
                }
            }
        }
    };
    check_edges(linear_space);
    check_edges(quadratic_space);
    check_edges(cubic_space);

    auto first_cell = cubic_space.dof_handler().cell(0);
    auto second_cell = cubic_space.dof_handler().cell(1);
    Eigen::VectorXi first_trace;
    Eigen::VectorXi second_trace;
    for (int local_edge = 0; local_edge < mesh.n_edges_per_cell; ++local_edge) {
        if (!first_cell.edge(local_edge).on_boundary()) first_trace = first_cell.edge(local_edge).dofs();
        if (!second_cell.edge(local_edge).on_boundary()) second_trace = second_cell.edge(local_edge).dofs();
    }
    ASSERT_EQ(first_trace.size(), 4);
    ASSERT_EQ(second_trace.size(), 4);
    for (int first_dof : first_trace) {
        for (int second_dof : second_trace) EXPECT_NE(first_dof, second_dof);
    }
}

TEST(dg_dof_handler, maps_interval_boundary_to_cell_local_dofs) {
    auto mesh = Triangulation<1, 1>::UnitInterval(4);
    FeSpace space(mesh, DG<1, 1>);
    auto coordinates = space.dof_handler().dofs_coords();

    EXPECT_EQ(space.n_dofs(), 6);
    EXPECT_EQ(space.dof_handler().n_boundary_dofs(), 2);
    EXPECT_FALSE(space.dof_handler().dof_sharing());
    for (int dof = 0; dof < space.n_dofs(); ++dof) {
        bool expected_boundary = almost_equal(coordinates(dof, 0), 0.0) || almost_equal(coordinates(dof, 0), 1.0);
        EXPECT_EQ(space.dof_handler().is_dof_on_boundary(dof), expected_boundary);
    }
}

TEST(dg_dof_handler, maps_tetrahedron_boundary_to_cell_local_dofs) {
    auto mesh = Triangulation<3, 3>::UnitCube(2);
    FeSpace space(mesh, DG<1, 1>);

    EXPECT_EQ(mesh.n_cells(), 6);
    EXPECT_EQ(space.n_dofs(), 24);
    EXPECT_EQ(space.dof_handler().n_boundary_dofs(), 24);
    EXPECT_FALSE(space.dof_handler().dof_sharing());
    for (int dof = 0; dof < space.n_dofs(); ++dof) EXPECT_TRUE(space.dof_handler().is_dof_on_boundary(dof));
}

TEST(dg_dof_handler, does_not_share_high_order_tetrahedron_faces) {
    auto mesh = Triangulation<3, 3>::UnitCube(2);
    FeSpace space(mesh, DG<4, 1>);

    EXPECT_EQ(space.n_dofs(), 6 * 35);
    EXPECT_EQ(space.dof_handler().n_boundary_dofs(), 6 * 25);
    for (int first_cell = 0; first_cell < mesh.n_cells(); ++first_cell) {
        auto first = space.dof_handler().active_dofs(first_cell);
        for (int second_cell = first_cell + 1; second_cell < mesh.n_cells(); ++second_cell) {
            auto second = space.dof_handler().active_dofs(second_cell);
            for (int first_dof : first) {
                for (int second_dof : second) EXPECT_NE(first_dof, second_dof);
            }
        }
    }
}

}   // namespace fdapde
