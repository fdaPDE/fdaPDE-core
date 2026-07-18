#include <fdaPDE/finite_elements.h>

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

}   // namespace fdapde
