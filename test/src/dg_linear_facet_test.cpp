#include <fdaPDE/finite_elements.h>
#include <gtest/gtest.h>

namespace fdapde {

namespace {

Eigen::Matrix<double, 6, 1> exact_jump_load() {
    const double half_diagonal = std::sqrt(2.0) / 2.0;
    Eigen::Matrix<double, 6, 1> load;
    load << 0, half_diagonal, half_diagonal, -half_diagonal, -half_diagonal, 0;
    return load;
}

Eigen::Matrix<double, 6, 1> exact_average_load() {
    const double quarter_diagonal = std::sqrt(2.0) / 4.0;
    Eigen::Matrix<double, 6, 1> load;
    load << 0, quarter_diagonal, quarter_diagonal, quarter_diagonal, quarter_diagonal, 0;
    return load;
}

}   // namespace

TEST(dg_linear_facet_assembler, assembles_both_jump_traces_once) {
    auto mesh = Triangulation<2, 2>::UnitSquare(2);
    FeSpace space(mesh, DG<1, 1>);
    TestFunction v(space);

    Eigen::Matrix<double, 6, 1> assembled = integral(mesh)(jump(v)).assemble();

    EXPECT_LT((assembled - exact_jump_load()).norm(), 1e-12);
    EXPECT_GT(assembled.template head<3>().norm(), 0.0);
    EXPECT_GT(assembled.template tail<3>().norm(), 0.0);
    EXPECT_NEAR(assembled.sum(), 0.0, 1e-12);
}

TEST(dg_linear_facet_assembler, preserves_cell_terms_in_mixed_form) {
    auto mesh = Triangulation<2, 2>::UnitSquare(2);
    FeSpace space(mesh, DG<1, 1>);
    TestFunction v(space);

    Eigen::Matrix<double, 6, 1> assembled = integral(mesh)(v + jump(v)).assemble();
    Eigen::Matrix<double, 6, 1> expected = Eigen::Matrix<double, 6, 1>::Constant(1.0 / 6.0) + exact_jump_load();

    EXPECT_LT((assembled - expected).norm(), 1e-12);
}

TEST(dg_linear_facet_assembler, average_has_no_cell_contribution) {
    auto mesh = Triangulation<2, 2>::UnitSquare(2);
    FeSpace space(mesh, DG<1, 1>);
    TestFunction v(space);

    Eigen::Matrix<double, 6, 1> assembled = integral(mesh)(avg(v)).assemble();

    EXPECT_LT((assembled - exact_average_load()).norm(), 1e-12);
    EXPECT_NEAR(assembled.sum(), std::sqrt(2.0), 1e-12);
}

TEST(dg_facet_assembler, rejects_filtered_cell_ranges_explicitly) {
    auto mesh = Triangulation<2, 2>::UnitSquare(2);
    mesh.mark_cells(3);
    FeSpace space(mesh, DG<1, 1>);
    TrialFunction u(space);
    TestFunction v(space);
    auto begin = mesh.cells_begin(3);
    auto end = mesh.cells_end(3);

    EXPECT_THROW(integral(begin, end)(jump(v)).assemble(), std::invalid_argument);
    EXPECT_THROW(integral(begin, end)(jump(u) * jump(v)).assemble(), std::invalid_argument);
}

}   // namespace fdapde
