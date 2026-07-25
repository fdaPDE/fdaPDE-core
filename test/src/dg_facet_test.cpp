#include <fdaPDE/finite_elements.h>
#include <gtest/gtest.h>

namespace fdapde {

namespace {

Eigen::Matrix<double, 6, 6> jump_matrix() {
    Eigen::Matrix<double, 6, 6> matrix;
    matrix << 0, 0, 0, 0, 0, 0, 0, 2, 1, -2, -1, 0, 0, 1, 2, -1, -2, 0, 0, -2, -1, 2, 1, 0, 0, -1, -2, 1, 2, 0, 0, 0, 0,
      0, 0, 0;
    return matrix;
}

Eigen::Matrix<double, 6, 6> consistency_matrix() {
    Eigen::Matrix<double, 6, 6> matrix;
    matrix << 0, 1, 1, -1, -1, 0, 1, -1, -1, 1, 1, -1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 0,
      -1, -1, 1, 1, 0;
    return 0.5 * matrix;
}

Eigen::Matrix<double, 6, 6> volume_matrix() {
    Eigen::Matrix<double, 6, 6> matrix = Eigen::Matrix<double, 6, 6>::Zero();
    matrix.template topLeftCorner<3, 3>() << 1, -0.5, -0.5, -0.5, 0.5, 0, -0.5, 0, 0.5;
    matrix.template bottomRightCorner<3, 3>() << 0.5, 0, -0.5, 0, 0.5, -0.5, -0.5, -0.5, 1;
    return matrix;
}

}   // namespace

TEST(dg_geometry, planar_line_normal_is_perpendicular) {
    Eigen::Matrix<double, 2, 2> coordinates;
    coordinates << 0, 2, 0, 0;
    Simplex<1, 2> edge(coordinates);

    EXPECT_NEAR(edge.normal().norm(), 1.0, 1e-14);
    EXPECT_NEAR(edge.normal().dot(edge.J().col(0)), 0.0, 1e-14);
}

TEST(dg_facet_assembler, assembles_all_four_jump_blocks_once) {
    auto mesh = Triangulation<2, 2>::UnitSquare(2);
    FeSpace space(mesh, DG<1, 1>);
    TrialFunction u(space);
    TestFunction v(space);

    auto h = facet_size(mesh);
    Eigen::Matrix<double, 6, 6> assembled = Eigen::MatrixXd(integral(mesh)((1.0 / h) * jump(u) * jump(v)).assemble());
    Eigen::Matrix<double, 6, 6> expected = jump_matrix() / 6.0;

    EXPECT_LT((assembled - expected).norm(), 1e-12);
    Eigen::Matrix<double, 6, 1> constant = Eigen::Matrix<double, 6, 1>::Ones();
    EXPECT_NEAR(constant.dot(assembled * constant), 0.0, 1e-12);
}

TEST(dg_facet_assembler, sipg_form_matches_exact_two_cell_patch) {
    auto mesh = Triangulation<2, 2>::UnitSquare(2);
    FeSpace space(mesh, DG<1, 1>);
    TrialFunction u(space);
    TestFunction v(space);
    auto n = facet_normal(mesh);
    auto h = facet_size(mesh);
    constexpr double penalty = 10.0;

    auto form = dot(grad(u), grad(v)) - dot(avg(grad(u)), n) * jump(v) - dot(avg(grad(v)), n) * jump(u) +
                (penalty / h) * jump(u) * jump(v);
    Eigen::Matrix<double, 6, 6> assembled = Eigen::MatrixXd(integral(mesh)(form).assemble());
    Eigen::Matrix<double, 6, 6> expected = volume_matrix() + consistency_matrix() + (penalty / 6.0) * jump_matrix();

    EXPECT_LT((assembled - expected).norm(), 1e-12);
    EXPECT_LT((assembled - assembled.transpose()).norm(), 1e-12);

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> eigensolver(assembled);
    ASSERT_EQ(eigensolver.info(), Eigen::Success);
    EXPECT_GT(eigensolver.eigenvalues()[1], 0.49);
    EXPECT_GT(eigensolver.eigenvalues().minCoeff(), -1e-11);
}

}   // namespace fdapde
