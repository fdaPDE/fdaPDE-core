#include <fdaPDE/finite_elements.h>

namespace fdapde {

TEST(dg_facet_assembler, evaluates_spatial_coefficients_at_facet_quadrature_nodes) {
    auto mesh = Triangulation<2, 2>::UnitSquare(2);
    FeSpace space(mesh, DG<1, 1>);
    TrialFunction u(space);
    TestFunction v(space);
    auto x = [](const Eigen::Matrix<double, 2, 1>& point) { return point[0]; };
    ScalarField<2, decltype(x)> coefficient(x);

    Eigen::Matrix<double, 6, 6> assembled =
      Eigen::MatrixXd(integral(mesh)(coefficient * jump(u) * jump(v)).assemble());
    Eigen::Matrix<double, 6, 6> expected = Eigen::Matrix<double, 6, 6>::Zero();
    expected.template block<4, 4>(1, 1) <<
       1.0 / 4.0,  1.0 / 12.0, -1.0 / 4.0, -1.0 / 12.0,
       1.0 / 12.0, 1.0 / 12.0, -1.0 / 12.0, -1.0 / 12.0,
      -1.0 / 4.0, -1.0 / 12.0,  1.0 / 4.0,  1.0 / 12.0,
      -1.0 / 12.0, -1.0 / 12.0, 1.0 / 12.0,  1.0 / 12.0;
    expected *= std::sqrt(2.0);

    EXPECT_LT((assembled - expected).norm(), 1e-12);
}

TEST(dg_linear_facet_assembler, evaluates_spatial_coefficients_at_facet_quadrature_nodes) {
    auto mesh = Triangulation<2, 2>::UnitSquare(2);
    FeSpace space(mesh, DG<1, 1>);
    TestFunction v(space);
    auto x = [](const Eigen::Matrix<double, 2, 1>& point) { return point[0]; };
    ScalarField<2, decltype(x)> coefficient(x);

    Eigen::Matrix<double, 6, 1> assembled = integral(mesh)(coefficient * jump(v)).assemble();
    Eigen::Matrix<double, 6, 1> expected;
    expected << 0, 1.0 / 3.0, 1.0 / 6.0, -1.0 / 3.0, -1.0 / 6.0, 0;
    expected *= std::sqrt(2.0);

    EXPECT_LT((assembled - expected).norm(), 1e-12);
}

TEST(dg_linear_facet_assembler, evaluates_vector_coefficients_at_facet_quadrature_nodes) {
    auto mesh = Triangulation<2, 2>::UnitSquare(2);
    FeSpace space(mesh, DG<1, 1>);
    TestFunction v(space);
    VectorField<2, 2, std::function<double(Eigen::Matrix<double, 2, 1>)>> coefficient;
    coefficient[0] = [](const Eigen::Matrix<double, 2, 1>& point) { return point[0]; };
    coefficient[1] = []([[maybe_unused]] const Eigen::Matrix<double, 2, 1>& point) { return 0.0; };

    Eigen::Matrix<double, 6, 1> assembled =
      integral(mesh)(dot(coefficient, facet_normal(mesh)) * jump(v)).assemble();
    Eigen::Matrix<double, 6, 1> expected;
    expected << 0, 1.0 / 3.0, 1.0 / 6.0, -1.0 / 3.0, -1.0 / 6.0, 0;

    EXPECT_LT((assembled - expected).norm(), 1e-12);
}

TEST(dg_linear_facet_assembler, rejects_cell_quadrature_arrays_on_facets) {
    auto mesh = Triangulation<2, 2>::UnitSquare(2);
    FeSpace space(mesh, DG<1, 1>);
    TestFunction v(space);
    Eigen::VectorXd cell_values = Eigen::VectorXd::Ones(6);
    FeCoeff<2, 1, 1, Eigen::VectorXd> coefficient(cell_values);

    EXPECT_THROW(integral(mesh)(coefficient * jump(v)).assemble(), std::logic_error);
}

}   // namespace fdapde
