#include <fdaPDE/finite_elements.h>
#include <gtest/gtest.h>

namespace fdapde {

namespace {

struct two_by_three_coefficient : MatrixFieldBase<2, two_by_three_coefficient> {
    using InputType = Eigen::Vector2d;
    using Scalar = double;
    static constexpr int StaticInputSize = 2;
    [[maybe_unused]] static constexpr int NestAsRef = 0;
    [[maybe_unused]] static constexpr int XprBits = 0;
    [[maybe_unused]] static constexpr int ReadOnly = 1;
    static constexpr int Rows = 2;
    static constexpr int Cols = 3;

    template <typename Point> Eigen::Matrix<double, Rows, Cols> operator()(const Point& point) const {
        Eigen::Matrix<double, Rows, Cols> value;
        value << point[0], point[1], point[0] + point[1], point[0] - point[1], 2.0 * point[0], 2.0 * point[1];
        return value;
    }

    template <typename Point> double eval(int i, int j, const Point& point) const { return operator()(point)(i, j); }

    constexpr int rows() const { return Rows; }
    constexpr int cols() const { return Cols; }
    constexpr int input_size() const { return StaticInputSize; }
};

}   // namespace

TEST(fe_map, preserves_matrix_field_coefficients_on_cell_quadrature_nodes) {
    Eigen::MatrixXd nodes(2, 2);
    nodes << 2.0, 3.0, -1.0, 4.0;
    FeMap mapped(two_by_three_coefficient {});
    mapped.init(nodes, 0, 0);

    internals::fe_assembler_packet<2> packet;
    packet.quad_node_id = 0;
    Eigen::Matrix<double, 2, 3> expected;
    expected << 2.0, 3.0, 5.0, -1.0, 4.0, 6.0;
    EXPECT_EQ(mapped(packet), expected);
    EXPECT_DOUBLE_EQ(mapped.eval(1, 2, packet), expected(1, 2));

    packet.quad_node_id = 1;
    expected << -1.0, 4.0, 3.0, -5.0, -2.0, 8.0;
    EXPECT_EQ(mapped(packet), expected);
    EXPECT_DOUBLE_EQ(mapped.eval(1, 2, packet), expected(1, 2));
}

TEST(dg_facet_assembler, evaluates_spatial_coefficients_at_facet_quadrature_nodes) {
    auto mesh = Triangulation<2, 2>::UnitSquare(2);
    FeSpace space(mesh, DG<1, 1>);
    TrialFunction u(space);
    TestFunction v(space);
    auto x = [](const Eigen::Matrix<double, 2, 1>& point) { return point[0]; };
    ScalarField<2, decltype(x)> coefficient(x);

    Eigen::Matrix<double, 6, 6> assembled = Eigen::MatrixXd(integral(mesh)(coefficient * jump(u) * jump(v)).assemble());
    Eigen::Matrix<double, 6, 6> expected = Eigen::Matrix<double, 6, 6>::Zero();
    expected.template block<4, 4>(1, 1) << 1.0 / 4.0, 1.0 / 12.0, -1.0 / 4.0, -1.0 / 12.0, 1.0 / 12.0, 1.0 / 12.0,
      -1.0 / 12.0, -1.0 / 12.0, -1.0 / 4.0, -1.0 / 12.0, 1.0 / 4.0, 1.0 / 12.0, -1.0 / 12.0, -1.0 / 12.0, 1.0 / 12.0,
      1.0 / 12.0;
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

    Eigen::Matrix<double, 6, 1> assembled = integral(mesh)(dot(coefficient, facet_normal(mesh)) * jump(v)).assemble();
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
