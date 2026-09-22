// SPDX-License-Identifier: GPL-3.0-or-later
#include <fdaPDE/geometric_finite_elements_fem.h>
#include <gtest/gtest.h>

namespace {
using namespace fdapde;
/// @brief checks FEM stencil assembly against independent gradients on the unit right triangle
TEST(GeometricSmoothingFEM, NativeSpaceAndScalarAssembly) {
    Eigen::Matrix<double, 3, 2> vertices;
    vertices << 0, 0, 1, 0, 0, 1;
    Eigen::Matrix<int, 1, 3> cells;
    cells << 0, 1, 2;
    const Triangulation<2, 2> mesh(vertices, cells, Eigen::Matrix<int, 3, 1>::Ones());
    using LE = manifold::LogEuclideanSPDGeometry<double, 2>;
    const GeometricFeSpace space(mesh, P1<1>, LE {});
    const auto packet = gfe::p1_fem_cell_quadrature(space, 0, QS2DP4);
    const auto stencil = gfe::p1_lumped_laplacian_stencil(space, QS2DP4);
    // P1 lumping divides area 1/2 equally between the three scalar nodal DOFs
    for (double mass : stencil.lumped_masses) EXPECT_NEAR(mass, 1. / 6, 1e-15);
    // orthogonal gradients at the last two vertices give zero stiffness between them
    ASSERT_EQ(stencil.edges.size(), 2u);
    // the first basis gradient has inner product -1 with each remaining gradient and area is 1/2
    EXPECT_NEAR(stencil.edges[0].stiffness, -.5, 1e-15);
    // quadrature integration weights reproduce the exact area
    EXPECT_NEAR(std::accumulate(packet.integration_weights.begin(), packet.integration_weights.end(), 0.), .5, 1e-15);
    const FeSpace scalar(mesh, P1<1>);
    const auto reference = gfe::p1_lumped_laplacian_stencil(scalar, QS2DP4);
    // wrapping a scalar space must not change masses or DOF order
    EXPECT_EQ(reference.lumped_masses, stencil.lumped_masses);
    // out-of-mesh cells are rejected before accessing the DOF handler
    EXPECT_THROW(gfe::p1_fem_cell_quadrature(space, 1, QS2DP4), std::out_of_range);
}
/// @brief checks the recovered lumped tension against the smooth Neumann eigenfunction energy
TEST(GeometricSmoothingFEM, NeumannRefinement) {
    constexpr double pi = 3.14159265358979323846;
    double previous = std::numeric_limits<double>::infinity();
    for (int side : {9, 17, 33}) {
        const auto mesh = Triangulation<2, 2>::UnitSquare(side);
        using LE = manifold::LogEuclideanSPDGeometry<double, 2, Usage::InterpolationNodes>;
        const GeometricFeSpace space(mesh, P1<1>, LE {});
        const auto stencil = gfe::p1_lumped_laplacian_stencil(space, QS2DP4);
        MatrixBatch<LE::Point> coefficients(space.n_dofs());
        for (int i = 0; i < space.n_dofs(); ++i) {
            const double h = std::cos(pi * mesh.nodes()(i, 0)) * std::cos(pi * mesh.nodes()(i, 1));
            coefficients[i] = Matrix<double, 2, 2>({std::exp(h), 0., 0., 1.});
        }
        const auto energy = gfe::p1_discrete_tension_value(LE {}, coefficients, stencil);
        const double relative = std::abs(energy.value / (.5 * pi * pi * pi * pi) - 1);
        // the smooth eigenfunction has zero normal derivative and exact half-bienergy pi^4/2
        EXPECT_LT(relative, previous);
        previous = relative;
    }
    // the finest grid reproduces the continuum Neumann oracle within the historical discretization error
    EXPECT_LT(previous, .002);
}
}   // namespace

namespace {
/// @brief changes rho between evaluations while retaining spatial plans and matrix coefficients
TEST(GeometricSmoothingFEM, MutableRhoPlans) {
    const auto mesh = fdapde::Triangulation<2, 2>::UnitSquare(3);
    using G = fdapde::manifold::CheegerLogEuclideanSPDGeometry<double, 3, fdapde::Usage::InterpolationNodes>;
    const G geometry;
    const fdapde::GeometricFeSpace space(mesh, fdapde::P1<1>, geometry);
    fdapde::MatrixBatch<G::Point> nodes(space.n_dofs());
    for (int i = 0; i < space.n_dofs(); ++i) {
        G::Tangent x;
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c <= r; ++c)
                x(r, c) = r == c ? .1 * (r + 1) * (1 + mesh.nodes()(i, 0)) : .08 * std::sin(i + r + c);
        nodes[i] = fdapde::matrix_exp(x);
    }
    fdapde::GeometricFeFunction field(space, nodes);
    fdapde::MatrixBatch<fdapde::Vector<double, 2>> points(2);
    points[0] = fdapde::Vector<double, 2>(.2, .3);
    points[1] = fdapde::Vector<double, 2>(.65, .7);
    const auto plan = space.prepare_evaluation(points);
    const auto initial = plan(field);
    // visiting interior sites prepares metric-dependent cell data
    EXPECT_GT(field.prepared_cells(), 0u);
    std::vector<double> rho(space.n_dofs());
    for (int i = 0; i < space.n_dofs(); ++i) rho[i] = .4 + .03 * i;
    const auto* cached_log = field.coeff()[0].cache().data();
    field.set_rho(rho);
    // replacing rho drops old cell lifts before a plan can reuse them
    EXPECT_EQ(field.prepared_cells(), 0u);
    // rho-only replacement preserves the original matrix cache buffer
    EXPECT_EQ(field.coeff()[0].cache().data(), cached_log);
    const auto updated = plan(field);
    const fdapde::GeometricFeSpace reference_space(mesh, fdapde::P1<1>, geometry, std::span<const double>(rho));
    const fdapde::GeometricFeFunction reference(reference_space, nodes);
    const auto expected = reference.eval_at(points);
    double change = 0;
    for (std::size_t i = 0; i < points.size(); ++i) {
        // the retained spatial plan agrees with a new field built using the replacement metric
        EXPECT_LT((fdapde::Matrix<double, 3, 3>(updated[i] - expected[i]).norm()), 1e-10);
        change += fdapde::Matrix<double, 3, 3>(updated[i] - initial[i]).norm();
    }
    // noncommuting data expose a material value change when the local rotational penalty changes
    EXPECT_GT(change, 1e-6);
    auto bad = rho;
    bad[0] = 0;
    // public updates reject nonpositive rho before discarding the last valid cache
    EXPECT_THROW(field.set_rho(bad), std::invalid_argument);
    // failed updates preserve the active scalar coefficients
    EXPECT_DOUBLE_EQ(field.rho_coefficients()[0], rho[0]);
    field.set_rho({});
    const auto restored = plan(field);
    // an empty override restores the geometry's constant rho using the same spatial plan
    EXPECT_LT((fdapde::Matrix<double, 3, 3>(restored[0] - initial[0]).norm()), 1e-10);
}
}   // namespace

namespace {
/// @brief recovers roundoff-stable boundary support from the TSPDE physical-point regression
TEST(GeometricSmoothingFEM, BarycentricBoundarySupport) {
    const auto mesh = Triangulation<2, 2>::UnitSquare(11);
    const FeSpace space(mesh, P1<1>);
    Eigen::MatrixXd locations(2, 2);
    locations << 493. / 499., 6. / 499., .123, .234;
    const auto cells = mesh.locate(locations);
    const auto edge = gfe::p1_fem_barycentric_weights(space, cells[0], locations.row(0).transpose());
    // the captured point x+y=1 has exactly two active shape functions despite inverse-map roundoff
    EXPECT_EQ(std::count(edge.begin(), edge.end(), 0.), 1);
    const auto interior = gfe::p1_fem_barycentric_weights(space, cells[1], locations.row(1).transpose());
    // interior support is preserved instead of being projected onto a boundary
    EXPECT_GT(*std::min_element(interior.begin(), interior.end()), 0);
    // normalization preserves the P1 partition of unity
    EXPECT_NEAR(std::accumulate(interior.begin(), interior.end(), 0.), 1, 1e-15);
    // a genuinely outside point is rejected rather than clamped onto the cell
    EXPECT_THROW(gfe::p1_fem_barycentric_weights(space, cells[0], Eigen::Vector2d(-1., -1.)), std::domain_error);
    // cell indices are checked before asking the DOF handler for its geometry
    EXPECT_THROW(gfe::p1_fem_barycentric_weights(space, mesh.n_cells(), Eigen::Vector2d(0., 0.)), std::out_of_range);
}
}   // namespace
