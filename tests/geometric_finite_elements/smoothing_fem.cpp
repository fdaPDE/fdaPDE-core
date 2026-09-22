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
