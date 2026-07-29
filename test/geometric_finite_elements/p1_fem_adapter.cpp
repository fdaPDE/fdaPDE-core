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

#include <fdaPDE/geometric_finite_elements_fem.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
#include <numeric>
#include <vector>

namespace {

using SegmentSpace = fdapde::FeSpace<fdapde::Triangulation<1, 1>, fdapde::FeP<1, 1>>;
using TriangleSpace = fdapde::FeSpace<fdapde::Triangulation<2, 2>, fdapde::FeP<1, 1>>;
using SurfaceTriangleSpace = fdapde::FeSpace<fdapde::Triangulation<2, 3>, fdapde::FeP<1, 1>>;
using TetrahedronSpace = fdapde::FeSpace<fdapde::Triangulation<3, 3>, fdapde::FeP<1, 1>>;

template <typename Space>
concept HasDefaultP1FEMAdapter =
  requires(const Space& space) { fdapde::gfe::p1_fem_cell_quadrature(space, std::size_t {0}); };

static_assert(HasDefaultP1FEMAdapter<SegmentSpace>);
static_assert(HasDefaultP1FEMAdapter<TriangleSpace>);
static_assert(HasDefaultP1FEMAdapter<SurfaceTriangleSpace>);
static_assert(HasDefaultP1FEMAdapter<TetrahedronSpace>);
static_assert(!HasDefaultP1FEMAdapter<fdapde::FeSpace<fdapde::Triangulation<2, 2>, fdapde::FeP<2, 1>>>);
static_assert(!HasDefaultP1FEMAdapter<fdapde::FeSpace<fdapde::Triangulation<2, 2>, fdapde::FeP<1, 2>>>);
static_assert(!HasDefaultP1FEMAdapter<fdapde::FeSpace<fdapde::Triangulation<2, 2>, fdapde::FeDG<1, 1>>>);

template <typename Packet> void expect_partition_and_scale(const Packet& packet, double measure) {
    double integration_weight_sum = 0;
    for (std::size_t q = 0; q < Packet::quadrature_size; ++q) {
        double weight_sum = 0;
        for (const double weight : packet.barycentric_weights[q]) {
            EXPECT_GE(weight, 0);
            weight_sum += weight;
        }
        EXPECT_NEAR(weight_sum, 1, 1.0e-14);
        EXPECT_GT(packet.integration_weights[q], 0);
        integration_weight_sum += packet.integration_weights[q];
    }
    EXPECT_NEAR(integration_weight_sum, measure, 1.0e-13 * std::max(1.0, measure));

    for (std::size_t axis = 0; axis < Packet::embed_dim; ++axis) {
        const double gradient_sum = std::accumulate(
          packet.physical_weight_gradients[axis].begin(), packet.physical_weight_gradients[axis].end(), 0.0);
        EXPECT_NEAR(gradient_sum, 0, 1.0e-14);
    }
}

fdapde::Triangulation<2, 2> make_triangle() {
    Eigen::Matrix<double, fdapde::Dynamic, fdapde::Dynamic> nodes(3, 2);
    nodes << 0, 0, 2, 0, 0, 3;
    Eigen::Matrix<int, fdapde::Dynamic, fdapde::Dynamic> cells(1, 3);
    cells << 0, 1, 2;
    Eigen::Matrix<int, fdapde::Dynamic, fdapde::Dynamic> boundary =
      Eigen::Matrix<int, fdapde::Dynamic, fdapde::Dynamic>::Ones(3, 1);
    return {nodes, cells, boundary};
}

fdapde::Triangulation<2, 2> make_flat_surface_triangle() {
    Eigen::Matrix<double, fdapde::Dynamic, fdapde::Dynamic> nodes(3, 2);
    nodes << 0, 0, 2, 0, 0, 5;
    Eigen::Matrix<int, fdapde::Dynamic, fdapde::Dynamic> cells(1, 3);
    cells << 0, 1, 2;
    Eigen::Matrix<int, fdapde::Dynamic, fdapde::Dynamic> boundary =
      Eigen::Matrix<int, fdapde::Dynamic, fdapde::Dynamic>::Ones(3, 1);
    return {nodes, cells, boundary};
}

fdapde::Triangulation<2, 2> make_two_triangles_with_permuted_second_cell() {
    Eigen::Matrix<double, fdapde::Dynamic, fdapde::Dynamic> nodes(4, 2);
    nodes << 0, 0, 2, 0, 0, 3, 2, 3;
    Eigen::Matrix<int, fdapde::Dynamic, fdapde::Dynamic> cells(2, 3);
    cells << 0, 1, 2, 3, 2, 1;
    Eigen::Matrix<int, fdapde::Dynamic, fdapde::Dynamic> boundary =
      Eigen::Matrix<int, fdapde::Dynamic, fdapde::Dynamic>::Ones(4, 1);
    return {nodes, cells, boundary};
}

fdapde::Triangulation<2, 3> make_surface_triangle() {
    Eigen::Matrix<double, fdapde::Dynamic, fdapde::Dynamic> nodes(3, 3);
    nodes << 0, 0, 0, 2, 0, 0, 0, 3, 4;
    Eigen::Matrix<int, fdapde::Dynamic, fdapde::Dynamic> cells(1, 3);
    cells << 0, 1, 2;
    Eigen::Matrix<int, fdapde::Dynamic, fdapde::Dynamic> boundary =
      Eigen::Matrix<int, fdapde::Dynamic, fdapde::Dynamic>::Ones(3, 1);
    return {nodes, cells, boundary};
}

fdapde::Triangulation<3, 3> make_tetrahedron() {
    Eigen::Matrix<double, fdapde::Dynamic, fdapde::Dynamic> nodes(4, 3);
    nodes << 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 4;
    Eigen::Matrix<int, fdapde::Dynamic, fdapde::Dynamic> cells(1, 4);
    cells << 0, 1, 2, 3;
    Eigen::Matrix<int, fdapde::Dynamic, fdapde::Dynamic> boundary =
      Eigen::Matrix<int, fdapde::Dynamic, fdapde::Dynamic>::Ones(4, 1);
    return {nodes, cells, boundary};
}

template <typename Packet, std::size_t NodeCount>
void expect_dofs(const Packet& packet, const std::array<std::size_t, NodeCount>& expected) {
    static_assert(Packet::node_count == NodeCount);
    EXPECT_EQ(packet.dofs, expected);
}

template <typename Space> void expect_lumped_stencil_matches_classical_assembly(const Space& space) {
    const auto stencil = fdapde::gfe::p1_lumped_laplacian_stencil(space);
    fdapde::TrialFunction u(space);
    fdapde::TestFunction v(space);
    const auto stiffness =
      fdapde::integral(space.triangulation())(fdapde::dot(fdapde::grad(u), fdapde::grad(v))).assemble();
    const auto mass = fdapde::integral(space.triangulation())(u * v).assemble();

    Eigen::MatrixXd reconstructed = Eigen::MatrixXd::Zero(space.n_dofs(), space.n_dofs());
    for (const auto& edge : stencil.edges) {
        reconstructed(static_cast<Eigen::Index>(edge.first), static_cast<Eigen::Index>(edge.second)) = edge.stiffness;
        reconstructed(static_cast<Eigen::Index>(edge.second), static_cast<Eigen::Index>(edge.first)) = edge.stiffness;
        reconstructed(static_cast<Eigen::Index>(edge.first), static_cast<Eigen::Index>(edge.first)) -= edge.stiffness;
        reconstructed(static_cast<Eigen::Index>(edge.second), static_cast<Eigen::Index>(edge.second)) -= edge.stiffness;
    }
    const Eigen::MatrixXd expected_stiffness(stiffness);
    ASSERT_EQ(reconstructed.rows(), expected_stiffness.rows());
    ASSERT_EQ(reconstructed.cols(), expected_stiffness.cols());
    for (Eigen::Index row = 0; row < reconstructed.rows(); ++row) {
        for (Eigen::Index col = 0; col < reconstructed.cols(); ++col) {
            EXPECT_NEAR(reconstructed(row, col), expected_stiffness(row, col), 2.0e-13);
        }
    }

    const Eigen::VectorXd expected_masses = mass * Eigen::VectorXd::Ones(space.n_dofs());
    ASSERT_EQ(stencil.lumped_masses.size(), static_cast<std::size_t>(expected_masses.size()));
    for (Eigen::Index node = 0; node < expected_masses.size(); ++node) {
        EXPECT_NEAR(stencil.lumped_masses[static_cast<std::size_t>(node)], expected_masses[node], 2.0e-13);
    }
}

}   // namespace

TEST(P1FEMAdapter, MapsSegmentTopologyGradientsAndQuadratureScale) {
    const auto mesh = fdapde::Triangulation<1, 1>::Interval(2, 5, 2);
    const SegmentSpace space(mesh, fdapde::P1<1>);
    const auto packet = fdapde::gfe::p1_fem_cell_quadrature(space, 0);

    static_assert(decltype(packet)::local_dim == 1);
    static_assert(decltype(packet)::embed_dim == 1);
    static_assert(decltype(packet)::quadrature_size == 2);
    expect_dofs(packet, std::array<std::size_t, 2> {0, 1});
    EXPECT_NEAR(packet.physical_weight_gradients[0][0], -1.0 / 3.0, 1.0e-15);
    EXPECT_NEAR(packet.physical_weight_gradients[0][1], 1.0 / 3.0, 1.0e-15);
    expect_partition_and_scale(packet, 3);
}

TEST(P1FEMAdapter, MapsTriangleTopologyGradientsAndQuadratureScale) {
    const auto mesh = make_triangle();
    const TriangleSpace space(mesh, fdapde::P1<1>);
    const auto packet = fdapde::gfe::p1_fem_cell_quadrature(space, 0);

    expect_dofs(packet, std::array<std::size_t, 3> {0, 1, 2});
    EXPECT_NEAR(packet.physical_weight_gradients[0][0], -0.5, 1.0e-15);
    EXPECT_NEAR(packet.physical_weight_gradients[0][1], 0.5, 1.0e-15);
    EXPECT_NEAR(packet.physical_weight_gradients[0][2], 0, 1.0e-15);
    EXPECT_NEAR(packet.physical_weight_gradients[1][0], -1.0 / 3.0, 1.0e-15);
    EXPECT_NEAR(packet.physical_weight_gradients[1][1], 0, 1.0e-15);
    EXPECT_NEAR(packet.physical_weight_gradients[1][2], 1.0 / 3.0, 1.0e-15);
    expect_partition_and_scale(packet, 3);
}

TEST(P1FEMAdapter, PreservesCellVertexOrderInDofsAndBarycentricColumns) {
    const auto mesh = make_two_triangles_with_permuted_second_cell();
    const TriangleSpace space(mesh, fdapde::P1<1>);
    const auto packet = fdapde::gfe::p1_fem_cell_quadrature(space, 1);

    expect_dofs(packet, std::array<std::size_t, 3> {3, 2, 1});
    EXPECT_NEAR(packet.barycentric_weights[1][0], 1.0 / 6.0, 1.0e-14);
    EXPECT_NEAR(packet.barycentric_weights[1][1], 2.0 / 3.0, 1.0e-14);
    EXPECT_NEAR(packet.barycentric_weights[1][2], 1.0 / 6.0, 1.0e-14);
}

TEST(P1FEMAdapter, MapsEmbeddedTriangleGradientsInEveryPhysicalAxis) {
    const auto mesh = make_surface_triangle();
    const SurfaceTriangleSpace space(mesh, fdapde::P1<1>);
    const auto packet = fdapde::gfe::p1_fem_cell_quadrature(space, 0);

    expect_dofs(packet, std::array<std::size_t, 3> {0, 1, 2});
    const std::array<std::array<double, 3>, 3> expected {
      {
       {{-0.5, 0.5, 0}},
       {{-3.0 / 25.0, 0, 3.0 / 25.0}},
       {{-4.0 / 25.0, 0, 4.0 / 25.0}},
       }
    };
    for (std::size_t axis = 0; axis < expected.size(); ++axis) {
        for (std::size_t node = 0; node < expected[axis].size(); ++node) {
            EXPECT_NEAR(packet.physical_weight_gradients[axis][node], expected[axis][node], 1.0e-15);
        }
    }
    expect_partition_and_scale(packet, 5);
}

TEST(P1FEMAdapter, MapsTetrahedronTopologyGradientsAndQuadratureScale) {
    const auto mesh = make_tetrahedron();
    const TetrahedronSpace space(mesh, fdapde::P1<1>);
    const auto packet = fdapde::gfe::p1_fem_cell_quadrature(space, 0);

    expect_dofs(packet, std::array<std::size_t, 4> {0, 1, 2, 3});
    const std::array<std::array<double, 4>, 3> expected {
      {
       {{-0.5, 0.5, 0, 0}},
       {{-1.0 / 3.0, 0, 1.0 / 3.0, 0}},
       {{-0.25, 0, 0, 0.25}},
       }
    };
    for (std::size_t axis = 0; axis < expected.size(); ++axis) {
        for (std::size_t node = 0; node < expected[axis].size(); ++node) {
            EXPECT_NEAR(packet.physical_weight_gradients[axis][node], expected[axis][node], 1.0e-15);
        }
    }
    expect_partition_and_scale(packet, 4);
}

TEST(P1FEMAdapter, SupportsExplicitPositiveRulesAndRejectsNegativeRules) {
    const auto mesh = make_triangle();
    const TriangleSpace space(mesh, fdapde::P1<1>);
    const auto packet = fdapde::gfe::p1_fem_cell_quadrature(space, 0, fdapde::QS2DP4);
    static_assert(decltype(packet)::quadrature_size == 6);
    expect_partition_and_scale(packet, 3);

    EXPECT_THROW(fdapde::gfe::p1_fem_cell_quadrature(space, 0, fdapde::QS2DP3), std::invalid_argument);
    EXPECT_THROW(fdapde::gfe::p1_fem_cell_quadrature(space, 1), std::out_of_range);

    const TriangleSpace uninitialized {};
    EXPECT_THROW(fdapde::gfe::p1_fem_cell_quadrature(uninitialized, 0), std::logic_error);
    EXPECT_THROW(fdapde::gfe::p1_lumped_laplacian_stencil(uninitialized), std::logic_error);
}

TEST(P1FEMAdapter, BuildsNativeLumpedStencilMatchingClassicalFiniteElementAssembly) {
    const auto segment_mesh = fdapde::Triangulation<1, 1>::Interval(2, 5, 2);
    const SegmentSpace segment_space(segment_mesh, fdapde::P1<1>);
    expect_lumped_stencil_matches_classical_assembly(segment_space);

    const auto triangle_mesh = make_triangle();
    const TriangleSpace triangle_space(triangle_mesh, fdapde::P1<1>);
    expect_lumped_stencil_matches_classical_assembly(triangle_space);
    const auto triangle_stencil = fdapde::gfe::p1_lumped_laplacian_stencil(triangle_space);
    ASSERT_EQ(triangle_stencil.edges.size(), 2);
    EXPECT_EQ(triangle_stencil.edges[0].first, 0);
    EXPECT_EQ(triangle_stencil.edges[0].second, 1);
    EXPECT_NEAR(triangle_stencil.edges[0].stiffness, -0.75, 1.0e-15);
    EXPECT_EQ(triangle_stencil.edges[1].first, 0);
    EXPECT_EQ(triangle_stencil.edges[1].second, 2);
    EXPECT_NEAR(triangle_stencil.edges[1].stiffness, -1.0 / 3.0, 1.0e-15);

    const auto surface_mesh = make_surface_triangle();
    const SurfaceTriangleSpace surface_space(surface_mesh, fdapde::P1<1>);
    expect_lumped_stencil_matches_classical_assembly(surface_space);
    const auto flat_surface_mesh = make_flat_surface_triangle();
    const TriangleSpace flat_surface_space(flat_surface_mesh, fdapde::P1<1>);
    const auto flat_surface_stencil = fdapde::gfe::p1_lumped_laplacian_stencil(flat_surface_space);
    const auto surface_stencil = fdapde::gfe::p1_lumped_laplacian_stencil(surface_space);
    ASSERT_EQ(flat_surface_stencil.lumped_masses.size(), surface_stencil.lumped_masses.size());
    for (std::size_t node = 0; node < surface_stencil.node_count(); ++node) {
        EXPECT_NEAR(flat_surface_stencil.lumped_masses[node], surface_stencil.lumped_masses[node], 2.0e-13);
    }
    ASSERT_EQ(flat_surface_stencil.edges.size(), surface_stencil.edges.size());
    for (std::size_t edge = 0; edge < surface_stencil.edges.size(); ++edge) {
        EXPECT_EQ(flat_surface_stencil.edges[edge].first, surface_stencil.edges[edge].first);
        EXPECT_EQ(flat_surface_stencil.edges[edge].second, surface_stencil.edges[edge].second);
        EXPECT_NEAR(flat_surface_stencil.edges[edge].stiffness, surface_stencil.edges[edge].stiffness, 2.0e-13);
    }

    const auto tetrahedron_mesh = make_tetrahedron();
    const TetrahedronSpace tetrahedron_space(tetrahedron_mesh, fdapde::P1<1>);
    expect_lumped_stencil_matches_classical_assembly(tetrahedron_space);
}

TEST(P1FEMAdapter, LumpedStencilIsCanonicalUnderCellAndLocalNodePermutation) {
    const auto mesh = make_two_triangles_with_permuted_second_cell();
    const TriangleSpace space(mesh, fdapde::P1<1>);
    using Packet = decltype(fdapde::gfe::p1_fem_cell_quadrature(space, 0));
    std::vector<Packet> packets {
      fdapde::gfe::p1_fem_cell_quadrature(space, 0), fdapde::gfe::p1_fem_cell_quadrature(space, 1)};
    const auto forward = fdapde::gfe::p1_lumped_laplacian_stencil(4, std::span<const Packet>(packets));
    std::reverse(packets.begin(), packets.end());
    std::swap(packets[0].dofs[0], packets[0].dofs[2]);
    for (auto& direction : packets[0].physical_weight_gradients) { std::swap(direction[0], direction[2]); }
    for (auto& weights : packets[0].barycentric_weights) { std::swap(weights[0], weights[2]); }
    const auto reversed = fdapde::gfe::p1_lumped_laplacian_stencil(4, std::span<const Packet>(packets));

    EXPECT_EQ(forward.lumped_masses, reversed.lumped_masses);
    ASSERT_EQ(forward.edges.size(), reversed.edges.size());
    for (std::size_t edge = 0; edge < forward.edges.size(); ++edge) {
        EXPECT_EQ(forward.edges[edge].first, reversed.edges[edge].first);
        EXPECT_EQ(forward.edges[edge].second, reversed.edges[edge].second);
        EXPECT_DOUBLE_EQ(forward.edges[edge].stiffness, reversed.edges[edge].stiffness);
    }
}

TEST(P1FEMAdapter, LumpedStencilPreservesPositiveOffDiagonalStiffness) {
    using Packet = fdapde::gfe::P1FEMCellQuadrature<2, 2, 1>;
    Packet packet {};
    packet.dofs = {0, 1, 2};
    packet.physical_weight_gradients = {{{{-1, 1, 0}}, {{-1.5, 0.5, 1}}}};
    packet.barycentric_weights = {{{{1.0 / 3, 1.0 / 3, 1.0 / 3}}}};
    packet.integration_weights = {0.5};
    const std::array<Packet, 1> packets {packet};
    const auto stencil = fdapde::gfe::p1_lumped_laplacian_stencil(3, std::span<const Packet>(packets));

    ASSERT_EQ(stencil.edges.size(), 3);
    EXPECT_NEAR(stencil.edges[0].stiffness, -0.875, 1.0e-15);
    EXPECT_NEAR(stencil.edges[1].stiffness, -0.75, 1.0e-15);
    EXPECT_NEAR(stencil.edges[2].stiffness, 0.25, 1.0e-15);
}

TEST(P1FEMAdapter, LumpedStencilRejectsMalformedPacketsAndIsolatedNodes) {
    const auto mesh = make_triangle();
    const TriangleSpace space(mesh, fdapde::P1<1>);
    using Packet = decltype(fdapde::gfe::p1_fem_cell_quadrature(space, 0));
    const Packet valid = fdapde::gfe::p1_fem_cell_quadrature(space, 0);
    std::array<Packet, 1> packets {valid};

    EXPECT_THROW(fdapde::gfe::p1_lumped_laplacian_stencil(0, std::span<const Packet>(packets)), std::invalid_argument);
    EXPECT_THROW(fdapde::gfe::p1_lumped_laplacian_stencil(3, std::span<const Packet> {}), std::invalid_argument);
    EXPECT_THROW(fdapde::gfe::p1_lumped_laplacian_stencil(4, std::span<const Packet>(packets)), std::domain_error);

    packets[0].dofs[1] = packets[0].dofs[0];
    EXPECT_THROW(fdapde::gfe::p1_lumped_laplacian_stencil(3, std::span<const Packet>(packets)), std::invalid_argument);
    packets[0] = valid;
    packets[0].dofs[2] = 3;
    EXPECT_THROW(fdapde::gfe::p1_lumped_laplacian_stencil(3, std::span<const Packet>(packets)), std::out_of_range);

    packets[0] = valid;
    packets[0].physical_weight_gradients[0][0] = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(fdapde::gfe::p1_lumped_laplacian_stencil(3, std::span<const Packet>(packets)), std::invalid_argument);
    packets[0] = valid;
    packets[0].physical_weight_gradients[0][0] += 1;
    EXPECT_THROW(fdapde::gfe::p1_lumped_laplacian_stencil(3, std::span<const Packet>(packets)), std::invalid_argument);
    packets[0] = valid;
    packets[0].integration_weights[0] = -1;
    EXPECT_THROW(fdapde::gfe::p1_lumped_laplacian_stencil(3, std::span<const Packet>(packets)), std::invalid_argument);
    packets[0] = valid;
    packets[0].integration_weights[0] = std::numeric_limits<double>::infinity();
    EXPECT_THROW(fdapde::gfe::p1_lumped_laplacian_stencil(3, std::span<const Packet>(packets)), std::invalid_argument);
    packets[0] = valid;
    packets[0].integration_weights.fill(0);
    EXPECT_THROW(fdapde::gfe::p1_lumped_laplacian_stencil(3, std::span<const Packet>(packets)), std::domain_error);
    packets[0] = valid;
    packets[0].physical_weight_gradients[0] = {
      std::numeric_limits<double>::max(), -std::numeric_limits<double>::max(), 0};
    EXPECT_THROW(fdapde::gfe::p1_lumped_laplacian_stencil(3, std::span<const Packet>(packets)), std::domain_error);
}
