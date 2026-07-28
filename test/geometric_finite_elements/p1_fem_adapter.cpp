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

#include <array>
#include <cstddef>
#include <numeric>

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
}
