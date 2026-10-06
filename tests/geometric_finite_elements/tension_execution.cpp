// SPDX-License-Identifier: GPL-3.0-or-later
#include <fdaPDE/geometric_finite_elements.h>
#include <gtest/gtest.h>

namespace {
using namespace fdapde;

/// @brief generates noncommuting SPD values with signed edges shared by several nodes
template <typename Geometry> auto tension_execution_data(const Geometry& geometry) {
    MatrixBatch<typename Geometry::Point> nodes(41, geometry.order(), geometry.order());
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        typename Geometry::Tangent chart;
        if constexpr (Geometry::Tangent::Rows == Dynamic) chart.resize(geometry.order(), geometry.order());
        for (int row = 0; row < geometry.order(); ++row)
            for (int col = 0; col <= row; ++col)
                chart(row, col) = row == col ? .1 * row + .018 * i : .012 * std::sin(1 + i + 2 * row + col);
        nodes[i] = matrix_exp(chart);
    }
    gfe::P1LumpedLaplacianStencil stencil;
    stencil.lumped_masses.resize(nodes.size());
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        stencil.lumped_masses[i] = .7 + .13 * i;
        for (std::size_t j = i + 1; j < nodes.size(); ++j)
            if (j - i < 4) stencil.edges.push_back({i, j, (j - i == 3 ? .07 : -.2) * (1 + i)});
    }
    return std::pair {std::move(nodes), std::move(stencil)};
}

/// @brief checks the parallel gather against the unchanged serial edge-scatter implementation
template <typename Geometry> void check_tension_execution(const Geometry& geometry) {
    auto [nodes, stencil] = tension_execution_data(geometry);
    for (int repeat = 0; repeat < 2; ++repeat) {
        const auto parallel = gfe::p1_discrete_tension_contribution(geometry, nodes, stencil, execution_par);
        const auto serial = gfe::p1_discrete_tension_contribution(geometry, nodes, stencil);
        const auto sequential = gfe::p1_discrete_tension_contribution(geometry, nodes, stencil, execution_seq);
        // ordered nodal sums and the scalar reduction retain every serial rounding step
        EXPECT_EQ(serial.value, parallel.value);
        // an explicit sequential policy dispatches to the preserved original implementation
        EXPECT_EQ(serial.value, sequential.value);
        // skipping gradient preparation leaves exactly the serial value-only result
        EXPECT_EQ(
          gfe::p1_discrete_tension_value(geometry, nodes, stencil).value,
          gfe::p1_discrete_tension_value(geometry, nodes, stencil, execution_par).value);
        for (std::size_t i = 0; i < nodes.size(); ++i)
            for (int row = 0; row < geometry.order(); ++row)
                for (int col = 0; col <= row; ++col) {
                    // the original edge and direction order is retained for each individual gradient coefficient
                    EXPECT_EQ(serial.nodal_gradient[i](row, col), parallel.nodal_gradient[i](row, col));
                    // policy forwarding cannot alter any coefficient on the sequential path
                    EXPECT_EQ(serial.nodal_gradient[i](row, col), sequential.nodal_gradient[i](row, col));
                }
    }
    std::vector<int> selection(nodes.size());
    for (std::size_t i = 0; i < nodes.size(); ++i) selection[i] = int(nodes.size() - 1 - i);
    const auto selected = nodes.select(selection);
    const auto serial = gfe::p1_discrete_tension_contribution(geometry, selected, stencil);
    const auto parallel = gfe::p1_discrete_tension_contribution(geometry, selected, stencil, execution_par);
    // a borrowed reordered batch selection uses the same owner caches under both execution policies
    EXPECT_EQ(serial.value, parallel.value);
    for (std::size_t i = 0; i < nodes.size(); ++i)
        for (int row = 0; row < geometry.order(); ++row)
            for (int col = 0; col <= row; ++col) {
                // selected-node pullbacks retain the exact serial node ordering and arithmetic
                EXPECT_EQ(serial.nodal_gradient[i](row, col), parallel.nodal_gradient[i](row, col));
            }
}

/// @brief compares planar and general C-LE pair pullbacks with their original serial covectors
template <typename Geometry> void check_cheeger_tension_execution(const Geometry& geometry) {
    auto [nodes, stencil] = tension_execution_data(geometry);
    std::vector<double> rho(nodes.size());
    for (std::size_t i = 0; i < nodes.size(); ++i) rho[i] = .5 + .03 * i;
    for (int repeat = 0; repeat < 2; ++repeat) {
        const auto parallel =
          gfe::p1_cheeger_discrete_tension_log_contribution(geometry, nodes, stencil, rho, execution_par);
        const auto serial = gfe::p1_cheeger_discrete_tension_log_contribution(geometry, nodes, stencil, rho);
        // independent metric evaluations reduce in the unchanged serial nodal order
        EXPECT_EQ(serial.value, parallel.value);
        for (std::size_t i = 0; i < nodes.size(); ++i) {
            // each directed branch's rho contribution reaches its base in the original edge order
            EXPECT_EQ(serial.rho_gradient[i], parallel.rho_gradient[i]);
            for (int row = 0; row < geometry.order(); ++row)
                for (int col = 0; col <= row; ++col) {
                    // ungrouped coordinate summands prevent parallel scheduling from changing roundoff
                    EXPECT_EQ(serial.nodal_gradient[i](row, col), parallel.nodal_gradient[i](row, col));
                }
        }
    }
    const auto serial = gfe::p1_cheeger_discrete_tension_log_contribution(geometry, nodes, stencil);
    const auto parallel = gfe::p1_cheeger_discrete_tension_log_contribution(geometry, nodes, stencil, execution_par);
    // the constant-rho policy overload forwards the empty nodal span unchanged
    EXPECT_EQ(serial.value, parallel.value);
}

/// @brief checks exact serial-parallel parity for flat and intrinsic static and dynamic matrix kernels
TEST(GeometricTensionExecution, FlatAndIntrinsicParity) {
    check_tension_execution(manifold::LogEuclideanSPDGeometry<double, 2, Usage::InterpolationNodes> {});
    check_tension_execution(manifold::LogCholeskySPDGeometry<double, 2, Usage::InterpolationNodes> {});
    check_tension_execution(manifold::AffineInvariantSPDGeometry<double, 2, Usage::BasePointMaps> {});
    check_tension_execution(manifold::BuresWassersteinSPDGeometry<double, 2, Usage::BasePointMaps> {});
    check_tension_execution(manifold::LogEuclideanSPDGeometry<double, 3> {});
    check_tension_execution(manifold::LogCholeskySPDGeometry<double, Dynamic>(3));
    check_tension_execution(manifold::AffineInvariantSPDGeometry<double, Dynamic>(3));
    check_tension_execution(manifold::BuresWassersteinSPDGeometry<double, 3> {});
}

/// @brief checks exact covector and rho-gradient parity for all C-LE tension specializations
TEST(GeometricTensionExecution, CheegerParity) {
    check_cheeger_tension_execution(manifold::CheegerLogEuclideanSPDGeometry<double, 2, Usage::InterpolationNodes> {});
    check_cheeger_tension_execution(manifold::CheegerLogEuclideanSPDGeometry<double, 3, Usage::InterpolationNodes> {});
    check_cheeger_tension_execution(manifold::CheegerLogEuclideanSPDGeometry<double, Dynamic>(3));
}

/// @brief verifies worker exceptions join safely and leave the execution runtime reusable
TEST(GeometricTensionExecution, WorkerFailureAndRecovery) {
    using Geometry = manifold::LogEuclideanSPDGeometry<double, 2>;
    Geometry geometry;
    auto [nodes, stencil] = tension_execution_data(geometry);
    nodes[0] = Matrix<double, 2, 2>({std::exp(4.), 0., 0., std::exp(4.)});
    stencil.edges[0].stiffness = -1e308;
    // finite inputs overflow a residual in a worker and the domain failure is rethrown to the caller
    EXPECT_THROW(gfe::p1_discrete_tension_contribution(geometry, nodes, stencil, execution_par), std::domain_error);
    stencil.edges[0].stiffness = -.2;
    // all tasks are joined before rethrowing, so another tension evaluation can use the same executor
    EXPECT_NO_THROW(gfe::p1_discrete_tension_contribution(geometry, nodes, stencil, execution_par));
    auto invalid = stencil;
    invalid.lumped_masses[1] = 0;
    // public mass validation rejects zero inverse masses before any workers are submitted
    EXPECT_THROW(gfe::p1_discrete_tension_value(geometry, nodes, invalid, execution_par), std::invalid_argument);
}
}   // namespace
