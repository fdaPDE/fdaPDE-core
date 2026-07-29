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

#ifndef __FDAPDE_GFE_P1_LUMPED_LAPLACIAN_STENCIL_H__
#define __FDAPDE_GFE_P1_LUMPED_LAPLACIAN_STENCIL_H__

#include "header_check.h"

namespace fdapde {
namespace gfe {

struct P1LumpedLaplacianEdge {
    std::size_t first = 0;
    std::size_t second = 0;
    double stiffness = 0;
};

struct P1LumpedLaplacianStencil {
    std::vector<double> lumped_masses;
    std::vector<P1LumpedLaplacianEdge> edges;

    std::size_t node_count() const noexcept { return lumped_masses.size(); }
};

namespace internals {

struct p1_lumped_mass_contribution {
    std::size_t node;
    double value;
};

struct p1_stiffness_contribution {
    std::size_t first;
    std::size_t second;
    double value;
};

inline void p1_laplacian_validate_spatial_direction(std::span<const double> direction) {
    long double sum = 0;
    long double absolute_sum = 0;
    for (const double coefficient : direction) {
        if (!std::isfinite(coefficient)) {
            throw std::invalid_argument("P1 Laplacian physical weight gradients must be finite");
        }
        sum += static_cast<long double>(coefficient);
        absolute_sum += std::abs(static_cast<long double>(coefficient));
    }
    const long double tolerance =
      static_cast<long double>(p1_weight_sum_tolerance(direction.size())) * std::max(1.0L, absolute_sum);
    if (std::abs(sum) > tolerance) {
        throw std::invalid_argument("P1 Laplacian physical weight gradients must sum to zero");
    }
}

inline double p1_laplacian_finite_double(long double value, const char* message) {
    const double result = static_cast<double>(value);
    if (!std::isfinite(value) || !std::isfinite(result)) { throw std::domain_error(message); }
    return result;
}

inline void p1_laplacian_compensated_add(long double value, long double& sum, long double& correction) {
    const long double corrected_value = value - correction;
    const long double updated_sum = sum + corrected_value;
    correction = (updated_sum - sum) - corrected_value;
    sum = updated_sum;
}

inline void validate_p1_lumped_laplacian_stencil(const P1LumpedLaplacianStencil& stencil) {
    if (stencil.lumped_masses.empty()) {
        throw std::invalid_argument("P1 lumped Laplacian stencil requires at least one node");
    }
    for (const double mass : stencil.lumped_masses) {
        if (!std::isfinite(mass) || mass <= 0) {
            throw std::invalid_argument("P1 lumped Laplacian masses must be finite and positive");
        }
    }
    std::optional<std::pair<std::size_t, std::size_t>> previous;
    for (const auto& edge : stencil.edges) {
        if (edge.first >= edge.second || edge.second >= stencil.node_count()) {
            throw std::invalid_argument("P1 lumped Laplacian edges require ordered in-range distinct nodes");
        }
        if (!std::isfinite(edge.stiffness) || edge.stiffness == 0) {
            throw std::invalid_argument("P1 lumped Laplacian edge stiffnesses must be finite and nonzero");
        }
        const std::pair key {edge.first, edge.second};
        if (previous && *previous >= key) {
            throw std::invalid_argument("P1 lumped Laplacian edges must be unique and lexicographically sorted");
        }
        previous = key;
    }
}

}   // namespace internals

template <std::size_t LocalDim, std::size_t EmbedDim, std::size_t QuadratureSize>
P1LumpedLaplacianStencil p1_lumped_laplacian_stencil(
  std::size_t node_count, std::span<const P1FEMCellQuadrature<LocalDim, EmbedDim, QuadratureSize>> packets) {
    using Packet = P1FEMCellQuadrature<LocalDim, EmbedDim, QuadratureSize>;
    static_assert(LocalDim >= 1 && LocalDim <= 3);
    static_assert(EmbedDim >= LocalDim);
    static_assert(QuadratureSize > 0);

    if (node_count == 0) { throw std::invalid_argument("P1 lumped Laplacian stencil requires at least one node"); }
    if (packets.empty()) { throw std::invalid_argument("P1 lumped Laplacian stencil requires at least one cell"); }

    std::vector<internals::p1_lumped_mass_contribution> mass_contributions;
    std::vector<internals::p1_stiffness_contribution> stiffness_contributions;
    for (const Packet& packet : packets) {
        for (std::size_t local_node = 0; local_node < Packet::node_count; ++local_node) {
            const std::size_t dof = packet.dofs[local_node];
            if (dof >= node_count) {
                throw std::out_of_range("P1 lumped Laplacian cell degree of freedom is out of range");
            }
            for (std::size_t previous = 0; previous < local_node; ++previous) {
                if (packet.dofs[previous] == dof) {
                    throw std::invalid_argument("P1 lumped Laplacian cell degrees of freedom must be distinct");
                }
            }
        }
        for (const auto& direction : packet.physical_weight_gradients) {
            internals::p1_laplacian_validate_spatial_direction(std::span<const double>(direction));
        }

        long double cell_measure = 0;
        long double cell_measure_correction = 0;
        for (std::size_t site = 0; site < Packet::quadrature_size; ++site) {
            const double integration_weight = packet.integration_weights[site];
            if (!std::isfinite(integration_weight) || integration_weight < 0) {
                throw std::invalid_argument("P1 lumped Laplacian integration weights must be finite and non-negative");
            }
            internals::p1_laplacian_compensated_add(
              static_cast<long double>(integration_weight), cell_measure, cell_measure_correction);
        }
        if (!std::isfinite(cell_measure) || cell_measure <= 0) {
            throw std::domain_error("P1 lumped Laplacian cells must have positive finite measure");
        }
        const double cell_mass = internals::p1_laplacian_finite_double(
          cell_measure / static_cast<long double>(Packet::node_count),
          "P1 lumped Laplacian mass contribution is nonfinite");
        for (const std::size_t dof : packet.dofs) { mass_contributions.push_back({dof, cell_mass}); }

        for (std::size_t first = 0; first < Packet::node_count; ++first) {
            for (std::size_t second = first + 1; second < Packet::node_count; ++second) {
                long double gradient_inner_product = 0;
                for (std::size_t axis = 0; axis < Packet::embed_dim; ++axis) {
                    gradient_inner_product += static_cast<long double>(packet.physical_weight_gradients[axis][first]) *
                                              static_cast<long double>(packet.physical_weight_gradients[axis][second]);
                }
                const double stiffness = internals::p1_laplacian_finite_double(
                  cell_measure * gradient_inner_product, "P1 lumped Laplacian stiffness contribution is nonfinite");
                if (stiffness == 0) continue;
                const auto [global_first, global_second] = std::minmax(packet.dofs[first], packet.dofs[second]);
                stiffness_contributions.push_back({global_first, global_second, stiffness});
            }
        }
    }

    std::sort(mass_contributions.begin(), mass_contributions.end(), [](const auto& left, const auto& right) {
        if (left.node != right.node) return left.node < right.node;
        return left.value < right.value;
    });
    P1LumpedLaplacianStencil result;
    result.lumped_masses.assign(node_count, 0);
    for (std::size_t begin = 0; begin < mass_contributions.size();) {
        const std::size_t node = mass_contributions[begin].node;
        long double mass = 0;
        long double correction = 0;
        std::size_t end = begin;
        while (end < mass_contributions.size() && mass_contributions[end].node == node) {
            internals::p1_laplacian_compensated_add(
              static_cast<long double>(mass_contributions[end].value), mass, correction);
            ++end;
        }
        result.lumped_masses[node] =
          internals::p1_laplacian_finite_double(mass, "P1 lumped Laplacian nodal mass is nonfinite");
        begin = end;
    }
    for (const double mass : result.lumped_masses) {
        if (mass <= 0) { throw std::domain_error("P1 lumped Laplacian contains a node with zero mass"); }
    }

    std::sort(stiffness_contributions.begin(), stiffness_contributions.end(), [](const auto& left, const auto& right) {
        if (left.first != right.first) return left.first < right.first;
        if (left.second != right.second) return left.second < right.second;
        return left.value < right.value;
    });
    for (std::size_t begin = 0; begin < stiffness_contributions.size();) {
        const std::size_t first = stiffness_contributions[begin].first;
        const std::size_t second = stiffness_contributions[begin].second;
        long double stiffness = 0;
        long double correction = 0;
        std::size_t end = begin;
        while (end < stiffness_contributions.size() && stiffness_contributions[end].first == first &&
               stiffness_contributions[end].second == second) {
            internals::p1_laplacian_compensated_add(
              static_cast<long double>(stiffness_contributions[end].value), stiffness, correction);
            ++end;
        }
        const double value =
          internals::p1_laplacian_finite_double(stiffness, "P1 lumped Laplacian assembled stiffness is nonfinite");
        if (value != 0) { result.edges.push_back({first, second, value}); }
        begin = end;
    }
    internals::validate_p1_lumped_laplacian_stencil(result);
    return result;
}

}   // namespace gfe
}   // namespace fdapde

#endif   // __FDAPDE_GFE_P1_LUMPED_LAPLACIAN_STENCIL_H__
