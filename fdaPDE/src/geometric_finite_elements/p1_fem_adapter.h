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

#ifndef __FDAPDE_GFE_P1_FEM_ADAPTER_H__
#define __FDAPDE_GFE_P1_FEM_ADAPTER_H__

#ifndef __FDAPDE_GEOMETRIC_FINITE_ELEMENTS_FEM_MODULE_H__
#    error "Include fdaPDE/geometric_finite_elements_fem.h instead of including this adapter directly."
#endif

namespace fdapde {
namespace gfe {

namespace internals {

template <typename Space, typename Quadrature>
concept p1_fem_adapter_source =
  Space::local_dim >= 1 && Space::local_dim <= 3 && Space::embed_dim >= Space::local_dim && Space::FeType::order == 1 &&
  Space::n_components == 1 && Space::cell_dof_descriptor::dof_sharing && Quadrature::local_dim == Space::local_dim &&
  Quadrature::order > 0;

inline bool approximately_zero(double sum, double absolute_sum, std::size_t count) {
    const double tolerance = p1_weight_sum_tolerance(count) * std::max(1.0, absolute_sum);
    return std::isfinite(sum) && std::abs(sum) <= tolerance;
}

}   // namespace internals

template <typename Space, typename Quadrature = typename Space::FeType::template cell_quadrature_t<Space::local_dim>>
    requires(internals::p1_fem_adapter_source<Space, Quadrature>)
auto p1_fem_cell_quadrature(const Space& space, std::size_t cell_id, const Quadrature& quadrature = Quadrature {}) {
    constexpr std::size_t local_dim = static_cast<std::size_t>(Space::local_dim);
    constexpr std::size_t embed_dim = static_cast<std::size_t>(Space::embed_dim);
    constexpr std::size_t quadrature_size = static_cast<std::size_t>(Quadrature::order);
    using Result = P1FEMCellQuadrature<local_dim, embed_dim, quadrature_size>;

    if (!space.dof_handler()) { throw std::logic_error("P1 FEM adapter requires an initialized finite-element space"); }
    if (cell_id >= static_cast<std::size_t>(space.triangulation().n_cells())) {
        throw std::out_of_range("P1 FEM adapter cell index is out of range");
    }

    const auto cell = space.dof_handler().cell(static_cast<int>(cell_id));
    const double measure = cell.measure();
    if (!std::isfinite(measure) || measure <= 0) {
        throw std::domain_error("P1 FEM adapter requires a cell with positive finite measure");
    }

    Result result {};
    std::vector<int> active_dofs;
    active_dofs.reserve(Result::node_count);
    space.dof_handler().active_dofs(static_cast<int>(cell_id), active_dofs);
    if (active_dofs.size() != Result::node_count) {
        throw std::logic_error("P1 FEM adapter requires one local degree of freedom per cell vertex");
    }
    for (std::size_t node = 0; node < Result::node_count; ++node) {
        if (active_dofs[node] < 0 || active_dofs[node] >= space.n_dofs()) {
            throw std::domain_error("P1 FEM adapter encountered an invalid degree of freedom");
        }
        result.dofs[node] = static_cast<std::size_t>(active_dofs[node]);
    }

    const auto& inverse_jacobian = cell.invJ();
    for (std::size_t axis = 0; axis < embed_dim; ++axis) {
        double first_gradient = 0;
        double absolute_sum = 0;
        for (std::size_t reference_axis = 0; reference_axis < local_dim; ++reference_axis) {
            const double coefficient = inverse_jacobian(static_cast<int>(reference_axis), static_cast<int>(axis));
            if (!std::isfinite(coefficient)) {
                throw std::domain_error("P1 FEM adapter requires finite physical shape gradients");
            }
            result.physical_weight_gradients[axis][reference_axis + 1] = coefficient;
            first_gradient -= coefficient;
            absolute_sum += std::abs(coefficient);
        }
        result.physical_weight_gradients[axis][0] = first_gradient;
        absolute_sum += std::abs(first_gradient);

        double sum = 0;
        for (const double coefficient : result.physical_weight_gradients[axis]) { sum += coefficient; }
        if (!internals::approximately_zero(sum, absolute_sum, Result::node_count)) {
            throw std::domain_error("P1 FEM adapter physical shape gradients must sum to zero");
        }
    }

    double quadrature_weight_sum = 0;
    double quadrature_weight_absolute_sum = 0;
    for (std::size_t q = 0; q < quadrature_size; ++q) {
        double coordinate_sum = 0;
        for (std::size_t reference_axis = 0; reference_axis < local_dim; ++reference_axis) {
            const double coordinate = quadrature.nodes(static_cast<int>(q), static_cast<int>(reference_axis));
            if (!std::isfinite(coordinate) || coordinate < 0) {
                throw std::invalid_argument("P1 FEM adapter quadrature nodes must lie in the reference simplex");
            }
            result.barycentric_weights[q][reference_axis + 1] = coordinate;
            coordinate_sum += coordinate;
        }
        const double first_weight = 1 - coordinate_sum;
        if (!std::isfinite(first_weight) || first_weight < 0) {
            throw std::invalid_argument("P1 FEM adapter quadrature nodes must lie in the reference simplex");
        }
        result.barycentric_weights[q][0] = first_weight;
        internals::validate_p1_data(Result::node_count, result.barycentric_weights[q]);

        const double quadrature_weight = quadrature.weights[static_cast<int>(q)];
        if (!std::isfinite(quadrature_weight) || quadrature_weight < 0) {
            throw std::invalid_argument("P1 FEM adapter requires finite non-negative quadrature weights");
        }
        quadrature_weight_sum += quadrature_weight;
        quadrature_weight_absolute_sum += std::abs(quadrature_weight);
        result.integration_weights[q] = measure * quadrature_weight;
        if (!std::isfinite(result.integration_weights[q]) || result.integration_weights[q] < 0) {
            throw std::domain_error("P1 FEM adapter integration weights must be finite and non-negative");
        }
    }
    if (!internals::approximately_zero(quadrature_weight_sum - 1, quadrature_weight_absolute_sum, quadrature_size)) {
        throw std::invalid_argument("P1 FEM adapter quadrature weights must sum to one");
    }
    return result;
}

}   // namespace gfe
}   // namespace fdapde

#endif   // __FDAPDE_GFE_P1_FEM_ADAPTER_H__
