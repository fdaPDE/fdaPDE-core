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

#ifndef __FDAPDE_GEOMETRIC_SPLINE_INTERPOLANT_H__
#define __FDAPDE_GEOMETRIC_SPLINE_INTERPOLANT_H__
#include "header_check.h"

namespace fdapde::gfe::internals {
/// @brief borrows active control matrices and applies the shared weighted mean and differential kernels
template <typename Geometry, typename Nodes> class spline_local_interpolant {
   public:
    using Scalar = typename Geometry::Scalar;
    using Point = typename Geometry::Point;
    using Weights = std::vector<double>;
    static constexpr int Rows = Point::Rows;
    /// @brief retains a selection binding, solver options and optional local scalar metric coefficients
    spline_local_interpolant(
      Geometry geometry, Nodes nodes, P1GeodesicLinearizationOptions options, std::vector<double> rho) :
        geometry_(std::move(geometry)), nodes_(std::move(nodes)), options_(options), rho_(std::move(rho)) { }
    /// @brief returns the matrix order shared by the active control coefficients
    int order() const { return geometry_.order(); }
    /// @brief reports the weighted mean and its convergence diagnostics without hiding iterative failures
    auto result(const Weights& weights) const {
        const auto geometry = metric_(weights);
        if constexpr (is_cheeger_geometry<Geometry>)
            return p1_geodesic_value(geometry, nodes_, weights, options_.mean);
        else if constexpr (is_flat_spd_geometry<Geometry>)
            return p1_geodesic_value(geometry, nodes_, weights);
        else
            return p1_geodesic_value(geometry, nodes_, weights, options_.mean);
    }
    /// @brief materializes a geometric value only after certifying its weighted mean
    Point evaluate(const Weights& weights) const {
        auto value = result(weights);
        require_converged(value);
        return std::move(value.value);
    }
    /// @brief prepares derivatives in active control order while retaining the selection binding
    auto linearization(const Weights& weights) const {
        if constexpr (is_cheeger_geometry<Geometry>)
            return p1_geodesic_linearization(geometry_, nodes_, weights, options_, std::span<const double>(rho_));
        else if constexpr (is_flat_spd_geometry<Geometry>)
            return p1_geodesic_linearization(geometry_, nodes_, weights);
        else
            return p1_geodesic_linearization(geometry_, nodes_, weights, options_);
    }
   private:
    /// @brief evaluates a varying Cheeger metric with the same basis weights as the matrix field
    Geometry metric_(const Weights& weights) const {
        if constexpr (is_cheeger_geometry<Geometry>) {
            if (!rho_.empty()) {
                double rho = 0;
                for (std::size_t i = 0; i < rho_.size(); ++i) rho += weights[i] * rho_[i];
                return geometry_.with_rho(rho);
            }
        }
        return geometry_;
    }
    Geometry geometry_;
    Nodes nodes_;
    P1GeodesicLinearizationOptions options_;
    std::vector<double> rho_;
};

/// @brief retains local spline coefficient bindings while sharing the existing weighted geometric kernels
template <typename Geometry, typename Domain, typename Values> class spline_field_interpolant {
   public:
    using Adapter = typename Domain::Adapter;
    using Mesh = typename Domain::Triangulation;
    using Element = typename Adapter::Element;
    using NodeType = typename Element::NodeType;
    using Dofs = typename Adapter::Dofs;
    using Weights = typename Adapter::Weights;
    using Selection = decltype(std::declval<const std::remove_cvref_t<Values>&>().select(std::declval<Dofs>()));
    using Local = spline_local_interpolant<Geometry, Selection>;
    /// @brief borrows the scalar space and controls and prepares an independent interval locator
    spline_field_interpolant(
      Geometry geometry, const Domain& domain, Values values, P1GeodesicLinearizationOptions options) :
        geometry_(std::move(geometry)),
        domain_(&domain),
        values_(values),
        options_(options),
        locator_(&domain.triangulation()) { }
    /// @brief borrows stable local data and owns the weights of a deferred spatial evaluation
    auto operator()(const NodeType& x) const& {
        const auto id = locate_(x);
        const auto dofs = Adapter::dofs(domain_->scalar_space(), id);
        const auto& local = cell_(id, dofs);
        return p1_interpolant_expr<const Local&>(local, weights_(id, dofs, x));
    }
    /// @brief prevents expressions from escaping an expiring local cache
    void operator()(const NodeType&) const&& = delete;
    /// @brief returns weighted mean diagnostics at a physical interval coordinate
    auto result(const NodeType& x) const {
        const auto id = locate_(x);
        const auto dofs = Adapter::dofs(domain_->scalar_space(), id);
        return cell_(id, dofs).result(weights_(id, dofs, x));
    }
    /// @brief prepares nodal and weight derivatives in the active spline control ordering
    auto linearization(const NodeType& x) const& {
        const auto id = locate_(x);
        const auto dofs = Adapter::dofs(domain_->scalar_space(), id);
        return cell_(id, dofs).linearization(weights_(id, dofs, x));
    }
    /// @brief prevents derivative bindings from escaping an expiring interpolant
    void linearization(const NodeType&) const&& = delete;
    /// @brief counts visited knot spans retaining local coefficient bindings
    std::size_t prepared_cells() const {
        const std::lock_guard lock(mutex_);
        return cells_.size();
    }
    /// @brief invalidates coefficient-dependent local data before replacing owned controls
    void clear_cache() {
        const std::lock_guard lock(mutex_);
        cells_.clear();
    }
    /// @brief validates replacement Cheeger scalar controls before invalidating local metric data
    void set_rho(std::vector<double> rho)
        requires is_cheeger_geometry<Geometry>
    {
        fdapde_strong_assert(
          rho.empty() || rho.size() == values_.size(), std::invalid_argument, "rho count must match spline controls");
        for (double value : rho) geometry_.with_rho(value);
        clear_cache();
        rho_override_.emplace(std::move(rho));
    }
    /// @brief borrows the field override or the immutable space defaults for the Cheeger metric
    std::span<const double> rho_coefficients() const&
        requires is_cheeger_geometry<Geometry>
    {
        if (rho_override_) return *rho_override_;
        return domain_->rho_coefficients();
    }
    /// @brief prevents scalar spans from escaping temporary interpolation owners
    void rho_coefficients() const&& = delete;
    /// @brief uses retained spatial weights with the current control and metric bindings
    typename Geometry::Point
    evaluate_prepared_(int id, const Element&, const Dofs& dofs, const Weights& weights) const {
        return cell_(id, dofs).evaluate(weights);
    }
   private:
    /// @brief rejects nonfinite coordinates and coordinates outside the physical spline interval
    int locate_(const NodeType& x) const {
        fdapde_strong_assert(x.allFinite(), std::invalid_argument, "spline field spatial point must be finite");
        const int id = Adapter::locate(domain_->scalar_space(), locator_, x);
        fdapde_strong_assert(id >= 0, std::invalid_argument, "spline field spatial point must belong to the interval");
        return id;
    }
    /// @brief evaluates the physical scalar basis on an already located knot span
    Weights weights_(int id, const Dofs& dofs, const NodeType& x) const {
        const Element element(domain_->dof_handler().cell(id).nodes());
        return Adapter::weights(domain_->scalar_space(), element, dofs, x);
    }
    /// @brief synchronizes first span preparation while leaving mean evaluation outside the lock
    const Local& cell_(int id, const Dofs& dofs) const {
        const std::lock_guard lock(mutex_);
        if (const auto it = cells_.find(id); it != cells_.end()) return it->second;
        std::vector<double> rho;
        if constexpr (is_cheeger_geometry<Geometry>) {
            const auto controls = rho_coefficients();
            if (!controls.empty())
                for (int index : dofs) rho.push_back(controls[index]);
        }
        return cells_.try_emplace(id, geometry_, values_.select(dofs), options_, std::move(rho)).first->second;
    }
    Geometry geometry_;
    const Domain* domain_;
    Values values_;
    P1GeodesicLinearizationOptions options_;
    TreeSearch<Mesh> locator_;
    std::optional<std::vector<double>> rho_override_;
    mutable std::mutex mutex_;
    mutable std::unordered_map<int, Local> cells_;
};
}   // namespace fdapde::gfe::internals
#endif
