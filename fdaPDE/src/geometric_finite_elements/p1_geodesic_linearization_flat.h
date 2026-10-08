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

#ifndef __FDAPDE_GFE_P1_GEODESIC_LINEARIZATION_FLAT_H__
#define __FDAPDE_GFE_P1_GEODESIC_LINEARIZATION_FLAT_H__

#include "header_check.h"

namespace fdapde {
namespace gfe {

/// @brief retains a mean and its differential data, borrowing immutable batches or owning span snapshots
template <typename Geometry_, typename Nodes>
    requires internals::is_flat_spd_geometry<Geometry_>
class P1GeodesicLinearization<Geometry_, Nodes> {
   public:
    using Geometry = Geometry_;
    using Scalar_ = typename Geometry::Scalar;
    static constexpr int Order_ = internals::flat_spd_order<Geometry>;
    using Point = typename Geometry::Point;
    using Tangent = typename Geometry::Tangent;

    /// @brief borrows the mean candidate and convergence diagnostics
    const P1ValueResult<Point>& result() const& noexcept { return result_; }
    /// @brief prevents borrowing storage from a temporary object
    const P1ValueResult<Point>& result() const&& = delete;

    /// @brief differentiates the represented normalized mean along a finite zero-sum weight direction
    /// @details the direction is based at result().normalized_weights. It must be
    /// @details finite and sum to zero; it is never projected or renormalized. The
    /// @details centered quotient formula differentiates the represented normalized
    /// @details mean using the same compensated weight total as the value
    Tangent weight_jvp(std::span<const double> direction) const {
        require_ready_();
        validate_weight_direction_(direction);

        bool has_nonzero_direction = false;
        auto chart_direction = make_accumulation_tangent_();
        auto correction = make_accumulation_tangent_();
        for (std::size_t node_index = 0; node_index < nodes_.size(); ++node_index) {
            const double coefficient = direction[node_index];
            if (coefficient == 0) continue;
            has_nonzero_direction = true;

            for (int i = 0; i < geometry_.order(); ++i) {
                for (int j = 0; j <= i; ++j) {
                    const AccumulationScalar contribution =
                      (static_cast<AccumulationScalar>(coefficient) /
                       static_cast<AccumulationScalar>(normalized_total_)) *
                      (static_cast<AccumulationScalar>(node_chart_(node_index)(i, j)) -
                       static_cast<AccumulationScalar>((*mean_chart_)(i, j)));
                    compensated_add_(chart_direction, correction, i, j, contribution);
                }
            }
        }
        if (!has_nonzero_direction) return make_zero_tangent_();
        return internals::p1_flat_frame_inverse_chart_jvp(
          geometry_, mean_frame_reference_(), to_tangent_(chart_direction));
    }

    /// @brief differentiates active nodal values in the geometry tangent representation
    /// @details directions at nodes with zero effective weight are not inspected,
    /// @details because their contribution is identically zero
    Tangent nodal_jvp(std::span<const Tangent> directions) const {
        require_ready_();
        fdapde_strong_assert(
          directions.size() == nodes_.size(), std::invalid_argument,
          "P1 geodesic nodal-direction and nodal-value counts must match");

        if (vertex_index_) {
            static_cast<void>(internals::p1_flat_frame_chart_jvp(
              geometry_, nodes_[*vertex_index_], *node_frames_[*vertex_index_], directions[*vertex_index_]));
            return directions[*vertex_index_];
        }

        auto chart_direction = make_accumulation_tangent_();
        auto correction = make_accumulation_tangent_();
        for (std::size_t node_index = 0; node_index < nodes_.size(); ++node_index) {
            const double weight = result_.normalized_weights[node_index];
            if (weight == 0) continue;

            const auto node_direction = internals::p1_flat_frame_chart_jvp(
              geometry_, nodes_[node_index], *node_frames_[node_index], directions[node_index]);
            const AccumulationScalar coefficient =
              static_cast<AccumulationScalar>(weight) / static_cast<AccumulationScalar>(normalized_total_);
            for (int i = 0; i < geometry_.order(); ++i) {
                for (int j = 0; j <= i; ++j) {
                    const AccumulationScalar contribution =
                      coefficient * static_cast<AccumulationScalar>(node_direction(i, j));
                    compensated_add_(chart_direction, correction, i, j, contribution);
                }
            }
        }
        return internals::p1_flat_frame_inverse_chart_jvp(
          geometry_, mean_frame_reference_(), to_tangent_(chart_direction));
    }

    /// @brief returns the Riemannian metric adjoint of the nodal differential
    /// @details this is the Riemannian adjoint of nodal_jvp. The argument and
    /// @details returned covectors are represented by metric-dual tangent vectors
    /// @details for the flat chart metric, not by ambient Frobenius gradients
    std::vector<Tangent> nodal_vjp(const Tangent& value_gradient) const {
        require_ready_();
        const Tangent value_chart_gradient =
          internals::p1_flat_frame_chart_jvp(geometry_, result_.value, mean_frame_reference_(), value_gradient);
        std::vector<Tangent> result(nodes_.size(), make_zero_tangent_());
        if (vertex_index_) {
            result[*vertex_index_] = value_gradient;
            return result;
        }

        for (std::size_t node_index = 0; node_index < nodes_.size(); ++node_index) {
            const double weight = result_.normalized_weights[node_index];
            if (weight == 0) continue;
            const Tangent scaled = scaled_tangent_(
              value_chart_gradient,
              static_cast<AccumulationScalar>(weight) / static_cast<AccumulationScalar>(normalized_total_));
            result[node_index] =
              internals::p1_flat_frame_inverse_chart_jvp(geometry_, *node_frames_[node_index], scaled);
        }
        return result;
    }

    /// @brief differentiates the weight action covariantly along nodal variations
    /// @details covariant nodal derivative of the spatial/weight JVP. The weight
    /// @details direction is based at result().normalized_weights, must be finite and
    /// @details sum to zero, and follows the same represented normalized-total
    /// @details convention as weight_jvp
    Tangent covariant_mixed_nodal_jvp(
      std::span<const double> weight_direction, std::span<const Tangent> nodal_directions) const {
        require_ready_();
        validate_weight_direction_(weight_direction);
        fdapde_strong_assert(
          nodal_directions.size() == nodes_.size(), std::invalid_argument,
          "P1 geodesic nodal-direction and nodal-value counts must match");

        const AccumulationScalar direction_total =
          static_cast<AccumulationScalar>(compensated_weight_total_(weight_direction));
        auto chart_direction = make_accumulation_tangent_();
        auto correction = make_accumulation_tangent_();
        bool has_nonzero_coefficient = false;
        for (std::size_t node_index = 0; node_index < nodes_.size(); ++node_index) {
            const AccumulationScalar coefficient =
              mixed_nodal_coefficient_(weight_direction[node_index], direction_total, node_index);
            if (coefficient == 0) continue;
            has_nonzero_coefficient = true;

            const auto node_direction = internals::p1_flat_frame_chart_jvp(
              geometry_, nodes_[node_index], *node_frames_[node_index], nodal_directions[node_index]);
            for (int i = 0; i < geometry_.order(); ++i) {
                for (int j = 0; j <= i; ++j) {
                    const AccumulationScalar contribution =
                      coefficient * static_cast<AccumulationScalar>(node_direction(i, j));
                    compensated_add_(chart_direction, correction, i, j, contribution);
                }
            }
        }
        if (!has_nonzero_coefficient) return make_zero_tangent_();
        return internals::p1_flat_frame_inverse_chart_jvp(
          geometry_, mean_frame_reference_(), to_tangent_(chart_direction));
    }

    /// @details riemannian adjoint, in the nodal variable, of
    /// @details covariant_mixed_nodal_jvp for a fixed weight direction. The argument
    /// @details and returned covectors are flat-chart metric-dual tangents
    /// @brief returns the nodal metric adjoint of the covariant mixed differential
    std::vector<Tangent>
    covariant_mixed_nodal_vjp(std::span<const double> weight_direction, const Tangent& value_gradient) const {
        require_ready_();
        validate_weight_direction_(weight_direction);
        const Tangent value_chart_gradient =
          internals::p1_flat_frame_chart_jvp(geometry_, result_.value, mean_frame_reference_(), value_gradient);
        std::vector<Tangent> result(nodes_.size(), make_zero_tangent_());
        const AccumulationScalar direction_total =
          static_cast<AccumulationScalar>(compensated_weight_total_(weight_direction));
        for (std::size_t node_index = 0; node_index < nodes_.size(); ++node_index) {
            const AccumulationScalar coefficient =
              mixed_nodal_coefficient_(weight_direction[node_index], direction_total, node_index);
            if (coefficient == 0) continue;
            const Tangent scaled = scaled_tangent_(value_chart_gradient, coefficient);
            result[node_index] =
              internals::p1_flat_frame_inverse_chart_jvp(geometry_, *node_frames_[node_index], scaled);
        }
        return result;
    }
   private:
    using AccumulationScalar = std::common_type_t<Scalar_, double>;
    using AccumulationTangent = fdapde::SymmetricMatrix<AccumulationScalar, Order_>;
    using ChartFrame = typename internals::FlatSPDChartFrame<Geometry>::type;
   public:
    /// @brief retains the nodal binding and prepares derivatives at the requested barycentric point
    P1GeodesicLinearization(const Geometry& geometry, Nodes nodal_values, std::span<const double> barycentric_weights) :
        geometry_(geometry),
        nodes_(std::forward<Nodes>(nodal_values)),
        result_(p1_geodesic_value(geometry_, nodes_, barycentric_weights)),
        node_frames_(nodes_.size()) {
        std::size_t positive_count = 0;
        for (std::size_t node_index = 0; node_index < nodes_.size(); ++node_index) {
            check_node_shape_(nodes_[node_index]);
            node_frames_[node_index].emplace(internals::p1_flat_chart_frame(geometry_, nodes_[node_index]));
            if (result_.normalized_weights[node_index] > 0) {
                ++positive_count;
                vertex_index_ = node_index;
            }
        }
        if (positive_count != 1) vertex_index_.reset();
        if (!result_.converged()) return;

        normalized_total_ = compensated_weight_total_(result_.normalized_weights);
        auto mean_chart = make_accumulation_tangent_();
        auto correction = make_accumulation_tangent_();
        for (std::size_t node_index = 0; node_index < nodes_.size(); ++node_index) {
            const double weight = result_.normalized_weights[node_index];
            if (weight == 0) continue;
            for (int i = 0; i < geometry_.order(); ++i) {
                for (int j = 0; j <= i; ++j) {
                    const AccumulationScalar contribution =
                      static_cast<AccumulationScalar>(weight) *
                      static_cast<AccumulationScalar>(node_chart_(node_index)(i, j));
                    compensated_add_(mean_chart, correction, i, j, contribution);
                }
            }
        }
        for (int i = 0; i < geometry_.order(); ++i) {
            for (int j = 0; j <= i; ++j) {
                mean_chart(i, j) = static_cast<AccumulationScalar>(mean_chart(i, j)) /
                                   static_cast<AccumulationScalar>(normalized_total_);
            }
        }
        mean_chart_.emplace(to_tangent_(mean_chart));
        if constexpr (!internals::is_log_euclidean_spd_geometry<Geometry>)
            mean_frame_.emplace(geometry_.chart_frame(result_.value));
        ready_ = true;
    }
   private:
    /// @brief reuses the original log chart or the retained triangular factor at the mean
    const ChartFrame& mean_frame_reference_() const {
        if constexpr (internals::is_log_euclidean_spd_geometry<Geometry>)
            return *mean_chart_;
        else
            return *mean_frame_;
    }

    /// @brief borrows one node's isometric coordinates without copying its retained factor
    const Tangent& node_chart_(std::size_t index) const {
        return internals::p1_flat_frame_coordinates<Geometry>(*node_frames_[index]);
    }

    /// @brief accumulates one chart coefficient while retaining its rounding correction
    static void compensated_add_(
      AccumulationTangent& sum, AccumulationTangent& correction, int i, int j, AccumulationScalar contribution) {
        const AccumulationScalar current_sum = static_cast<AccumulationScalar>(sum(i, j));
        const AccumulationScalar corrected = contribution - static_cast<AccumulationScalar>(correction(i, j));
        const AccumulationScalar next = current_sum + corrected;
        correction(i, j) = (next - current_sum) - corrected;
        sum(i, j) = next;
    }

    /// @brief sums represented weights with compensated arithmetic
    static double compensated_weight_total_(std::span<const double> weights) {
        double total = 0;
        double correction = 0;
        for (const double weight : weights) {
            if (weight == 0) continue;
            const double corrected = weight - correction;
            const double next = total + corrected;
            correction = (next - total) - corrected;
            total = next;
        }
        return total;
    }

    /// @brief allocates a zero chart using the promoted accumulation scalar
    AccumulationTangent make_accumulation_tangent_() const {
        AccumulationTangent result;
        if constexpr (Order_ == fdapde::Dynamic) { result.resize(geometry_.order(), geometry_.order()); }
        for (int i = 0; i < geometry_.order(); ++i) {
            for (int j = 0; j <= i; ++j) { result(i, j) = AccumulationScalar(0); }
        }
        return result;
    }

    /// @brief allocates a zero tangent with the geometry order
    Tangent make_zero_tangent_() const {
        Tangent result;
        if constexpr (Order_ == fdapde::Dynamic) { result.resize(geometry_.order(), geometry_.order()); }
        for (int i = 0; i < geometry_.order(); ++i) {
            for (int j = 0; j <= i; ++j) { result(i, j) = Scalar_(0); }
        }
        return result;
    }

    /// @brief rounds the accumulated chart to the geometry scalar
    Tangent to_tangent_(const AccumulationTangent& source) const {
        Tangent result = make_zero_tangent_();
        for (int i = 0; i < geometry_.order(); ++i) {
            for (int j = 0; j <= i; ++j) { result(i, j) = static_cast<Scalar_>(source(i, j)); }
        }
        return result;
    }

    /// @brief scales a chart with promoted accumulation arithmetic
    Tangent scaled_tangent_(const Tangent& source, AccumulationScalar coefficient) const {
        Tangent result = make_zero_tangent_();
        for (int i = 0; i < geometry_.order(); ++i) {
            for (int j = 0; j <= i; ++j) {
                result(i, j) = static_cast<Scalar_>(
                  static_cast<AccumulationScalar>(coefficient) * static_cast<AccumulationScalar>(source(i, j)));
            }
        }
        return result;
    }

    /// @brief differentiates one represented normalized nodal weight
    AccumulationScalar mixed_nodal_coefficient_(
      double weight_direction, AccumulationScalar direction_total, std::size_t node_index) const {
        const AccumulationScalar total = static_cast<AccumulationScalar>(normalized_total_);
        return (static_cast<AccumulationScalar>(weight_direction) -
                direction_total * static_cast<AccumulationScalar>(result_.normalized_weights[node_index]) / total) /
               total;
    }

    /// @brief rejects a nodal matrix whose order differs from the geometry
    template <SPDLike Node> void check_node_shape_(const Node& node) const {
        fdapde_strong_assert(
          node.rows() == geometry_.order() && node.cols() == geometry_.order(), std::invalid_argument,
          "P1 geodesic nodal value has incompatible dimensions");
    }

    /// @brief requires finite coefficients with zero sum up to the P1 rounding tolerance
    void validate_weight_direction_(std::span<const double> direction) const {
        fdapde_strong_assert(
          direction.size() == nodes_.size(), std::invalid_argument,
          "P1 geodesic weight-direction and nodal-value counts must match");

        long double sum = 0;
        long double correction = 0;
        long double absolute_sum = 0;
        for (const double coefficient : direction) {
            fdapde_strong_assert(
              std::isfinite(coefficient), std::invalid_argument, "P1 geodesic weight directions must be finite");
            const long double value = static_cast<long double>(coefficient);
            const long double corrected = value - correction;
            const long double next = sum + corrected;
            correction = (next - sum) - corrected;
            sum = next;
            absolute_sum += std::abs(value);
        }
        const long double tolerance =
          static_cast<long double>(p1_weight_sum_tolerance(direction.size())) * std::max(1.0L, absolute_sum);
        fdapde_strong_assert(
          std::abs(sum) <= tolerance, std::invalid_argument, "P1 geodesic weight directions must sum to zero");
    }

    /// @brief rejects derivative evaluation when the mean has not converged
    void require_ready_() const {
        fdapde_strong_assert(ready_, std::logic_error, "P1 geodesic derivatives require a converged value");
    }

    Geometry geometry_;
    Nodes nodes_;
    P1ValueResult<Point> result_;
    std::vector<std::optional<ChartFrame>> node_frames_;
    std::optional<ChartFrame> mean_frame_;
    double normalized_total_ = 0;
    std::optional<Tangent> mean_chart_;
    std::optional<std::size_t> vertex_index_;
    bool ready_ = false;
};

}   // namespace gfe
}   // namespace fdapde

#endif   // __FDAPDE_GFE_P1_GEODESIC_LINEARIZATION_FLAT_H__
