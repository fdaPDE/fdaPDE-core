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

#ifndef __FDAPDE_GFE_P1_GEODESIC_LINEARIZATION_H__
#define __FDAPDE_GFE_P1_GEODESIC_LINEARIZATION_H__

#include "header_check.h"

namespace fdapde {
namespace gfe {

template <typename Geometry> class P1GeodesicLinearization;

struct P1GeodesicLinearizationOptions {
    manifold::WeightedKarcherMeanOptions mean;
    manifold::PositiveDefiniteCGOptions linear_solve;
};

template <typename Derivative> struct P1DerivativeResult {
    // The differential candidate. On iterative paths it is derived from the
    // last linear-solve iterate and is certified only when converged() is true.
    Derivative derivative;
    double residual_norm = std::numeric_limits<double>::quiet_NaN();
    std::size_t iterations = 0;
    manifold::PositiveDefiniteCGStopReason stop_reason = manifold::PositiveDefiniteCGStopReason::max_iterations;

    bool converged() const { return stop_reason == manifold::PositiveDefiniteCGStopReason::residual_tolerance; }
};

struct P1LinearSolveStatus {
    double residual_norm = std::numeric_limits<double>::quiet_NaN();
    std::size_t iterations = 0;
    manifold::PositiveDefiniteCGStopReason stop_reason = manifold::PositiveDefiniteCGStopReason::max_iterations;

    bool converged() const { return stop_reason == manifold::PositiveDefiniteCGStopReason::residual_tolerance; }
};

template <typename Derivative> struct P1MixedDerivativeResult {
    // The derivative candidate is assembled from the three last solver
    // iterates and is certified only when converged() is true. Each method
    // documents the dependency order represented by solve_statuses.
    Derivative derivative;
    std::array<P1LinearSolveStatus, 3> solve_statuses;

    bool converged() const {
        for (const P1LinearSolveStatus& status : solve_statuses) {
            if (!status.converged()) return false;
        }
        return true;
    }
};

// Builds an owning derivative snapshot and validates all nodal geometry
// points, including zero-weight nodes. The value-only evaluator retains its
// support-skipping behavior. Derivative actions require a converged result.
template <typename Scalar_, int Order_>
P1GeodesicLinearization<manifold::LogEuclideanSPDGeometry<Scalar_, Order_>> p1_geodesic_linearization(
  const manifold::LogEuclideanSPDGeometry<Scalar_, Order_>& geometry,
  std::span<const typename manifold::LogEuclideanSPDGeometry<Scalar_, Order_>::Point> nodal_values,
  std::span<const double> barycentric_weights);

template <typename Scalar_, int Order_>
P1GeodesicLinearization<manifold::AffineInvariantSPDGeometry<Scalar_, Order_>> p1_geodesic_linearization(
  const manifold::AffineInvariantSPDGeometry<Scalar_, Order_>& geometry,
  std::span<const typename manifold::AffineInvariantSPDGeometry<Scalar_, Order_>::Point> nodal_values,
  std::span<const double> barycentric_weights, const P1GeodesicLinearizationOptions& options = {});

template <typename Scalar_, int Order_>
P1GeodesicLinearization<manifold::AffineInvariantSPDGeometry<Scalar_, Order_>> p1_geodesic_linearization(
  const manifold::AffineInvariantSPDGeometry<Scalar_, Order_>& geometry,
  std::span<const typename manifold::AffineInvariantSPDGeometry<Scalar_, Order_>::Point> nodal_values,
  std::span<const double> barycentric_weights,
  const typename manifold::AffineInvariantSPDGeometry<Scalar_, Order_>::Point& initial,
  const P1GeodesicLinearizationOptions& options = {});

template <typename Scalar_, int Order_>
class P1GeodesicLinearization<manifold::LogEuclideanSPDGeometry<Scalar_, Order_>> {
   public:
    using Geometry = manifold::LogEuclideanSPDGeometry<Scalar_, Order_>;
    using Point = typename Geometry::Point;
    using Tangent = typename Geometry::Tangent;

    const P1ValueResult<Point>& result() const& noexcept { return result_; }
    const P1ValueResult<Point>& result() const&& = delete;

    // The direction is based at result().normalized_weights. It must be
    // finite and sum to zero; it is never projected or renormalized. The
    // centered quotient formula differentiates the represented normalized
    // mean using the same compensated weight total as the value.
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
                      (static_cast<AccumulationScalar>((*node_logs_[node_index])(i, j)) -
                       static_cast<AccumulationScalar>((*mean_chart_)(i, j)));
                    compensated_add_(chart_direction, correction, i, j, contribution);
                }
            }
        }
        if (!has_nonzero_direction) return make_zero_tangent_();
        return Tangent(fdapde::linalg::matrix_exp_frechet(*mean_chart_, to_tangent_(chart_direction)));
    }

    // Directions at nodes with zero effective weight are not inspected,
    // because their contribution is identically zero.
    Tangent nodal_jvp(std::span<const Tangent> directions) const {
        require_ready_();
        if (directions.size() != nodes_.size()) {
            throw std::invalid_argument("P1 geodesic nodal-direction and nodal-value counts must match");
        }

        if (vertex_index_) {
            static_cast<void>(fdapde::linalg::matrix_log_frechet(nodes_[*vertex_index_], directions[*vertex_index_]));
            return directions[*vertex_index_];
        }

        auto chart_direction = make_accumulation_tangent_();
        auto correction = make_accumulation_tangent_();
        for (std::size_t node_index = 0; node_index < nodes_.size(); ++node_index) {
            const double weight = result_.normalized_weights[node_index];
            if (weight == 0) continue;

            const auto node_direction = fdapde::linalg::matrix_log_frechet(nodes_[node_index], directions[node_index]);
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
        return Tangent(fdapde::linalg::matrix_exp_frechet(*mean_chart_, to_tangent_(chart_direction)));
    }

    // This is the Riemannian adjoint of nodal_jvp. The argument and
    // returned covectors are represented by metric-dual tangent vectors
    // for the log-Euclidean metric, not by ambient Frobenius gradients.
    std::vector<Tangent> nodal_vjp(const Tangent& value_gradient) const {
        require_ready_();
        const Tangent value_chart_gradient(fdapde::linalg::matrix_log_frechet(result_.value, value_gradient));
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
            result[node_index] = Tangent(fdapde::linalg::matrix_exp_frechet(*node_logs_[node_index], scaled));
        }
        return result;
    }

    // Covariant nodal derivative of the spatial/weight JVP. The weight
    // direction is based at result().normalized_weights, must be finite and
    // sum to zero, and follows the same represented normalized-total
    // convention as weight_jvp.
    Tangent covariant_mixed_nodal_jvp(
      std::span<const double> weight_direction, std::span<const Tangent> nodal_directions) const {
        require_ready_();
        validate_weight_direction_(weight_direction);
        if (nodal_directions.size() != nodes_.size()) {
            throw std::invalid_argument("P1 geodesic nodal-direction and nodal-value counts must match");
        }

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

            const auto node_direction =
              fdapde::linalg::matrix_log_frechet(nodes_[node_index], nodal_directions[node_index]);
            for (int i = 0; i < geometry_.order(); ++i) {
                for (int j = 0; j <= i; ++j) {
                    const AccumulationScalar contribution =
                      coefficient * static_cast<AccumulationScalar>(node_direction(i, j));
                    compensated_add_(chart_direction, correction, i, j, contribution);
                }
            }
        }
        if (!has_nonzero_coefficient) return make_zero_tangent_();
        return Tangent(fdapde::linalg::matrix_exp_frechet(*mean_chart_, to_tangent_(chart_direction)));
    }

    // Riemannian adjoint, in the nodal variable, of
    // covariant_mixed_nodal_jvp for a fixed weight direction. The argument
    // and returned covectors are log-Euclidean metric-dual tangents.
    std::vector<Tangent>
    covariant_mixed_nodal_vjp(std::span<const double> weight_direction, const Tangent& value_gradient) const {
        require_ready_();
        validate_weight_direction_(weight_direction);
        const Tangent value_chart_gradient(fdapde::linalg::matrix_log_frechet(result_.value, value_gradient));
        std::vector<Tangent> result(nodes_.size(), make_zero_tangent_());
        const AccumulationScalar direction_total =
          static_cast<AccumulationScalar>(compensated_weight_total_(weight_direction));
        for (std::size_t node_index = 0; node_index < nodes_.size(); ++node_index) {
            const AccumulationScalar coefficient =
              mixed_nodal_coefficient_(weight_direction[node_index], direction_total, node_index);
            if (coefficient == 0) continue;
            const Tangent scaled = scaled_tangent_(value_chart_gradient, coefficient);
            result[node_index] = Tangent(fdapde::linalg::matrix_exp_frechet(*node_logs_[node_index], scaled));
        }
        return result;
    }
   private:
    using AccumulationScalar = std::common_type_t<Scalar_, double>;
    using AccumulationTangent = fdapde::linalg::SymmetricMatrix<AccumulationScalar, Order_, Order_>;

    P1GeodesicLinearization(
      const Geometry& geometry, std::span<const Point> nodal_values, std::span<const double> barycentric_weights) :
        geometry_(geometry),
        nodes_(nodal_values.begin(), nodal_values.end()),
        result_(p1_geodesic_value(geometry_, std::span<const Point>(nodes_), barycentric_weights)),
        node_logs_(nodes_.size()) {
        std::size_t positive_count = 0;
        for (std::size_t node_index = 0; node_index < nodes_.size(); ++node_index) {
            check_node_shape_(nodes_[node_index]);
            node_logs_[node_index].emplace(fdapde::linalg::matrix_log(nodes_[node_index]));
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
                      static_cast<AccumulationScalar>((*node_logs_[node_index])(i, j));
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
        ready_ = true;
    }

    static void compensated_add_(
      AccumulationTangent& sum, AccumulationTangent& correction, int i, int j, AccumulationScalar contribution) {
        const AccumulationScalar current_sum = static_cast<AccumulationScalar>(sum(i, j));
        const AccumulationScalar corrected = contribution - static_cast<AccumulationScalar>(correction(i, j));
        const AccumulationScalar next = current_sum + corrected;
        correction(i, j) = (next - current_sum) - corrected;
        sum(i, j) = next;
    }

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

    AccumulationTangent make_accumulation_tangent_() const {
        AccumulationTangent result;
        if constexpr (Order_ == fdapde::Dynamic) { result.resize(geometry_.order(), geometry_.order()); }
        for (int i = 0; i < geometry_.order(); ++i) {
            for (int j = 0; j <= i; ++j) { result(i, j) = AccumulationScalar(0); }
        }
        return result;
    }

    Tangent make_zero_tangent_() const {
        Tangent result;
        if constexpr (Order_ == fdapde::Dynamic) { result.resize(geometry_.order(), geometry_.order()); }
        for (int i = 0; i < geometry_.order(); ++i) {
            for (int j = 0; j <= i; ++j) { result(i, j) = Scalar_(0); }
        }
        return result;
    }

    Tangent to_tangent_(const AccumulationTangent& source) const {
        Tangent result = make_zero_tangent_();
        for (int i = 0; i < geometry_.order(); ++i) {
            for (int j = 0; j <= i; ++j) { result(i, j) = static_cast<Scalar_>(source(i, j)); }
        }
        return result;
    }

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

    AccumulationScalar mixed_nodal_coefficient_(
      double weight_direction, AccumulationScalar direction_total, std::size_t node_index) const {
        const AccumulationScalar total = static_cast<AccumulationScalar>(normalized_total_);
        return (
                 static_cast<AccumulationScalar>(weight_direction) -
                 direction_total * static_cast<AccumulationScalar>(result_.normalized_weights[node_index]) / total) /
               total;
    }

    void check_node_shape_(const Point& node) const {
        if (node.rows() != geometry_.order() || node.cols() != geometry_.order()) {
            throw std::invalid_argument("P1 geodesic nodal value has incompatible dimensions");
        }
    }

    void validate_weight_direction_(std::span<const double> direction) const {
        if (direction.size() != nodes_.size()) {
            throw std::invalid_argument("P1 geodesic weight-direction and nodal-value counts must match");
        }

        long double sum = 0;
        long double correction = 0;
        long double absolute_sum = 0;
        for (const double coefficient : direction) {
            if (!std::isfinite(coefficient)) {
                throw std::invalid_argument("P1 geodesic weight directions must be finite");
            }
            const long double value = static_cast<long double>(coefficient);
            const long double corrected = value - correction;
            const long double next = sum + corrected;
            correction = (next - sum) - corrected;
            sum = next;
            absolute_sum += std::abs(value);
        }
        const long double tolerance =
          static_cast<long double>(p1_weight_sum_tolerance(direction.size())) * std::max(1.0L, absolute_sum);
        if (std::abs(sum) > tolerance) {
            throw std::invalid_argument("P1 geodesic weight directions must sum to zero");
        }
    }

    void require_ready_() const {
        if (!ready_) { throw std::logic_error("P1 geodesic derivatives require a converged value"); }
    }

    template <typename OtherScalar_, int OtherOrder_>
    friend P1GeodesicLinearization<manifold::LogEuclideanSPDGeometry<OtherScalar_, OtherOrder_>>
    p1_geodesic_linearization(
      const manifold::LogEuclideanSPDGeometry<OtherScalar_, OtherOrder_>& geometry,
      std::span<const typename manifold::LogEuclideanSPDGeometry<OtherScalar_, OtherOrder_>::Point> nodal_values,
      std::span<const double> barycentric_weights);

    Geometry geometry_;
    std::vector<Point> nodes_;
    P1ValueResult<Point> result_;
    std::vector<std::optional<Tangent>> node_logs_;
    double normalized_total_ = 0;
    std::optional<Tangent> mean_chart_;
    std::optional<std::size_t> vertex_index_;
    bool ready_ = false;
};

template <typename Scalar_, int Order_>
P1GeodesicLinearization<manifold::LogEuclideanSPDGeometry<Scalar_, Order_>> p1_geodesic_linearization(
  const manifold::LogEuclideanSPDGeometry<Scalar_, Order_>& geometry,
  std::span<const typename manifold::LogEuclideanSPDGeometry<Scalar_, Order_>::Point> nodal_values,
  std::span<const double> barycentric_weights) {
    return P1GeodesicLinearization<manifold::LogEuclideanSPDGeometry<Scalar_, Order_>>(
      geometry, nodal_values, barycentric_weights);
}

template <typename Scalar_, int Order_>
class P1GeodesicLinearization<manifold::AffineInvariantSPDGeometry<Scalar_, Order_>> {
   public:
    using Geometry = manifold::AffineInvariantSPDGeometry<Scalar_, Order_>;
    using Point = typename Geometry::Point;
    using Tangent = typename Geometry::Tangent;

    const P1ValueResult<Point>& result() const& noexcept { return result_; }
    const P1ValueResult<Point>& result() const&& = delete;

    // The direction is based at result().normalized_weights. It must be
    // finite and sum to zero; it is never projected or renormalized.
    P1DerivativeResult<Tangent> weight_jvp(std::span<const double> direction) const {
        require_ready_();
        validate_weight_direction_(direction);
        return solve_(weight_right_hand_side_(direction));
    }

    // Directions at nodes with zero effective weight are not inspected,
    // because their contribution is identically zero.
    P1DerivativeResult<Tangent> nodal_jvp(std::span<const Tangent> directions) const {
        require_ready_();
        if (directions.size() != nodes_.size()) {
            throw std::invalid_argument("P1 geodesic nodal-direction and nodal-value counts must match");
        }
        if (vertex_index_) {
            require_finite_tangent_(
              nodes_[*vertex_index_], directions[*vertex_index_], "P1 geodesic nodal directions must be finite");
            return {directions[*vertex_index_], 0, 0, manifold::PositiveDefiniteCGStopReason::residual_tolerance};
        }

        Tangent right_hand_side = geometry_.zero_tangent(result_.value);
        for (std::size_t node_index = 0; node_index < nodes_.size(); ++node_index) {
            const double weight = effective_weights_[node_index];
            if (weight == 0) continue;
            require_finite_tangent_(
              nodes_[node_index], directions[node_index], "P1 geodesic nodal directions must be finite");
            const Tangent target_action =
              geometry_.logarithm_target_jvp(result_.value, nodes_[node_index], directions[node_index]);
            right_hand_side = geometry_.linear_combination(result_.value, 1, right_hand_side, weight, target_action);
        }
        return solve_(right_hand_side);
    }

    // This is the Riemannian adjoint of nodal_jvp. The argument and
    // returned covectors are represented by AIRM metric-dual tangent vectors,
    // not by ambient Frobenius gradients.
    P1DerivativeResult<std::vector<Tangent>> nodal_vjp(const Tangent& value_gradient) const {
        require_ready_();
        require_finite_tangent_(result_.value, value_gradient, "P1 geodesic value gradients must be finite");

        std::vector<Tangent> pullback;
        pullback.reserve(nodes_.size());
        for (const Point& node : nodes_) { pullback.push_back(geometry_.zero_tangent(node)); }
        if (vertex_index_) {
            pullback[*vertex_index_] = value_gradient;
            return {std::move(pullback), 0, 0, manifold::PositiveDefiniteCGStopReason::residual_tolerance};
        }

        const auto solve = solve_tangent_(value_gradient);
        for (std::size_t node_index = 0; node_index < nodes_.size(); ++node_index) {
            const double weight = effective_weights_[node_index];
            if (weight == 0) continue;
            const Tangent node_dual = geometry_.logarithm_target_vjp(result_.value, nodes_[node_index], solve.solution);
            pullback[node_index] =
              geometry_.linear_combination(nodes_[node_index], weight, node_dual, 0, pullback[node_index]);
        }
        return {std::move(pullback), solve.residual_norm, solve.iterations, solve.stop_reason};
    }

    // Covariant nodal derivative of the spatial/weight JVP. The three
    // solve statuses are ordered as the weight, nodal, and mixed solves.
    P1MixedDerivativeResult<Tangent> covariant_mixed_nodal_jvp(
      std::span<const double> weight_direction, std::span<const Tangent> nodal_directions) const {
        require_ready_();
        validate_weight_direction_(weight_direction);
        if (nodal_directions.size() != nodes_.size()) {
            throw std::invalid_argument("P1 geodesic nodal-direction and nodal-value counts must match");
        }

        const double direction_total = compensated_weight_total_(weight_direction);
        Tangent weight_right_hand_side = weight_right_hand_side_(weight_direction);
        Tangent nodal_right_hand_side = geometry_.zero_tangent(result_.value);
        std::vector<Tangent> target_actions(nodes_.size(), geometry_.zero_tangent(result_.value));
        for (std::size_t node_index = 0; node_index < nodes_.size(); ++node_index) {
            const double alpha = effective_weights_[node_index];
            const double gamma = mixed_weight_coefficient_(weight_direction[node_index], direction_total, node_index);
            if (alpha == 0 && gamma == 0) continue;

            require_finite_tangent_(
              nodes_[node_index], nodal_directions[node_index], "P1 geodesic nodal directions must be finite");
            target_actions[node_index] =
              geometry_.logarithm_target_jvp(result_.value, nodes_[node_index], nodal_directions[node_index]);
            if (alpha != 0) {
                nodal_right_hand_side = geometry_.linear_combination(
                  result_.value, 1, nodal_right_hand_side, alpha, target_actions[node_index]);
            }
        }

        auto weight_solve = solve_tangent_(weight_right_hand_side);
        auto nodal_solve = solve_tangent_(nodal_right_hand_side);
        Tangent mixed_right_hand_side = geometry_.zero_tangent(result_.value);
        for (std::size_t node_index = 0; node_index < nodes_.size(); ++node_index) {
            const double alpha = effective_weights_[node_index];
            const double gamma = mixed_weight_coefficient_(weight_direction[node_index], direction_total, node_index);
            if (gamma != 0) {
                const Tangent hessian_action = geometry_.half_squared_distance_hessian_vector(
                  result_.value, nodes_[node_index], nodal_solve.solution);
                const Tangent component = geometry_.linear_combination(
                  result_.value, 1, target_actions[node_index], -1, hessian_action);
                mixed_right_hand_side = geometry_.linear_combination(
                  result_.value, 1, mixed_right_hand_side, gamma, component);
            }
            if (alpha == 0) continue;

            const Tangent mixed_action = geometry_.half_squared_distance_hessian_covariant_jvp(
              result_.value, nodes_[node_index], nodal_solve.solution, nodal_directions[node_index],
              weight_solve.solution);
            mixed_right_hand_side =
              geometry_.linear_combination(result_.value, 1, mixed_right_hand_side, -alpha, mixed_action);
        }
        auto mixed_solve = solve_tangent_(mixed_right_hand_side);
        return {
          std::move(mixed_solve.solution),
          {solve_status_(weight_solve), solve_status_(nodal_solve), solve_status_(mixed_solve)}};
    }

    // Riemannian adjoint, in the nodal variable, of
    // covariant_mixed_nodal_jvp for a fixed weight direction. The three
    // solve statuses are ordered as the weight, output, and pullback solves.
    P1MixedDerivativeResult<std::vector<Tangent>>
    covariant_mixed_nodal_vjp(std::span<const double> weight_direction, const Tangent& value_gradient) const {
        require_ready_();
        validate_weight_direction_(weight_direction);
        require_finite_tangent_(result_.value, value_gradient, "P1 geodesic value gradients must be finite");

        const double direction_total = compensated_weight_total_(weight_direction);
        auto weight_solve = solve_tangent_(weight_right_hand_side_(weight_direction));
        auto output_solve = solve_tangent_(value_gradient);
        Tangent pullback_right_hand_side = geometry_.zero_tangent(result_.value);
        std::vector<std::optional<Tangent>> mixed_target_duals(nodes_.size());
        for (std::size_t node_index = 0; node_index < nodes_.size(); ++node_index) {
            const double alpha = effective_weights_[node_index];
            const double gamma = mixed_weight_coefficient_(weight_direction[node_index], direction_total, node_index);
            if (gamma != 0) {
                const Tangent hessian_action = geometry_.half_squared_distance_hessian_vector(
                  result_.value, nodes_[node_index], output_solve.solution);
                pullback_right_hand_side = geometry_.linear_combination(
                  result_.value, 1, pullback_right_hand_side, -gamma, hessian_action);
            }
            if (alpha == 0) continue;

            auto mixed_duals = geometry_.half_squared_distance_hessian_covariant_vjp(
              result_.value, nodes_[node_index], weight_solve.solution, output_solve.solution);
            pullback_right_hand_side = geometry_.linear_combination(
              result_.value, 1, pullback_right_hand_side, -alpha, mixed_duals.first);
            mixed_target_duals[node_index].emplace(std::move(mixed_duals.second));
        }
        auto pullback_solve = solve_tangent_(pullback_right_hand_side);

        std::vector<Tangent> pullback;
        pullback.reserve(nodes_.size());
        for (const Point& node : nodes_) { pullback.push_back(geometry_.zero_tangent(node)); }
        for (std::size_t node_index = 0; node_index < nodes_.size(); ++node_index) {
            const double alpha = effective_weights_[node_index];
            const double gamma = mixed_weight_coefficient_(weight_direction[node_index], direction_total, node_index);
            if (gamma != 0) {
                const Tangent direct =
                  geometry_.logarithm_target_vjp(result_.value, nodes_[node_index], output_solve.solution);
                pullback[node_index] =
                  geometry_.linear_combination(nodes_[node_index], 1, pullback[node_index], gamma, direct);
            }
            if (alpha == 0) continue;

            pullback[node_index] = geometry_.linear_combination(
              nodes_[node_index], 1, pullback[node_index], -alpha, *mixed_target_duals[node_index]);
            const Tangent indirect =
              geometry_.logarithm_target_vjp(result_.value, nodes_[node_index], pullback_solve.solution);
            pullback[node_index] =
              geometry_.linear_combination(nodes_[node_index], 1, pullback[node_index], alpha, indirect);
        }
        return {
          std::move(pullback),
          {solve_status_(weight_solve), solve_status_(output_solve), solve_status_(pullback_solve)}};
    }
   private:
    P1GeodesicLinearization(
      const Geometry& geometry, std::span<const Point> nodal_values, std::span<const double> barycentric_weights,
      const P1GeodesicLinearizationOptions& options) :
        geometry_(geometry),
        nodes_(nodal_values.begin(), nodal_values.end()),
        linear_solver_(options.linear_solve),
        result_(p1_geodesic_value(geometry_, std::span<const Point>(nodes_), barycentric_weights, options.mean)) {
        initialize_();
    }

    P1GeodesicLinearization(
      const Geometry& geometry, std::span<const Point> nodal_values, std::span<const double> barycentric_weights,
      const Point& initial, const P1GeodesicLinearizationOptions& options) :
        geometry_(geometry),
        nodes_(nodal_values.begin(), nodal_values.end()),
        linear_solver_(options.linear_solve),
        result_(
          p1_geodesic_value(geometry_, std::span<const Point>(nodes_), barycentric_weights, initial, options.mean)) {
        initialize_();
    }

    void initialize_() {
        for (const Point& node : nodes_) {
            const double self_distance = geometry_.distance(node, node);
            if (!std::isfinite(self_distance)) {
                throw std::invalid_argument("P1 geodesic nodal values must have finite geometry");
            }
        }
        if (!result_.converged()) return;

        normalized_total_ = compensated_weight_total_(result_.normalized_weights);
        effective_weights_.reserve(result_.normalized_weights.size());
        std::size_t positive_count = 0;
        for (const double weight : result_.normalized_weights) {
            effective_weights_.push_back(weight / normalized_total_);
            if (weight > 0) {
                ++positive_count;
                vertex_index_ = effective_weights_.size() - 1;
            }
        }
        if (positive_count != 1) vertex_index_.reset();

        node_logs_.reserve(nodes_.size());
        for (const Point& node : nodes_) {
            node_logs_.push_back(geometry_.logarithm(result_.value, node));
            require_finite_tangent_(result_.value, node_logs_.back(), "P1 geodesic nodal logarithms must be finite");
        }

        mean_residual_.emplace(geometry_.zero_tangent(result_.value));
        for (std::size_t node_index = 0; node_index < nodes_.size(); ++node_index) {
            const double weight = effective_weights_[node_index];
            if (weight == 0) continue;
            *mean_residual_ =
              geometry_.linear_combination(result_.value, 1, *mean_residual_, weight, node_logs_[node_index]);
        }
        ready_ = true;
    }

    Tangent hessian_action_(const Tangent& direction) const {
        Tangent result = geometry_.zero_tangent(result_.value);
        for (std::size_t node_index = 0; node_index < nodes_.size(); ++node_index) {
            const double weight = effective_weights_[node_index];
            if (weight == 0) continue;
            const Tangent component =
              geometry_.half_squared_distance_hessian_vector(result_.value, nodes_[node_index], direction);
            result = geometry_.linear_combination(result_.value, 1, result, weight, component);
        }
        return result;
    }

    manifold::PositiveDefiniteCGResult<Tangent> solve_tangent_(const Tangent& right_hand_side) const {
        auto hessian = [this](const Tangent& direction) { return hessian_action_(direction); };
        return linear_solver_.solve(hessian, geometry_, result_.value, right_hand_side);
    }

    P1DerivativeResult<Tangent> solve_(const Tangent& right_hand_side) const {
        auto solve = solve_tangent_(right_hand_side);
        return {std::move(solve.solution), solve.residual_norm, solve.iterations, solve.stop_reason};
    }

    Tangent weight_right_hand_side_(std::span<const double> direction) const {
        Tangent right_hand_side = geometry_.zero_tangent(result_.value);
        for (std::size_t node_index = 0; node_index < nodes_.size(); ++node_index) {
            const double coefficient = direction[node_index];
            if (coefficient == 0) continue;
            const Tangent centered_log =
              geometry_.linear_combination(result_.value, 1, node_logs_[node_index], -1, *mean_residual_);
            right_hand_side = geometry_.linear_combination(
              result_.value, 1, right_hand_side, coefficient / normalized_total_, centered_log);
        }
        return right_hand_side;
    }

    static P1LinearSolveStatus solve_status_(const manifold::PositiveDefiniteCGResult<Tangent>& solve) {
        return {solve.residual_norm, solve.iterations, solve.stop_reason};
    }

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

    double mixed_weight_coefficient_(
      double weight_direction, double direction_total, std::size_t node_index) const {
        return (
                 weight_direction -
                 direction_total * result_.normalized_weights[node_index] / normalized_total_) /
               normalized_total_;
    }

    void validate_weight_direction_(std::span<const double> direction) const {
        if (direction.size() != nodes_.size()) {
            throw std::invalid_argument("P1 geodesic weight-direction and nodal-value counts must match");
        }

        long double sum = 0;
        long double correction = 0;
        long double absolute_sum = 0;
        for (const double coefficient : direction) {
            if (!std::isfinite(coefficient)) {
                throw std::invalid_argument("P1 geodesic weight directions must be finite");
            }
            const long double value = static_cast<long double>(coefficient);
            const long double corrected = value - correction;
            const long double next = sum + corrected;
            correction = (next - sum) - corrected;
            sum = next;
            absolute_sum += std::abs(value);
        }
        const long double tolerance =
          static_cast<long double>(p1_weight_sum_tolerance(direction.size())) * std::max(1.0L, absolute_sum);
        if (std::abs(sum) > tolerance) {
            throw std::invalid_argument("P1 geodesic weight directions must sum to zero");
        }
    }

    void require_finite_tangent_(const Point& point, const Tangent& tangent, const char* description) const {
        if (!std::isfinite(geometry_.norm(point, tangent))) { throw std::invalid_argument(description); }
    }

    void require_ready_() const {
        if (!ready_) { throw std::logic_error("P1 geodesic derivatives require a converged value"); }
    }

    template <typename OtherScalar_, int OtherOrder_>
    friend P1GeodesicLinearization<manifold::AffineInvariantSPDGeometry<OtherScalar_, OtherOrder_>>
    p1_geodesic_linearization(
      const manifold::AffineInvariantSPDGeometry<OtherScalar_, OtherOrder_>& geometry,
      std::span<const typename manifold::AffineInvariantSPDGeometry<OtherScalar_, OtherOrder_>::Point> nodal_values,
      std::span<const double> barycentric_weights, const P1GeodesicLinearizationOptions& options);

    template <typename OtherScalar_, int OtherOrder_>
    friend P1GeodesicLinearization<manifold::AffineInvariantSPDGeometry<OtherScalar_, OtherOrder_>>
    p1_geodesic_linearization(
      const manifold::AffineInvariantSPDGeometry<OtherScalar_, OtherOrder_>& geometry,
      std::span<const typename manifold::AffineInvariantSPDGeometry<OtherScalar_, OtherOrder_>::Point> nodal_values,
      std::span<const double> barycentric_weights,
      const typename manifold::AffineInvariantSPDGeometry<OtherScalar_, OtherOrder_>::Point& initial,
      const P1GeodesicLinearizationOptions& options);

    Geometry geometry_;
    std::vector<Point> nodes_;
    manifold::PositiveDefiniteConjugateGradient linear_solver_;
    P1ValueResult<Point> result_;
    std::vector<double> effective_weights_;
    std::vector<Tangent> node_logs_;
    double normalized_total_ = 0;
    std::optional<Tangent> mean_residual_;
    std::optional<std::size_t> vertex_index_;
    bool ready_ = false;
};

template <typename Scalar_, int Order_>
P1GeodesicLinearization<manifold::AffineInvariantSPDGeometry<Scalar_, Order_>> p1_geodesic_linearization(
  const manifold::AffineInvariantSPDGeometry<Scalar_, Order_>& geometry,
  std::span<const typename manifold::AffineInvariantSPDGeometry<Scalar_, Order_>::Point> nodal_values,
  std::span<const double> barycentric_weights, const P1GeodesicLinearizationOptions& options) {
    return P1GeodesicLinearization<manifold::AffineInvariantSPDGeometry<Scalar_, Order_>>(
      geometry, nodal_values, barycentric_weights, options);
}

template <typename Scalar_, int Order_>
P1GeodesicLinearization<manifold::AffineInvariantSPDGeometry<Scalar_, Order_>> p1_geodesic_linearization(
  const manifold::AffineInvariantSPDGeometry<Scalar_, Order_>& geometry,
  std::span<const typename manifold::AffineInvariantSPDGeometry<Scalar_, Order_>::Point> nodal_values,
  std::span<const double> barycentric_weights,
  const typename manifold::AffineInvariantSPDGeometry<Scalar_, Order_>::Point& initial,
  const P1GeodesicLinearizationOptions& options) {
    return P1GeodesicLinearization<manifold::AffineInvariantSPDGeometry<Scalar_, Order_>>(
      geometry, nodal_values, barycentric_weights, initial, options);
}

}   // namespace gfe
}   // namespace fdapde

#endif   // __FDAPDE_GFE_P1_GEODESIC_LINEARIZATION_H__
