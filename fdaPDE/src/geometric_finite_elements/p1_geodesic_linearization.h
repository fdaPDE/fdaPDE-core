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

// Builds an owning derivative snapshot and validates every nodal geometry
// point, including zero-weight nodes. The value-only evaluator retains its
// support-skipping behavior. Derivative actions require a converged result.
template <typename Scalar_, int Order_>
P1GeodesicLinearization<manifold::LogEuclideanSPDGeometry<Scalar_, Order_>> p1_geodesic_linearization(
  const manifold::LogEuclideanSPDGeometry<Scalar_, Order_>& geometry,
  std::span<const typename manifold::LogEuclideanSPDGeometry<Scalar_, Order_>::Point> nodal_values,
  std::span<const double> barycentric_weights);

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
   private:
    using AccumulationScalar = std::common_type_t<Scalar_, double>;
    using AccumulationTangent = fdapde::linalg::SymmetricMatrix<AccumulationScalar, Order_, Order_>;

    P1GeodesicLinearization(
      const Geometry& geometry, std::span<const Point> nodal_values, std::span<const double> barycentric_weights) :
        geometry_(geometry),
        nodes_(nodal_values.begin(), nodal_values.end()),
        result_(p1_geodesic_value(geometry_, std::span<const Point>(nodes_), barycentric_weights)),
        node_logs_(nodes_.size()) {
        if (!result_.converged()) return;

        normalized_total_ = compensated_weight_total_(result_.normalized_weights);
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

}   // namespace gfe
}   // namespace fdapde

#endif   // __FDAPDE_GFE_P1_GEODESIC_LINEARIZATION_H__
