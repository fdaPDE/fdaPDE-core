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

#ifndef __FDAPDE_GFE_P1_INTERPOLANT_H__
#define __FDAPDE_GFE_P1_INTERPOLANT_H__
#include "header_check.h"

namespace fdapde::gfe {
namespace internals {
/// @brief attaches mean diagnostics to a failed matrix materialization
inline void require_converged(const auto& result) {
    if (result.converged()) return;
    const char* reason = "unknown";
    switch (result.stop_reason) {
    case manifold::BarycenterStopReason::max_iterations:
        reason = "max_iterations";
        break;
    case manifold::BarycenterStopReason::line_search_failed:
        reason = "line_search_failed";
        break;
    case manifold::BarycenterStopReason::non_finite_cost:
        reason = "non_finite_cost";
        break;
    case manifold::BarycenterStopReason::non_finite_gradient:
        reason = "non_finite_gradient";
        break;
    default:
        break;
    }
    std::ostringstream message;
    message << "P1 interpolation failed: stop_reason=" << reason << ", iterations=" << result.iterations
            << ", residual=" << result.stationarity_norm;
    fdapde_strong_assert(result.converged(), std::runtime_error, message.str());
}
/// @brief retains a prepared interpolant binding and copies barycentric coordinates for deferred evaluation
template <typename Interpolant> class p1_interpolant_expr : public GeometryExpr<p1_interpolant_expr<Interpolant>> {
   public:
    using Prepared = std::remove_cvref_t<Interpolant>;
    using Scalar = typename Prepared::Scalar;
    static constexpr int Rows = Prepared::Rows;
    static constexpr int Cols = Rows;
    static constexpr int StorageOrder = RowMajor;
    static constexpr int NestAsRef = 0;
    static constexpr int ReadOnly = 1;
    using assignment_executor = fdapde::internals::deleted_assignment_executor;
    /// @brief stores the prepared binding and the spatial parameter without evaluating the interpolated matrix
    p1_interpolant_expr(Interpolant interpolant, typename Prepared::Weights weights) :
        interpolant_(std::forward<Interpolant>(interpolant)), weights_(weights) { }
    /// @brief returns the prepared matrix order
    int rows() const { return interpolant_.order(); }
    /// @brief returns the prepared matrix order
    int cols() const { return rows(); }
    /// @brief evaluates a certified SPD point, throwing with diagnostics if its mean does not converge
    template <typename Policy> auto eval() const {
        return SPDMatrix<Scalar, Rows, Cols, Policy>(interpolant_.evaluate(weights_));
    }
   private:
    Interpolant interpolant_;
    typename Prepared::Weights weights_;
};
}   // namespace internals

/// @brief owns a simplex and prepared edge curves while borrowing immutable MatrixBatch coefficients and caches
/// @details the source batch must outlive this object and its expressions and must not be modified during their use
template <typename Geometry, typename Element, typename Nodes> class P1Interpolant {
   public:
    using Scalar = typename Geometry::Scalar;
    using Point = typename Geometry::Point;
    using NodeType = typename Element::NodeType;
    static constexpr int Rows = Point::Rows;
    static constexpr int NodeCount = Element::n_nodes;
    using Weights = std::array<double, NodeCount>;
    using Curve = decltype(std::declval<Geometry>().geodesic(std::declval<Point>(), std::declval<Point>()));
    /// @brief snapshots the cell geometry, retains the batch binding and prepares each edge once
    P1Interpolant(Geometry geometry, Element element, Nodes nodes, P1GeodesicLinearizationOptions options) :
        geometry_(std::move(geometry)),
        element_(std::move(element)),
        nodes_(std::forward<Nodes>(nodes)),
        options_(options) {
        fdapde_strong_assert(
          nodes_.size() == NodeCount, std::invalid_argument,
          "P1 interpolant requires one nodal value per simplex vertex");
        for (std::size_t i = 0; i < nodes_.size(); ++i)
            manifold::internals::check_spd_geometry_shape(nodes_[i], geometry_.order());
        if constexpr (Element::local_dim != Element::embed_dim) element_.supporting_plane();
        edges_.reserve(NodeCount * (NodeCount - 1) / 2);
        for (int j = 1; j < NodeCount; ++j)
            for (int i = 0; i < j; ++i) edges_.push_back(geometry_.geodesic(nodes_[i], nodes_[j]));
    }
    /// @brief returns the matrix order of the prepared geometry
    int order() const { return geometry_.order(); }
    /// @brief borrows this interpolant and stores the coordinates of a spatial point by value
    auto operator()(const NodeType& x) const& {
        return internals::p1_interpolant_expr<const P1Interpolant&>(*this, weights_(x));
    }
    /// @brief keeps a temporary interpolant alive in its deferred spatial expression
    auto operator()(const NodeType& x) && {
        const auto weights = weights_(x);
        return internals::p1_interpolant_expr<P1Interpolant>(std::move(*this), weights);
    }
    /// @brief prevents expressions from borrowing a const temporary interpolant
    void operator()(const NodeType&) const&& = delete;
    /// @brief exposes the mean result and convergence diagnostics at a spatial point
    P1ValueResult<Point> result(const NodeType& x) const { return result_(weights_(x)); }
    /// @brief prepares weight, nodal and mixed derivatives using the converged relative workspace
    auto linearization(const NodeType& x) const {
        const auto weights = weights_(x);
        if constexpr (internals::is_log_euclidean_spd_geometry<Geometry>)
            return p1_geodesic_linearization(geometry_, nodes_, weights);
        else
            return p1_geodesic_linearization(geometry_, nodes_, weights, options_);
    }
   private:
    template <typename> friend class internals::p1_interpolant_expr;
    /// @brief evaluates certified coefficients from stored barycentric coordinates
    Point evaluate(const Weights& weights) const {
        auto result = result_(weights);
        internals::require_converged(result);
        return std::move(result.value);
    }
    /// @brief maps a point on the simplex to its validated barycentric coordinates
    Weights weights_(const NodeType& x) const {
        fdapde_strong_assert(
          element_.contains(x) != Element::OUTSIDE, std::invalid_argument,
          "P1 interpolant spatial point must belong to the simplex");
        const auto coordinates = element_.barycentric_coords(x);
        Weights weights;
        for (int i = 0; i < NodeCount; ++i) weights[i] = coordinates[i];
        internals::validate_p1_data(NodeCount, weights);
        return weights;
    }
    /// @brief evaluates vertices and edges directly, using the Karcher solver for larger AIRM supports
    P1ValueResult<Point> result_(const Weights& weights) const {
        const auto vertex = internals::validate_p1_data(NodeCount, weights);
        if (vertex) return internals::p1_vertex_result(geometry_, nodes_, *vertex);
        int first = -1, second = -1, active = 0;
        for (int i = 0; i < NodeCount; ++i)
            if (weights[i] > 0) {
                if (active == 0) first = i;
                if (active == 1) second = i;
                ++active;
            }
        if (active == 2) {
            const auto& curve = edges_[second * (second - 1) / 2 + first];
            const double parameter = weights[second] / (weights[first] + weights[second]);
            return {
              Point(curve(parameter)),
              manifold::internals::normalize_karcher_weights(weights),
              0,
              manifold::BarycenterStopReason::closed_form,
              manifold::BarycenterUniqueness::globally_unique,
              0};
        }
        if constexpr (internals::is_log_euclidean_spd_geometry<Geometry>)
            return p1_geodesic_value(geometry_, nodes_, weights);
        else
            return p1_geodesic_value(geometry_, nodes_, weights, options_.mean);
    }
    Geometry geometry_;
    Element element_;
    Nodes nodes_;
    P1GeodesicLinearizationOptions options_;
    std::vector<Curve> edges_;
};
}   // namespace fdapde::gfe

namespace fdapde::manifold {
/// @brief prepares native SPD interpolation with an owning copy of the spatial simplex
template <typename Scalar_, int Order_, Usage Uses_>
template <typename Element, typename Nodes>
    requires(std::is_lvalue_reference_v<Nodes &&> || std::remove_cvref_t<Nodes>::NestAsRef == 0)
auto LogEuclideanSPDGeometry<Scalar_, Order_, Uses_>::interpolant(const Element& element, Nodes&& nodes) const {
    return interpolant(element, std::forward<Nodes>(nodes), gfe::P1GeodesicLinearizationOptions {});
}
/// @brief prepares native SPD interpolation with an owning copy of the spatial simplex
template <typename Scalar_, int Order_, Usage Uses_>
template <typename Element, typename Nodes>
    requires(std::is_lvalue_reference_v<Nodes &&> || std::remove_cvref_t<Nodes>::NestAsRef == 0)
auto LogEuclideanSPDGeometry<Scalar_, Order_, Uses_>::interpolant(
  const Element& element, Nodes&& nodes, const gfe::P1GeodesicLinearizationOptions& options) const {
    using Stored = std::conditional_t<
      std::remove_cvref_t<Nodes>::NestAsRef == 0, std::remove_cvref_t<Nodes>, const std::remove_cvref_t<Nodes>&>;
    using Cell = Simplex<Element::local_dim, Element::embed_dim>;
    return gfe::P1Interpolant<LogEuclideanSPDGeometry, Cell, Stored>(
      *this, Cell(element.nodes()), std::forward<Nodes>(nodes), options);
}
/// @brief prepares native SPD interpolation with an owning copy of the spatial simplex
template <typename Scalar_, int Order_, Usage Uses_>
template <typename Element, typename Nodes>
    requires(std::is_lvalue_reference_v<Nodes &&> || std::remove_cvref_t<Nodes>::NestAsRef == 0)
auto AffineInvariantSPDGeometry<Scalar_, Order_, Uses_>::interpolant(const Element& element, Nodes&& nodes) const {
    return interpolant(element, std::forward<Nodes>(nodes), gfe::P1GeodesicLinearizationOptions {});
}
/// @brief prepares native SPD interpolation with an owning copy of the spatial simplex
template <typename Scalar_, int Order_, Usage Uses_>
template <typename Element, typename Nodes>
    requires(std::is_lvalue_reference_v<Nodes &&> || std::remove_cvref_t<Nodes>::NestAsRef == 0)
auto AffineInvariantSPDGeometry<Scalar_, Order_, Uses_>::interpolant(
  const Element& element, Nodes&& nodes, const gfe::P1GeodesicLinearizationOptions& options) const {
    using Stored = std::conditional_t<
      std::remove_cvref_t<Nodes>::NestAsRef == 0, std::remove_cvref_t<Nodes>, const std::remove_cvref_t<Nodes>&>;
    using Cell = Simplex<Element::local_dim, Element::embed_dim>;
    return gfe::P1Interpolant<AffineInvariantSPDGeometry, Cell, Stored>(
      *this, Cell(element.nodes()), std::forward<Nodes>(nodes), options);
}
}   // namespace fdapde::manifold
#endif
