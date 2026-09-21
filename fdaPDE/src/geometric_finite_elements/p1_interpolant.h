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
#include <mutex>
#include <unordered_map>

#include "header_check.h"

namespace fdapde {
template <typename, typename> class GeometricFeEvaluation;
}

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
    auto linearization(const NodeType& x) const { return linearization_(weights_(x)); }
   private:
    template <typename> friend class internals::p1_interpolant_expr;
    template <typename, typename, typename> friend class P1FieldInterpolant;
    /// @brief shares derivative preparation between simplex coordinates and finite element shape values
    auto linearization_(const Weights& weights) const {
        if constexpr (internals::is_log_euclidean_spd_geometry<Geometry>)
            return p1_geodesic_linearization(geometry_, nodes_, weights);
        else
            return p1_geodesic_linearization(geometry_, nodes_, weights, options_);
    }
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
namespace internals {
/// @brief obtains the triangulation from a finite element space or a directly supplied mesh
template <typename Domain> const auto& p1_mesh(const Domain& domain) {
    if constexpr (requires { domain.triangulation(); })
        return domain.triangulation();
    else
        return domain;
}
/// @brief evaluates the existing scalar reference basis on an independently stored spatial cell
template <typename Space, typename Element, typename Point>
auto p1_shape_weights(const Space& space, const Element& element, const Point& x) {
    const auto mapped = (element.invJ() * (x - element.node(0))).eval();
    Matrix<double, Space::local_dim, 1> reference;
    for (int i = 0; i < Space::local_dim; ++i) reference[i] = mapped[i];
    std::array<double, Element::n_nodes> weights;
    for (int i = 0; i < Element::n_nodes; ++i) weights[i] = space.eval_shape_value(i, reference);
    validate_p1_data(Element::n_nodes, weights);
    return weights;
}
}   // namespace internals

/// @brief borrows an immutable mesh or finite element space and batch, retaining only visited P1 cells
/// @details expressions and linearizations borrow this immovable field; all bindings must outlive their use
template <typename Geometry, typename Domain, typename Values> class P1FieldInterpolant {
   public:
    using Mesh = std::remove_cvref_t<decltype(internals::p1_mesh(std::declval<const Domain&>()))>;
    using Element = Simplex<Mesh::local_dim, Mesh::embed_dim>;
    using NodeType = typename Element::NodeType;
    using Selection = decltype(std::declval<const std::remove_cvref_t<Values>&>().select(
      std::declval<std::array<int, Element::n_nodes>>()));
    using Local = P1Interpolant<Geometry, Element, Selection>;
    /// @brief builds an independent spatial index without preparing any nodal interpolation data
    P1FieldInterpolant(Geometry geometry, const Domain& domain, Values values, P1GeodesicLinearizationOptions options) :
        geometry_(std::move(geometry)),
        domain_(&domain),
        mesh_(checked_mesh_(domain, values)),
        values_(std::forward<Values>(values)),
        options_(options),
        locator_(mesh_) { }
    /// @brief locates a cell and returns its deferred expression with a stable borrowed cache binding
    auto operator()(const NodeType& x) const& {
        const auto& cell = cell_(x);
        return internals::p1_interpolant_expr<const Local&>(cell, weights_(cell, x));
    }
    /// @brief rejects expressions that would borrow an expiring field cache
    void operator()(const NodeType&) const&& = delete;
    /// @brief returns the local mean and convergence diagnostics at a spatial point
    auto result(const NodeType& x) const {
        const auto& cell = cell_(x);
        return cell.result_(weights_(cell, x));
    }
    /// @brief prepares derivatives in the located cell's local vertex order
    auto linearization(const NodeType& x) const& {
        const auto& cell = cell_(x);
        return cell.linearization_(weights_(cell, x));
    }
    /// @brief rejects derivative workspaces that could borrow an expiring field cache
    void linearization(const NodeType&) const&& = delete;
    /// @brief returns the number of visited cells whose edge curves have been prepared
    std::size_t prepared_cells() const {
        const std::lock_guard lock(mutex_);
        return cells_.size();
    }
    /// @brief discards coefficient-dependent cell data while retaining the spatial index
    /// @details invalidates existing expressions and linearizations; updates require exclusive access
    void clear_cache() {
        const std::lock_guard lock(mutex_);
        cells_.clear();
    }
   private:
    /// @brief validates the mesh and nodal DOF count before constructing the spatial index
    static const Mesh* checked_mesh_(const Domain& domain, const std::remove_cvref_t<Values>& values) {
        const auto& mesh = internals::p1_mesh(domain);
        fdapde_strong_assert(mesh.n_cells() > 0, std::invalid_argument, "P1 field requires a nonempty mesh");
        const int count = [&]() {
            if constexpr (requires { domain.n_dofs(); })
                return domain.n_dofs();
            else
                return mesh.n_nodes();
        }();
        fdapde_strong_assert(
          std::cmp_equal(values.size(), count), std::invalid_argument,
          "P1 field requires one value per nodal degree of freedom");
        return &mesh;
    }
    /// @brief evaluates finite element shape weights or the simplex barycentric map on a located cell
    typename Local::Weights weights_(const Local& cell, const NodeType& x) const {
        if constexpr (requires { domain_->dof_handler(); })
            return internals::p1_shape_weights(*domain_, cell.element_, x);
        else
            return cell.weights_(x);
    }
    /// @brief selects global DOFs in local vertex order and prepares each visited cell at most once
    const Local& cell_(const NodeType& x) const {
        fdapde_strong_assert(x.allFinite(), std::invalid_argument, "P1 field spatial point must be finite");
        const int id = locator_.locate(x);
        fdapde_strong_assert(id >= 0, std::invalid_argument, "P1 field spatial point must belong to the mesh");
        return cached_cell_(id, [&] {
            std::array<int, Element::n_nodes> ids;
            for (int i = 0; i < Element::n_nodes; ++i) {
                if constexpr (requires { domain_->dof_handler(); })
                    ids[i] = domain_->dof_handler().dofs()(id, i);
                else
                    ids[i] = mesh_->cells()(id, i);
            }
            const typename Mesh::CellType cell(id, mesh_);
            return Local(geometry_, Element(cell.nodes()), values_.select(ids), options_);
        });
    }
    template <typename, typename> friend class fdapde::GeometricFeEvaluation;
    /// @brief evaluates prepared spatial data using the current coefficient-dependent local cache
    typename Geometry::Point evaluate_prepared_(
      int id, const Element& element, const std::array<int, Element::n_nodes>& dofs,
      const typename Local::Weights& weights) const {
        const auto& cell = cached_cell_(id, [&] { return Local(geometry_, element, values_.select(dofs), options_); });
        return cell.evaluate(weights);
    }
    /// @brief synchronizes first cell preparation while leaving numerical evaluation outside the lock
    template <typename Prepare> const Local& cached_cell_(int id, Prepare prepare) const {
        // ponytail: one lock protects cell preparation, shard only if concurrent cache misses dominate
        const std::lock_guard lock(mutex_);
        if (const auto it = cells_.find(id); it != cells_.end()) return it->second;
        return cells_.emplace(id, prepare()).first->second;
    }
    Geometry geometry_;
    const Domain* domain_;
    const Mesh* mesh_;
    Values values_;
    P1GeodesicLinearizationOptions options_;
    TreeSearch<Mesh> locator_;
    mutable std::mutex mutex_;
    mutable std::unordered_map<int, Local> cells_;
};

namespace internals {
/// @brief dispatches both SPD geometries to the same simplex or mesh preparation path
template <typename Geometry, typename Element, typename Nodes>
auto make_p1_interpolant(
  Geometry geometry, Element&& element, Nodes&& nodes, const P1GeodesicLinearizationOptions& options) {
    using Spatial = std::remove_cvref_t<Element>;
    using Stored = std::conditional_t<
      std::remove_cvref_t<Nodes>::NestAsRef == 0, std::remove_cvref_t<Nodes>, const std::remove_cvref_t<Nodes>&>;
    if constexpr (requires { typename Spatial::CellType; })
        return P1FieldInterpolant<Geometry, Spatial, Stored>(
          std::move(geometry), element, std::forward<Nodes>(nodes), options);
    else {
        using Cell = Simplex<Spatial::local_dim, Spatial::embed_dim>;
        return P1Interpolant<Geometry, Cell, Stored>(
          std::move(geometry), Cell(element.nodes()), std::forward<Nodes>(nodes), options);
    }
}
}   // namespace internals
}   // namespace fdapde::gfe

namespace fdapde::manifold {
/// @brief delegates native SPD interpolation to the shared simplex or mesh preparation path
template <typename Scalar_, int Order_, Usage Uses_>
template <typename Element, typename Nodes>
    requires gfe::P1InterpolationBinding<Element, Nodes>
auto LogEuclideanSPDGeometry<Scalar_, Order_, Uses_>::interpolant(Element&& element, Nodes&& nodes) const {
    return interpolant(
      std::forward<Element>(element), std::forward<Nodes>(nodes), gfe::P1GeodesicLinearizationOptions {});
}
/// @brief delegates native SPD interpolation to the shared simplex or mesh preparation path
template <typename Scalar_, int Order_, Usage Uses_>
template <typename Element, typename Nodes>
    requires gfe::P1InterpolationBinding<Element, Nodes>
auto LogEuclideanSPDGeometry<Scalar_, Order_, Uses_>::interpolant(
  Element&& element, Nodes&& nodes, const gfe::P1GeodesicLinearizationOptions& options) const {
    return gfe::internals::make_p1_interpolant(
      *this, std::forward<Element>(element), std::forward<Nodes>(nodes), options);
}
/// @brief delegates native SPD interpolation to the shared simplex or mesh preparation path
template <typename Scalar_, int Order_, Usage Uses_>
template <typename Element, typename Nodes>
    requires gfe::P1InterpolationBinding<Element, Nodes>
auto AffineInvariantSPDGeometry<Scalar_, Order_, Uses_>::interpolant(Element&& element, Nodes&& nodes) const {
    return interpolant(
      std::forward<Element>(element), std::forward<Nodes>(nodes), gfe::P1GeodesicLinearizationOptions {});
}
/// @brief delegates native SPD interpolation to the shared simplex or mesh preparation path
template <typename Scalar_, int Order_, Usage Uses_>
template <typename Element, typename Nodes>
    requires gfe::P1InterpolationBinding<Element, Nodes>
auto AffineInvariantSPDGeometry<Scalar_, Order_, Uses_>::interpolant(
  Element&& element, Nodes&& nodes, const gfe::P1GeodesicLinearizationOptions& options) const {
    return gfe::internals::make_p1_interpolant(
      *this, std::forward<Element>(element), std::forward<Nodes>(nodes), options);
}
}   // namespace fdapde::manifold
#endif
