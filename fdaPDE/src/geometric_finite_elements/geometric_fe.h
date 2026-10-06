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

#ifndef __FDAPDE_GEOMETRIC_FE_H__
#define __FDAPDE_GEOMETRIC_FE_H__
#include "header_check.h"

namespace fdapde {
template <typename, typename> class FeSpace;
template <int, int> struct FeP;

namespace gfe::internals {
/// @brief limits geometric point loops to the policies supplied by the execution module
template <typename Policy>
concept PointExecutionPolicy = std::same_as<Policy, execution_seq_t> || std::same_as<Policy, execution_par_t>;
/// @brief runs independent point work and rethrows the lowest-index failure after parallel workers complete
template <PointExecutionPolicy Policy, typename Body> void for_each_point(std::size_t count, Policy, Body body) {
    fdapde_assert(
      count <= static_cast<std::size_t>(std::numeric_limits<int>::max()), std::length_error,
      "geometric evaluation point count exceeds the execution index range");
    if constexpr (std::same_as<Policy, execution_seq_t>) {
        for (std::size_t i = 0; i < count; ++i) body(i);
    } else {
        if (count == 0) return;
        std::exception_ptr failure;
        std::size_t first_failure = count;
        std::mutex mutex;
        parallel_for(0, static_cast<int>(count), [&](int i) {
            try {
                body(static_cast<std::size_t>(i));
            } catch (...) {
                const std::lock_guard lock(mutex);
                if (static_cast<std::size_t>(i) < first_failure) {
                    first_failure = i;
                    failure = std::current_exception();
                }
            }
        });
        if (failure) std::rethrow_exception(failure);
    }
}
}   // namespace gfe::internals

/// @brief pairs an existing scalar P1 finite element space with its target matrix geometry
/// @details include finite_elements.h for spatial types; the immutable mesh must outlive this space
template <typename Triangulation_, typename FeType_, typename Geometry_>
    requires std::same_as<std::remove_cvref_t<FeType_>, FeP<1, 1>>
class GeometricFeSpace {
   public:
    using Triangulation = std::remove_cvref_t<Triangulation_>;
    using FeType = std::remove_cvref_t<FeType_>;
    using Geometry = std::remove_cvref_t<Geometry_>;
    using ScalarSpace = FeSpace<Triangulation, FeType>;
    using DofHandlerType = typename ScalarSpace::DofHandlerType;
    using cell_dof_descriptor = typename ScalarSpace::cell_dof_descriptor;
    static constexpr int n_components = 1;
    static constexpr int local_dim = Triangulation::local_dim;
    static constexpr int embed_dim = Triangulation::embed_dim;
    /// @brief enumerates scalar nodal DOFs and owns a copy of the target geometry
    GeometricFeSpace(Triangulation_& mesh, FeType_ element, Geometry_ geometry) :
        space_(mesh, element), geometry_(std::move(geometry)) { }
    /// @brief owns positive continuous P1 rho coefficients in the scalar nodal DOF ordering
    GeometricFeSpace(Triangulation_& mesh, FeType_ element, Geometry_ geometry, std::span<const double> rho)
        requires gfe::internals::is_cheeger_geometry<Geometry>
        : space_(mesh, element), geometry_(std::move(geometry)), rho_(rho.begin(), rho.end()) {
        fdapde_strong_assert(
          std::cmp_equal(rho_.size(), n_dofs()), std::invalid_argument,
          "rho coefficient count must match the scalar space DOFs");
        for (double value : rho_) geometry_.with_rho(value);
    }
    /// @brief rejects a rho field whose space would borrow an expiring mesh
    GeometricFeSpace(const Triangulation&&, FeType_, Geometry_, std::span<const double>) = delete;
    /// @brief exposes immutable nodal rho coefficients, with empty storage denoting the constant geometry value
    std::span<const double> rho_coefficients() const& { return rho_; }
    /// @brief prevents coefficient views from escaping an expiring space
    void rho_coefficients() const&& = delete;
    /// @brief rejects temporary meshes even when the space template explicitly names a const triangulation
    GeometricFeSpace(const Triangulation&&, FeType_, Geometry_) = delete;
    /// @brief preserves the address of the space borrowed by geometric functions
    GeometricFeSpace(const GeometricFeSpace&) = delete;
    /// @brief prevents replacing a space while functions refer to its DOFs and geometry
    GeometricFeSpace& operator=(const GeometricFeSpace&) = delete;
    /// @brief preserves the scalar DOF handler's internal bindings and external function references
    GeometricFeSpace(GeometricFeSpace&&) = delete;
    /// @brief prevents transferring a space while functions retain its address
    GeometricFeSpace& operator=(GeometricFeSpace&&) = delete;
    /// @brief borrows the unchanged spatial triangulation
    const Triangulation& triangulation() const { return space_.triangulation(); }
    /// @brief exposes the scalar nodal enumeration without allowing its mutation
    const DofHandlerType& dof_handler() const { return space_.dof_handler(); }
    /// @brief returns the number of geometric nodal coefficients
    int n_dofs() const { return space_.n_dofs(); }
    /// @brief evaluates a scalar P1 weight through the existing reference basis
    template <typename Input> double eval_shape_value(int i, const Input& reference_point) const {
        return space_.eval_shape_value(i, reference_point);
    }
    /// @brief borrows the target geometry shared by geometric functions on this space
    const Geometry& geometry() const { return geometry_; }
    /// @brief owns location data and prepares reusable scalar shape weights and cell-local DOFs
    template <typename Location, gfe::internals::PointExecutionPolicy Policy = execution_seq_t>
    auto prepare_evaluation(MatrixBatch<Location> locations, Policy policy = execution_seq) const& {
        return GeometricFeEvaluation<GeometricFeSpace, Location>(*this, std::move(locations), policy);
    }
    /// @brief prevents a prepared evaluation from borrowing an expiring space
    template <typename Location, gfe::internals::PointExecutionPolicy Policy = execution_seq_t>
    void prepare_evaluation(MatrixBatch<Location>, Policy = execution_seq) const&& = delete;
   private:
    ScalarSpace space_;
    Geometry geometry_;
    std::vector<double> rho_;
};

/// @brief owns matrix coefficients and their prepared P1 cells while borrowing an immutable geometric space
/// @details the space must outlive the function; coefficient replacement invalidates expressions and linearizations
template <typename Space_, typename MatrixType_> class GeometricFeFunction {
   public:
    using Space = std::remove_cvref_t<Space_>;
    using Geometry = typename Space::Geometry;
    using Coefficients = MatrixBatch<MatrixType_>;
    using Interpolant = gfe::P1FieldInterpolant<Geometry, Space, const Coefficients&>;
    using InputType = typename Interpolant::NodeType;
    /// @brief copies an lvalue batch or transfers an rvalue, then prepares only the spatial index
    GeometricFeFunction(
      Space_& space, Coefficients coefficients, const gfe::P1GeodesicLinearizationOptions& options = {}) :
        space_(&space),
        coefficients_(checked_coeff_(space, std::move(coefficients))),
        interpolant_(space.geometry(), space, coefficients_, options) { }
    /// @brief rejects temporary spaces even when the function template explicitly names a const space
    GeometricFeFunction(const Space&&, Coefficients, const gfe::P1GeodesicLinearizationOptions& = {}) = delete;
    /// @brief borrows the immutable function space
    const Space& function_space() const { return *space_; }
    /// @brief borrows the owned coefficients without permitting changes that bypass cache invalidation
    const Coefficients& coeff() const& { return coefficients_; }
    /// @brief prevents coefficient references from escaping a temporary function
    void coeff() const&& = delete;
    /// @brief validates replacement data before discarding prepared cells and swapping the owned batch
    /// @details updates and destruction require exclusive access to the function and its borrowed evaluations
    void set_coeff(Coefficients coefficients) {
        validate_coeff_(*space_, coefficients);
        interpolant_.clear_cache();
        coefficients_.swap(coefficients);
    }
    /// @brief replaces this field's rho coefficients while preserving nodal matrix caches and spatial plans
    /// @details an empty vector restores the geometry's constant rho; updates require exclusive access
    void set_rho(std::vector<double> rho)
        requires gfe::internals::is_cheeger_geometry<Geometry>
    {
        interpolant_.set_rho(std::move(rho));
    }
    /// @brief borrows this field's active rho coefficients in scalar DOF order
    std::span<const double> rho_coefficients() const&
        requires gfe::internals::is_cheeger_geometry<Geometry>
    {
        return interpolant_.rho_coefficients();
    }
    /// @brief rejects scalar references escaping a temporary field
    void rho_coefficients() const&& = delete;
    /// @brief locates a cell and defers its native geometric interpolation using scalar shape weights
    auto operator()(const InputType& x) const& { return interpolant_(x); }
    /// @brief rejects expressions that would borrow an expiring coefficient owner
    void operator()(const InputType&) const&& = delete;
    /// @brief exposes the local value and mean convergence diagnostics
    auto result(const InputType& x) const { return interpolant_.result(x); }
    /// @brief prepares derivatives in the located cell's local DOF order
    auto linearization(const InputType& x) const& { return interpolant_.linearization(x); }
    /// @brief rejects derivatives that could outlive their coefficient owner
    void linearization(const InputType&) const&& = delete;
    /// @brief reports the number of prepared cells for the current coefficients
    std::size_t prepared_cells() const { return interpolant_.prepared_cells(); }
    /// @brief prepares and evaluates locations sequentially through the reusable multipoint engine
    template <typename Location> auto eval_at(MatrixBatch<Location> locations) const {
        return space_->prepare_evaluation(std::move(locations))(*this);
    }
   private:
    template <typename, typename> friend class GeometricFeEvaluation;
    /// @brief checks the nodal count and matrix order without evaluating any cell
    static void validate_coeff_(const Space& space, const Coefficients& coefficients) {
        fdapde_strong_assert(
          std::cmp_equal(coefficients.size(), space.n_dofs()), std::invalid_argument,
          "GeometricFeFunction coefficient count must match the space DOFs");
        fdapde_strong_assert(
          coefficients.rows() == space.geometry().order() && coefficients.cols() == space.geometry().order(),
          std::invalid_argument, "GeometricFeFunction coefficient shape must match the target geometry");
    }
    /// @brief validates construction data before transferring its complete coefficient and cache buffers
    static Coefficients checked_coeff_(const Space& space, Coefficients coefficients) {
        validate_coeff_(space, coefficients);
        return coefficients;
    }
    const Space* space_;
    Coefficients coefficients_;
    Interpolant interpolant_;
};
/// @brief owns locations and spatial P1 preparation independently of any function's coefficients
/// @details the immutable space must outlive this object; copying or moving retains that space binding
template <typename Space, typename Location> class GeometricFeEvaluation {
   public:
    using Geometry = typename Space::Geometry;
    using Element = Simplex<Space::local_dim, Space::embed_dim>;
    using Weights = std::array<double, Element::n_nodes>;
    using Locations = MatrixBatch<Location>;
    using Values = MatrixBatch<typename Geometry::Point>;
    /// @brief copies or transfers locations and prepares each point with the selected execution policy
    template <gfe::internals::PointExecutionPolicy Policy = execution_seq_t>
    GeometricFeEvaluation(const Space& space, Locations locations, Policy policy = execution_seq) :
        space_(&space), locations_(std::move(locations)) {
        fdapde_strong_assert(
          locations_.rows() == Space::embed_dim && locations_.cols() == 1, std::invalid_argument,
          "geometric evaluation locations must be embedding-coordinate column vectors");
        fdapde_strong_assert(
          locations_.size() <= static_cast<std::size_t>(std::numeric_limits<int>::max()), std::length_error,
          "geometric evaluation point count exceeds the execution index range");
        points_.resize(locations_.size());
        if (points_.empty()) return;
        const TreeSearch<typename Space::Triangulation> locator(&space.triangulation());
        cells_.reserve(std::min(locations_.size(), static_cast<std::size_t>(space.triangulation().n_cells())));
        std::mutex mutex;
        gfe::internals::for_each_point(locations_.size(), policy, [&](std::size_t i) {
            typename Element::NodeType x;
            const auto location = std::as_const(locations_)[i];
            for (int d = 0; d < Space::embed_dim; ++d) x[d] = location(d, 0);
            fdapde_strong_assert(x.allFinite(), std::invalid_argument, "P1 field spatial point must be finite");
            const int id = locator.locate(x);
            fdapde_strong_assert(id >= 0, std::invalid_argument, "P1 field spatial point must belong to the mesh");
            const Cell& cell = [&]() -> const Cell& {
                const std::lock_guard lock(mutex);
                return cells_.try_emplace(id, space, id).first->second;
            }();
            points_[i] = {id, gfe::internals::p1_shape_weights(space, cell.element, x)};
        });
    }
    /// @brief rejects an expiring space even for direct construction of the preparation object
    template <gfe::internals::PointExecutionPolicy Policy = execution_seq_t>
    GeometricFeEvaluation(const Space&&, Locations, Policy = execution_seq) = delete;
    /// @brief copies owned locations and spatial metadata while retaining the same borrowed space
    GeometricFeEvaluation(const GeometricFeEvaluation&) = default;
    /// @brief transfers the complete spatial preparation without rebinding its space
    GeometricFeEvaluation(GeometricFeEvaluation&&) noexcept = default;
    /// @brief replaces a preparation only after a complete copy or transfer has succeeded
    GeometricFeEvaluation& operator=(GeometricFeEvaluation other) & noexcept {
        std::swap(space_, other.space_);
        locations_.swap(other.locations_);
        cells_.swap(other.cells_);
        points_.swap(other.points_);
        return *this;
    }
    /// @brief returns the number of retained locations in output order
    std::size_t size() const { return locations_.size(); }
    /// @brief returns the number of spatial cells shared by the prepared points
    std::size_t prepared_cells() const { return cells_.size(); }
    /// @brief borrows the retained native location batch without allowing stale spatial preparation
    const Locations& locations() const& { return locations_; }
    /// @brief prevents location views from escaping an expiring preparation object
    void locations() const&& = delete;
    /// @brief evaluates current coefficients into independent native matrix batch slots in location order
    template <
      typename FunctionSpace, typename MatrixType, gfe::internals::PointExecutionPolicy Policy = execution_seq_t>
        requires std::same_as<std::remove_cvref_t<FunctionSpace>, Space>
    Values
    operator()(const GeometricFeFunction<FunctionSpace, MatrixType>& function, Policy policy = execution_seq) const {
        fdapde_strong_assert(
          &function.function_space() == space_, std::invalid_argument,
          "prepared geometric evaluation requires a function of the same space");
        Values values(size(), space_->geometry().order(), space_->geometry().order());
        gfe::internals::for_each_point(size(), policy, [&](std::size_t i) {
            const auto& point = points_[i];
            const auto& cell = cells_.at(point.cell);
            if constexpr (gfe::internals::is_flat_spd_geometry<Geometry>) {
                auto result =
                  gfe::p1_geodesic_value(space_->geometry(), function.coeff().select(cell.dofs), point.weights);
                gfe::internals::require_converged(result);
                values[i] = result.value;
            } else {
                values[i] =
                  function.interpolant_.evaluate_prepared_(point.cell, cell.element, cell.dofs, point.weights);
            }
        });
        return values;
    }
   private:
    /// @brief shares immutable spatial geometry and local DOF numbering between points in one cell
    struct Cell {
        Element element;
        std::array<int, Element::n_nodes> dofs;
        /// @brief snapshots cell geometry and scalar DOFs without reading any matrix coefficients
        Cell(const Space& space, int id) : element(space.dof_handler().cell(id).nodes()) {
            for (int i = 0; i < Element::n_nodes; ++i) dofs[i] = space.dof_handler().dofs()(id, i);
        }
    };
    /// @brief retains a cell id and scalar shape weights for one location
    struct PointData {
        int cell;
        Weights weights;
    };
    const Space* space_;
    Locations locations_;
    std::unordered_map<int, Cell> cells_;
    std::vector<PointData> points_;
};
}   // namespace fdapde
#endif
