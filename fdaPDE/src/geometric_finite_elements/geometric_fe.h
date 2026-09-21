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
    static constexpr int local_dim = Triangulation::local_dim;
    static constexpr int embed_dim = Triangulation::embed_dim;
    /// @brief enumerates scalar nodal DOFs and owns a copy of the target geometry
    GeometricFeSpace(Triangulation_& mesh, FeType_ element, Geometry_ geometry) :
        space_(mesh, element), geometry_(std::move(geometry)) { }
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
   private:
    ScalarSpace space_;
    Geometry geometry_;
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
   private:
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
}   // namespace fdapde
#endif
