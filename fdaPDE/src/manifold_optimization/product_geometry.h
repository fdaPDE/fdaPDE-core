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

#ifndef __FDAPDE_PRODUCT_GEOMETRY_H__
#define __FDAPDE_PRODUCT_GEOMETRY_H__

#include "header_check.h"

namespace fdapde::manifold {

/// @brief lifts one native matrix geometry to a product of uniformly shaped matrix owners
/// @details the problem retains a single joint cost, gradient and Hessian across all batch elements
template <FirstOrderGeometry Geometry> class ProductGeometry {
   public:
    using MatrixType = point_t<Geometry>;
    using TangentType = tangent_t<Geometry>;
    using Point = MatrixBatch<MatrixType>;
    using Tangent = MatrixBatch<TangentType>;
    using Scalar = typename MatrixType::Scalar;

    /// @brief owns the element metric and validates a positive count with its fixed or dynamic shape
    ProductGeometry(Geometry geometry, std::size_t count) :
        geometry_(std::move(geometry)), count_(count), rows_(matrix_rows_(geometry_)), cols_(matrix_cols_(geometry_)) {
        fdapde_strong_assert(count > 0, std::invalid_argument, "batch geometry: count must be positive");
        fdapde_strong_assert(
          rows_ > 0 && cols_ > 0, std::invalid_argument, "batch geometry: element dimensions must be positive");
        fdapde_strong_assert(
          geometry_.dimension() > 0 && count <= std::numeric_limits<std::size_t>::max() / geometry_.dimension(),
          std::length_error, "batch geometry: dimension exceeds supported range");
    }
    /// @brief validates the initial batch count and shape before objective evaluation
    void validate_point(const Point& point) const { check_batch_(point); }
    /// @brief counts independent coordinates across every matrix in the product
    std::size_t dimension() const { return count_ * geometry_.dimension(); }
    /// @brief sums element metric contractions in batch order
    double inner_product(const Point& point, const Tangent& u, const Tangent& v) const {
        check_batch_(point);
        check_batch_(u);
        check_batch_(v);
        double value = 0;
        for (std::size_t i = 0; i < count_; ++i) value += geometry_.inner_product(point[i], u[i], v[i]);
        return value;
    }
    /// @brief combines element metric norms without squaring large finite values
    double norm(const Point& point, const Tangent& u) const {
        check_batch_(point);
        check_batch_(u);
        double value = 0;
        for (std::size_t i = 0; i < count_; ++i) value = std::hypot(value, geometry_.norm(point[i], u[i]));
        return value;
    }
    /// @brief projects each ambient matrix into the corresponding tangent space
    Tangent project(const Point& point, const Tangent& u) const {
        check_batch_(point);
        check_batch_(u);
        Tangent result(count_, rows_, cols_);
        for (std::size_t i = 0; i < count_; ++i) result[i] = geometry_.project(point[i], u[i]);
        return result;
    }
    /// @brief allocates an independent zero tangent batch of the configured shape
    Tangent zero_tangent(const Point& point) const {
        check_batch_(point);
        return Tangent(count_, rows_, cols_);
    }
    /// @brief materializes elementwise linear combinations into an owning tangent batch
    Tangent linear_combination(const Point& point, double a, const Tangent& u, double b, const Tangent& v) const {
        check_batch_(point);
        check_batch_(u);
        check_batch_(v);
        Tangent result(count_, rows_, cols_);
        for (std::size_t i = 0; i < count_; ++i) result[i] = geometry_.linear_combination(point[i], a, u[i], b, v[i]);
        return result;
    }
    /// @brief takes one joint retraction step while preserving each element's native point type
    Point retract(const Point& point, const Tangent& u, double step) const {
        check_batch_(point);
        check_batch_(u);
        Point result(count_, rows_, cols_);
        for (std::size_t i = 0; i < count_; ++i) result[i] = geometry_.retract(point[i], u[i], step);
        return result;
    }
    /// @brief follows the product geodesic with the same finite parameter on every factor
    Point exponential(const Point& point, const Tangent& u, double step = 1) const
        requires GeodesicGeometry<Geometry>
    {
        check_batch_(point);
        check_batch_(u);
        Point result(count_, rows_, cols_);
        for (std::size_t i = 0; i < count_; ++i) result[i] = geometry_.exponential(point[i], u[i], step);
        return result;
    }
    /// @brief collects the initial geodesic velocities between corresponding batch elements
    Tangent logarithm(const Point& from, const Point& to) const
        requires GeodesicGeometry<Geometry>
    {
        check_batch_(from);
        check_batch_(to);
        Tangent result(count_, rows_, cols_);
        for (std::size_t i = 0; i < count_; ++i) result[i] = geometry_.logarithm(from[i], to[i]);
        return result;
    }
    /// @brief combines factor distances with the product metric's scale-safe norm
    double distance(const Point& from, const Point& to) const
        requires GeodesicGeometry<Geometry>
    {
        check_batch_(from);
        check_batch_(to);
        double value = 0;
        for (std::size_t i = 0; i < count_; ++i) value = std::hypot(value, geometry_.distance(from[i], to[i]));
        return value;
    }
    /// @brief transports every direction along its corresponding source-to-target geodesic
    Tangent transport(const Point& from, const Point& to, const Tangent& u) const
        requires VectorTransportGeometry<Geometry>
    {
        check_batch_(from);
        check_batch_(to);
        check_batch_(u);
        Tangent result(count_, rows_, cols_);
        for (std::size_t i = 0; i < count_; ++i) result[i] = geometry_.transport(from[i], to[i], u[i]);
        return result;
    }
    /// @brief converts each ambient gradient to its element Riemannian metric dual
    template <typename Ambient>
    Tangent euclidean_to_riemannian_gradient(const Point& point, const MatrixBatch<Ambient>& u) const
        requires requires(const Geometry& geometry, const MatrixType& matrix, const Ambient& gradient) {
            { geometry.euclidean_to_riemannian_gradient(matrix, gradient) } -> std::same_as<TangentType>;
        }
    {
        check_batch_(point);
        check_batch_(u);
        Tangent result(count_, rows_, cols_);
        for (std::size_t i = 0; i < count_; ++i) result[i] = geometry_.euclidean_to_riemannian_gradient(point[i], u[i]);
        return result;
    }
    /// @brief converts full objective Hessian components while retaining all cross-factor coupling
    template <typename Gradient, typename Hessian>
    Tangent euclidean_to_riemannian_hessian(
      const Point& point, const MatrixBatch<Gradient>& gradient, const MatrixBatch<Hessian>& hessian,
      const Tangent& direction) const
        requires requires(
          const Geometry& geometry, const MatrixType& matrix, const Gradient& g, const Hessian& h,
          const TangentType& u) {
            { geometry.euclidean_to_riemannian_hessian(matrix, g, h, u) } -> std::same_as<TangentType>;
        }
    {
        check_batch_(point);
        check_batch_(gradient);
        check_batch_(hessian);
        check_batch_(direction);
        Tangent result(count_, rows_, cols_);
        for (std::size_t i = 0; i < count_; ++i)
            result[i] = geometry_.euclidean_to_riemannian_hessian(point[i], gradient[i], hessian[i], direction[i]);
        return result;
    }
    /// @brief materializes each factor's ambient velocity when tangents use different coordinates
    auto to_ambient(const Point& point, const Tangent& direction) const
        requires requires(const Geometry& geometry, const MatrixType& matrix, const TangentType& u) {
            geometry.to_ambient(matrix, u);
        }
    {
        check_batch_(point);
        check_batch_(direction);
        using Ambient = std::remove_cvref_t<decltype(geometry_.to_ambient(
          std::declval<const MatrixType&>(), std::declval<const TangentType&>()))>;
        MatrixBatch<Ambient> result(count_, rows_, cols_);
        for (std::size_t i = 0; i < count_; ++i) result[i] = geometry_.to_ambient(point[i], direction[i]);
        return result;
    }
   private:
    /// @brief rejects mismatched batch operands before element access
    template <typename Batch> void check_batch_(const Batch& batch) const {
        fdapde_strong_assert(batch.size() == count_, std::invalid_argument, "batch geometry: incompatible count");
        fdapde_strong_assert(
          batch.rows() == rows_ && batch.cols() == cols_, std::invalid_argument,
          "batch geometry: incompatible element dimensions");
    }
    /// @brief obtains the element row extent from its metric or fixed native type
    static int matrix_rows_(const Geometry& geometry) {
        if constexpr (requires { geometry.order(); })
            return geometry.order();
        else if constexpr (requires { geometry.rows(); })
            return geometry.rows();
        else
            return MatrixType::Rows;
    }
    /// @brief obtains the element column extent from its metric or fixed native type
    static int matrix_cols_(const Geometry& geometry) {
        if constexpr (requires { geometry.order(); })
            return geometry.order();
        else if constexpr (requires { geometry.cols(); })
            return geometry.cols();
        else
            return MatrixType::Cols;
    }
    Geometry geometry_;
    std::size_t count_;
    int rows_, cols_;
};

}   // namespace fdapde::manifold

#endif   // __FDAPDE_PRODUCT_GEOMETRY_H__
