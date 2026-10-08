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

#ifndef __FDAPDE_EUCLIDEAN_GEOMETRY_H__
#define __FDAPDE_EUCLIDEAN_GEOMETRY_H__

#include "header_check.h"

namespace fdapde::manifold {

/// @brief supplies the Frobenius metric and affine steps for native dense or symmetric matrix owners
/// @details vectors use their ordinary Euclidean metric; symmetric off-diagonal entries count twice
template <typename Point_ = Vector<double, Dynamic>> class EuclideanGeometry {
   public:
    using Point = Point_;
    using Tangent = Point;
    using Scalar = typename Point::Scalar;
    fdapde_static_assert(
      (std::same_as<Point, Matrix<Scalar, Point::Rows, Point::Cols, Point::StorageOrder>> ||
       (NativeSymmetricLike<Point> && Point::NestAsRef == 1)),
      EUCLIDEAN_GEOMETRY_REQUIRES_A_NATIVE_DENSE_OR_SYMMETRIC_OWNER);
    fdapde_static_assert(std::is_floating_point_v<Scalar>, EUCLIDEAN_GEOMETRY_REQUIRES_A_FLOATING_POINT_SCALAR);

    /// @brief constructs the single-factor metric of a fixed native matrix shape
    EuclideanGeometry()
        requires(Point::Rows != Dynamic && Point::Cols != Dynamic)
        : EuclideanGeometry(Point::Rows, Point::Cols) { }
    /// @brief preserves the single-factor dynamic column-vector shape constructor
    explicit EuclideanGeometry(int rows)
        requires(Point::Rows == Dynamic && Point::Cols == 1)
        : EuclideanGeometry(rows, 1) { }
    /// @brief validates nonempty element dimensions against the native matrix shape
    EuclideanGeometry(int rows, int cols) : rows_(rows), cols_(cols) {
        fdapde_strong_assert(
          rows > 0 && cols > 0, std::invalid_argument, "EuclideanGeometry: dimensions must be positive");
        fdapde_strong_assert(
          (Point::Rows == Dynamic || Point::Rows == rows) && (Point::Cols == Dynamic || Point::Cols == cols),
          std::invalid_argument, "EuclideanGeometry: incompatible static dimensions");
        fdapde_strong_assert(
          std::int64_t(rows) * cols <= std::numeric_limits<int>::max(), std::length_error,
          "EuclideanGeometry: matrix size exceeds supported range");
        if constexpr (is_symmetric_matrix_v<Point>)
            fdapde_strong_assert(
              rows == cols, std::invalid_argument, "EuclideanGeometry: symmetric matrices must be square");
    }
    /// @brief returns the configured element row extent
    int rows() const { return rows_; }
    /// @brief returns the configured element column extent
    int cols() const { return cols_; }
    /// @brief counts independent entries of one element, preserving the packed symmetric dimension
    std::size_t dimension() const {
        if constexpr (is_symmetric_matrix_v<Point>)
            return std::size_t(rows_) * (std::size_t(rows_) + 1) / 2;
        else
            return std::size_t(rows_) * cols_;
    }
    /// @brief contracts full matrix entries so both mirrored symmetric coefficients contribute
    template <typename P, typename U, typename V> double inner_product(const P& point, const U& u, const V& v) const {
        check_matrix_(point);
        check_matrix_(u);
        check_matrix_(v);
        double value = 0;
        for (int i = 0; i < rows_; ++i)
            for (int j = 0; j < cols_; ++j) value += static_cast<double>(u(i, j)) * static_cast<double>(v(i, j));
        return value;
    }
    /// @brief measures a direction with the native scale-safe Frobenius norm
    template <typename P, typename U> double norm(const P& point, const U& u) const {
        check_matrix_(point);
        check_matrix_(u);
        return u.norm();
    }
    /// @brief copies an already admissible direction into independent owning storage
    Tangent project(const Point& point, const Tangent& u) const {
        check_matrix_(point);
        check_matrix_(u);
        return u;
    }
    /// @brief retains the Frobenius gradient because the Euclidean metric is constant
    template <typename P, typename U> Tangent euclidean_to_riemannian_gradient(const P& point, const U& u) const {
        check_matrix_(point);
        check_matrix_(u);
        return Tangent(u);
    }
    /// @brief retains the ambient Hessian action for the constant Euclidean connection
    template <typename P, typename G, typename H, typename U>
    Tangent
    euclidean_to_riemannian_hessian(const P& point, const G& gradient, const H& hessian, const U& direction) const {
        check_matrix_(point);
        check_matrix_(gradient);
        check_matrix_(hessian);
        check_matrix_(direction);
        return Tangent(hessian);
    }
    /// @brief creates a zero matrix with the geometry dimensions
    Tangent zero_tangent(const Point& point) const {
        check_matrix_(point);
        Tangent result;
        if constexpr (Point::Rows == Dynamic || Point::Cols == Dynamic) result.resize(rows_, cols_);
        return result;
    }
    /// @brief materializes a linear combination without retaining borrowed expressions
    Tangent linear_combination(const Point& point, double a, const Tangent& u, double b, const Tangent& v) const {
        check_matrix_(point);
        check_matrix_(u);
        check_matrix_(v);
        return Tangent(a * u + b * v);
    }
    /// @brief takes an affine step while preserving the native matrix structure
    Point retract(const Point& point, const Tangent& u, double step) const {
        return linear_combination(point, 1, point, step, u);
    }
   private:
    /// @brief checks solver operands and borrowed element views against the fixed shape in debug mode
    template <typename Xpr> void check_matrix_(const Xpr& matrix) const {
        fdapde_assert(
          matrix.rows() == rows_ && matrix.cols() == cols_, std::invalid_argument,
          "EuclideanGeometry: incompatible matrix dimensions");
    }
    int rows_, cols_;
};

}   // namespace fdapde::manifold

#endif   // __FDAPDE_EUCLIDEAN_GEOMETRY_H__
