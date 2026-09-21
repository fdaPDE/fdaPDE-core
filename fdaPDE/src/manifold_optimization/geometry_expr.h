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

#ifndef __FDAPDE_MANIFOLD_GEOMETRY_EXPR_H__
#define __FDAPDE_MANIFOLD_GEOMETRY_EXPR_H__

#include "header_check.h"

namespace fdapde {

/// @brief integrates deferred geometric operations with whole-matrix native materialization
/// @details expressions have no coefficient evaluator and acquire verified SPD status only after evaluation
template <typename Derived> struct GeometryExpr : public MatrixExpr<Derived> {
    using MatrixExpr<Derived>::derived;
    /// @brief evaluates a complete matrix for native dense construction or assignment
    auto eval_matrix() const { return derived().template eval<Cache::None>(); }
};

namespace internals {

/// @brief binds a parameter by value to borrowed or owned prepared geodesic data
template <typename Curve> class spd_geodesic_expr : public GeometryExpr<spd_geodesic_expr<Curve>> {
   public:
    using CurveType = std::remove_cvref_t<Curve>;
    using Scalar = typename CurveType::Scalar;
    static constexpr int Rows = CurveType::Rows;
    static constexpr int Cols = Rows;
    static constexpr int StorageOrder = RowMajor;
    static constexpr int NestAsRef = 0;
    static constexpr int ReadOnly = 1;
    using assignment_executor = deleted_assignment_executor;
    /// @brief retains the prepared curve and parameter without evaluating a matrix
    spd_geodesic_expr(Curve curve, double parameter) : curve_(std::forward<Curve>(curve)), parameter_(parameter) { }
    /// @brief returns the prepared matrix order
    int rows() const { return curve_.rows(); }
    /// @brief returns the prepared matrix order
    int cols() const { return rows(); }
    /// @brief reconstructs symmetric coefficients without certifying numerical positive definiteness
    auto eval_matrix() const { return curve_.eval(parameter_); }
    /// @brief certifies the same reconstructed coefficients using the SPD destination cache policy
    template <typename Policy> auto eval() const { return SPDMatrix<Scalar, Rows, Cols, Policy>(eval_matrix()); }
   private:
    Curve curve_;
    double parameter_;
};

/// @brief owns a snapshot of the factors or logarithmic charts defining an SPD geodesic
/// @details affine-invariant curves store F and log(lambda); log-Euclidean curves store log(A) and log(B)-log(A)
template <typename Scalar_, int Order_, bool AffineInvariant> class spd_geodesic {
   public:
    using Scalar = Scalar_;
    static constexpr int Rows = Order_;
    using Symmetric = SymmetricMatrix<Scalar, Rows, Rows>;
    using First = std::conditional_t<AffineInvariant, Matrix<Scalar, Rows, Rows>, Symmetric>;
    using Second = std::conditional_t<AffineInvariant, Vector<Scalar, Rows>, Symmetric>;
    /// @brief owns prepared data independently of endpoint and geometry lifetimes
    spd_geodesic(First first, Second second) : first_(std::move(first)), second_(std::move(second)) { }
    /// @brief returns the fixed or runtime order of the prepared data
    int rows() const { return first_.rows(); }
    /// @brief borrows a persistent curve and stores the parameter in a deferred matrix expression
    auto operator()(double parameter) const& { return spd_geodesic_expr<const spd_geodesic&>(*this, parameter); }
    /// @brief keeps a temporary prepared curve alive inside its deferred matrix expression
    auto operator()(double parameter) && { return spd_geodesic_expr<spd_geodesic>(std::move(*this), parameter); }
    /// @brief prevents borrowing a const temporary prepared curve
    void operator()(double) const&& = delete;
    /// @brief reconstructs finite symmetric coefficients once without an SPD certification
    auto eval(double parameter) const {
        const Scalar t = static_cast<Scalar>(parameter);
        fdapde_strong_assert(std::isfinite(t), std::invalid_argument, "geodesic: parameter must be finite");
        Symmetric value;
        if constexpr (Rows == Dynamic) value.resize(rows(), rows());
        if constexpr (AffineInvariant) {
            Vector<Scalar, Rows> powers;
            if constexpr (Rows == Dynamic) powers.resize(rows());
            for (int k = 0; k < rows(); ++k) {
                powers[k] = std::exp(t * second_[k]);
                fdapde_strong_assert(
                  std::isfinite(powers[k]), std::domain_error, "geodesic: unrepresentable spectral power");
            }
            for (int i = 0; i < rows(); ++i)
                for (int j = 0; j <= i; ++j) {
                    Scalar coefficient = 0;
                    for (int k = 0; k < rows(); ++k) coefficient += first_(i, k) * powers[k] * first_(j, k);
                    fdapde_strong_assert(
                      std::isfinite(coefficient), std::domain_error, "geodesic: nonfinite reconstruction");
                    value(i, j) = coefficient;
                }
            return value;
        } else {
            for (int i = 0; i < rows(); ++i)
                for (int j = 0; j <= i; ++j) value(i, j) = first_(i, j) + t * second_(i, j);
            validate_finite_symmetric(value);
            const EVD<Symmetric> evd(value);
            require_computed(evd);
            const auto powers = transform_eigenvalues(evd, rows(), [](auto x) { return std::exp(x); });
            return reconstruct_symmetric(evd, rows(), powers);
        }
    }
   private:
    First first_;
    Second second_;
};

/// @brief stores an unevaluated log-Euclidean combination with borrowed geometry and safely nested operands
template <typename Geometry, typename Points, typename Weights>
class log_euclidean_mean_expr : public GeometryExpr<log_euclidean_mean_expr<Geometry, Points, Weights>> {
   public:
    using Scalar = typename Geometry::Scalar;
    static constexpr int Rows = Geometry::Point::Rows;
    static constexpr int Cols = Geometry::Point::Cols;
    static constexpr int StorageOrder = RowMajor;
    static constexpr int NestAsRef = 0;
    static constexpr int ReadOnly = 1;
    using assignment_executor = deleted_assignment_executor;
    /// @brief retains references or temporary expression nodes without computing logarithms or coefficients
    log_euclidean_mean_expr(const Geometry& geometry, Points points, Weights weights) :
        geometry_(geometry), points_(std::forward<Points>(points)), weights_(std::forward<Weights>(weights)) { }
    /// @brief reports the result order through the current geometry
    int rows() const { return geometry_.order(); }
    /// @brief reports the result order through the current geometry
    int cols() const { return geometry_.order(); }
    /// @brief evaluates one complete combination and certifies its rounded result with the destination cache policy
    template <typename Policy> auto eval() const {
        fdapde_strong_assert(
          points_.rows() == rows() && points_.cols() == cols(), std::invalid_argument,
          "weighted_mean: points and geometry have incompatible orders");
        fdapde_strong_assert(
          (weights_.rows() == 1 || weights_.cols() == 1) && std::cmp_equal(weights_.size(), points_.size()),
          std::invalid_argument, "weighted_mean: expected one weight per point");
        SymmetricMatrix<Scalar, Rows, Cols> sum;
        if constexpr (Rows == Dynamic) sum.resize(rows(), cols());
        for (int i = 0; i < rows(); ++i)
            for (int j = 0; j <= i; ++j) sum(i, j) = Scalar(0);
        bool has_nonzero_weight = false;
        for (std::size_t k = 0; k < points_.size(); ++k) {
            const auto weight = weights_[static_cast<int>(k)];
            static_assert(
              std::is_arithmetic_v<std::remove_cvref_t<decltype(weight)>>,
              "weighted_mean requires real arithmetic weights");
            const Scalar scalar = static_cast<Scalar>(weight);
            fdapde_strong_assert(std::isfinite(scalar), std::invalid_argument, "weighted_mean: weights must be finite");
            if (scalar == Scalar(0)) continue;
            has_nonzero_weight = true;
            const auto point = points_[k];
            fdapde_strong_assert(
              point.rows() == rows() && point.cols() == cols(), std::invalid_argument,
              "weighted_mean: nonuniform point dimensions");
            const auto logarithm = matrix_log(point);
            for (int i = 0; i < rows(); ++i)
                for (int j = 0; j <= i; ++j) {
                    sum(i, j) = Scalar(sum(i, j)) + scalar * logarithm(i, j);
                    fdapde_strong_assert(
                      std::isfinite(Scalar(sum(i, j))), std::domain_error, "weighted_mean: nonfinite log combination");
                }
        }
        if (!has_nonzero_weight) return SPDMatrix<Scalar, Rows, Cols, Policy>::Identity(rows());
        return matrix_exp<Policy>(sum);
    }
   private:
    const Geometry& geometry_;
    Points points_;
    Weights weights_;
};

}   // namespace internals
}   // namespace fdapde
#endif   // __FDAPDE_MANIFOLD_GEOMETRY_EXPR_H__
