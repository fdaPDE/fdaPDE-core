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

#ifndef __FDAPDE_MANIFOLD_SO_H__
#define __FDAPDE_MANIFOLD_SO_H__
#include "geometry_expr.h"
#include "header_check.h"
#include "so_differential.h"
#include "spd_geometry_common.h"

namespace fdapde {
/// @brief identifies per-point rotation operations that can reuse the logarithm at identity
enum class RotationUsage : unsigned {
    None = 0,
    IdentityDistance = 1,
    IdentityLog = 2,
    RotationPenalty = 4
};
/// @brief combines independent identity-based rotation uses
constexpr RotationUsage operator|(RotationUsage a, RotationUsage b) {
    return static_cast<RotationUsage>(static_cast<unsigned>(a) | static_cast<unsigned>(b));
}
namespace internals {
/// @brief binds a copied parameter to a borrowed or owned prepared rotation curve
template <typename Curve> class rotation_geodesic_expr : public GeometryExpr<rotation_geodesic_expr<Curve>> {
   public:
    using CurveType = std::remove_cvref_t<Curve>;
    using Scalar = typename CurveType::Scalar;
    static constexpr int Rows = CurveType::Rows, Cols = Rows, StorageOrder = RowMajor, NestAsRef = 0, ReadOnly = 1;
    using assignment_executor = deleted_assignment_executor;
    /// @brief retains prepared curve data without reconstructing coefficients
    rotation_geodesic_expr(Curve curve, double t) : curve_(std::forward<Curve>(curve)), t_(t) { }
    /// @brief returns the prepared matrix order
    int rows() const { return curve_.rows(); }
    /// @brief returns the prepared matrix order
    int cols() const { return rows(); }
    /// @brief reconstructs one matrix without rotation certification for ordinary destinations
    auto eval_matrix() const { return curve_.eval(t_); }
    /// @brief certifies a reconstructed rotation with the requested cache policy
    template <typename Policy> auto eval() const { return RotationMatrix<Scalar, Rows, Cols, Policy>(eval_matrix()); }
   private:
    Curve curve_;
    double t_;
};
/// @brief owns a chosen rotation branch and the factors needed to evaluate it at any finite parameter
template <typename S, int N> class rotation_geodesic {
   public:
    using Scalar = S;
    static constexpr int Rows = N;
    /// @brief prepares half-angle planes so the selected logarithm branch survives even at angle pi
    template <RotationLike Q>
    rotation_geodesic(const Q& first, const RotationLogResult<S, N>& branch) : diagnostics_(branch.diagnostics) {
        const auto half = rotation_exp(branch.tangent, 0.5);
        const rotation_schur<S, N> schur(half);
        right_ = schur.vectors();
        left_ = first * right_;
        angles_ = schur.angles();
        angles_ *= S(2);
    }
    /// @brief returns the prepared matrix order
    int rows() const { return right_.rows(); }
    /// @brief returns diagnostics for the retained branch
    RotationLogDiagnostics diagnostics() const { return diagnostics_; }
    /// @brief borrows persistent prepared factors and copies the parameter
    auto operator()(double t) const& { return rotation_geodesic_expr<const rotation_geodesic&>(*this, t); }
    /// @brief owns expiring factors inside the deferred expression
    auto operator()(double t) && { return rotation_geodesic_expr<rotation_geodesic>(std::move(*this), t); }
    /// @brief rejects borrowing a const temporary curve
    void operator()(double) const&& = delete;
    /// @brief reconstructs the selected curve using scalar trigonometry and stored factors
    auto eval(double parameter) const {
        const S t = static_cast<S>(parameter);
        fdapde_strong_assert(std::isfinite(t), std::invalid_argument, "rotation geodesic parameter must be finite");
        Matrix<S, N, N> result;
        if constexpr (N == Dynamic) result.resize(rows(), rows());
        result.set_zero();
        for (int k = 0; k < rows(); ++k) {
            if (angles_[k] < S(0)) continue;
            if (angles_[k] == S(0)) {
                for (int i = 0; i < rows(); ++i)
                    for (int j = 0; j < rows(); ++j) result(i, j) += left_(i, k) * right_(j, k);
            } else {
                const S angle = t * angles_[k];
                fdapde_strong_assert(std::isfinite(angle), std::domain_error, "rotation geodesic angle overflow");
                const S c = std::cos(angle), s = std::sin(angle);
                for (int i = 0; i < rows(); ++i)
                    for (int j = 0; j < rows(); ++j)
                        result(i, j) += c * (left_(i, k) * right_(j, k) + left_(i, k + 1) * right_(j, k + 1)) +
                                        s * (left_(i, k + 1) * right_(j, k) - left_(i, k) * right_(j, k + 1));
            }
        }
        return result;
    }
   private:
    Matrix<S, N, N> left_, right_;
    Vector<S, N> angles_;
    RotationLogDiagnostics diagnostics_;
};
}   // namespace internals
namespace manifold {
/// @brief defines the full-Frobenius bi-invariant metric on SO(n) with body-coordinate skew tangents
/// @details a tangent Omega denotes ambient velocity Q Omega and has norm squared tr(Omega transpose Omega)
template <typename S, int N, RotationUsage Uses = RotationUsage::None> class SOGeometry {
   public:
    static_assert(N == Dynamic || N > 0, "SO geometry order must be positive");
    static_assert((static_cast<unsigned>(Uses) & ~7u) == 0, "invalid SO geometry usage");
    using Scalar = S;
    using CachePolicy = std::conditional_t<Uses == RotationUsage::None, RotationCache::None, RotationCache::Log>;
    using Point = RotationMatrix<S, N, N, CachePolicy>;
    using Tangent = SkewSymmetricMatrix<S, N, N>;
    /// @brief retains one relative logarithm and optional immutable differential preparation
    struct RelativeFrame {
        RotationMatrix<S, N, N, RotationCache::Log> relative;
        std::optional<internals::SOLogDifferential<S, N>> differential;
    };
    /// @brief prepares spatial rotation interpolation over a simplex or an immutable mesh
    template <typename Element, typename Nodes>
        requires gfe::P1InterpolationBinding<Element, Nodes>
    auto interpolant(Element&& element, Nodes&& nodes) const;
    /// @brief prepares rotation interpolation with explicit mean and differential tolerances
    template <typename Element, typename Nodes>
        requires gfe::P1InterpolationBinding<Element, Nodes>
    auto interpolant(Element&& element, Nodes&& nodes, const gfe::P1GeodesicLinearizationOptions& options) const;
    /// @brief constructs a fixed positive-order rotation geometry
    SOGeometry()
        requires(N != Dynamic)
    = default;
    /// @brief constructs a dynamic geometry with a bounded positive matrix order
    explicit SOGeometry(int n)
        requires(N == Dynamic)
        : n_(n) {
        fdapde_strong_assert(
          n > 0 && std::int64_t(n) * n <= std::numeric_limits<int>::max() / 4, std::invalid_argument,
          "invalid SO geometry order");
    }
    /// @brief returns the rotation matrix order
    int order() const { return n_; }
    /// @brief returns the number of independent skew tangent coefficients
    std::size_t dimension() const { return std::size_t(n_) * (n_ - 1) / 2; }
    /// @brief pairs body-coordinate tangents using the full Frobenius metric
    template <RotationLike Q> double inner_product(const Q& q, const Tangent& u, const Tangent& v) const {
        check_point_(q);
        check_tangent_(u);
        check_tangent_(v);
        long double value = 0;
        for (int i = 0; i < n_; ++i)
            for (int j = i + 1; j < n_; ++j)
                value += 2 * static_cast<long double>(u(i, j)) * static_cast<long double>(v(i, j));
        return internals::checked_geometry_result(double(value));
    }
    /// @brief computes the scale-safe full Frobenius norm of a body tangent
    template <RotationLike Q> double norm(const Q& q, const Tangent& u) const {
        check_point_(q);
        check_tangent_(u);
        double value = 0;
        for (int i = 0; i < n_; ++i)
            for (int j = i + 1; j < n_; ++j) {
                value = std::hypot(value, double(u(i, j)));
                value = std::hypot(value, double(u(i, j)));
            }
        return value;
    }
    /// @brief copies a tangent already expressed in body skew coordinates
    template <RotationLike Q> Tangent project(const Q& q, const Tangent& u) const {
        check_point_(q);
        check_tangent_(u);
        return u;
    }
    /// @brief projects an ambient matrix to body coordinates using skew(Q transpose A)
    template <RotationLike Q, typename A> Tangent project(const Q& q, const MatrixExpr<A>& ambient) const {
        return from_ambient(q, ambient);
    }
    /// @brief converts ambient velocity or gradient to its orthogonal body-coordinate projection
    template <RotationLike Q, typename A> Tangent from_ambient(const Q& q, const MatrixExpr<A>& ambient) const {
        check_point_(q);
        check_shape_(ambient);
        const Matrix<S, N, N> body(q.transpose() * ambient);
        Tangent result = zero_tangent(q);
        for (int i = 0; i < n_; ++i)
            for (int j = i + 1; j < n_; ++j)
                result(i, j) = internals::checked_geometry_result(S(0.5) * (body(i, j) - body(j, i)));
        return result;
    }
    /// @brief returns the ambient velocity Q Omega represented by a body tangent
    template <RotationLike Q> auto to_ambient(const Q& q, const Tangent& u) const {
        check_point_(q);
        check_tangent_(u);
        return Matrix<S, N, N>(q * u);
    }
    /// @brief converts a Euclidean gradient to the body gradient for the full Frobenius metric
    template <RotationLike Q, typename A>
    Tangent euclidean_to_riemannian_gradient(const Q& q, const MatrixExpr<A>& gradient) const {
        return from_ambient(q, gradient);
    }
    /// @brief creates an owning zero skew tangent of the geometry order
    template <RotationLike Q> Tangent zero_tangent(const Q& q) const {
        check_point_(q);
        Tangent result;
        if constexpr (N == Dynamic) result.resize(n_, n_);
        return result;
    }
    /// @brief combines finite body tangents with finite scalar weights
    template <RotationLike Q>
    Tangent linear_combination(const Q& q, double alpha, const Tangent& u, double beta, const Tangent& v) const {
        check_tangent_(u);
        check_tangent_(v);
        const S a = internals::geometry_coefficient<S>(alpha), b = internals::geometry_coefficient<S>(beta);
        Tangent result = zero_tangent(q);
        for (int i = 0; i < n_; ++i)
            for (int j = i + 1; j < n_; ++j)
                result(i, j) = internals::checked_geometry_result(a * S(u(i, j)) + b * S(v(i, j)));
        return result;
    }
    /// @brief follows Q exp(step Omega) and certifies the rounded rotation
    template <RotationLike Q> Point exponential(const Q& q, const Tangent& u, double step = 1) const {
        check_point_(q);
        check_tangent_(u);
        const auto delta = fdapde::internals::skew_exponential(u, step);
        return Point(q * delta);
    }
    /// @brief uses the exact exponential as a rotation-preserving retraction
    template <RotationLike Q> Point retract(const Q& q, const Tangent& u, double step = 1) const {
        return exponential(q, u, step);
    }
    /// @brief returns the unique minimum body logarithm of from transpose times to
    template <RotationLike Q, RotationLike R> Tangent logarithm(const Q& from, const R& to) const {
        return rotation_log(relative_(from, to));
    }
    /// @brief prepares a candidate-relative logarithm cache shared by the mean cost and gradient
    template <RotationLike Q, RotationLike R> RelativeFrame relative_frame(const Q& from, const R& to) const {
        check_point_(from);
        check_point_(to);
        return {RotationMatrix<S, N, N, RotationCache::Log>(from.transpose() * to), std::nullopt};
    }
    /// @brief reuses the retained minimum body logarithm without repeating the relative decomposition
    Tangent logarithm(const RelativeFrame& frame) const {
        check_point_(frame.relative);
        return rotation_log(frame.relative);
    }
    /// @brief reuses the retained distance even when the logarithm branch is ambiguous
    double distance(const RelativeFrame& frame) const {
        check_point_(frame.relative);
        return frame.relative.cache().distance();
    }
    /// @brief applies the target logarithm differential in body coordinates
    Tangent logarithm_target_jvp(const RelativeFrame& frame, const Tangent& v) const {
        return with_differential_(frame, [&](const auto& d) { return d.target_action(v); });
    }
    /// @brief applies the full-Frobenius metric adjoint of the target logarithm differential
    Tangent logarithm_target_vjp(const RelativeFrame& frame, const Tangent& z) const {
        return with_differential_(frame, [&](const auto& d) { return d.target_action(z, true); });
    }
    /// @brief applies the covariant base Hessian of one half the squared distance
    Tangent half_squared_distance_hessian_vector(const RelativeFrame& frame, const Tangent& u) const {
        return with_differential_(frame, [&](const auto& d) { return d.hessian_action(u); });
    }
    /// @brief differentiates the Hessian in both endpoints while parallel-transporting its input
    Tangent half_squared_distance_hessian_covariant_jvp(
      const RelativeFrame& frame, const Tangent& u, const Tangent& v, const Tangent& w) const {
        return with_differential_(frame, [&](const auto& d) { return d.mixed_action(u, v, w); });
    }
    /// @brief returns both endpoint metric adjoints of the covariant Hessian differential
    auto
    half_squared_distance_hessian_covariant_vjp(const RelativeFrame& frame, const Tangent& w, const Tangent& z) const {
        return with_differential_(frame, [&](const auto& d) { return d.mixed_adjoint(w, z); });
    }
    /// @brief retains differential spectra and rejects a singular or nonpositive local mean Hessian
    void
    prepare_linearization(std::vector<std::optional<RelativeFrame>>& frames, std::span<const double> weights) const {
        fdapde_strong_assert(
          frames.size() == weights.size(), std::invalid_argument, "SO derivative frame and weight counts must match");
        const auto m = dimension();
        fdapde_strong_assert(
          m <= std::size_t(std::sqrt(double(std::numeric_limits<int>::max()))), std::length_error,
          "SO differential workspace exceeds the native matrix index range");
        Matrix<S, Dynamic, Dynamic> h(static_cast<int>(m), static_cast<int>(m));
        h.set_zero();
        for (std::size_t i = 0; i < frames.size(); ++i) {
            if (!frames[i]) continue;
            auto& frame = *frames[i];
            require_regular_(frame);
            if (!frame.differential) frame.differential.emplace(logarithm(frame));
            if (weights[i] != 0) h += S(weights[i]) * frame.differential->hessian();
        }
        if (m == 0) return;
        const EVD evd(h.template as_symmetric<Lower>());
        S scale = 1;
        for (int i = 0; i < static_cast<int>(m); ++i) scale = std::max(scale, std::abs(S(evd.eigenvalues()[i])));
        for (int i = 0; i < static_cast<int>(m); ++i)
            fdapde_strong_assert(
              evd.eigenvalues()[i] > 64 * m * std::numeric_limits<S>::epsilon() * scale, std::domain_error,
              "SO interpolation derivatives require a positive definite mean Hessian");
    }
    /// @brief explicitly chooses a minimum body logarithm and reports ambiguity or numerical branch limitations
    template <RotationLike Q, RotationLike R> auto minimum_logarithm(const Q& from, const R& to) const {
        return minimum_rotation_log(relative_(from, to));
    }
    /// @brief measures the full-Frobenius geodesic distance including relative eigenvalues equal to minus one
    template <RotationLike Q, RotationLike R> double distance(const Q& from, const R& to) const {
        return rotation_distance_identity(relative_(from, to));
    }
    /// @brief reuses a per-point logarithmic cache for distance from identity
    template <RotationLike Q> double distance_from_identity(const Q& q) const {
        check_point_(q);
        return rotation_distance_identity(q);
    }
    /// @brief returns one half the squared identity distance including the cut locus
    template <RotationLike Q> double rotation_penalty(const Q& q) const {
        const double d = distance_from_identity(q);
        return S(0.5) * d * d;
    }
    /// @brief returns the identity-penalty gradient only on a numerically regular logarithm branch
    template <RotationLike Q> Tangent rotation_penalty_gradient(const Q& q) const {
        check_point_(q);
        if constexpr (Q::CachePolicy::Flags != 0) {
            fdapde_strong_assert(
              q.cache().diagnostics().regular(), std::domain_error,
              "rotation penalty gradient requires a regular branch");
            return rotation_log(q);
        } else {
            const auto branch = minimum_rotation_log(q);
            fdapde_strong_assert(
              branch.diagnostics.regular(), std::domain_error, "rotation penalty gradient requires a regular branch");
            return branch.tangent;
        }
    }
    /// @brief prepares the unique minimum branch and owns all endpoint data needed by the curve
    template <RotationLike Q, RotationLike R> auto geodesic(const Q& from, const R& to) const {
        const auto branch = minimum_logarithm(from, to);
        fdapde_strong_assert(
          branch.diagnostics.unique(), std::domain_error,
          "rotation geodesic requires an explicit branch at the cut locus");
        return fdapde::internals::rotation_geodesic<S, N>(from, branch);
    }
    /// @brief retains an explicitly selected minimum branch after checking its endpoint and length
    template <RotationLike Q, RotationLike R>
    auto geodesic(const Q& from, const R& to, const RotationLogResult<S, N>& branch) const {
        check_point_(from);
        check_point_(to);
        check_tangent_(branch.tangent);
        const auto endpoint = exponential(from, branch.tangent);
        const RotationMatrix<S, N, N, RotationCache::Log> relative(from.transpose() * to);
        const double tolerance = 512 * n_ * std::numeric_limits<S>::epsilon();
        double error = 0;
        for (int i = 0; i < n_; ++i)
            for (int j = 0; j < n_; ++j) error = std::max(error, double(std::abs(endpoint(i, j) - to(i, j))));
        fdapde_strong_assert(
          error <= tolerance && std::abs(norm(from, branch.tangent) - relative.cache().distance()) <= tolerance * n_,
          std::invalid_argument, "rotation branch must be a minimum logarithm of the supplied endpoints");
        return fdapde::internals::rotation_geodesic<S, N>(from, {branch.tangent, relative.cache().diagnostics()});
    }
    /// @brief parallel-transports body tangents along the uniquely resolved shortest geodesic
    template <RotationLike Q, RotationLike R> Tangent transport(const Q& from, const R& to, const Tangent& u) const {
        check_tangent_(u);
        const auto omega = logarithm(from, to);
        const auto half = rotation_exp(omega, 0.5);
        const Matrix<S, N, N> value(half.transpose() * u * half);
        return Tangent(value.template as_skew_symmetric<Upper>());
    }
   private:
    /// @brief rejects branch-sensitive derivatives at ambiguous or numerically near-cut relative rotations
    void require_regular_(const RelativeFrame& frame) const {
        check_point_(frame.relative);
        fdapde_strong_assert(
          frame.relative.cache().diagnostics().regular(), std::domain_error,
          "SO derivatives require a regular relative logarithm branch");
    }
    /// @brief reuses immutable derivative preparation or constructs a private one for an isolated geometry call
    template <typename Apply> auto with_differential_(const RelativeFrame& frame, Apply apply) const {
        require_regular_(frame);
        if (frame.differential) return apply(*frame.differential);
        return apply(internals::SOLogDifferential<S, N>(logarithm(frame)));
    }
    /// @brief checks matrix dimensions before any geometry loop accesses coefficients
    template <typename Xpr> void check_shape_(const Xpr& q) const {
        fdapde_strong_assert(
          q.rows() == n_ && q.cols() == n_, std::invalid_argument, "SO geometry: incompatible shape");
    }
    /// @brief accepts only verified rotations with this geometry's order and scalar
    template <RotationLike Q> void check_point_(const Q& q) const {
        static_assert(std::same_as<S, typename Q::Scalar>, "SO geometry scalar mismatch");
        check_shape_(q);
    }
    /// @brief checks finite body tangent coefficients and the geometry order
    void check_tangent_(const Tangent& u) const {
        check_shape_(u);
        for (int i = 0; i < n_; ++i)
            for (int j = i + 1; j < n_; ++j)
                fdapde_strong_assert(std::isfinite(S(u(i, j))), std::invalid_argument, "SO tangent must be finite");
    }
    /// @brief forms the relative rotation without assuming endpoint logarithms can be subtracted
    template <RotationLike Q, RotationLike R> auto relative_(const Q& from, const R& to) const {
        check_point_(from);
        check_point_(to);
        return RotationMatrix<S, N, N>(from.transpose() * to);
    }
    int n_ = N;
};
/// @brief solves a local rotation mean and reuses the local stationarity refinement when necessary
template <typename S, int N, RotationUsage Uses, typename Samples>
WeightedKarcherMeanResult<typename SOGeometry<S, N, Uses>::Point> weighted_karcher_mean(
  const SOGeometry<S, N, Uses>& geometry, const Samples& samples, std::span<const double> weights,
  const typename SOGeometry<S, N, Uses>::Point& initial, const WeightedKarcherMeanOptions& options = {},
  internals::KarcherWorkspace<SOGeometry<S, N, Uses>>* retained = nullptr) {
    using Geometry = SOGeometry<S, N, Uses>;
    internals::KarcherWorkspace<Geometry> workspace;
    auto result = weighted_karcher_mean<Geometry>(geometry, samples, weights, initial, options, &workspace);
    internals::polish_karcher_mean(geometry, samples, options, result, workspace);
    if (retained) *retained = std::move(workspace);
    return result;
}
}   // namespace manifold
}   // namespace fdapde
#endif
