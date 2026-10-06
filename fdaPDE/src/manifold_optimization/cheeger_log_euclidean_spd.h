// SPDX-License-Identifier: GPL-3.0-or-later
#ifndef __FDAPDE_CHEEGER_LOG_EUCLIDEAN_SPD_H__
#define __FDAPDE_CHEEGER_LOG_EUCLIDEAN_SPD_H__
#include "header_check.h"

namespace fdapde::manifold {
namespace internals {
/// @brief stores the trace and two traceless coordinates of a symmetric matrix
struct CheegerChart {
    double s = 0, x = 0, y = 0;
};
/// @brief extracts the three independent symmetric chart coordinates
template <typename M> CheegerChart cheeger_chart(const M& m) {
    return {double(m(0, 0) + m(1, 1)) / 2, double(m(0, 0) - m(1, 1)) / 2, double(m(0, 1))};
}
/// @brief reconstructs a native symmetric matrix from chart coordinates
template <typename Scalar> auto cheeger_matrix(CheegerChart a) {
    SymmetricMatrix<Scalar, 2, 2> m;
    m(0, 0) = Scalar(a.s + a.x);
    m(0, 1) = Scalar(a.y);
    m(1, 1) = Scalar(a.s - a.x);
    return m;
}
/// @brief conjugates a symmetric chart by the planar rotation using its doubled angle
inline CheegerChart cheeger_rotate(CheegerChart a, double phi) {
    const double c = std::cos(2 * phi), s = std::sin(2 * phi);
    return {a.s, c * a.x - s * a.y, s * a.x + c * a.y};
}
/// @brief pairs chart directions in the full Frobenius metric
inline double cheeger_dot(CheegerChart a, CheegerChart b) { return 2 * (a.s * b.s + a.x * b.x + a.y * b.y); }
/// @brief retains the minimum pair distance and all numerically tied minimizing rotation angles
struct CheegerPair {
    double squared_distance;
    std::vector<double> rotations;
    /// @brief reports whether the planar search found exactly one minimizing angle
    bool unique() const { return rotations.size() == 1; }
};
/// @brief enumerates stationary points of the SPD2 pair rotation objective at fixed rho
inline CheegerPair cheeger_pair(CheegerChart a, CheegerChart b, double rho) {
    const double pi = std::acos(-1.), aa = std::hypot(a.x, a.y), ab = std::hypot(b.x, b.y), p = aa * ab, e = rho;
    const double angle = std::atan2(b.y, b.x) - std::atan2(a.y, a.x);
    const double delta = std::atan2(std::sin(angle), std::cos(angle)) / 2;
    auto cost = [&](double phi) {
        return 2 * std::pow(b.s - a.s, 2) + 2 * std::pow(ab - aa, 2) + 8 * p * std::pow(std::sin(delta - phi), 2) +
               2 * e * phi * phi;
    };
    auto gradient = [&](double phi) { return -8 * p * std::sin(2 * (delta - phi)) + 4 * e * phi; };
    std::vector<double> bounds {-pi / 2, pi / 2}, candidates {-pi / 2, pi / 2};
    if (e <= 4 * p) {
        const double offset = std::acos(-e / (4 * p)) / 2;
        for (int sign : {-1, 1})
            for (int k : {-1, 0, 1}) {
                const double phi = delta + sign * offset + k * pi;
                if (phi > -pi / 2 && phi < pi / 2) bounds.push_back(phi);
            }
    }
    std::sort(bounds.begin(), bounds.end());
    for (double phi : bounds)
        if (std::abs(gradient(phi)) <= 1e-14 * (e + p)) candidates.push_back(phi);
    for (std::size_t i = 1; i < bounds.size(); ++i) {
        double lo = bounds[i - 1], hi = bounds[i], gl = gradient(lo);
        if (gl * gradient(hi) >= 0) continue;
        double root = 0;
        bool refined = false;
        const double epsilon = std::numeric_limits<double>::epsilon();
        // keep amplified cost scales and nearly degenerate curvature on the established bisection path
        if (e <= 1 / epsilon && e > 4 * p && e - 4 * p > std::sqrt(epsilon) * (e + 4 * p)) {
            // keep Newton inside the original convex bracket and avoid a residual-only stop near flat curvature
            double left = lo, right = hi, phi = 0;
            for (int iteration = 0; iteration < 16; ++iteration) {
                const double g = gradient(phi), curvature = 16 * p * std::cos(2 * (delta - phi)) + 4 * e;
                if (!std::isfinite(g) || !std::isfinite(curvature) || !(curvature > 0)) break;
                const double radius = 8 * epsilon * std::max(1., std::abs(phi)),
                             near_left = std::max(left, phi - radius), near_right = std::min(right, phi + radius),
                             error =
                               4 * epsilon * (8 * p + 4 * e * std::max(std::abs(near_left), std::abs(near_right)));
                // require opposite signs beyond estimated evaluation roundoff across a machine-scale bracket
                if (near_left < near_right && gradient(near_left) < -error && gradient(near_right) > error) {
                    root = (near_left + near_right) / 2;
                    refined = true;
                    break;
                }
                if ((g > 0) == (gl > 0))
                    left = phi;
                else
                    right = phi;
                const double next = phi - g / curvature;
                phi = std::isfinite(next) && next > left && next < right ? next : (left + right) / 2;
                if (!(phi > left && phi < right)) break;
            }
        }
        if (!refined) {
            // unresolved or poorly conditioned refinements retain the original sixty-step bisection interval
            for (int k = 0; k < 60; ++k) {
                const double mid = (lo + hi) / 2;
                if ((gradient(mid) > 0) == (gl > 0))
                    lo = mid;
                else
                    hi = mid;
            }
            root = (lo + hi) / 2;
        }
        candidates.push_back(root);
    }
    double best = std::numeric_limits<double>::infinity();
    for (double phi : candidates) best = std::min(best, cost(phi));
    std::sort(candidates.rbegin(), candidates.rend());
    CheegerPair result {best, {}};
    for (double phi : candidates)
        if (
          cost(phi) - best <= 1e-12 * std::max(1., best) &&
          (result.rotations.empty() || std::abs(phi - result.rotations.back()) > 1e-8))
            result.rotations.push_back(phi);
    return result;
}
}   // namespace internals

/// @brief declares the Cheeger log-Euclidean metric with ambient symmetric tangents
/// @details pair diagnostics retain detected ties without claiming a global uniqueness certificate
template <typename Scalar_, int Order_ = 2, Usage Uses_ = Usage::None> class CheegerLogEuclideanSPDGeometry;

/// @brief specializes the C-LE metric and exact pair search for planar tensors
template <typename Scalar_, Usage Uses_> class CheegerLogEuclideanSPDGeometry<Scalar_, 2, Uses_> {
    static constexpr int Order_ = 2;
    using Base = LogEuclideanSPDGeometry<Scalar_, Order_, Uses_>;
    using Chart = internals::CheegerChart;
    Base ambient_;
    double rho_;
   public:
    using Scalar = Scalar_;
    using Point = typename Base::Point;
    using Tangent = typename Base::Tangent;
    /// @brief preserves the epsilon constructor while storing its positive finite square
    explicit CheegerLogEuclideanSPDGeometry(double epsilon = 0.5) : rho_(epsilon * epsilon) {
        fdapde_strong_assert(
          std::isfinite(epsilon) && epsilon > 0 && std::isfinite(rho_) && rho_ > 0, std::invalid_argument,
          "Cheeger-LE requires a finite positive epsilon squared");
    }
    /// @brief constructs the geometry from rho directly without a square-root round trip
    static CheegerLogEuclideanSPDGeometry from_rho(double rho) {
        fdapde_strong_assert(
          std::isfinite(rho) && rho > 0, std::invalid_argument, "Cheeger-LE requires finite rho > 0");
        CheegerLogEuclideanSPDGeometry result;
        result.rho_ = rho;
        return result;
    }
    /// @brief replaces only the local metric parameter
    CheegerLogEuclideanSPDGeometry with_rho(double rho) const { return from_rho(rho); }
    /// @brief returns the local squared rotation penalty
    double rho() const { return rho_; }
    /// @brief prepares spatial C-LE interpolation on a persistent batch and simplex or mesh
    template <typename Element, typename Nodes>
        requires gfe::P1InterpolationBinding<Element, Nodes>
    auto interpolant(Element&& element, Nodes&& nodes) const;
    /// @brief prepares C-LE interpolation with explicit numerical tolerances
    template <typename Element, typename Nodes>
        requires gfe::P1InterpolationBinding<Element, Nodes>
    auto interpolant(Element&& element, Nodes&& nodes, const gfe::P1GeodesicLinearizationOptions& options) const;
    // metric-dependent operations use C-LE formulas; only tangent algebra is shared with LE
    /// @brief retains the optimal pair lift for repeated fixed-rho geodesic evaluation
    class Curve {
       public:
        /// @brief copies endpoint charts and the unique minimizing alignment
        Curve(Chart x, Chart y, double angle, double squared_distance) :
            x_(x), y_(internals::cheeger_rotate(y, -angle)), angle_(angle), squared_distance_(squared_distance) { }
        /// @brief evaluates the lifted straight segment and maps its rotation back to SPD2
        Point operator()(double t) const {
            fdapde_strong_assert(std::isfinite(t), std::invalid_argument, "nonfinite geodesic parameter");
            return from_chart(
              internals::cheeger_rotate(
                {(1 - t) * x_.s + t * y_.s, (1 - t) * x_.x + t * y_.x, (1 - t) * x_.y + t * y_.y}, t * angle_));
        }
        /// @brief returns the prepared endpoint distance for exact two-node mean diagnostics
        double squared_distance() const { return squared_distance_; }
       private:
        Chart x_, y_;
        double angle_, squared_distance_;
    };
    /// @brief prepares a fixed-rho pair geodesic after rejecting minimizing alignment ties
    template <SPDLike A, SPDLike B> Curve geodesic(const A& a, const B& b) const {
        const auto x = chart(a), y = chart(b);
        const auto branch = internals::cheeger_pair(x, y, rho_);
        fdapde_strong_assert(branch.rotations.size() == 1, std::domain_error, "ambiguous Cheeger geodesic");
        return Curve(x, y, branch.rotations.front(), branch.squared_distance);
    }
    /// @brief returns the supported tensor order
    int order() const { return ambient_.order(); }
    /// @brief returns the symmetric tangent dimension
    std::size_t dimension() const { return ambient_.dimension(); }
    /// @brief copies a metric-independent ambient symmetric tangent
    Tangent project(const Point& p, const Tangent& u) const { return ambient_.project(p, u); }
    /// @brief creates a zero ambient tangent
    Tangent zero_tangent(const Point& p) const { return ambient_.zero_tangent(p); }
    /// @brief combines ambient symmetric tangents with finite scalar coefficients
    Tangent linear_combination(const Point& p, double a, const Tangent& u, double b, const Tangent& v) const {
        return ambient_.linear_combination(p, a, u, b, v);
    }
    /// @brief returns the legacy square-root parameter
    double epsilon() const { return std::sqrt(rho_); }
    /// @brief solves the two-point rotational alignment using cached native logarithms
    template <SPDLike A, SPDLike B> auto pair(const A& a, const B& b) const {
        return internals::cheeger_pair(chart(a), chart(b), rho_);
    }
    /// @brief evaluates a point logarithm and extracts its symmetric coordinates
    template <SPDLike P> static Chart chart(const P& p) {
        fdapde_strong_assert(p.rows() == 2 && p.cols() == 2, std::invalid_argument, "Cheeger-LE requires SPD2 nodes");
        return internals::cheeger_chart(matrix_log(p));
    }
    /// @brief exponentiates a symmetric chart into a certified SPD point
    static Point from_chart(Chart x) { return Point(matrix_exp(internals::cheeger_matrix<Scalar>(x))); }
    /// @brief returns the intrinsic pair distance including detected minimizing ties
    double distance(const Point& a, const Point& b) const { return std::sqrt(pair(a, b).squared_distance); }
    /// @brief pairs ambient tangents through the rho-dependent logarithmic metric
    double inner_product(const Point& p, const Tangent& u, const Tangent& v) const {
        const auto x = chart(p), h = internals::cheeger_chart(matrix_log_frechet(p, u)),
                   k = internals::cheeger_chart(matrix_log_frechet(p, v));
        return internals::cheeger_dot(h, k) -
               8 * (x.x * h.y - x.y * h.x) * (x.x * k.y - x.y * k.x) / (rho_ + 4 * (x.x * x.x + x.y * x.y));
    }
    /// @brief computes the norm induced by the local C-LE metric
    double norm(const Point& p, const Tangent& u) const { return std::sqrt(std::max(0., inner_product(p, u, u))); }
    /// @brief follows the quotient geodesic defined by an ambient initial tangent
    Point exponential(const Point& p, const Tangent& u, double step = 1) const {
        fdapde_strong_assert(std::isfinite(step), std::invalid_argument, "nonfinite exponential step");
        const auto x = chart(p), h = internals::cheeger_chart(matrix_log_frechet(p, u));
        const double phi = 2 * (x.x * h.y - x.y * h.x) / (rho_ + 4 * (x.x * x.x + x.y * x.y));
        return from_chart(
          internals::cheeger_rotate(
            {x.s + step * h.s, x.x + step * (h.x + 2 * phi * x.y), x.y + step * (h.y - 2 * phi * x.x)}, step * phi));
    }
    /// @brief uses the exact local exponential as a retraction
    Point retract(const Point& p, const Tangent& u, double step) const { return exponential(p, u, step); }
    /// @brief returns the unique minimizing ambient logarithm or rejects a pair tie
    Tangent logarithm(const Point& p, const Point& q) const {
        const auto x = chart(p);
        const auto branch = pair(p, q);
        fdapde_strong_assert(branch.rotations.size() == 1, std::domain_error, "ambiguous Cheeger logarithm");
        const double phi = branch.rotations.front();
        const auto y = internals::cheeger_rotate(chart(q), -phi);
        return Tangent(matrix_exp_frechet(
          internals::cheeger_matrix<Scalar>(x),
          internals::cheeger_matrix<Scalar>({y.s - x.s, y.x - x.x - 2 * phi * x.y, y.y - x.y + 2 * phi * x.x})));
    }
};
}   // namespace fdapde::manifold
#endif
