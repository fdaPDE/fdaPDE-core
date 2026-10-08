// SPDX-License-Identifier: GPL-3.0-or-later
#ifndef __FDAPDE_CHEEGER_P1_H__
#define __FDAPDE_CHEEGER_P1_H__
#include "header_check.h"

namespace fdapde::gfe {
/// @brief records minimizing ties and optional global uniqueness certificates for a Cheeger P1 value
template <typename Point> struct CheegerP1ValueResult : P1ValueResult<Point> {
    bool detected_ambiguity = false;   // multistart evidence, never a global uniqueness certificate
    double objective = 0;
};

namespace internals {
template <typename Geometry> inline constexpr bool is_cheeger_geometry = false;
template <typename S, int N, Usage U, typename Point_>
inline constexpr bool is_cheeger_geometry<manifold::CheegerLogEuclideanSPDGeometry<S, N, U, Point_>> = true;
}   // namespace internals
/// @brief prepares native SPD2 C-LE interpolation and implicit first derivatives at fixed nodal rho
/// @details weight derivatives require interior weights; nodal pullbacks retain inactive-coordinate support
/// @details chart data are snapshots; all matrix coefficients remain in the native input batch
template <typename S, Usage U, typename Point_, typename Nodes>
class P1GeodesicLinearization<manifold::CheegerLogEuclideanSPDGeometry<S, 2, U, Point_>, Nodes> {
    static constexpr int N = 2;
   public:
    using Geometry = manifold::CheegerLogEuclideanSPDGeometry<S, N, U, Point_>;
    using Point = typename Geometry::Point;
    using Tangent = typename Geometry::Tangent;
    using Chart = manifold::internals::CheegerChart;
    using Dense = fdapde::Matrix<double, Dynamic, Dynamic>;
    using Vector = fdapde::Vector<double, Dynamic>;
    /// @brief snapshots cached nodal logarithms, selects a mean branch and optionally factors its Hessian
    P1GeodesicLinearization(
      const Geometry& geometry, Nodes nodes, std::span<const double> weights,
      const P1GeodesicLinearizationOptions& options = {}, std::span<const double> rho_nodes = {},
      bool derivatives = true) :
        geometry_(geometry),
        binding_(std::forward<Nodes>(nodes)),
        options_(options),
        w_(weights.begin(), weights.end()),
        rho_nodes_(rho_nodes.begin(), rho_nodes.end()) {
        auto vertex = internals::validate_p1_data(binding_.size(), weights);
        fdapde_strong_assert(
          options.mean.solver.gradient_tolerance > 0 && std::isfinite(options.mean.solver.gradient_tolerance) &&
            options.linear_solve.residual_tolerance > 0 && std::isfinite(options.linear_solve.residual_tolerance),
          std::invalid_argument, "Cheeger P1 requires positive finite tolerances");
        nodes_.reserve(binding_.size());
        for (std::size_t i = 0; i < binding_.size(); ++i) nodes_.push_back(Geometry::chart(binding_[i]));
        if (!rho_nodes_.empty()) {
            fdapde_strong_assert(
              rho_nodes_.size() == w_.size(), std::invalid_argument, "rho count must match nodal DOFs");
            for (double rho : rho_nodes_) Geometry::from_rho(rho);
        }
        double total = 0;
        for (double w : w_) total += w;
        weight_total_ = total;
        for (double& w : w_) w /= total;
        if (!rho_nodes_.empty()) {
            double rho = 0;
            for (std::size_t i = 0; i < w_.size(); ++i) rho += w_[i] * rho_nodes_[i];
            geometry_ = Geometry::from_rho(rho);
        }
        result_.normalized_weights = w_;
        std::vector<std::size_t> active;
        for (std::size_t i = 0; i < w_.size(); ++i)
            if (w_[i] > 0) active.push_back(i);
        double threshold = 0;
        for (auto i : active) {
            double maximum = 0;
            for (auto j : active)
                if (i != j) maximum = std::max(maximum, std::hypot(nodes_[j].x, nodes_[j].y));
            threshold += 4 * w_[i] * std::hypot(nodes_[i].x, nodes_[i].y) * maximum;
        }
        const bool certified = geometry_.rho() > threshold;
        if (active.size() <= 2) {
            std::vector<double> phi(w_.size(), 0);
            if (active.size() == 2) {
                const auto pair =
                  manifold::internals::cheeger_pair(nodes_[active[0]], nodes_[active[1]], geometry_.rho());
                const double angle = pair.rotations.front();
                phi[active[0]] = -w_[active[1]] * angle;
                phi[active[1]] = w_[active[0]] * angle;
                result_.detected_ambiguity = pair.rotations.size() > 1;
            }
            fit_ = evaluate_(phi);
            result_.value = vertex ? Point(binding_[*vertex]) : Geometry::from_chart(fit_.mean);
            result_.stationarity_norm = 0;
            result_.stop_reason = manifold::BarycenterStopReason::closed_form;
            if (vertex) result_.uniqueness = manifold::BarycenterUniqueness::globally_unique;
        } else if (certified) {
            fit_ = optimize_convex_();
            result_.value = Geometry::from_chart(fit_.mean);
            result_.stationarity_norm = norm_(fit_.gradient);
            result_.stop_reason = result_.stationarity_norm <= options.mean.solver.gradient_tolerance ?
                                    manifold::BarycenterStopReason::stationarity_tolerance :
                                    manifold::BarycenterStopReason::max_iterations;
        } else {
            // ponytail: node-aligned multistart is local, add a certificate before optimizing rho for uniqueness
            std::vector<std::vector<double>> starts(1, std::vector<double>(w_.size(), 0));
            for (const auto& reference : nodes_) {
                std::vector<double> phi;
                const double theta = std::atan2(reference.y, reference.x) / 2;
                for (const auto& node : nodes_) phi.push_back(wrap_(std::atan2(node.y, node.x) / 2 - theta));
                starts.push_back(std::move(phi));
            }
            std::vector<Fit> fits;
            for (const auto& start : starts) fits.push_back(optimize_(start));
            std::sort(fits.begin(), fits.end(), [](const Fit& a, const Fit& b) { return a.cost < b.cost; });
            fit_ = fits.front();
            result_.value = Geometry::from_chart(fit_.mean);
            result_.stationarity_norm = norm_(fit_.gradient);
            result_.stop_reason = result_.stationarity_norm <= options.mean.solver.gradient_tolerance ?
                                    manifold::BarycenterStopReason::stationarity_tolerance :
                                    manifold::BarycenterStopReason::max_iterations;
            for (const auto& f : fits)
                if (
                  norm_(f.gradient) <= options_.mean.solver.gradient_tolerance &&
                  f.cost - fit_.cost < 1e-10 * std::max(1., fit_.cost) &&
                  std::hypot(f.mean.x - fit_.mean.x, f.mean.y - fit_.mean.y) > 1e-5)
                    result_.detected_ambiguity = true;
        }
        if (certified) result_.uniqueness = manifold::BarycenterUniqueness::globally_unique;
        result_.objective = fit_.cost;
        result_.iterations = fit_.iterations;
        if (derivatives && result_.converged() && !result_.detected_ambiguity) {
            hessian_.emplace(lifted_hessian_(fit_));
            try {
                const SPDMatrix<double, Dynamic> positive(*hessian_);
                static_cast<void>(positive);
                lu_.emplace(*hessian_);
                mean_log_.emplace(manifold::internals::cheeger_matrix<S>(fit_.mean));
                mean_log_->cache();
            } catch (const std::domain_error&) {
                // a value can be resolved while its local derivative remains singular
            }
        }
    }
    /// @brief exposes the selected candidate and its convergence and ambiguity diagnostics
    const CheegerP1ValueResult<Point>& result() const& { return result_; }
    /// @brief prevents diagnostics references from escaping a temporary workspace
    const CheegerP1ValueResult<Point>& result() const&& = delete;
    /// @brief returns the local metric after interpolating the optional nodal rho field
    const Geometry& geometry() const& { return geometry_; }
    /// @brief prevents geometry references from escaping a temporary workspace
    void geometry() const&& = delete;
    /// @brief differentiates weights including the P1 rho variation when a nodal field is bound
    P1DerivativeResult<Tangent> weight_jvp(std::span<const double> direction) const {
        fdapde_strong_assert(direction.size() == w_.size(), std::invalid_argument, "weight direction size mismatch");
        fdapde_strong_assert(
          std::all_of(w_.begin(), w_.end(), [](double w) { return w > 0; }), std::domain_error,
          "Cheeger weight derivatives require interior weights");
        double sum = 0, sumabs = 0;
        for (double d : direction) {
            fdapde_strong_assert(std::isfinite(d), std::invalid_argument, "nonfinite weight direction");
            sum += d;
            sumabs += std::abs(d);
        }
        fdapde_strong_assert(
          std::isfinite(sum) && std::isfinite(sumabs) &&
            std::abs(sum) <= p1_weight_sum_tolerance(w_.size()) * std::max(1., sumabs),
          std::invalid_argument, "weight direction must sum to zero");
        std::vector<double> dw(w_.size());
        double drho = 0;
        for (std::size_t i = 0; i < w_.size(); ++i) {
            dw[i] = (direction[i] - w_[i] * sum) / weight_total_;
            if (!rho_nodes_.empty()) drho += dw[i] * rho_nodes_[i];
        }
        return differentiate_(dw, {}, drho);
    }
    /// @brief differentiates the local squared penalty at fixed weights and matrix data
    P1DerivativeResult<Tangent> rho_jvp(double direction = 1) const {
        fdapde_strong_assert(std::isfinite(direction), std::invalid_argument, "nonfinite rho direction");
        return differentiate_({}, {}, direction);
    }
    /// @brief differentiates nodal rho coefficients through their local P1 interpolation
    P1DerivativeResult<Tangent> nodal_rho_jvp(std::span<const double> direction) const {
        fdapde_strong_assert(direction.size() == w_.size(), std::invalid_argument, "rho direction size mismatch");
        double drho = 0;
        for (std::size_t i = 0; i < w_.size(); ++i) {
            fdapde_strong_assert(std::isfinite(direction[i]), std::invalid_argument, "nonfinite rho direction");
            drho += w_[i] * direction[i];
        }
        return rho_jvp(drho);
    }
    /// @brief differentiates along ambient symmetric nodal tangents using native logarithmic differentials
    P1DerivativeResult<Tangent> nodal_jvp(std::span<const Tangent> directions) const {
        fdapde_strong_assert(directions.size() == w_.size(), std::invalid_argument, "nodal direction size mismatch");
        std::vector<Chart> charts;
        for (std::size_t i = 0; i < directions.size(); ++i) {
            charts.push_back(manifold::internals::cheeger_chart(matrix_log_frechet(binding_[i], directions[i])));
        }
        return differentiate_({}, charts, 0);
    }
    /// @brief preserves the smoothing adapter for symmetric nodal log-coordinate directions
    Tangent nodal_log_jvp(std::span<const Tangent> directions) const {
        fdapde_strong_assert(directions.size() == w_.size(), std::invalid_argument, "nodal direction size mismatch");
        std::vector<Chart> charts;
        for (const auto& direction : directions) charts.push_back(manifold::internals::cheeger_chart(direction));
        auto result = differentiate_({}, charts, 0);
        fdapde_strong_assert(result.converged(), std::domain_error, "Cheeger nodal derivative solve failed");
        return std::move(result.derivative);
    }
    /// @brief pulls an ambient Frobenius covector back to nodal logs and optionally to local rho with one adjoint solve
    std::vector<Tangent> nodal_log_vjp(const Tangent& covector, double* rho_derivative = nullptr) const {
        fdapde_strong_assert(
          result_.converged() && !result_.detected_ambiguity, std::domain_error, "unresolved Cheeger mean branch");
        for (std::size_t i = 0; i < w_.size(); ++i)
            if (w_[i] == 1.) {
                Tangent zero;
                for (int row = 0; row < N; ++row)
                    for (int col = 0; col <= row; ++col) zero(row, col) = 0;
                std::vector<Tangent> result(w_.size(), zero);
                result[i] = matrix_exp_frechet(manifold::internals::cheeger_matrix<S>(nodes_[i]), covector);
                if (rho_derivative) *rho_derivative = 0;
                return result;
            }
        fdapde_strong_assert(
          lu_.has_value() && lu_->info() == 0 && mean_log_.has_value(), std::domain_error,
          "singular Cheeger mean Hessian");
        const auto b = manifold::internals::cheeger_chart(matrix_exp_frechet(*mean_log_, covector));
        const int n = int(w_.size());
        Vector rhs(n);
        for (int i = 0; i < n; ++i) rhs[i] = 4 * w_[i] * (b.x * fit_.z[i].y - b.y * fit_.z[i].x);
        const Vector adjoint(lu_->solve(rhs));
        Chart v;
        double drho = 0;
        for (int i = 0; i < n; ++i) {
            v.x += adjoint[i] * w_[i] * fit_.z[i].x;
            v.y += adjoint[i] * w_[i] * fit_.z[i].y;
            drho -= 4 * adjoint[i] * w_[i] * fit_.phi[i];
        }
        std::vector<Tangent> result;
        result.reserve(w_.size());
        for (int i = 0; i < n; ++i)
            result.emplace_back(
              manifold::internals::cheeger_matrix<S>(manifold::internals::cheeger_rotate(
                {w_[i] * b.s, w_[i] * (b.x + 4 * v.y - 4 * adjoint[i] * fit_.mean.y),
                 w_[i] * (b.y - 4 * v.x + 4 * adjoint[i] * fit_.mean.x)},
                fit_.phi[i])));
        if (rho_derivative) *rho_derivative = drho;
        return result;
    }
   private:
    /// @brief reuses the lifted Hessian for weight, nodal chart and local metric perturbations
    P1DerivativeResult<Tangent>
    differentiate_(std::span<const double> dw, std::span<const Chart> dx, double drho) const {
        for (const auto& d : dx)
            fdapde_strong_assert(
              std::isfinite(d.s) && std::isfinite(d.x) && std::isfinite(d.y), std::invalid_argument,
              "nonfinite nodal direction");
        if (dw.empty())
            for (std::size_t i = 0; i < w_.size(); ++i)
                if (w_[i] == 1.) {
                    const auto log = manifold::internals::cheeger_matrix<S>(nodes_[i]);
                    const auto delta = manifold::internals::cheeger_matrix<S>(dx.empty() ? Chart {} : dx[i]);
                    return {
                      Tangent(matrix_exp_frechet(log, delta)), 0, 0,
                      manifold::PositiveDefiniteCGStopReason::residual_tolerance};
                }
        fdapde_strong_assert(
          lu_.has_value() && lu_->info() == 0, std::domain_error,
          "Cheeger derivatives require interior weights and a resolved positive Hessian");
        Chart delta;
        std::vector<Chart> dz(w_.size());
        for (std::size_t i = 0; i < w_.size(); ++i) {
            if (!dx.empty()) { dz[i] = manifold::internals::cheeger_rotate(dx[i], -fit_.phi[i]); }
            const double d = dw.empty() ? 0 : dw[i];
            delta.s += d * fit_.z[i].s + w_[i] * dz[i].s;
            delta.x += d * fit_.z[i].x + w_[i] * dz[i].x;
            delta.y += d * fit_.z[i].y + w_[i] * dz[i].y;
        }
        Vector rhs(w_.size());
        for (std::size_t i = 0; i < w_.size(); ++i) {
            const double d = dw.empty() ? 0 : dw[i];
            rhs[i] =
              -8 * w_[i] *
                (delta.y * fit_.z[i].x - delta.x * fit_.z[i].y + fit_.mean.y * dz[i].x - fit_.mean.x * dz[i].y) -
              4 * drho * w_[i] * fit_.phi[i] -
              d * (8 * (fit_.mean.y * fit_.z[i].x - fit_.mean.x * fit_.z[i].y) + 4 * geometry_.rho() * fit_.phi[i]);
        }
        const Vector dphi(lu_->solve(rhs));
        double residual = 0, rhsnorm = 0;
        for (std::size_t i = 0; i < w_.size(); ++i) {
            double r = -rhs[i];
            for (std::size_t j = 0; j < w_.size(); ++j) r += (*hessian_)(i, j) * dphi[j];
            residual += r * r;
            rhsnorm += rhs[i] * rhs[i];
            delta.x += 2 * w_[i] * fit_.z[i].y * dphi[i];
            delta.y -= 2 * w_[i] * fit_.z[i].x * dphi[i];
        }
        residual = std::sqrt(residual);
        const auto log_delta = manifold::internals::cheeger_matrix<S>(delta);
        return {
          Tangent(matrix_exp_frechet(*mean_log_, log_delta)), residual, 1,
          residual <= options_.linear_solve.residual_tolerance * std::sqrt(rhsnorm) ?
            manifold::PositiveDefiniteCGStopReason::residual_tolerance :
            manifold::PositiveDefiniteCGStopReason::numerical_breakdown};
    }
    /// @brief stores a lifted candidate and its analytic stationarity residual
    struct Fit {
        std::vector<double> phi, gradient;
        std::vector<Chart> z;
        Chart mean;
        double cost = 0;
        std::size_t iterations = 0;
    };
    /// @brief maps rotation lifts to the canonical half-turn interval
    static double wrap_(double p) { return std::atan2(std::sin(2 * p), std::cos(2 * p)) / 2; }
    /// @brief pairs dynamic rotation coefficient arrays
    static double dot_(const std::vector<double>& a, const std::vector<double>& b) {
        double s = 0;
        for (std::size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
        return s;
    }
    /// @brief measures the lifted stationarity residual
    static double norm_(const std::vector<double>& a) { return std::sqrt(dot_(a, a)); }
    /// @brief evaluates the lifted objective and its analytic angle gradient
    Fit evaluate_(const std::vector<double>& phi) const {
        Fit f;
        f.phi = phi;
        f.z.reserve(w_.size());
        f.gradient.reserve(w_.size());
        for (std::size_t i = 0; i < w_.size(); ++i) {
            const auto z = manifold::internals::cheeger_rotate(nodes_[i], -phi[i]);
            f.z.push_back(z);
            f.mean.s += w_[i] * z.s;
            f.mean.x += w_[i] * z.x;
            f.mean.y += w_[i] * z.y;
        }
        for (std::size_t i = 0; i < w_.size(); ++i) {
            const auto z = f.z[i];
            const double rho = geometry_.rho();
            f.cost += 2 * w_[i] *
                      (std::pow(z.s - f.mean.s, 2) + std::pow(z.x - f.mean.x, 2) + std::pow(z.y - f.mean.y, 2) +
                       rho * phi[i] * phi[i]);
            f.gradient.push_back(8 * w_[i] * (f.mean.y * z.x - f.mean.x * z.y) + 4 * rho * w_[i] * phi[i]);
        }
        return f;
    }
    /// @brief forms the active rotation Hessian with unit diagonal on unused coordinates
    Dense lifted_hessian_(const Fit& f) const {
        const int n = int(w_.size());
        Dense h(n, n);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) {
                h(i, j) = -16 * w_[i] * w_[j] * (f.z[i].x * f.z[j].x + f.z[i].y * f.z[j].y);
                if (i == j)
                    h(i, j) += w_[i] == 0 ?
                                 1 :
                                 16 * w_[i] * (f.mean.x * f.z[i].x + f.mean.y * f.z[i].y) + 4 * geometry_.rho() * w_[i];
            }
        return h;
    }
    /// @brief solves the globally strongly convex rotation problem by damped Newton
    Fit optimize_convex_() const {
        const int n = int(w_.size());
        std::vector<double> phi(n, 0);
        Fit f = evaluate_(phi);
        for (std::size_t iteration = 0; iteration < options_.mean.solver.max_iterations; ++iteration) {
            const double norm = norm_(f.gradient);
            if (norm <= options_.mean.solver.gradient_tolerance * .1) break;
            const fdapde::PartialPivLU lu(lifted_hessian_(f));
            if (lu.info() != 0) break;
            Vector rhs(n);
            for (int i = 0; i < n; ++i) rhs[i] = -f.gradient[i];
            const Vector direction(lu.solve(rhs));
            double slope = 0;
            for (int i = 0; i < n; ++i) slope += direction[i] * f.gradient[i];
            bool accepted = false;
            double step = 1;
            for (int search = 0; search < 40; ++search) {
                for (int i = 0; i < n; ++i) phi[i] = f.phi[i] + step * direction[i];
                auto next = evaluate_(phi);
                if (
                  next.cost <= f.cost + 1e-4 * step * slope ||
                  (next.cost <= f.cost + 1e-14 * std::max(1., f.cost) && norm_(next.gradient) < norm)) {
                    next.iterations = f.iterations + 1;
                    f = std::move(next);
                    accepted = true;
                    break;
                }
                step /= 2;
            }
            if (!accepted) break;
        }
        return f;
    }
    /// @brief refines one lifted branch by BFGS and local stationarity polishing
    Fit optimize_(std::vector<double> phi) const {
        Fit f = evaluate_(phi);
        const int n = int(w_.size());
        Dense inverse(n, n);
        auto reset = [&] {
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < n; ++j)
                    inverse(i, j) =
                      i == j && w_[i] > 0 ?
                        1 / (4 * w_[i] *
                             (geometry_.rho() + 4 * (nodes_[i].x * nodes_[i].x + nodes_[i].y * nodes_[i].y))) :
                        0;
        };
        reset();
        for (std::size_t iteration = 0; iteration < options_.mean.solver.max_iterations; ++iteration) {
            if (norm_(f.gradient) < options_.mean.solver.gradient_tolerance * .1) break;
            std::vector<double> direction(w_.size(), 0);
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < n; ++j) direction[i] -= inverse(i, j) * f.gradient[j];
            if (dot_(direction, f.gradient) >= 0) {
                reset();
                for (int i = 0; i < n; ++i) direction[i] = -inverse(i, i) * f.gradient[i];
            }
            double step = 1;
            Fit next;
            for (int search = 0; search < 40; ++search) {
                for (int i = 0; i < n; ++i) phi[i] = wrap_(f.phi[i] + step * direction[i]);
                next = evaluate_(phi);
                if (next.cost <= f.cost + 1e-4 * step * dot_(direction, f.gradient)) break;
                step /= 2;
            }
            if (next.cost > f.cost || (next.cost == f.cost && norm_(next.gradient) >= norm_(f.gradient))) break;
            std::vector<double> s(w_.size()), y(w_.size()), hy(w_.size(), 0);
            for (int i = 0; i < n; ++i) {
                s[i] = next.phi[i] - f.phi[i];
                y[i] = next.gradient[i] - f.gradient[i];
            }
            const double sy = dot_(s, y);
            if (sy > 1e-20) {
                for (int i = 0; i < n; ++i)
                    for (int j = 0; j < n; ++j) hy[i] += inverse(i, j) * y[j];
                const double factor = (sy + dot_(y, hy)) / (sy * sy);
                for (int i = 0; i < n; ++i)
                    for (int j = 0; j < n; ++j)
                        inverse(i, j) += factor * s[i] * s[j] - (hy[i] * s[j] + s[i] * hy[j]) / sy;
            } else
                reset();
            next.iterations = f.iterations + 1;
            f = std::move(next);
        }
        // polish stationarity near the cost roundoff floor without leaving the nearby branch
        for (int polish = 0;
             polish < 5 && norm_(f.gradient) > options_.mean.solver.gradient_tolerance * .1 && norm_(f.gradient) < 1e-5;
             ++polish) {
            const Dense h = lifted_hessian_(f);
            const fdapde::PartialPivLU lu(h);
            if (lu.info() != 0) break;
            Vector rhs(n);
            for (int i = 0; i < n; ++i) rhs[i] = -f.gradient[i];
            const Vector delta(lu.solve(rhs));
            double maximum = 0;
            for (int i = 0; i < n; ++i) {
                maximum = std::max(maximum, std::abs(delta[i]));
                phi[i] = wrap_(f.phi[i] + delta[i]);
            }
            if (maximum > 0.01) break;
            auto next = evaluate_(phi);
            if (norm_(next.gradient) >= norm_(f.gradient) || next.cost > f.cost + 1e-14 * std::max(1., f.cost)) break;
            next.iterations = f.iterations + 1;
            f = std::move(next);
        }
        return f;
    }
    Geometry geometry_;
    Nodes binding_;
    P1GeodesicLinearizationOptions options_;
    std::vector<double> w_, rho_nodes_;
    double weight_total_ = 1;
    std::optional<Dense> hessian_;
    std::optional<PartialPivLU<Dense>> lu_;
    std::optional<SymmetricMatrix<S, N, Cache::Spectral>> mean_log_;
    std::vector<Chart> nodes_;
    Fit fit_;
    CheegerP1ValueResult<Point> result_ {
      {Geometry::from_chart({}), {}}
    };
};

/// @brief evaluates a native batch at a fixed local rho without preparing derivative factorizations
template <typename S, int N, Usage U, typename Point_, typename Nodes>
auto p1_geodesic_value(
  const manifold::CheegerLogEuclideanSPDGeometry<S, N, U, Point_>& geometry, const Nodes& nodes,
  std::span<const double> weights, const manifold::WeightedKarcherMeanOptions& options = {}) {
    P1GeodesicLinearizationOptions combined;
    combined.mean = options;
    P1GeodesicLinearization<manifold::CheegerLogEuclideanSPDGeometry<S, N, U, Point_>, const Nodes&> fit(
      geometry, nodes, weights, combined, {}, false);
    return fit.result();
}
}   // namespace fdapde::gfe
#endif
