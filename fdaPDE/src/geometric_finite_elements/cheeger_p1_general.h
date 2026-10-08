// SPDX-License-Identifier: GPL-3.0-or-later
#ifndef __FDAPDE_CHEEGER_P1_GENERAL_H__
#define __FDAPDE_CHEEGER_P1_GENERAL_H__
#include "header_check.h"
namespace fdapde::gfe {
/// @brief prepares general C-LE means and their implicit derivatives on the selected rotation branch
template <typename S, int N, Usage U, typename Point_, typename Nodes>
    requires(N != 2)
class P1GeodesicLinearization<manifold::CheegerLogEuclideanSPDGeometry<S, N, U, Point_>, Nodes> {
   public:
    using Geometry = manifold::CheegerLogEuclideanSPDGeometry<S, N, U, Point_>;
    using Point = typename Geometry::Point;
    using Tangent = typename Geometry::Tangent;
    using Lift = manifold::internals::CheegerLift<S, N>;
    using Vec = typename Lift::Vec;
    /// @brief snapshots nodal logarithms and factors a resolved active lift once
    P1GeodesicLinearization(
      const Geometry& geometry, Nodes nodes, std::span<const double> weights,
      const P1GeodesicLinearizationOptions& options = {}, std::span<const double> rho_nodes = {},
      bool derivatives = true) :
        geometry_(geometry),
        binding_(std::forward<Nodes>(nodes)),
        w_(weights.begin(), weights.end()),
        rho_(rho_nodes.begin(), rho_nodes.end()),
        options_(options) {
        const auto vertex = internals::validate_p1_data(binding_.size(), weights);
        fdapde_strong_assert(
          options.mean.solver.gradient_tolerance > 0 && std::isfinite(options.mean.solver.gradient_tolerance) &&
            options.linear_solve.residual_tolerance > 0 && std::isfinite(options.linear_solve.residual_tolerance),
          std::invalid_argument, "C-LE tolerances must be finite and positive");
        fdapde_strong_assert(rho_.empty() || rho_.size() == w_.size(), std::invalid_argument, "rho size mismatch");
        total_ = 0;
        for (double w : w_) total_ += w;
        double rho = 0;
        for (std::size_t i = 0; i < w_.size(); ++i) {
            fdapde_strong_assert(
              binding_[i].rows() == geometry.order() && binding_[i].cols() == geometry.order(), std::invalid_argument,
              "C-LE node order mismatch");
            w_[i] /= total_;
            if (w_[i] > 0) active_.push_back(i);
            if (!rho_.empty()) {
                geometry.with_rho(rho_[i]);
                rho += w_[i] * rho_[i];
            }
        }
        if (!rho_.empty()) geometry_ = geometry.with_rho(rho);
        MatrixBatch<Tangent> charts(active_.size(), geometry.order(), geometry.order());
        std::vector<double> active_weights;
        for (std::size_t i = 0; i < active_.size(); ++i) {
            charts[i] = matrix_log(binding_[active_[i]]);
            active_weights.push_back(w_[active_[i]]);
        }
        lift_.emplace(std::move(charts), std::move(active_weights), geometry_.rho());
        state_.emplace(lift_->solve(options.mean.solver.gradient_tolerance, options.mean.solver.max_iterations));
        result_.emplace(
          CheegerP1ValueResult<Point> {
            {vertex ? Point(binding_[*vertex]) : Point(matrix_exp(state_->mean)), w_}
        });
        result_->stationarity_norm = state_->gradient.norm();
        result_->stop_reason = lift_->converged() ? manifold::BarycenterStopReason::stationarity_tolerance :
                                                    manifold::BarycenterStopReason::max_iterations;
        result_->iterations = state_->iterations;
        result_->detected_ambiguity = lift_->ambiguous();
        result_->objective = state_->cost;
        if (vertex) result_->uniqueness = manifold::BarycenterUniqueness::globally_unique;
        if (derivatives && result_->converged() && !result_->detected_ambiguity) {
            try {
                lift_->prepare(*state_);
                ready_ = true;
            } catch (const std::domain_error&) { }
        }
        mean_log_.emplace(state_->mean);
    }
    /// @brief borrows the selected mean and solver diagnostics
    const auto& result() const& { return *result_; }
    /// @brief forbids references to temporary diagnostics
    void result() const&& = delete;
    /// @brief borrows the metric at the interpolated rho value
    const Geometry& geometry() const& { return geometry_; }
    /// @brief forbids a geometry reference escaping a temporary linearization
    void geometry() const&& = delete;
    /// @brief differentiates the local rho parameter
    P1DerivativeResult<Tangent> rho_jvp(double direction = 1) const { return derivative_({}, {}, direction); }
    /// @brief differentiates rho coefficients through the P1 weights
    P1DerivativeResult<Tangent> nodal_rho_jvp(std::span<const double> direction) const {
        fdapde_strong_assert(direction.size() == w_.size(), std::invalid_argument, "rho direction count mismatch");
        double d = 0;
        for (std::size_t i = 0; i < w_.size(); ++i) {
            fdapde_strong_assert(std::isfinite(direction[i]), std::invalid_argument, "nonfinite rho direction");
            d += w_[i] * direction[i];
        }
        return rho_jvp(d);
    }
    /// @brief differentiates normalized weights including spatial rho variation
    P1DerivativeResult<Tangent> weight_jvp(std::span<const double> direction) const {
        fdapde_strong_assert(
          direction.size() == w_.size() && active_.size() == w_.size(), std::invalid_argument,
          "weight derivatives require interior weights");
        double sum = 0, scale = 0;
        for (double d : direction) {
            fdapde_strong_assert(std::isfinite(d), std::invalid_argument, "nonfinite weight direction");
            sum += d;
            scale += std::abs(d);
        }
        fdapde_strong_assert(
          std::isfinite(sum) && std::isfinite(scale) &&
            std::abs(sum) <= p1_weight_sum_tolerance(w_.size()) * std::max(1., scale),
          std::invalid_argument, "weight direction must sum to zero");
        std::vector<double> dw(w_.size());
        double drho = 0;
        for (std::size_t i = 0; i < w_.size(); ++i) {
            dw[i] = (direction[i] - w_[i] * sum) / total_;
            if (!rho_.empty()) drho += dw[i] * rho_[i];
        }
        return derivative_(dw, {}, drho);
    }
    /// @brief differentiates nodal logarithmic coordinates at fixed weights and rho
    Tangent nodal_log_jvp(std::span<const Tangent> direction) const {
        fdapde_strong_assert(direction.size() == w_.size(), std::invalid_argument, "nodal direction count mismatch");
        for (const auto& d : direction) check_tangent_(d);
        return derivative_({}, direction, 0).derivative;
    }
    /// @brief converts ambient SPD perturbations into nodal log directions
    P1DerivativeResult<Tangent> nodal_jvp(std::span<const Tangent> direction) const {
        fdapde_strong_assert(direction.size() == w_.size(), std::invalid_argument, "nodal direction count mismatch");
        std::vector<Tangent> dx;
        for (std::size_t i = 0; i < w_.size(); ++i) {
            check_tangent_(direction[i]);
            dx.emplace_back(matrix_log_frechet(binding_[i], direction[i]));
        }
        return derivative_({}, dx, 0);
    }
    /// @brief computes nodal log covectors and a local rho derivative with one adjoint solve
    auto nodal_log_rho_vjp(const Tangent& ambient_dual) const {
        require_ready_();
        check_tangent_(ambient_dual);
        const Tangent c(matrix_exp_frechet(*mean_log_, ambient_dual));
        Vec rhs(state_->gradient.rows());
        const int k = lift_->skew_size();
        for (std::size_t i = 0; i < active_.size(); ++i)
            for (int a = 0; a < k; ++a) {
                const auto b = lift_->basis(a);
                rhs[int(i) * k + a] =
                  S(w_[active_[i]] * manifold::internals::cheeger_inner(c, state_->z[i] * b - b * state_->z[i]));
            }
        const auto solution = lift_->solve_linear(rhs, options_.linear_solve.residual_tolerance);
        Tangent coupled = lift_->zero();
        double drho = 0;
        for (std::size_t i = 0; i < active_.size(); ++i) {
            const auto l = lift_->tangent(solution, i);
            coupled +=
              Tangent((S(2 * w_[active_[i]]) * (l * state_->z[i] - state_->z[i] * l)).template as_symmetric<Lower>());
            drho -= 2 * w_[active_[i]] * manifold::internals::cheeger_inner(l, state_->omega[i]);
        }
        std::vector<Tangent> out(w_.size(), lift_->zero());
        for (std::size_t i = 0; i < active_.size(); ++i) {
            const auto l = lift_->tangent(solution, i);
            const auto q = state_->rotations[i];
            const Tangent dual((S(w_[active_[i]]) * (c - coupled - S(2) * (state_->mean * l - l * state_->mean)))
                                 .template as_symmetric<Lower>());
            out[active_[i]] = Tangent((q * dual * q.transpose()).template as_symmetric<Lower>());
        }
        return std::pair {std::move(out), drho};
    }
   private:
    /// @brief checks user-supplied logarithmic directions and ambient covectors
    void check_tangent_(const Tangent& d) const {
        fdapde_strong_assert(
          d.rows() == geometry_.order() && d.cols() == geometry_.order(), std::invalid_argument,
          "C-LE tangent order mismatch");
        for (int i = 0; i < d.rows(); ++i)
            for (int j = 0; j <= i; ++j)
                fdapde_strong_assert(std::isfinite(double(d(i, j))), std::invalid_argument, "nonfinite C-LE tangent");
    }
    /// @brief rejects nonstationary, ambiguous or singular branches before differentiating
    void require_ready_() const {
        fdapde_strong_assert(ready_, std::domain_error, "C-LE derivative requires a resolved positive lifted Hessian");
    }
    /// @brief applies the implicit function theorem in orthonormal rotation coordinates
    P1DerivativeResult<Tangent>
    derivative_(std::span<const double> dw, std::span<const Tangent> dx, double drho) const {
        require_ready_();
        fdapde_strong_assert(std::isfinite(drho), std::invalid_argument, "nonfinite rho direction");
        Tangent delta = lift_->zero();
        std::vector<Tangent> dz(active_.size(), lift_->zero());
        for (std::size_t i = 0; i < active_.size(); ++i) {
            const auto id = active_[i];
            const auto q = state_->rotations[i];
            if (!dx.empty()) dz[i] = Tangent((q.transpose() * dx[id] * q).template as_symmetric<Lower>());
            delta += S(w_[id]) * dz[i];
            if (!dw.empty()) delta += S(dw[id]) * state_->z[i];
        }
        Vec rhs(state_->gradient.rows());
        const int k = lift_->skew_size();
        for (std::size_t i = 0; i < active_.size(); ++i) {
            const auto id = active_[i];
            const double d = dw.empty() ? 0 : dw[id];
            const Matrix<S, N, N> dg(
              S(2 * w_[id]) * (delta * state_->z[i] - state_->z[i] * delta + state_->mean * dz[i] -
                               dz[i] * state_->mean + S(drho) * state_->omega[i]) +
              S(2 * d) *
                (state_->mean * state_->z[i] - state_->z[i] * state_->mean + S(geometry_.rho()) * state_->omega[i]));
            int a = int(i) * k;
            for (int r = 0; r < geometry_.order(); ++r)
                for (int c = r + 1; c < geometry_.order(); ++c) rhs[a++] = -S(std::sqrt(2.)) * dg(r, c);
        }
        const auto solution = lift_->solve_linear(rhs, options_.linear_solve.residual_tolerance);
        for (std::size_t i = 0; i < active_.size(); ++i) {
            const auto omega = lift_->tangent(solution, i);
            delta += Tangent(
              (S(w_[active_[i]]) * (state_->z[i] * omega - omega * state_->z[i])).template as_symmetric<Lower>());
        }
        return {
          Tangent(matrix_exp_frechet(*mean_log_, delta)), lift_->residual(solution, rhs), 1,
          manifold::PositiveDefiniteCGStopReason::residual_tolerance};
    }
    Geometry geometry_;
    Nodes binding_;
    std::vector<double> w_, rho_;
    std::vector<std::size_t> active_;
    P1GeodesicLinearizationOptions options_;
    double total_ = 1;
    bool ready_ = false;
    std::optional<Lift> lift_;
    std::optional<typename Lift::State> state_;
    std::optional<SymmetricMatrix<S, N, Cache::Spectral>> mean_log_;
    std::optional<CheegerP1ValueResult<Point>> result_;
};
}   // namespace fdapde::gfe
#endif
