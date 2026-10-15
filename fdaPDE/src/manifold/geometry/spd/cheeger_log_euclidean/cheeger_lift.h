// SPDX-License-Identifier: GPL-3.0-or-later
#ifndef __FDAPDE_CHEEGER_LIFT_H__
#define __FDAPDE_CHEEGER_LIFT_H__
#include "../../../header_check.h"

namespace fdapde::manifold::internals {
/// @brief pairs full matrix coefficients without assuming a storage layout
inline double cheeger_inner(const auto& a, const auto& b) {
    double value = 0;
    for (int i = 0; i < a.rows(); ++i)
        for (int j = 0; j < a.cols(); ++j) value += double(a(i, j)) * double(b(i, j));
    return value;
}
/// @brief copies cached eigenvectors when available and orients the frame as a proper rotation
template <typename S, int N, typename X> Matrix<S, N, N> cheeger_eigenframe(const X& chart) {
    const EVD evd(chart);
    Matrix<S, N, N> q(evd.eigenvectors());
    if (PartialPivLU(q).determinant() < 0)
        for (int r = 0; r < chart.rows(); ++r) q(r, 0) = -q(r, 0);
    return q;
}
/// @brief specializes rotation differentials in dimensions two and three and reuses SO(n) otherwise
template <typename S, int N> class CheegerRotationDifferential {
   public:
    using Tangent = SkewSymmetricMatrix<S, N>;
    /// @brief prepares scalar Rodrigues coefficients or the general adjoint spectrum
    explicit CheegerRotationDifferential(const Tangent& omega) : omega_(omega), squared_(cheeger_inner(omega, omega)) {
        if (omega.rows() > 3) general_.emplace(omega);
        if (omega.rows() == 3) {
            if (squared_ < .02) {
                radial_ = 1. / 24 +
                          squared_ *
                            (1. / 2880 + squared_ * (1. / 241920 + squared_ * (1. / 19353600 + squared_ / 1532805120)));
                radial_derivative_ =
                  1. / 2880 + squared_ * (1. / 120960 + squared_ * (1. / 6451200 + squared_ / 383201280));
                coefficient_ = 1 - squared_ * radial_;
                coefficient_derivative_ = -radial_ - squared_ * radial_derivative_;
            } else {
                const double half = std::sqrt(squared_ / 2) / 2;
                const double sine = std::sin(half);
                coefficient_ = half / std::tan(half);
                radial_ = (1 - coefficient_) / squared_;
                coefficient_derivative_ = (coefficient_ - half * half / (sine * sine)) / (2 * squared_);
                radial_derivative_ = -(coefficient_derivative_ + radial_) / squared_;
            }
        }
    }
    /// @brief applies the bi-invariant squared-distance Hessian in body coordinates
    Tangent hessian_action(const Tangent& direction) const {
        if (general_) return general_->hessian_action(direction);
        const double projection = radial_ * cheeger_inner(omega_, direction);
        return Tangent((S(coefficient_) * direction + S(projection) * omega_).template as_skew_symmetric<Lower>());
    }
    /// @brief differentiates the group logarithm along a right body perturbation
    Tangent target_action(const Tangent& direction) const {
        if (general_) return general_->target_action(direction);
        const auto h = hessian_action(direction);
        return Tangent((h + S(.5) * (omega_ * direction - direction * omega_)).template as_skew_symmetric<Lower>());
    }
    /// @brief differentiates the Hessian action with respect to ordinary logarithm coefficients
    Tangent hessian_action_jvp(const Tangent& direction, const Tangent& omega_direction) const {
        fdapde_strong_assert(
          !general_, std::logic_error, "Cheeger rotation Hessian derivative requires order at most three");
        const double squared_direction = 2 * cheeger_inner(omega_, omega_direction);
        const double projection = cheeger_inner(omega_, direction);
        const double projection_direction = cheeger_inner(omega_direction, direction);
        return Tangent(
          (S(coefficient_derivative_ * squared_direction) * direction +
           S(radial_derivative_ * squared_direction * projection + radial_ * projection_direction) * omega_ +
           S(radial_ * projection) * omega_direction)
            .template as_skew_symmetric<Lower>());
    }
    /// @brief differentiates the target logarithm action with respect to ordinary logarithm coefficients
    Tangent target_action_jvp(const Tangent& direction, const Tangent& omega_direction) const {
        const auto h = hessian_action_jvp(direction, omega_direction);
        return Tangent((h + S(.5) * (omega_direction * direction - direction * omega_direction))
                         .template as_skew_symmetric<Lower>());
    }
   private:
    Tangent omega_;
    double squared_, coefficient_ = 1, radial_ = 0, coefficient_derivative_ = 0, radial_derivative_ = 0;
    std::optional<SOLogDifferential<S, N>> general_;
};
/// @brief minimizes rotational lifts and retains their analytic Hessian for implicit differentiation
/// @details pair searches use one fixed chart; mean searches eliminate the weighted symmetric chart
/// @details multistart is a local search and does not certify global optimality
// ponytail: dense lifted Hessians scale quadratically in nodes*n*(n-1)/2, use matrix-free solves for large orders
template <typename S, int N, int Nodes = Dynamic> class CheegerLift {
   public:
    using Sym = SymmetricMatrix<S, N>;
    using Skew = SkewSymmetricMatrix<S, N>;
    using Rotation = RotationMatrix<S, N, RotationCache::Log>;
    using Batch = MatrixBatch<Rotation>;
    static constexpr int Coordinates = N == Dynamic || Nodes == Dynamic ? Dynamic : Nodes * N * (N - 1) / 2;
    using Dense = Matrix<S, Coordinates, Coordinates>;
    using Vec = Vector<S, Coordinates>;
    using Eigenframe = Matrix<S, N, N>;
    /// @brief stores one evaluation's rotations, aligned charts and stationarity residual
    struct State {
        Batch rotations;
        std::vector<Sym> z;
        std::vector<Skew> omega;
        Sym mean;
        Vec gradient;
        double cost = 0;
        std::size_t iterations = 0;
    };
    /// @brief retains a selected lift's directional changes without repeating its rotation search
    struct Variation {
        std::vector<Sym> z;
        std::vector<Skew> omega;
        Sym mean;
        Vec rotations;
        Dense hessian;
        double rho = 0;
    };
    /// @brief binds a copied chart batch, positive weights and one local metric parameter
    CheegerLift(MatrixBatch<Sym> charts, std::vector<double> weights, double rho, std::optional<Sym> base = {}) :
        charts_(std::move(charts)),
        w_(std::move(weights)),
        rho_(rho),
        base_(std::move(base)),
        n_(charts_.rows()),
        k_(n_ * (n_ - 1) / 2) {
        fdapde_assert(std::isfinite(rho_) && rho_ > 0, std::invalid_argument, "C-LE requires positive finite rho");
        fdapde_assert(charts_.size() == w_.size() && !w_.empty(), std::invalid_argument, "invalid lifted nodes");
        fdapde_assert(
          Nodes == Dynamic || w_.size() == std::size_t(Nodes), std::invalid_argument,
          "invalid static lifted node count");
        fdapde_strong_assert(
          std::int64_t(k_) * w_.size() <= std::sqrt(double(std::numeric_limits<int>::max())), std::length_error,
          "C-LE lifted workspace exceeds native index range");
        for (double w : w_)
            fdapde_assert(std::isfinite(w) && w > 0, std::invalid_argument, "active lift weights must be positive");
    }
    /// @brief builds an orthonormal skew direction by its plane index
    Skew basis(int axis) const {
        Skew b;
        if constexpr (N == Dynamic) b.resize(n_, n_);
        for (int r = 0; r < n_; ++r)
            for (int c = r + 1; c < n_; ++c) b(r, c) = 0;
        int p = 0;
        for (int i = 0; i < n_; ++i)
            for (int j = i + 1; j < n_; ++j)
                if (p++ == axis) b(i, j) = S(1 / std::sqrt(2.));
        return b;
    }
    /// @brief reconstructs one block of body coordinates
    Skew tangent(const Vec& v, std::size_t block) const {
        Skew b;
        if constexpr (N == Dynamic) b.resize(n_, n_);
        for (int r = 0; r < n_; ++r)
            for (int c = r + 1; c < n_; ++c) b(r, c) = 0;
        int p = int(block) * k_;
        for (int i = 0; i < n_; ++i)
            for (int j = i + 1; j < n_; ++j) b(i, j) = v[p++] / S(std::sqrt(2.));
        return b;
    }
    /// @brief creates a zero symmetric chart of the bound order
    Sym zero() const {
        Sym x;
        if constexpr (N == Dynamic) x.resize(n_, n_);
        for (int r = 0; r < n_; ++r)
            for (int c = 0; c <= r; ++c) x(r, c) = 0;
        return x;
    }
    /// @brief evaluates the eliminated lift and its analytic body gradient
    State evaluate(const Batch& rotations) const {
        State f {rotations, {}, {}, zero(), Vec(int(w_.size()) * k_)};
        f.z.reserve(w_.size());
        f.omega.reserve(w_.size());
        for (std::size_t i = 0; i < w_.size(); ++i) {
            const Matrix<S, N, N> z(rotations[i].transpose() * charts_[i] * rotations[i]);
            f.z.emplace_back(z.template as_symmetric<Lower>());
            f.omega.emplace_back(rotation_log(rotations[i]));
            if (!base_) f.mean += S(w_[i]) * f.z.back();
        }
        if (base_) f.mean = *base_;
        for (std::size_t i = 0; i < w_.size(); ++i) {
            const Sym residual(f.z[i] - f.mean);
            f.cost += w_[i] * (cheeger_inner(residual, residual) + rho_ * cheeger_inner(f.omega[i], f.omega[i]));
            const Matrix<S, N, N> g(S(2 * w_[i]) * (f.mean * f.z[i] - f.z[i] * f.mean + S(rho_) * f.omega[i]));
            int p = int(i) * k_;
            for (int r = 0; r < n_; ++r)
                for (int c = r + 1; c < n_; ++c) f.gradient[p++] = S(std::sqrt(2.)) * g(r, c);
        }
        return f;
    }
    /// @brief assembles the covariant rotation Hessian including coupling through the eliminated mean
    Dense hessian(const State& f) const {
        const int size = int(w_.size()) * k_;
        Dense h;
        if constexpr (Coordinates == Dynamic) h.resize(size, size);
        h.set_zero();
        std::vector<CheegerRotationDifferential<S, N>> differentials;
        differentials.reserve(w_.size());
        for (const auto& omega : f.omega) differentials.emplace_back(omega);
        for (std::size_t j = 0; j < w_.size(); ++j) {
            const Matrix<S, N, N> g(S(2 * w_[j]) * (f.mean * f.z[j] - f.z[j] * f.mean));
            for (int a = 0; a < k_; ++a) {
                const Skew b = basis(a);
                const Sym dz((f.z[j] * b - b * f.z[j]).template as_symmetric<Lower>());
                const Sym dm(base_ ? zero() : Sym(S(w_[j]) * dz));
                const auto curvature = differentials[j].hessian_action(b);
                const auto diagonal =
                  S(2 * w_[j]) * (f.mean * dz - dz * f.mean + S(rho_) * curvature) + S(.5) * (b * g - g * b);
                for (std::size_t i = 0; i < w_.size(); ++i) {
                    const auto coupling = S(2 * w_[i]) * (dm * f.z[i] - f.z[i] * dm);
                    // only skew coordinates enter the Hessian; the other matrix entries are discarded
                    int p = int(i) * k_;
                    for (int r = 0; r < n_; ++r)
                        for (int c = r + 1; c < n_; ++c) {
                            S value = coupling(r, c);
                            if (i == j) value += diagonal(r, c);
                            h(p++, int(j) * k_ + a) = S(std::sqrt(2.)) * value;
                        }
                }
            }
        }
        return h;
    }
    /// @brief differentiates a resolved rotation branch in nodal log coordinates and its local metric parameter
    Variation differentiate(
      const State& state, std::span<const Sym> direction, double rho_direction = 0,
      std::optional<Sym> base_direction = {}, double tolerance = 1e-9) const
        requires(N == 2 || N == 3)
    {
        fdapde_assert(direction.size() == w_.size(), std::invalid_argument, "lift direction count mismatch");
        Variation result {{}, {}, base_direction.value_or(zero()), Vec(int(w_.size()) * k_), {}, rho_direction};
        result.z.reserve(w_.size());
        result.omega.reserve(w_.size());
        for (std::size_t i = 0; i < w_.size(); ++i) {
            const auto q = state.rotations[i];
            result.z.emplace_back((q.transpose() * direction[i] * q).template as_symmetric<Lower>());
            if (!base_) result.mean += S(w_[i]) * result.z.back();
        }
        Vec rhs(int(w_.size()) * k_);
        for (std::size_t i = 0; i < w_.size(); ++i) {
            const Matrix<S, N, N> change(
              S(-2 * w_[i]) * (result.mean * state.z[i] - state.z[i] * result.mean + state.mean * result.z[i] -
                               result.z[i] * state.mean + S(rho_direction) * state.omega[i]));
            int p = int(i) * k_;
            for (int r = 0; r < n_; ++r)
                for (int c = r + 1; c < n_; ++c) rhs[p++] = S(std::sqrt(2.)) * change(r, c);
        }
        result.rotations = solve_linear(rhs, tolerance);
        for (std::size_t i = 0; i < w_.size(); ++i) {
            const auto body = tangent(result.rotations, i);
            const Sym change((state.z[i] * body - body * state.z[i]).template as_symmetric<Lower>());
            result.z[i] += change;
            if (!base_) result.mean += S(w_[i]) * change;
            result.omega.push_back(CheegerRotationDifferential<S, N>(state.omega[i]).target_action(body));
        }
        result.hessian = hessian_jvp(state, result);
        return result;
    }
    /// @brief differentiates the analytic body Hessian along a resolved lift variation
    Dense hessian_jvp(const State& state, const Variation& direction) const
        requires(N == 2 || N == 3)
    {
        const int size = int(w_.size()) * k_;
        Dense result;
        if constexpr (Coordinates == Dynamic) result.resize(size, size);
        result.set_zero();
        for (std::size_t j = 0; j < w_.size(); ++j)
            for (int a = 0; a < k_; ++a) {
                const auto b = basis(a);
                const Sym dz((state.z[j] * b - b * state.z[j]).template as_symmetric<Lower>());
                const Sym ddz((direction.z[j] * b - b * direction.z[j]).template as_symmetric<Lower>());
                const Sym dm(base_ ? zero() : Sym(S(w_[j]) * dz));
                const Sym ddm(base_ ? zero() : Sym(S(w_[j]) * ddz));
                for (std::size_t i = 0; i < w_.size(); ++i) {
                    Matrix<S, N, N> change(
                      S(2 * w_[i]) * (ddm * state.z[i] - state.z[i] * ddm + dm * direction.z[i] - direction.z[i] * dm));
                    if (i == j) {
                        const CheegerRotationDifferential<S, N> differential(state.omega[i]);
                        const auto curvature = differential.hessian_action(b);
                        const auto curvature_change = differential.hessian_action_jvp(b, direction.omega[i]);
                        const Matrix<S, N, N> dg(
                          S(2 * w_[i]) * (direction.mean * state.z[i] - state.z[i] * direction.mean +
                                          state.mean * direction.z[i] - direction.z[i] * state.mean));
                        change += S(2 * w_[i]) *
                                    (direction.mean * dz - dz * direction.mean + state.mean * ddz - ddz * state.mean +
                                     S(direction.rho) * curvature + S(rho_) * curvature_change) +
                                  S(.5) * (b * dg - dg * b);
                    }
                    int p = int(i) * k_;
                    for (int r = 0; r < n_; ++r)
                        for (int c = r + 1; c < n_; ++c) result(p++, int(j) * k_ + a) = S(std::sqrt(2.)) * change(r, c);
                }
            }
        return result;
    }
    /// @brief supplies the existing trust-region solver with product rotations and flat body coordinates
    struct Product {
        using Point = Batch;
        using Tangent = Vec;
        const CheegerLift* lift;
        /// @brief returns the number of active rotation coordinates
        std::size_t dimension() const { return lift->w_.size() * lift->k_; }
        /// @brief pairs orthonormal body coordinates
        double inner_product(const Point&, const Vec& a, const Vec& b) const { return cheeger_inner(a, b); }
        /// @brief measures a product body tangent
        double norm(const Point&, const Vec& a) const { return a.norm(); }
        /// @brief preserves body coordinates
        Vec project(const Point&, const Vec& a) const { return a; }
        /// @brief constructs the zero product direction
        Vec zero_tangent(const Point&) const {
            Vec v(static_cast<int>(dimension()));
            v.set_zero();
            return v;
        }
        /// @brief combines body directions in their common product tangent space
        Vec linear_combination(const Point&, double a, const Vec& x, double b, const Vec& y) const {
            return Vec(S(a) * x + S(b) * y);
        }
        /// @brief retracts each rotation with the native skew exponential
        Point retract(const Point& p, const Vec& v, double step) const {
            return fdapde::internals::generate_matrix_batch<Rotation>(
              p.size(), lift->n_, lift->n_, execution_seq, [&](std::size_t i) {
                  const auto point = p[i];
                  const auto delta = rotation_exp(lift->tangent(v, i), step);
                  return Rotation(point * delta);
              });
        }
    };
    /// @brief exposes a lifted objective through the existing optimizer workspace contract
    struct Problem {
        /// @brief retains the current candidate and lazily assembled Hessian
        struct Workspace {
            std::optional<State> state;
            std::optional<Dense> hessian;
        };
        const CheegerLift* lift;
        /// @brief evaluates one candidate and retains its shared data
        double cost(const Batch& p, Workspace& w) {
            try {
                w.state = lift->evaluate(p);
                return w.state->cost;
            } catch (const std::domain_error&) { return std::numeric_limits<double>::infinity(); }
        }
        /// @brief reuses the candidate's body gradient
        Vec grad(const Batch& p, Workspace& w) {
            if (!w.state) w.state = lift->evaluate(p);
            return w.state->gradient;
        }
        /// @brief reuses the analytic candidate Hessian across subproblem iterations
        Vec hess(const Batch& p, const Vec& v, Workspace& w) {
            if (!w.state) w.state = lift->evaluate(p);
            if (!w.hessian) w.hessian = lift->hessian(*w.state);
            return Vec(*w.hessian * v);
        }
    };
    /// @brief solves several deterministic local starts and retains competing stationary candidates
    State solve(double tolerance = 1e-10, std::size_t iterations = 200) {
        return solve({}, nullptr, tolerance, iterations);
    }
    /// @brief reuses prepared oriented eigenframes without changing the local starts or solver certificates
    State solve(
      std::span<const Eigenframe> eigenframes, const Eigenframe* base_eigenframe, double tolerance = 1e-10,
      std::size_t iterations = 200) {
        if constexpr (N == 3 && Nodes == Dynamic) {
            switch (w_.size()) {
            case 1:
                return solve_fixed_<1>(eigenframes, base_eigenframe, tolerance, iterations);
            case 2:
                return solve_fixed_<2>(eigenframes, base_eigenframe, tolerance, iterations);
            case 3:
                return solve_fixed_<3>(eigenframes, base_eigenframe, tolerance, iterations);
            case 4:
                return solve_fixed_<4>(eigenframes, base_eigenframe, tolerance, iterations);
            }
        }
        Batch initial(w_.size(), n_, n_);
        if (!base_ && w_.size() == 1) {
            converged_ = true;
            ambiguous_ = false;
            return evaluate(initial);
        }
        std::vector<Batch> starts;
        starts.reserve(1 + (base_ ? 1 : w_.size()) + 2 * k_);
        starts.push_back(initial);
        // seed whole eigenframe alignments as well as individual rotation planes
        std::vector<Eigenframe> prepared;
        if (eigenframes.empty()) {
            prepared.reserve(w_.size());
            for (std::size_t i = 0; i < w_.size(); ++i) prepared.push_back(cheeger_eigenframe<S, N>(charts_[i]));
            eigenframes = prepared;
        }
        fdapde_assert(eigenframes.size() == w_.size(), std::invalid_argument, "invalid lifted eigenframe count");
        std::span<const Eigenframe> references = eigenframes;
        std::optional<Eigenframe> prepared_base;
        if (base_) {
            if (!base_eigenframe) {
                prepared_base.emplace(cheeger_eigenframe<S, N>(*base_));
                base_eigenframe = &*prepared_base;
            }
            references = {base_eigenframe, 1};
        }
        for (const auto& reference : references) {
            Batch seed(initial);
            bool regular = true;
            for (std::size_t i = 0; i < w_.size(); ++i) {
                seed[i] = eigenframes[i] * reference.transpose();
                try {
                    rotation_log(seed[i]);
                } catch (const std::domain_error&) { regular = false; }
            }
            if (regular) starts.push_back(std::move(seed));
        }
        // pairwise plane starts retain both senses of a large rotational alignment
        for (int axis = 0; axis < k_; ++axis)
            for (double sign : {-1., 1.}) {
                Batch seed(initial);
                for (std::size_t i = 0; i < w_.size(); ++i)
                    seed[i] = rotation_exp(basis(axis), sign * (i % 2 ? -1. : 1.) * std::numbers::pi / std::sqrt(2.));
                starts.push_back(std::move(seed));
            }
        TrustRegionOptions options;
        options.max_iterations = iterations;
        options.gradient_tolerance = tolerance;
        options.initial_radius = .5;
        options.subproblem.residual_tolerance = 1e-8;
        Problem problem {this};
        Product geometry {this};
        std::vector<State> candidates;
        candidates.reserve(starts.size());
        for (const auto& start : starts) {
            auto result = RiemannianTrustRegion(options).optimize(problem, geometry, start);
            if (std::isfinite(result.cost)) {
                auto candidate = evaluate(result.point);
                candidate.iterations = result.iterations;
                // polish analytic stationarity when objective decreases fall below floating-point resolution
                for (int polish = 0;
                     polish < 6 && candidate.gradient.norm() > tolerance && candidate.gradient.norm() < 1e-4;
                     ++polish) {
                    const auto h = hessian(candidate);
                    const PartialPivLU lu(equilibrated_hessian_(h));
                    if (lu.info() != 0) break;
                    const Vec rhs(S(-1) * candidate.gradient), step(equilibrated_solve_(lu, rhs));
                    if (step.norm() > .01) break;
                    const auto point = geometry.retract(candidate.rotations, step, 1);
                    auto next = evaluate(point);
                    if (
                      next.cost >
                        candidate.cost + 32 * std::numeric_limits<S>::epsilon() * std::max(1., candidate.cost) ||
                      next.gradient.norm() >= candidate.gradient.norm())
                        break;
                    next.iterations = candidate.iterations + 1;
                    candidate = std::move(next);
                }
                candidates.push_back(std::move(candidate));
            }
        }
        fdapde_strong_assert(!candidates.empty(), std::domain_error, "C-LE alignment has no finite candidate");
        std::sort(candidates.begin(), candidates.end(), [](const State& a, const State& b) { return a.cost < b.cost; });
        auto best = candidates.begin();
        // prefer a stationary candidate when its cost is indistinguishable at machine precision
        const double cost_floor = 32 * std::numeric_limits<S>::epsilon() * std::max(1., best->cost);
        for (auto candidate = candidates.begin(); candidate != candidates.end(); ++candidate)
            if (
              candidate->cost - candidates.front().cost <= cost_floor &&
              candidate->gradient.norm() < best->gradient.norm())
                best = candidate;
        converged_ = best->gradient.norm() <= tolerance;
        ambiguous_ = false;
        for (const auto& c : candidates)
            if (c.gradient.norm() <= tolerance && c.cost - best->cost <= 1e-10 * std::max(1., best->cost)) {
                double separation = 0;
                if (base_) {
                    separation = (c.rotations - best->rotations).redux(0., [](double sum, const auto& difference) {
                        return sum + difference.norm();
                    });
                } else
                    separation = Sym(c.mean - best->mean).norm();
                if (separation > 1e-5) ambiguous_ = true;
            }
        return std::move(*best);
    }
    /// @brief validates and factors one resolved positive lifted Hessian
    void prepare(const State& state) {
        h_ = hessian(state);
        if (h_->rows() == 0) return;
        const auto equilibrated = equilibrated_hessian_(*h_);
        const SPDMatrix<S, Coordinates> positive(equilibrated);
        lu_.emplace(equilibrated);
    }
    /// @brief solves a retained implicit system and checks its relative residual
    Vec solve_linear(const Vec& rhs, double tolerance = 1e-9) const {
        if (rhs.rows() == 0) return rhs;
        fdapde_strong_assert(lu_.has_value(), std::domain_error, "C-LE derivative requires a positive lifted Hessian");
        Vec x(equilibrated_solve_(*lu_, rhs));
        fdapde_strong_assert(
          Vec(*h_ * x - rhs).norm() <= tolerance * rhs.norm() + 32 * std::numeric_limits<S>::epsilon(),
          std::domain_error, "C-LE implicit solve failed");
        return x;
    }
    /// @brief measures the residual of a prepared implicit solve
    double residual(const Vec& x, const Vec& rhs) const { return rhs.rows() == 0 ? 0 : Vec(*h_ * x - rhs).norm(); }
    /// @brief reports stationarity of the chosen local candidate
    bool converged() const { return converged_; }
    /// @brief reports detected equal-cost competing stationary candidates
    bool ambiguous() const { return ambiguous_; }
    /// @brief returns the bound scalar metric coefficient
    double rho() const { return rho_; }
    /// @brief returns the number of skew coordinates per rotation
    int skew_size() const { return k_; }
   private:
    /// @brief runs small active lifts with inline optimizer coordinates and retains the dynamic state contract
    template <int Count>
    State solve_fixed_(
      std::span<const Eigenframe> eigenframes, const Eigenframe* base_eigenframe, double tolerance,
      std::size_t iterations) {
        CheegerLift<S, N, Count> fixed(charts_, w_, rho_, base_);
        auto state = fixed.solve(eigenframes, base_eigenframe, tolerance, iterations);
        converged_ = fixed.converged();
        ambiguous_ = fixed.ambiguous();
        return {std::move(state.rotations), std::move(state.z), std::move(state.omega), std::move(state.mean),
                Vec(state.gradient),        state.cost,         state.iterations};
    }
    /// @brief removes nodal weight units by a positive diagonal congruence without changing the lifted system
    Dense equilibrated_hessian_(const Dense& h) const {
        Dense scaled(h);
        for (int i = 0; i < h.rows(); ++i)
            for (int j = 0; j < h.cols(); ++j)
                scaled(i, j) = h(i, j) / S(std::sqrt(w_[i / k_])) / S(std::sqrt(w_[j / k_]));
        return scaled;
    }
    /// @brief converts both sides of an equilibrated solve back to the original rotation coordinates
    Vec equilibrated_solve_(const PartialPivLU<Dense>& lu, const Vec& rhs) const {
        Vec scaled(rhs);
        for (int i = 0; i < rhs.rows(); ++i) scaled[i] /= S(std::sqrt(w_[i / k_]));
        Vec result(lu.solve(scaled));
        for (int i = 0; i < result.rows(); ++i) result[i] /= S(std::sqrt(w_[i / k_]));
        return result;
    }
    MatrixBatch<Sym> charts_;
    std::vector<double> w_;
    double rho_;
    std::optional<Sym> base_;
    int n_, k_;
    bool converged_ = false, ambiguous_ = false;
    std::optional<Dense> h_;
    std::optional<PartialPivLU<Dense>> lu_;
};
}   // namespace fdapde::manifold::internals
#endif
