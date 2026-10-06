// SPDX-License-Identifier: GPL-3.0-or-later
#ifndef __FDAPDE_CHEEGER_LIFT_H__
#define __FDAPDE_CHEEGER_LIFT_H__
#include "../../header_check.h"

namespace fdapde::manifold::internals {
/// @brief pairs full matrix coefficients without assuming a storage layout
inline double cheeger_inner(const auto& a, const auto& b) {
    double value = 0;
    for (int i = 0; i < a.rows(); ++i)
        for (int j = 0; j < a.cols(); ++j) value += double(a(i, j)) * double(b(i, j));
    return value;
}
/// @brief specializes rotation differentials in dimensions two and three and reuses SO(n) otherwise
template <typename S, int N> class CheegerRotationDifferential {
   public:
    using Tangent = SkewSymmetricMatrix<S, N, N>;
    /// @brief prepares the scalar Rodrigues coefficient or the general adjoint spectrum
    explicit CheegerRotationDifferential(const Tangent& omega) : omega_(omega), squared_(cheeger_inner(omega, omega)) {
        if (omega.rows() > 3) general_.emplace(omega);
        if (omega.rows() == 3) {
            const double theta2 = squared_ / 2;
            const double half = std::sqrt(theta2) / 2;
            coefficient_ = theta2 < 1e-6 ? 1 - theta2 / 12 - theta2 * theta2 / 720 - theta2 * theta2 * theta2 / 30240 :
                                           half / std::tan(half);
        }
    }
    /// @brief applies the bi-invariant squared-distance Hessian in body coordinates
    Tangent hessian_action(const Tangent& direction) const {
        if (general_) return general_->hessian_action(direction);
        const double projection = squared_ == 0 ? 0 : (1 - coefficient_) * cheeger_inner(omega_, direction) / squared_;
        return Tangent((S(coefficient_) * direction + S(projection) * omega_).template as_skew_symmetric<Lower>());
    }
    /// @brief differentiates the group logarithm along a right body perturbation
    Tangent target_action(const Tangent& direction) const {
        if (general_) return general_->target_action(direction);
        const auto h = hessian_action(direction);
        return Tangent((h + S(.5) * (omega_ * direction - direction * omega_)).template as_skew_symmetric<Lower>());
    }
   private:
    Tangent omega_;
    double squared_, coefficient_ = 1;
    std::optional<SOLogDifferential<S, N>> general_;
};
/// @brief applies the C-LE log-coordinate metric using a symmetric spectral frame
template <typename S, int N, typename X>
SymmetricMatrix<S, N, N> cheeger_metric(const X& x, const SymmetricMatrix<S, N, N>& h, double rho) {
    const EVD evd(x);
    const auto& q = evd.eigenvectors();
    const Matrix<S, N, N> local(q.transpose() * h * q);
    Matrix<S, N, N> scaled(local);
    for (int i = 0; i < x.rows(); ++i)
        for (int j = 0; j < x.rows(); ++j) {
            const double gap = double(evd.eigenvalues()[i]) - double(evd.eigenvalues()[j]);
            scaled(i, j) *= S(rho / (rho + gap * gap));
        }
    return SymmetricMatrix<S, N, N>((q * scaled * q.transpose()).template as_symmetric<Lower>());
}
/// @brief minimizes rotational lifts and retains their analytic Hessian for implicit differentiation
/// @details pair searches use one fixed chart; mean searches eliminate the weighted symmetric chart
/// @details multistart is a local search and does not certify global optimality
// ponytail: dense lifted Hessians scale quadratically in nodes*n*(n-1)/2, use matrix-free solves for large orders
template <typename S, int N> class CheegerLift {
   public:
    using Sym = SymmetricMatrix<S, N, N>;
    using Skew = SkewSymmetricMatrix<S, N, N>;
    using Rotation = RotationMatrix<S, N, N, RotationCache::Log>;
    using Batch = MatrixBatch<Rotation>;
    using Dense = Matrix<S, Dynamic, Dynamic>;
    using Vec = Vector<S, Dynamic>;
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
        Dense h(size, size);
        h.set_zero();
        std::vector<CheegerRotationDifferential<S, N>> differentials;
        for (const auto& omega : f.omega) differentials.emplace_back(omega);
        for (std::size_t j = 0; j < w_.size(); ++j)
            for (int a = 0; a < k_; ++a) {
                const Skew b = basis(a);
                const Sym dz((f.z[j] * b - b * f.z[j]).template as_symmetric<Lower>());
                const Sym dm(base_ ? zero() : Sym(S(w_[j]) * dz));
                for (std::size_t i = 0; i < w_.size(); ++i) {
                    Matrix<S, N, N> dg(S(2 * w_[i]) * (dm * f.z[i] - f.z[i] * dm));
                    if (i == j) {
                        const auto curvature = differentials[i].hessian_action(b);
                        const Matrix<S, N, N> g(S(2 * w_[i]) * (f.mean * f.z[i] - f.z[i] * f.mean));
                        dg +=
                          S(2 * w_[i]) * (f.mean * dz - dz * f.mean + S(rho_) * curvature) + S(.5) * (b * g - g * b);
                    }
                    int p = int(i) * k_;
                    for (int r = 0; r < n_; ++r)
                        for (int c = r + 1; c < n_; ++c) h(p++, int(j) * k_ + a) = S(std::sqrt(2.)) * dg(r, c);
                }
            }
        return h;
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
            Point q(p.size(), lift->n_, lift->n_);
            for (std::size_t i = 0; i < p.size(); ++i) {
                const auto delta = rotation_exp(lift->tangent(v, i), step);
                q[i] = p[i] * delta;
            }
            return q;
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
        Vec gradient(const Batch& p, Workspace& w) {
            if (!w.state) w.state = lift->evaluate(p);
            return w.state->gradient;
        }
        /// @brief reuses the analytic candidate Hessian across subproblem iterations
        Vec hessian_vector(const Batch& p, const Vec& v, Workspace& w) {
            if (!w.state) w.state = lift->evaluate(p);
            if (!w.hessian) w.hessian = lift->hessian(*w.state);
            return Vec(*w.hessian * v);
        }
    };
    /// @brief solves several deterministic local starts and retains competing stationary candidates
    State solve(double tolerance = 1e-10, std::size_t iterations = 200) {
        Matrix<S, N, N> identity(n_, n_);
        identity.set_zero();
        for (int i = 0; i < n_; ++i) identity(i, i) = 1;
        Batch initial(w_.size(), n_, n_);
        for (std::size_t i = 0; i < w_.size(); ++i) initial[i] = identity;
        if (!base_ && w_.size() == 1) {
            converged_ = true;
            ambiguous_ = false;
            return evaluate(initial);
        }
        std::vector<Batch> starts;
        starts.push_back(initial);
        // seed whole eigenframe alignments as well as individual rotation planes
        std::vector<Matrix<S, N, N>> eigenframes;
        for (std::size_t i = 0; i < w_.size(); ++i) {
            const EVD evd(charts_[i]);
            Matrix<S, N, N> q(evd.eigenvectors());
            if (PartialPivLU(q).determinant() < 0)
                for (int r = 0; r < n_; ++r) q(r, 0) = -q(r, 0);
            eigenframes.push_back(std::move(q));
        }
        std::vector<Matrix<S, N, N>> references = eigenframes;
        if (base_) {
            const EVD evd(*base_);
            Matrix<S, N, N> q(evd.eigenvectors());
            if (PartialPivLU(q).determinant() < 0)
                for (int r = 0; r < n_; ++r) q(r, 0) = -q(r, 0);
            references = {std::move(q)};
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
                    const PartialPivLU lu(h);
                    if (lu.info() != 0) break;
                    const Vec rhs(S(-1) * candidate.gradient), step(lu.solve(rhs));
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
        State best = candidates.front();
        converged_ = best.gradient.norm() <= tolerance;
        ambiguous_ = false;
        for (const auto& c : candidates)
            if (c.gradient.norm() <= tolerance && c.cost - best.cost <= 1e-10 * std::max(1., best.cost)) {
                double separation = 0;
                if (base_) {
                    for (std::size_t i = 0; i < w_.size(); ++i)
                        separation += Matrix<S, N, N>(c.rotations[i] - best.rotations[i]).norm();
                } else
                    separation = Sym(c.mean - best.mean).norm();
                if (separation > 1e-5) ambiguous_ = true;
            }
        return best;
    }
    /// @brief validates and factors one resolved positive lifted Hessian
    void prepare(const State& state) {
        h_ = hessian(state);
        if (h_->rows() == 0) return;
        const SPDMatrix<S, Dynamic, Dynamic> positive(*h_);
        lu_.emplace(*h_);
    }
    /// @brief solves a retained implicit system and checks its relative residual
    Vec solve_linear(const Vec& rhs, double tolerance = 1e-9) const {
        if (rhs.rows() == 0) return rhs;
        fdapde_strong_assert(lu_.has_value(), std::domain_error, "C-LE derivative requires a positive lifted Hessian");
        Vec x(lu_->solve(rhs));
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
