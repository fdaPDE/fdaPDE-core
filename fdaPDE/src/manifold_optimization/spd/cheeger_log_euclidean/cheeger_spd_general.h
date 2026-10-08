// SPDX-License-Identifier: GPL-3.0-or-later
#ifndef __FDAPDE_CHEEGER_SPD_GENERAL_H__
#define __FDAPDE_CHEEGER_SPD_GENERAL_H__
#include "../../header_check.h"
namespace fdapde::manifold::internals {
/// @brief retains one resolved pair lift and reuses its factorization for logarithm pullbacks
template <typename S, int N> class CheegerPairFrame {
   public:
    using Lift = CheegerLift<S, N>;
    using Sym = typename Lift::Sym;
    using Skew = typename Lift::Skew;
    using Vec = typename Lift::Vec;
    using Rotation = typename Lift::Rotation;
    /// @brief prepares the scalar planar branch or the general native rotation search
    CheegerPairFrame(const Sym& x, const Sym& y, double rho, bool derivatives = true) :
        x_(x), lift_(batch_(y), {1}, rho, x), state_(solve_(x, y, rho)) {
        if (derivatives) {
            fdapde_strong_assert(!ambiguous_, std::domain_error, "ambiguous C-LE tension edge");
            lift_.prepare(state_);
            differential_.emplace(state_.omega[0]);
        }
    }
    /// @brief returns the initial logarithmic-coordinate velocity of the selected quotient geodesic
    Sym logarithm() const {
        return Sym((state_.z[0] - x_ + state_.omega[0] * x_ - x_ * state_.omega[0]).template as_symmetric<Lower>());
    }
    /// @brief pulls an ambient chart covector back to both endpoint charts and local rho
    auto pullback(const Sym& dual) const {
        fdapde_strong_assert(differential_.has_value(), std::logic_error, "pair derivatives were not prepared");
        Vec rhs(lift_.skew_size());
        for (int a = 0; a < rhs.rows(); ++a) {
            const auto b = lift_.basis(a), domega = differential_->target_action(b);
            const Sym dl(
              (state_.z[0] * b - b * state_.z[0] + domega * x_ - x_ * domega).template as_symmetric<Lower>());
            rhs[a] = S(cheeger_inner(dual, dl));
        }
        const auto lambda = lift_.tangent(lift_.solve_linear(rhs), 0);
        const auto& q = state_.rotations[0];
        const auto& omega = state_.omega[0];
        const Sym base(
          (S(-1) * dual + dual * omega - omega * dual - S(2) * (lambda * state_.z[0] - state_.z[0] * lambda))
            .template as_symmetric<Lower>());
        const Sym target(
          (q * (dual - S(2) * (x_ * lambda - lambda * x_)) * q.transpose()).template as_symmetric<Lower>());
        return std::tuple {base, target, -2 * cheeger_inner(lambda, omega)};
    }
    /// @brief borrows the selected rotation and its aligned chart
    const auto& state() const { return state_; }
    /// @brief reports a detected minimizing tie
    bool ambiguous() const { return ambiguous_; }
   private:
    /// @brief packs one chart without converting through an SPD exponential
    static MatrixBatch<Sym> batch_(const Sym& y) {
        MatrixBatch<Sym> result(1, y.rows(), y.cols());
        result[0] = y;
        return result;
    }
    /// @brief preserves the exact planar branch enumeration and otherwise runs the general solver
    typename Lift::State solve_(const Sym& x, const Sym& y, double rho) {
        if (x.rows() == 2) {
            const auto pair = cheeger_pair(cheeger_chart(x), cheeger_chart(y), rho);
            ambiguous_ = pair.rotations.size() != 1;
            Skew omega;
            if constexpr (N == Dynamic) omega.resize(2, 2);
            for (int r = 0; r < omega.rows(); ++r)
                for (int c = r + 1; c < omega.rows(); ++c) omega(r, c) = 0;
            omega(0, 1) = S(-pair.rotations.front());
            typename Lift::Batch rotations(1, 2, 2);
            rotations[0] = rotation_exp(omega);
            return lift_.evaluate(rotations);
        }
        auto state = lift_.solve();
        fdapde_strong_assert(lift_.converged(), std::domain_error, "C-LE pair alignment did not converge");
        ambiguous_ = lift_.ambiguous();
        return state;
    }
    Sym x_;
    Lift lift_;
    bool ambiguous_ = false;
    typename Lift::State state_;
    std::optional<CheegerRotationDifferential<S, N>> differential_;
};
}   // namespace fdapde::manifold::internals
namespace fdapde::manifold {
/// @brief defines the C-LE quotient metric for fixed or dynamic SPD order
/// @details general alignment uses local multistart and never certifies global uniqueness
template <typename S, int N, Usage U, typename Point_> class CheegerLogEuclideanSPDGeometry {
    // the rotation lift applies to tensor orders at least two; scalar smoothing uses LE
    static_assert(N == Dynamic || N >= 2, "C-LE requires tensor order at least two");
    using Base = LogEuclideanSPDGeometry<S, N, U, Point_>;
    Base ambient_;
    double rho_;
   public:
    using Self = CheegerLogEuclideanSPDGeometry<S, N, U, Point_>;
    using Scalar = S;
    using Point = typename Base::Point;
    using CachePolicy = typename Point::CachePolicy;
    using Tangent = typename Base::Tangent;
    using Chart = Tangent;
    using Rotation = RotationMatrix<S, N, N, RotationCache::Log>;
    /// @brief constructs a fixed-order geometry from the legacy epsilon parameter
    explicit CheegerLogEuclideanSPDGeometry(double epsilon = .5)
        requires(N != Dynamic)
        : rho_(checked_(epsilon * epsilon)) {
        fdapde_strong_assert(epsilon > 0, std::invalid_argument, "epsilon must be positive");
    }
    /// @brief constructs a dynamic-order geometry and checks its metric parameter
    explicit CheegerLogEuclideanSPDGeometry(int order, double epsilon = .5)
        requires(N == Dynamic)
        : ambient_(order), rho_(checked_(epsilon * epsilon)) {
        fdapde_strong_assert(order >= 2, std::invalid_argument, "C-LE requires tensor order at least two");
        fdapde_strong_assert(epsilon > 0, std::invalid_argument, "epsilon must be positive");
    }
    /// @brief constructs a fixed-order geometry directly from rho
    static auto from_rho(double rho)
        requires(N != Dynamic)
    {
        CheegerLogEuclideanSPDGeometry g;
        g.rho_ = checked_(rho);
        return g;
    }
    /// @brief constructs a dynamic-order geometry directly from rho and order
    static auto from_rho(double rho, int order)
        requires(N == Dynamic)
    {
        CheegerLogEuclideanSPDGeometry g(order);
        g.rho_ = checked_(rho);
        return g;
    }
    /// @brief replaces the metric parameter while retaining the tensor order
    auto with_rho(double rho) const {
        auto g = *this;
        g.rho_ = checked_(rho);
        return g;
    }
    /// @brief returns the squared rotation penalty
    double rho() const { return rho_; }
    /// @brief returns the legacy square-root parameter
    double epsilon() const { return std::sqrt(rho_); }
    /// @brief returns the tensor order
    int order() const { return ambient_.order(); }
    /// @brief returns the symmetric tangent dimension
    std::size_t dimension() const { return ambient_.dimension(); }
    /// @brief borrows a cached logarithm into an owning chart
    template <SPDLike P> static Chart chart(const P& p) { return Chart(matrix_log(p)); }
    /// @brief exponentiates a symmetric logarithmic chart
    template <typename OutputPolicy = CachePolicy> static auto from_chart(const Chart& x) {
        return matrix_exp<OutputPolicy>(x);
    }
    /// @brief projects an ambient symmetric tangent
    template <SPDLike P> Tangent project(const P& p, const Tangent& v) const { return ambient_.project(p, v); }
    /// @brief constructs a zero tangent at the requested point
    template <SPDLike P> Tangent zero_tangent(const P& p) const { return ambient_.zero_tangent(p); }
    /// @brief combines ambient symmetric tangents
    template <SPDLike P>
    Tangent linear_combination(const P& p, double a, const Tangent& x, double b, const Tangent& y) const {
        return ambient_.linear_combination(p, a, x, b, y);
    }
    /// @brief pairs logarithmic differentials through the C-LE spectral metric
    template <SPDLike P> double inner_product(const P& p, const Tangent& u, const Tangent& v) const {
        check_(p);
        const Tangent a(matrix_log_frechet(p, u)), b(matrix_log_frechet(p, v));
        return internals::cheeger_inner(a, internals::cheeger_metric<S, N>(matrix_log(p), b, rho_));
    }
    /// @brief measures an ambient tangent in the local metric
    template <SPDLike P> double norm(const P& p, const Tangent& v) const {
        return std::sqrt(std::max(0., inner_product(p, v, v)));
    }
    /// @brief records a selected minimizing rotation with separate ambiguity diagnostics
    struct Pair {
        double squared_distance;
        std::vector<Rotation> rotations;
        bool detected_ambiguity;
        /// @brief reports absence of detected ties without certifying global uniqueness
        bool unique() const { return !detected_ambiguity; }
    };
    /// @brief solves a local rotational alignment in logarithmic coordinates
    template <SPDLike P, SPDLike Q> Pair pair(const P& p, const Q& q) const {
        check_(p);
        check_(q);
        const internals::CheegerPairFrame<S, N> frame(chart(p), chart(q), rho_, false);
        return {frame.state().cost, {Rotation(frame.state().rotations[0])}, frame.ambiguous()};
    }
    /// @brief retains a pair lift for repeated quotient-geodesic evaluation
    class Curve {
        Chart x_, v_;
        SkewSymmetricMatrix<S, N, N> omega_;
        double distance_;
       public:
        /// @brief retains endpoint charts and the selected rotation logarithm
        Curve(const Chart& x, const Chart& y, const Rotation& q, double distance) :
            x_(x),
            v_((q.transpose() * y * q - x).template as_symmetric<Lower>()),
            omega_(rotation_log(q)),
            distance_(distance) { }
        /// @brief evaluates the quotient of a straight symmetric path and a group geodesic
        template <typename OutputPolicy = CachePolicy> auto operator()(double t) const {
            const auto q = rotation_exp(omega_, t);
            return from_chart<OutputPolicy>(
              Chart((q * (x_ + S(t) * v_) * q.transpose()).template as_symmetric<Lower>()));
        }
        /// @brief returns the prepared squared endpoint distance
        double squared_distance() const { return distance_; }
    };
    /// @brief prepares a repeated curve after rejecting detected alignment ties
    template <SPDLike P, SPDLike Q> Curve geodesic(const P& p, const Q& q) const {
        const auto fit = pair(p, q);
        fdapde_strong_assert(fit.unique(), std::domain_error, "ambiguous C-LE geodesic");
        return Curve(chart(p), chart(q), fit.rotations.front(), fit.squared_distance);
    }
    /// @brief returns uniformly spaced geodesic samples including both endpoints with the selected output cache policy
    /// @details defaults to the geometry point policy; prepares one curve and requires count >= 2
    /// sample i uses t = i / (count - 1); execution defaults to sequential and parallel calls join before returning
    template <
      typename OutputPolicy = CachePolicy, SPDLike From, SPDLike To,
      fdapde::internals::BatchExecutionPolicy ExecutionPolicy = execution_seq_t>
    auto geodesic(const From& from, const To& to, int count, ExecutionPolicy policy = {}) const {
        return internals::sample_spd_geodesic<OutputPolicy>(*this, from, to, count, policy);
    }
    /// @brief returns the selected quotient distance
    template <SPDLike P, SPDLike Q> double distance(const P& p, const Q& q) const {
        return std::sqrt(pair(p, q).squared_distance);
    }
    /// @brief maps the selected initial chart velocity into an ambient SPD tangent
    template <SPDLike P, SPDLike Q> Tangent logarithm(const P& p, const Q& q) const {
        check_(p);
        check_(q);
        const internals::CheegerPairFrame<S, N> frame(chart(p), chart(q), rho_, false);
        fdapde_strong_assert(!frame.ambiguous(), std::domain_error, "ambiguous C-LE logarithm");
        return Tangent(matrix_exp_frechet(matrix_log(p), frame.logarithm()));
    }
    /// @brief computes the horizontal lift and its exact quotient exponential
    template <SPDLike P> Point exponential(const P& p, const Tangent& u, double step = 1) const {
        check_(p);
        const Chart x = chart(p), h(matrix_log_frechet(p, u));
        const EVD evd(x);
        const auto& q = evd.eigenvectors();
        const Matrix<S, N, N> local(q.transpose() * h * q);
        SkewSymmetricMatrix<S, N, N> omega;
        if constexpr (N == Dynamic) omega.resize(order(), order());
        for (int r = 0; r < omega.rows(); ++r)
            for (int c = r + 1; c < omega.rows(); ++c) omega(r, c) = 0;
        for (int i = 0; i < order(); ++i)
            for (int j = i + 1; j < order(); ++j) {
                const S gap = evd.eigenvalues()[j] - evd.eigenvalues()[i];
                omega(i, j) = local(i, j) * gap / S(rho_ + gap * gap);
            }
        const SkewSymmetricMatrix<S, N, N> world((q * omega * q.transpose()).template as_skew_symmetric<Lower>());
        const Chart v((h - world * x + x * world).template as_symmetric<Lower>());
        const auto rotation = rotation_exp(world, step);
        return from_chart(Chart((rotation * (x + S(step) * v) * rotation.transpose()).template as_symmetric<Lower>()));
    }
    /// @brief retracts by the quotient exponential
    template <SPDLike P> Point retract(const P& p, const Tangent& u, double step) const {
        return exponential(p, u, step);
    }
    /// @brief prepares native spatial interpolation with default mean tolerances
    template <typename Element, typename Nodes>
        requires gfe::P1InterpolationBinding<Element, Nodes>
    auto interpolant(Element&& element, Nodes&& nodes) const;
    /// @brief prepares native spatial interpolation with explicit mean tolerances
    template <typename Element, typename Nodes>
        requires gfe::P1InterpolationBinding<Element, Nodes>
    auto interpolant(Element&& element, Nodes&& nodes, const gfe::P1GeodesicLinearizationOptions& options) const;
   private:
    /// @brief rejects invalid public metric parameters
    static double checked_(double rho) {
        fdapde_strong_assert(std::isfinite(rho) && rho > 0, std::invalid_argument, "C-LE requires finite rho > 0");
        return rho;
    }
    /// @brief validates point dimensions against the bound geometry
    void check_(const auto& p) const {
        fdapde_strong_assert(
          p.rows() == order() && p.cols() == order(), std::invalid_argument, "C-LE point order mismatch");
    }
};
}   // namespace fdapde::manifold
#endif
