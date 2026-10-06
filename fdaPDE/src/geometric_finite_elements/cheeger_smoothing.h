// SPDX-License-Identifier: GPL-3.0-or-later
#ifndef __FDAPDE_CHEEGER_SMOOTHING_H__
#define __FDAPDE_CHEEGER_SMOOTHING_H__
#include "header_check.h"
namespace fdapde::gfe {
/// @brief returns Frobenius log-coordinate covectors and scalar rho derivatives in nodal order
/// @details these are coordinate gradients, not ambient Riemannian metric gradients
template <typename Tangent> struct P1CheegerLogContributionResult : P1ObjectiveContributionResult<Tangent> {
    std::vector<double> rho_gradient;
};
/// @brief evaluates planar discrete tension and its nodal log and rho covectors with scalar rotation branches
/// @details cached nodal logarithms are shared with other contributions; no general rotation lift is constructed
template <typename S, Usage U, typename Nodes>
P1CheegerLogContributionResult<typename manifold::CheegerLogEuclideanSPDGeometry<S, 2, U>::Tangent>
p1_cheeger_discrete_tension_log_contribution(
  const manifold::CheegerLogEuclideanSPDGeometry<S, 2, U>& geometry, const Nodes& nodes,
  const P1LumpedLaplacianStencil& stencil, std::span<const double> rho_nodes = {}) {
    using G = manifold::CheegerLogEuclideanSPDGeometry<S, 2, U>;
    using C = manifold::internals::CheegerChart;
    using V = std::array<double, 3>;
    internals::p1_discrete_tension_validate(geometry, nodes, stencil);
    fdapde_strong_assert(
      rho_nodes.empty() || rho_nodes.size() == nodes.size(), std::invalid_argument, "rho count must match nodes");
    for (double rho : rho_nodes) geometry.with_rho(rho);
    std::vector<C> q;
    q.reserve(nodes.size());
    for (std::size_t i = 0; i < nodes.size(); ++i) q.push_back(G::chart(nodes[i]));
    std::vector<V> residual(nodes.size(), V {}), dual(nodes.size(), V {}), gradient(nodes.size(), V {});
    const auto rho = [&](std::size_t i) { return rho_nodes.empty() ? geometry.rho() : rho_nodes[i]; };
    /// @brief retains one directed edge's scalar minimizing angle and aligned target chart
    struct Edge {
        std::size_t i, j;
        double weight, beta;
        C z;
    };
    std::vector<Edge> edges;
    edges.reserve(2 * stencil.edges.size());
    for (const auto& edge : stencil.edges)
        for (int reverse = 0; reverse < 2; ++reverse) {
            const auto i = reverse ? edge.second : edge.first;
            const auto j = reverse ? edge.first : edge.second;
            const auto pair = manifold::internals::cheeger_pair(q[i], q[j], rho(i));
            fdapde_strong_assert(pair.rotations.size() == 1, std::domain_error, "ambiguous Cheeger tension edge");
            const double beta = pair.rotations.front();
            const auto z = manifold::internals::cheeger_rotate(q[j], -beta);
            residual[i][0] += edge.stiffness * (z.s - q[i].s);
            residual[i][1] += edge.stiffness * (z.x - q[i].x - 2 * beta * q[i].y);
            residual[i][2] += edge.stiffness * (z.y - q[i].y + 2 * beta * q[i].x);
            edges.push_back({i, j, edge.stiffness, beta, z});
        }
    P1CheegerLogContributionResult<typename G::Tangent> out;
    out.nodal_gradient.reserve(nodes.size());
    out.rho_gradient.assign(nodes.size(), 0);
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        const double d = rho(i) + 4 * (q[i].x * q[i].x + q[i].y * q[i].y);
        const double c = q[i].x * residual[i][2] - q[i].y * residual[i][1], mass = stencil.lumped_masses[i];
        out.value += (residual[i][0] * residual[i][0] + residual[i][1] * residual[i][1] +
                      residual[i][2] * residual[i][2] - 4 * c * c / d) /
                     mass;
        dual[i] = {
          2 * residual[i][0] / mass, (2 * residual[i][1] + 8 * c * q[i].y / d) / mass,
          (2 * residual[i][2] - 8 * c * q[i].x / d) / mass};
        gradient[i][1] = (-8 * c * residual[i][2] / d + 32 * c * c * q[i].x / (d * d)) / mass;
        gradient[i][2] = (8 * c * residual[i][1] / d + 32 * c * c * q[i].y / (d * d)) / mass;
        out.rho_gradient[i] = 4 * c * c / (d * d * mass);
    }
    for (const auto& edge : edges) {
        const auto i = edge.i, j = edge.j;
        const auto z = edge.z;
        const double d = rho(i) + 4 * (q[i].x * z.x + q[i].y * z.y);
        fdapde_strong_assert(d > 1e-12, std::domain_error, "singular Cheeger edge branch");
        out.rho_gradient[i] -=
          2 * edge.weight * edge.beta / d * (dual[i][1] * (z.y - q[i].y) + dual[i][2] * (q[i].x - z.x));
        for (int endpoint = 0; endpoint < 2; ++endpoint)
            for (int coord = 0; coord < 3; ++coord) {
                C u {}, v {};
                C& perturb = endpoint ? v : u;
                if (coord == 0) perturb.s = 1;
                if (coord == 1) perturb.x = 1;
                if (coord == 2) perturb.y = 1;
                const auto dz = manifold::internals::cheeger_rotate(v, -edge.beta);
                const double db = -2 * (-u.x * z.y + u.y * z.x - q[i].x * dz.y + q[i].y * dz.x) / d;
                const V dv {
                  v.s - u.s, dz.x - u.x - 2 * edge.beta * u.y + 2 * (z.y - q[i].y) * db,
                  dz.y - u.y + 2 * edge.beta * u.x + 2 * (q[i].x - z.x) * db};
                for (int k = 0; k < 3; ++k) gradient[endpoint ? j : i][coord] += edge.weight * dual[i][k] * dv[k];
            }
    }
    for (const auto& value : gradient)
        out.nodal_gradient.push_back(
          manifold::internals::cheeger_matrix<S>({value[0] / 2, value[1] / 2, value[2] / 2}));
    fdapde_strong_assert(std::isfinite(out.value), std::domain_error, "nonfinite C-LE tension");
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        internals::p1_objective_require_finite_shape<std::domain_error>(
          out.nodal_gradient[i], geometry.order(), "nonfinite C-LE tension gradient");
        fdapde_strong_assert(std::isfinite(out.rho_gradient[i]), std::domain_error, "nonfinite C-LE rho gradient");
    }
    return out;
}
/// @brief evaluates the existing squared discrete tension using rho_i at each residual's base node
/// @details rho coefficients remain fixed during one call; an empty span uses the geometry's constant rho
template <typename S, int N, Usage U, typename Nodes>
P1CheegerLogContributionResult<typename manifold::CheegerLogEuclideanSPDGeometry<S, N, U>::Tangent>
p1_cheeger_discrete_tension_log_contribution(
  const manifold::CheegerLogEuclideanSPDGeometry<S, N, U>& geometry, const Nodes& nodes,
  const P1LumpedLaplacianStencil& stencil, std::span<const double> rho_nodes = {}) {
    using Sym = typename manifold::CheegerLogEuclideanSPDGeometry<S, N, U>::Tangent;
    using Frame = manifold::internals::CheegerPairFrame<S, N>;
    internals::p1_discrete_tension_validate(geometry, nodes, stencil);
    fdapde_strong_assert(
      rho_nodes.empty() || rho_nodes.size() == nodes.size(), std::invalid_argument, "rho count must match nodes");
    std::vector<double> rho(nodes.size(), geometry.rho());
    if (!rho_nodes.empty()) rho.assign(rho_nodes.begin(), rho_nodes.end());
    for (double r : rho) geometry.with_rho(r);
    MatrixBatch<CachedSymmetricMatrix<S, N, N>> charts(nodes.size(), geometry.order(), geometry.order());
    Sym zero;
    if constexpr (N == Dynamic) zero.resize(geometry.order(), geometry.order());
    for (int i = 0; i < geometry.order(); ++i)
        for (int j = 0; j <= i; ++j) zero(i, j) = 0;
    std::vector<Sym> residual(nodes.size(), zero), dual(nodes.size(), zero);
    for (std::size_t i = 0; i < nodes.size(); ++i) charts[i] = matrix_log(nodes[i]);
    std::vector<std::array<Frame, 2>> frames;
    frames.reserve(stencil.edges.size());
    for (const auto& edge : stencil.edges) {
        frames.push_back(
          {Frame(Sym(charts[edge.first]), Sym(charts[edge.second]), rho[edge.first]),
           Frame(Sym(charts[edge.second]), Sym(charts[edge.first]), rho[edge.second])});
        const auto a = frames.back()[0].logarithm(), b = frames.back()[1].logarithm();
        residual[edge.first] += S(edge.stiffness) * a;
        residual[edge.second] += S(edge.stiffness) * b;
    }
    P1CheegerLogContributionResult<Sym> out;
    out.nodal_gradient.assign(nodes.size(), zero);
    out.rho_gradient.assign(nodes.size(), 0);
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        const auto metric = manifold::internals::cheeger_metric<S, N>(charts[i], residual[i], rho[i]);
        const double mass = stencil.lumped_masses[i];
        out.value += .5 * manifold::internals::cheeger_inner(residual[i], metric) / mass;
        dual[i] = metric / S(mass);
        const Matrix<S, N, N> commutator(charts[i] * metric - metric * charts[i]);
        out.nodal_gradient[i] =
          Sym(((-S(1 / (mass * rho[i]))) * (commutator * metric - metric * commutator)).template as_symmetric<Lower>());
        out.rho_gradient[i] =
          .5 * manifold::internals::cheeger_inner(commutator, commutator) / (mass * rho[i] * rho[i]);
    }
    for (std::size_t e = 0; e < stencil.edges.size(); ++e)
        for (int reverse = 0; reverse < 2; ++reverse) {
            const auto& edge = stencil.edges[e];
            const auto i = reverse ? edge.second : edge.first, j = reverse ? edge.first : edge.second;
            const Sym covector(S(edge.stiffness) * dual[i]);
            const auto [base, target, rho_derivative] = frames[e][reverse].pullback(covector);
            out.nodal_gradient[i] += base;
            out.nodal_gradient[j] += target;
            out.rho_gradient[i] += rho_derivative;
        }
    fdapde_strong_assert(std::isfinite(out.value), std::domain_error, "nonfinite C-LE tension");
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        internals::p1_objective_require_finite_shape<std::domain_error>(
          out.nodal_gradient[i], geometry.order(), "nonfinite C-LE tension gradient");
        fdapde_strong_assert(std::isfinite(out.rho_gradient[i]), std::domain_error, "nonfinite C-LE rho gradient");
    }
    return out;
}
/// @brief pulls a Frobenius data residual back through C-LE interpolation to nodal logs and rho
/// @details rho gradients are with respect to nodal scalar coefficients even when their values are constant
template <typename S, int N, Usage U, typename Nodes>
P1CheegerLogContributionResult<typename manifold::CheegerLogEuclideanSPDGeometry<S, N, U>::Tangent>
p1_cheeger_frobenius_data_site_log_contribution(
  const manifold::CheegerLogEuclideanSPDGeometry<S, N, U>& geometry, const Nodes& nodes,
  std::span<const double> weights,
  const typename manifold::CheegerLogEuclideanSPDGeometry<S, N, U>::Tangent& observation,
  const P1GeodesicLinearizationOptions& options = {}, std::span<const double> rho_nodes = {}) {
    using Sym = typename manifold::CheegerLogEuclideanSPDGeometry<S, N, U>::Tangent;
    const auto vertex = internals::validate_p1_data(nodes.size(), weights);
    internals::p1_objective_require_finite_shape(observation, geometry.order(), "invalid C-LE observation");
    fdapde_strong_assert(
      rho_nodes.empty() || rho_nodes.size() == nodes.size(), std::invalid_argument, "rho count mismatch");
    for (double r : rho_nodes) geometry.with_rho(r);
    for (std::size_t i = 0; i < nodes.size(); ++i)
        internals::p1_objective_require_finite_shape(nodes[i], geometry.order(), "invalid C-LE node");
    Sym zero;
    if constexpr (N == Dynamic) zero.resize(geometry.order(), geometry.order());
    for (int i = 0; i < geometry.order(); ++i)
        for (int j = 0; j <= i; ++j) zero(i, j) = 0;
    P1CheegerLogContributionResult<Sym> out;
    out.nodal_gradient.assign(nodes.size(), zero);
    out.rho_gradient.assign(nodes.size(), 0);
    if (vertex) {
        const Sym residual(nodes[*vertex] - observation);
        out.value = .5 * manifold::internals::cheeger_inner(residual, residual);
        out.nodal_gradient[*vertex] = matrix_exp_frechet(matrix_log(nodes[*vertex]), residual);
        fdapde_strong_assert(std::isfinite(out.value), std::domain_error, "nonfinite C-LE data loss");
        internals::p1_objective_require_finite_shape<std::domain_error>(
          out.nodal_gradient[*vertex], geometry.order(), "nonfinite C-LE data gradient");
        return out;
    }
    std::vector<int> active;
    std::vector<double> w, rho;
    for (std::size_t i = 0; i < weights.size(); ++i)
        if (weights[i] > 0) {
            active.push_back(int(i));
            w.push_back(weights[i]);
            if (!rho_nodes.empty()) rho.push_back(rho_nodes[i]);
        }
    auto local = [&] {
        if constexpr (requires { nodes.select(active); })
            return nodes.select(active);
        else {
            MatrixBatch<typename manifold::CheegerLogEuclideanSPDGeometry<S, N, U>::Point> selected(
              active.size(), geometry.order(), geometry.order());
            for (std::size_t i = 0; i < active.size(); ++i) selected[i] = nodes[active[i]];
            return selected;
        }
    }();
    auto fit = p1_geodesic_linearization(geometry, local, w, options, rho);
    const auto value = internals::p1_frobenius_data_site_value_impl(geometry, fit.result(), observation);
    out.value = value.value;
    out.first_failure = value.first_failure;
    if (!out.converged()) return out;
    fdapde_strong_assert(!fit.result().detected_ambiguity, std::domain_error, "ambiguous C-LE data interpolation");
    const Sym residual(fit.result().value - observation);
    double local_rho = 0;
    if constexpr (N != 2) {
        const auto pullback = fit.nodal_log_rho_vjp(residual);
        for (std::size_t i = 0; i < active.size(); ++i) out.nodal_gradient[active[i]] = pullback.first[i];
        local_rho = pullback.second;
    } else {
        std::vector<Sym> direction(active.size(), zero);
        for (std::size_t i = 0; i < active.size(); ++i)
            for (int r = 0; r < N; ++r)
                for (int c = 0; c <= r; ++c) {
                    direction[i](r, c) = 1;
                    const auto d = fit.nodal_log_jvp(direction);
                    out.nodal_gradient[active[i]](r, c) =
                      S(manifold::internals::cheeger_inner(residual, d) / (r == c ? 1 : 2));
                    direction[i](r, c) = 0;
                }
        const auto scalar = fit.rho_jvp();
        fdapde_strong_assert(scalar.converged(), std::domain_error, "C-LE scalar derivative solve failed");
        local_rho = manifold::internals::cheeger_inner(residual, scalar.derivative);
    }
    double total = 0;
    for (double wi : w) total += wi;
    for (std::size_t i = 0; i < active.size(); ++i) out.rho_gradient[active[i]] = w[i] * local_rho / total;
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        internals::p1_objective_require_finite_shape<std::domain_error>(
          out.nodal_gradient[i], geometry.order(), "nonfinite C-LE data gradient");
        fdapde_strong_assert(
          std::isfinite(out.rho_gradient[i]), std::domain_error, "nonfinite C-LE data rho derivative");
    }
    return out;
}
}   // namespace fdapde::gfe
#endif
