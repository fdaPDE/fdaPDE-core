// SPDX-License-Identifier: GPL-3.0-or-later
#include <fdaPDE/manifold_geometry.h>

#include <iostream>

namespace {
using namespace fdapde;

/// @brief checks lifted curvature against scalar costs for fixed and eliminated means
template <int N, int Nodes> void check_curvature(int order, int count, bool fixed, bool small_weight) {
    using Lift = manifold::internals::CheegerLift<double, N, Nodes>;
    using Sym = typename Lift::Sym;
    using Vec = typename Lift::Vec;
    MatrixBatch<Sym> charts(count, order, order);
    for (int i = 0; i < count; ++i) {
        Sym value;
        if constexpr (N == Dynamic) value.resize(order, order);
        for (int r = 0; r < order; ++r)
            for (int c = 0; c <= r; ++c) value(r, c) = .2 * std::sin(1 + i + 3 * r + c);
        charts[i] = value;
    }
    std::vector<double> weights(count, 1. / count);
    if (small_weight && count > 1) {
        weights[0] = 1e-13;
        for (int i = 1; i < count; ++i) weights[i] = (1 - weights[0]) / (count - 1);
    }
    Lift lift(charts, weights, .7, fixed ? std::optional<Sym>(charts[0]) : std::nullopt);
    typename Lift::Batch identity(count, order, order);
    Vec displacement(count * order * (order - 1) / 2), first(displacement.rows()), second(displacement.rows());
    for (int i = 0; i < displacement.rows(); ++i) {
        displacement[i] = .13 * std::cos(i + 1);
        first[i] = .2 * std::sin(i + 1);
        second[i] = .2 * std::cos(2 * i + 1);
    }
    typename Lift::Product geometry {&lift};
    const auto point = geometry.retract(identity, displacement, 1);
    const auto state = lift.evaluate(point);
    const auto hessian = lift.hessian(state);
    // a covariant scalar Hessian is symmetric even away from a stationary lift
    fdapde_strong_assert(
      (typename Lift::Dense(hessian - hessian.transpose())).norm() < 1e-12, std::runtime_error,
      "lifted Hessian lost symmetry");
    constexpr double step = 1e-4;
    const auto cost = [&](const Vec& direction) {
        return lift.evaluate(geometry.retract(point, direction, step)).cost;
    };
    for (const Vec& direction : {first, second, Vec(first + second)}) {
        const double numeric = (cost(direction) - 2 * state.cost + cost(Vec(-1. * direction))) / (step * step);
        const double analytic = manifold::internals::cheeger_inner(direction, Vec(hessian * direction));
        // geodesic scalar second differences independently check the assembled quadratic form
        fdapde_strong_assert(
          std::abs(numeric - analytic) < 2e-6 * std::max(1., std::abs(analytic)), std::runtime_error,
          "lifted Hessian disagrees with scalar second differences");
    }
}
}   // namespace

// cover the specialized pair and mean sizes, dynamic order, and nearly inactive mean weights
int main() {
    for (bool fixed : {false, true})
        for (bool small_weight : {false, true}) {
            check_curvature<3, 1>(3, 1, fixed, small_weight);
            check_curvature<3, 2>(3, 2, fixed, small_weight);
            check_curvature<3, 3>(3, 3, fixed, small_weight);
            check_curvature<3, 4>(3, 4, fixed, small_weight);
            check_curvature<3, Dynamic>(3, 3, fixed, small_weight);
            check_curvature<2, Dynamic>(2, 3, fixed, small_weight);
            check_curvature<4, Dynamic>(4, 3, fixed, small_weight);
            check_curvature<Dynamic, Dynamic>(3, 3, fixed, small_weight);
        }
    std::cout << "32 lifted Hessian cases passed\n";
}
