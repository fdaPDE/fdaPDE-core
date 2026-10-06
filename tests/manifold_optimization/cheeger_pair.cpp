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

#include <fdaPDE/manifold_optimization.h>
#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <limits>
#include <numbers>
#include <random>
#include <vector>

namespace {
using Chart = fdapde::manifold::internals::CheegerChart;
using Pair = fdapde::manifold::internals::CheegerPair;

/// @brief retains the pre-optimization enumeration and sixty-step bisection as an independent test oracle
Pair bisection_pair(Chart a, Chart b, double rho) {
    const double pi = std::acos(-1.), aa = std::hypot(a.x, a.y), ab = std::hypot(b.x, b.y), p = aa * ab, e = rho;
    const double angle = std::atan2(b.y, b.x) - std::atan2(a.y, a.x);
    const double delta = std::atan2(std::sin(angle), std::cos(angle)) / 2;
    const auto cost = [&](double phi) {
        return 2 * std::pow(b.s - a.s, 2) + 2 * std::pow(ab - aa, 2) + 8 * p * std::pow(std::sin(delta - phi), 2) +
               2 * e * phi * phi;
    };
    const auto gradient = [&](double phi) { return -8 * p * std::sin(2 * (delta - phi)) + 4 * e * phi; };
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
        for (int k = 0; k < 60; ++k) {
            const double mid = (lo + hi) / 2;
            if ((gradient(mid) > 0) == (gl > 0))
                lo = mid;
            else
                hi = mid;
        }
        candidates.push_back((lo + hi) / 2);
    }
    double best = std::numeric_limits<double>::infinity();
    for (double phi : candidates) best = std::min(best, cost(phi));
    std::sort(candidates.rbegin(), candidates.rend());
    Pair result {best, {}};
    for (double phi : candidates)
        if (
          cost(phi) - best <= 1e-12 * std::max(1., best) &&
          (result.rotations.empty() || std::abs(phi - result.rotations.back()) > 1e-8))
            result.rotations.push_back(phi);
    return result;
}

/// @brief compares retained branch count, order, distance and angle accuracy with the original pair solver
void expect_pair(Chart a, Chart b, double rho) {
    const Pair expected = bisection_pair(a, b, rho), actual = fdapde::manifold::internals::cheeger_pair(a, b, rho);
    // the unchanged candidate and tie policies retain exactly the same number of minimizing angles
    ASSERT_EQ(actual.rotations.size(), expected.rotations.size());
    // the public uniqueness report follows the retained branch count rather than analytic convexity alone
    EXPECT_EQ(actual.unique(), expected.unique());
    const double epsilon = std::numeric_limits<double>::epsilon();
    // minimizing distances agree at floating-point accuracy relative to the baseline objective scale
    EXPECT_NEAR(
      actual.squared_distance, expected.squared_distance, 32 * epsilon * std::max(1., expected.squared_distance));
    for (std::size_t i = 0; i < expected.rotations.size(); ++i) {
        // descending branch order and the root angle agree with the sixty-step bisection oracle
        EXPECT_NEAR(
          actual.rotations[i], expected.rotations[i], 32 * epsilon * std::max(1., std::abs(expected.rotations[i])));
    }
}

/// @brief builds a chart with known traceless amplitude, doubled orientation and trace coordinate
Chart chart(double amplitude, double angle, double trace = 0) {
    return {trace, amplitude * std::cos(angle), amplitude * std::sin(angle)};
}

// strictly convex pairs preserve original distances and minimizing angles over deterministic noncommuting charts
TEST(CheegerPair, ConvexRootsMatchBisection) {
    std::mt19937_64 generator(739);
    std::uniform_real_distribution<double> coordinate(-2., 2.);
    for (int i = 0; i < 128; ++i) {
        const Chart a {coordinate(generator), coordinate(generator), coordinate(generator)},
          b {coordinate(generator), coordinate(generator), coordinate(generator)};
        const double p = std::hypot(a.x, a.y) * std::hypot(b.x, b.y);
        for (double margin : {.001, .05, .5, 4., 100.})
            // the baseline oracle independently brackets the unique convex stationary point
            expect_pair(a, b, 4 * p * (1 + margin));
    }
}

// almost-flat curvature and nonconvex branches retain the original bisection and numerical ambiguity policy
TEST(CheegerPair, ConvexityThresholdAndBoundaryAngles) {
    const double pi = std::numbers::pi, infinity = std::numeric_limits<double>::infinity();
    for (double angle : {0., .7, pi, std::nextafter(pi, 0.), pi - 1e-14, -pi, -pi + 1e-14}) {
        const Chart a = chart(1, 0), b = chart(1, angle);
        const double threshold = 4 * std::hypot(a.x, a.y) * std::hypot(b.x, b.y);
        for (double rho :
             {std::nextafter(threshold, 0.), threshold, std::nextafter(threshold, infinity), threshold * (1 + 1e-10),
              2 * threshold})
            // crossing the curvature threshold must not invent, suppress or reorder minimizing branches
            expect_pair(a, b, rho);
        const double rho = std::nextafter(threshold, infinity);
        const Pair expected = bisection_pair(a, b, rho), actual = fdapde::manifold::internals::cheeger_pair(a, b, rho);
        // the immediately adjacent convex case uses the untouched bisection fallback exactly
        EXPECT_EQ(actual.rotations, expected.rotations);
    }
}

// scale changes and tiny costs preserve endpoint ties instead of claiming automatic uniqueness for convex pairs
TEST(CheegerPair, NumericalTiesAndScaledCharts) {
    const Chart isotropic {};
    const auto tied = fdapde::manifold::internals::cheeger_pair(isotropic, isotropic, 1e-14);
    // the original absolute cost floor retains both endpoints and the zero root for a tiny positive rho
    ASSERT_EQ(tied.rotations.size(), 3);
    // analytic strict convexity does not bypass the established numerical tie report
    EXPECT_FALSE(tied.unique());
    // an independent copy of the baseline solver checks the tied minimum and its ordered angles
    expect_pair(isotropic, isotropic, 1e-14);
    for (double scale : {1e-6, 1e-3, 1., 1e3, 1e6}) {
        const Chart a = chart(scale, .2, .3 * scale), b = chart(1.3 * scale, -2.4, -.7 * scale);
        for (double rho_ratio : {.1, 4., 8., 100.})
            // scaling chart amplitudes and rho together exercises the unchanged cost-relative tie threshold
            expect_pair(a, b, rho_ratio * scale * scale);
    }
    // a large trace residual keeps near-minimal rotation endpoints under the existing relative cost floor
    expect_pair(chart(1, 0, 1e8), chart(1, 1, -1e8), 10);
}

// extreme rho retains the original root discretization and any existing nonfinite outcome without new overflow
TEST(CheegerPair, ExtremeRhoRetainsBisectionFallback) {
    const Chart zero {}, tiny = chart(1e-100, .7);
    for (const Chart b : {zero, tiny})
        for (double rho : {1e16, 1e30, 1e300, std::numeric_limits<double>::max()}) {
            const Pair expected = bisection_pair(zero, b, rho),
                       actual = fdapde::manifold::internals::cheeger_pair(zero, b, rho);
            // the scale guard preserves the exact baseline angles instead of changing the tiny-angle root grid
            EXPECT_EQ(actual.rotations, expected.rotations);
            // extremely amplified or nonfinite baseline costs remain the same under the untouched fallback
            EXPECT_EQ(actual.squared_distance, expected.squared_distance);
        }
}
}   // namespace
