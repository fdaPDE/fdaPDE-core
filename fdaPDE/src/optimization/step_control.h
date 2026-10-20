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

#ifndef __FDAPDE_STEP_CONTROL_H__
#define __FDAPDE_STEP_CONTROL_H__
#include <span>

#include "header_check.h"

namespace fdapde {
/// @brief records a bounded Armijo search using an objective increment oracle
struct ArmijoStep {
    double step = 0, change = 0, bound = 0;
    int evaluations = 0;
    bool accepted = false;
};

/// @brief searches using f(x + step d) - f(x) supplied without subtracting full objective values
/// @details the oracle must evaluate the represented trial point and return infinity for invalid trials
template <typename Difference>
ArmijoStep armijo_step(
  Difference difference, double slope, double initial_step = 1, double contraction = .5,
  double sufficient_decrease = 1e-4, int max_trials = 30) {
    fdapde_strong_assert(
      std::isfinite(slope) && slope < 0 && std::isfinite(initial_step) && initial_step > 0 && contraction > 0 &&
        contraction < 1 && sufficient_decrease > 0 && sufficient_decrease < 1 && max_trials > 0,
      std::invalid_argument, "invalid Armijo parameters or non-descent direction");
    ArmijoStep result;
    result.step = initial_step;
    for (int i = 0; i < max_trials; ++i) {
        result.change = difference(result.step);
        ++result.evaluations;
        result.bound = sufficient_decrease * result.step * slope;
        if (std::isfinite(result.change) && result.bound < 0 && result.change <= result.bound) {
            result.accepted = true;
            return result;
        }
        if (i + 1 < max_trials) result.step *= contraction;
    }
    return result;
}

/// @brief records a safeguarded Wolfe search and its explicitly marked roundoff fallback
struct ApproximateWolfeStep : ArmijoStep {
    double derivative = 0, quadratic_change = 0, roundoff_allowance = 0;
    bool approximate = false;
};

/// @brief searches with ordinary Wolfe away from roundoff and derivative-based approximate Wolfe near it
/// @details the oracle returns the full cost and directional derivative; the fallback is not an exact cost increment
/// @see Hager and Zhang, ACM TOMS 32(1), 2006, equations 22, 23 and 27
template <typename Evaluate>
ApproximateWolfeStep approximate_wolfe_step(
  Evaluate evaluate, double value, double slope, double initial_step = 1, double maximum_step = 1,
  double sufficient_decrease = 1e-4, double curvature = .9, int max_trials = 30) {
    fdapde_strong_assert(
      std::isfinite(value) && std::isfinite(slope) && slope < 0 && std::isfinite(initial_step) && initial_step > 0 &&
        std::isfinite(maximum_step) && maximum_step >= initial_step && sufficient_decrease > 0 &&
        sufficient_decrease < .5 && curvature >= sufficient_decrease && curvature < 1 && max_trials > 0,
      std::invalid_argument, "invalid approximate Wolfe settings or non-descent direction");
    ApproximateWolfeStep result;
    result.step = initial_step;
    result.roundoff_allowance = 64 * std::numeric_limits<double>::epsilon() * std::max(1., std::abs(value));
    double lower = 0, lower_slope = slope, upper = std::numeric_limits<double>::infinity(), upper_slope = 0;
    bool upper_derivative = false;
    for (int trial = 0; trial < max_trials; ++trial) {
        const auto [candidate, derivative] = evaluate(result.step);
        ++result.evaluations;
        result.change = candidate - value;
        result.derivative = derivative;
        result.bound = sufficient_decrease * result.step * slope;
        result.quadratic_change = .5 * result.step * (slope + derivative);
        const bool finite = std::isfinite(candidate) && std::isfinite(derivative);
        const bool near_roundoff = finite && std::abs(result.change) <= result.roundoff_allowance;
        const bool weak_curvature = finite && derivative >= curvature * slope;
        const bool ordinary = !near_roundoff && result.change <= result.bound && weak_curvature;
        const bool approximate = near_roundoff && weak_curvature && derivative <= (2 * sufficient_decrease - 1) * slope;
        if (finite && result.bound < 0 && (ordinary || approximate)) {
            result.accepted = true;
            result.approximate = approximate;
            return result;
        }
        if (trial + 1 == max_trials) return result;
        // maintain a descent-side lower bound and an invalid, rising or insufficient-decrease upper bound
        if (!finite || (!near_roundoff && result.change > result.bound) || derivative >= 0) {
            upper = result.step;
            upper_slope = derivative;
            upper_derivative = finite;
        } else {
            lower = result.step;
            lower_slope = derivative;
        }
        if (std::isfinite(upper)) {
            const double width = upper - lower;
            double next = lower + .5 * width;
            if (upper_derivative && upper_slope > lower_slope)
                next = std::clamp(
                  lower - lower_slope * width / (upper_slope - lower_slope), lower + .1 * width, upper - .1 * width);
            if (!(next > lower && next < upper)) return result;
            result.step = next;
        } else {
            const double next = std::min(2 * result.step, maximum_step);
            if (!(next > result.step)) return result;
            result.step = next;
        }
    }
    return result;
}

/// @brief identifies the coefficient controlling the raw and lumped mass dual infinity norms
struct LumpedGradientSummary {
    double raw_inf = 0, density_inf = 0, volume = 0;
    std::size_t raw_index = 0, density_index = 0;
};

/// @brief measures interleaved nodal covectors without rescaling their stopping tolerances
/// @details density has inverse volume units and changes under a change of spatial coordinate unit
inline LumpedGradientSummary
lumped_gradient_summary(std::span<const double> gradient, std::span<const double> masses, std::size_t components = 1) {
    fdapde_strong_assert(
      components > 0 && !masses.empty() && gradient.size() / components == masses.size() &&
        gradient.size() % components == 0,
      std::invalid_argument, "incompatible lumped gradient dimensions");
    LumpedGradientSummary result;
    CompensatedSum volume;
    for (std::size_t i = 0; i < masses.size(); ++i) {
        fdapde_strong_assert(std::isfinite(masses[i]) && masses[i] > 0, std::invalid_argument, "invalid lumped mass");
        volume.add(masses[i]);
        for (std::size_t a = 0; a < components; ++a) {
            const std::size_t index = components * i + a;
            fdapde_strong_assert(std::isfinite(gradient[index]), std::invalid_argument, "nonfinite gradient");
            const double magnitude = std::abs(gradient[index]);
            if (magnitude > result.raw_inf) {
                result.raw_inf = magnitude;
                result.raw_index = index;
            }
            const double density = magnitude / masses[i];
            fdapde_strong_assert(std::isfinite(density), std::overflow_error, "nonfinite gradient density");
            if (density > result.density_inf) {
                result.density_inf = density;
                result.density_index = index;
            }
        }
    }
    result.volume = volume.value();
    return result;
}
}   // namespace fdapde
#endif
