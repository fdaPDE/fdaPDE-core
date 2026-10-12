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

#ifndef __FDAPDE_LINALG_KERNELS_SMALL_SPECTRAL_H__
#define __FDAPDE_LINALG_KERNELS_SMALL_SPECTRAL_H__

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <type_traits>

#include "../header_check.h"

namespace fdapde {
namespace internals {

/// @brief selects the scalar spectral transform used by the eigenvector-free two-dimensional kernel
enum class SmallSymmetricFunction {
    Log,
    Exp,
    Sqrt,
    InverseSqrt
};

/// @brief evaluates a two-dimensional symmetric function with stable repeated-eigenvalue limits
/// @details verified SPD input is required except for the exponential, whose caller checks finite symmetry
template <SmallSymmetricFunction Function, typename Xpr> auto small_symmetric_function_2(const Xpr& value) {
    using Scalar = std::remove_cv_t<typename Xpr::Scalar>;
    SymmetricMatrix<Scalar, 2> result;
    if constexpr (Function == SmallSymmetricFunction::Exp) {
        const Scalar center = Scalar(.5) * value(0, 0) + Scalar(.5) * value(1, 1);
        const Scalar difference = Scalar(.5) * value(0, 0) - Scalar(.5) * value(1, 1);
        const Scalar radius = std::hypot(difference, value(1, 0));
        const Scalar upper = center + radius, lower = center - radius;
        const Scalar positive = std::exp(upper), negative = std::exp(lower);
        fdapde_strong_assert(
          std::isfinite(positive) && std::isfinite(negative) && positive > Scalar(0) &&
            negative > Scalar(128) * std::numeric_limits<Scalar>::epsilon() * positive,
          std::domain_error, "SPDMatrix: exponential is not numerically positive definite");
        if (radius == Scalar(0)) {
            result(0, 0) = result(1, 1) = positive;
            result(1, 0) = Scalar(0);
        } else {
            // nonnegative spectral weights retain the minor diagonal without subtracting nearly equal half-sums
            const Scalar relative_off_diagonal = value(1, 0) / radius;
            const Scalar minor =
              Scalar(.5) * relative_off_diagonal * relative_off_diagonal / (Scalar(1) + std::abs(difference) / radius);
            const Scalar first = difference >= Scalar(0) ? Scalar(1) - minor : minor;
            const Scalar second = difference >= Scalar(0) ? minor : Scalar(1) - minor;
            const Scalar gap = positive - negative;
            const Scalar slope = positive * (-std::expm1(-Scalar(2) * radius) / (Scalar(2) * radius));
            result(0, 0) = negative + gap * first;
            result(1, 1) = negative + gap * second;
            result(1, 0) = slope * value(1, 0);
        }
    } else {
        const Scalar magnitude = std::max({std::abs(value(0, 0)), std::abs(value(1, 0)), std::abs(value(1, 1))});
        // binary scaling preserves coefficients before the compensated determinant is formed
        const Scalar scale = std::scalbn(Scalar(1), std::ilogb(magnitude));
        const Scalar a = value(0, 0) / scale, b = value(1, 0) / scale, d = value(1, 1) / scale;
        const Scalar center = Scalar(.5) * a + Scalar(.5) * d;
        const Scalar difference = Scalar(.5) * a - Scalar(.5) * d;
        const Scalar radius = std::hypot(difference, b);
        const Scalar upper = center + radius;
        const Scalar square = b * b;
        const Scalar determinant = std::fma(a, d, -square) + std::fma(-b, b, square);
        const Scalar lower = determinant / upper;
        [[maybe_unused]] Scalar average, slope;
        if constexpr (Function == SmallSymmetricFunction::Log) {
            average = std::log(scale) + Scalar(.5) * std::log(upper) + Scalar(.5) * std::log(lower);
            if (radius <= Scalar(.125) * center) {
                const Scalar ratio = radius / center, squared = ratio * ratio;
                // atanh(r)/r avoids subtracting logarithms in a nearly repeated spectrum
                const Scalar relative =
                  ratio == Scalar(0) ? Scalar(1) : Scalar(.5) * (std::log1p(ratio) - std::log1p(-ratio)) / ratio;
                slope = relative / center;
                average = std::log(scale) + std::log(center) + Scalar(.5) * std::log1p(-squared);
            } else
                slope = (std::log(upper) - std::log(lower)) / (Scalar(2) * radius);
        } else {
            const Scalar root_determinant = std::sqrt(determinant);
            const Scalar denominator = std::sqrt(a + d + Scalar(2) * root_determinant);
            const Scalar factor = Function == SmallSymmetricFunction::Sqrt ?
                                    std::sqrt(scale) / denominator :
                                    Scalar(1) / (std::sqrt(scale) * root_determinant * denominator);
            if constexpr (Function == SmallSymmetricFunction::Sqrt) {
                result(0, 0) = factor * (a + root_determinant);
                result(1, 1) = factor * (d + root_determinant);
                result(1, 0) = factor * b;
            } else {
                result(0, 0) = factor * (d + root_determinant);
                result(1, 1) = factor * (a + root_determinant);
                result(1, 0) = -factor * b;
            }
        }
        if constexpr (Function == SmallSymmetricFunction::Log) {
            result(0, 0) = average + slope * difference;
            result(1, 1) = average - slope * difference;
            result(1, 0) = slope * b;
        }
    }
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j <= i; ++j)
            fdapde_strong_assert(
              std::isfinite(result(i, j)), std::domain_error, "SPD spectral operation: nonfinite result");
    return result;
}

/// @brief applies symmetric divided differences through fixed scalar workspaces for a checked small direction
/// @details the caller validates the direction and supplies divided differences symmetric in the eigenpair indices
template <int Dimension, typename Spectral, typename Direction, typename DividedDifference>
auto small_frechet_symmetric(
  const Spectral& spectral, const Direction& direction, DividedDifference&& divided_difference) {
    using Scalar = std::remove_cv_t<typename Spectral::Scalar>;
    Scalar q[Dimension][Dimension];
    Scalar h[Dimension][Dimension];
    Scalar hq[Dimension][Dimension];
    Scalar coefficients[Dimension][Dimension];
    Scalar q_coefficients[Dimension][Dimension];
    const auto& eigenvectors = spectral.eigenvectors();
    const auto& eigenvalues = spectral.eigenvalues();
    for (int i = 0; i < Dimension; ++i) {
        for (int j = 0; j < Dimension; ++j) q[i][j] = eigenvectors(i, j);
        for (int j = 0; j <= i; ++j) h[i][j] = h[j][i] = static_cast<Scalar>(direction(i, j));
    }

    // symmetric Loewner coefficients need one evaluation per lower-triangular entry
    for (int i = 0; i < Dimension; ++i) {
        for (int j = 0; j < Dimension; ++j) {
            Scalar value = Scalar(0);
            for (int k = 0; k < Dimension; ++k) value += h[i][k] * q[k][j];
            hq[i][j] = value;
        }
    }
    for (int i = 0; i < Dimension; ++i) {
        for (int j = 0; j <= i; ++j) {
            Scalar value = Scalar(0);
            for (int k = 0; k < Dimension; ++k) value += q[k][i] * hq[k][j];
            if constexpr (std::is_invocable_v<DividedDifference, Scalar, Scalar, int, int>)
                value *= divided_difference(eigenvalues[i], eigenvalues[j], i, j);
            else
                value *= divided_difference(eigenvalues[i], eigenvalues[j]);
            fdapde_strong_assert(
              std::isfinite(value), std::domain_error, "SPD spectral operation: nonfinite Frechet derivative");
            coefficients[i][j] = coefficients[j][i] = value;
        }
    }
    for (int i = 0; i < Dimension; ++i) {
        for (int j = 0; j < Dimension; ++j) {
            Scalar value = Scalar(0);
            for (int k = 0; k < Dimension; ++k) value += q[i][k] * coefficients[k][j];
            q_coefficients[i][j] = value;
        }
    }
    SymmetricMatrix<Scalar, Dimension> result;
    for (int i = 0; i < Dimension; ++i) {
        for (int j = 0; j <= i; ++j) {
            Scalar value = Scalar(0);
            for (int k = 0; k < Dimension; ++k) value += q_coefficients[i][k] * q[j][k];
            fdapde_strong_assert(
              std::isfinite(value), std::domain_error, "SPD spectral operation: nonfinite Frechet derivative");
            result(i, j) = value;
        }
    }
    return result;
}

}   // namespace internals
}   // namespace fdapde

#endif   // __FDAPDE_LINALG_KERNELS_SMALL_SPECTRAL_H__
