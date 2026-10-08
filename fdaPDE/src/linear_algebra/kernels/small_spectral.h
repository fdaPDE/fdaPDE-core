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

#include <cmath>
#include <stdexcept>
#include <type_traits>

#include "../header_check.h"

namespace fdapde {
namespace internals {

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
    const auto eigenvectors = spectral.eigenvectors();
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
