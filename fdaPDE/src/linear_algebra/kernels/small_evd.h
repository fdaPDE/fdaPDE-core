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

#ifndef __FDAPDE_LINALG_KERNELS_SMALL_EVD_H__
#define __FDAPDE_LINALG_KERNELS_SMALL_EVD_H__

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>

#include "../header_check.h"

namespace fdapde {
namespace internals {

/// @brief annihilates one normalized pivot and updates its matching eigenvector columns
/// @details the pivot exceeds dimension times epsilon, keeping the squared rotation ratio finite
template <int Dimension, int P, int Q, typename Scalar>
constexpr void
small_jacobi_rotation(Scalar (&diagonalized)[Dimension][Dimension], Scalar (&eigenvectors)[Dimension][Dimension]) {
    const Scalar app = diagonalized[P][P];
    const Scalar aqq = diagonalized[Q][Q];
    const Scalar apq = diagonalized[P][Q];
    const Scalar tau = (aqq - app) / (Scalar(2) * apq);
    const Scalar tangent = std::copysign(Scalar(1), tau) / (std::abs(tau) + std::sqrt(Scalar(1) + tau * tau));
    const Scalar cosine = Scalar(1) / std::sqrt(Scalar(1) + tangent * tangent);
    const Scalar sine = tangent * cosine;

    diagonalized[P][P] = app - tangent * apq;
    diagonalized[Q][Q] = aqq + tangent * apq;
    diagonalized[P][Q] = diagonalized[Q][P] = Scalar(0);
    if constexpr (Dimension == 3) {
        constexpr int R = 3 - P - Q;
        const Scalar arp = diagonalized[R][P];
        const Scalar arq = diagonalized[R][Q];
        diagonalized[R][P] = diagonalized[P][R] = cosine * arp - sine * arq;
        diagonalized[R][Q] = diagonalized[Q][R] = sine * arp + cosine * arq;
    }
    for (int row = 0; row < Dimension; ++row) {
        const Scalar erp = eigenvectors[row][P];
        const Scalar erq = eigenvectors[row][Q];
        eigenvectors[row][P] = cosine * erp - sine * erq;
        eigenvectors[row][Q] = sine * erp + cosine * erq;
    }
}

/// @brief computes scaled fixed two- or three-dimensional eigenpairs through scalar workspaces
/// @details publishes row-major eigenvectors and matching unsorted eigenvalues only after successful convergence
template <int Dimension, typename MatrixType, typename Scalar>
constexpr void small_symmetric_evd(const MatrixType& matrix, Scalar* output_vectors, Scalar* output_values) {
    Scalar diagonalized[Dimension][Dimension];
    Scalar eigenvectors[Dimension][Dimension] {};
    Scalar matrix_scale = Scalar(0);
    for (int row = 0; row < Dimension; ++row) {
        for (int col = 0; col < Dimension; ++col) {
            const Scalar value = static_cast<Scalar>(matrix(row, col));
            fdapde_strong_assert(
              std::isfinite(value), std::invalid_argument, "EVD requires finite matrix coefficients");
            diagonalized[row][col] = value;
            matrix_scale = std::max(matrix_scale, std::abs(value));
        }
    }
    const Scalar normalization = matrix_scale == Scalar(0) ? Scalar(1) : matrix_scale;
    for (int row = 0; row < Dimension; ++row) {
        for (int col = 0; col < Dimension; ++col) diagonalized[row][col] /= normalization;
        eigenvectors[row][row] = Scalar(1);
    }
    const Scalar tolerance =
      std::max(std::numeric_limits<Scalar>::min(), std::numeric_limits<Scalar>::epsilon() * Scalar(Dimension));
    if constexpr (Dimension == 2) {
        if (std::abs(diagonalized[0][1]) > tolerance) small_jacobi_rotation<2, 0, 1>(diagonalized, eigenvectors);
    } else {
        for (std::size_t iteration = 0;; ++iteration) {
            const auto a01 = std::abs(diagonalized[0][1]);
            const auto a02 = std::abs(diagonalized[0][2]);
            const auto a12 = std::abs(diagonalized[1][2]);
            if (a01 <= tolerance && a02 <= tolerance && a12 <= tolerance) break;
            fdapde_strong_assert(iteration != 450, std::runtime_error, "EVD: Jacobi iteration did not converge");
            if (a01 >= a02 && a01 >= a12)
                small_jacobi_rotation<3, 0, 1>(diagonalized, eigenvectors);
            else if (a02 >= a12)
                small_jacobi_rotation<3, 0, 2>(diagonalized, eigenvectors);
            else
                small_jacobi_rotation<3, 1, 2>(diagonalized, eigenvectors);
        }
    }
    for (int row = 0; row < Dimension; ++row) {
        output_values[row] = diagonalized[row][row] * normalization;
        for (int col = 0; col < Dimension; ++col) output_vectors[row * Dimension + col] = eigenvectors[row][col];
    }
}

}   // namespace internals
}   // namespace fdapde

#endif   // __FDAPDE_LINALG_KERNELS_SMALL_EVD_H__
