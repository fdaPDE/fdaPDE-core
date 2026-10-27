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

#ifndef __FDAPDE_LINALG_KERNELS_SMALL_EVD3_H__
#define __FDAPDE_LINALG_KERNELS_SMALL_EVD3_H__

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

#include "../header_check.h"

namespace fdapde {
namespace internals {

/// @brief pairs three scalar coordinates with fused multiply-add accumulation
inline double small_evd3_dot(const std::array<double, 3>& a, const std::array<double, 3>& b) {
    return std::fma(a[0], b[0], std::fma(a[1], b[1], a[2] * b[2]));
}
/// @brief forms a three-dimensional cross product without dynamic vector expressions
inline auto small_evd3_cross(const std::array<double, 3>& a, const std::array<double, 3>& b) {
    return std::array<double, 3> {
      std::fma(a[1], b[2], -a[2] * b[1]), std::fma(a[2], b[0], -a[0] * b[2]), std::fma(a[0], b[1], -a[1] * b[0])};
}
/// @brief refines a cancellation-sensitive eigenvalue through compensated scalar triple products
template <int Dimension>
inline double
small_evd_rayleigh(const double (&matrix)[Dimension][Dimension], const std::array<double, Dimension>& vector) {
    double high = 0, low = 0;
    auto append = [&](double x, double y, double z) {
        const double first = x * y, error = std::fma(x, y, -first);
        const double term = first * z, remainder = std::fma(first, z, -term) + error * z;
        const double next = high + term, offset = next - high;
        low += (high - (next - offset)) + (term - offset) + remainder;
        high = next;
    };
    for (int i = 0; i < Dimension; ++i) {
        append(vector[i], vector[i], matrix[i][i]);
        for (int j = i + 1; j < Dimension; ++j) append(vector[i], vector[j], 2 * matrix[i][j]);
    }
    if constexpr (Dimension == 3)
        return (high + low) / small_evd3_dot(vector, vector);
    else
        return (high + low) / std::fma(vector[0], vector[0], vector[1] * vector[1]);
}
/// @brief diagonalizes an isolated analytic eigenvector's orthogonal complement and certifies the eigenpairs
/// @details power-of-two scaling preserves input coefficients; compensated Rayleigh quotients refine small eigenvalues
inline bool
small_symmetric_evd3(const double (&matrix)[3][3], double scale, double* output_vectors, double* output_values) {
    if (matrix[0][1] == 0 && matrix[0][2] == 0 && matrix[1][2] == 0) {
        for (int i = 0; i < 3; ++i) {
            output_values[i] = matrix[i][i];
            for (int j = 0; j < 3; ++j) output_vectors[3 * i + j] = double(i == j);
        }
        return true;
    }
    const double normalization = std::scalbn(1., std::ilogb(scale));
    double normalized[3][3], centered[3][3];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) normalized[i][j] = matrix[i][j] / normalization;
    const double first = normalized[0][0] - normalized[2][2], second = normalized[1][1] - normalized[2][2];
    const double offset = (first + second) / 3, shift = normalized[2][2] + offset;
    double spread = 0;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            centered[i][j] = i == j ? (i == 0 ? first - offset : i == 1 ? second - offset : -offset) : normalized[i][j];
            spread += centered[i][j] * centered[i][j];
        }
    const double radius = std::sqrt(spread / 6);
    if (!(radius > 0)) return false;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) centered[i][j] /= radius;
    const double determinant =
      centered[0][0] * (centered[1][1] * centered[2][2] - centered[1][2] * centered[1][2]) +
      centered[0][1] * (2 * centered[1][2] * centered[0][2] - centered[0][1] * centered[2][2]) -
      centered[0][2] * centered[0][2] * centered[1][1];
    const double cosine = std::clamp(.5 * determinant, -1., 1.);
    const double phase = std::acos(cosine) / 3;
    const double isolated = 2 * std::cos(phase + (cosine < 0 ? 2.0943951023931954923 : 0.));
    std::array<std::array<double, 3>, 3> rows;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) rows[i][j] = centered[i][j] - (i == j ? isolated : 0.);
    const std::array candidates {
      small_evd3_cross(rows[0], rows[1]), small_evd3_cross(rows[0], rows[2]), small_evd3_cross(rows[1], rows[2])};
    int pivot = 0;
    for (int i = 1; i < 3; ++i)
        if (small_evd3_dot(candidates[i], candidates[i]) > small_evd3_dot(candidates[pivot], candidates[pivot]))
            pivot = i;
    std::array<double, 3> axis = candidates[pivot], u;
    const double norm = std::sqrt(small_evd3_dot(axis, axis));
    if (!(norm > 0)) return false;
    for (double& coefficient : axis) coefficient /= norm;
    if (std::abs(axis[0]) > std::abs(axis[1])) {
        const double inverse = 1 / std::sqrt(axis[0] * axis[0] + axis[2] * axis[2]);
        u = {-axis[2] * inverse, 0, axis[0] * inverse};
    } else {
        const double inverse = 1 / std::sqrt(axis[1] * axis[1] + axis[2] * axis[2]);
        u = {0, axis[2] * inverse, -axis[1] * inverse};
    }
    const auto v = small_evd3_cross(axis, u);
    std::array<double, 3> image_u, image_v;
    for (int i = 0; i < 3; ++i) {
        image_u[i] = small_evd3_dot(rows[i], u);
        image_v[i] = small_evd3_dot(rows[i], v);
    }
    const double upper = small_evd3_dot(u, image_u), lower = small_evd3_dot(v, image_v),
                 off = small_evd3_dot(u, image_v);
    double tangent = 0;
    if (std::abs(off) > std::numeric_limits<double>::epsilon()) {
        const double tau = (lower - upper) / (2 * off);
        tangent = std::copysign(1., tau) / (std::abs(tau) + std::sqrt(1 + tau * tau));
    }
    const double c = 1 / std::sqrt(1 + tangent * tangent), s = tangent * c;
    std::array<std::array<double, 3>, 3> vectors;
    for (int i = 0; i < 3; ++i) {
        vectors[0][i] = axis[i];
        vectors[1][i] = c * u[i] - s * v[i];
        vectors[2][i] = s * u[i] + c * v[i];
    }
    double values[3] {
      shift + radius * isolated, shift + radius * (upper - tangent * off + isolated),
      shift + radius * (lower + tangent * off + isolated)};
    for (int col = 0; col < 3; ++col) {
        std::array<double, 3> image;
        for (int i = 0; i < 3; ++i)
            image[i] = small_evd3_dot({normalized[i][0], normalized[i][1], normalized[i][2]}, vectors[col]);
        values[col] = std::abs(values[col]) < .1 * scale / normalization ?
                        small_evd_rayleigh<3>(normalized, vectors[col]) :
                        small_evd3_dot(image, vectors[col]) / small_evd3_dot(vectors[col], vectors[col]);
        for (int i = 0; i < 3; ++i)
            if (!(std::abs(image[i] - values[col] * vectors[col][i]) <= 32 * std::numeric_limits<double>::epsilon()))
                return false;
        if (!std::isfinite(values[col] * normalization)) return false;
    }
    for (int col = 0; col < 3; ++col) {
        output_values[col] = values[col] * normalization;
        for (int row = 0; row < 3; ++row) output_vectors[3 * row + col] = vectors[col][row];
    }
    return true;
}
}   // namespace internals
}   // namespace fdapde
#endif
