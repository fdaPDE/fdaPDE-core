// This file is part of fdaPDE, a C++ library for physics-informed
// spatial and functional data analysis.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

#ifndef __FDAPDE_LINALG_SPECTRAL_H__
#define __FDAPDE_LINALG_SPECTRAL_H__

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>

#include "header_check.h"

namespace fdapde {

namespace internals {

/// @brief materializes arithmetic input as double after checking square shape, finite values and relative symmetry
template <typename XprType_> auto spectral_matrix(const MatrixExpr<XprType_>& matrix) {
    using XprType = std::decay_t<XprType_>;
    using Scalar = std::remove_cv_t<typename XprType::Scalar>;
    static_assert(std::is_arithmetic_v<Scalar>, "spectral matrix functions require arithmetic coefficients");

    constexpr int Rows = XprType::Rows;
    constexpr int Cols = XprType::Cols;
    const auto& input = matrix.derived();
    fdapde_strong_assert(
      input.rows() > 0 && input.rows() == input.cols(), std::invalid_argument,
      "spectral matrix function requires a nonempty square matrix");
    const std::int64_t dimension = input.rows();
    fdapde_strong_assert(
      dimension * dimension <= std::numeric_limits<int>::max(), std::length_error,
      "spectral matrix function dense workspace exceeds supported range");

    Matrix<double, Rows, Cols> dense(input);
    double scale = 0.0;
    for (int i = 0; i < input.rows(); ++i) {
        for (int j = 0; j < input.cols(); ++j) {
            const double value = dense(i, j);
            fdapde_strong_assert(
              std::isfinite(value), std::invalid_argument, "spectral matrix function requires finite coefficients");
            scale = std::max(scale, std::abs(value));
        }
    }
    const double tolerance = 32.0 * static_cast<double>(input.rows()) * std::numeric_limits<double>::epsilon() * scale;
    for (int i = 0; i < input.rows(); ++i) {
        for (int j = 0; j < i; ++j) {
            fdapde_strong_assert(
              std::abs(dense(i, j) - dense(j, i)) <= tolerance, std::invalid_argument,
              "spectral matrix function requires a symmetric matrix");
        }
    }
    return dense;
}

/// @brief transforms a symmetric eigenspectrum and reconstructs an owned double matrix with the input static shape
template <typename XprType_, typename UnaryOp_>
auto apply_spectral_function(const MatrixExpr<XprType_>& matrix, UnaryOp_&& operation) {
    using XprType = std::decay_t<XprType_>;
    constexpr int Rows = XprType::Rows;
    constexpr int Cols = XprType::Cols;

    if constexpr (Rows != Dynamic && Cols != Dynamic && Rows != Cols) {
        fdapde_strong_assert(false, std::invalid_argument, "spectral matrix function requires a square matrix");
    } else {
        auto dense = spectral_matrix(matrix);
        const int dimension = dense.rows();
        const EVD decomposition(dense.template as_symmetric<Lower>());
        fdapde_strong_assert(decomposition.computed(), std::domain_error, "spectral eigendecomposition failed");

        double spectrum_scale = 0.0;
        for (int i = 0; i < dimension; ++i) {
            const double eigenvalue = decomposition.eigenvalues()[i];
            fdapde_strong_assert(std::isfinite(eigenvalue), std::domain_error, "spectral eigendecomposition failed");
            spectrum_scale = std::max(spectrum_scale, std::abs(eigenvalue));
        }
        const double spectrum_tolerance =
          64.0 * static_cast<double>(dimension) * std::numeric_limits<double>::epsilon() * spectrum_scale;

        Vector<double, Rows> eigenvalues;
        if constexpr (Rows == Dynamic) { eigenvalues.resize(dimension); }
        for (int i = 0; i < dimension; ++i) {
            eigenvalues[i] = operation(decomposition.eigenvalues()[i], spectrum_tolerance);
            fdapde_strong_assert(
              std::isfinite(eigenvalues[i]), std::domain_error, "spectral matrix function produced a nonfinite result");
        }

        Matrix<double, Rows, Cols> result;
        if constexpr (Rows == Dynamic || Cols == Dynamic) { result.resize(dimension, dimension); }
        const auto eigenvectors = decomposition.eigenvectors();
        for (int i = 0; i < dimension; ++i) {
            for (int j = 0; j < dimension; ++j) {
                double value = 0.0;
                for (int k = 0; k < dimension; ++k) {
                    value += eigenvectors(i, k) * eigenvalues[k] * eigenvectors(j, k);
                }
                fdapde_strong_assert(
                  std::isfinite(value), std::domain_error, "spectral matrix function produced a nonfinite result");
                result(i, j) = value;
            }
        }
        return result;
    }
}

}   // namespace internals

/// @brief returns the real symmetric matrix logarithm as an owned double matrix
/// @details requires eigenvalues above 64 * dimension * double epsilon * spectral radius
template <typename XprType_> auto logm(const MatrixExpr<XprType_>& matrix) {
    return internals::apply_spectral_function(matrix, [](double eigenvalue, double tolerance) {
        fdapde_strong_assert(eigenvalue > tolerance, std::domain_error, "logm requires a positive definite matrix");
        return std::log(eigenvalue);
    });
}

/// @brief returns the real symmetric matrix exponential as an owned double matrix, rejecting nonfinite results
template <typename XprType_> auto expm(const MatrixExpr<XprType_>& matrix) {
    return internals::apply_spectral_function(matrix, [](double eigenvalue, double) { return std::exp(eigenvalue); });
}

/// @brief returns an integer power of a real symmetric matrix as an owned double matrix
/// @details negative powers require every eigenvalue magnitude above the relative spectral tolerance
template <typename XprType_> auto powm(const MatrixExpr<XprType_>& matrix, int exponent) {
    return internals::apply_spectral_function(matrix, [exponent](double eigenvalue, double tolerance) {
        fdapde_strong_assert(
          exponent >= 0 || std::abs(eigenvalue) > tolerance, std::domain_error,
          "powm with a negative exponent requires an invertible matrix");
        return std::pow(eigenvalue, exponent);
    });
}

/// @brief returns the principal square root of a real symmetric positive-semidefinite matrix
/// @details negative eigenvalues within the relative spectral tolerance are clamped to zero; others are rejected
template <typename XprType_> auto sqrtm(const MatrixExpr<XprType_>& matrix) {
    return internals::apply_spectral_function(matrix, [](double eigenvalue, double tolerance) {
        fdapde_strong_assert(
          eigenvalue >= -tolerance, std::domain_error, "sqrtm requires a positive semidefinite matrix");
        return std::sqrt(std::max(eigenvalue, 0.0));
    });
}

}   // namespace fdapde

#endif   // __FDAPDE_LINALG_SPECTRAL_H__
