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

template <typename XprType_> auto spectral_matrix(const MatrixExpr<XprType_>& matrix) {
    using XprType = std::decay_t<XprType_>;
    using Scalar = std::remove_cv_t<typename XprType::Scalar>;
    static_assert(std::is_arithmetic_v<Scalar>, "spectral matrix functions require arithmetic coefficients");

    constexpr int Rows = XprType::Rows;
    constexpr int Cols = XprType::Cols;
    const auto& input = matrix.derived();
    if (input.rows() <= 0 || input.rows() != input.cols()) {
        throw std::invalid_argument("spectral matrix function requires a nonempty square matrix");
    }
    const std::int64_t dimension = input.rows();
    if (dimension * dimension > std::numeric_limits<int>::max()) {
        throw std::length_error("spectral matrix function dense workspace exceeds supported range");
    }

    Matrix<double, Rows, Cols> dense(input);
    double scale = 0.0;
    for (int i = 0; i < input.rows(); ++i) {
        for (int j = 0; j < input.cols(); ++j) {
            const double value = dense(i, j);
            if (!std::isfinite(value)) {
                throw std::invalid_argument("spectral matrix function requires finite coefficients");
            }
            scale = std::max(scale, std::abs(value));
        }
    }
    const double tolerance = 32.0 * static_cast<double>(input.rows()) * std::numeric_limits<double>::epsilon() * scale;
    for (int i = 0; i < input.rows(); ++i) {
        for (int j = 0; j < i; ++j) {
            if (std::abs(dense(i, j) - dense(j, i)) > tolerance) {
                throw std::invalid_argument("spectral matrix function requires a symmetric matrix");
            }
        }
    }
    return dense;
}

template <typename XprType_, typename UnaryOp_>
auto apply_spectral_function(const MatrixExpr<XprType_>& matrix, UnaryOp_&& operation) {
    using XprType = std::decay_t<XprType_>;
    constexpr int Rows = XprType::Rows;
    constexpr int Cols = XprType::Cols;

    if constexpr (Rows != Dynamic && Cols != Dynamic && Rows != Cols) {
        throw std::invalid_argument("spectral matrix function requires a square matrix");
    } else {
        auto dense = spectral_matrix(matrix);
        const int dimension = dense.rows();
        const EVD decomposition(dense.template as_symmetric<Lower>());
        if (!decomposition.computed()) { throw std::domain_error("spectral eigendecomposition failed"); }

        double spectrum_scale = 0.0;
        for (int i = 0; i < dimension; ++i) {
            const double eigenvalue = decomposition.eigenvalues()[i];
            if (!std::isfinite(eigenvalue)) { throw std::domain_error("spectral eigendecomposition failed"); }
            spectrum_scale = std::max(spectrum_scale, std::abs(eigenvalue));
        }
        const double spectrum_tolerance =
          64.0 * static_cast<double>(dimension) * std::numeric_limits<double>::epsilon() * spectrum_scale;

        Vector<double, Rows> eigenvalues;
        if constexpr (Rows == Dynamic) { eigenvalues.resize(dimension); }
        for (int i = 0; i < dimension; ++i) {
            eigenvalues[i] = operation(decomposition.eigenvalues()[i], spectrum_tolerance);
            if (!std::isfinite(eigenvalues[i])) {
                throw std::domain_error("spectral matrix function produced a nonfinite result");
            }
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
                if (!std::isfinite(value)) {
                    throw std::domain_error("spectral matrix function produced a nonfinite result");
                }
                result(i, j) = value;
            }
        }
        return result;
    }
}

}   // namespace internals

template <typename XprType_> auto logm(const MatrixExpr<XprType_>& matrix) {
    return internals::apply_spectral_function(matrix, [](double eigenvalue, double tolerance) {
        if (!(eigenvalue > tolerance)) { throw std::domain_error("logm requires a positive definite matrix"); }
        return std::log(eigenvalue);
    });
}

template <typename XprType_> auto expm(const MatrixExpr<XprType_>& matrix) {
    return internals::apply_spectral_function(matrix, [](double eigenvalue, double) { return std::exp(eigenvalue); });
}

template <typename XprType_> auto powm(const MatrixExpr<XprType_>& matrix, int exponent) {
    return internals::apply_spectral_function(matrix, [exponent](double eigenvalue, double tolerance) {
        if (exponent < 0 && std::abs(eigenvalue) <= tolerance) {
            throw std::domain_error("powm with a negative exponent requires an invertible matrix");
        }
        return std::pow(eigenvalue, exponent);
    });
}

template <typename XprType_> auto sqrtm(const MatrixExpr<XprType_>& matrix) {
    return internals::apply_spectral_function(matrix, [](double eigenvalue, double tolerance) {
        if (eigenvalue < -tolerance) { throw std::domain_error("sqrtm requires a positive semidefinite matrix"); }
        return std::sqrt(std::max(eigenvalue, 0.0));
    });
}

template <typename XprType_> constexpr bool is_empty(const MatrixExpr<XprType_>& matrix) {
    const auto& value = matrix.derived();
    return value.rows() == 0 || value.cols() == 0;
}

template <typename Scalar_> constexpr bool is_empty(const SparseMatrix<Scalar_>& matrix) {
    return matrix.rows() == 0 || matrix.cols() == 0;
}

}   // namespace fdapde

#endif   // __FDAPDE_LINALG_SPECTRAL_H__
