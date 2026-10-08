// SPDX-License-Identifier: GPL-3.0-or-later
#ifndef __FDAPDE_CHEEGER_DIFFERENTIAL_H__
#define __FDAPDE_CHEEGER_DIFFERENTIAL_H__
#include "../../header_check.h"
#include "../spd_geometry_common.h"

namespace fdapde::manifold::internals {
/// @brief divides a symmetric commutator numerator without overflowing the reciprocal of a small rho
template <typename S, int N> SymmetricMatrix<S, N> cheeger_divide(const SymmetricMatrix<S, N>& numerator, double rho) {
    auto result = make_symmetric<S, N>(numerator.rows());
    for (int i = 0; i < numerator.rows(); ++i)
        for (int j = i; j < numerator.rows(); ++j)
            result(i, j) = checked_geometry_result(S(static_cast<long double>(numerator(i, j)) / rho));
    return result;
}

/// @brief applies the C-LE log-coordinate metric using a symmetric spectral frame
template <typename S, int N, typename X>
SymmetricMatrix<S, N> cheeger_metric(const X& x, const SymmetricMatrix<S, N>& h, double rho) {
    const EVD evd(x);
    const auto& q = evd.eigenvectors();
    const Matrix<S, N, N> local(q.transpose() * h * q);
    Matrix<S, N, N> scaled(local);
    for (int i = 0; i < x.rows(); ++i)
        for (int j = 0; j < x.rows(); ++j) {
            const double gap = double(evd.eigenvalues()[i]) - double(evd.eigenvalues()[j]);
            scaled(i, j) *= S(rho / (rho + gap * gap));
        }
    return SymmetricMatrix<S, N>((q * scaled * q.transpose()).template as_symmetric<Lower>());
}

/// @brief applies the inverse C-LE log metric without differentiating spectral eigenvectors
template <typename S, int N>
SymmetricMatrix<S, N>
cheeger_inverse_metric(const SymmetricMatrix<S, N>& x, const SymmetricMatrix<S, N>& h, double rho) {
    const Matrix<S, N, N> commutator(x * h - h * x);
    const SymmetricMatrix<S, N> numerator((x * commutator - commutator * x).template as_symmetric<Lower>());
    const auto correction = cheeger_divide(numerator, rho);
    return SymmetricMatrix<S, N>(h + correction);
}

/// @brief differentiates the inverse log metric along a symmetric chart direction
template <typename S, int N>
SymmetricMatrix<S, N> cheeger_inverse_metric_derivative(
  const SymmetricMatrix<S, N>& x, const SymmetricMatrix<S, N>& direction, const SymmetricMatrix<S, N>& h, double rho) {
    const Matrix<S, N, N> base_commutator(x * h - h * x), direction_commutator(direction * h - h * direction);
    const SymmetricMatrix<S, N> numerator(
      (direction * base_commutator - base_commutator * direction + x * direction_commutator - direction_commutator * x)
        .template as_symmetric<Lower>());
    return cheeger_divide(numerator, rho);
}

/// @brief converts a log-chart Euclidean Hessian action with the C-LE Levi-Civita connection
template <typename S, int N>
SymmetricMatrix<S, N> cheeger_chart_hessian(
  const SymmetricMatrix<S, N>& x, const SymmetricMatrix<S, N>& gradient, const SymmetricMatrix<S, N>& hessian,
  const SymmetricMatrix<S, N>& direction, double rho) {
    const auto metric_direction = cheeger_metric<S, N>(x, direction, rho);
    const auto riemannian_gradient = cheeger_inverse_metric(x, gradient, rho);
    const Matrix<S, N, N> direction_commutator(x * metric_direction - metric_direction * x);
    const Matrix<S, N, N> gradient_commutator(x * gradient - gradient * x);
    // the adjoint metric derivative supplies the final Koszul term in logarithmic coordinates
    const SymmetricMatrix<S, N> numerator((gradient_commutator * metric_direction -
                                           metric_direction * gradient_commutator + direction_commutator * gradient -
                                           gradient * direction_commutator)
                                            .template as_symmetric<Lower>());
    const auto adjoint = cheeger_divide(numerator, rho);
    const SymmetricMatrix<S, N> corrected(hessian + S(.5) * adjoint);
    const auto raised = cheeger_inverse_metric(x, corrected, rho);
    const auto direct = cheeger_inverse_metric_derivative(x, direction, gradient, rho);
    const auto connection = cheeger_inverse_metric_derivative(x, riemannian_gradient, metric_direction, rho);
    return SymmetricMatrix<S, N>(raised + S(.5) * direct - S(.5) * connection);
}

/// @brief converts a native ambient gradient through the exponential chart and inverse C-LE metric
template <SPDLike P, typename S, int N>
SymmetricMatrix<S, N> cheeger_ambient_gradient(const P& point, const SymmetricMatrix<S, N>& gradient, double rho) {
    const SymmetricMatrix<S, N> x(matrix_log(point)), chart_gradient(spd_exp_log_frechet(point, gradient));
    const auto raised = cheeger_inverse_metric(x, chart_gradient, rho);
    return SymmetricMatrix<S, N>(spd_exp_log_frechet(point, raised));
}

/// @brief converts an ambient Hessian action including chart curvature and the C-LE connection
template <SPDLike P, typename S, int N>
SymmetricMatrix<S, N> cheeger_ambient_hessian(
  const P& point, const SymmetricMatrix<S, N>& gradient, const SymmetricMatrix<S, N>& hessian,
  const SymmetricMatrix<S, N>& direction, double rho) {
    const SymmetricMatrix<S, N> x(matrix_log(point)), chart_direction(matrix_log_frechet(point, direction));
    const SymmetricMatrix<S, N> chart_gradient(spd_exp_log_frechet(point, gradient));
    const auto chart_curvature = matrix_exp_second_frechet(x, chart_direction, gradient);
    const auto chart_action = spd_exp_log_frechet(point, hessian);
    const SymmetricMatrix<S, N> chart_hessian(chart_curvature + chart_action);
    const auto converted = cheeger_chart_hessian(x, chart_gradient, chart_hessian, chart_direction, rho);
    return SymmetricMatrix<S, N>(spd_exp_log_frechet(point, converted));
}
}   // namespace fdapde::manifold::internals
#endif
