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

#ifndef __FDAPDE_MANIFOLD_SO_DIFFERENTIAL_H__
#define __FDAPDE_MANIFOLD_SO_DIFFERENTIAL_H__
#include "header_check.h"

namespace fdapde::manifold::internals {
/// @brief caches analytic logarithm and Hessian operators in orthonormal body skew coordinates
/// @details with K = ad(log(Q)), the distance Hessian is f(-K*K), f(x) = sqrt(x)/2 cot(sqrt(x)/2)
template <typename S, int N> class SOLogDifferential {
   public:
    using Tangent = SkewSymmetricMatrix<S, N, N>;
    /// @brief prepares the native symmetric spectrum and divided differences once per relative logarithm
    explicit SOLogDifferential(const Tangent& logarithm) : n_(logarithm.rows()), k_(ad(logarithm)) {
        const int m = k_.rows();
        h_.resize(m, m);
        divided_.resize(m, m);
        basis_.resize(m, m);
        if (m == 0) return;
        const Matrix<S, Dynamic, Dynamic> squared(k_.transpose() * k_);
        const EVD evd(squared.template as_symmetric<Lower>());
        basis_ = evd.eigenvectors();
        Vector<S, Dynamic> values(m);
        for (int i = 0; i < m; ++i) values[i] = std::max(S(0), S(evd.eigenvalues()[i]));
        Matrix<S, Dynamic, Dynamic> scaled(basis_);
        for (int j = 0; j < m; ++j) {
            const S value = function_(values[j]);
            for (int i = 0; i < m; ++i) {
                scaled(i, j) *= value;
                divided_(i, j) = divided_difference_(values[i], values[j]);
            }
        }
        h_ = scaled * basis_.transpose();
    }
    /// @brief borrows the self-adjoint Hessian matrix for local nondegeneracy checks
    const auto& hessian() const { return h_; }
    /// @brief returns orthonormal coordinates for a finite body tangent of the retained order
    Vector<S, Dynamic> coordinates(const Tangent& tangent) const {
        fdapde_strong_assert(
          tangent.rows() == n_ && tangent.cols() == n_, std::invalid_argument,
          "SO differential tangent has incompatible shape");
        Vector<S, Dynamic> result(k_.rows());
        int p = 0;
        for (int i = 0; i < n_; ++i)
            for (int j = i + 1; j < n_; ++j) {
                const S value = tangent(i, j);
                fdapde_strong_assert(
                  std::isfinite(value), std::invalid_argument, "SO differential tangent must be finite");
                result[p++] = std::sqrt(S(2)) * value;
            }
        return result;
    }
    /// @brief reconstructs a skew tangent from orthonormal coordinates
    template <typename X> Tangent tangent(const X& x) const {
        Tangent result;
        if constexpr (N == Dynamic) result.resize(n_, n_);
        int p = 0;
        for (int i = 0; i < n_; ++i)
            for (int j = i + 1; j < n_; ++j) result(i, j) = x[p++] / std::sqrt(S(2));
        return result;
    }
    /// @brief applies the cached covariant base Hessian
    Tangent hessian_action(const Tangent& u) const {
        const auto x = coordinates(u);
        return tangent(Vector<S, Dynamic>(h_ * x));
    }
    /// @brief applies the target logarithm differential or its Frobenius metric adjoint
    Tangent target_action(const Tangent& v, bool adjoint = false) const {
        const auto x = coordinates(v);
        return tangent(Vector<S, Dynamic>(h_ * x + (adjoint ? S(-.5) : S(.5)) * (k_ * x)));
    }
    /// @brief differentiates the Hessian covariantly in both endpoints with a parallel input tangent
    Tangent mixed_action(const Tangent& u, const Tangent& v, const Tangent& w) const {
        const auto a = coordinates(u), b = coordinates(v), c = coordinates(w);
        const Vector<S, Dynamic> delta(S(-1) * (h_ * a) + S(.5) * (k_ * a) + h_ * b + S(.5) * (k_ * b));
        const auto dk = ad(tangent(delta));
        const Matrix<S, Dynamic, Dynamic> da(S(-1) * (dk * k_ + k_ * dk));
        const auto dh = frechet_(da);
        const auto connection = ad(u);
        return tangent(Vector<S, Dynamic>(dh * c + S(.5) * (connection * (h_ * c) - h_ * (connection * c))));
    }
    /// @brief contracts the mixed Hessian analytically to obtain its two endpoint metric adjoints
    std::pair<Tangent, Tangent> mixed_adjoint(const Tangent& w, const Tangent& z) const {
        const auto c = coordinates(w), d = coordinates(z);
        Matrix<S, Dynamic, Dynamic> dual(k_.rows(), k_.rows());
        for (int i = 0; i < k_.rows(); ++i)
            for (int j = 0; j < k_.rows(); ++j) dual(i, j) = S(.5) * (c[i] * d[j] + d[i] * c[j]);
        const auto b = frechet_(dual);
        const Matrix<S, Dynamic, Dynamic> k_dual(b * k_ + k_ * b);
        const auto g = coordinates(ad_adjoint_(k_dual));
        const auto hw = tangent(Vector<S, Dynamic>(h_ * c));
        const auto hz = tangent(Vector<S, Dynamic>(h_ * d));
        const auto hw_ad = ad(hw), w_ad = ad(w);
        const auto hz_coords = coordinates(hz);
        const Vector<S, Dynamic> connection(S(.5) * (hw_ad * d - w_ad * hz_coords));
        return {
          tangent(Vector<S, Dynamic>(S(-1) * (h_ * g) - S(.5) * (k_ * g) + connection)),
          tangent(Vector<S, Dynamic>(h_ * g - S(.5) * (k_ * g)))};
    }
   private:
    /// @brief forms the commutator operator using the sparse entries of the skew coordinate basis
    Matrix<S, Dynamic, Dynamic> ad(const Tangent& l) const {
        const std::int64_t count = std::int64_t(n_) * (n_ - 1) / 2;
        fdapde_strong_assert(
          count == 0 || count <= std::numeric_limits<int>::max() / count, std::length_error,
          "SO differential workspace exceeds the native matrix index range");
        const int m = static_cast<int>(count);
        Matrix<S, Dynamic, Dynamic> result(m, m);
        int p = 0;
        for (int i = 0; i < n_; ++i)
            for (int j = i + 1; j < n_; ++j, ++p) {
                int q = 0;
                for (int a = 0; a < n_; ++a)
                    for (int b = a + 1; b < n_; ++b, ++q)
                        result(p, q) = (j == b ? S(l(i, a)) : S(0)) - (j == a ? S(l(i, b)) : S(0)) -
                                       (i == a ? S(l(b, j)) : S(0)) + (i == b ? S(l(a, j)) : S(0));
            }
        return result;
    }
    /// @brief reverses the sparse commutator construction in the full-Frobenius tangent metric
    Tangent ad_adjoint_(const Matrix<S, Dynamic, Dynamic>& dual) const {
        Matrix<S, Dynamic, Dynamic> dense(n_, n_);
        dense.set_zero();
        int p = 0;
        for (int i = 0; i < n_; ++i)
            for (int j = i + 1; j < n_; ++j, ++p) {
                int q = 0;
                for (int a = 0; a < n_; ++a)
                    for (int b = a + 1; b < n_; ++b, ++q) {
                        const S value = dual(p, q);
                        if (j == b) dense(i, a) += value;
                        if (j == a) dense(i, b) -= value;
                        if (i == a) dense(b, j) -= value;
                        if (i == b) dense(a, j) += value;
                    }
            }
        Tangent result;
        if constexpr (N == Dynamic) result.resize(n_, n_);
        for (int i = 0; i < n_; ++i)
            for (int j = i + 1; j < n_; ++j) result(i, j) = S(.5) * (dense(i, j) - dense(j, i));
        return result;
    }
    /// @brief applies the self-adjoint spectral Frechet derivative without differentiating eigenvectors
    Matrix<S, Dynamic, Dynamic> frechet_(const Matrix<S, Dynamic, Dynamic>& direction) const {
        Matrix<S, Dynamic, Dynamic> local(basis_.transpose() * direction * basis_);
        for (int i = 0; i < local.rows(); ++i)
            for (int j = 0; j < local.cols(); ++j) local(i, j) *= divided_(i, j);
        return Matrix<S, Dynamic, Dynamic>(basis_ * local * basis_.transpose());
    }
    /// @brief evaluates the analytic Hessian kernel with its removable singularity at zero
    static S function_(S x) {
        if (x < S(.01)) return S(1) - x * (S(1) / 12 + x * (S(1) / 720 + x * (S(1) / 30240 + x / S(1209600))));
        const S t = std::sqrt(x) / 2;
        return t / std::tan(t);
    }
    /// @brief evaluates sin(x)/x continuously for repeated spectral frequencies
    static S sinc_(S x) {
        if (std::abs(x) < S(.001)) {
            const S x2 = x * x;
            return S(1) - x2 * (S(1) / 6 - x2 * (S(1) / 120 - x2 / S(5040)));
        }
        return std::sin(x) / x;
    }
    /// @brief evaluates repeated and clustered divided differences without subtracting nearby Hessian values
    static S divided_difference_(S a, S b) {
        if (std::max(a, b) < S(.01))
            return -S(1) / 12 - (a + b) / 720 - (a * a + a * b + b * b) / 30240 -
                   (a * a * a + a * a * b + a * b * b + b * b * b) / 1209600;
        if (std::min(a, b) < S(.25) * std::max(a, b)) return (function_(a) - function_(b)) / (a - b);
        const S t = std::sqrt(a) / 2, u = std::sqrt(b) / 2;
        return (sinc_(t + u) - sinc_(t - u)) / (8 * std::sin(t) * std::sin(u));
    }
    int n_;
    Matrix<S, Dynamic, Dynamic> k_, h_, basis_, divided_;
};
}   // namespace fdapde::manifold::internals
#endif
