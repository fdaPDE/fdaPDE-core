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

#ifndef __FDAPDE_LINALG_ROTATION_SCHUR_H__
#define __FDAPDE_LINALG_ROTATION_SCHUR_H__
#include <numbers>

#include "header_check.h"

namespace fdapde {
/// @brief distinguishes resolved branches, proximity to the cut locus and nonunique or unresolved logarithms
enum class RotationLogStatus {
    Regular,
    NearCut,
    Ambiguous,
    Unresolved
};
/// @brief reports the numerical branch status separately from distance and reconstruction accuracy
struct RotationLogDiagnostics {
    RotationLogStatus status = RotationLogStatus::Regular;
    double cut_gap = std::numbers::pi;
    double residual = 0;
    /// @brief reports whether the ordinary logarithm has a resolved unique branch
    bool unique() const { return status == RotationLogStatus::Regular || status == RotationLogStatus::NearCut; }
    /// @brief reports whether branch-sensitive derivatives are numerically regular
    bool regular() const { return status == RotationLogStatus::Regular; }
};
namespace internals {

/// @brief computes real orthogonal rotation planes by jointly resolving the symmetric and skew parts
/// @details each positive angle starts a two-column plane; its second column stores the negative angle
template <typename S, int N> class rotation_schur {
   public:
    using Scalar = S;
    static constexpr int Rows = N, Cols = N;
    /// @brief decomposes a verified rotation using native symmetric eigensolvers and checks the reconstruction
    template <typename Xpr> explicit rotation_schur(const Xpr& q) {
        const int n = q.rows();
        if constexpr (N == Dynamic) {
            vectors_.resize(n, n);
            angles_.resize(n);
        }
        vectors_.set_zero();
        angles_.set_zero();
        const S eps = std::numeric_limits<S>::epsilon();
        const S tolerance = S(64) * n * eps;
        SymmetricMatrix<S, N> h;
        if constexpr (N == Dynamic) h.resize(n, n);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j <= i; ++j) h(i, j) = S(0.5) * (q(i, j) + q(j, i));
        const EVD decomposition(h);
        const auto basis = decomposition.eigenvectors();
        std::vector<int> indices(n);
        for (int i = 0; i < n; ++i) indices[i] = i;
        std::sort(indices.begin(), indices.end(), [&](int a, int b) {
            return decomposition.eigenvalues()[a] < decomposition.eigenvalues()[b];
        });
        int columns = 0;
        for (int first = 0; first < n;) {
            int last = first + 1;
            const S first_cosine = decomposition.eigenvalues()[indices[first]];
            if (first_cosine < S(-0.5) - tolerance) {
                while (last < n && decomposition.eigenvalues()[indices[last]] < S(-0.5) + tolerance) ++last;
            } else if (first_cosine > S(0.5) - tolerance)
                last = n;
            else
                while (last < n && decomposition.eigenvalues()[indices[last]] - first_cosine <= tolerance) ++last;
            const int m = last - first;
            Matrix<S, Dynamic, Dynamic> group(n, m);
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < m; ++j) group(i, j) = basis(i, indices[first + j]);
            Matrix<S, Dynamic, Dynamic> skew(m, m), skew_group(n, m);
            skew.set_zero();
            for (int i = 0; i < n; ++i)
                for (int a = 0; a < m; ++a) {
                    S value = 0;
                    for (int j = 0; j < n; ++j) value += S(0.5) * (q(i, j) - q(j, i)) * group(j, a);
                    skew_group(i, a) = value;
                }
            for (int a = 0; a < m; ++a)
                for (int b = a + 1; b < m; ++b) {
                    S value = 0;
                    for (int i = 0; i < n; ++i) value += group(i, a) * skew_group(i, b);
                    skew(a, b) = value;
                    skew(b, a) = -value;
                }
            // the real lift of -i times the skew part resolves near-real and repeated cosine subspaces without squaring
            // angles
            SymmetricMatrix<S, Dynamic> lift(2 * m, 2 * m);
            for (int i = 0; i < m; ++i)
                for (int j = 0; j < m; ++j) lift(m + i, j) = skew(i, j);
            const EVD sine_evd(lift);
            const auto sine_basis = sine_evd.eigenvectors();
            std::vector<int> order(2 * m);
            for (int i = 0; i < 2 * m; ++i) order[i] = i;
            std::sort(order.begin(), order.end(), [&](int a, int b) {
                return sine_evd.eigenvalues()[a] > sine_evd.eigenvalues()[b];
            });
            const int start = columns;
            for (int index : order) {
                if (columns - start + 1 >= m || sine_evd.eigenvalues()[index] <= S(0)) continue;
                Vector<S, N> u, v;
                if constexpr (N == Dynamic) {
                    u.resize(n);
                    v.resize(n);
                }
                for (int i = 0; i < n; ++i) {
                    u[i] = 0;
                    for (int j = 0; j < m; ++j) u[i] += group(i, j) * sine_basis(j, index);
                }
                if (!normalize_(u, columns, std::sqrt(tolerance))) continue;
                Vector<S, Dynamic> local(m), image(m);
                for (int a = 0; a < m; ++a) {
                    local[a] = 0;
                    for (int i = 0; i < n; ++i) local[a] += group(i, a) * u[i];
                }
                for (int a = 0; a < m; ++a) {
                    image[a] = 0;
                    for (int b = 0; b < m; ++b) image[a] += skew(a, b) * local[b];
                }
                for (int i = 0; i < n; ++i) {
                    v[i] = 0;
                    for (int a = 0; a < m; ++a) v[i] += group(i, a) * image[a];
                }
                const S sine = v.norm();
                if (sine == S(0)) continue;
                v /= sine;
                if (!normalize_(v, columns, std::sqrt(tolerance))) continue;
                S projection = u.dot(v);
                v -= projection * u;
                const S vn = v.norm();
                if (vn <= S(0.5)) continue;
                v /= vn;
                S cosine = 0, signed_sine = 0;
                for (int i = 0; i < n; ++i)
                    for (int j = 0; j < n; ++j) {
                        cosine += u[i] * q(i, j) * u[j];
                        signed_sine += v[i] * q(i, j) * u[j];
                    }
                if (signed_sine < S(0)) {
                    v *= S(-1);
                    signed_sine = -signed_sine;
                }
                const S angle = std::atan2(signed_sine, cosine);
                if (angle == S(0)) continue;
                for (int i = 0; i < n; ++i) {
                    vectors_(i, columns) = u[i];
                    vectors_(i, columns + 1) = v[i];
                }
                angles_[columns] = angle;
                angles_[columns + 1] = -angle;
                columns += 2;
                update_status_(angle, signed_sine, tolerance, false);
            }
            // complete real eigenspaces at +1 or -1; negative singleton directions are paired explicitly
            std::vector<Vector<S, N>> remainder;
            for (int j = 0; j < m; ++j) {
                Vector<S, N> u(group.col(j));
                if (!normalize_(u, columns, S(0.5))) continue;
                for (const auto& previous : remainder) u -= previous.dot(u) * previous;
                const S un = u.norm();
                if (un <= S(0.5)) continue;
                u /= un;
                remainder.push_back(u);
            }
            const bool negative = decomposition.eigenvalues()[indices[first]] < S(0);
            fdapde_strong_assert(
              !negative || remainder.size() % 2 == 0, std::domain_error,
              "rotation Schur: unresolved negative eigenspace");
            for (std::size_t j = 0; j < remainder.size(); ++j) {
                for (int i = 0; i < n; ++i) vectors_(i, columns) = remainder[j][i];
                if (negative) angles_[columns] = (j % 2 == 0 ? S(1) : S(-1)) * std::numbers::pi_v<S>;
                ++columns;
            }
            if (negative && !remainder.empty()) {
                bool exact = true;
                for (int i = 0; i < m; ++i)
                    for (int j = 0; j < m; ++j) exact = exact && skew(i, j) == S(0);
                update_status_(std::numbers::pi_v<S>, S(0), tolerance, exact);
            }
            first = last;
        }
        fdapde_strong_assert(columns == n, std::domain_error, "rotation Schur: incomplete orthogonal basis");
        const auto reconstructed = evaluate(S(1));
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                diagnostics_.residual =
                  std::max(diagnostics_.residual, double(std::abs(reconstructed(i, j) - q(i, j))));
        fdapde_strong_assert(
          is_orthogonal(vectors_) && diagnostics_.residual <= S(8) * tolerance, std::domain_error,
          "rotation Schur: unresolved rotation planes");
    }
    /// @brief borrows the real orthogonal basis containing adjacent rotation-plane columns
    const auto& vectors() const& { return vectors_; }
    /// @brief rejects a basis reference escaping a temporary decomposition
    void vectors() const&& = delete;
    /// @brief borrows paired signed principal angles and zero singleton angles
    const auto& angles() const& { return angles_; }
    /// @brief rejects angle references escaping a temporary decomposition
    void angles() const&& = delete;
    /// @brief returns branch diagnostics without selecting an ambiguous logarithm
    RotationLogDiagnostics diagnostics() const { return diagnostics_; }
    /// @brief reconstructs the rotation obtained by scaling all selected plane angles
    auto evaluate(S t) const { return evaluate_planes(vectors_, angles_, t); }
    /// @brief evaluates a real orthogonal plane representation without recomputing its decomposition
    template <typename Basis, typename Angles>
    static auto evaluate_planes(const Basis& basis, const Angles& angles, S t) {
        fdapde_strong_assert(std::isfinite(t), std::invalid_argument, "rotation parameter must be finite");
        Matrix<S, N, N> result;
        const int n = basis.rows();
        if constexpr (N == Dynamic) result.resize(n, n);
        result.set_zero();
        for (int k = 0; k < n; ++k) {
            if (angles[k] < S(0)) continue;
            if (angles[k] == S(0)) {
                for (int i = 0; i < n; ++i)
                    for (int j = 0; j < n; ++j) result(i, j) += basis(i, k) * basis(j, k);
            } else {
                const S angle = t * angles[k];
                fdapde_strong_assert(std::isfinite(angle), std::domain_error, "rotation angle overflow");
                const S c = std::cos(angle), s = std::sin(angle);
                for (int i = 0; i < n; ++i)
                    for (int j = 0; j < n; ++j)
                        result(i, j) += c * (basis(i, k) * basis(j, k) + basis(i, k + 1) * basis(j, k + 1)) +
                                        s * (basis(i, k + 1) * basis(j, k) - basis(i, k) * basis(j, k + 1));
            }
        }
        return result;
    }
    /// @brief reconstructs the selected minimum skew logarithm from its rotation planes
    template <typename Basis, typename Angles> static auto logarithm(const Basis& basis, const Angles& angles) {
        SkewSymmetricMatrix<S, N, N> result;
        const int n = basis.rows();
        if constexpr (N == Dynamic) result.resize(n, n);
        for (int i = 0; i < n; ++i)
            for (int j = i + 1; j < n; ++j) {
                S value = 0;
                for (int k = 0; k < n; ++k)
                    if (angles[k] > S(0))
                        value += angles[k] * (basis(i, k + 1) * basis(j, k) - basis(i, k) * basis(j, k + 1));
                result(i, j) = value;
            }
        return result;
    }
   private:
    /// @brief removes previously selected columns twice and normalizes a nondependent candidate
    bool normalize_(Vector<S, N>& v, int count, S tolerance) const {
        for (int pass = 0; pass < 2; ++pass)
            for (int k = 0; k < count; ++k) {
                S dot = 0;
                for (int i = 0; i < v.rows(); ++i) dot += vectors_(i, k) * v[i];
                for (int i = 0; i < v.rows(); ++i) v[i] -= dot * vectors_(i, k);
            }
        const S norm = v.norm();
        if (norm <= tolerance) return false;
        v /= norm;
        return true;
    }
    /// @brief distinguishes exact negative real planes from numerically unresolved nearby branches
    void update_status_(S angle, S sine, S tolerance, bool exact) {
        diagnostics_.cut_gap = std::min(diagnostics_.cut_gap, double(std::max(S(0), std::numbers::pi_v<S> - angle)));
        if (angle <= std::numbers::pi_v<S> / S(2)) return;
        RotationLogStatus status = RotationLogStatus::Regular;
        if (sine <= tolerance)
            status = exact ? RotationLogStatus::Ambiguous : RotationLogStatus::Unresolved;
        else if (diagnostics_.cut_gap <= std::sqrt(tolerance))
            status = RotationLogStatus::NearCut;
        if (
          status == RotationLogStatus::Ambiguous || (diagnostics_.status != RotationLogStatus::Ambiguous &&
                                                     static_cast<int>(status) > static_cast<int>(diagnostics_.status)))
            diagnostics_.status = status;
    }
    Matrix<S, N, N> vectors_;
    Vector<S, N> angles_;
    RotationLogDiagnostics diagnostics_;
};

/// @brief computes distance from cosine eigenvalues when rotation planes cannot be resolved numerically
template <typename Q> double rotation_distance_cosines(const Q& q) {
    using S = typename Q::Scalar;
    SymmetricMatrix<S, Q::Rows> symmetric;
    if constexpr (Q::Rows == Dynamic) symmetric.resize(q.rows(), q.cols());
    for (int i = 0; i < q.rows(); ++i)
        for (int j = 0; j <= i; ++j) symmetric(i, j) = S(0.5) * (q(i, j) + q(j, i));
    const EVD evd(symmetric);
    double result = 0;
    for (int i = 0; i < q.rows(); ++i)
        result = std::hypot(result, std::acos(std::clamp(double(evd.eigenvalues()[i]), -1., 1.)));
    return result;
}

/// @brief exponentiates a finite skew matrix through scaling, a converged Taylor series and squaring
template <typename Xpr> auto skew_exponential(const SkewSymmetricMatrixExpr<Xpr>& source, double step = 1) {
    using S = std::remove_cv_t<typename Xpr::Scalar>;
    constexpr int N = Xpr::Rows;
    const S t = static_cast<S>(step);
    fdapde_strong_assert(std::isfinite(t), std::invalid_argument, "rotation step must be finite");
    Matrix<S, N, N> scaled(source);
    scaled *= t;
    S norm = 0;
    for (int i = 0; i < scaled.rows(); ++i)
        for (int j = 0; j < scaled.cols(); ++j) {
            fdapde_strong_assert(std::isfinite(scaled(i, j)), std::domain_error, "nonfinite rotation tangent");
            norm = std::hypot(norm, scaled(i, j));
        }
    fdapde_strong_assert(std::isfinite(norm), std::domain_error, "rotation tangent norm overflow");
    int squarings = 0;
    while (norm > S(0.5)) {
        norm *= S(0.5);
        scaled *= S(0.5);
        ++squarings;
    }
    Matrix<S, N, N> result(IdentityMatrix<S, N, N>(scaled.rows(), scaled.cols()));
    Matrix<S, N, N> term(result);
    for (int k = 1; k <= 32; ++k) {
        term = term * scaled / S(k);
        result += term;
        if (term.norm() <= std::numeric_limits<S>::epsilon() * result.norm()) break;
    }
    for (int i = 0; i < squarings; ++i) result = result * result;
    return result;
}
}   // namespace internals
}   // namespace fdapde
#endif
