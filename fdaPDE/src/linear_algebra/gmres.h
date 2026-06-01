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

#ifndef __FDAPDE_LINALG_GMRES_H__
#define __FDAPDE_LINALG_GMRES_H__

#include "header_check.h"

namespace fdapde {

// preconditioned m-step generalized minimum residual (GMRES) method with restart
template <typename XprType_, typename Preconditioner_> class GMRES {
    using XprType = std::decay_t<XprType_>;
    fdapde_static_assert(
      XprType::Rows == Dynamic || XprType::Cols == Dynamic || XprType::Rows == XprType::Cols,
      THIS_CLASS_IS_FOR_SQUARE_MATRICES_ONLY);
    static constexpr int Rows = XprType::Rows;
    static constexpr int Cols = XprType::Cols;
    using Scalar = typename XprType::Scalar;
   public:
    template <typename Preconditioner__>
        requires(std::is_constructible_v<Preconditioner_, Preconditioner__>)
    constexpr GMRES(Preconditioner__&& P, int max_iter, int restart, double tolerance) :
        max_iter_(max_iter), restart_(restart), tolerance_(tolerance), P_(std::forward<Preconditioner__>(P)) { }
    template <typename Preconditioner__>
        requires(std::is_constructible_v<Preconditioner_, Preconditioner__>)
    constexpr GMRES(Preconditioner__&& P) : GMRES(std::forward<Preconditioner__>(P), 500, 50, 1e-6) { }

    template <typename XprType__, typename Preconditioner__>
        requires(std::is_constructible_v<Preconditioner_, Preconditioner__> &&
                 std::is_same_v<XprType, std::decay_t<XprType__>>)
    constexpr explicit GMRES(
      const LinearOperatorExpr<XprType__>& m, Preconditioner__&& P, int max_iter, int restart, double tolerance) :
        max_iter_(max_iter), restart_(restart), tolerance_(tolerance), P_(std::forward<Preconditioner__>(P)) {
        if constexpr (Rows == Dynamic || Cols == Dynamic) {
            fdapde_assert(m.rows() == m.cols());
            fdapde_assert(max_iter > 0 && tolerance > 0);
        }
        compute(m);
    }
    template <typename XprType__, typename Preconditioner__>
    constexpr explicit GMRES(const LinearOperatorExpr<XprType__>& m, Preconditioner__&& P) :
        GMRES(m, std::forward<Preconditioner__>(P), 500, 50, 1e-6) { }

    template <typename XprType__>
        requires(std::is_same_v<XprType, std::decay_t<XprType__>>)
    constexpr void compute(const LinearOperatorExpr<XprType__>& m) {
        P_.compute(m.derived());
        m_ = std::addressof(m.derived());
        // pre-allocate memory
        H_.resize(restart_ + 1, restart_);
        V_.resize(m.rows(), restart_ + 1);
        c_.resize(restart_);
        s_.resize(restart_);
        g_.resize(restart_ + 1);
        if constexpr (Rows == Dynamic) { r_.resize(m_->rows()); }
        return;
    }
    template <typename RhsXprType, typename InitXprType>
    constexpr auto solve(const MatrixExpr<RhsXprType>& b, const MatrixExpr<InitXprType>& x0) {
        fdapde_assert(m_ != nullptr);
        fdapde_assert(b.cols() == 1 && b.rows() == m_->rows() && x0.cols() == 1 && x0.rows() == m_->rows());
        Vector<Scalar, Rows> x = x0;   // x = \argmin_{x \in x0 + V_} \| b - m * x \|_2

        int iter = 0;
        while (iter < max_iter_) {
            bool converged = false;
            // compute preconditioned residual r = P^-1 * (b - A * x)
            r_ = P_.solve(b - m_->apply(x));
            double beta = r_.norm();
            double r0_norm = beta;
            // initial guess already good
            if (r0_norm < tolerance_) { return x; }

            // start iterative procedure
            g_.set_zero();
            g_[0] = beta;
            int j = 0;
            for (; j < restart_ && iter < max_iter_; ++j, ++iter) {
                // update Krylov subspace by Arnoldi process
                // See "Golub, G. H., & Van Loan, C. F. (2013). Matrix computations. JHU press. Alg.10.5.1"
                V_.col(j) = r_ / beta;
                r_ = P_.solve(m_->apply(V_.col(j)));
                for (int i = 0; i <= j; ++i) {
                    H_(i, j) = V_.col(i).dot(r_);
                    r_ -= H_(i, j) * V_.col(i);
                }
                H_(j + 1, j) = r_.norm();
                beta = H_(j + 1, j);

                // QR update by Givens rotation
                // apply G_1, \ldots, G_{j-1} to j-th Hessenberg column H(1:j, j)
                for (int i = 0; i < j; ++i) {
                    double tmp = c_[i] * H_(i, j) + s_[i] * H_(i + 1, j);
                    H_(i + 1, j) = -s_[i] * H_(i, j) + c_[i] * H_(i + 1, j);
                    H_(i, j) = tmp;
                }

                // compute new rotation G_j to eliminate H(j + 1, j)
                // See "Golub, G. H., & Van Loan, C. F. (2013). Matrix computations. JHU press. Alg.5.1.3"
                double a_ = H_(j, j), b_ = H_(j + 1, j);
                if (almost_zero(b_)) {
                    c_[j] = 1.0;
                    s_[j] = 0.0;
                } else if (std::abs(b_) > std::abs(a_)) {
                    double tau = a_ / b_;
                    s_[j] = 1.0 / std::sqrt(1.0 + tau * tau);
                    c_[j] = s_[j] * tau;
                } else {
                    double tau = b_ / a_;
                    c_[j] = 1.0 / std::sqrt(1.0 + tau * tau);
                    s_[j] = c_[j] * tau;
                }

                // compute G_j^\top * (G_{j-1}^\top * G_1 * H)
                H_(j, j) = c_[j] * H_(j, j) + s_[j] * H_(j + 1, j);
                H_(j + 1, j) = 0.0;
                // compute G_j^\top * (G_{j-1}^\top * G_1 * (beta * e_1))
                double tmp = g_[j];
                g_[j] = c_[j] * tmp;        // s_[j] * g_[j + 1] = 0
                g_[j + 1] = -s_[j] * tmp;   // c_[j] * g_[j + 1] = 0

                // convergence check
                if (std::abs(g_[j + 1]) / r0_norm < tolerance_) {
                    j++;
                    converged = true;
                    break;
                }
            }
            // solve \min_{y \in R^restart} \| beta * e_1 - H * y \|_2 by QR-solve H * y = g
            auto y = H_.block(0, 0, j, j).template triangular_block<Upper>().solve(g_.top_rows(j));
            // update solution in Krylov space x = x + V_ * y
            x += V_.left_cols(j) * y;
            if (converged) { return x; }
        }
        return x;
    }
    template <typename RhsXprType> constexpr auto solve(const MatrixExpr<RhsXprType>& b) {
        Vector<Scalar, Rows> x0 = Vector<Scalar, Rows>::Zero(b.rows());
        return solve(b, x0);
    }
   private:
    int max_iter_;       // maximum number of iterations
    int restart_;        // maximum Krylov subspace dimension
    double tolerance_;   // accepted tolerance on l2-norm of relative residual

    const XprType_* m_ = nullptr;
    Preconditioner_ P_;
    Matrix<Scalar, Dynamic, Dynamic> H_;             // Hessenberg matrix
    Matrix<Scalar, Dynamic, Dynamic, ColMajor> V_;   // Krylov subspace
    Vector<Scalar, Dynamic> c_, s_;                  // Givens rotation G_i = (c[i], s[i]; -s[i], c[i])
    Vector<Scalar, Dynamic> g_;                      // G_i^\top * G_{i-1}^\top * ... * G_1^\top * (beta * e_1)
    Vector<Scalar, Rows> r_;                         // Arnoldi vector
};

}   // namespace fdapde

#endif   // __FDAPDE_LINALG_GMRES_H__
