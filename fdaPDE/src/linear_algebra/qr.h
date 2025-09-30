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

#ifndef __FDAPDE_LINALG_QR_H__
#define __FDAPDE_LINALG_QR_H__

#include "header_check.h"

namespace fdapde {

// implementation of Householder QR as detailed in "Golub, G. H., & Van Loan, C. F. (2013). Matrix computations. JHU
// press. Sec.5.2.2"
template <typename Scalar_, int Rows_, int Cols_>
struct HouseholderQR {
public:
    using Scalar = Scalar_;
    static constexpr int Rows = Rows_;
    static constexpr int Cols = Cols_;

    constexpr HouseholderQR() : A_(), R_(), Q_(), taus_() { }

    template <int Rows_, int Cols_, typename XprType_>
    constexpr explicit HouseholderQR(const MatrixExpr<Rows_, Cols_, XprType_>& m) {
        compute(m);
    }

    template <int Rows_, int Cols_, typename XprType_>
    constexpr void compute(const MatrixExpr<Rows_, Cols_, XprType_>& m) {
        // initialization
        const int n_rows = m.rows();
        const int n_cols = m.cols();
        const int kmax = std::min(n_rows, n_cols);
        if constexpr (Rows == Dynamic || Cols == Dynamic) {
            A_.resize(n_rows, n_cols);
            taus_.resize(kmax);
        }
        A_ = m.derived();   // assign expression to dense storage

        // Householder QR iteration
        for (int k = 0; k < kmax; ++k) {
            // build Householder reflector
            // see "Golub, G. H., & Van Loan, C. F. (2013). Matrix computations. JHU press. Alg.5.1.1"
            int len = n_rows - k;
            Vector<Scalar, Rows> x(len);   // vector A[k:n-1, k]
            for (int i = 0; i < len; ++i) { x[i] = A_(k + i, k); }

            // compute reflector
            Scalar alpha = x[0];
            Scalar sigma = x.tail(len - 1).squared_norm();
            taus_[k] = Scalar(0);
            if (almost_zero(sigma)) {   // x = c * e_k, e_k: k-th canonical basis vector
                tau = Scalar(0);
            } else {
                Scalar norm_x = fdapde::sqrt(alpha * alpha + sigma);
                Scalar beta = (alpha <= Scalar(0)) ? (norm_x) : (-sigma / (alpha + norm_x));
                tau = (beta - alpha) / beta;
                x[0] = Scalar(1);
                x.tail(len - 1) /= (alpha - beta);

                // apply H = I - tau * v v^T to trailing submatrix A[k:m-1, k:n-1]
                for (int j = k; j < n_cols; ++j) {
                    Scalar dot = Scalar(0);
                    for (int i = 0; i < len; ++i) dot += x[i] * A_(k + i, j);
                    dot *= tau;
                    for (int i = 0; i < len; ++i) A_(k + i, j) -= x[i] * dot;
                }

		// store beta in diagonal
                A_(k, k) = beta;
                for (int i = 1; i < len; ++i) A_(k + i, k) = x[i];   // store R_ in the upper triangular part
                taus_[k] = tau;
            }
        }
    }

    // Build Q explicitly, apply householder vectors in reverse order
    constexpr const auto& Q() {
        if (Q_cached_) return Q_;
        const int n_rows = A_.rows();
        const int n_cols = A_.cols();
        const int kmax = std::min(n_rows, n_cols);

        Q_.setZero();
        for (int i = 0; i < n_rows; ++i) Q_(i, i) = Scalar(1);

        // apply reflectors backwards
        for (int k = kmax - 1; k >= 0; --k) {
            Scalar tau = taus_[k];
            if (tau == Scalar(0)) continue;

            int len = n_rows - k;
            Vector<Scalar, Rows> v(len);
            v[0] = 1;
            for (int i = 1; i < len; ++i) v[i] = A_(k + i, k);

            for (int j = 0; j < n_rows; ++j) {
                Scalar dot = 0;
                for (int i = 0; i < len; ++i) dot += v[i] * Q_(k + i, j);
                dot *= tau;
                for (int i = 0; i < len; ++i) Q_(k + i, j) -= v[i] * dot;
            }
        }

        Q_cached_ = true;
        return Q_;
    }

    constexpr auto R() const { return A_.template triangular_block<Upper>(); }
   private:
    Matrix<Scalar, Rows, Cols> A_;
    Vector<Scalar, Rows> taus_;      // Householder scalars
};
  
}   // namespace fdapde

#endif   // _FDAPDE_LINALG_QR_H__
