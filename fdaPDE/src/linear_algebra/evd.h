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

#ifndef __FDAPDE_EVD_H__
#define __FDAPDE_EVD_H__

namespace fdapde {

// computes the eigen-decomposition of a matrix
template <typename Scalar_, int Size_> class EVD {
    static constexpr int Size = Size_;
    using Scalar = Scalar_;

    // tridiagonalize a symmetric matrix via householder reflectors
    // see "Golub, G. H., & Van Loan, C. F. (2013). Matrix computations. JHU press. Sec.8.3.1"
    template <typename XprType>
    std::tuple<Matrix<Scalar, Size, Size>, OrthogonalMatrix<Scalar, Size>>
    householder_tridiagonalize_(const XprType& m) const {
        const int n = m.rows();
        Matrix<Scalar, Size, Size> T = m;
        Matrix<Scalar, Size, Size> Q = Matrix<Scalar, Size, Size>::Identity(n, n);

        for (int k = 0; k < n - 2; ++k) {
            const int m = n - k - 1;
            Vector<Scalar, Dynamic> u = T.block(k + 1, k, m, 1);
            Scalar u_norm = u.norm();
            if (almost_equal(u_norm, 0.0, 1e-14)) continue;   // column is already zero below diagonal

            // compute householder reflector
            Scalar alpha = -std::copysign(u_norm, u[0]);
            u[0] -= alpha;
            Scalar beta = Scalar(2) / u.squared_norm();
            // update symmetric block A_22 = T[k+1:n-1, k+1:n-1]
            auto A22 = T.block(k + 1, k + 1, m, m);
            Vector<Scalar, Dynamic> w = beta * (A22 * u);
            Scalar tau = 0.5 * beta * u.dot(w);
            w -= tau * u;
            A22 -= u * w.transpose() + w * u.transpose();
            // update top-right block A_01 = T[0:k, k+1:n-1]
            auto A01 = T.block(0, k + 1, k + 1, m);
            Vector<Scalar, Dynamic> z = beta * (A01 * u);
            A01 -= z * u.transpose();
            // symmetrize (block A_10)
            T.block(k + 1, 0, m, k + 1) = A01.transpose();
            // set (k+1,k) bidiagonal element, zero column below it
            T(k + 1, k) = alpha;
            T(k, k + 1) = alpha;
            for (int i = k + 2; i < n; ++i) { T(i, k) = T(k, i) = Scalar(0); }

            // Q update
            Vector<Scalar, Dynamic> zQ = Q.block(0, k + 1, n, m) * u;
            Q.block(0, k + 1, n, m) -= zQ * (beta * u.transpose());
        }
        return std::make_pair(T, internals::orthogonal_wrapper<Size, Size, Matrix<Scalar, Size, Size>>(Q));
    }

    int max_iter_ = 30;   // taken from LAPACK, actual number of iteration is scaled by matrix size
   public:
    constexpr EVD() = default;

    template <int Rows, int Cols, typename XprType>
    constexpr explicit EVD(const SymmetricMatrixExpr<Rows, Cols, XprType>& m) {
        compute(m);
    }

    // computes the EVD of a symmetric matrix using the implicit QR-iteration with Wilkinson shift
    // see "Golub, G. H., & Van Loan, C. F. (2013). Matrix computations. JHU press. Ch.8.3"
    template <int Rows, int Cols, typename XprType>
    constexpr void compute(const SymmetricMatrixExpr<Rows, Cols, XprType>& mtx) {
        fdapde_static_assert(Rows == Cols && Rows == Size, THIS_METHOD_IS_FOR_SQUARE_MATRICES_ONLY);
        auto [T, Q_] = householder_tridiagonalize_(mtx.derived());
        const int n = T.rows();
        const int max_iter = max_iter_ * n;
        Matrix<Scalar, Rows, Cols> Q = Matrix<Scalar, Rows, Cols>::Identity(n, n);
        // extract diagonal and subdiagonal
        Vector<Scalar, Size> dd;
        Vector<Scalar, Size == Dynamic ? Dynamic : (Size - 1)> sd;
        if constexpr (Size == Dynamic) {
            dd.resize(n);
            sd.resize(n - 1);
        }
        for (int i = 0; i < n; ++i) { dd[i] = T(i, i); }
        for (int i = 0; i < n - 1; ++i) { sd[i] = T(i + 1, i); }
        int i = 0, j = n - 1;   // active diagonal range
        int iter = 0;           // iteration counter

        while (j > 1 && iter < max_iter) {
            // deflate small subdiagonals
            for (int k = i; k < j; ++k) {
                Scalar s = sd[k];
                if (abs(s) < std::numeric_limits<Scalar>::min()) {   // underflow, force to zero
                    sd[k] = Scalar(0);
                    continue;
                }
                // check relative size against neighboring diagonals
                Scalar scaled_s = s / std::numeric_limits<Scalar>::epsilon();
                if (scaled_s * scaled_s <= (abs(dd[k]) + abs(dd[k + 1]))) { sd[k] = Scalar(0); }
            }
            // adapt active diagonal range
            while (j > 1 && almost_zero(sd[j - 1])) { j--; }
            if (j == 0) break;
            i = j - 1;
            while (i > 0 && !almost_zero(sd[i - 1])) { i--; }
            // stable computation of Wilkinson shift from 2x2 block
            // [T(m-2, m-2) T(m-2, m-1)
            //  T(m-1, m-2) T(m-2, m-2)]
            Scalar t = (dd[j - 1] - dd[j]) * 0.5;
            Scalar e = sd[j - 1];
            Scalar mu = dd[j];
            if (almost_zero(t)) {
                mu -= abs(e);
            } else {
                Scalar e2 = e * e;
                Scalar h = std::hypot(t, e);
                mu -= e2 / (t + (t > 0 ? h : -h));
            }
            // Francis implicit tridiagonal-QR step
            Scalar x = dd[i] - mu;
            Scalar z = sd[i];

            for (int k = i; k < j && !almost_zero(z); ++k) {
                Scalar r = std::hypot(x, z);
                Scalar c = (r == Scalar(0)) ? Scalar(1) : x / r;
                Scalar s = (r == Scalar(0)) ? Scalar(0) : z / r;
                // apply Givens rotation G = [c -s 0; s c 0; 0 0 1] to 3 x 3 block
                // [T(k, k)     T(k, k + 1)     0
                //  T(k + 1, k) T(k + 1, k + 1) T(k + 1, k + 2)
                //  0           T(k + 2, k + 1) T(k + 2, k + 2)]
                Scalar m0 = dd[k];
                Scalar m1 = sd[k];
                Scalar m3 = dd[k + 1];
                // compute G^\top T_{k:k+1, k:k+1} G
                dd[k] = c * c * m0 + 2 * c * s * m1 + s * s * m3;
                sd[k] = c * s * (m3 - m0) + (c * c - s * s) * m1;
                dd[k + 1] = s * s * m0 - 2 * c * s * m1 + c * c * m3;
                // update previous subdiagonal
                if (k > i) { sd[k - 1] = c * sd[k - 1] + s * z; }
                // handle 3 x 3 block
                x = sd[k];
                if (k < j - 1) {
                    Scalar m4 = sd[k + 1];
                    sd[k + 1] = c * m4;
                    z = s * m4;
                }

                // update matrix Q <- Q * G
                for (int h = 0; h < n; ++h) {
                    Scalar q1 = Q(h, k), q2 = Q(h, k + 1);
                    Q(h, k) = c * q1 + s * q2;
                    Q(h, k + 1) = -s * q1 + c * q2;
                }
            }
            ++iter;
        }
        // store decomposition
        eigenvectors_ = Q_ * Q;
        eigenvalues_ = dd;
    }
    // observers
    constexpr const Vector<Scalar, Size>& eigenvalues() const { return eigenvalues_; }
    constexpr auto eigenvectors() const {
      return internals::orthogonal_wrapper<Size, Size, Matrix<Scalar, Size, Size>>(eigenvectors_);
    }
   private:
    Matrix<Scalar, Size, Size> eigenvectors_;
    Vector<Scalar, Size> eigenvalues_;
};

}   // namespace fdapde

#endif   // __FDAPDE_EVD_H__
