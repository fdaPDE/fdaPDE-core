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

#ifndef __FDAPDE_MATRIX_DECOMPOSITION_H__
#define __FDAPDE_MATRIX_DECOMPOSITION_H__

#include "header_check.h"
#include "square_matrix_base.h"

namespace fdapde {

// forward declarations
template <typename MatrixType> class PartialPivLU;
template <typename MatrixType, int Solver> class EigenDecomposition;
template <typename MatrixType> class QRDecomposition;

[[maybe_unused]] constexpr bool Success = true;

// LU with partial (row) pivoting and threshold check
template <typename MatrixType>
class PartialPivLU {
    fdapde_static_assert(MatrixType::Rows == MatrixType::Cols, LU_DECOMPOSITION_IS_ONLY_FOR_SQUARE_INVERTIBLE_MATRICES);

public:
    static constexpr int N = MatrixType::Rows;
    using Scalar = typename MatrixType::Scalar;
    static constexpr Scalar pivot_threshold_ = std::numeric_limits<Scalar>::epsilon();

    constexpr PartialPivLU() : lu_(), P_() { }

    template <typename Xpr>
    constexpr explicit PartialPivLU(const SquareMatrixBase<N, Xpr>& m) {
        compute(m);
    }

    // factorize: overwrite `lu_` in place, build P_
    template <typename Xpr>
    constexpr void compute(const SquareMatrixBase<N, Xpr>& m) {
        // copy input
        lu_ = m;

        // track row swaps in a simple array first
        std::array<int,N> perm;
        for (int i = 0; i < N; ++i) perm[i] = i;

        // Doolittle algorithm with partial pivoting
        for (int i = 0; i < N-1; ++i) {
            // 1) find pivot row among i..N-1 (O(N); could be optimized with a blocked or parallel scan for large N)
            int pivot_index = find_pivot_row(i, perm);

            // 1b) check threshold to detect singular or rank‐deficient matrices
            Scalar max_val = fdapde::abs(lu_(pivot_index,i));
            if (max_val < pivot_threshold_) {
                info_ = !Success;
                break;
            }

            // 2) swap rows i <-> pivot_index if needed
            if (pivot_index != i) {
                swap_rows(i, pivot_index, perm);
            }

            // 3) eliminate below pivot
            eliminate_column(i);

            // check invertibility
            Scalar diag = fdapde::abs(lu_(i+1,i+1));
            if (diag < pivot_threshold_ ) info_ = !Success;
        }

        // build the final PermutationMatrix
        P_ = PermutationMatrix<N>(perm);
    }

    // access the permutation
    constexpr PermutationMatrix<N> P() const { return P_; }

    // solve Ax = b via PA = LU
    template <typename Rhs>
    constexpr Vector<Scalar,N> solve(const Rhs& b) const {
        fdapde_static_assert(std::is_same_v<Scalar FDAPDE_COMMA typename Rhs::Scalar>, INVALID_SCALAR_TYPE_FOR_RHS_OPERAND);
        fdapde_constexpr_assert(b.rows() == N && b.cols() == 1);

        // apply P to RHS, then forward/backward substitute
        auto y = P_ * b;
        auto z = forward_sub(lu_.template triangular_view<UnitLower>(), y);
        return backward_sub(lu_.template triangular_view<Upper>(), z);
    }

    [[nodiscard]] bool info() const { return info_; }

private:
    // helper: find pivot row with maximum absolute value in column 'col'
    constexpr int find_pivot_row(int col, const std::array<int,N>& /*perm*/) const {
        int pivot = col;
        Scalar maxval = Scalar(0);
        for (int r = col; r < N; ++r) {
            Scalar av = fdapde::abs(lu_(r, col));
            if (av > maxval) {
                maxval = av;
                pivot = r;
            }
        }
        return pivot;
    }

    // helper: swap two rows in 'lu_' and record in 'perm'
    constexpr void swap_rows(int i, int j, std::array<int,N>& perm) {
        std::swap(perm[i], perm[j]);
        for (int c = 0; c < N; ++c)
            std::swap(lu_(i, c), lu_(j, c));
    }

    // helper: eliminate entries below pivot in column 'col'
    constexpr void eliminate_column(int col) {
        for (int r = col+1; r < N; ++r) {
            Scalar alpha = lu_(r, col) / lu_(col, col);
            lu_(r, col) = alpha;  // store L
            for (int c = col+1; c < N; ++c) {
                lu_(r, c) -= alpha * lu_(col, c);
            }
        }
    }

    Matrix<Scalar, N, N, RowMajor> lu_;  // will hold both L (unit lower) and U (upper)
    PermutationMatrix<N> P_; // row‐permutation matrix
    bool info_ = Success;
};

[[maybe_unused]] constexpr int Best = 0;
[[maybe_unused]] constexpr int QR = 1;

// eigen decomposition
template <typename MatrixType, int Solver_ = Best>
class EigenDecomposition {
    fdapde_static_assert(MatrixType::Rows == MatrixType::Cols, "EVD is only defined for square matrices");
    fdapde_static_assert(MatrixType::Rows >= 2, "EVD: size must be >= 2 for analytic specializations");

public:
    static constexpr int N = MatrixType::Rows;
    using Scalar  = typename MatrixType::Scalar;
    using VectorType = Vector<Scalar, N>;
    using MatrixN = Matrix<Scalar, N, N>;
    static constexpr int Solver = Solver_;

    constexpr EigenDecomposition() = default;

    template <typename Xpr>
    constexpr explicit EigenDecomposition(const SquareMatrixBase<N, Xpr>& m) {
        static_assert(std::is_same_v<Scalar, typename Xpr::Scalar>, "EVD: scalar types must match");
        fdapde_static_assert(internals::is_symmetric_v<Xpr>, "EVD compute requires symmetric input expression");
        compute(m);
    }

    template <typename Xpr>
    constexpr void compute(const SquareMatrixBase<N, Xpr>& xpr) {
        static_assert(std::is_same_v<Scalar, typename Xpr::Scalar>, "EVD: scalar types must match");
        fdapde_static_assert(internals::is_symmetric_v<Xpr>, "EVD compute requires symmetric input expression");
        // 2×2 analytic closed‐form
        if constexpr (N == 2 && Solver == Best) {

            const Scalar a = xpr.derived()(0,0), b = xpr.derived()(0,1), c = xpr.derived()(1,1);
            const Scalar tr = a + c;
            const Scalar det = a*c - b*b;
            const Scalar disc = fdapde::sqrt(tr*tr/4 - det);

            eigenvalues_[0] = tr/2 + disc;
            eigenvalues_[1] = tr/2 - disc;

            for (int k = 0; k < 2; ++k) {
                Scalar lambda = eigenvalues_[k];
                VectorType v{};
                if (b != Scalar(0)) {
                    v[0] = lambda - c;
                    v[1] = b;
                } else {
                    v[0] = Scalar(1);
                    v[1] = Scalar(0);
                }
                Scalar nrm = fdapde::sqrt(v[0]*v[0] + v[1]*v[1]);
                eigenvectors_(0,k) = v[0] / nrm;
                eigenvectors_(1,k) = v[1] / nrm;
            }
            return;
        }

        // 3×3 analytic closed‐form (symmetric)
        else if constexpr (N == 3 && Solver == Best) {

            // matrix entries
            const Scalar m00 = xpr.derived()(0,0), m01 = xpr.derived()(0,1), m02 = xpr.derived()(0,2);
            const Scalar m11 = xpr.derived()(1,1), m12 = xpr.derived()(1,2), m22 = xpr.derived()(2,2);
            // compute trace and centered moments
            const Scalar trace = (m00 + m11 + m22) / Scalar(3);
            const Scalar a00 = m00 - trace;
            const Scalar a11 = m11 - trace;
            const Scalar a22 = m22 - trace;
            const Scalar p2 = a00*a00 + a11*a11 + a22*a22 + Scalar(2)*(m01*m01 + m02*m02 + m12*m12);
            const Scalar p = fdapde::sqrt(p2 / Scalar(6));
            // build normalized matrix B = (A - trace*I)/p
            MatrixN B;
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    const Scalar aij = xpr.derived()(i,j) - (i==j ? trace : Scalar(0));
                    B(i,j) = aij / p;
                }
            }
            // compute det(B)/2
            const Scalar detB = B(0,0)*(B(1,1)*B(2,2) - B(1,2)*B(2,1))
                              - B(0,1)*(B(1,0)*B(2,2) - B(1,2)*B(2,0))
                              + B(0,2)*(B(1,0)*B(2,1) - B(1,1)*B(2,0));
            Scalar r = detB / Scalar(2);
            r = r < -Scalar(1) ? -Scalar(1) : (r > Scalar(1) ? Scalar(1) : r);
            constexpr Scalar pi = Scalar(3.14159265358979323846);
            const Scalar phi = std::acos(r) / Scalar(3);
            // eigenvalues
            eigenvalues_[0] = trace + Scalar(2)*p * std::cos(phi);
            eigenvalues_[2] = trace + Scalar(2)*p * std::cos(phi + Scalar(2)*pi/Scalar(3));
            eigenvalues_[1] = Scalar(3)*trace - eigenvalues_[0] - eigenvalues_[2];
            // eigenvectors via cross‐product
            for (int k = 0; k < 3; ++k) {
                const Scalar lambda = eigenvalues_[k];
                VectorType row0, row1, row2;
                row0[0] = m00 - lambda; row0[1] = m01;          row0[2] = m02;
                row1[0] = m01;          row1[1] = m11 - lambda; row1[2] = m12;
                row2[0] = m02;          row2[1] = m12;          row2[2] = m22 - lambda;
                auto cross = [&](const VectorType& u,const VectorType& v){
                    VectorType c;
                    c[0] = u[1]*v[2] - u[2]*v[1];
                    c[1] = u[2]*v[0] - u[0]*v[2];
                    c[2] = u[0]*v[1] - u[1]*v[0];
                    return c;
                };
                VectorType v = cross(row0, row1);
                Scalar n2 = v[0]*v[0] + v[1]*v[1] + v[2]*v[2];
                if (n2 < Scalar(1e-12)) { v = cross(row0, row2); n2 = v[0]*v[0] + v[1]*v[1] + v[2]*v[2]; }
                if (n2 < Scalar(1e-12)) { v = cross(row1, row2); n2 = v[0]*v[0] + v[1]*v[1] + v[2]*v[2]; }
                if (n2 < Scalar(1e-12)) { v[0]=Scalar(1); v[1]=v[2]=Scalar(0); n2 = Scalar(1); }
                const Scalar n = fdapde::sqrt(n2);
                eigenvectors_(0,k) = v[0] / n;
                eigenvectors_(1,k) = v[1] / n;
                eigenvectors_(2,k) = v[2] / n;
            }
            return;
        }

        // Generic fallback: simple QR iteration
        // ....
    }

    constexpr VectorType eigenvalues() const  { return eigenvalues_; }
    constexpr MatrixN  eigenvectors() const { return eigenvectors_; }

private:
    MatrixN eigenvectors_{};
    VectorType eigenvalues_{};
};

// QR decomposition (using classical GS)
template <typename MatrixType>
class QRDecomposition {
    fdapde_static_assert(MatrixType::Rows == MatrixType::Cols, "QR is only defined for square matrices");

public:
    static constexpr int N = MatrixType::Rows;
    using Scalar  = typename MatrixType::Scalar;
    using MatrixN = Matrix<Scalar, N, N>;

    constexpr QRDecomposition() = default;

    template <typename Xpr>
    constexpr explicit QRDecomposition(const SquareMatrixBase<N, Xpr>& m) {
        static_assert(std::is_same_v<Scalar, typename Xpr::Scalar>, "QR: scalar types must match");
        compute(m);
    }

    template <typename Xpr>
    constexpr void compute(const SquareMatrixBase<N, Xpr>& xpr) {
        MatrixN A(xpr.derived());
        Q_.setZero();
        R_.setZero();

        // Classical Gram–Schmidt
        for (int j = 0; j < N; ++j) {
            // v = column j of A
            Vector<Scalar, N> v{};
            for (int i = 0; i < N; ++i) {
                v[i] = A(i, j);
            }

            // Orthogonalization
            for (int k = 0; k < j; ++k) {
                Scalar dot = Scalar(0);
                for (int i = 0; i < N; ++i) {
                    dot += Q_(i, k) * A(i, j);
                }
                R_(k, j) = dot;
                for (int i = 0; i < N; ++i) {
                    v[i] -= dot * Q_(i, k);
                }
            }

            // Normalize
            Scalar norm2 = Scalar(0);
            for (int i = 0; i < N; ++i) {
                norm2 += v[i] * v[i];
            }
            Scalar norm = fdapde::sqrt(norm2);
            R_(j, j) = norm;
            // Avoid divide-by-zero
            if (norm == Scalar(0)) {
                info_ = !Success;
            }
            for (int i = 0; i < N; ++i) {
                Q_(i, j) = v[i] / norm;
            }
        }
    }

    constexpr MatrixN Q() const { return Q_; }
    constexpr auto R() const { return R_.template triangular_view<Upper>(); }
    [[nodiscard]] bool info() const { return info_; }

private:
    MatrixN Q_{};
    MatrixN R_{};
    bool info_ = Success;
};

} // namespace fdapde

#endif   // __FDAPDE_MATRIX_DECOMPOSITION_H__