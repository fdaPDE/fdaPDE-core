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
#include "matrix_algorithms.h"
#include "orthogonal_matrix.h"
#include "square_matrix_base.h"
#include "triangular_matrix.h"

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
        P_ = PermutationOp<N>(perm);
    }

    // access the permutation
    constexpr PermutationOp<N> P() const { return P_; }
    // access L and U
    constexpr LowerTriangularMatrix<Scalar, N> L() const {
        return LowerTriangularMatrix<Scalar, N>(lu_.template triangular_view<UnitLower>());
    }
    constexpr UpperTriangularMatrix<Scalar, N> U() const {
        return UpperTriangularMatrix<Scalar, N>(lu_.template triangular_view<Upper>());
    }

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
        Scalar max_val = Scalar(0);
        for (int r = col; r < N; ++r) {
            Scalar av = fdapde::abs(lu_(r, col));
            if (av > max_val) {
                max_val = av;
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
    PermutationOp<N> P_; // row‐permutation matrix
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
    static constexpr int StorageOrder = MatrixType::StorageOrder;
    static constexpr int NestAsRefBit = MatrixType::NestAsRefBit;
    using VectorType = Vector<Scalar, N, NestAsRefBit>;
    using MatrixN = Matrix<Scalar, N, N, StorageOrder, NestAsRefBit>;
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

        // Generic fallback: simple QR iteration
        MatrixN A(xpr.derived());
        QRDecomposition<decltype(A)> qr;
        MatrixN V(MatrixN::Identity());
        Scalar tol = N * std::numeric_limits<Scalar>::epsilon() * A.norm();
        int iter = 0;
        for (; iter < 100; ++iter) {
            qr.compute(A);
            A = MatrixN(qr.R() * qr.Q());
            V =  MatrixN(V * qr.Q());
            if (A.off_diagonal_norm() < tol) break;
        }
        for (int i = 0; i < N; ++i) eigenvalues_[i] = A(i,i);
        eigenvectors_ = V;
    }

    constexpr auto& eigenvalues() const  { return eigenvalues_; }
    constexpr auto& eigenvectors() const { return eigenvectors_; }

private:
    OrthogonalMatrix<Scalar, N> eigenvectors_{};
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

        MatrixView<Scalar, N, N, Xpr::StorageOrder, true> A(xpr.derived());
        R_.setZero();

        // MGS with basis completion
        Q_ = modified_gram_schmidt<decltype(A)>(A);

        // coefficients matrix
        for (int i = 0; i < N; ++i) {
            for (int j = i; j < N; ++j) {
                R_(i,j) = Q_.col(i).dot(A.col(j));
            }
        }
    }

    constexpr auto Q() const { return Q_; }
    constexpr auto R() const { return R_; }
    [[nodiscard]] bool info() const { return info_; }

private:
    OrthogonalMatrix<Scalar, N> Q_;
    UpperTriangularMatrix<Scalar, N> R_;
    bool info_ = Success;
};

} // namespace fdapde

#endif   // __FDAPDE_MATRIX_DECOMPOSITION_H__