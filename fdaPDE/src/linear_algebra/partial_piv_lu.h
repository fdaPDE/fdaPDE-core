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

#ifndef __FDAPDE_LINALG_PARTIAL_PIV_LU_H__
#define __FDAPDE_LINALG_PARTIAL_PIV_LU_H__

#include "header_check.h"

namespace fdapde {

// LU with partial (row) pivoting and threshold check
template <typename XprType_> class PartialPivLU {
    using XprType = std::decay_t<XprType_>;
    fdapde_static_assert(
      XprType::Rows == Dynamic || XprType::Cols == Dynamic || XprType::Rows == XprType::Cols,
      THIS_CLASS_IS_FOR_SQUARE_MATRICES_ONLY);
    static constexpr int Rows = XprType::Rows;
    static constexpr int Cols = XprType::Cols;
    using Scalar = typename XprType::Scalar;
   public:
    // constructors
    constexpr PartialPivLU() : L_(), U_(), P_(), info_(0), rank_(0) { }
    template <typename XprType>
    constexpr explicit PartialPivLU(const MatrixExpr<XprType>& m) : L_(), U_(), P_(), info_(0), rank_(0) {
        compute(m);
    }

    // build LU factorization of m via Doolittle LU with partial pivoting
    template <typename XprType> constexpr void compute(const MatrixExpr<XprType>& m) {
        const int n = m.rows();
        Matrix<Scalar, Rows, Cols> lu = m;
        Scalar pivot_threshold = std::numeric_limits<Scalar>::epsilon() * m.inf_norm();
        // initialization
        Vector<int, Rows> perm;
        if constexpr (Rows == Dynamic || Cols == Dynamic) { perm.resize(n); }
        for (int i = 0; i < n; ++i) perm[i] = i;
        info_ = 0;
        rank_ = n;
        // Doolittle with partial pivoting
        for (int k = 0; k < n; ++k) {
            // find pivot row
            int pivot_index = k;
            Scalar max_val = Scalar(0);
            for (int r = k; r < n; ++r) {
                Scalar av = fdapde::abs(lu(r, k));
                if (av > max_val) {
                    max_val = av;
                    pivot_index = r;
                }
            }
            // check threshold to detect singular or rank-deficient matrices
            if (max_val < pivot_threshold) {
                if (info_ == 0) info_ = k;   // LAPACK convention: first zero pivot index (0-based)
                rank_ = k;
                break;
            }
            if (pivot_index != k) {   // row swap
                for (int c = 0; c < m.cols(); ++c) { std::swap(lu(k, c), lu(pivot_index, c)); }
                std::swap(perm[k], perm[pivot_index]);
            }
            // eliminate column below pivot
            for (int r = k + 1; r < n; ++r) {
                Scalar alpha = lu(r, k) / lu(k, k);
                lu(r, k) = alpha;
                for (int c = k + 1; c < m.cols(); ++c) { lu(r, c) -= alpha * lu(k, c); }
            }
        }
        L_ = lu.template triangular_block<Lower>();
        L_.diagonal().cwise() = 1;   // L_ is unit-lower
        U_ = lu.template triangular_block<Upper>();
        P_ = PermutationMatrix<Rows, Cols>(perm);
        return;
    }
    // observers
    constexpr const PermutationMatrix<Rows, Cols>& P() const { return P_; }
    constexpr auto L() const { return L_; }
    constexpr auto U() const { return U_; }
    constexpr int info() const { return info_; }   // 0 = success, >0 = first zero pivot
    constexpr int rank() const { return rank_; }
    constexpr Scalar determinant() const {
        Scalar d = 1;
        for (int i = 0, n = U_.rows(); i < n; ++i) { d *= U_(i, i); }
        return P_.determinant() * d;
    }
    // solve Ax = b via PA = LU
    template <typename RhsXprType> constexpr auto solve(const MatrixExpr<RhsXprType>& b) const {
        fdapde_assert(b.rows() == L_.rows() && b.rows() == U_.rows() && b.cols() > 0);
        fdapde_assert(info_ == 0);

        constexpr int RhsRows = RhsXprType::Rows;
        constexpr int RhsCols = RhsXprType::Cols;
        Matrix<Scalar, RhsRows, RhsCols> y = P_ * b;
        auto z = L_.solve(y);   // forward  substitute
        auto x = U_.solve(z);   // backward substitute
        return x;
    }
   private:
    TriangularMatrix<Scalar, Rows, Cols, Lower> L_;
    TriangularMatrix<Scalar, Rows, Cols, Upper> U_;
    PermutationMatrix<Rows, Cols> P_;
    int info_;
    int rank_;
};

}   // namespace fdapde

#endif   // __FDAPDE_LINALG_PARTIAL_PIV_LU_H__
