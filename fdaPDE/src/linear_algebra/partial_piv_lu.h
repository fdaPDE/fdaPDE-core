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
template <typename Scalar_, int Size_> class PartialPivLU {
   public:
    static constexpr int Size = Size_;
    using Scalar = Scalar_;

    constexpr PartialPivLU() : lu_(), P_(), info_(0), rank_(0) { }
    template <int Rows, int Cols, typename XprType>
    constexpr explicit PartialPivLU(const MatrixExpr<Rows, Cols, XprType>& m) : lu_(), P_(), info_(0), rank_(0) {
        compute(m);
    }

    // build LU factorization of m
    template <int Rows, int Cols, typename XprType> constexpr void compute(const MatrixExpr<Rows, Cols, XprType>& m) {
        const int n = m.rows();
        lu_ = m;
        Scalar pivot_threshold = std::numeric_limits<Scalar>::epsilon() * m.inf_norm();
        // initialization
        Vector<int, Size> perm;
        if constexpr (Size == Dynamic) { perm.resize(n); }
        for (int i = 0; i < n; ++i) perm[i] = i;
        info_ = 0;
        rank_ = n;
        // Doolittle with partial pivoting
        for (int k = 0; k < n; ++k) {
            // find pivot row
            int pivot_index = k;
            Scalar max_val = Scalar(0);
            for (int r = k; r < n; ++r) {
                Scalar av = fdapde::abs(lu_(r, k));
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
                for (int c = 0; c < m.cols(); ++c) { std::swap(lu_(k, c), lu_(pivot_index, c)); }
                std::swap(perm[k], perm[pivot_index]);
            }
            // eliminate column below pivot
            for (int r = k + 1; r < n; ++r) {
                Scalar alpha = lu_(r, k) / lu_(k, k);
                lu_(r, k) = alpha;
                for (int c = k + 1; c < m.cols(); ++c) { lu_(r, c) -= alpha * lu_(k, c); }
            }
        }
        P_ = PermutationMatrix<Size>(perm);
	return;
    }

    // observers
    constexpr const PermutationMatrix<Size>& P() const { return P_; }
    constexpr auto L() const { return lu_.template triangular_block<UnitLower>(); }
    constexpr auto U() const { return lu_.template triangular_block<Upper>(); }
    constexpr int info() const { return info_; }   // 0 = success, >0 = first zero pivot
    constexpr int rank() const { return rank_; }
    constexpr Scalar determinant() const {
        Scalar d = 1;
        for (int i = 0, n = lu_.rows(); i < n; ++i) { d *= lu_(i, i); }
        return P_.determinant() * d;
    }

    // solve Ax = b via PA = LU
    template <int RhsRows, int RhsCols, typename RhsXprType>
    constexpr Matrix<Scalar, RhsRows, RhsCols> solve(const MatrixExpr<RhsRows, RhsCols, RhsXprType>& b) const {
        fdapde_assert(b.rows() == lu_.rows() && b.cols() > 0);

        Matrix<Scalar, RhsRows, RhsCols> y = P_ * b;
        auto z = L().solve(y);   // forward  substitute
        auto x = U().solve(z);   // backward substitute
        return x;
    }
   private:
    Matrix<Scalar, Size, Size, RowMajor> lu_;   // holds both L (unit lower) and U (upper)
    PermutationMatrix<Size> P_;
    int info_;
    int rank_;
};

}   // namespace fdapde

#endif   // __FDAPDE_LINALG_PARTIAL_PIV_LU_H__
