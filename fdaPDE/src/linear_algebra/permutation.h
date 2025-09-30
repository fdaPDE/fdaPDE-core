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

#ifndef __FDAPDE_LINALG_PERMUTATION_H__
#define __FDAPDE_LINALG_PERMUTATION_H__

#include "header_check.h"

namespace fdapde {

// implementation of the symmetric group S_n

template <int Rows_, int Cols_, typename XprType_>
struct PermutationMatrixExpr :
    public OrthogonalMatrixExpr<Rows_, Cols_, PermutationMatrixExpr<Rows_, Cols_, XprType_>> {
    fdapde_static_assert(
      Rows_ == Dynamic || Cols_ == Dynamic || Rows_ == Cols_, PERMUTATION_TYPE_SYSTEM_IS_FOR_SQUARED_MATRICES_ONLY);
    using Base = OrthogonalMatrixExpr<Rows_, Cols_, PermutationMatrixExpr<Rows_, Cols_, XprType_>>;
    using Base::derived;

    constexpr auto inverse() const { return PermutationInverseOp<XprType_>(derived()); }
    constexpr auto determinant() const {
        using Scalar = typename XprType_::Scalar;
        static constexpr int Rows = XprType_::Rows;

        Vector<Scalar, Rows> permutation = derived().permutation();
        int n = permutation.size();
        Vector<char, Rows> visited(n, 0);
        int cycles = 0;
        for (int i = 0; i < n; ++i) {
            if (!visited[i]) {
                ++cycles;
                int j = i;
                while (!visited[j]) {
                    visited[j] = true;
                    j = permutation[j];
                }
            }
        }
        // sign = (-1)^(n - cycles)
        return ((n - cycles) % 2 == 0) ? Scalar(1) : Scalar(-1);
    }
    // reductions
    constexpr auto squared_norm() const { return derived().rows(); }
    constexpr auto norm() const { return sqrt(squared_norm()); }
};

// expression of the inverse of a permutation
template <typename XprType>
struct PermutationInverseOp :
    public PermutationMatrixExpr<XprType::Rows, XprType::Cols, PermutationInverseOp<XprType>> {
    using Base = PermutationMatrixExpr<XprType::Rows, XprType::Cols, PermutationInverseOp<XprType>>;
    using XprTypeNested = internals::ref_select_t<XprType>;
    using Scalar = typename XprType::Scalar;
    static constexpr int Rows = XprType::Rows;
    static constexpr int Cols = XprType::Cols;
    static constexpr int NestAsRef = 0;
    static constexpr int ReadOnly = 1;

    template <typename XprType_>
        requires(std::is_constructible_v<XprTypeNested, XprType_>)
    constexpr explicit PermutationInverseOp(XprType_&& xpr) : xpr_(std::forward<XprType_>(xpr)) { }

    constexpr Scalar operator()(int i, int j) const { return xpr_.image(j) == i ? 1 : 0; }   // matrix transposition
    constexpr int image(int i) const {   // image(i) returns pi^{-1}(i)
        fdapde_constexpr_assert(i >= 0 && i < xpr_.rows());
        for (int j = 0, n = xpr_.rows(); j < n; ++j) {
            if (xpr_.image(j) == i) { return j; }
        }
    }
    constexpr Vector<Scalar, Rows> permutation() const {   // materialize the permutation vector
        Vector<Scalar, Rows> p;
        if constexpr (Rows == Dynamic) { p.resize(xpr_.rows()); }
        for (int i = 0, n = xpr_.rows(); i < n; ++i) { p[i] = image(i); }
        return p;
    }
    // observers
    constexpr int rows() const { return xpr_.rows(); }
    constexpr int cols() const { return xpr_.cols(); }
   private:
    XprTypeNested xpr_;
};

// action products
namespace internals {

template <typename LhsXprType, typename RhsXprType, int ProductMode> struct permutation_product_executor {
    static constexpr auto run(int i, int j, const LhsXprType& lhs, const RhsXprType& rhs) {
        if constexpr (ProductMode == LhsMode) { return rhs(lhs.image(i), j); }   // RowPermutation
        if constexpr (ProductMode == RhsMode) { return lhs(j, rhs.image(i)); }   // ColPermutation
    }
};

}   // namespace internals

// P * M (RowPermutation)
template <typename LhsXprType, typename RhsXprType>
constexpr auto operator*(
  const PermutationMatrixExpr<LhsXprType::Rows, LhsXprType::Cols, LhsXprType>& lhs,
  const MatrixExpr<RhsXprType::Rows, RhsXprType::Cols, RhsXprType>& rhs) {
    return MatrixProductOp<
      LhsXprType, RhsXprType, internals::permutation_product_executor<LhsXprType, RhsXprType>, LhsMode> {
      lhs.derived(), rhs.derived()};
}
// M * P (ColPermutation)
template <typename LhsXprType, typename RhsXprType>
constexpr auto operator*(
  const MatrixExpr<LhsXprType::Rows, LhsXprType::Cols, LhsXprType>& lhs,
  const PermutationMatrixExpr<RhsXprType::Rows, RhsXprType::Cols, RhsXprType>& rhs) {
    return MatrixProductOp<
      LhsXprType, RhsXprType, internals::permutation_product_executor<LhsXprType, RhsXprType>, RhsMode> {
      lhs.derived(), rhs.derived()};
}

// symmetric group product closure
template <typename LhsXprType, typename RhsXprType>
struct PermutationCompositionOp :
    public PermutationMatrixExpr<LhsXprType::Rows, RhsXprType::Cols, PermutationCompositionOp<LhsXprType, RhsXprType>> {
    fdapde_static_assert(
      internals::is_dynamic_sized_v<LhsXprType> || internals::is_dynamic_sized_v<RhsXprType> ||
        LhsXprType::Cols == RhsXprType::Rows,
      INVALID_STATIC_SIZED_OPERANDS_FOR_PERMUTATION_COMPOSITION);
    using Base =
      PermutationMatrixExpr<LhsXprType::Rows, RhsXprType::Cols, PermutationCompositionOp<LhsXprType, RhsXprType>>;
    using LhsXprTypeNested = internals::ref_select_t<LhsXprType>;
    using RhsXprTypeNested = internals::ref_select_t<RhsXprType>;
    using Scalar = int;
    static constexpr int Rows = LhsXprType::Rows;
    static constexpr int Cols = LhsXprType::Cols;
    static constexpr int NestAsRef = 0;
    static constexpr int ReadOnly = 1;

    template <typename LhsXprType_, typename RhsXprType_>
        requires(std::is_constructible_v<LhsXprTypeNested, LhsXprType_> &&
                 std::is_constructible_v<RhsXprTypeNested, RhsXprType_>)
    constexpr PermutationCompositionOp(LhsXprType_&& lhs, RhsXprType_&& rhs) :
        lhs_(std::forward<LhsXprType_>(lhs)), rhs_(std::forward<RhsXprType_>(rhs)) {
        constexpr int LhsRows = LhsXprType::Rows, LhsCols = LhsXprType::Cols;
        constexpr int RhsRows = RhsXprType::Rows, RhsCols = RhsXprType::Cols;
        if constexpr (internals::is_dynamic_sized_v<LhsXprType> || internals::is_dynamic_sized_v<RhsXprType>) {
            fdapde_constexpr_assert(lhs_.rows() == rhs_.rows() && lhs_.cols() == rhs_.cols());
        }
    }

    constexpr Scalar operator()(int i, int j) const { return (image(i) == j) ? Scalar(1) : Scalar(0); }
    // image of i under the permutation
    constexpr int image(int i) const {
        fdapde_constexpr_assert(i >= 0 && i < lhs_.rows());
        return lhs_.image(rhs_.image(i));
    }
    constexpr Vector<Scalar, Rows> permutation() const {   // materialize the permutation vector
        Vector<Scalar, Rows> p;
        if constexpr (Rows == Dynamic) { p.resize(lhs_.rows()); }
        for (int i = 0, n = lhs_.rows(); i < n; ++i) { p[i] = image(i); }
        return p;
    }
    constexpr int rows() const { return lhs_.rows(); }
    constexpr int cols() const { return rhs_.cols(); }
   private:
    LhsXprTypeNested lhs_;
    RhsXprTypeNested rhs_;
};
template <typename LhsXprType, typename RhsXprType>
constexpr auto operator*(
  const PermutationMatrixExpr<LhsXprType::Rows, LhsXprType::Cols, LhsXprType>& lhs,
  const PermutationMatrixExpr<RhsXprType::Rows, RhsXprType::Cols, RhsXprType>& rhs) {
    return PermutationCompositionOp<LhsXprType, RhsXprType>(lhs.derived(), rhs.derived());
}
  
// permutation matrix
template <int Size_> struct PermutationMatrix : public PermutationMatrixExpr<Size_, Size_, PermutationMatrix<Size_>> {
    using Base = PermutationMatrixExpr<Size_, Size_, PermutationMatrix<Size_>>;
    using Scalar = int;
    using StorageType = Vector<Scalar, Size_>;
    static constexpr int Rows = Size_;
    static constexpr int Cols = Size_;
    static constexpr int StorageSize = Size_;
    static constexpr int NestAsRef = 0;
    static constexpr int ReadOnly = 1;

    // constructors
    constexpr PermutationMatrix() = delete;   // empty permutations are ill-formed
    template <typename DataT>
        requires(internals::is_vector_like_v<DataT> && !internals::is_matrix_like_v<DataT>)
    constexpr explicit PermutationMatrix(DataT&& permutation) : permutation_(permutation) {
        if constexpr (Size_ != Dynamic) { fdapde_constexpr_assert(permutation_.size() == Size_); }
    }
    template <std::size_t RhsSize>
    constexpr explicit PermutationMatrix(const Scalar (&permutation)[RhsSize]) : permutation_(permutation) {
        fdapde_static_assert(Size_ != Dynamic && StorageSize == RhsSize, THIS_METHOD_IS_FOR_STATIC_SIZED_MATRICES_ONLY);
    }

    // observers
    constexpr int rows() const { return permutation_.size(); }
    constexpr int cols() const { return permutation_.size(); }
    constexpr Scalar operator()(int i, int j) const { return permutation_[i] == j ? 1 : 0; }
    constexpr const StorageType& permutation() const { return permutation_; }
    constexpr int image(int i) const {
        fdapde_constexpr_assert(i >= 0 && i < rows());
        return permutation_[i];
    }
   private:
    StorageType permutation_;
};
  
}   // namespace fdapde

#endif   // __FDAPDE_LINALG_PERMUTATION_H__
