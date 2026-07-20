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

namespace fdapde::linalg {

// implementation of the symmetric group S_n

template <typename XprType> struct PermutationInverseOp;

template <typename XprType_>
struct PermutationMatrixExpr : public OrthogonalMatrixExpr<XprType_> {
    using XprType = std::decay_t<XprType_>;
    using OrthogonalMatrixExpr<XprType_>::derived;

    constexpr auto inverse() const & { return PermutationInverseOp<XprType_>(this->derived()); }
    constexpr auto inverse() const && requires(XprType::NestAsRef == 0) {
        return PermutationInverseOp<XprType_>(this->derived());
    }
    constexpr void inverse() const && requires(XprType::NestAsRef != 0) = delete;
    constexpr auto determinant() const {
        using Scalar = typename XprType_::Scalar;
        constexpr int Rows = XprType_::Rows;

        Vector<Scalar, Rows> permutation = derived().permutation();
        const int n = permutation.size();
        Vector<char, Rows> visited;
        if constexpr (Rows == Dynamic) visited.resize(n);
        for (int i = 0; i < n; ++i) visited[i] = 0;
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
    constexpr auto norm() const { return fdapde::sqrt(static_cast<double>(squared_norm())); }
    // ostream
    friend std::ostream& operator<<(std::ostream& out, const PermutationMatrixExpr& m) {
        const int rows = m.derived().rows();
        const int cols = m.derived().cols();
        if (rows == 0 || cols == 0) return out;
        auto perm = m.derived().permutation();   // evaluate permutation mapping
        for (int i = 0; i < rows - 1; ++i) {
            for (int j = 0; j < cols; ++j) { out << (perm[i] == j ? 1 : 0) << " "; }
            out << "\n";
        }
        // print last row without carriage return
        for (int j = 0; j < cols; ++j) { out << (perm[rows - 1] == j ? 1 : 0) << " "; }
        return out;
    }
};

// expression of the inverse of a permutation
template <typename XprType_> struct PermutationInverseOp : public PermutationMatrixExpr<PermutationInverseOp<XprType_>> {
   private:
    using XprType = std::decay_t<XprType_>;
    using XprTypeNested = fdapde::internals::ref_select_t<const XprType>;
   public:
    using Scalar = typename XprType::Scalar;
    static constexpr int Rows = XprType::Rows;
    static constexpr int Cols = XprType::Cols;
    static constexpr int StorageOrder = XprType::StorageOrder;
    static constexpr int NestAsRef = 0;
    static constexpr int ReadOnly = 1;

    constexpr PermutationInverseOp(const PermutationInverseOp&) = default;
    template <typename XprType__>
        requires(
          !std::same_as<std::remove_cvref_t<XprType__>, PermutationInverseOp> &&
          internals::safely_nestable<XprTypeNested, XprType__>)
    constexpr explicit PermutationInverseOp(XprType__&& xpr) : xpr_(std::forward<XprType__>(xpr)) { }
    constexpr Scalar operator()(int i, int j) const { return xpr_.image(j) == i ? 1 : 0; }   // matrix transposition
    constexpr int image(int i) const {   // image(i) returns pi^{-1}(i)
        fdapde_assert(i >= 0 && i < xpr_.rows());
        for (int j = 0, n = xpr_.rows(); j < n; ++j) {
            if (xpr_.image(j) == i) { return j; }
        }
        fdapde_assert(false);
        return 0;
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
        using Scalar = promote_type_t<typename LhsXprType::Scalar, typename RhsXprType::Scalar>;
        if constexpr (ProductMode == LhsMode) { return Scalar(rhs(lhs.image(i), j)); }   // RowPermutation
        if constexpr (ProductMode == RhsMode) {
            for (int k = 0; k < rhs.rows(); ++k) {
                if (rhs.image(k) == j) return Scalar(lhs(i, k));
            }
            fdapde_assert(false);
            return Scalar(0);
        }
    }
};

}   // namespace internals

// P * M (RowPermutation)
template <typename LhsXprType, typename RhsXprType>
    requires(
      !is_orthogonal_matrix_v<RhsXprType> && !is_diagonal_matrix_v<RhsXprType> &&
      !is_triangular_matrix_v<RhsXprType>)
constexpr auto operator*(const PermutationMatrixExpr<LhsXprType>& lhs, const MatrixExpr<RhsXprType>& rhs) {
    return MatrixMultiplicationOp<
      LhsXprType, RhsXprType, internals::permutation_product_executor<LhsXprType, RhsXprType, LhsMode>> {
      lhs.derived(), rhs.derived()};
}
// M * P (ColPermutation)
template <typename LhsXprType, typename RhsXprType>
    requires(
      !is_orthogonal_matrix_v<LhsXprType> && !is_diagonal_matrix_v<LhsXprType> &&
      !is_triangular_matrix_v<LhsXprType>)
constexpr auto operator*(const MatrixExpr<LhsXprType>& lhs, const PermutationMatrixExpr<RhsXprType>& rhs) {
    return MatrixMultiplicationOp<
      LhsXprType, RhsXprType, internals::permutation_product_executor<LhsXprType, RhsXprType, RhsMode>> {
      lhs.derived(), rhs.derived()};
}

// symmetric group product closure
template <typename LhsXprType_, typename RhsXprType_>
struct PermutationCompositionOp : public PermutationMatrixExpr<PermutationCompositionOp<LhsXprType_, RhsXprType_>> {
   private:
    using LhsXprType = std::decay_t<LhsXprType_>;
    using RhsXprType = std::decay_t<RhsXprType_>;
    fdapde_static_assert(
      internals::is_dynamic_sized_v<LhsXprType> || internals::is_dynamic_sized_v<RhsXprType> ||
        LhsXprType::Cols == RhsXprType::Rows,
      INVALID_COMPOSITION_OPERATION__NOT_MATCHING_OPERANDS_STATIC_SIZE);
    using LhsXprTypeNested = fdapde::internals::ref_select_t<const LhsXprType>;
    using RhsXprTypeNested = fdapde::internals::ref_select_t<const RhsXprType>;
   public:
    using Scalar = promote_type_t<typename LhsXprType::Scalar, typename RhsXprType::Scalar>;
    static constexpr int Rows = LhsXprType::Rows;
    static constexpr int Cols = RhsXprType::Cols;
    static constexpr int StorageOrder =
      internals::promote_storage_order_v<LhsXprType::StorageOrder, RhsXprType::StorageOrder>;
    static constexpr int NestAsRef = 0;
    static constexpr int ReadOnly = 1;

    template <typename LhsXprType__, typename RhsXprType__>
        requires(internals::safely_nestable<LhsXprTypeNested, LhsXprType__> &&
                 internals::safely_nestable<RhsXprTypeNested, RhsXprType__>)
    constexpr PermutationCompositionOp(LhsXprType__&& lhs, RhsXprType__&& rhs) :
        lhs_(std::forward<LhsXprType__>(lhs)), rhs_(std::forward<RhsXprType__>(rhs)) {
        if constexpr (internals::is_dynamic_sized_v<LhsXprType> || internals::is_dynamic_sized_v<RhsXprType>) {
            fdapde_assert(lhs_.rows() == rhs_.rows() && lhs_.cols() == rhs_.cols());
        }
    }
    constexpr Scalar operator()(int i, int j) const { return (image(i) == j) ? Scalar(1) : Scalar(0); }
    constexpr int image(int i) const {   // image of i under the permutation
        fdapde_assert(i >= 0 && i < lhs_.rows());
        return rhs_.image(lhs_.image(i));
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
constexpr auto operator*(const PermutationMatrixExpr<LhsXprType>& lhs, const PermutationMatrixExpr<RhsXprType>& rhs) {
    return PermutationCompositionOp<LhsXprType, RhsXprType>(lhs.derived(), rhs.derived());
}

// owning permutation matrix type.
template <int Rows_, int Cols_>
class PermutationMatrix : public PermutationMatrixExpr<PermutationMatrix<Rows_, Cols_>> {
    fdapde_static_assert(
      Rows_ == Dynamic || Cols_ == Dynamic || Rows_ == Cols_, THIS_CLASS_IS_FOR_SQUARE_MATRICES_ONLY);
    using StorageType = Vector<int, Rows_>;
   public:
    using Scalar = int;
    static constexpr int Rows = Rows_;
    static constexpr int Cols = Cols_;
    static constexpr int StorageOrder = StorageType::StorageOrder;
    static constexpr int NestAsRef = 1;
    static constexpr int ReadOnly = 1;
    using assignment_executor = typename StorageType::assignment_executor;
    // constructors
    constexpr PermutationMatrix() noexcept : permutation_() {
        if constexpr (Rows_ != Dynamic || Cols_ != Dynamic) set_identity_(static_shape_());
    }
    // copy-semantic
    constexpr PermutationMatrix(const PermutationMatrix& other) { clone_(other); }
    constexpr PermutationMatrix& operator=(const PermutationMatrix& rhs) & {
        clone_(rhs);
        return *this;
    }
    template <typename RhsXprType_>
        requires(std::is_same_v<typename std::decay_t<RhsXprType_>::Scalar, Scalar>)
    constexpr explicit PermutationMatrix(const MatrixExpr<RhsXprType_>& rhs) : permutation_(rhs) {
        using RhsXprType = std::decay_t<RhsXprType_>;
        fdapde_static_assert(
          RhsXprType::Rows == 1 || RhsXprType::Cols == 1, THIS_METHOD_IS_ONLY_FOR_ROW_OR_COLUMN_VECTORS);
        validate_or_identity_();
    }
    template <typename Scalar__>
        requires(std::integral<std::remove_cv_t<Scalar__>>)
    constexpr explicit PermutationMatrix(const std::vector<Scalar__>& vec) : permutation_() {
        load_(vec);
    }
    template <typename Scalar__, std::size_t Size>
        requires(std::integral<std::remove_cv_t<Scalar__>>)
    constexpr explicit PermutationMatrix(const Scalar__ (&permutation)[Size]) : permutation_() {
        fdapde_static_assert(
          Rows_ != Dynamic && Cols_ != Dynamic && Rows_ == Size && Cols_ == Size,
          THIS_METHOD_IS_FOR_STATIC_SIZED_MATRICES_ONLY);
        load_(permutation);
    }
    // observers
    constexpr int rows() const { return permutation_.size(); }
    constexpr int cols() const { return permutation_.size(); }
    constexpr Scalar operator()(int i, int j) const {
        if (i < 0 || i >= rows() || j < 0 || j >= cols()) {
            fdapde_assert(i >= 0 && i < rows() && j >= 0 && j < cols());
            return 0;
        }
        return permutation_[i] == j ? 1 : 0;
    }
    constexpr const StorageType& permutation() const { return permutation_; }
    constexpr int image(int i) const {
        if (i < 0 || i >= rows()) {
            fdapde_assert(i >= 0 && i < rows());
            return 0;
        }
        return permutation_[i];
    }
   private:
    static constexpr int static_shape_() {
        if constexpr (Rows_ != Dynamic) return Rows_;
        if constexpr (Cols_ != Dynamic) return Cols_;
        return 0;
    }
    constexpr void set_identity_(int size) {
        if constexpr (Rows_ == Dynamic) permutation_.resize(size);
        for (int i = 0; i < permutation_.size(); ++i) permutation_[i] = i;
    }
    template <typename Data> constexpr void load_(const Data& data) {
        const int size = static_cast<int>(std::size(data));
        if constexpr (Rows_ == Dynamic) permutation_.resize(size);
        if (permutation_.size() != size) {
            fdapde_assert(permutation_.size() == size);
            set_identity_(static_shape_());
            return;
        }
        for (int i = 0; i < size; ++i) {
            if (!std::in_range<int>(data[i])) {
                fdapde_assert(std::in_range<int>(data[i]));
                set_identity_(static_shape_() > 0 ? static_shape_() : size);
                return;
            }
            permutation_[i] = static_cast<int>(data[i]);
        }
        validate_or_identity_();
    }
    constexpr void validate_or_identity_() {
        const int size = permutation_.size();
        bool valid = (Rows_ == Dynamic || size == Rows_) && (Cols_ == Dynamic || size == Cols_);
        for (int i = 0; valid && i < size; ++i) {
            valid = permutation_[i] >= 0 && permutation_[i] < size;
            for (int j = 0; valid && j < i; ++j) valid = permutation_[i] != permutation_[j];
        }
        if (!valid) {
            fdapde_assert(valid);
            set_identity_(static_shape_() > 0 ? static_shape_() : size);
        }
    }
    template <typename RhsXprType> constexpr void clone_(const RhsXprType& rhs) {
        if constexpr (Rows_ == Dynamic || Cols_ == Dynamic) { permutation_.resize(rhs.rows()); }
        assignment_executor::run(permutation_, rhs.permutation(), [](auto&& l, const auto& r) { l = r; });
        validate_or_identity_();
    }
    StorageType permutation_;
};

// detection trait
template <typename XprType> struct is_permutation_matrix {
    using Type = std::remove_cvref_t<XprType>;
    static constexpr bool value = std::is_base_of_v<PermutationMatrixExpr<Type>, Type>;
};
template <typename XprType> static constexpr bool is_permutation_matrix_v = is_permutation_matrix<XprType>::value;

}   // namespace fdapde::linalg

#endif   // __FDAPDE_LINALG_PERMUTATION_H__
