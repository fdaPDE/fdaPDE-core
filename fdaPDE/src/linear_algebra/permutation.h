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

/// @brief represents permutation inverse op
template <typename XprType> struct PermutationInverseOp;

/// @brief represents permutation matrix expr
template <typename XprType_> struct PermutationMatrixExpr : public OrthogonalMatrixExpr<XprType_> {
   private:
    using Base = OrthogonalMatrixExpr<XprType_>;
    using XprType = std::remove_cvref_t<XprType_>;
   public:
    using Base::derived;

    /// @brief returns the inverse matrix expression
    constexpr auto inverse() const& { return PermutationInverseOp<XprType_>(derived()); }
    /// @brief returns the inverse matrix expression
    constexpr auto inverse() const&&
        requires(XprType::NestAsRef == 0)
    {
        return PermutationInverseOp<XprType_>(derived());
    }
    /// @brief returns the inverse matrix expression
    constexpr void inverse() const&&
        requires(XprType::NestAsRef != 0)
    = delete;
    /// @brief returns the matrix determinant
    constexpr auto determinant() const {
        using Scalar = typename XprType::Scalar;
        constexpr int Rows = XprType::Rows;

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
    /// @brief returns the sum of squared coefficients
    constexpr auto squared_norm() const { return derived().rows(); }
    /// @brief returns the Euclidean or Frobenius norm
    constexpr auto norm() const { return fdapde::sqrt(static_cast<double>(squared_norm())); }
    // ostream
    /// @brief implements the operator<< expression operation
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
/// @brief represents permutation inverse op
template <typename XprType_>
struct PermutationInverseOp : public PermutationMatrixExpr<PermutationInverseOp<XprType_>> {
   private:
    using XprType = std::remove_cvref_t<XprType_>;
    using XprTypeNested = internals::ref_select_t<const XprType>;
   public:
    using Scalar = typename XprType::Scalar;
    static constexpr int Rows = XprType::Rows;
    static constexpr int Cols = XprType::Cols;
    static constexpr int StorageOrder = XprType::StorageOrder;
    static constexpr int NestAsRef = 0;
    static constexpr int ReadOnly = 1;

    /// @brief constructs permutation inverse op from the supplied state
    constexpr PermutationInverseOp(const PermutationInverseOp&) = default;
    /// @brief constructs permutation inverse op from the supplied state
    template <typename XprType__>
        requires(
          !std::same_as<std::remove_cvref_t<XprType__>, PermutationInverseOp> &&
          internals::safely_nestable<XprTypeNested, XprType__>)
    constexpr explicit PermutationInverseOp(XprType__&& xpr) : xpr_(std::forward<XprType__>(xpr)) { }
    /// @brief accesses or evaluates the requested coefficient
    constexpr Scalar operator()(int i, int j) const {
        internals::validate_matrix_index(i, j, rows(), cols());
        return xpr_.image(j) == i ? Scalar(1) : Scalar(0);
    }
    /// @brief returns the destination index of a permutation entry
    constexpr int image(int i) const {   // image(i) returns pi^{-1}(i)
        fdapde_assert(!(i < 0 || i >= xpr_.rows()), std::out_of_range, "permutation index out of range");
        for (int j = 0, n = xpr_.rows(); j < n; ++j) {
            if (xpr_.image(j) == i) { return j; }
        }
        throw std::logic_error("invalid permutation inverse");
    }
    /// @brief returns the permutation index vector
    constexpr Vector<Scalar, Rows> permutation() const {   // materialize the permutation vector
        Vector<Scalar, Rows> p;
        if constexpr (Rows == Dynamic) { p.resize(xpr_.rows()); }
        for (int i = 0, n = xpr_.rows(); i < n; ++i) { p[i] = image(i); }
        return p;
    }
    // observers
    /// @brief returns the row count
    constexpr int rows() const { return xpr_.rows(); }
    /// @brief returns the column count
    constexpr int cols() const { return xpr_.cols(); }
   private:
    XprTypeNested xpr_;
};

// action products
namespace internals {

/// @brief represents permutation product executor
template <typename LhsXprType, typename RhsXprType, int ProductMode> struct permutation_product_executor {
    /// @brief executes the coefficient operation over the supplied expressions
    static constexpr auto run(int i, int j, const LhsXprType& lhs, const RhsXprType& rhs) {
        using Scalar = promote_type_t<typename LhsXprType::Scalar, typename RhsXprType::Scalar>;
        if constexpr (ProductMode == LhsMode) { return Scalar(rhs(lhs.image(i), j)); }   // RowPermutation
        if constexpr (ProductMode == RhsMode) {
            for (int k = 0; k < rhs.rows(); ++k) {
                if (rhs.image(k) == j) return Scalar(lhs(i, k));
            }
            throw std::logic_error("invalid right permutation action");
        }
    }
};

}   // namespace internals

// p * M (RowPermutation)
/// @brief implements the operator* expression operation
template <typename LhsXprType, typename RhsXprType>
    requires(
      !is_orthogonal_matrix_v<RhsXprType> && !is_diagonal_matrix_v<RhsXprType> && !is_triangular_matrix_v<RhsXprType>)
constexpr auto operator*(const PermutationMatrixExpr<LhsXprType>& lhs, const MatrixExpr<RhsXprType>& rhs) {
    return MatrixMultiplicationOp<
      LhsXprType, RhsXprType, internals::permutation_product_executor<LhsXprType, RhsXprType, LhsMode>> {
      lhs.derived(), rhs.derived()};
}
// m * P (ColPermutation)
/// @brief implements the operator* expression operation
template <typename LhsXprType, typename RhsXprType>
    requires(
      !is_orthogonal_matrix_v<LhsXprType> && !is_diagonal_matrix_v<LhsXprType> && !is_triangular_matrix_v<LhsXprType>)
constexpr auto operator*(const MatrixExpr<LhsXprType>& lhs, const PermutationMatrixExpr<RhsXprType>& rhs) {
    return MatrixMultiplicationOp<
      LhsXprType, RhsXprType, internals::permutation_product_executor<LhsXprType, RhsXprType, RhsMode>> {
      lhs.derived(), rhs.derived()};
}

// symmetric group product closure
/// @brief represents permutation composition op
template <typename LhsXprType_, typename RhsXprType_>
struct PermutationCompositionOp : public PermutationMatrixExpr<PermutationCompositionOp<LhsXprType_, RhsXprType_>> {
   private:
    using LhsXprType = std::remove_cvref_t<LhsXprType_>;
    using RhsXprType = std::remove_cvref_t<RhsXprType_>;
    fdapde_static_assert(
      internals::is_dynamic_sized_v<LhsXprType> || internals::is_dynamic_sized_v<RhsXprType> ||
        LhsXprType::Cols == RhsXprType::Rows,
      INVALID_COMPOSITION_OPERATION__NOT_MATCHING_OPERANDS_STATIC_SIZE);
    using LhsXprTypeNested = internals::ref_select_t<const LhsXprType>;
    using RhsXprTypeNested = internals::ref_select_t<const RhsXprType>;
   public:
    using Scalar = promote_type_t<typename LhsXprType::Scalar, typename RhsXprType::Scalar>;
    static constexpr int Rows = LhsXprType::Rows;
    static constexpr int Cols = RhsXprType::Cols;
    static constexpr int StorageOrder =
      internals::promote_storage_order_v<LhsXprType::StorageOrder, RhsXprType::StorageOrder>;
    static constexpr int NestAsRef = 0;
    static constexpr int ReadOnly = 1;

    /// @brief constructs permutation composition op from the supplied state
    template <typename LhsXprType__, typename RhsXprType__>
        requires(internals::safely_nestable<LhsXprTypeNested, LhsXprType__> &&
                 internals::safely_nestable<RhsXprTypeNested, RhsXprType__>)
    constexpr PermutationCompositionOp(LhsXprType__&& lhs, RhsXprType__&& rhs) :
        lhs_(std::forward<LhsXprType__>(lhs)), rhs_(std::forward<RhsXprType__>(rhs)) {
        fdapde_assert(
          !(lhs_.rows() != rhs_.rows() || lhs_.cols() != rhs_.cols()), std::invalid_argument,
          "permutation composition requires matching shapes");
    }
    /// @brief accesses or evaluates the requested coefficient
    constexpr Scalar operator()(int i, int j) const {
        internals::validate_matrix_index(i, j, rows(), cols());
        return (image(i) == j) ? Scalar(1) : Scalar(0);
    }
    /// @brief returns the destination index of a permutation entry
    constexpr int image(int i) const {   // image of i under the permutation
        fdapde_assert(!(i < 0 || i >= lhs_.rows()), std::out_of_range, "permutation index out of range");
        return rhs_.image(lhs_.image(i));
    }
    /// @brief returns the permutation index vector
    constexpr Vector<Scalar, Rows> permutation() const {   // materialize the permutation vector
        Vector<Scalar, Rows> p;
        if constexpr (Rows == Dynamic) { p.resize(lhs_.rows()); }
        for (int i = 0, n = lhs_.rows(); i < n; ++i) { p[i] = image(i); }
        return p;
    }
    /// @brief returns the row count
    constexpr int rows() const { return lhs_.rows(); }
    /// @brief returns the column count
    constexpr int cols() const { return rhs_.cols(); }
   private:
    LhsXprTypeNested lhs_;
    RhsXprTypeNested rhs_;
};
/// @brief implements the operator* expression operation
template <typename LhsXprType, typename RhsXprType>
constexpr auto operator*(const PermutationMatrixExpr<LhsXprType>& lhs, const PermutationMatrixExpr<RhsXprType>& rhs) {
    return PermutationCompositionOp<LhsXprType, RhsXprType>(lhs.derived(), rhs.derived());
}

// owning permutation matrix type.
/// @brief represents a permutation matrix
template <int Rows_, int Cols_>
class PermutationMatrix : public PermutationMatrixExpr<PermutationMatrix<Rows_, Cols_>> {
    fdapde_static_assert((Rows_ == Dynamic || Rows_ > 0) && (Cols_ == Dynamic || Cols_ > 0), INVALID_MATRIX_DIMENSIONS);
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
    /// @brief constructs permutation matrix from the supplied state
    constexpr PermutationMatrix() : permutation_() {
        if constexpr (Rows_ != Dynamic || Cols_ != Dynamic) set_identity_(static_shape_());
    }
    // copy-semantic
    /// @brief constructs permutation matrix from the supplied state
    constexpr PermutationMatrix(const PermutationMatrix& other) : permutation_() { clone_(other); }
    /// @brief assigns the supplied coefficients
    constexpr PermutationMatrix& operator=(const PermutationMatrix& rhs) & {
        clone_(rhs);
        return *this;
    }
    /// @brief constructs permutation matrix from the supplied state
    template <typename RhsXprType_>
        requires(std::integral<std::remove_cv_t<typename std::remove_cvref_t<RhsXprType_>::Scalar>>)
    constexpr explicit PermutationMatrix(const MatrixExpr<RhsXprType_>& rhs) : permutation_() {
        using RhsXprType = std::remove_cvref_t<RhsXprType_>;
        fdapde_static_assert(
          RhsXprType::Rows == Dynamic || RhsXprType::Cols == Dynamic || RhsXprType::Rows == 1 || RhsXprType::Cols == 1,
          THIS_METHOD_IS_ONLY_FOR_ROW_OR_COLUMN_VECTORS);
        fdapde_assert(
          !(rhs.rows() != 1 && rhs.cols() != 1), std::invalid_argument,
          "permutation input must be a row or column vector");
        load_(rhs.size(), [&rhs](int i) { return rhs.rows() == 1 ? rhs.derived()(0, i) : rhs.derived()(i, 0); });
    }
    /// @brief constructs permutation matrix from the supplied state
    template <typename Scalar__>
        requires(std::integral<std::remove_cv_t<Scalar__>>)
    constexpr explicit PermutationMatrix(const std::vector<Scalar__>& vec) : permutation_() {
        const int size = internals::checked_matrix_data_size(vec.size());
        load_(size, [&vec](int i) { return vec[static_cast<std::size_t>(i)]; });
    }
    /// @brief constructs permutation matrix from the supplied state
    template <typename Scalar__, std::size_t Size>
        requires(std::integral<std::remove_cv_t<Scalar__>>)
    constexpr explicit PermutationMatrix(const Scalar__ (&permutation)[Size]) : permutation_() {
        fdapde_static_assert(
          Rows_ != Dynamic && Cols_ != Dynamic && Rows_ == Size && Cols_ == Size,
          THIS_METHOD_IS_FOR_STATIC_SIZED_MATRICES_ONLY);
        load_(static_cast<int>(Size), [&permutation](int i) { return permutation[static_cast<std::size_t>(i)]; });
    }
    // observers
    /// @brief returns the row count
    constexpr int rows() const { return permutation_.size(); }
    /// @brief returns the column count
    constexpr int cols() const { return permutation_.size(); }
    /// @brief accesses or evaluates the requested coefficient
    constexpr Scalar operator()(int i, int j) const {
        internals::validate_matrix_index(i, j, rows(), cols());
        return permutation_[i] == j ? Scalar(1) : Scalar(0);
    }
    /// @brief returns the permutation index vector
    constexpr const StorageType& permutation() const { return permutation_; }
    /// @brief returns the destination index of a permutation entry
    constexpr int image(int i) const {
        fdapde_assert(!(i < 0 || i >= rows()), std::out_of_range, "permutation index out of range");
        return permutation_[i];
    }
   private:
    /// @brief returns the compile-time permutation dimension
    static constexpr int static_shape_() {
        if constexpr (Rows_ != Dynamic) return Rows_;
        if constexpr (Cols_ != Dynamic) return Cols_;
        return 0;
    }
    /// @brief sets identity
    constexpr void set_identity_(int size) {
        if constexpr (Rows_ == Dynamic) permutation_.resize(size);
        for (int i = 0; i < size; ++i) permutation_[i] = i;
    }
    /// @brief validates the dimension and prepares permutation storage
    constexpr void prepare_size_(int size) {
        fdapde_assert(
          !(size < 0 || (Rows_ != Dynamic && size != Rows_) || (Cols_ != Dynamic && size != Cols_)),
          std::invalid_argument, "permutation size does not match its static shape");
        if constexpr (Rows_ == Dynamic) permutation_.resize(size);
    }
    /// @brief loads and validates a permutation
    template <typename Reader> constexpr void load_(int size, Reader&& read) {
        prepare_size_(size);
        for (int i = 0; i < size; ++i) {
            const auto raw = read(i);
            fdapde_assert(!(!std::in_range<int>(raw)), std::invalid_argument, "permutation entry is outside int range");
            const int value = static_cast<int>(raw);
            fdapde_assert(
              !(value < 0 || value >= size), std::invalid_argument, "permutation entry is outside its index range");
            for (int j = 0; j < i; ++j) {
                fdapde_assert(!(permutation_[j] == value), std::invalid_argument, "permutation entries must be unique");
            }
            permutation_[i] = value;
        }
    }
    /// @brief returns an owning copy of the coefficients
    constexpr void clone_(const PermutationMatrix& rhs) {
        if (this == std::addressof(rhs)) return;
        if constexpr (Rows_ == Dynamic) permutation_.resize(rhs.rows());
        for (int i = 0; i < rhs.rows(); ++i) permutation_[i] = rhs.permutation_[i];
    }
    StorageType permutation_;
};

// detection trait
/// @brief detects is permutation matrix
template <typename XprType> struct is_permutation_matrix {
    using Type = std::remove_cvref_t<XprType>;
    static constexpr bool value = std::is_base_of_v<PermutationMatrixExpr<Type>, Type>;
};
template <typename XprType> static constexpr bool is_permutation_matrix_v = is_permutation_matrix<XprType>::value;

}   // namespace fdapde

#endif   // __FDAPDE_LINALG_PERMUTATION_H__
