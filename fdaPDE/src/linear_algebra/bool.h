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

#ifndef __FDAPDE_LINALG_BOOL_H__
#define __FDAPDE_LINALG_BOOL_H__

#include "header_check.h"

#include <bitset>


namespace fdapde {

// linalg API specialization for the case Scalar = bool. Due to the boolean algebra semantic and the packed
// storage layout, boolean linalg API is different than that of the rest of the linalg module

template <int Rows_, int Cols_, typename XprType_> struct BoolMatrixExpr;

namespace internals {

struct bitpack_assignment_executor {
    template <typename DstMatrixType, typename SrcXprType, typename AssignmentOp>
        requires(
          requires(AssignmentOp op, typename DstMatrixType::bitpack_t& l, const typename SrcXprType::bitpack_t& r) {
              { op(l, r) } -> std::same_as<void>;
          })
    static constexpr void run(DstMatrixType& dst, const SrcXprType& src, AssignmentOp&& op) {
        fdapde_static_assert(DstMatrixType::ReadOnly == 0, ASSIGNMENT_TO_READ_ONLY_EXPRESSION);
        fdapde_static_assert(
          is_dynamic_sized_v<DstMatrixType> || is_dynamic_sized_v<SrcXprType> ||
            same_static_shape_v<DstMatrixType FDAPDE_COMMA SrcXprType>,
          INVALID_ASSIGNMENT__LHS_AND_RHS_STATIC_SIZES_DOES_NOT_MATCH);
        if constexpr (internals::is_dynamic_sized_v<DstMatrixType> || internals::is_dynamic_sized_v<SrcXprType>) {
            fdapde_assert(dst.rows() == src.rows() && dst.cols() == src.cols());
        }
        auto& d = dst.derived();
        const auto& s = src.derived();
        // fast bitpack assignment
        for (int i = 0, bitpacks_ = d.bitpacks(); i < bitpacks_; ++i) { op(d.bitpack(i), s.bitpack(i)); }
        return;
    }
};

}   // namespace internals

template <int Rows_, int Cols_, int StorageOrder_, typename BoolMatrixType>
class MatrixBase<bool, Rows_, Cols_, StorageOrder_, BoolMatrixType> :
    public BoolMatrixExpr<Rows_, Cols_, BoolMatrixType> {
    fdapde_static_assert((Rows_ == Dynamic || Rows_ > 0) && (Cols_ == Dynamic || Cols_ > 0), INVALID_MATRIX_DIMENSIONS);
   public:
    using Base = BoolMatrixExpr<Rows_, Cols_, BoolMatrixType>;
    using Base::derived;
    using Scalar = typename Base::Scalar;
    static constexpr int Rows = Rows_;
    static constexpr int Cols = Cols_;
    static constexpr int StorageOrder = StorageOrder_;
    static constexpr int NestAsRef = BoolMatrixType::NestAsRef;
    static constexpr int ReadOnly = std::is_const_v<Scalar> ? 1 : 0;
    static constexpr std::size_t PackSize = Base::PackSize;
    using assignment_executor = internals::bitpack_assignment_executor;
    using bitpack_t = typename Base::bitpack_t;

    // struct to proxy the behaviour of a reference to a single bit of a bitpack
    template <typename BitPackT>
        requires(std::is_same_v<std::decay_t<BitPackT>, bitpack_t>)
    struct bit_proxy {
       private:
        constexpr std::pair<int, int> pack_of_(int i, int j, int row_stride, int col_stride) const {
            int map = i * row_stride + j * col_stride;
            if constexpr ((PackSize & (PackSize - 1)) == 0) {   // avoid integer division if PackSize is a power of 2
                constexpr int PackShift = std::countr_zero(PackSize);
                return std::make_pair(map >> PackShift, map & (PackSize - 1));
            } else {   // generic fallback
                return std::make_pair(map / PackSize, map % PackSize);
            }
        }
       public:
        friend bit_proxy<bitpack_t>;
        friend bit_proxy<const bitpack_t>;
      
        constexpr bit_proxy() noexcept : data_(nullptr), pack_id_(0), bitmask_(0) { }
        template <typename BitPackT_>
        constexpr bit_proxy(const bit_proxy<BitPackT_>& other) :
            data_(const_cast<BitPackT*>(other.data_)), pack_id_(other.pack_id_), bitmask_(other.bitmask_) { }
        template <typename BitPackT_> constexpr bit_proxy& operator=(const bit_proxy<BitPackT_>& other) {
            data_ = const_cast<BitPackT*>(other.data_);
            pack_id_ = other.pack_id_;
            bitmask_ = other.bitmask_;
            return *this;
        }
        constexpr bit_proxy(BitPackT* data, int row, int col, int row_stride, int col_stride) :
            data_(data), pack_id_(), bitmask_() {
            auto [pack_id, bit_off] = pack_of_(row, col, row_stride, col_stride);
            pack_id_ = pack_id;
            bitmask_ = bitpack_t(1) << bit_off;
        }
        constexpr bit_proxy(BitPackT* data, int row) :
            data_(data), pack_id_(row / PackSize), bitmask_(bitpack_t(1) << row % PackSize) { }
        // modifiers
        constexpr void set()   { data_[pack_id_] |=  bitmask_; }
        constexpr void clear() { data_[pack_id_] &= ~bitmask_; }
        template <typename T>
            requires(std::is_convertible_v<T, bool>)
        constexpr bit_proxy& operator=(T b) {
            b ? set() : clear();
	    return *this;
        }
        // observers
        constexpr operator bool() const { return (data_[pack_id_] & bitmask_) != 0; }
        constexpr operator bool() { return (data_[pack_id_] & bitmask_) != 0; }
       private:
        BitPackT* data_;
        int pack_id_;
        bitpack_t bitmask_;
    };
    using reference = bit_proxy<bitpack_t>;
    using const_reference = bit_proxy<const bitpack_t>;

    // constructors
    constexpr MatrixBase() noexcept :
        rows_(Rows_ == Dynamic ? 0 : Rows_),
        cols_(Cols_ == Dynamic ? 0 : Cols_),
        row_stride_(StorageOrder == RowMajor ? cols_ : 1),
        col_stride_(StorageOrder == RowMajor ? 1 : rows_) {
    }
    constexpr MatrixBase(int rows, int cols) :
        rows_(Rows == Dynamic ? rows : Rows),
        cols_(Cols == Dynamic ? cols : Cols),
        row_stride_(StorageOrder == RowMajor ? cols_ : 1),
        col_stride_(StorageOrder == RowMajor ? 1 : rows_) {
        fdapde_static_assert(Rows_ != 1 && Cols_ != 1, THIS_METHOD_IS_FOR_PROPER_MATRICES);
    }
    constexpr MatrixBase(int size) :
        rows_(Rows == 1 ? 1 : size),
        cols_(Cols == 1 ? 1 : size),
        row_stride_(StorageOrder == RowMajor ? cols_ : 1),
        col_stride_(StorageOrder == RowMajor ? 1 : rows_) {
        fdapde_static_assert(Rows == 1 || Cols == 1, THIS_METHOD_IS_FOR_ROW_OR_COLUMN_VECTORS_ONLY);
    }
    // copy assignment
    constexpr BoolMatrixType& operator=(const BoolMatrixType& other) {
        fdapde_static_assert(ReadOnly == 0, ASSIGNMENT_TO_READ_ONLY_LOCATION);
        if (this == std::addressof(other)) { return derived(); }
        if constexpr (Rows_ == Dynamic || Cols_ == Dynamic) {
            if (rows_ != other.rows() || cols_ != other.cols()) { derived().resize(other.rows(), other.cols()); }
        }
        assignment_executor::run(*this, other, [](bitpack_t& l, const bitpack_t& r) { l = r; });
        return derived();
    }
    // inherit assignment from base
    using Base::operator=;
  
    // access
    constexpr reference operator()(int i, int j) {
        fdapde_assert(i >= 0 && i < rows_ && j >= 0 && j < cols_);
        return reference(derived().data(), i, j, row_stride_, col_stride_);
    }
    constexpr reference operator[](int i) {
        fdapde_static_assert(Rows == 1 || Cols == 1, THIS_METHOD_IS_FOR_ROW_OR_COLUMN_VECTORS_ONLY);
        fdapde_assert(i >= 0 && i < rows_ * cols_);
        return reference(derived().data(), i);
    }
    constexpr const_reference operator()(int i, int j) const {
        fdapde_static_assert(ReadOnly == 0, ASSIGNMENT_TO_READ_ONLY_LOCATION_IS_INVALID);
        fdapde_assert(i >= 0 && i < rows_ && j >= 0 && j < cols_);
        return const_reference(derived().data(), i, j, row_stride_, col_stride_);
    }
    constexpr const_reference operator[](int i) const {
        fdapde_static_assert(ReadOnly == 0, ASSIGNMENT_TO_READ_ONLY_LOCATION_IS_INVALID);
        fdapde_static_assert(Rows == 1 || Cols == 1, THIS_METHOD_IS_FOR_ROW_OR_COLUMN_VECTORS_ONLY);
        fdapde_assert(i >= 0 && i < rows_ * cols_);
        return const_reference(derived().data(), i);
    }
    // observers
    constexpr int rows() const { return Rows != Dynamic ? Rows : rows_; }
    constexpr int cols() const { return Cols != Dynamic ? Cols : cols_; }
    constexpr int size() const { return rows() * cols(); }
   protected:
    int rows_, cols_;
    int row_stride_, col_stride_;
};

template <int Rows_, int Cols_, int StorageOrder_>
class Matrix<bool, Rows_, Cols_, StorageOrder_> :
    public MatrixBase<bool, Rows_, Cols_, StorageOrder_, Matrix<bool, Rows_, Cols_, StorageOrder_>> {
   public:
    using Base = MatrixBase<bool, Rows_, Cols_, StorageOrder_, Matrix<bool, Rows_, Cols_, StorageOrder_>>;
    using bitpack_t = typename Base::bitpack_t;
    using Scalar = typename Base::Scalar;
    static constexpr std::size_t PackSize = Base::PackSize;
    static constexpr int NestAsRef = 1;
    static constexpr int Rows = Rows_;
    static constexpr int Cols = Cols_;
    static constexpr std::size_t StorageSize =
      Rows_ == Dynamic || Cols_ == Dynamic ? Dynamic : 1 + (Rows * Cols) / PackSize;
    using StorageType = std::conditional_t<
      Rows_ == Dynamic || Cols_ == Dynamic, std::vector<bitpack_t>,
      std::array<bitpack_t, (StorageSize < 0) ? 0 : static_cast<std::size_t>(StorageSize)>>;   // avoid clang narrowing
    using iterator = typename StorageType::iterator;
    using const_iterator = typename StorageType::const_iterator;

    constexpr Matrix() noexcept :
        Base(), data_(), bitpacks_(StorageSize == Dynamic ? 0 : 1 + fdapde::ceil((Rows * Cols) / PackSize)) { }
    constexpr Matrix(const Matrix& other) :
        Base(), data_(), bitpacks_(StorageSize == Dynamic ? 0 : 1 + fdapde::ceil((Rows * Cols) / PackSize)) {
        if constexpr (Rows_ == Dynamic || Cols_ == Dynamic) { resize(other.rows(), other.cols()); }
        using assignment = typename Base::assignment_executor;
        assignment::run(*this, other, [](bitpack_t& l, const bitpack_t& r) { l = r; });
    }
    constexpr Matrix& operator=(const Matrix& other) {
        bitpacks_ = other.bitpacks_;
        Base::operator=(other);
        return *this;
    }
    template <int RhsRows_, int RhsCols_, typename RhsXprType_>   // construct from plain BoolMatrixExpr
    constexpr Matrix(const BoolMatrixExpr<RhsRows_, RhsCols_, RhsXprType_>& rhs) :
        Base(), bitpacks_(StorageSize == Dynamic ? 0 : 1 + fdapde::ceil((Rows * Cols) / PackSize)) {
        if constexpr (Rows_ == Dynamic || Cols_ == Dynamic) { resize(rhs.rows(), rhs.cols()); }
        using assignment = typename Base::assignment_executor;
        assignment::run(*this, rhs.derived(), [](bitpack_t& l, const bitpack_t& r) { l = r; });
    }
    // inherit assignment from base
    using Base::operator=;
  
    // Matrix API
    // value-initialized static-sized matrix
    constexpr explicit Matrix(bool v)
        requires(Rows_ != Dynamic && Cols_ != Dynamic)
        : data_(), bitpacks_(StorageSize == Dynamic ? 0 : 1 + fdapde::ceil((Rows * Cols) / PackSize)) {
        bitpack_t v_ = v ? -1 : 0;
        for (int i = 0; i < bitpacks_; ++i) { data_[i] = v_; }
    }
    // false-initialized dynamic-sized matrix. For static-sized matrices does nothing (exposed for API compatibility)
    constexpr Matrix(int rows, int cols)
        requires(Rows_ != 1 && Cols_ != 1)
        : Base(rows, cols), bitpacks_(1 + std::ceil((rows * cols) / PackSize)) {
        if constexpr (Rows_ == Dynamic || Cols_ == Dynamic) { data_.resize(bitpacks_, 0); }
    }
    // value-initialized dynamic-sized matrix, avoid vectors
    constexpr Matrix(int rows, int cols, bool v)
        requires(Rows_ == Dynamic && Cols_ == Dynamic)
        : Matrix(rows, cols) {
        bitpack_t v_ = v ? -1 : 0;
        for (int i = 0; i < bitpacks_; ++i) { data_[i] = v_; }
    }

    // Vector API
    // false-initialized dynamic-sized vector
    constexpr explicit Matrix(int size)
        requires(Rows_ == Dynamic || Cols_ == Dynamic)
        : Base(size), data_(), bitpacks_(1 + std::ceil(size / PackSize)) {
        fdapde_static_assert(Rows_ == 1 || Cols_ == 1, THIS_METHOD_IS_FOR_ROW_OR_COLUMN_VECTORS_ONLY);
        if constexpr (Rows_ == Dynamic || Cols_ == Dynamic) { data_.resize(bitpacks_, 0); }
    }
    // value-initialized dynamic-sized vector
    constexpr Matrix(int size, Scalar v)
        requires(Rows_ == Dynamic || Cols_ == Dynamic)
        : Matrix(size) {
        fdapde_static_assert(
          (Rows_ == 1 && Cols_ == Dynamic) || (Cols_ == 1 && Rows_ == Dynamic),
          THIS_METHOD_IS_FOR_DYNAMIC_SIZED_ROW_OR_COLUMN_VECTORS_ONLY);
        bitpack_t v_ = v ? -1 : 0;
        for (int i = 0, n = data_.size(); i < n; ++i) { data_[i] = v_; }
    }

    // constructors taking external data
    template <typename Scalar, std::size_t Size>
        requires(std::is_convertible_v<Scalar, bool>)
    constexpr explicit Matrix(const Scalar (&data)[Size]) :
        Base(), data_(), bitpacks_(1 + fdapde::ceil(Size / PackSize)) {
        fdapde_static_assert(
          Rows_ != Dynamic && Cols_ != Dynamic && Rows * Cols == Size, THIS_METHOD_IS_FOR_STATIC_SIZED_MATRICES_ONLY);
        for (int i = 0; i < Rows; ++i) {
            for (int j = 0; j < Cols; ++j) {
                Base::operator()(i, j) = data[i * Base::row_stride_ + j * Base::col_stride_];
            }
        }
        return;
    }
    template <typename DataT>
        requires(internals::is_vector_like_v<DataT>)
    constexpr explicit Matrix(DataT&& data) : Base(), data_() {
        fdapde_static_assert(
          (Rows_ != Dynamic && Cols_ != Dynamic) || (Rows_ == 1 && Cols_ == Dynamic) ||
            (Cols_ == 1 && Rows_ == Dynamic),
          THIS_METHOD_IS_EITHER_FOR_ROW_OR_COLUMN_VECTORS_OR_FOR_STATIC_SIZED_MATRICES);
        if constexpr (Rows_ == Dynamic || Cols_ == Dynamic) { resize(data.size()); }
        fdapde_assert(data_.size() == data.size());
        for (int i = 0, size = data.size(); i < size; ++i) {
            if (data[i]) set(i / Base::rows_, i % Base::cols_);
        }
        bitpacks_ = 1 + fdapde::ceil(data_.size() / PackSize);
        return;
    }
    // observers
    constexpr int bitpacks() const { return bitpacks_; }
    constexpr bitpack_t bitpack(int i) const { return data_[i]; }
    constexpr bitpack_t& bitpack(int i) { return data_[i]; }

    // modifiers
    void resize(int rows, int cols) {
        fdapde_static_assert(Rows_ == Dynamic || Cols_ == Dynamic, THIS_METHOD_IS_FOR_DYNAMIC_SIZED_MATRICES_ONLY);
        const int rows_ = Rows_ == Dynamic ? rows : Rows_;
        const int cols_ = Cols_ == Dynamic ? cols : Cols_;
        if (rows_ == Base::rows_ && cols_ == Base::cols_) return;   // do not reallocate memory if sizes didn't changed
        // update and reallocate memory
        Base::rows_ = rows_;
        Base::cols_ = cols_;
        Base::row_stride_ = StorageOrder_ == RowMajor ? cols_ : 1;
        Base::col_stride_ = StorageOrder_ == RowMajor ? 1 : rows_;
        bitpacks_ = 1 + std::ceil((rows_ * cols_) / PackSize);
        data_.resize(bitpacks_, 0);
        return;
    }
    void resize(int size) {
        fdapde_static_assert(
          (Rows_ == 1 && Cols_ == Dynamic) || (Cols_ == 1 && Rows_ == Dynamic),
          THIS_METHOD_IS_FOR_DYNAMIC_SIZED_ROW_OR_COLUMN_VECTORS_ONLY);
        resize(Rows_ == Dynamic ? size : Rows_, Cols_ == Dynamic ? size : Cols_);
        return;
    }
    constexpr void set(int i, int j) { Base::operator()(i, j).set(); }
    constexpr void set(int i) {
        fdapde_static_assert(Rows == 1 || Cols == 1, THIS_METHOD_IS_FOR_ROW_OR_COLUMN_VECTORS_ONLY);
        Base::operator[](i).set();
    }
    constexpr void set() {   // sets all coeffients
        for (int i = 0, n = data_.size(); i < n; ++i) { data_[i] = -1; }
    }
    constexpr void clear(int i, int j) { Base::operator()(i, j).clear(); }
    constexpr void clear(int i) {
        fdapde_static_assert(Rows == 1 || Cols == 1, THIS_METHOD_IS_FOR_ROW_OR_COLUMN_VECTORS_ONLY);
        Base::operator[](i).clear();
    }
    constexpr void clear() {   // clears all coeffients
        for (int i = 0, n = data_.size(); i < n; ++i) { data_[i] = 0; }
    }

    // data pointers
    constexpr const bitpack_t* data() const { return data_.data(); }
    constexpr bitpack_t* data() { return data_.data(); }

    // iterators
    constexpr iterator begin() { return data_.begin(); }
    constexpr const_iterator begin() const { return data_.begin(); }
    constexpr iterator end() { return data_.end(); }
    constexpr const_iterator end() const { return data_.end(); }
   private:
    StorageType data_;
    int bitpacks_;
};

// unary bitwise operation on binary expression
template <typename XprType, typename BitWiseOperation, typename BitPackOperation>
struct BoolMatrixBitWiseOp :
    public BoolMatrixExpr<
      XprType::Rows, XprType::Cols, BoolMatrixBitWiseOp<XprType, BitWiseOperation, BitPackOperation>> {
    using Base =
      BoolMatrixExpr<XprType::Rows, XprType::Cols, BoolMatrixBitWiseOp<XprType, BitWiseOperation, BitPackOperation>>;
    using XprTypeNested = internals::ref_select_t<const XprType>;
    using XprTypeClean = std::decay_t<XprType>;
    using bitpack_t = typename Base::bitpack_t;
    static constexpr int Rows = XprTypeClean::Rows;
    static constexpr int Cols = XprTypeClean::Cols;
    static constexpr int NestAsRef = 0;
    static constexpr int ReadOnly = 1;

    template <typename XprType_>
        requires(std::is_constructible_v<XprTypeNested, XprType_>)
    constexpr BoolMatrixBitWiseOp(XprType_&& xpr, BitWiseOperation bitwise_op, BitPackOperation bitpack_op) :
        xpr_(std::forward<XprType_>(xpr)), bitwise_op_(bitwise_op), bitpack_op_(bitpack_op) { }

    constexpr bool operator()(int i, int j) const { return bitwise_op_(xpr_(i, j)); }
    constexpr bool operator[](int i) const {
        fdapde_static_assert(Rows == 1 || Cols == 1, THIS_METHOD_IS_FOR_ROW_OR_COLUMN_VECTORS_ONLY);
        return bitwise_op_(xpr_[i]);
    }
    constexpr bitpack_t bitpack(int i) const { return bitpack_op_(xpr_.bitpack(i)); }
    constexpr int rows() const { return Rows != Dynamic ? Rows : xpr_.rows(); }
    constexpr int cols() const { return Cols != Dynamic ? Cols : xpr_.cols(); }
    constexpr int bitpacks() const { return xpr_.bitpacks(); }
   private:
    XprTypeNested xpr_;
    BitWiseOperation bitwise_op_;
    BitPackOperation bitpack_op_;
};

template <typename LhsXprType, typename RhsXprType, typename BitWiseOperation, typename BitPackOperation>
struct BoolMatrixBinOp :
    public BoolMatrixExpr<
      LhsXprType::Rows, LhsXprType::Cols, BoolMatrixBinOp<LhsXprType, RhsXprType, BitWiseOperation, BitPackOperation>> {
    fdapde_static_assert(
      internals::is_dynamic_sized_v<LhsXprType> || internals::is_dynamic_sized_v<RhsXprType> ||
        (LhsXprType::Rows == RhsXprType::Rows && LhsXprType::Cols == RhsXprType::Cols),
      YOU_MIXED_MATRICES_OF_DIFFERENT_STATIC_SIZE);
    fdapde_static_assert(
      std::is_same_v<typename LhsXprType::bitpack_t FDAPDE_COMMA typename RhsXprType::bitpack_t>,
      BOOLEAN_OPERATION_BETWEEN_MATRICES_OF_DIFFERENT_BITPACK_LAYOUT);
    using Base = BoolMatrixExpr<
      LhsXprType::Rows, LhsXprType::Cols, BoolMatrixBinOp<LhsXprType, RhsXprType, BitWiseOperation, BitPackOperation>>;
    using LhsXprTypeNested = internals::ref_select_t<const LhsXprType>;
    using RhsXprTypeNested = internals::ref_select_t<const RhsXprType>;
    using Scalar = typename Base::Scalar;
    using bitpack_t = typename Base::bitpack_t;
    static constexpr int Rows =
      (LhsXprType::Rows == Dynamic || RhsXprType::Rows == Dynamic) ? Dynamic : LhsXprType::Rows;
    static constexpr int Cols =
      (LhsXprType::Cols == Dynamic || RhsXprType::Cols == Dynamic) ? Dynamic : LhsXprType::Cols;
    static constexpr int NestAsRef = 0;
    static constexpr int ReadOnly = 1;

    template <typename LhsXprType_, typename RhsXprType_>
        requires(std::is_constructible_v<LhsXprTypeNested, LhsXprType_> &&
                 std::is_constructible_v<RhsXprTypeNested, RhsXprType_>)
    constexpr BoolMatrixBinOp(
      LhsXprType_&& lhs, RhsXprType_&& rhs, BitWiseOperation bitwise_op, BitPackOperation bitpack_op) :
        lhs_(std::forward<LhsXprType_>(lhs)),
        rhs_(std::forward<RhsXprType_>(rhs)),
        bitwise_op_(bitwise_op),
        bitpack_op_(bitpack_op) {
        if constexpr (internals::is_dynamic_sized_v<LhsXprType> || internals::is_dynamic_sized_v<RhsXprType>) {
            fdapde_assert(
              std::cmp_equal(lhs_.rows() FDAPDE_COMMA rhs_.rows()) &&
              std::cmp_equal(lhs_.cols() FDAPDE_COMMA rhs_.cols()));
        }
    }
    constexpr Scalar operator()(int i, int j) const { return bitwise_op_(lhs_(i, j), rhs_(i, j)); }
    constexpr Scalar operator[](int i) const {
        fdapde_static_assert(
          (LhsXprType::Cols == 1 && RhsXprType::Cols == 1) || (LhsXprType::Rows == 1 && RhsXprType::Rows == 1),
          THIS_METHOD_IS_FOR_ROW_OR_COLUMN_VECTORS_ONLY);
        return bitwise_op_(lhs_[i], rhs_[i]);
    }
    constexpr bitpack_t bitpack(int i) const { return bitpack_op_(lhs_.bitpack(i), rhs_.bitpack(i)); }
    constexpr int rows() const { return Rows != Dynamic ? Rows : lhs_.rows(); }
    constexpr int cols() const { return Cols != Dynamic ? Cols : lhs_.cols(); }
    constexpr int bitpacks() const { return lhs_.bitpacks(); }
   private:
    LhsXprTypeNested lhs_;
    RhsXprTypeNested rhs_;
    BitWiseOperation bitwise_op_;
    BitPackOperation bitpack_op_;
};
// boolean arithmetic
template <typename LhsXprType, typename RhsXprType>
constexpr auto operator&(
  const BoolMatrixExpr<LhsXprType::Rows, LhsXprType::Cols, LhsXprType>& lhs,
  const BoolMatrixExpr<RhsXprType::Rows, RhsXprType::Cols, RhsXprType>& rhs) {
    return BoolMatrixBinOp<LhsXprType, RhsXprType, std::bit_and<>, std::bit_and<>>(
      lhs.derived(), rhs.derived(), std::bit_and<>(), std::bit_and<>());
}
template <typename LhsXprType, typename RhsXprType>
constexpr auto operator|(
  const BoolMatrixExpr<LhsXprType::Rows, LhsXprType::Cols, LhsXprType>& lhs,
  const BoolMatrixExpr<RhsXprType::Rows, RhsXprType::Cols, RhsXprType>& rhs) {
    return BoolMatrixBinOp<LhsXprType, RhsXprType, std::bit_or<>, std::bit_or<>>(
      lhs.derived(), rhs.derived(), std::bit_or<>(), std::bit_or<>());
}
template <typename LhsXprType, typename RhsXprType>
constexpr auto operator^(
  const BoolMatrixExpr<LhsXprType::Rows, LhsXprType::Cols, LhsXprType>& lhs,
  const BoolMatrixExpr<RhsXprType::Rows, RhsXprType::Cols, RhsXprType>& rhs) {
    return BoolMatrixBinOp<LhsXprType, RhsXprType, std::bit_xor<>, std::bit_xor<>>(
      lhs.derived(), rhs.derived(), std::bit_xor<>(), std::bit_xor<>());
}

// dense-block of binary matrix
template <int BlockRows_, int BlockCols_, typename XprType>
class BoolMatrixBlock :
    public BoolMatrixExpr<BlockRows_, BlockCols_, BoolMatrixBlock<BlockRows_, BlockCols_, XprType>> {
    fdapde_static_assert(
      internals::is_dynamic_sized_v<XprType> ||
        ((BlockRows_ == Dynamic || (BlockRows_ > 0 && BlockRows_ <= XprType::Rows)) &&
         (BlockCols_ == Dynamic || (BlockCols_ > 0 && BlockCols_ <= XprType::Cols))),
      INVALID_STATIC_SIZED_BLOCK);
   public:
    using Base = BoolMatrixExpr<BlockRows_, BlockCols_, BoolMatrixBlock<BlockRows_, BlockCols_, XprType>>;
    using XprTypeNested = internals::ref_select_t<XprType>;
    using bitpack_t = typename Base::bitpack_t;
    using Scalar = typename Base::Scalar;
    static constexpr int PackSize = Base::PackSize;
    static constexpr int Rows = BlockRows_;
    static constexpr int Cols = BlockCols_;
    static constexpr int NestAsRef = 0;
    static constexpr int ReadOnly = XprType::ReadOnly;
    using assignment_executor = internals::generic_assignment_executor;   // bit-level assignment loop

    // row/column constructor
    template <typename XprType_>
        requires(std::is_constructible_v<XprTypeNested, XprType_>)
    constexpr BoolMatrixBlock(XprType_&& xpr, int i) :
        start_row_(BlockRows_ == 1 ? i : 0),
        start_col_(BlockCols_ == 1 ? i : 0),
        block_rows_(BlockRows_ == 1 ? 1 : xpr.rows()),
        block_cols_(BlockCols_ == 1 ? 1 : xpr.cols()),
        start_bitpack_((start_row_ * xpr.cols() + start_col_) / PackSize),
        end_bitpack_(1 + ((start_row_ + block_rows_) * xpr.cols() + start_col_ + block_cols_) / PackSize),
        xpr_(std::forward<XprType_>(xpr)) {
        fdapde_static_assert(BlockRows_ == 1 || BlockCols_ == 1, THIS_METHOD_IS_FOR_ROW_AND_COLUMN_BLOCKS_ONLY);
        fdapde_assert(
          i >= 0 && ((BlockRows_ == 1 && i < xpr_.rows()) || (BlockCols_ == 1 && i < xpr_.cols())));
    }
    // static-sized constructor
    template <typename XprType_>
        requires(std::is_constructible_v<XprTypeNested, XprType_>)
    constexpr BoolMatrixBlock(XprType_&& xpr, int start_row, int start_col) :
        start_row_(start_row),
        start_col_(start_col),
        block_rows_(BlockRows_),
        block_cols_(BlockCols_),
        start_bitpack_((start_row_ * xpr.cols() + start_col_) / PackSize),
        end_bitpack_(1 + ((start_row_ + block_rows_) * xpr.cols() + start_col_ + block_cols_) / PackSize),
        xpr_(std::forward<XprType_>(xpr)) {
        fdapde_static_assert(
          BlockRows_ != Dynamic && BlockCols_ != Dynamic, THIS_METHOD_IS_FOR_STATIC_SIZED_BLOCKS_ONLY);
        fdapde_assert(
          start_row >= 0 && start_row + block_rows_ <= xpr_.rows() && start_col >= 0 &&
          start_col + block_cols_ <= xpr.cols());
    }
    // dynamic-sized constructor
    template <typename XprType_>
        requires(std::is_constructible_v<XprTypeNested, XprType_>)
    constexpr BoolMatrixBlock(XprType_&& xpr, int start_row, int start_col, int block_rows, int block_cols) :
        start_row_(start_row),
        start_col_(start_col),
        block_rows_(block_rows),
        block_cols_(block_cols),
        start_bitpack_((start_row_ * xpr.cols() + start_col_) / PackSize),
        end_bitpack_(1 + ((start_row_ + block_rows_) * xpr.cols() + start_col_ + block_cols_) / PackSize),
        xpr_(std::forward<XprType_>(xpr)) {
        fdapde_static_assert(
          BlockRows_ == Dynamic && BlockCols_ == Dynamic, THIS_METHOD_IS_FOR_DYNAMIC_SIZED_BLOCKS_ONLY);
        fdapde_assert(
          start_row >= 0 && start_row + block_rows_ <= xpr_.rows() && start_col >= 0 &&
          start_col + block_cols_ <= xpr.cols());
    }
    // inherit assignment from base
    using Base::operator=;

    constexpr int rows() const noexcept { return BlockRows_ != Dynamic ? BlockRows_ : block_rows_; }
    constexpr int cols() const noexcept { return BlockCols_ != Dynamic ? BlockCols_ : block_cols_; }
    constexpr int size() const { return rows() * cols(); }
    constexpr int bitpacks() const { return (1 + fdapde::ceil(size() / PackSize)); }
    constexpr Scalar operator()(int i, int j) const {
        fdapde_assert(i >= 0 && i < block_rows_ && j >= 0 && j < block_cols_);
        return xpr_(i + start_row_, j + start_col_);
    }
    constexpr Scalar operator[](int i) const {
        fdapde_static_assert(BlockRows_ == 1 || BlockCols_ == 1, THIS_METHOD_IS_FOR_ROW_AND_COLUMN_BLOCKS_ONLY);
        if constexpr (Rows == 1) return xpr_(start_row_, start_col_ + i);
        if constexpr (Cols == 1) return xpr_(start_row_ + i, start_col_);
    }
    constexpr auto operator()(int i, int j) {
        fdapde_static_assert(XprType::ReadOnly == 0, ASSIGNMENT_TO_READ_ONLY_LOCATION);
        return xpr_(start_row_ + i, start_col_ + j);
    }
    constexpr auto operator[](int i) {
        fdapde_static_assert(BlockRows_ == 1 || BlockCols_ == 1, THIS_METHOD_IS_FOR_ROW_AND_COLUMN_BLOCKS_ONLY);
        fdapde_static_assert(XprType::ReadOnly == 0, ASSIGNMENT_TO_READ_ONLY_LOCATION);
        if constexpr (Rows == 1) return xpr_(start_row_, start_col_ + i);
        if constexpr (Cols == 1) return xpr_(start_row_ + i, start_col_);
    }

    // modifiers
    constexpr void set(int i, int j) noexcept {
        fdapde_assert(i >= 0 && i < block_rows_ && j >= 0 && j < block_cols_);
        xpr_.set(i + start_row_, j + start_col_);
    }
    constexpr void set() noexcept {
      for (int i = start_bitpack_; i < end_bitpack_; ++i) { xpr_.bitpack(i) |= bitpack_mask_(i); }
    }
    constexpr void clear(int i, int j) noexcept {
        fdapde_assert(i >= 0 && i < block_rows_ && j >= 0 && j < block_cols_);
        xpr_.clear(i + start_row_, j + start_col_);
    }
    constexpr void clear() noexcept {
        for (int i = start_bitpack_; i < end_bitpack_; ++i) { xpr_.bitpack(i) &= ~bitpack_mask_(i); }
    }
    constexpr bitpack_t bitpack(int i) const {
        fdapde_assert(i >= 0 && i < bitpacks());
        bitpack_t out = bitpack_t(0);
        // precompute block bit layout
        const int base_bit = i * PackSize;
        const int col_offset = start_col_ + ((base_bit) % cols());
        const int row_offset = start_row_ + ((base_bit) / cols());
	const int max_col = start_col_ + block_cols_;
        // assembly block bitpack
        constexpr bitpack_t mask = bitpack_t(1);
        int row = row_offset, col = col_offset;
        for (int j = 0, size_ = size(); j < PackSize && base_bit + j < size_; ++j) {
            out |= (mask & xpr_(row, col)) << j;
            if (++col == max_col) {
                col = start_col_;
                row++;
            }
        }
        return out;
    }
   private:
    // generates a mask for active block bits in i-th global bitpack
    constexpr bitpack_t bitpack_mask_(int i) const {
        fdapde_assert(i >= 0 && i < xpr_.bitpacks());
        bitpack_t out = bitpack_t(0);
        // precompute block bit layout
        const int max_col = start_col_ + block_cols_;
	const int max_row = start_row_ + block_rows_;
        int col = (i * PackSize % xpr_.cols());
        int row = (i * PackSize / xpr_.cols());
	// flag as 1 all block bits intersected by this bitpack
        constexpr bitpack_t mask = bitpack_t(1);
        for (int j = 0; j < PackSize; ++j) {
            out |= (mask & (row >= start_row_ && row < max_row && col >= start_col_ && col < max_col)) << j;
            if (++col == xpr_.cols()) {
                col = 0;
                row++;
            }
        }
        return out;
    }

    int start_row_ = 0, start_col_ = 0;
    int block_rows_ = 0, block_cols_ = 0;
    int start_bitpack_ = 0, end_bitpack_ = 0;   // index of first and last intersected bitpack
    XprTypeNested xpr_;
};

// boolean redux operation
namespace internals {

// bitpack reduction loop on matrix expressions
template <typename XprType, typename Functor> struct matrix_bitpack_redux_executor {
    using XprTypeClean = std::decay_t<XprType>;
    static constexpr int PackSize = XprTypeClean::PackSize;

    template <typename Scalar> static constexpr Scalar run(const XprType& xpr, Scalar init, Functor visitor) {
        fdapde_assert(xpr.size() > 0);
        const int size = xpr.size();
        Scalar res = init;

        if (size < PackSize) {
            res = visitor(res, xpr.bitpack(0), size);   // size: number of valid bits in bitpack
            return res;
        }
        int k = 0, i = 0;   // k: current bitpack, i: last bit processed
        for (; i + PackSize <= size; ++k, i += PackSize) {
            res = visitor(res, xpr.bitpack(k));
            if (visitor.early_exit()) { return res; }
        }
        // process last, partially filled, bitpack
        if (i < size) { res = visitor(res, xpr.bitpack(k), size - i); }
        return res;
    }
};

// evaluates true if all the coefficients of XprType are true
template <typename XprType> struct all_redux_executor {
    using XprTypeClean = std::decay_t<XprType>;
    using bitpack_t = typename XprTypeClean::bitpack_t;
    static constexpr int PackSize = XprTypeClean::PackSize;

    constexpr all_redux_executor() noexcept : all_(true) { }

    constexpr bool operator()([[maybe_unused]] bool b, bitpack_t p) noexcept {
        all_ &= (p == std::numeric_limits<bitpack_t>::max());
        return all_;
    }
    constexpr bool operator()([[maybe_unused]] bool b, bitpack_t p, int size) noexcept {
        const bitpack_t mask = (bitpack_t(1) << size) - 1;
        all_ &= ((p & mask) == mask);
        return all_;
    }
    constexpr bool early_exit() const noexcept { return !all_; }
   private:
    bool all_;
};

// evaluates true if at least one coefficient of XprType is true
template <typename XprType> struct any_redux_executor {
    using XprTypeClean = std::decay_t<XprType>;
    using bitpack_t = typename XprTypeClean::bitpack_t;
    static constexpr int PackSize = XprTypeClean::PackSize;

    constexpr any_redux_executor() noexcept : any_(false) { }

    constexpr bool operator()([[maybe_unused]] bool b, bitpack_t p) noexcept {
        any_ |= (p != bitpack_t(0));
        return any_;
    }
    constexpr bool operator()([[maybe_unused]] bool b, bitpack_t p, int size) noexcept {
        const bitpack_t mask = ((bitpack_t(1) << size) - 1);
        any_ |= ((p & mask) != bitpack_t(0));
        return any_;
    }
    constexpr bool early_exit() const noexcept { return any_; }
   private:
    bool any_;
};
 
// returns the number of true coefficients in XprType, based on popcnt machine istruction for fast scalar counting
template <typename XprType> struct cnt_redux_executor {
    using XprTypeClean = std::decay_t<XprType>;
    using bitpack_t = typename XprTypeClean::bitpack_t;
    static constexpr int PackSize = XprTypeClean::PackSize;

    constexpr cnt_redux_executor() noexcept = default;

    constexpr int operator()(int cnt, bitpack_t p) const noexcept { return cnt + std::popcount(p); }
    constexpr int operator()(int cnt, bitpack_t p, int size) const noexcept {
        const bitpack_t mask = ((bitpack_t(1) << size) - 1);
        return cnt + std::popcount(p & mask);
    }
    constexpr bool early_exit() const noexcept { return false; }   // never stop
}; 
  
}   // namespace internals

// base class for boolean expressions
template <int Rows_, int Cols_, typename XprType_> struct BoolMatrixExpr {
    using XprType = XprType_;
    using bitpack_t = std::uintmax_t;
    using Scalar = bool;
    static constexpr int Rows = Rows_;
    static constexpr int Cols = Cols_;
    static constexpr std::size_t PackSize = sizeof(bitpack_t) * 8;   // number of bits in a packet

      // assignment
    template <int RhsRows_, int RhsCols_, typename RhsXprType_>
    constexpr XprType& operator=(const BoolMatrixExpr<RhsRows_, RhsCols_, RhsXprType_>& rhs) {
        using executor = typename XprType::assignment_executor;
        if constexpr (Rows_ == Dynamic || Cols_ == Dynamic) {   // resize to rhs size, if lhs is Dynamic
            if (derived().rows() != rhs.rows() || derived().cols() != rhs.cols()) {
                derived().resize(rhs.rows(), rhs.cols());
            }
        }
        executor::run(derived(), rhs.derived(), [](auto&& l, const auto& r) { l = r; });
        return derived();
    }
    // compound boolean algebra
    template <int RhsXprRows_, int RhsXprCols_, typename RhsXprType_>
    constexpr XprType& operator&=(const BoolMatrixExpr<RhsXprRows_, RhsXprCols_, RhsXprType_>& other) {
	using executor = typename XprType::assignment_executor;
        executor::run(derived(), other.derived(), [](auto&& l, const auto& r) { l &= r; });
        return derived();
    }
    template <int RhsXprRows_, int RhsXprCols_, typename RhsXprType_>
    constexpr XprType& operator|=(const BoolMatrixExpr<RhsXprRows_, RhsXprCols_, RhsXprType_>& other) {
	using executor = typename XprType::assignment_executor;
        executor::run(derived(), other.derived(), [](auto&& l, const auto& r) { l |= r; });
        return derived();
    }
    template <int RhsXprRows_, int RhsXprCols_, typename RhsXprType_>
    constexpr XprType& operator^=(const BoolMatrixExpr<RhsXprRows_, RhsXprCols_, RhsXprType_>& other) {
	using executor = typename XprType::assignment_executor;
        executor::run(derived(), other.derived(), [](auto&& l, const auto& r) { l ^= r; });
        return derived();
    }
    // observers
    constexpr int rows() const { return derived().rows(); }
    constexpr int cols() const { return derived().cols(); }
    constexpr int size() const {
        return (Rows != Dynamic && Cols != Dynamic) ? Rows * Cols : derived().rows() * derived().cols();
    }
    constexpr const XprType& derived() const { return static_cast<const XprType&>(*this); }
    constexpr XprType& derived() { return static_cast<XprType&>(*this); }
    // ostream
    friend std::ostream& operator<<(std::ostream& out, const BoolMatrixExpr& m) {
        const int rows = m.derived().rows();
        const int cols = m.derived().cols();
        const auto& d = m.derived();
        for (int i = 0; i < rows - 1; ++i) {
            for (int j = 0; j < cols; ++j) { out << d(i, j) << " "; }
            out << "\n";
        }
        // print last row without carriage return
        for (int j = 0; j < cols; ++j) { out << d(rows - 1, j) << " "; }
        return out;
    }

    // returns all the indices (in row-major order) having coefficients equal to b
    std::vector<int> which(bool b) const {
        std::vector<int> result;
        const int bitpacks_ = derived().bitpacks();
        const int size_ = derived().size();
        int j = 0;
        for (int i = 0; i < bitpacks_; ++i) {
            bitpack_t pack = derived().bitpack(i);
            for (int bit = 0; bit < PackSize && j < size_; ++bit, ++j) {
                if ((pack & 0x1) == static_cast<bitpack_t>(b)) result.push_back(j);
                pack >>= 1;
            }
        }
        return result;
    }
    // unary bitwise negation
    constexpr BoolMatrixBitWiseOp<XprType, std::logical_not<>, std::bit_not<>> operator~() const {
        return BoolMatrixBitWiseOp<XprType, std::logical_not<>, std::bit_not<>>(
          derived(), std::logical_not<>(), std::bit_not<>());
    }
    // block accessors
    // static-sized block
    template <int BlockRows, int BlockCols>
    constexpr BoolMatrixBlock<BlockRows, BlockCols, XprType> block(int i, int j) {
        return BoolMatrixBlock<BlockRows, BlockCols, XprType>(derived(), i, j);
    }
    template <int BlockRows, int BlockCols>
    constexpr BoolMatrixBlock<BlockRows, BlockCols, const XprType> block(int i, int j) const {
        return BoolMatrixBlock<BlockRows, BlockCols, const XprType>(derived(), i, j);
    }
    // dynamic-sized block
    constexpr BoolMatrixBlock<Dynamic, Dynamic, XprType> block(int i, int j, int rows, int cols) {
        return BoolMatrixBlock<Dynamic, Dynamic, XprType>(derived(), i, j, rows, cols);
    }
    constexpr BoolMatrixBlock<Dynamic, Dynamic, const XprType> block(int i, int j, int rows, int cols) const {
        return BoolMatrixBlock<Dynamic, Dynamic, const XprType>(derived(), i, j, rows, cols);
    }
    // row/col accessors
    constexpr BoolMatrixBlock<Rows, 1, XprType> col(int i) { return BoolMatrixBlock<Rows, 1, XprType>(derived(), i); }
    constexpr BoolMatrixBlock<Rows, 1, const XprType> col(int i) const {
        return BoolMatrixBlock<Rows, 1, const XprType>(derived(), i);
    }
    constexpr BoolMatrixBlock<1, Cols, XprType> row(int i) { return BoolMatrixBlock<1, Cols, XprType>(derived(), i); }
    constexpr BoolMatrixBlock<1, Cols, const XprType> row(int i) const {
        return BoolMatrixBlock<1, Cols, const XprType>(derived(), i);
    }
    // other block-type accessors
    template <int BlockRows> constexpr auto top_rows() { return block<BlockRows, Cols>(0, 0); }
    template <int BlockRows> constexpr auto top_rows() const { return block<BlockRows, Cols>(0, 0); }
    constexpr auto top_rows(int rows) { return block(0, 0, rows, derived().cols()); }
    constexpr auto top_rows(int rows) const { return block(0, 0, rows, derived().cols()); }

    template <int BlockRows> constexpr auto bottom_rows() { return block<BlockRows, Cols>(Rows - BlockRows, 0); }
    template <int BlockRows> constexpr auto bottom_rows() const { return block<BlockRows, Cols>(Rows - BlockRows, 0); }
    constexpr auto bottom_rows(int rows) { return block(derived().rows() - rows, 0, rows, derived().cols()); }
    constexpr auto bottom_rows(int rows) const { return block(derived().rows() - rows, 0, rows, derived().cols()); }

    template <int BlockCols> constexpr auto left_cols() { return block<Rows, BlockCols>(0, 0); }
    template <int BlockCols> constexpr auto left_cols() const { return block<Rows, BlockCols>(0, 0); }
    constexpr auto left_cols(int cols) { return block(0, 0, derived().rows(), cols); }
    constexpr auto left_cols(int cols) const { return block(0, 0, derived().rows(), cols); }

    template <int BlockCols> constexpr auto right_cols() { return block<Rows, BlockCols>(0, Cols - BlockCols); }
    template <int BlockCols> constexpr auto right_cols() const { return block<Rows, BlockCols>(0, Cols - BlockCols); }
    constexpr auto right_cols(int cols) { return block(0, derived().cols() - cols, derived().rows(), cols); }
    constexpr auto right_cols(int cols) const { return block(0, derived().cols() - cols, derived().rows(), cols); }

    // visitors support
    constexpr bool all() const {
        return internals::matrix_bitpack_redux_executor<XprType, internals::all_redux_executor<XprType>>().run(
          derived(), true, internals::all_redux_executor<XprType>());
    }
    constexpr bool any() const {
        return internals::matrix_bitpack_redux_executor<XprType, internals::any_redux_executor<XprType>>().run(
          derived(), true, internals::any_redux_executor<XprType>());
    }
    constexpr int count() const {
        return internals::matrix_bitpack_redux_executor<XprType, internals::cnt_redux_executor<XprType>>().run(
          derived(), 0, internals::cnt_redux_executor<XprType>());
    }

    // make generic select, not bounded to eigen types (maybe with lazy evaluation)

    //     // block-repeat operation
    //     BinMtxRepeatOp<Dynamic, Dynamic, XprType> repeat(int rep_row, int rep_col) const {
    //         return BinMtxRepeatOp<Dynamic, Dynamic, XprType>(get(), rep_row, rep_col);
    //     }
    //     // reshape a binary matrix to another matrix of different sizes
    //     BinMtxReshapeOp<Dynamic, Dynamic, XprType> reshape(int n_row, int n_col) const {
    //         return BinMtxReshapeOp<Dynamic, Dynamic, XprType>(get(), n_row, n_col);
    //     }
    //     BinMtxReshapeOp<Dynamic, Dynamic, XprType> vector_view() const { return reshape(get().size(), 1); }
    //    private:
    //     template <typename Visitor, template <typename, typename> typename VisitStrategy> inline auto visit_apply_()
    //     const {
    //         Visitor visitor;
    //         VisitStrategy<XprType, Visitor>::run(get(), visitor);
    //         return visitor.res;
    //     }
};
  
// a non-writable expression of a block-repeat operation
// template <int Rows, int Cols, typename XprTypeNested>
// class BinMtxRepeatOp : public BinMtxBase<Rows, Cols, BinMtxRepeatOp<Rows, Cols, XprTypeNested>> {
//    public:
//     using XprType = BinMtxRepeatOp<Rows, Cols, XprTypeNested>;
//     using Base = BinMtxBase<Rows, Cols, XprType>;
//     using BitPackType = typename Base::BitPackType;
//     static constexpr int PackSize = Base::PackSize;   // number of bits in a packet
//     static constexpr int NestAsRef = 0;   // whether to store this node by reference or by copy in an expression
//     using Base::cols_;
//     using Base::rows_;

//     BinMtxRepeatOp(const XprTypeNested& xpr, int rep_row, int rep_col) :
//         Base(xpr.rows() * rep_row, xpr.cols() * rep_col), xpr_(xpr), rep_row_(rep_row), rep_col_(rep_col) {
//     }
//     bool operator()(int i, int j) const {
//         fdapde_assert(i < rows_ && j < cols_);
//         return xpr_(i % xpr_.rows(), j % xpr_.cols());
//     }
//     BitPackType bitpack(int i) const {
//         BitPackType out = 0x0;
//         for (int j = 0; j < PackSize && i * PackSize + j < Base::size(); ++j) {
//             out |=
//               ((BitPackType)1 & xpr_(((i * PackSize + j) / cols_) % xpr_.rows(), (i * PackSize + j) % xpr_.cols()))
//               << j;
//         }
//         return out;
//     }
//    private:
//     // internal data
//     typename internals::ref_select<const XprTypeNested>::type xpr_;
//     int rep_row_, rep_col_;
// };

// reshape operation
// template <int Rows, int Cols, typename XprTypeNested>
// class BinMtxReshapeOp : public BinMtxBase<Rows, Cols, BinMtxReshapeOp<Rows, Cols, XprTypeNested>> {
// public:
//     using XprType = BinMtxReshapeOp<Rows, Cols, XprTypeNested>;
//     using Base = BinMtxBase<Rows, Cols, XprType>;
//     using BitPackType = typename Base::BitPackType;
//     static constexpr int PackSize = Base::PackSize;   // number of bits in a packet
//     static constexpr int NestAsRef = 0;   // whether to store this node by reference or by copy in an expression
  
//     BinMtxReshapeOp(const XprTypeNested& xpr, int reshaped_rows, int reshaped_cols) :
//         Base(reshaped_rows, reshaped_cols), xpr_(xpr), reshaped_rows_(reshaped_rows), reshaped_cols_(reshaped_cols) {
//         fdapde_assert(reshaped_rows * reshaped_cols == xpr.rows() * xpr.cols());
//     }
//     bool operator()(int i, int j) const {
//         fdapde_assert(i < reshaped_rows_ && j < reshaped_cols_);
//         return xpr_((i * reshaped_cols_ + j) / xpr_.cols(), (i * reshaped_cols_ + j) % xpr_.cols());
//     }
//     BitPackType bitpack(int i) const { return xpr_.bitpack(i); }   // no changes in storage layout
//    private:
//     // internal data
//     typename internals::ref_select<const XprTypeNested>::type xpr_;
//     int reshaped_rows_, reshaped_cols_;
// };
  
// comparison operator
template <int Rows1, int Cols1, typename XprType1, int Rows2, int Cols2, typename XprType2>
constexpr bool
operator==(const BoolMatrixExpr<Rows1, Cols1, XprType1>& op1, const BoolMatrixExpr<Rows2, Cols2, XprType2>& op2) {
    fdapde_static_assert(
      (internals::is_dynamic_sized_v<XprType1> || internals::is_dynamic_sized_v<XprType2> ||
       (Rows1 == Rows2 && Cols1 == Cols2)),
      INVALID_COMPARISON__OPERANDS_HAVE_DIFFERENT_SIZES);
    if constexpr (internals::is_dynamic_sized_v<XprType1> || internals::is_dynamic_sized_v<XprType2>) {
        fdapde_assert(op1.rows() == op2.rows() && op1.cols() == op2.cols());
    }
    using bitpack_t = typename XprType1::bitpack_t;
    constexpr int pack_size = sizeof(bitpack_t) * 8;

    const auto& d1 = op1.derived();
    const auto& d2 = op2.derived();
    bool result = true;
    int n = d1.bitpacks() - 1;
    // fast first n bitpacks comparison
    for (int i = 0; i < n && result; i++) { result &= (d1.bitpack(i) == d2.bitpack(i)); }
    // process last bitpack
    bitpack_t mask = ~(bitpack_t)0 >> (pack_size - (d1.size() - pack_size * n));
    result &= ((mask & d1.bitpack(n)) == (mask & d2.bitpack(n)));
    return result;
}
template <int Rows1, int Cols1, typename XprType1, int Rows2, int Cols2, typename XprType2>
constexpr bool
operator!=(const BoolMatrixExpr<Rows1, Cols1, XprType1>& op1, const BoolMatrixExpr<Rows2, Cols2, XprType2>& op2) {
    return !(op1 == op2);
}

// // out-of-class which function
// template <int Rows, int Cols, typename XprType> std::vector<int> which(const BinMtxBase<Rows, Cols, XprType>& mtx) {
//     return mtx.which(true);
// }

// // move the iterator first-last to a binary vector v such that v[i] = true \iff *(first + i) == c
// template <typename Iterator>
// BinaryVector<Dynamic> make_binary_vector(const Iterator& first, const Iterator& last, typename Iterator::value_type c) {
//     int n_rows = std::distance(first, last);
//     BinaryVector<Dynamic> vec(n_rows);
//     for (int i = 0; i < n_rows; ++i) {
//         if (*(first + i) == c) vec.set(i);
//     }
//     return vec;
// }

// template <typename Data>
//     requires(internals::is_vector_like_v<Data> || internals::is_matrix_like_v<Data>)
// auto na_matrix(const Data& data) {
//     using storage_t =
//       std::conditional_t<internals::is_vector_like_v<Data>, BinaryVector<Dynamic>, BinaryMatrix<Dynamic, Dynamic>>;
//     storage_t na_mask;
//     if constexpr (internals::is_vector_like_v<Data>) {
//         na_mask.resize(data.size());
//         for (int i = 0; i < data.size(); ++i) {
//             if (std::isnan(internals::vector_like_access(data, i))) { na_mask.set(i); }
//         }
//     } else {
//         na_mask.resize(data.rows(), data.cols());
//         for (int i = 0; i < data.rows(); ++i) {
//             for (int j = 0; j < data.cols(); ++j) {
//                 if (std::isnan(data(i, j))) { na_mask.set(i, j); }
//             }
//         }
//     }
//     return na_mask;
// }

// // map a memory region to a BinaryMatrix
// template <int Rows, int Cols, typename XprTypeNested>
// class BinaryMap : public BinMtxBase<Rows, Cols, BinaryMap<Rows, Cols, XprTypeNested>> {
//    public:
//     using XprType = BinaryMap<Rows, Cols, XprTypeNested>;
//     using Base = BinMtxBase<Rows, Cols, XprType>;
//     using BitPackType = std::decay_t<XprTypeNested>;
//     static constexpr int PackSize = sizeof(XprTypeNested) * 8;   // number of bits in a packet
//     static constexpr int NestAsRef = 0;   // whether to store this node by reference or by copy in an expression
//     using Base::cols_;
//     using Base::rows_;

//     BinaryMap(XprTypeNested* data)
//         requires(Rows != Dynamic && Cols != Dynamic)
//       : Base(), data_(data) {
//         fdapde_static_assert(std::is_integral_v<XprTypeNested>, ONLY_INTEGRAL_TYPES_CAN_BE_BINARY_MAPPED);
//     }
//     BinaryMap(XprTypeNested* data, int row) : Base(row), data_(data) {
//         fdapde_static_assert(std::is_integral_v<XprTypeNested>, ONLY_INTEGRAL_TYPES_CAN_BE_BINARY_MAPPED);
//         fdapde_static_assert(Cols == 1 || Rows == 1, THIS_METHOD_IS_ONLY_FOR_VECTORS);
//     }
//     BinaryMap(XprTypeNested* data, int row, int col) : Base(row, col), data_(data) {
//         fdapde_static_assert(std::is_integral_v<XprTypeNested>, ONLY_INTEGRAL_TYPES_CAN_BE_BINARY_MAPPED);
//     }
//     // const access
//     bool operator()(int i, int j) const {
//         return (data_[pack_of(i, j)] & BitPackType(1) << ((i * cols_ + j) % PackSize)) != 0;
//     }
//     bool operator[](int i) const {   // vector-like (subscript) access
//         fdapde_static_assert(Cols == 1 || Rows == 1, THIS_METHOD_IS_ONLY_FOR_VECTORS);
//         return operator()(i, 0);
//     }
//     BitPackType bitpack(int i) const { return data_[i]; }
//     BitPackType& bitpack(int i) { return data_[i]; }   // non-const access to i-th bitpack

//     void set(int i, int j) {   // set (i,j)-th bit
//         fdapde_assert(i < rows_ && j < cols_);
//         data_[pack_of(i, j)] |= (BitPackType(1) << ((i * cols_ + j) % PackSize));
//     }
//     void set(int i) {
//         fdapde_static_assert(Cols == 1 || Rows == 1, THIS_METHOD_IS_ONLY_FOR_VECTORS);
//         set(i, 0);
//     }
//     void set() {   // sets all coeffients in the matrix
//         for (int i = 0; i < rows_; ++i) {
//             for (int j = 0; j < cols_; ++j) {
//                 data_[pack_of(i, j)] |= (BitPackType(1) << ((i * cols_ + j) % PackSize));
//             }
//         }
//     }  
//     void clear(int i, int j) {   // clear (i,j)-th bit (sets to 0)
//         fdapde_assert(i < rows_ && j < cols_);
//         data_[pack_of(i, j)] &= ~(BitPackType(1) << ((i * cols_ + j) % PackSize));
//     }
//     void clear(int i) {
//         fdapde_static_assert(Cols == 1 || Rows == 1, THIS_METHOD_IS_ONLY_FOR_VECTORS);
//         clear(i, 0);
//     }
//     void clear() {   // clears all coeffients in the matrix
//         for (int i = 0; i < rows_; ++i) {
//             for (int j = 0; j < cols_; ++j) {
//                 data_[pack_of(i, j)] &= ~(BitPackType(1) << ((i * cols_ + j) % PackSize));
//             }
//         }
//     }
//    private:
//     XprTypeNested* data_;
//     // recover the byte-pack for the (i,j)-th element
//     inline int pack_of(int i, int j) const { return (i * cols_ + j) / PackSize; }
// };

  
}   // namespace fdapde

#endif   // __FDAPDE_LINALG_BOOL_H__
