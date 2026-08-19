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

namespace fdapde {

// linalg API specialization for the case Scalar = bool. Due to the boolean algebra semantic and the packed
// storage layout, boolean linalg API is different than that of the rest of the linalg module

template <typename XprType_> struct BoolMatrixExpr;

namespace internals {

template <typename T>
concept bool_matrix_expression =
  std::derived_from<std::remove_cvref_t<T>, BoolMatrixExpr<std::remove_cvref_t<T>>>;

constexpr int bitpack_count(int size, int pack_size) {
    return size <= 0 ? 0 : 1 + (size - 1) / pack_size;
}

template <typename BitPack> constexpr BitPack low_bits_mask(int used_bits) {
    constexpr int pack_size = sizeof(BitPack) * 8;
    return used_bits == pack_size ? ~BitPack(0) : (BitPack(1) << used_bits) - 1;
}

struct bitpack_assignment_executor {
    template <typename DstMatrixType, typename SrcXprType, typename AssignmentOp>
        requires(
          requires(AssignmentOp op, typename DstMatrixType::bitpack_t& l, const typename SrcXprType::bitpack_t& r) {
              { op(l, r) } -> std::same_as<void>;
          })
    static constexpr void run(DstMatrixType& dst, const SrcXprType& src, AssignmentOp&& op) {
        fdapde_static_assert(DstMatrixType::ReadOnly == 0, ASSIGNMENT_TO_READ_ONLY_EXPRESSION);
        fdapde_static_assert(
          internals::same_static_shape_weak_v<DstMatrixType FDAPDE_COMMA SrcXprType>,
          INVALID_ASSIGNMENT__NOT_MATCHING_LHS_AND_RHS_STATIC_SIZES);
        if constexpr (internals::is_dynamic_sized_v<DstMatrixType> || internals::is_dynamic_sized_v<SrcXprType>) {
            if (dst.rows() != src.rows() || dst.cols() != src.cols()) {
                throw std::invalid_argument("Boolean matrix dimensions do not match");
            }
        }
        if (dst.size() == 0) return;
        auto& d = dst.derived();
        const auto& s = src.derived();
        // fast bitpack assignment
        for (int i = 0, bitpacks_ = d.bitpacks(); i < bitpacks_; ++i) { op(d.bitpack(i), s.bitpack(i)); }
        return;
    }
};

}   // namespace internals

template <int Rows_, int Cols_, int StorageOrder_, typename BoolMatrixType_>
class MatrixBase<bool, Rows_, Cols_, StorageOrder_, BoolMatrixType_> : public BoolMatrixExpr<BoolMatrixType_> {
    fdapde_static_assert(
      (Rows_ == Dynamic || Rows_ > 0) && (Cols_ == Dynamic || Cols_ > 0), ZERO_STATICALLY_SIZED_MATRICES_ARE_INVALID);
    fdapde_static_assert(
      Rows_ == Dynamic || Cols_ == Dynamic ||
        static_cast<std::uint64_t>(Rows_) * static_cast<std::uint64_t>(Cols_) <=
          static_cast<std::uint64_t>(std::numeric_limits<int>::max()),
      MATRIX_SIZE_EXCEEDS_SUPPORTED_RANGE);
   private:
    using BoolMatrixType = std::decay_t<BoolMatrixType_>;
    using Base = BoolMatrixExpr<BoolMatrixType>;
   public:
    using Scalar = typename Base::Scalar;
    using bitpack_t = typename Base::bitpack_t;
    static constexpr int Rows = Rows_;
    static constexpr int Cols = Cols_;
    static constexpr int StorageOrder = StorageOrder_;
    static constexpr int NestAsRef = BoolMatrixType::NestAsRef;
    static constexpr int ReadOnly = std::is_const_v<Scalar>;
    static constexpr std::size_t PackSize = Base::PackSize;
    using assignment_executor = internals::bitpack_assignment_executor;
    using Base::derived;
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
            requires(std::is_convertible_v<BitPackT_* FDAPDE_COMMA BitPackT*>)
        constexpr bit_proxy(const bit_proxy<BitPackT_>& other) :
            data_(other.data_), pack_id_(other.pack_id_), bitmask_(other.bitmask_) { }
        constexpr bit_proxy& operator=(const bit_proxy& other) requires(!std::is_const_v<BitPackT>) {
            return operator=(bool(other));
        }
        template <typename BitPackT_>
            requires(!std::is_const_v<BitPackT>)
        constexpr bit_proxy& operator=(const bit_proxy<BitPackT_>& other) {
            return operator=(bool(other));
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
        constexpr void set() requires(!std::is_const_v<BitPackT>) { data_[pack_id_] |= bitmask_; }
        constexpr void clear() requires(!std::is_const_v<BitPackT>) { data_[pack_id_] &= ~bitmask_; }
        template <typename T>
            requires(!std::is_const_v<BitPackT> && std::is_convertible_v<T, bool>)
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
        col_stride_(StorageOrder == RowMajor ? 1 : rows_) { }
    constexpr MatrixBase(int rows, int cols) :
        rows_(Rows == Dynamic ? rows : Rows),
        cols_(Cols == Dynamic ? cols : Cols),
        row_stride_(StorageOrder == RowMajor ? cols_ : 1),
        col_stride_(StorageOrder == RowMajor ? 1 : rows_) { }
    constexpr MatrixBase(int size) :
        rows_(Rows == 1 ? 1 : size),
        cols_(Cols == 1 ? 1 : size),
        row_stride_(StorageOrder == RowMajor ? cols_ : 1),
        col_stride_(StorageOrder == RowMajor ? 1 : rows_) {
        fdapde_static_assert(Rows == 1 || Cols == 1, THIS_METHOD_IS_FOR_ROW_OR_COLUMN_VECTORS_ONLY);
    }
    // copy assignment
    constexpr BoolMatrixType& operator=(const BoolMatrixType& other) & {
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
        fdapde_assert(i >= 0 && i < rows_ && j >= 0 && j < cols_);
        return const_reference(derived().data(), i, j, row_stride_, col_stride_);
    }
    constexpr const_reference operator[](int i) const {
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
   private:
    using This = Matrix<bool, Rows_, Cols_, StorageOrder_>;
    using Base = MatrixBase<bool, Rows_, Cols_, StorageOrder_, This>;
    static constexpr int checked_size_(int rows, int cols) {
        internals::validate_matrix_shape<Rows_, Cols_>(rows, cols);
        return internals::checked_matrix_size(Rows_ == Dynamic ? rows : Rows_, Cols_ == Dynamic ? cols : Cols_);
    }
    static constexpr int checked_vector_size_(int size) {
        internals::validate_matrix_vector_size<Rows_, Cols_>(size);
        return internals::checked_matrix_size(Rows_ == Dynamic ? size : Rows_, Cols_ == Dynamic ? size : Cols_);
    }
   public:
    using Scalar = typename Base::Scalar;
    using bitpack_t = typename Base::bitpack_t;
    static constexpr int Rows = Rows_;
    static constexpr int Cols = Cols_;
    static constexpr int NestAsRef = 1;
    static constexpr int ReadOnly = std::is_const_v<Scalar>;
    static constexpr std::size_t PackSize = Base::PackSize;
    static constexpr int StorageSize = [] {
        if constexpr (Rows == Dynamic || Cols == Dynamic) {
            return Dynamic;
        } else {
            constexpr std::uint64_t size =
              static_cast<std::uint64_t>(Rows) * static_cast<std::uint64_t>(Cols);
            if constexpr (size > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
                return 0;
            } else {
                return internals::bitpack_count(static_cast<int>(size), static_cast<int>(PackSize));
            }
        }
    }();
    using StorageType = std::conditional_t<
      Rows == Dynamic || Cols == Dynamic, std::vector<bitpack_t>,
      std::array<bitpack_t, static_cast<std::size_t>(StorageSize)>>;   // avoid clang narrowing
    using iterator = typename StorageType::iterator;
    using const_iterator = typename StorageType::const_iterator;
    using assignment_executor = typename Base::assignment_executor;
  
    constexpr Matrix() noexcept :
        Base(), data_(), bitpacks_(StorageSize == Dynamic ? 0 : StorageSize) { }
    constexpr Matrix(const Matrix& other) :
        Base(), data_(), bitpacks_(StorageSize == Dynamic ? 0 : StorageSize) {
        if constexpr (Rows_ == Dynamic || Cols_ == Dynamic) { resize(other.rows(), other.cols()); }
        assignment_executor::run(*this, other, [](bitpack_t& l, const bitpack_t& r) { l = r; });
    }
    constexpr Matrix& operator=(const Matrix& other) & {
        Base::operator=(other);
        return *this;
    }
    constexpr void operator=(const Matrix&) && = delete;
    template <typename RhsXprType_>   // construct from plain BoolMatrixExpr
    constexpr Matrix(const BoolMatrixExpr<RhsXprType_>& rhs) :
        Base(), data_(), bitpacks_(StorageSize == Dynamic ? 0 : StorageSize) {
        if constexpr (Rows_ == Dynamic || Cols_ == Dynamic) { resize(rhs.rows(), rhs.cols()); }
        internals::generic_assignment_executor::run(
          *this, rhs.derived(), [](auto&& l, const auto& r) { l = bool(r); });
    }
    template <typename RhsXprType_>   // cast MatrixExpr to bool
    constexpr Matrix(const MatrixExpr<RhsXprType_>& rhs) :
        Base(), data_(), bitpacks_(StorageSize == Dynamic ? 0 : StorageSize) {
        fdapde_static_assert(
          internals::same_static_shape_weak_v<This FDAPDE_COMMA RhsXprType_>,
          INVALID_ASSIGNMENT__NOT_MATCHING_LHS_AND_RHS_STATIC_SIZES);
        if constexpr (internals::is_dynamic_sized_v<This>) { resize(rhs.rows(), rhs.cols()); }
        if (this->rows() != rhs.rows() || this->cols() != rhs.cols()) {
            throw std::invalid_argument("matrix dimensions do not match its static shape");
        }
        const int rows = this->rows();
        const int cols = this->cols();
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) { this->operator()(i, j) = rhs.derived()(i, j); }
        }
    }
    template <typename RhsXprType_>   // materialize a coefficient-wise comparison
    constexpr Matrix(const MatrixCoeffWiseExpr<RhsXprType_>& rhs) : Matrix(rhs.mwise()) { }
    // inherit assignment from base
    using Base::operator=;

    // Matrix API
    // value-initialized static-sized matrix
    constexpr explicit Matrix(bool v)
        requires(Rows_ != Dynamic && Cols_ != Dynamic)
        : data_(), bitpacks_(StorageSize == Dynamic ? 0 : StorageSize) {
        bitpack_t v_ = v ? -1 : 0;
        for (int i = 0; i < bitpacks_; ++i) { data_[i] = v_; }
    }
    // false-initialized dynamic-sized matrix. For static-sized matrices does nothing (exposed for API compatibility)
    constexpr Matrix(int rows, int cols)
        requires(Rows_ != 1 && Cols_ != 1)
        : Base(rows, cols), data_(),
          bitpacks_(internals::bitpack_count(checked_size_(rows, cols), static_cast<int>(PackSize))) {
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
        : Base(size), data_(),
          bitpacks_(internals::bitpack_count(checked_vector_size_(size), static_cast<int>(PackSize))) {
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
        Base(), data_(), bitpacks_(StorageSize) {
        fdapde_static_assert(
          Rows_ != Dynamic && Cols_ != Dynamic && Rows * Cols == Size, THIS_METHOD_IS_FOR_STATIC_SIZED_MATRICES_ONLY);
        for (int i = 0; i < Rows; ++i) {
            for (int j = 0; j < Cols; ++j) {
                this->operator()(i, j) = data[i * Cols + j];
            }
        }
        return;
    }
    template <typename Scalar_>
        requires(std::is_convertible_v<Scalar_, Scalar>)
    constexpr explicit Matrix(const std::vector<Scalar_>& data) :
        Base(), data_(), bitpacks_(StorageSize == Dynamic ? 0 : StorageSize) {
        fdapde_static_assert(
          (Rows_ != Dynamic && Cols_ != Dynamic) || (Rows_ == 1 || Cols_ == 1),
          THIS_METHOD_IS_FOR_STATIC_SIZED_MATRICES_OR_VECTORS);
        const int input_size = internals::checked_matrix_data_size(data.size());
        if constexpr (Rows_ == Dynamic || Cols_ == Dynamic) {
            resize(input_size);
        } else if (this->size() != input_size) {
            throw std::invalid_argument("matrix input size does not match its shape");
        }
        const int rows = this->rows();
        const int cols = this->cols();
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                this->operator()(i, j) = data[static_cast<std::size_t>(i * cols + j)];
            }
        }
        return;
    }
    // static named constructors
    static constexpr auto Zero() { return ZeroMatrix<bool, Rows_, Cols_>(); }
    static constexpr auto Zero(int rows) {
        fdapde_static_assert(Rows_ == 1 || Cols_ == 1, THIS_METHOD_IS_FOR_ROW_OR_COLUMN_VECTORS_ONLY);
        return ZeroMatrix<bool, Rows_ == 1 ? Rows_ : Dynamic, Cols_ == 1 ? Cols_ : Dynamic>(rows);
    }
    static constexpr auto Zero(int rows, int cols) { return ZeroMatrix<bool, Dynamic, Dynamic>(rows, cols); }
    static constexpr auto Ones() { return OnesMatrix<bool, Rows_, Cols_>(); }
    static constexpr auto Ones(int rows) {
        fdapde_static_assert(Rows_ == 1 || Cols_ == 1, THIS_METHOD_IS_FOR_ROW_OR_COLUMN_VECTORS_ONLY);
        return OnesMatrix<bool, Rows_ == 1 ? Rows_ : Dynamic, Cols_ == 1 ? Cols_ : Dynamic>(rows);
    }
    static constexpr auto Ones(int rows, int cols) { return OnesMatrix<bool, Dynamic, Dynamic>(rows, cols); }
    // observers
    constexpr int bitpacks() const { return bitpacks_; }
    constexpr bitpack_t bitpack(int i) const { return data_[i]; }
    constexpr bitpack_t& bitpack(int i) { return data_[i]; }
    // modifiers
    void resize(int rows, int cols) {
        fdapde_static_assert(Rows_ == Dynamic || Cols_ == Dynamic, THIS_METHOD_IS_FOR_DYNAMIC_SIZED_MATRICES_ONLY);
        const int size = checked_size_(rows, cols);
        const int new_rows = Rows_ == Dynamic ? rows : Rows_;
        const int new_cols = Cols_ == Dynamic ? cols : Cols_;
        if (new_rows == this->rows_ && new_cols == this->cols_) return;
        const int old_size = this->size();
        const int new_bitpacks = internals::bitpack_count(size, static_cast<int>(PackSize));
        data_.resize(static_cast<std::size_t>(new_bitpacks), 0);
        for (int i = old_size; i < size; ++i) {
            data_[static_cast<std::size_t>(i / static_cast<int>(PackSize))] &=
              ~(bitpack_t(1) << (i % static_cast<int>(PackSize)));
        }
        if (new_bitpacks > 0) {
            const int used_bits = size - (new_bitpacks - 1) * static_cast<int>(PackSize);
            data_.back() &= internals::low_bits_mask<bitpack_t>(used_bits);
        }
        bitpacks_ = new_bitpacks;
        this->rows_ = new_rows;
        this->cols_ = new_cols;
        this->row_stride_ = StorageOrder_ == RowMajor ? new_cols : 1;
        this->col_stride_ = StorageOrder_ == RowMajor ? 1 : new_rows;
        return;
    }
    void resize(int size) {
        fdapde_static_assert(
          (Rows_ == 1 && Cols_ == Dynamic) || (Cols_ == 1 && Rows_ == Dynamic),
          THIS_METHOD_IS_FOR_DYNAMIC_SIZED_ROW_OR_COLUMN_VECTORS_ONLY);
        resize(Rows_ == Dynamic ? size : Rows_, Cols_ == Dynamic ? size : Cols_);
        return;
    }
    constexpr void set(int i, int j) { this->operator()(i, j).set(); }
    constexpr void set(int i) {
        fdapde_static_assert(Rows == 1 || Cols == 1, THIS_METHOD_IS_FOR_ROW_OR_COLUMN_VECTORS_ONLY);
        this->operator[](i).set();
    }
    constexpr void set() {   // sets all coeffients
        for (int i = 0, n = data_.size(); i < n; ++i) { data_[i] = bitpack_t(-1); }
    }
    constexpr void clear(int i, int j) { this->operator()(i, j).clear(); }
    constexpr void clear(int i) {
        fdapde_static_assert(Rows == 1 || Cols == 1, THIS_METHOD_IS_FOR_ROW_OR_COLUMN_VECTORS_ONLY);
        this->operator[](i).clear();
    }
    constexpr void clear() {   // clears all coeffients
        for (int i = 0, n = data_.size(); i < n; ++i) { data_[i] = bitpack_t(0); }
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
template <typename XprType_, typename BitWiseOperation, typename BitPackOperation>
struct BoolMatrixBitWiseOp : public BoolMatrixExpr<BoolMatrixBitWiseOp<XprType_, BitWiseOperation, BitPackOperation>> {
   private:
    using XprType = std::decay_t<XprType_>;
    using XprTypeNested = internals::ref_select_t<const XprType>;
   public:
    using Scalar = typename XprType::Scalar;
    using bitpack_t = typename XprType::bitpack_t;
    static constexpr int Rows = XprType::Rows;
    static constexpr int Cols = XprType::Cols;
    static constexpr int StorageOrder = XprType::StorageOrder;
    static constexpr int NestAsRef = 0;
    static constexpr int ReadOnly = 1;

    template <typename XprType__>
        requires(internals::safely_nestable<XprTypeNested, XprType__>)
    constexpr BoolMatrixBitWiseOp(XprType__&& xpr, BitWiseOperation bitwise_op, BitPackOperation bitpack_op) :
        xpr_(std::forward<XprType__>(xpr)), bitwise_op_(bitwise_op), bitpack_op_(bitpack_op) { }

    constexpr Scalar operator()(int i, int j) const { return bitwise_op_(xpr_(i, j)); }
    constexpr Scalar operator[](int i) const {
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

template <typename LhsXprType_, typename RhsXprType_, typename BitWiseOperation, typename BitPackOperation>
struct BoolMatrixBinOp :
    public BoolMatrixExpr<BoolMatrixBinOp<LhsXprType_, RhsXprType_, BitWiseOperation, BitPackOperation>> {
   private:
    using LhsXprType = std::decay_t<LhsXprType_>;
    using RhsXprType = std::decay_t<RhsXprType_>;
    fdapde_static_assert(
      internals::same_static_shape_weak_v<LhsXprType FDAPDE_COMMA RhsXprType>,
      INVALID_BINARY_OPERATION__OPERANDS_OF_DIFFERENT_STATIC_SIZE);
    fdapde_static_assert(
      std::is_same_v<typename LhsXprType::bitpack_t FDAPDE_COMMA typename RhsXprType::bitpack_t>,
      INVALID_BINARY_OPERATION__OPERANDS_OF_DIFFERENT_BITPACK_LAYOUT);
    using LhsXprTypeNested = internals::ref_select_t<const LhsXprType>;
    using RhsXprTypeNested = internals::ref_select_t<const RhsXprType>;
   public:
    using Scalar = promote_type_t<typename LhsXprType::Scalar, typename RhsXprType::Scalar>;
    using bitpack_t = typename LhsXprType::bitpack_t;
    static constexpr int PackSize = sizeof(bitpack_t) * 8;
    static constexpr int Rows =
      (LhsXprType::Rows == Dynamic || RhsXprType::Rows == Dynamic) ? Dynamic : LhsXprType::Rows;
    static constexpr int Cols =
      (LhsXprType::Cols == Dynamic || RhsXprType::Cols == Dynamic) ? Dynamic : LhsXprType::Cols;
    static constexpr int StorageOrder =
      internals::promote_storage_order_v<LhsXprType::StorageOrder, RhsXprType::StorageOrder>;
    static constexpr int NestAsRef = 0;
    static constexpr int ReadOnly = 1;

    template <typename LhsXprType__, typename RhsXprType__>
        requires(internals::safely_nestable<LhsXprTypeNested, LhsXprType__> &&
                 internals::safely_nestable<RhsXprTypeNested, RhsXprType__>)
    constexpr BoolMatrixBinOp(
      LhsXprType__&& lhs, RhsXprType__&& rhs, BitWiseOperation bitwise_op, BitPackOperation bitpack_op) :
        lhs_(std::forward<LhsXprType__>(lhs)),
        rhs_(std::forward<RhsXprType__>(rhs)),
        bitwise_op_(bitwise_op),
        bitpack_op_(bitpack_op) {
        if constexpr (internals::is_dynamic_sized_v<LhsXprType> || internals::is_dynamic_sized_v<RhsXprType>) {
            if (!std::cmp_equal(lhs_.rows(), rhs_.rows()) || !std::cmp_equal(lhs_.cols(), rhs_.cols())) {
                throw std::invalid_argument("Boolean binary operation requires matching dimensions");
            }
        }
    }
    constexpr Scalar operator()(int i, int j) const { return bitwise_op_(lhs_(i, j), rhs_(i, j)); }
    constexpr Scalar operator[](int i) const {
        fdapde_static_assert(
          (LhsXprType::Cols == 1 && RhsXprType::Cols == 1) || (LhsXprType::Rows == 1 && RhsXprType::Rows == 1),
          THIS_METHOD_IS_FOR_ROW_OR_COLUMN_VECTORS_ONLY);
        return bitwise_op_(lhs_[i], rhs_[i]);
    }
    constexpr bitpack_t bitpack(int i) const {
        if constexpr (LhsXprType::StorageOrder == RhsXprType::StorageOrder) {
            return bitpack_op_(lhs_.bitpack(i), rhs_.bitpack(i));
        } else {
            bitpack_t out = bitpack_t(0);
            const int base = i * PackSize;
            for (int offset = 0; offset < PackSize && base + offset < rows() * cols(); ++offset) {
                const int index = base + offset;
                const int row = StorageOrder == RowMajor ? index / cols() : index % rows();
                const int col = StorageOrder == RowMajor ? index % cols() : index / rows();
                if (bool(bitwise_op_(lhs_(row, col), rhs_(row, col)))) out |= bitpack_t(1) << offset;
            }
            return out;
        }
    }
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
constexpr auto operator&(const BoolMatrixExpr<LhsXprType>& lhs, const BoolMatrixExpr<RhsXprType>& rhs) {
    return BoolMatrixBinOp<LhsXprType, RhsXprType, std::bit_and<>, std::bit_and<>>(
      lhs.derived(), rhs.derived(), std::bit_and<>(), std::bit_and<>());
}
template <typename LhsXprType, typename RhsXprType>
constexpr auto operator|(const BoolMatrixExpr<LhsXprType>& lhs, const BoolMatrixExpr<RhsXprType>& rhs) {
    return BoolMatrixBinOp<LhsXprType, RhsXprType, std::bit_or<>, std::bit_or<>>(
      lhs.derived(), rhs.derived(), std::bit_or<>(), std::bit_or<>());
}
template <typename LhsXprType, typename RhsXprType>
constexpr auto operator^(const BoolMatrixExpr<LhsXprType>& lhs, const BoolMatrixExpr<RhsXprType>& rhs) {
    return BoolMatrixBinOp<LhsXprType, RhsXprType, std::bit_xor<>, std::bit_xor<>>(
      lhs.derived(), rhs.derived(), std::bit_xor<>(), std::bit_xor<>());
}

template <internals::bool_matrix_expression Lhs, internals::bool_matrix_expression Rhs>
    requires(
      internals::is_owning_rvalue_expression_v<Lhs&&> || internals::is_owning_rvalue_expression_v<Rhs&&>)
constexpr void operator&(Lhs&&, Rhs&&) = delete;

template <internals::bool_matrix_expression Lhs, internals::bool_matrix_expression Rhs>
    requires(
      internals::is_owning_rvalue_expression_v<Lhs&&> || internals::is_owning_rvalue_expression_v<Rhs&&>)
constexpr void operator|(Lhs&&, Rhs&&) = delete;

template <internals::bool_matrix_expression Lhs, internals::bool_matrix_expression Rhs>
    requires(
      internals::is_owning_rvalue_expression_v<Lhs&&> || internals::is_owning_rvalue_expression_v<Rhs&&>)
constexpr void operator^(Lhs&&, Rhs&&) = delete;

// dense-block of binary matrix
template <int BlockRows_, int BlockCols_, typename XprType_>
class BoolMatrixBlock : public BoolMatrixExpr<BoolMatrixBlock<BlockRows_, BlockCols_, XprType_>> {
   private:
    using Base = BoolMatrixExpr<BoolMatrixBlock<BlockRows_, BlockCols_, XprType_>>;
    using XprType = std::decay_t<XprType_>;
    fdapde_static_assert(
      internals::is_dynamic_sized_v<XprType> ||
        ((BlockRows_ == Dynamic || (BlockRows_ > 0 && BlockRows_ <= XprType::Rows)) &&
         (BlockCols_ == Dynamic || (BlockCols_ > 0 && BlockCols_ <= XprType::Cols))),
      INVALID_BLOCK__STATIC_SIZES_DONT_FIT_WRAPPED_EXPRESSION);
    using XprTypeNested = internals::ref_select_t<XprType_>;   // derive constness from wrapped expression
    static constexpr int PackSize = Base::PackSize;
   public:
    using Scalar = typename XprType::Scalar;
    using bitpack_t = typename XprType::bitpack_t;
    static constexpr int Rows = BlockRows_;
    static constexpr int Cols = BlockCols_;
    static constexpr int NestAsRef = 0;
    static constexpr int StorageOrder = XprType::StorageOrder;
    static constexpr int ReadOnly = XprType::ReadOnly;
    using assignment_executor = internals::generic_assignment_executor;   // bitwise assignment loop

    // row/column constructor
    template <typename XprType__>
        requires(std::is_constructible_v<XprTypeNested, XprType__>)
    constexpr BoolMatrixBlock(XprType__&& xpr, int i) :
        start_row_(BlockRows_ == 1 ? i : 0),
        start_col_(BlockCols_ == 1 ? i : 0),
        block_rows_(BlockRows_ == 1 ? 1 : xpr.rows()),
        block_cols_(BlockCols_ == 1 ? 1 : xpr.cols()),
        xpr_(std::forward<XprType__>(xpr)) {
        fdapde_static_assert(BlockRows_ == 1 || BlockCols_ == 1, THIS_METHOD_IS_FOR_ROW_AND_COLUMN_BLOCKS_ONLY);
        fdapde_assert(
          i >= 0 && ((BlockRows_ == 1 && i < xpr_.rows()) || (BlockCols_ == 1 && i < xpr_.cols())));
    }
    // static-sized constructor
    template <typename XprType__>
        requires(std::is_constructible_v<XprTypeNested, XprType__>)
    constexpr BoolMatrixBlock(XprType__&& xpr, int start_row, int start_col) :
        start_row_(start_row),
        start_col_(start_col),
        block_rows_(BlockRows_),
        block_cols_(BlockCols_),
        xpr_(std::forward<XprType__>(xpr)) {
        fdapde_static_assert(
          BlockRows_ != Dynamic && BlockCols_ != Dynamic, THIS_METHOD_IS_FOR_STATIC_SIZED_BLOCKS_ONLY);
        fdapde_assert(
          start_row >= 0 && start_row + block_rows_ <= xpr_.rows() && start_col >= 0 &&
          start_col + block_cols_ <= xpr.cols());
    }
    // dynamic-sized constructor
    template <typename XprType__>
        requires(std::is_constructible_v<XprTypeNested, XprType__>)
    constexpr BoolMatrixBlock(XprType__&& xpr, int start_row, int start_col, int block_rows, int block_cols) :
        start_row_(start_row),
        start_col_(start_col),
        block_rows_(block_rows),
        block_cols_(block_cols),
        xpr_(std::forward<XprType__>(xpr)) {
        fdapde_static_assert(
          BlockRows_ == Dynamic && BlockCols_ == Dynamic, THIS_METHOD_IS_FOR_DYNAMIC_SIZED_BLOCKS_ONLY);
        fdapde_assert(
          start_row >= 0 && start_row + block_rows_ <= xpr_.rows() && start_col >= 0 &&
          start_col + block_cols_ <= xpr.cols());
    }
    // inherit assignment from base
    using Base::operator=;
    // observers
    constexpr int rows() const noexcept { return BlockRows_ != Dynamic ? BlockRows_ : block_rows_; }
    constexpr int cols() const noexcept { return BlockCols_ != Dynamic ? BlockCols_ : block_cols_; }
    constexpr int size() const { return rows() * cols(); }
    constexpr int bitpacks() const { return internals::bitpack_count(size(), PackSize); }
    constexpr decltype(auto) operator()(int i, int j) const {
        fdapde_assert(i >= 0 && i < block_rows_ && j >= 0 && j < block_cols_);
        return xpr_(i + start_row_, j + start_col_);
    }
    constexpr decltype(auto) operator[](int i) const {
        fdapde_static_assert(BlockRows_ == 1 || BlockCols_ == 1, THIS_METHOD_IS_FOR_ROW_AND_COLUMN_BLOCKS_ONLY);
        if constexpr (Rows == 1) return xpr_(start_row_, start_col_ + i);
        if constexpr (Cols == 1) return xpr_(start_row_ + i, start_col_);
    }
    constexpr decltype(auto) operator()(int i, int j) {
        fdapde_assert(i >= 0 && i < block_rows_ && j >= 0 && j < block_cols_);
        return xpr_(start_row_ + i, start_col_ + j);
    }
    constexpr decltype(auto) operator[](int i) {
        fdapde_static_assert(BlockRows_ == 1 || BlockCols_ == 1, THIS_METHOD_IS_FOR_ROW_AND_COLUMN_BLOCKS_ONLY);
        if constexpr (Rows == 1) return xpr_(start_row_, start_col_ + i);
        if constexpr (Cols == 1) return xpr_(start_row_ + i, start_col_);
    }
    // modifiers
    constexpr void set(int i, int j) noexcept {
        fdapde_assert(i >= 0 && i < block_rows_ && j >= 0 && j < block_cols_);
        xpr_.set(i + start_row_, j + start_col_);
    }
    constexpr void set() noexcept {
        for (int i = 0; i < rows(); ++i) {
            for (int j = 0; j < cols(); ++j) { xpr_.set(start_row_ + i, start_col_ + j); }
        }
    }
    constexpr void clear(int i, int j) noexcept {
        fdapde_assert(i >= 0 && i < block_rows_ && j >= 0 && j < block_cols_);
        xpr_.clear(i + start_row_, j + start_col_);
    }
    constexpr void clear() noexcept {
        for (int i = 0; i < rows(); ++i) {
            for (int j = 0; j < cols(); ++j) { xpr_.clear(start_row_ + i, start_col_ + j); }
        }
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
    int start_row_ = 0, start_col_ = 0;
    int block_rows_ = 0, block_cols_ = 0;
    XprTypeNested xpr_;
};

// boolean redux operation
namespace internals {

// bitpack reduction loop on matrix expressions
struct matrix_redux_bitpack_executor {
    template <typename XprType, typename Scalar, typename Executor>
    static constexpr Scalar run(const XprType& xpr, Scalar init, Executor executor) {
        fdapde_assert(xpr.size() > 0);
        constexpr int PackSize = std::decay_t<XprType>::PackSize;
        const int size = xpr.size();
        Scalar res = init;

        if (size < PackSize) {
            res = executor(res, xpr.bitpack(0), size);   // size: number of valid bits in bitpack
            return res;
        }
        int k = 0, i = 0;   // k: current bitpack, i: last bit processed
        for (; i + PackSize <= size; ++k, i += PackSize) {
            res = executor(res, xpr.bitpack(k));
            if (executor.early_exit()) { return res; }
        }
        // process last, partially filled, bitpack
        if (i < size) { res = executor(res, xpr.bitpack(k), size - i); }
        return res;
    }
};

// evaluates true if all the coefficients of XprType are true
struct all_redux_bitpack_executor {
    constexpr all_redux_bitpack_executor() noexcept : all_(true) { }
    template <typename bitpack_t> constexpr bool operator()([[maybe_unused]] bool b, bitpack_t p) noexcept {
        all_ &= (p == std::numeric_limits<bitpack_t>::max());
        return all_;
    }
    template <typename bitpack_t> constexpr bool operator()([[maybe_unused]] bool b, bitpack_t p, int size) noexcept {
        const bitpack_t mask = (bitpack_t(1) << size) - 1;
        all_ &= ((p & mask) == mask);
        return all_;
    }
    constexpr bool early_exit() const noexcept { return !all_; }
   private:
    bool all_;
};
// evaluates true if at least one coefficient of XprType is true
struct any_redux_bitpack_executor {
    constexpr any_redux_bitpack_executor() noexcept : any_(false) { }
    template <typename bitpack_t> constexpr bool operator()([[maybe_unused]] bool b, bitpack_t p) noexcept {
        any_ |= (p != bitpack_t(0));
        return any_;
    }
    template <typename bitpack_t> constexpr bool operator()([[maybe_unused]] bool b, bitpack_t p, int size) noexcept {
        const bitpack_t mask = ((bitpack_t(1) << size) - 1);
        any_ |= ((p & mask) != bitpack_t(0));
        return any_;
    }
    constexpr bool early_exit() const noexcept { return any_; }
   private:
    bool any_;
};
// number of true coefficients in XprType, based on popcnt machine istruction for fast scalar counting
struct cnt_redux_bitpack_executor {
    constexpr cnt_redux_bitpack_executor() noexcept = default;
    template <typename bitpack_t> constexpr int operator()(int cnt, bitpack_t p) const noexcept {
        return cnt + std::popcount(p);
    }
    template <typename bitpack_t> constexpr int operator()(int cnt, bitpack_t p, int size) const noexcept {
        const bitpack_t mask = ((bitpack_t(1) << size) - 1);
        return cnt + std::popcount(p & mask);
    }
    constexpr bool early_exit() const noexcept { return false; }   // never stop
}; 
  
}   // namespace internals

// reshaping operation with bitpack support. As reshaping mantains the physical memory layout, bitpacks are preserved
template <int Rows_, int Cols_, typename XprType_>
class BoolReshapeOp : public BoolMatrixExpr<BoolReshapeOp<Rows_, Cols_, XprType_>> {
   private:
    using XprType = std::decay_t<XprType_>;
    using XprTypeNested = ReshapeOp<Rows_, Cols_, XprType_>;   // reuse standard reshaping
   public:
    static constexpr int Rows = Rows_;
    static constexpr int Cols = Cols_;
    static constexpr int NestAsRef = 0;
    static constexpr int ReadOnly = XprType::ReadOnly;

    constexpr BoolReshapeOp() = default;
    constexpr BoolReshapeOp(const BoolReshapeOp& other) : xpr_(other.xpr_) { }
    constexpr BoolReshapeOp& operator=(const BoolReshapeOp& other) {
        xpr_ = other.xpr_;
        return *this;
    }
    template <typename XprType__>
        requires(std::is_constructible_v<XprTypeNested, XprType__>)
    constexpr explicit BoolReshapeOp(XprType__&& xpr) : xpr_(std::forward<XprType__>(xpr)) { }
    template <typename XprType__>
        requires(std::is_constructible_v<XprTypeNested, XprType__>)
    constexpr BoolReshapeOp(XprType__&& xpr, int rows, int cols) :
        xpr_(std::forward<XprType__>(xpr), rows, cols) { }
    template <typename XprType__>
        requires(std::is_constructible_v<XprTypeNested, XprType__>)
    constexpr BoolReshapeOp(XprType__&& xpr, int rows) : xpr_(std::forward<XprType__>(xpr), rows) { }

    constexpr int rows() const { return xpr_.rows(); }
    constexpr int cols() const { return xpr_.cols(); }
    constexpr decltype(auto) bitpack(int i) const { return xpr_.bitpack(i); }
    // access
    constexpr decltype(auto) operator()(int i, int j) const { return xpr_(i, j); }
    constexpr decltype(auto) operator[](int i) const { return Rows == 1 ? operator()(i, 0) : operator()(0, i); }
    constexpr decltype(auto) operator()(int i, int j) { return xpr_(i, j); }
    constexpr decltype(auto) operator[](int i) { return Rows == 1 ? operator()(i, 0) : operator()(0, i); }
   private:
    XprTypeNested xpr_;
};
  
// base class for boolean expressions
template <typename XprType_> struct BoolMatrixExpr {
    using XprType = std::decay_t<XprType_>;
    using bitpack_t = std::uintmax_t;
    using Scalar = bool;
    static constexpr std::size_t PackSize = sizeof(bitpack_t) * 8;   // number of bits in a packet

    // assignment
    template <typename RhsXprType_>
        requires(XprType::ReadOnly == 0)
    constexpr XprType& operator=(const BoolMatrixExpr<RhsXprType_>& rhs) & {
        using RhsXprType = std::decay_t<RhsXprType_>;
        Matrix<bool, RhsXprType::Rows, RhsXprType::Cols, XprType::StorageOrder> tmp(rhs);
        using executor = typename XprType::assignment_executor;
        constexpr int Rows = XprType::Rows;
        constexpr int Cols = XprType::Cols;
        if constexpr (requires(XprType_ xpr, int i, int j) {
                          xpr.resize(i, j);
                      } && (Rows == Dynamic || Cols == Dynamic)) {
            if (derived().rows() != tmp.rows() || derived().cols() != tmp.cols()) {
                derived().resize(tmp.rows(), tmp.cols());
            }
        }
        executor::run(derived(), tmp, [](auto&& l, const auto& r) {
            if constexpr (std::is_same_v<std::remove_cvref_t<decltype(l)> FDAPDE_COMMA bitpack_t>) {
                l = r;
            } else {
                l = bool(r);
            }
        });
        return derived();
    }
    template <typename RhsXprType_>
    constexpr XprType operator=(const BoolMatrixExpr<RhsXprType_>& rhs) &&
        requires(XprType::NestAsRef == 0 && XprType::ReadOnly == 0)
    {
        static_cast<BoolMatrixExpr&>(*this).operator=(rhs);
        return derived();
    }
    template <typename RhsXprType_>
    constexpr void operator=(const BoolMatrixExpr<RhsXprType_>&) &&
        requires(XprType::NestAsRef != 0)
      = delete;
    // compound boolean algebra
    template <typename RhsXprType_>
        requires(XprType::ReadOnly == 0)
    constexpr XprType& operator&=(const BoolMatrixExpr<RhsXprType_>& other) & {
        using RhsXprType = std::decay_t<RhsXprType_>;
        Matrix<bool, RhsXprType::Rows, RhsXprType::Cols, XprType::StorageOrder> tmp(other);
        using executor = typename XprType::assignment_executor;
        executor::run(derived(), tmp, [](auto&& l, const auto& r) { l = std::bit_and<>()(l, r); });
        return derived();
    }
    template <typename RhsXprType_>
    constexpr XprType operator&=(const BoolMatrixExpr<RhsXprType_>& other) &&
        requires(XprType::NestAsRef == 0 && XprType::ReadOnly == 0)
    {
        static_cast<BoolMatrixExpr&>(*this).operator&=(other);
        return derived();
    }
    template <typename RhsXprType_>
    constexpr void operator&=(const BoolMatrixExpr<RhsXprType_>&) &&
        requires(XprType::NestAsRef != 0)
      = delete;
    template <typename RhsXprType_>
        requires(XprType::ReadOnly == 0)
    constexpr XprType& operator|=(const BoolMatrixExpr<RhsXprType_>& other) & {
        using RhsXprType = std::decay_t<RhsXprType_>;
        Matrix<bool, RhsXprType::Rows, RhsXprType::Cols, XprType::StorageOrder> tmp(other);
        using executor = typename XprType::assignment_executor;
        executor::run(derived(), tmp, [](auto&& l, const auto& r) { l = std::bit_or<>()(l, r); });
        return derived();
    }
    template <typename RhsXprType_>
    constexpr XprType operator|=(const BoolMatrixExpr<RhsXprType_>& other) &&
        requires(XprType::NestAsRef == 0 && XprType::ReadOnly == 0)
    {
        static_cast<BoolMatrixExpr&>(*this).operator|=(other);
        return derived();
    }
    template <typename RhsXprType_>
    constexpr void operator|=(const BoolMatrixExpr<RhsXprType_>&) &&
        requires(XprType::NestAsRef != 0)
      = delete;
    template <typename RhsXprType_>
        requires(XprType::ReadOnly == 0)
    constexpr XprType& operator^=(const BoolMatrixExpr<RhsXprType_>& other) & {
        using RhsXprType = std::decay_t<RhsXprType_>;
        Matrix<bool, RhsXprType::Rows, RhsXprType::Cols, XprType::StorageOrder> tmp(other);
        using executor = typename XprType::assignment_executor;
        executor::run(derived(), tmp, [](auto&& l, const auto& r) { l = std::bit_xor<>()(l, r); });
        return derived();
    }
    template <typename RhsXprType_>
    constexpr XprType operator^=(const BoolMatrixExpr<RhsXprType_>& other) &&
        requires(XprType::NestAsRef == 0 && XprType::ReadOnly == 0)
    {
        static_cast<BoolMatrixExpr&>(*this).operator^=(other);
        return derived();
    }
    template <typename RhsXprType_>
    constexpr void operator^=(const BoolMatrixExpr<RhsXprType_>&) &&
        requires(XprType::NestAsRef != 0)
      = delete;
    // observers
    constexpr int rows() const { return XprType::Rows == Dynamic ? derived().rows() : XprType::Rows; }
    constexpr int cols() const { return XprType::Cols == Dynamic ? derived().cols() : XprType::Cols; }
    constexpr int size() const {
        constexpr int Rows = XprType::Rows;
        constexpr int Cols = XprType::Cols;
        return (Rows != Dynamic && Cols != Dynamic) ? Rows * Cols : derived().rows() * derived().cols();
    }
    constexpr const XprType& derived() const & { return static_cast<const XprType&>(*this); }
    constexpr XprType& derived() & { return static_cast<XprType&>(*this); }
    constexpr void derived() const && = delete;
    constexpr void derived() && = delete;
    // ostream
    friend std::ostream& operator<<(std::ostream& out, const BoolMatrixExpr& m) {
        const int rows = m.derived().rows();
        const int cols = m.derived().cols();
        if (rows == 0 || cols == 0) return out;
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
    constexpr BoolMatrixBitWiseOp<XprType, std::logical_not<>, std::bit_not<>> operator~() const & {
        return BoolMatrixBitWiseOp<XprType, std::logical_not<>, std::bit_not<>>(
          derived(), std::logical_not<>(), std::bit_not<>());
    }
    constexpr BoolMatrixBitWiseOp<XprType, std::logical_not<>, std::bit_not<>> operator~() const &&
        requires(XprType::NestAsRef == 0)
    {
        return BoolMatrixBitWiseOp<XprType, std::logical_not<>, std::bit_not<>>(
          derived(), std::logical_not<>(), std::bit_not<>());
    }
    constexpr void operator~() const && requires(XprType::NestAsRef != 0) = delete;
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
    constexpr auto col(int i) { return BoolMatrixBlock<XprType::Rows, 1, XprType>(derived(), i); }
    constexpr auto col(int i) const { return BoolMatrixBlock<XprType::Rows, 1, const XprType>(derived(), i); }
    constexpr auto row(int i) { return BoolMatrixBlock<1, XprType::Cols, XprType>(derived(), i); }
    constexpr auto row(int i) const { return BoolMatrixBlock<1, XprType::Cols, const XprType>(derived(), i); }
    // other block-type accessors
    template <int BlockRows> constexpr auto top_rows() { return block<BlockRows, XprType::Cols>(0, 0); }
    template <int BlockRows> constexpr auto top_rows() const { return block<BlockRows, XprType::Cols>(0, 0); }
    constexpr auto top_rows(int rows) { return block(0, 0, rows, derived().cols()); }
    constexpr auto top_rows(int rows) const { return block(0, 0, rows, derived().cols()); }
    template <int BlockRows> constexpr auto bottom_rows() {
        return block<BlockRows, XprType::Cols>(derived().rows() - BlockRows, 0);
    }
    template <int BlockRows> constexpr auto bottom_rows() const {
        return block<BlockRows, XprType::Cols>(derived().rows() - BlockRows, 0);
    }
    constexpr auto bottom_rows(int rows) { return block(derived().rows() - rows, 0, rows, derived().cols()); }
    constexpr auto bottom_rows(int rows) const { return block(derived().rows() - rows, 0, rows, derived().cols()); }
    template <int BlockCols> constexpr auto left_cols() { return block<XprType::Rows, BlockCols>(0, 0); }
    template <int BlockCols> constexpr auto left_cols() const { return block<XprType::Rows, BlockCols>(0, 0); }
    constexpr auto left_cols(int cols) { return block(0, 0, derived().rows(), cols); }
    constexpr auto left_cols(int cols) const { return block(0, 0, derived().rows(), cols); }
    template <int BlockCols> constexpr auto right_cols() {
        return block<XprType::Rows, BlockCols>(0, derived().rows() - BlockCols);
    }
    template <int BlockCols> constexpr auto right_cols() const {
        return block<XprType::Rows, BlockCols>(0, derived().rows() - BlockCols);
    }
    constexpr auto right_cols(int cols) { return block(0, derived().cols() - cols, derived().rows(), cols); }
    constexpr auto right_cols(int cols) const { return block(0, derived().cols() - cols, derived().rows(), cols); }
    // visitor support
    constexpr bool all() const {
        if (derived().size() == 0) return true;
        return internals::matrix_redux_bitpack_executor::run(derived(), 1, internals::all_redux_bitpack_executor());
    }
    constexpr bool any() const {
        if (derived().size() == 0) return false;
        return internals::matrix_redux_bitpack_executor::run(derived(), 1, internals::any_redux_bitpack_executor());
    }
    constexpr int count() const {
        if (derived().size() == 0) return 0;
        return internals::matrix_redux_bitpack_executor::run(derived(), 0, internals::cnt_redux_bitpack_executor());
    }
    // binary selection
    template <typename TrueXprType, typename FalseXprType>
        requires(internals::is_matrix_like_v<TrueXprType> && internals::is_matrix_like_v<FalseXprType>)
    constexpr auto select(TrueXprType&& true_xpr, FalseXprType&& false_xpr) const {
        return TernaryOp<XprType, std::decay_t<TrueXprType>, std::decay_t<FalseXprType>>(
          derived(), std::forward<TrueXprType>(true_xpr), std::forward<FalseXprType>(false_xpr));
    }
    // reshaping
    // static-sized
    template <int ReshapedRows_, int ReshapedCols_> constexpr auto reshape() {
        return BoolReshapeOp<ReshapedRows_, ReshapedCols_, XprType>(derived());
    }
    template <int ReshapedRows_, int ReshapedCols_> constexpr auto reshape() const {
        return BoolReshapeOp<ReshapedRows_, ReshapedCols_, const XprType>(derived());
    }
    template <int ReshapedRows_> constexpr auto reshape() {
        return BoolReshapeOp<ReshapedRows_, 1, XprType>(derived());
    }
    template <int ReshapedRows_> constexpr auto reshape() const {
        return BoolReshapeOp<ReshapedRows_, 1, const XprType>(derived());
    }
    // dynamic-sized
    constexpr auto reshape(int rows, int cols) {
        return BoolReshapeOp<Dynamic, Dynamic, XprType>(derived(), rows, cols);
    }
    constexpr auto reshape(int rows, int cols) const {
        return BoolReshapeOp<Dynamic, Dynamic, const XprType>(derived(), rows, cols);
    }
    constexpr auto reshape(int rows) { return BoolReshapeOp<Dynamic, 1, XprType>(derived(), rows); }
    constexpr auto reshape(int rows) const { return BoolReshapeOp<Dynamic, 1, const XprType>(derived(), rows); }
};
    
// comparison operator
template <typename LhsXprType, typename RhsXprType>
constexpr bool operator==(const BoolMatrixExpr<LhsXprType>& lhs, const BoolMatrixExpr<RhsXprType>& rhs) {
    fdapde_static_assert(
      (internals::is_dynamic_sized_v<LhsXprType> || internals::is_dynamic_sized_v<RhsXprType> ||
       internals::same_static_shape_v<LhsXprType FDAPDE_COMMA RhsXprType>),
      INVALID_COMPARISON__MATRICES_OF_DIFFERENT_STATIC_SIZE);
    if constexpr (internals::is_dynamic_sized_v<LhsXprType> || internals::is_dynamic_sized_v<RhsXprType>) {
        if (lhs.rows() != rhs.rows() || lhs.cols() != rhs.cols()) {
            throw std::invalid_argument("Boolean matrix dimensions do not match");
        }
    }
    using bitpack_t = typename LhsXprType::bitpack_t;
    constexpr int pack_size = sizeof(bitpack_t) * 8;

    const auto& d1 = lhs.derived();
    const auto& d2 = rhs.derived();
    if (d1.size() == 0) return true;
    bool result = true;
    int n = d1.bitpacks() - 1;
    // fast first n bitpacks comparison
    for (int i = 0; i < n && result; i++) { result &= (d1.bitpack(i) == d2.bitpack(i)); }
    // process last bitpack
    const bitpack_t mask = internals::low_bits_mask<bitpack_t>(d1.size() - pack_size * n);
    result &= ((mask & d1.bitpack(n)) == (mask & d2.bitpack(n)));
    return result;
}
template <typename LhsXprType, typename RhsXprType>
constexpr bool operator!=(const BoolMatrixExpr<LhsXprType>& op1, const BoolMatrixExpr<RhsXprType>& op2) {
    return !(op1 == op2);
}

// detection trait
template <typename XprType> struct is_boolean_matrix {
    using CleanXprType = std::remove_cvref_t<XprType>;
    static constexpr bool value = std::is_base_of_v<BoolMatrixExpr<CleanXprType>, CleanXprType>;
};
template <typename XprType> static constexpr bool is_boolean_matrix_v = is_boolean_matrix<XprType>::value;
template <typename XprType> struct is_boolean_vector {
    using CleanXprType = std::remove_cvref_t<XprType>;
    static constexpr bool value = [] {
        if constexpr (is_boolean_matrix_v<CleanXprType>) {
            return CleanXprType::Cols == 1 || CleanXprType::Rows == 1;
        } else {
            return false;
        }
    }();
};
template <typename XprType> static constexpr bool is_boolean_vector_v = is_boolean_vector<XprType>::value;
  
// indexes of true elements in the boolean expression
template <typename XprType> std::vector<int> which(const BoolMatrixExpr<XprType>& mtx) { return mtx.which(true); }

// returns boolean vector v such that v[i] = true \iff i-th element in range [first, last] equals c
template <typename Iterator, typename Scalar>
    requires(requires(Iterator first, Iterator last, int i) {
        { *first } -> std::convertible_to<Scalar>;
        { first + i } -> std::convertible_to<Iterator>;
        { first != last } -> std::same_as<bool>;
	{ ++first } -> std::convertible_to<Iterator>;
    })
Vector<bool, Dynamic> value_indicator(const Iterator& first, const Iterator& last, Scalar c) {
    int n_rows = std::distance(first, last);
    Vector<bool, Dynamic> vec(n_rows);
    for (int i = 0; i < n_rows; ++i) {
        if (*(first + i) == c) { vec.set(i); }
    }
    return vec;
}

// return boolean matrix m such that m(i, j) = true \iff (i,j)-th element of DataType is nan
template <typename DataType>
    requires(internals::is_vector_like_v<DataType> || internals::is_matrix_like_v<DataType>)
Matrix<bool, Dynamic, Dynamic> nan_indicator(DataType&& data) {
    const int rows = internals::is_vector_like_v<DataType> ? data.size() : data.rows();
    const int cols = internals::is_vector_like_v<DataType> ? 1 : data.cols();
    Matrix<bool, Dynamic, Dynamic> mask(rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            const auto val =
              internals::is_vector_like_v<DataType> ? internals::vector_like_access(data, i) : data(i, j);
            if (std::isnan(val)) mask.set(i, j);
        }
    }
    return mask;
}

// non-owning Matrix view of an existing block of data
template <int Rows_, int Cols_, int StorageOrder_>
class MatrixView<bool, Rows_, Cols_, StorageOrder_> :
    public MatrixBase<bool, Rows_, Cols_, StorageOrder_, MatrixView<bool, Rows_, Cols_, StorageOrder_>> {
   private:
    using Base = MatrixBase<bool, Rows_, Cols_, StorageOrder_, MatrixView<bool, Rows_, Cols_, StorageOrder_>>;
   public:
    using Scalar = typename Base::Scalar;
    using bitpack_t = typename Base::bitpack_t;   // machine largest integer type for bit-packing
    using StorageType = bitpack_t*;
    static constexpr int Rows = Rows_;
    static constexpr int Cols = Cols_;
    static constexpr int PackSize = sizeof(bitpack_t) * 8;
    static constexpr int NestAsRef = 0;

    // constructors
    constexpr MatrixView() : Base(), data_(nullptr), bitpacks_(0), last_bitpack_mask_(0) { }
    template <typename Scalar_>
        requires(std::is_convertible_v<Scalar_, bitpack_t>)
    constexpr explicit MatrixView(Scalar_* data) :
        Base(), data_(reinterpret_cast<bitpack_t*>(data)), bitpacks_(0), last_bitpack_mask_(0) {
        fdapde_static_assert(Rows_ != Dynamic && Cols_ != Dynamic, THIS_METHOD_IS_FOR_STATIC_SIZED_MATRICES_ONLY);
        bitpacks_ = internals::bitpack_count(Rows * Cols, PackSize);
        // compute last bitpack
        const int last_used_bits = (Rows * Cols - (bitpacks_ - 1) * PackSize);
        last_bitpack_mask_ = internals::low_bits_mask<bitpack_t>(last_used_bits);
    }
    template <typename Scalar_>
        requires(std::is_convertible_v<Scalar_, bitpack_t>)
    constexpr MatrixView(Scalar_* data, int size) :
        Base(size), data_(reinterpret_cast<bitpack_t*>(data)), bitpacks_(0), last_bitpack_mask_(0) {
        fdapde_static_assert(Rows_ == 1 || Cols_ == 1, THIS_METHOD_IS_FOR_ROW_OR_COLUMN_VECTORS_ONLY);
        fdapde_assert(size > 0);
        bitpacks_ = internals::bitpack_count(this->size(), PackSize);
        if (bitpacks_ > 0) {
            const int last_used_bits = (this->size() - (bitpacks_ - 1) * PackSize);
            last_bitpack_mask_ = internals::low_bits_mask<bitpack_t>(last_used_bits);
        }
    }
    template <typename Scalar_>
        requires(std::is_convertible_v<Scalar_, bitpack_t>)
    constexpr MatrixView(Scalar_* data, int rows, int cols) :
        Base(rows, cols), data_(reinterpret_cast<bitpack_t*>(data)), bitpacks_(0), last_bitpack_mask_(0) {
        fdapde_assert(rows > 0 && cols > 0);
        bitpacks_ = internals::bitpack_count(this->size(), PackSize);
        if (bitpacks_ > 0) {
            const int last_used_bits = (this->size() - (bitpacks_ - 1) * PackSize);
            last_bitpack_mask_ = internals::low_bits_mask<bitpack_t>(last_used_bits);
        }
    }
    // inherit assignment from Base
    using Base::operator=;
    // observers
    constexpr int bitpacks() const { return bitpacks_; }
    bitpack_t bitpack(int i) const {
        fdapde_assert(i >= 0 && i < bitpacks_);
        if (i < bitpacks_ - 1) {
            return data_[i];
        } else {
            return data_[i] & last_bitpack_mask_;
        }
    }
    // data pointers
    constexpr const bitpack_t* data() const { return data_; }
    constexpr bitpack_t* data() { return data_; }
    // modifiers
    constexpr void set(int i, int j) { this->operator()(i, j).set(); }
    constexpr void set(int i) {
        fdapde_static_assert(Rows == 1 || Cols == 1, THIS_METHOD_IS_FOR_ROW_OR_COLUMN_VECTORS_ONLY);
        this->operator[](i).set();
    }
    constexpr void set() {
        if (bitpacks_ == 0) return;
        for (int i = 0; i < bitpacks_ - 1; ++i) { data_[i] = ~bitpack_t(0); }
        data_[bitpacks_ - 1] |= last_bitpack_mask_;
    }
    constexpr void clear(int i, int j) { this->operator()(i, j).clear(); }
    constexpr void clear(int i) {
        fdapde_static_assert(Rows == 1 || Cols == 1, THIS_METHOD_IS_FOR_ROW_OR_COLUMN_VECTORS_ONLY);
        this->operator[](i).clear();
    }
    constexpr void clear() {
        if (bitpacks_ == 0) return;
        for (int i = 0; i < bitpacks_ - 1; ++i) { data_[i] = bitpack_t(0); }
        data_[bitpacks_ - 1] &= ~last_bitpack_mask_;
    }
   private:
    StorageType data_;
    int bitpacks_;
    bitpack_t last_bitpack_mask_;
};

}   // namespace fdapde

#endif   // __FDAPDE_LINALG_BOOL_H__
