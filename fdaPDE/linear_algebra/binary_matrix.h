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

#ifndef __BINARY_MATRIX_H__
#define __BINARY_MATRIX_H__

#include "../utils/assert.h"
#include "../utils/symbols.h"

namespace fdapde {
namespace core {

// forward declarations
template <int Rows, int Cols, typename E> class BinMtxBase;

namespace internals {
template <typename XprType> struct ref_select {
    using type = typename std::conditional<
      XprType::NestAsRef == 0, std::remove_reference_t<XprType>, std::add_lvalue_reference_t<XprType>>::type;
};
}   // namespace internals

// a space-time efficient implementation for a dense-storage matrix type holding binary values
template <int Rows, int Cols = Rows> class BinaryMatrix : public BinMtxBase<Rows, Cols, BinaryMatrix<Rows, Cols>> {
   public:
    using This = BinaryMatrix<Rows, Cols>;
    using Base = BinMtxBase<Rows, Cols, This>;
    using BitPackType = std::uintmax_t;
    static constexpr std::size_t PackSize = sizeof(std::uintmax_t) * 8;   // number of bits in a packet
    static constexpr int NestAsRef = 1;   // whether to store this node by reference or by copy in an expression
    static constexpr int ct_rows = Rows;
    static constexpr int ct_cols = Cols;

    template <typename T> struct is_dynamic_sized {
        static constexpr bool value = (T::ct_rows == Dynamic || T::ct_cols == Dynamic);
    };
    using StorageType = typename std::conditional<
      is_dynamic_sized<This>::value, std::vector<BitPackType>,
      std::array<BitPackType, 1 + (Rows * Cols) / PackSize>>::type;
    using Base::n_cols_;
    using Base::n_rows_;
    // static default constructor
    fdapde_enable_constructor_if(!is_dynamic_sized, This) BinaryMatrix() : Base(Rows, Cols) {
        std::fill_n(data_.begin(), data_.size(), 0);
    };
    // dynamic sized constructors
    fdapde_enable_constructor_if(is_dynamic_sized, This) BinaryMatrix() : Base() {};
    fdapde_enable_constructor_if(is_dynamic_sized, This) BinaryMatrix(std::size_t n_rows, std::size_t n_cols) :
        Base(n_rows, n_cols) {
        data_.resize(1 + std::ceil((n_rows_ * n_cols_) / PackSize), 0);
    }
    // vector constructor
    fdapde_enable_constructor_if(is_dynamic_sized, This) BinaryMatrix(std::size_t n_rows) : BinaryMatrix(n_rows, 1) {
        fdapde_static_assert(Rows == Dynamic && Cols == 1, THIS_METHOD_IS_ONLY_FOR_VECTORS);
    }

    // construct from expression
    template <int Rows_, int Cols_, typename Rhs_> BinaryMatrix(const BinMtxBase<Rows_, Cols_, Rhs_>& rhs) {
        // !is_dynamic_sized \implies (Rows == Rows_ && Cols == Cols_)
        fdapde_static_assert(
          is_dynamic_sized<This>::value || (Rows == Rows_ && Cols == Cols_) ||
            (Cols == 1 && (Rows_ == 1 || Cols_ == 1)),
          INVALID_MATRIX_ASSIGNMENT);
        // perform assignment
        if constexpr (Cols == 1 && (Rows_ == 1 || Cols_ == 1)) {   // assignment to vector
            n_rows_ = Rows_ == 1 ? rhs.cols() : rhs.rows();
        } else {
            n_rows_ = rhs.rows();
        }
        n_cols_ = Cols == 1 ? 1 : rhs.cols();
        Base::n_bitpacks_ = rhs.bitpacks();
        if constexpr (is_dynamic_sized<This>::value) data_.resize(Base::n_bitpacks_, 0);
        for (std::size_t i = 0; i < rhs.bitpacks(); ++i) { data_[i] = rhs.bitpack(i); }
    }

    // constructs a matrix of ones
    static BinaryMatrix<Rows, Cols> Ones(std::size_t i, std::size_t j) {
        BinaryMatrix<Rows, Cols> result;
        result.resize(i, j);
        for (std::size_t k = 0; k < result.bitpacks(); ++k) { result.bitpack(k) = -1; }
        return result;
    }
    static BinaryMatrix<Rows, Cols> Ones() {   // static-sized factory
        fdapde_static_assert(Rows != Dynamic && Cols != Dynamic, THIS_METHOD_IS_ONLY_FOR_STATIC_SIZED_MATRICES);
        return Ones(Rows, Cols);
    }
    static BinaryMatrix<Rows, Cols> Ones(std::size_t i) {   // vector-like factory
        fdapde_static_assert(Rows == Dynamic && Cols == 1, THIS_METHOD_IS_ONLY_FOR_VECTORS);
        BinaryMatrix<Rows, 1> result(i);
        for (std::size_t k = 0; k < result.bitpacks(); ++k) { result.bitpack(k) = -1; }
        return result;
    }

    template <int Rows_ = Rows, int Cols_ = Cols>
    typename std::enable_if<Rows_ == Dynamic && Cols_ == Dynamic, void>::type
    resize(std::size_t rows, std::size_t cols) {
        n_rows_ = rows;
        n_cols_ = cols;
        Base::n_bitpacks_ = 1 + std::ceil((n_rows_ * n_cols_) / PackSize);
        data_ = std::vector<BitPackType>(Base::n_bitpacks_, 0);
    }
    template <int Rows_ = Rows, int Cols_ = Cols>   // vector-like resizing
    typename std::enable_if<Rows_ == Dynamic && Cols_ == 1, void>::type resize(std::size_t rows) {
        n_rows_ = rows;
        n_cols_ = 1;
        Base::n_bitpacks_ = 1 + std::ceil(n_rows_ / PackSize);
        data_ = std::vector<BitPackType>(Base::n_bitpacks_, 0);
    }

    // getters
    bool operator()(std::size_t i, std::size_t j) const {   // access to (i,j)-th element
        return (data_[pack_of(i, j)] & BitPackType(1) << ((i * Base::n_cols_ + j) % PackSize)) != 0;
    }
    template <int Cols_ = Cols>   // vector-like (subscript) access
    typename std::enable_if<Cols_ == 1, bool>::type operator[](std::size_t i) const {
        return operator()(i, 0);
    }
    BitPackType bitpack(std::size_t i) const { return data_[i]; }
    BitPackType& bitpack(std::size_t i) { return data_[i]; }   // non-const access to i-th bitpack
    // setters
    void set(std::size_t i, std::size_t j) {   // set (i,j)-th bit
        fdapde_assert(i < n_rows_ && j < n_cols_);
        data_[pack_of(i, j)] |= (BitPackType(1) << ((i * Base::n_cols_ + j) % PackSize));
    }
    void set(std::size_t i) {
        fdapde_static_assert(Cols == 1, THIS_METHOD_IS_ONLY_FOR_VECTORS);
        set(i, 0);
    }
    void clear(std::size_t i, std::size_t j) {   // clear (i,j)-th bit (sets to 0)
        fdapde_assert(i < n_rows_ && j < n_cols_);
        data_[pack_of(i, j)] &= ~(BitPackType(1) << ((i * Base::n_cols_ + j) % PackSize));
    }
    void clear(std::size_t i) {
        fdapde_static_assert(Cols == 1, THIS_METHOD_IS_ONLY_FOR_VECTORS);
        clear(i, 0);
    }
    // fast bitwise assignment from binary expression
    template <int Rows_, int Cols_, typename Rhs_> BinaryMatrix& operator=(const BinMtxBase<Rows_, Cols_, Rhs_>& rhs) {
        // !is_dynamic_sized \implies (Rows == Rows_ && Cols == Cols_)
        fdapde_static_assert(
          is_dynamic_sized<This>::value || (Rows == Rows_ && Cols == Cols_) ||
            (Cols == 1 && (Rows_ == 1 || Cols_ == 1)),
          INVALID_MATRIX_ASSIGNMENT);
        // perform assignment
        if constexpr (Cols == 1 && (Rows_ == 1 || Cols_ == 1)) {   // row/column block assignment to vector
            n_rows_ = Rows_ == 1 ? rhs.cols() : rhs.rows();
        } else {
            n_rows_ = rhs.rows();
        }
        n_cols_ = Cols == 1 ? 1 : rhs.cols();
        Base::n_bitpacks_ = rhs.bitpacks();
        if constexpr (is_dynamic_sized<This>::value) data_.resize(Base::n_bitpacks_, 0);
        for (std::size_t i = 0; i < rhs.bitpacks(); ++i) { data_[i] = rhs.bitpack(i); }
        return *this;
    }
   private:
    StorageType data_;
    // recover the byte-pack for the (i,j)-th element
    inline std::size_t pack_of(std::size_t i, std::size_t j) const { return (i * Base::n_cols_ + j) / PackSize; }
};

// unary bitwise operation of binary expressions
template <int Rows, int Cols, typename XprTypeNested, typename UnaryBitwiseOp, typename UnaryLogicalOp>
struct BinMtxUnaryOp :
    public BinMtxBase<Rows, Cols, BinMtxUnaryOp<Rows, Cols, XprTypeNested, UnaryBitwiseOp, UnaryLogicalOp>> {
    using XprType = BinMtxUnaryOp<Rows, Cols, XprTypeNested, UnaryBitwiseOp, UnaryLogicalOp>;
    using Base = BinMtxBase<Rows, Cols, XprType>;
    using BitPackType = typename Base::BitPackType;
    static constexpr int NestAsRef = 0;   // whether to store this node by reference or by copy in an expression
    typename internals::ref_select<const XprTypeNested>::type op_;
    UnaryBitwiseOp f_bitwise_;
    UnaryLogicalOp f_logical_;

    BinMtxUnaryOp(const XprTypeNested& op, UnaryBitwiseOp f_bitwise, UnaryLogicalOp f_logical) :
        Base(op.rows(), op.cols()), op_(op), f_bitwise_(f_bitwise), f_logical_(f_logical) {};
    bool operator()(std::size_t i, std::size_t j) const { return f_logical_(op_(i, j)); }
    BitPackType bitpack(std::size_t i) const { return f_bitwise_(op_.bitpack(i)); }
};

// binary bitwise operations between binary expression
template <int Rows, int Cols, typename Lhs, typename Rhs, typename BinaryOperation>
struct BinMtxBinaryOp : public BinMtxBase<Rows, Cols, BinMtxBinaryOp<Rows, Cols, Lhs, Rhs, BinaryOperation>> {
    using XprType = BinMtxBinaryOp<Rows, Cols, Lhs, Rhs, BinaryOperation>;
    using Base = BinMtxBase<Rows, Cols, XprType>;
    using BitPackType = typename Base::BitPackType;
    static constexpr int NestAsRef = 0;   // whether to store this node by reference or by copy in an expression
    typename internals::ref_select<const Lhs>::type op1_;
    typename internals::ref_select<const Rhs>::type op2_;
    BinaryOperation f_;

    BinMtxBinaryOp(const Lhs& op1, const Rhs& op2, BinaryOperation f) :
        Base(op1.rows(), op1.cols()), op1_(op1), op2_(op2), f_(f) {
        fdapde_assert(op1_.rows() == op2_.rows() && op1_.cols() == op2_.cols());
    }
    bool operator()(std::size_t i, std::size_t j) const { return f_(op1_(i, j), op2_(i, j)); }
    BitPackType bitpack(std::size_t i) const { return f_(op1_.bitpack(i), op2_.bitpack(i)); }
};

#define DEFINE_BITWISE_BINMTX_OPERATOR(OPERATOR, FUNCTOR)                                                              \
    template <int Rows, int Cols, typename OP1, typename OP2>                                                          \
    BinMtxBinaryOp<Rows, Cols, OP1, OP2, FUNCTOR> OPERATOR(                                                            \
      const BinMtxBase<Rows, Cols, OP1>& op1, const BinMtxBase<Rows, Cols, OP2>& op2) {                                \
        return BinMtxBinaryOp<Rows, Cols, OP1, OP2, FUNCTOR> {op1.get(), op2.get(), FUNCTOR()};                        \
    }

DEFINE_BITWISE_BINMTX_OPERATOR(operator&, std::bit_and<>)   // m1 & m2
DEFINE_BITWISE_BINMTX_OPERATOR(operator|, std::bit_or<>)    // m1 | m2
DEFINE_BITWISE_BINMTX_OPERATOR(operator^, std::bit_xor<>)   // m1 ^ m2

// dense-block of binary matrix
template <int BlockRows, int BlockCols, typename XprTypeNested>
class BinMtxBlock : public BinMtxBase<BlockRows, BlockCols, BinMtxBlock<BlockRows, BlockCols, XprTypeNested>> {
   public:
    using XprType = BinMtxBlock<BlockRows, BlockCols, XprTypeNested>;
    using Base = BinMtxBase<BlockRows, BlockCols, XprType>;
    using BitPackType = typename Base::BitPackType;
    static constexpr std::size_t PackSize = Base::PackSize;   // number of bits in a packet
    static constexpr int NestAsRef = 0;   // whether to store this node by reference or by copy in an expression
    using Base::n_cols_;
    using Base::n_rows_;
    // row/column constructor
    BinMtxBlock(XprTypeNested& xpr, std::size_t i) :
        Base(BlockRows == 1 ? 1 : xpr.rows(), BlockCols == 1 ? 1 : xpr.cols()),
        xpr_(xpr),
        start_row_(BlockRows == 1 ? i : 0),
        start_col_(BlockCols == 1 ? i : 0) {
        fdapde_static_assert(BlockRows == 1 || BlockCols == 1, THIS_METHOD_IS_ONLY_FOR_ROW_AND_COLUMN_BLOCKS);
        fdapde_assert(i >= 0 && ((BlockRows == 1 && i < xpr_.rows()) || (BlockCols == 1 && i < xpr_.cols())));
    }
    // fixed-sized constructor
    BinMtxBlock(XprTypeNested& xpr, std::size_t start_row, std::size_t start_col) :
        Base(BlockRows, BlockCols), xpr_(xpr), start_row_(start_row), start_col_(start_col) {
        fdapde_static_assert(BlockRows != Dynamic && BlockCols != Dynamic, THIS_METHOD_IS_ONLY_FOR_FIXED_SIZE);
        fdapde_assert(
          start_row_ >= 0 && BlockRows >= 0 && start_row_ + BlockRows <= xpr_.rows() && start_col_ >= 0 &&
          BlockCols >= 0 && start_col_ + BlockCols <= xpr_.cols());
    }
    // dynamic-sized constructor
    BinMtxBlock(
      XprTypeNested& xpr, std::size_t start_row, std::size_t start_col, std::size_t block_rows,
      std::size_t block_cols) :
        Base(block_rows, block_cols), xpr_(xpr), start_row_(start_row), start_col_(start_col) {
        fdapde_assert(BlockRows == Dynamic || BlockCols == Dynamic);
        fdapde_assert(
          start_row_ >= 0 && start_row_ + block_rows <= xpr_.rows() && start_col_ >= 0 &&
          start_col_ + block_cols <= xpr_.cols());
    }

    bool operator()(std::size_t i, std::size_t j) const {
        fdapde_assert(i < n_rows_ && j < n_cols_);
        return xpr_(i + start_row_, j + start_col_);
    }
    void set(std::size_t i, std::size_t j) { xpr_.set(i + start_row_, j + start_col_); }
    void set() {   // sets all coeffients in the block
        for (std::size_t i = 0; i < n_rows_; ++i) {
            for (std::size_t j = 0; j < n_cols_; ++j) { set(i, j); }
        }
    }
    void clear(std::size_t i, std::size_t j) { xpr_.clear(i + start_row_, j + start_col_); }
    void clear() {   // clears all coeffients in the block
        for (std::size_t i = 0; i < n_rows_; ++i) {
            for (std::size_t j = 0; j < n_cols_; ++j) { clear(i, j); }
        }
    }
    BitPackType bitpack(std::size_t i) const {
        BitPackType out = 0x0;
        // compute first (row,column) index of the bitpack
        std::size_t col_offset_ = start_col_ + (i == 0 ? 0 : (n_cols_ - (n_cols_ - ((i * PackSize) % n_cols_))));
        std::size_t row_offset_ = start_row_ + (std::floor(i * (PackSize / (double)n_cols_)));
        // assembly block bitpack
        BitPackType mask = 0x1;
        for (std::size_t j = 0; j < PackSize && i * PackSize + j < Base::size(); ++j) {
            out |= (mask & xpr_(row_offset_ + (j / n_cols_), col_offset_ + (j % n_cols_))) << j;
        }
        return out;
    }
   private:
    // internal data
    std::size_t start_row_, start_col_;
    typename internals::ref_select<XprTypeNested>::type xpr_;
};

// visitors
// performs a linear bitpack visit of the binary expression
template <typename XprType, typename Visitor> struct linear_bitpack_visit {
    static constexpr std::size_t PackSize = XprType::PackSize;
    // apply visitor by cycling over bitpacks
    static inline void run(const XprType& xpr, Visitor& visitor) {
        std::size_t size = xpr.size();
        if (size == 0) return;
        if (size < PackSize) {
            visitor.apply(xpr.bitpack(0), size);
            return;
        }
        std::size_t k = 0, i = 0;   // k: current bitpack, i: maximum coefficient index processed
        for (; i + PackSize - 1 < size; i += PackSize) {
            visitor.apply(xpr.bitpack(k));
            if (visitor) return;
            k++;
        }
        if (i < size) visitor.apply(xpr.bitpack(k), size - i);
        return;
    };
};

// performs a linear bit visit of the binary expression
template <typename XprType, typename Visitor> struct linear_bit_visit {
    using BitPackType = typename XprType::BitPackType;
    static constexpr std::size_t PackSize = XprType::PackSize;
    // apply visitor bit by bit
    static inline void run(const XprType& xpr, Visitor& visitor) {
        std::size_t size = xpr.size(), n_bitpacks = xpr.bitpacks();
        if (size == 0) return;
        for (std::size_t k = 0, i = 0; k < xpr.bitpacks(); k++) {
            // bitpack evaluation
            BitPackType bitpack = xpr.bitpack(k);
            if (i + PackSize < size) {   // cycle over entire bitpack
                for (std::size_t h = 0; h < PackSize; ++h) {
                    visitor.apply((bitpack & 1) == 1);   // provide LSB of bitpack
                    bitpack = bitpack >> 1;
                }
                i += PackSize;
            } else {   // last bitpack
                for (; i < size; ++i) {
                    visitor.apply((bitpack & 1) == 1);
                    bitpack = bitpack >> 1;
                }
            }
        }
        return;
    };
};

// evaluates true if all coefficients in the binary expression are true
template <typename XprType> struct all_visitor {
    using BitPackType = typename XprType::BitPackType;
    static constexpr std::size_t PackSize = XprType::PackSize;   // number of bits in a packet
    bool res = true;
    inline void apply(BitPackType b) { res &= (~b == 0); }
    inline void apply(BitPackType b, std::size_t size) { res &= (~((((BitPackType)1 << (size - 1)) - 1) | b) == 0); }
    operator bool() const { return res == false; }   // stop if already false
};
// evaluates true if at least one coefficient in the binary expression is true
template <typename XprType> struct any_visitor {
    using BitPackType = typename XprType::BitPackType;
    static constexpr std::size_t PackSize = XprType::PackSize;   // number of bits in a packet
    bool res = false;
    inline void apply(BitPackType b) { res |= (b != 0); }
    inline void apply(BitPackType b, std::size_t size) { res |= (((~(BitPackType)0 >> (PackSize - size)) & b) != 0); }
    operator bool() const { return res == true; }   // stop if already true
};
// counts the number of true coefficients in a binary expression
template <typename XprType> struct count_visitor {
    std::size_t res = 0;
    inline void apply(bool b) {
        if (b) res++;
    }
};
  
// a non-writable expression of a block-repeat operation
template <int Rows, int Cols, typename XprTypeNested>
class BinMtxBlockRepeatOp : public BinMtxBase<Rows, Cols, BinMtxBlockRepeatOp<Rows, Cols, XprTypeNested>> {
   public:
    using XprType = BinMtxBlockRepeatOp<Rows, Cols, XprTypeNested>;
    using Base = BinMtxBase<Rows, Cols, XprType>;
    using BitPackType = typename Base::BitPackType;
    static constexpr std::size_t PackSize = Base::PackSize;   // number of bits in a packet
    static constexpr int NestAsRef = 0;   // whether to store this node by reference or by copy in an expression
    using Base::n_cols_;
    using Base::n_rows_;

    BinMtxBlockRepeatOp(const XprTypeNested& xpr, std::size_t rep_row, std::size_t rep_col) :
        Base(xpr.rows() * rep_row, xpr.cols() * rep_col), xpr_(xpr), rep_row_(rep_row), rep_col_(rep_col) { }
    bool operator()(std::size_t i, std::size_t j) const {
        fdapde_assert(i < n_rows_ && j < n_cols_);
        return xpr_(i % xpr_.rows(), j % xpr_.cols());
    }
    BitPackType bitpack(std::size_t i) const {
        BitPackType out = 0x0;
        for (std::size_t j = 0; j < PackSize && i * PackSize + j < Base::size(); ++j) {
            out |=
              ((BitPackType)1 & xpr_(((i * PackSize + j) / n_cols_) % xpr_.rows(), (i * PackSize + j) % xpr_.cols()))
              << j;
        }
        return out;
    }
   private:
    // internal data
    std::size_t rep_row_, rep_col_;
    typename internals::ref_select<const XprTypeNested>::type xpr_;
};

// base class of any binary matrix expression
template <int Rows, int Cols, typename XprType> class BinMtxBase {
   protected:
    std::size_t n_rows_ = 0, n_cols_ = 0;
    std::size_t n_bitpacks_ = 0;   // number of required bitpacks
   public:
    using Scalar = bool;
    using BitPackType = std::uintmax_t;
    static constexpr std::size_t PackSize = sizeof(BitPackType) * 8;   // number of bits in a packet

    BinMtxBase() = default;
    BinMtxBase(std::size_t n_rows, std::size_t n_cols) :
        n_rows_(n_rows), n_cols_(n_cols), n_bitpacks_(1 + std::ceil((n_rows_ * n_cols_) / PackSize)) {};
    // getters
    inline std::size_t rows() const { return n_rows_; }
    inline std::size_t cols() const { return n_cols_; }
    inline std::size_t bitpacks() const { return n_bitpacks_; }
    inline std::size_t size() const { return n_rows_ * n_cols_; }
    XprType& get() { return static_cast<XprType&>(*this); }
    const XprType& get() const { return static_cast<const XprType&>(*this); }
    // access operator on base type E
    bool operator()(std::size_t i, std::size_t j) const {
        fdapde_assert(i < n_rows_ && j < n_cols_);
        return get().operator()(i, j);
    }
    // access to i-th bitpack of the expression
    BitPackType bitpack(std::size_t i) const { return get().bitpack(i); }
    // send matrix to out stream
    friend std::ostream& operator<<(std::ostream& out, const BinMtxBase& m) {
        // assign to temporary (triggers fast bitwise evaluation)
        BinaryMatrix<Rows, Cols> tmp;
        tmp = m;
        for (std::size_t i = 0; i < tmp.rows() - 1; ++i) {
            for (std::size_t j = 0; j < tmp.cols(); ++j) { out << tmp(i, j); }
            out << "\n";
        }
        // print last row without carriage return
        for (std::size_t j = 0; j < tmp.cols(); ++j) { out << tmp(tmp.rows() - 1, j); }
        return out;
    }
    // expression bitwise NOT
    BinMtxUnaryOp<Rows, Cols, XprType, std::bit_not<>, std::logical_not<>> operator~() const {
        return BinMtxUnaryOp<Rows, Cols, XprType, std::bit_not<>, std::logical_not<>>(
          get(), std::bit_not<>(), std::logical_not<>());
    }
    // block-type indexing
    BinMtxBlock<1, Cols, XprType> row(std::size_t row) { return BinMtxBlock<1, Cols, XprType>(get(), row); }
    BinMtxBlock<1, Cols, const XprType> row(std::size_t row) const {
        return BinMtxBlock<1, Cols, const XprType>(get(), row);
    }
    BinMtxBlock<Rows, 1, XprType> col(std::size_t col) { return BinMtxBlock<Rows, 1, XprType>(get(), col); }
    BinMtxBlock<Rows, 1, const XprType> col(std::size_t col) const {
        return BinMtxBlock<Rows, 1, const XprType>(get(), col);
    }
    template <int Rows_, int Cols_>
    BinMtxBlock<Rows_, Cols_, XprType> block(std::size_t start_row, std::size_t start_col) {
        return BinMtxBlock<Rows_, Cols_, XprType>(get(), start_row, start_col);
    }
    BinMtxBlock<Dynamic, Dynamic, XprType>
    block(std::size_t start_row, std::size_t start_col, std::size_t block_rows, std::size_t block_cols) {
        return BinMtxBlock<Dynamic, Dynamic, XprType>(get(), start_row, start_col, block_rows, block_cols);
    }
    // visitors support
    inline bool all() const { return visit_apply_<all_visitor<XprType>, linear_bitpack_visit>(); }
    inline bool any() const { return visit_apply_<any_visitor<XprType>, linear_bitpack_visit>(); }
    inline std::size_t count() const { return visit_apply_<count_visitor<XprType>, linear_bit_visit>(); }
    // selection on eigen expressions
    template <typename ExprType>
    DMatrix<typename ExprType::Scalar> select(const Eigen::MatrixBase<ExprType>& mtx) const {
        fdapde_assert(n_rows_ == mtx.rows() && n_cols_ == mtx.cols());
        using Scalar_ = typename ExprType::Scalar;
        DMatrix<Scalar_> masked_mtx = mtx;   // assign to dense storage
        for (std::size_t i = 0; i < mtx.rows(); ++i)
            for (std::size_t j = 0; j < mtx.cols(); ++j) {
                if (!get().operator()(i, j)) masked_mtx(i, j) = 0;
            }
        return std::move(masked_mtx);
    }
    template <typename ExprType>
    SpMatrix<typename ExprType::Scalar> select(const Eigen::SparseMatrixBase<ExprType>& mtx) const {
        fdapde_assert(n_rows_ == mtx.rows() && n_cols_ == mtx.cols());
        using Scalar_ = typename ExprType::Scalar;
        SpMatrix<Scalar_> masked_mtx = mtx;   // assign to sparse storage
        for (int k = 0; k < masked_mtx.outerSize(); ++k)
            for (typename SpMatrix<Scalar_>::InnerIterator it(masked_mtx, k); it; ++it) {
                if (!get().operator()(it.row(), it.col())) { it.valueRef() = 0; }
            }
        return std::move(masked_mtx);
    }
    // block-repeat operation
    BinMtxBlockRepeatOp<Dynamic, Dynamic, XprType> blk_repeat(std::size_t rep_row, std::size_t rep_col) const {
        return BinMtxBlockRepeatOp<Dynamic, Dynamic, XprType>(get(), rep_row, rep_col);
    }
   private:
    template <typename Visitor, template <typename, typename> typename VisitStrategy> inline auto visit_apply_() const {
        Visitor visitor;
        VisitStrategy<XprType, Visitor>::run(get(), visitor);
        return visitor.res;
    }
};
  
// comparison operator
template <int Rows1, int Cols1, typename XprType1, int Rows2, int Cols2, typename XprType2>
bool operator==(const BinMtxBase<Rows1, Cols1, XprType1>& op1, const BinMtxBase<Rows2, Cols2, XprType2>& op2) {
    fdapde_static_assert(
      !(Rows1 != Dynamic && Cols1 != Dynamic && Rows2 != Dynamic && Cols2 != Dynamic) ||
        (Rows1 == Cols1 && Rows2 == Cols2),
      YOU_MIXED_MATRICES_OF_DIFFERENT_SIZE);
    fdapde_assert(op1.rows() == op2.rows() && op1.cols() == op2.cols());
    using BitPackType = typename XprType1::BitPackType;
    static constexpr int PackSize = XprType1::PackSize;
    bool result = true;
    std::size_t n = op1.bitpacks() - 1;
    for (std::size_t i = 0; i < n && result; i++) { result &= (op1.bitpack(i) == op2.bitpack(i)); }
    // process last bitpack
    BitPackType mask = ~(BitPackType)0 >> (PackSize - (op1.size() - PackSize * n));
    result &= ((mask & op1.bitpack(n)) == (mask & op2.bitpack(n)));
    return result;
}
template <int Rows1, int Cols1, typename XprType1, int Rows2, int Cols2, typename XprType2>
bool operator!=(const BinMtxBase<Rows1, Cols1, XprType1>& op1, const BinMtxBase<Rows2, Cols2, XprType2>& op2) {
    fdapde_static_assert(
      !(Rows1 != Dynamic && Cols1 != Dynamic && Rows2 != Dynamic && Cols2 != Dynamic) ||
        (Rows1 == Cols1 && Rows2 == Cols2),
      YOU_MIXED_MATRICES_OF_DIFFERENT_SIZE);
    fdapde_assert(op1.rows() == op2.rows() && op1.cols() == op2.cols());
    using BitPackType = typename XprType1::BitPackType;
    static constexpr int PackSize = XprType1::PackSize;    
    bool result = false;
    std::size_t n = op1.bitpacks() - 1;
    for (std::size_t i = 0; i < n && !result; i++) { result |= (op1.bitpack(i) != op2.bitpack(i)); }
    // process last bitpack
    BitPackType mask = ~(BitPackType)0 >> (PackSize - (op1.size() - PackSize * n));
    result |= ((mask & op1.bitpack(n)) != (mask & op2.bitpack(n)));    
    return result;
}

// alias export for binary vectors
template <int Rows> using BinaryVector = BinaryMatrix<Rows, 1>;

}   // namespace core
}   // namespace fdapde

#endif   // __BINARY_MATRIX_H__
