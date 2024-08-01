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

#ifndef __MATRIX_FIELD_H__
#define __MATRIX_FIELD_H__

#include <array>
#include <type_traits>

#include "../utils/symbols.h"
#include "../linear_algebra/constexpr_matrix.h"
#include "norm.h"
#include "divergence.h"
#include "dot.h"

namespace fdapde {
  
template <int Size, typename Derived> class MatrixBase;

template <typename Lhs, typename Rhs>
class MatrixProduct : public fdapde::MatrixBase<Lhs::StaticInputSize, MatrixProduct<Lhs, Rhs>> {
    fdapde_static_assert(
      Lhs::StaticInputSize == Dynamic || Rhs::StaticInputSize == Dynamic ||
        (Lhs::StaticInputSize != Dynamic && Rhs::StaticInputSize != Dynamic &&
         Lhs::StaticInputSize == Rhs::StaticInputSize),
      YOU_MIXED_MATRICES_WITH_DIFFERENT_INPUT_SIZES);
    fdapde_static_assert(
      std::is_same_v<typename Lhs::InputType FDAPDE_COMMA typename Rhs::InputType>,
      YOU_MIXED_MATRICES_WITH_DIFFERENT_INPUT_TYPES);
   public:
    using Base = MatrixBase<Lhs::StaticInputSize, MatrixProduct<Lhs, Rhs>>;
    using InputType = typename Lhs::InputType;
    using Scalar = decltype(std::declval<typename Lhs::Scalar>() * std::declval<typename Rhs::Scalar>());
    static constexpr int StaticInputSize = Lhs::StaticInputSize;
    static constexpr int Rows = Lhs::Rows;
    static constexpr int Cols = Rhs::Cols;
    static constexpr int NestAsRef = 0;
    using Base::operator();

    constexpr MatrixProduct(const Lhs& lhs, const Rhs& rhs) requires(Rows != Dynamic && Cols != Dynamic)
        : Base(), lhs_(lhs), rhs_(rhs) {
        fdapde_static_assert(Lhs::Cols == Rhs::Rows, INVALID_OPERAND_SIZES_FOR_MATRIX_PRODUCT);
    }
    MatrixProduct(const Lhs& lhs, const Rhs& rhs) requires(Rows == Dynamic || Cols == Dynamic)
        : Base(), lhs_(lhs), rhs_(rhs) {
        fdapde_assert(lhs_.cols() == rhs_.rows() && lhs_.input_size() == rhs_.input_size());
    }
    constexpr int rows() const { return lhs_.rows(); }
    constexpr int cols() const { return rhs_.cols(); }
    constexpr int input_size() const { lhs_.input_size(); }
    constexpr int size() const { return rows() * cols(); }

    // for matrix multiplication, it is more convenient to evaluate the two operands at p, and take the product of the
    // evaluations
    template <typename VectorType, typename Dest> constexpr void eval_at(const VectorType& p, Dest& dest) const {
        fdapde_static_assert(
          std::is_invocable_v<Dest FDAPDE_COMMA int FDAPDE_COMMA int> ||
            fdapde::is_subscriptable<Dest FDAPDE_COMMA int>,
          DESTINATION_TYPE_MUST_EITHER_EXPOSE_A_MATRIX_LIKE_ACCESS_OPERATOR_OR_A_SUBSCRIPT_OPERATOR);

        // store evaluations in temporaries (O(n) function calls)
        using LhsStorageType = std::conditional_t<
          Lhs::Rows == Dynamic || Lhs::Cols == Dynamic, std::vector<Scalar>,
          std::array<Scalar, int(Lhs::Rows) * int(Lhs::Cols)>>;
        using RhsStorageType = std::conditional_t<
          Rhs::Rows == Dynamic || Rhs::Cols == Dynamic, std::vector<Scalar>,
          std::array<Scalar, int(Rhs::Rows) * int(Rhs::Cols)>>;
        LhsStorageType lhs_temp;
        RhsStorageType rhs_temp;
        if constexpr (Lhs::Rows == Dynamic || Lhs::Cols == Dynamic) lhs_temp.resize(lhs_.size());
        if constexpr (Rhs::Rows == Dynamic || Rhs::Cols == Dynamic) rhs_temp.resize(rhs_.size());
        for (int i = 0; i < lhs_.rows(); ++i) {
            for (int j = 0; j < lhs_.cols(); ++j) { lhs_temp[i * lhs_.cols() + j] = lhs_.eval(i, j, p); }
        }
        for (int i = 0; i < rhs_.cols(); ++i) {   // store col major to exploit cache locality in matrix product
            for (int j = 0; j < rhs_.rows(); ++j) { rhs_temp[i * rhs_.rows() + j] = rhs_.eval(j, i, p); }
        }
        // perform standard matrix-matrix product
        for (int i = 0; i < rows(); ++i) {
            for (int j = 0; j < cols(); ++j) {
                Scalar res = 0;
                for (int k = 0; k < lhs_.cols(); ++k) {
                    res += lhs_temp[i * lhs_.cols() + k] * rhs_temp[j * rhs_.rows() + k];
		}
                if constexpr (std::is_invocable_v<Dest, int, int>) {
                    dest(i, j) = res;
                } else {
                    dest[i * cols() + j] = res;
                }
            }
	}
        return;
    }  
    constexpr auto operator()(int i, int j) const {
        return [i, j, this](const InputType& p) {
            Scalar res = 0;
            for (int k = 0; k < lhs_.cols(); ++k) { res += lhs_.eval(i, k, p) * rhs_.eval(k, j, p); }
            return res;
        };
    }
    constexpr Scalar eval(int i, int j, const InputType& p) const {
        Scalar res = 0;
        for (int k = 0; k < lhs_.cols(); ++k) { res += lhs_.eval(i, k, p) * rhs_.eval(k, j, p); }
        return res;
    }
    template <typename... Args> constexpr MatrixProduct<Lhs, Rhs>& forward(Args&&... args) {
        lhs_.forward(std::forward<Args>(args)...);
        rhs_.forward(std::forward<Args>(args)...);
        return *this;
    }
   protected:
    typename internals::ref_select<const Lhs>::type lhs_;
    typename internals::ref_select<const Rhs>::type rhs_;
};
template <typename Lhs, typename Rhs>
constexpr MatrixProduct<Lhs, Rhs>
operator*(const MatrixBase<Lhs::StaticInputSize, Lhs>& lhs, const MatrixBase<Rhs::StaticInputSize, Rhs>& rhs) {
    return MatrixProduct<Lhs, Rhs> {lhs.derived(), rhs.derived()};
}

template <int BlockRows_, int BlockCols_, typename Derived>
class MatrixBlock : public fdapde::MatrixBase<Derived::StaticInputSize, MatrixBlock<BlockRows_, BlockCols_, Derived>> {
    fdapde_static_assert(
      (BlockRows_ == Dynamic || (Derived::Rows == Dynamic || (BlockRows_ > 0 && BlockRows_ <= Derived::Rows))) &&
        (BlockCols_ == Dynamic || (Derived::Cols == Dynamic || (BlockCols_ > 0 && BlockCols_ <= Derived::Cols))),
      INVALID_BLOCK_SIZES);
   public:
    using Base = MatrixBase<Derived::StaticInputSize, MatrixBlock<BlockRows_, BlockCols_, Derived>>;
    using InputType = typename Derived::InputType;
    using Scalar = typename Derived::Scalar;
    static constexpr int StaticInputSize = Derived::StaticInputSize;
    static constexpr int Rows = BlockRows_;
    static constexpr int Cols = BlockCols_;
    static constexpr int NestAsRef = 0;
    static constexpr int ReadOnly = Derived::ReadOnly;
    using Base::operator();

    // row/column constructor
    constexpr MatrixBlock(const Derived& xpr, int i) :
        Base(),
        xpr_(xpr),
        start_row_(BlockRows_ == 1 ? i : 0),
        start_col_(BlockCols_ == 1 ? i : 0),
        block_rows_(BlockRows_ == 1 ? 1 : xpr.rows()),
        block_cols_(BlockCols_ == 1 ? 1 : xpr.cols()) {
        fdapde_static_assert(BlockRows_ == 1 || BlockCols_ == 1, THIS_METHOD_IS_ONLY_FOR_ROW_AND_COLUMN_BLOCKS);
        fdapde_constexpr_assert(
          i >= 0 && ((BlockRows_ == 1 && i < xpr_.rows()) || (BlockCols_ == 1 && i < xpr_.cols())));
    }
    constexpr MatrixBlock(const Derived& xpr, int start_row, int start_col) :
        Base(), xpr_(xpr), start_row_(start_row), start_col_(start_col), block_rows_(BlockRows_),
        block_cols_(BlockCols_) {
        fdapde_static_assert(
          BlockRows_ != Dynamic && BlockCols_ != Dynamic, THIS_METHOD_IS_ONLY_FOR_STATIC_SIZED_BLOCKS);
        fdapde_constexpr_assert(
          start_row_ >= 0 && start_row_ + BlockRows_ <= xpr_.rows() && start_col_ >= 0 &&
          start_col_ + BlockCols_ <= xpr_.cols());
    }
    MatrixBlock(const Derived& xpr, int start_row, int start_col, int block_rows, int block_cols) :
        Base(), xpr_(xpr), start_row_(start_row), start_col_(start_col), block_rows_(block_rows),
        block_cols_(block_cols) {
        fdapde_static_assert(
          BlockRows_ == Dynamic || BlockCols_ == Dynamic, THIS_METHOD_IS_ONLY_FOR_DYNAMIC_SIZED_BLOCKS);
        fdapde_assert(
          start_row_ >= 0 && start_row_ + block_rows <= xpr_.rows() && start_col_ >= 0 &&
          start_col_ + block_cols <= xpr_.cols());
    }

    constexpr int rows() const { return block_rows_; }
    constexpr int cols() const { return block_cols_; }
    constexpr int input_size() const { return xpr_.input_size(); }
    constexpr int size() const { return rows() * cols(); }
    constexpr auto operator()(int i, int j) const { return xpr_(start_row_ + i, start_col_ + j); }
    constexpr auto operator[](int i) const {
        fdapde_static_assert(BlockRows_ == 1 || BlockCols_ == 1, THIS_METHOD_IS_ONLY_FOR_ROW_OR_COLUMN_BLOCKS);
        if constexpr (Rows == 1) return xpr_(start_row_, start_col_ + i);
        if constexpr (Cols == 1) return xpr_(start_row_ + i, start_col_);
    }
    constexpr Scalar eval(int i, int j, const InputType& p) const {
        return xpr_.eval(start_row_ + i, start_col_ + j, p);
    }
    constexpr Scalar eval(int i, const InputType& p) const {
        fdapde_static_assert(BlockRows_ == 1 || BlockCols_ == 1, THIS_METHOD_IS_ONLY_FOR_ROW_OR_COLUMN_BLOCKS);
        if constexpr (Rows == 1) return xpr_.eval(start_row_, start_col_ + i, p);
        if constexpr (Cols == 1) return xpr_.eval(start_row_ + i, start_col_, p);
    }
    template <typename... Args> constexpr MatrixBlock<BlockRows_, BlockCols_, Derived>& forward(Args&&... args) {
        xpr_.forward(std::forward<Args>(args)...);
        return *this;
    }
    // block assignment
    template <int Size_, typename RhsDerived>
    constexpr MatrixBlock<BlockRows_, BlockCols_, Derived>& operator=(const MatrixBase<Size_, RhsDerived>& rhs)
        requires(BlockRows_ != Dynamic && BlockCols_ != Dynamic) {
        fdapde_static_assert(Derived::ReadOnly != 0, BLOCK_ASSIGNMENT_TO_A_READ_ONLY_EXPRESSION_IS_INVALID);
        fdapde_static_assert(
          RhsDerived::Rows == BlockRows_ && RhsDerived::Cols == BlockCols_ &&
            std::is_convertible_v<typename RhsDerived::FunctorType FDAPDE_COMMA typename Derived::FunctorType>,
          INVALID_BLOCK_SIZE_OR_YOU_ARE_TRYING_TO_ASSIGN_A_BLOCK_WITH_NON_CONVERTIBLE_FUNCTOR_TYPE);
        for (int i = 0; i < xpr_.rows(); ++i) {
            for (int j = 0; j < xpr_.cols(); ++j) { xpr_(start_row_ + i, start_col_ + j) = rhs(i, j); }
        }
	return *this;
    }
    template <int Size_, typename RhsDerived>
    MatrixBlock<BlockRows_, BlockCols_, Derived>& operator=(const MatrixBase<Size_, RhsDerived>& rhs)
        requires(BlockRows_ == Dynamic || BlockCols_ == Dynamic) {
        fdapde_static_assert(Derived::ReadOnly != 0, BLOCK_ASSIGNMENT_TO_A_READ_ONLY_EXPRESSION_IS_INVALID);
        fdapde_static_assert(
          std::is_convertible_v<typename RhsDerived::FunctorType FDAPDE_COMMA typename Derived::FunctorType>,
          YOU_ARE_TRYING_TO_ASSIGN_A_BLOCK_WITH_NON_CONVERTIBLE_COEFFICIENT_TYPE);
        fdapde_assert(rhs.rows() == xpr_.rows() && rhs.cols() == xpr_.cols());
        for (int i = 0; i < xpr_.rows(); ++i) {
            for (int j = 0; j < xpr_.cols(); ++j) { xpr_(start_row_ + i, start_col_ + j) = rhs(i, j); }
        }
	return *this;
    }  
   private:
    int start_row_ = 0, start_col_ = 0;
    int block_rows_ = 0, block_cols_ = 0;
    typename internals::ref_select<Derived>::type xpr_;
};

template <typename Lhs, typename Rhs, typename BinaryOperation>
class MatrixBinOp : public fdapde::MatrixBase<Lhs::StaticInputSize, MatrixBinOp<Lhs, Rhs, BinaryOperation>> {
    fdapde_static_assert(
      (Lhs::StaticInputSize == Dynamic || Rhs::StaticInputSize == Dynamic ||
       Lhs::StaticInputSize == Rhs::StaticInputSize) &&
        (Lhs::Rows == Dynamic || Rhs::Rows == Dynamic || Lhs::Rows == Rhs::Rows) &&
        (Lhs::Cols == Dynamic || Rhs::Cols == Dynamic || Lhs::Cols == Rhs::Cols),
      YOU_MIXED_MATRIX_FIELDS_OF_DIFFERENT_SIZES);
    fdapde_static_assert(
      std::is_same_v<typename Lhs::InputType FDAPDE_COMMA typename Rhs::InputType>,
      YOU_MIXED_MATRIX_FIELDS_WITH_DIFFERENT_INPUT_TYPE);
    typename internals::ref_select<const Lhs>::type lhs_;
    typename internals::ref_select<const Rhs>::type rhs_;
    BinaryOperation op_;
   public:
    using Base = MatrixBase<Lhs::StaticInputSize, MatrixBinOp<Lhs, Rhs, BinaryOperation>>;
    using InputType = typename Lhs::InputType;
    using Scalar = decltype(std::declval<BinaryOperation>().operator()(
      std::declval<typename Lhs::Scalar>(), std::declval<typename Rhs::Scalar>()));
    static constexpr int StaticInputSize = Lhs::StaticInputSize;
    static constexpr int Rows = Lhs::Rows;
    static constexpr int Cols = Lhs::Cols;
    static constexpr int NestAsRef = 0;
    static constexpr int ReadOnly = 1;
    using Base::operator();

    constexpr MatrixBinOp(const Lhs& lhs, const Rhs& rhs, BinaryOperation op)
        requires(StaticInputSize != Dynamic && Rows != Dynamic && Cols != Dynamic)
        : Base(), lhs_(lhs), rhs_(rhs), op_(op) { }
    MatrixBinOp(const Lhs& lhs, const Rhs& rhs, BinaryOperation op)
        requires(StaticInputSize == Dynamic || Rows == Dynamic || Cols == Dynamic)
        : lhs_(lhs), rhs_(rhs), op_(op) {
        fdapde_assert(
          (lhs.input_size() == rhs.input_size()) && (Rows != Dynamic || lhs.rows() == rhs.rows()) &&
          (Cols != Dynamic || lhs.cols() == rhs.cols()));
    }

    constexpr Scalar eval(int i, int j, const InputType& p) const {
        return op_(lhs_.eval(i, j, p), rhs_.eval(i, j, p));
    }
    constexpr Scalar eval(int i, const InputType& p) const {
        fdapde_static_assert(Rows == 1 || Cols == 1, THIS_METHOD_IS_ONLY_FOR_ROW_OR_COLUMN_MATRICES);
        return op_(lhs_.eval(i, p), rhs_.eval(i, p));
    }
    constexpr auto operator()(int i, int j) const {
        return [i, j, this](const InputType& p) { return op_(lhs_.eval(i, j, p), rhs_.eval(i, j, p)); };
    }
    constexpr Scalar operator[](int i) const {
        fdapde_static_assert(Rows == 1 || Cols == 1, THIS_METHOD_IS_ONLY_FOR_ROW_OR_COLUMN_MATRICES);
	return [i, this](const InputType& p) { return op_(lhs_.eval(i, p), rhs_.eval(i, p)); };;
    }  
    template <typename... Args> constexpr MatrixBinOp<Lhs, Rhs, BinaryOperation>& forward(Args&&... args) {
        lhs_.forward(std::forward<Args>(args)...);
        rhs_.forward(std::forward<Args>(args)...);
        return *this;
    }
    constexpr int rows() const { return lhs_.rows(); }
    constexpr int cols() const { return lhs_.cols(); }
    constexpr int input_size() const { return lhs_.input_size(); }
    constexpr int size() const { return lhs_.size(); }
};
template <typename Lhs, typename Rhs>
constexpr MatrixBinOp<Lhs, Rhs, std::plus<>>
operator+(const MatrixBase<Lhs::StaticInputSize, Lhs>& lhs, const MatrixBase<Rhs::StaticInputSize, Rhs>& rhs) {
    return MatrixBinOp<Lhs, Rhs, std::plus<>>{lhs.derived(), rhs.derived(), std::plus<>()};
}
template <typename Lhs, typename Rhs>
constexpr MatrixBinOp<Lhs, Rhs, std::minus<>>
operator-(const MatrixBase<Lhs::StaticInputSize, Lhs>& lhs, const MatrixBase<Rhs::StaticInputSize, Rhs>& rhs) {
    return MatrixBinOp<Lhs, Rhs, std::minus<>>{lhs.derived(), rhs.derived(), std::minus<>()};
}

template <typename Lhs, typename Rhs, typename BinaryOperation>
class MatrixCoeffWiseOp :
    public fdapde::MatrixBase<
      std::conditional_t<std::is_arithmetic_v<Lhs>, Rhs, Lhs>::StaticInputSize,
      MatrixCoeffWiseOp<Lhs, Rhs, BinaryOperation>> {
    using CoeffType_ = std::conditional_t<std::is_arithmetic_v<Lhs>, Lhs, Rhs>;
    using Derived = std::conditional_t<std::is_arithmetic_v<Lhs>, Rhs, Lhs>;
    typename internals::ref_select<const Derived>::type xpr_;
    CoeffType_ coeff_;
    BinaryOperation op_;
   public:
    static constexpr int StaticInputSize = Derived::StaticInputSize;
    using Base = MatrixBase<StaticInputSize, MatrixCoeffWiseOp<Lhs, Rhs, BinaryOperation>>;
    using Scalar = decltype(std::declval<BinaryOperation>().operator()(
      std::declval<typename Derived::Scalar>(), std::declval<CoeffType_>()));
    using InputType = typename Derived::InputType;
    static constexpr int Rows = Derived::Rows;
    static constexpr int Cols = Derived::Cols;
    static constexpr int NestAsRef = 0;
    static constexpr int ReadOnly = 1;
    using Base::operator();

    constexpr MatrixCoeffWiseOp(const Derived& xpr, CoeffType_ coeff, BinaryOperation op)
        : Base(), xpr_(xpr), coeff_(coeff), op_(op) { }
    constexpr Scalar eval(int i, int j, const InputType& p) const { return op_(xpr_.eval(i, j, p), coeff_); }
    constexpr Scalar eval(int i, const InputType& p) const {
        fdapde_static_assert(Rows == 1 || Cols == 1, THIS_METHOD_IS_ONLY_FOR_ROW_OR_COLUMN_MATRICES);
        return op_(xpr_.eval(i, p), coeff_);
    }
    constexpr auto operator()(int i, int j) const {
        return [i, j, this](const InputType& p) { return op_(xpr_.eval(i, j, p), coeff_); };
    }
    constexpr Scalar operator[](int i) const {
        fdapde_static_assert(Rows == 1 || Cols == 1, THIS_METHOD_IS_ONLY_FOR_ROW_OR_COLUMN_MATRICES);
        return [i, this](const InputType& p) { return op_(xpr_.eval(i, p), coeff_); };
    }
    template <typename... Args> constexpr MatrixCoeffWiseOp<Lhs, Rhs, BinaryOperation>& forward(Args&&... args) {
        xpr_.forward(std::forward<Args>(args)...);
        return *this;
    }
    constexpr int rows() const { return xpr_.rows(); }
    constexpr int cols() const { return xpr_.cols(); }
    constexpr int input_size() const { return xpr_.input_size(); }
    constexpr int size() const { return xpr_.size(); }
};

template <int Size, typename Derived, typename Coeff>
constexpr MatrixCoeffWiseOp<Derived, Coeff, std::multiplies<>>
operator*(const MatrixBase<Size, Derived>& lhs, Coeff rhs)
    requires(std::is_arithmetic_v<Coeff>) {
    return MatrixCoeffWiseOp<Derived, Coeff, std::multiplies<>> {lhs.derived(), rhs, std::multiplies<>()};
}
template <int Size, typename Derived, typename Coeff>
constexpr MatrixCoeffWiseOp<Coeff, Derived, std::multiplies<>>
operator*(Coeff lhs, const MatrixBase<Size, Derived>& rhs)
    requires(std::is_arithmetic_v<Coeff>) {
    return MatrixCoeffWiseOp<Coeff, Derived, std::multiplies<>> {rhs.derived(), lhs, std::multiplies<>()};
}
template <int Size, typename Derived, typename Coeff>
constexpr MatrixCoeffWiseOp<Derived, Coeff, std::divides<>> operator/(const MatrixBase<Size, Derived>& lhs, Coeff rhs)
    requires(std::is_arithmetic_v<Coeff>) {
    return MatrixCoeffWiseOp<Derived, Coeff, std::divides<>> {lhs.derived(), rhs, std::divides<>()};
}
  
template <
  int StaticInputSize_, int Rows_, int Cols_,
  typename FunctorType_ = std::function<double(static_dynamic_vector_selector_t<StaticInputSize_>)>>
class MatrixField :
    public fdapde::MatrixBase<StaticInputSize_, MatrixField<StaticInputSize_, Rows_, Cols_, FunctorType_>> {
    template <typename T> struct is_dynamic_sized {
        static constexpr bool value = (StaticInputSize_ == Dynamic || Rows_ == Dynamic || Cols_ == Dynamic);
    };
    using This = MatrixField<StaticInputSize_, Rows_, Cols_, FunctorType_>;
    using StorageType = typename std::conditional<
      is_dynamic_sized<This>::value, std::vector<FunctorType_>, std::array<FunctorType_, Rows_ * Cols_>>::type;
    using Base = MatrixBase<StaticInputSize_, MatrixField<StaticInputSize_, Rows_, Cols_, FunctorType_>>;
    using traits = fn_ptr_traits<&FunctorType_::operator()>;
    fdapde_static_assert(traits::n_args == 1, PROVIDED_FUNCTOR_MUST_ACCEPT_ONLY_ONE_ARGUMENT);

    StorageType data_;
    int inner_size_ = 0;
    int n_rows_ = 0, n_cols_ = 0;
   public:
    using FunctorType = std::decay_t<FunctorType_>;
    using InputType = std::decay_t<std::tuple_element_t<0, typename traits::ArgsType>>;
    using Scalar = typename std::invoke_result<FunctorType, InputType>::type;
    static constexpr int StaticInputSize = StaticInputSize_;   // dimensionality of base space (can be Dynamic)
    static constexpr int NestAsRef = 0;
    static constexpr int ReadOnly = 0;
    static constexpr int Rows = Rows_;
    static constexpr int Cols = Cols_;
    using Base::operator();
    // static sized constructor
    constexpr MatrixField() requires(!is_dynamic_sized<This>::value)
        : Base(), data_(), inner_size_(StaticInputSize), n_rows_(Rows), n_cols_(Cols) { }
    // dynamic sized constructor
    MatrixField() requires(is_dynamic_sized<This>::value)
        : Base(), data_(), inner_size_(0), n_rows_(0), n_cols_(0) { }
    MatrixField(int inner_size, int rows, int cols) requires(is_dynamic_sized<This>::value)
        : Base(), inner_size_(inner_size), n_rows_(rows), n_cols_(cols) {
        fdapde_assert(rows > 0 && cols > 0);
        data_.resize(rows * cols);
    }
    // vector constructor
    explicit MatrixField(int rows) requires(is_dynamic_sized<This>::value) : Base(rows, 1) {
        fdapde_static_assert(Rows == Dynamic && Cols == 1, THIS_METHOD_IS_ONLY_FOR_VECTORS);
    }
    template <int Size_, typename RhsDerived>
    explicit constexpr MatrixField(const MatrixBase<Size_, RhsDerived>& other)
        requires(!is_dynamic_sized<This>::value)
        : Base(), data_() {
        fdapde_static_assert(
          StaticInputSize == Size_ && Rows == RhsDerived::Row && Cols == RhsDerived::Cols &&
            std::is_convertible_v<typename RhsDerived::FunctorType FDAPDE_COMMA FunctorType>,
          INVALID_RHS_SIZE_OR_NON_CONVERTIBLE_FUNCTOR_TYPE);
        for (int i = 0; i < Rows; ++i) {
            for (int j = 0; j < Cols; ++j) { operator()(i, j) = other(i, j); }
        }
    }
    template <int Size_, typename RhsDerived>
    explicit MatrixField(const MatrixBase<Size_, RhsDerived>& other)
        requires(is_dynamic_sized<This>::value)
        : Base(other.rows(), other.cols()) {
        fdapde_static_assert(
          StaticInputSize == Size_ && std::is_convertible_v<typename RhsDerived::FunctorType FDAPDE_COMMA FunctorType>,
          INVALID_INPUT_SIZE_OR_NON_CONVERTIBLE_FUNCTOR_TYPE);
	fdapde_assert(rows() == other.rows() && cols() == other.cols());
        for (int i = 0; i < n_rows_; ++i) {
            for (int j = 0; j < n_cols_; ++j) { operator()(i, j) = other(i, j); }
        }
    }

    // assignment
    template <int Size_, typename RhsDerived>
    constexpr MatrixField<StaticInputSize, Rows, Cols, FunctorType>& operator=(const MatrixBase<Size_, RhsDerived>& rhs)
        requires(Rows != Dynamic && Cols != Dynamic) {
        fdapde_static_assert(
          StaticInputSize_ == Size_ && Rows == RhsDerived::Rows && Cols == RhsDerived::Cols &&
            std::is_convertible_v<typename RhsDerived::FunctorType FDAPDE_COMMA FunctorType>,
          INVALID_RHS_SIZE_OR_NON_CONVERTIBLE_RHS_FUNCTOR_TYPE_IN_ASSIGNMENT);
        for (int i = 0; i < Rows; ++i) {
            for (int j = 0; j < Cols; ++j) { operator()(i, j) = rhs.derived()(i, j); }
        }
        return *this;
    }
    template <int Size_, typename RhsDerived>
    MatrixField<StaticInputSize, Rows, Cols, FunctorType>& operator=(const MatrixBase<Size_, RhsDerived>& rhs)
        requires(StaticInputSize == Dynamic || Rows == Dynamic || Cols == Dynamic) {
        using RhsFunctorType =
          decltype(std::declval<RhsDerived>().operator()(std::declval<int>(), std::declval<int>()));
        fdapde_static_assert(
          std::is_convertible_v<RhsFunctorType FDAPDE_COMMA FunctorType>,
          NON_CONVERTIBLE_RHS_FUNCTOR_TYPE_IN_ASSIGNMENT);
        if constexpr (Rows == Dynamic) n_rows_ = rhs.derived().rows();
        if constexpr (Cols == Dynamic) n_cols_ = rhs.derived().cols();
        if constexpr (StaticInputSize == Dynamic) inner_size_ = rhs.derived().input_size();
        data_.resize(n_rows_ * n_cols_);
        for (int i = 0; i < n_cols_; ++i) {
            for (int j = 0; j < n_rows_; ++j) { operator()(i, j) = rhs.derived()(i, j); }
        }
        return *this;
    }

    void resize(int inner_size, int rows, int cols) {
        fdapde_static_assert(Rows == Dynamic || Cols == Dynamic, THIS_METHOD_IS_ONLY_FOR_DYNAMIC_SIZED_MATRICES);
        fdapde_assert(inner_size > 0 && rows > 0 && cols > 0);
        if constexpr (Rows == Dynamic) n_rows_ = rows;
        if constexpr (Cols == Dynamic) n_cols_ = cols;
        data_ = std::vector<FunctorType>(n_rows_ * n_cols_);
        if constexpr (StaticInputSize == Dynamic) inner_size_ = inner_size;
        return;
    }
    void resize(int inner_size, int size) {
        fdapde_static_assert(
          (Rows == Dynamic && Cols == 1) || (Cols == Dynamic && Rows == 1),
          THIS_METHOD_IS_ONLY_FOR_DYNAMIC_SIZED_VECTORS);
        fdapde_assert(inner_size > 0 && size > 0);
        if constexpr (Rows == Dynamic) n_rows_ = size;
        n_cols_ = 1;
        data_ = std::vector<FunctorType>(n_rows_ * n_cols_);
        if constexpr (StaticInputSize == Dynamic) inner_size_ = inner_size;
        return;
    }
    // getters
    constexpr Scalar eval(int i, int j, const InputType& p) const {
        if constexpr (is_dynamic_sized<This>::value) { fdapde_assert(p.size() == inner_size_); }
        return data_[i * cols() + j](p);
    }
    constexpr Scalar eval(int i, const InputType& p) const {
        fdapde_static_assert(Rows == 1 || Cols == 1, THIS_METHOD_IS_ONLY_FOR_VECTORS);
        if constexpr (is_dynamic_sized<This>::value) { fdapde_assert(p.size() == inner_size_); }
        return data_[i](p);
    }
    constexpr const FunctorType& operator()(int i, int j) const { return data_[i * cols() + j]; }
    constexpr FunctorType& operator()(int i, int j) { return data_[i * cols() + j]; }
    constexpr const FunctorType& operator[](int i) const {
        fdapde_static_assert(Rows == 1 || Cols == 1, THIS_METHOD_IS_ONLY_FOR_VECTORS);
        return data_[i];
    }
    constexpr FunctorType& operator[](int i) {
        fdapde_static_assert(Rows == 1 || Cols == 1, THIS_METHOD_IS_ONLY_FOR_VECTORS);
        return data_[i];
    }
    constexpr int rows() const { return n_rows_; }
    constexpr int cols() const { return n_cols_; }
    constexpr int input_size() const { return inner_size_; }
    constexpr int size() const { return n_rows_ * n_cols_; }
};

template <int StaticInputSize, int Rows, typename Derived>
using VectorField = MatrixField<StaticInputSize, Rows, 1, Derived>;

template <typename Derived>
struct MatrixTranspose : public fdapde::MatrixBase<Derived::StaticInputSize, MatrixTranspose<Derived>> {
    using Base = MatrixBase<Derived::StaticInputSize, MatrixTranspose<Derived>>;
    using Scalar = typename Derived::Scalar;
    using InputType = typename Derived::InputType;
    static constexpr int StaticInputSize = Derived::StaticInputSize;
    static constexpr int Rows = Derived::Cols;
    static constexpr int Cols = Derived::Rows;
    static constexpr int NestAsRef = 0;
    static constexpr int ReadOnly = 1;
    using Base::operator();

    constexpr explicit MatrixTranspose(const Derived& xpr) : Base(), xpr_(xpr) { }
    constexpr Scalar eval(int i, int j, const InputType& p) const { return xpr_.eval(j, i, p); }
    constexpr Scalar eval(int i, const InputType& p) const {
        fdapde_static_assert(Rows == 1 || Cols == 1, THIS_METHOD_IS_ONLY_FOR_VECTORS);
        return xpr_.eval(i, p);
    }
    constexpr const auto& operator()(int i, int j) const { return xpr_(j, i); }
    constexpr const auto& operator[](int i) const { return xpr_[i]; }
    constexpr int rows() const { return xpr_.cols(); }
    constexpr int cols() const { return xpr_.rows(); }
    constexpr int input_size() const { return xpr_.input_size(); }
    constexpr int size() const { return xpr_.size(); }
    template <typename... Args> constexpr MatrixTranspose<Derived>& forward(Args&&... args) {
        xpr_.forward(std::forward<Args>(args)...);
        return *this;
    }
   protected:
    typename internals::ref_select<const Derived>::type xpr_;
};

template <typename Derived>
struct MatrixDiagonalBlock : public fdapde::MatrixBase<Derived::StaticInputSize, MatrixDiagonalBlock<Derived>> {
    fdapde_static_assert(Derived::Rows == Derived::Cols, DIAGONAL_BLOCK_DEFINED_ONLY_FOR_SQUARED_MATRICES);
    using Base = MatrixBase<Derived::StaticInputSize, MatrixDiagonalBlock<Derived>>;
    using Scalar = typename Derived::Scalar;
    using InputType = typename Derived::InputType;
    static constexpr int StaticInputSize = Derived::StaticInputSize;
    static constexpr int Rows = Derived::Rows;
    static constexpr int Cols = Derived::Cols;
    static constexpr int NestAsRef = 0;
    static constexpr int ReadOnly = 1;
    using Base::operator();

    constexpr explicit MatrixDiagonalBlock(const Derived& xpr) : Base(), xpr_(xpr) { }
    constexpr Scalar eval(int i, int j, const InputType& p) const { return i != j ? Scalar(0) : xpr_.eval(i, j, p); }
    constexpr const auto& operator[](int i) const { return xpr_(i, i); }
    constexpr auto& operator[](int i) { return xpr_(i, i); }
    constexpr int rows() const { return xpr_.rows(); }
    constexpr int cols() const { return xpr_.cols(); }
    constexpr int input_size() const { return xpr_.input_size(); }
    constexpr int size() const { return xpr_.size(); }
    template <typename... Args> constexpr MatrixDiagonalBlock<Derived>& forward(Args&&... args) {
        xpr_.forward(std::forward<Args>(args)...);
        return *this;
    }
    // diagonal assignment
    template <int Size_, typename RhsDerived>
    constexpr MatrixDiagonalBlock<Derived>& operator=(const MatrixBase<Size_, RhsDerived>& rhs)
        requires(Rows != Dynamic && Cols != Dynamic) {
        fdapde_static_assert(Derived::ReadOnly != 0, BLOCK_ASSIGNMENT_TO_A_READ_ONLY_EXPRESSION_IS_INVALID);
        fdapde_static_assert(
          RhsDerived::Cols == 1 && RhsDerived::Rows == Rows &&
            std::is_convertible_v<typename RhsDerived::FunctorType FDAPDE_COMMA typename Derived::FunctorType>,
          VECTOR_FIELD_REQUIRED_OR_YOU_ARE_TRYING_TO_ASSIGN_A_BLOCK_WITH_NON_CONVERTIBLE_FUNCTOR_TYPE);
        for (int i = 0; i < xpr_.rows(); ++i) { xpr_(i, i) = rhs[i]; }
        return *this;
    }
    template <int Size_, typename RhsDerived>
    MatrixDiagonalBlock<Derived>& operator=(const MatrixBase<Size_, RhsDerived>& rhs)
        requires(Rows == Dynamic || Cols == Dynamic) {
        fdapde_static_assert(Derived::ReadOnly != 0, BLOCK_ASSIGNMENT_TO_A_READ_ONLY_EXPRESSION_IS_INVALID);
        fdapde_static_assert(
          std::is_convertible_v<typename RhsDerived::FunctorType FDAPDE_COMMA typename Derived::FunctorType>,
          YOU_ARE_TRYING_TO_ASSIGN_A_BLOCK_WITH_NON_CONVERTIBLE_COEFFICIENT_TYPE);
        fdapde_assert(rhs.rows() == xpr_.rows() && rhs.cols() == 1);
        for (int i = 0; i < xpr_.rows(); ++i) { xpr_(i, i) = rhs[i]; }
        return *this;
    }
   protected:
    typename internals::ref_select<Derived>::type xpr_;
};

template <typename Derived, int ViewMode>
struct MatrixSymmetricView :
    public fdapde::MatrixBase<Derived::StaticInputSize, MatrixSymmetricView<Derived, ViewMode>> {
    fdapde_static_assert(
      (Derived::Rows == Dynamic || Derived::Cols == Dynamic) || Derived::Rows == Derived::Cols,
      SYMMETRIC_MATRIX_CONCEPT_DEFINED_ONLY_FOR_SQUARED_MATRICES);
    fdapde_static_assert(
      ViewMode == fdapde::Upper || ViewMode == fdapde::Lower, SYMMETRIC_VIEWS_MUST_BE_EITHER_LOWER_OR_UPPER);
    using Base = MatrixBase<Derived::StaticInputSize, MatrixSymmetricView<Derived, ViewMode>>;
    using Scalar = typename Derived::Scalar;
    using InputType = typename Derived::InputType;
    static constexpr int StaticInputSize = Derived::StaticInputSize;
    static constexpr int Rows = Derived::Rows;
    static constexpr int Cols = Derived::Cols;
    static constexpr int NestAsRef = 0;
    static constexpr int ReadOnly = 1;
    using Base::operator();

    constexpr MatrixSymmetricView() = default;
    constexpr MatrixSymmetricView(const Derived& xpr) requires(Rows != Dynamic && Cols != Dynamic)
        : Base(), xpr_(xpr) { }
    MatrixSymmetricView(const Derived& xpr) requires(Rows == Dynamic || Cols == Dynamic) : Base(), xpr_(xpr) {
        fdapde_assert(xpr_.rows() == xpr_.cols());
    }

    template <typename InputType, typename Dest> constexpr void eval_at(const InputType& p, Dest& dest) const {
        fdapde_static_assert(
          std::is_invocable_v<Dest FDAPDE_COMMA int FDAPDE_COMMA int> ||
            fdapde::is_subscriptable<Dest FDAPDE_COMMA int>,
          DESTINATION_TYPE_MUST_EITHER_EXPOSE_A_MATRIX_LIKE_ACCESS_OPERATOR_OR_A_SUBSCRIPT_OPERATOR);
        // just evaluate half of the coefficients
        Scalar tmp = 0;
        int row = 0, col = 0;
        for (int i = 0; i < xpr_.rows(); ++i) {
            for (int j = 0; j < i; ++j) {
                if constexpr (ViewMode == fdapde::Lower) { row = i; col = j; }
                if constexpr (ViewMode == fdapde::Upper) { row = j; col = i; }
		tmp = xpr_.eval(row, col, p);
                if constexpr (std::is_invocable_v<Dest, int, int>) {
                    dest(row, col) = tmp;
                    dest(col, row) = tmp;
                } else {
                    dest[row * xpr_.cols() + col] = tmp;
                    dest[col * xpr_.cols() + row] = tmp;
                }
            }
        }
        // evaluate coefficients on the diagonal
        for (int i = 0; i < xpr_.rows(); ++i) {
            if constexpr (std::is_invocable_v<Dest, int, int>) {
                dest(i, i) = xpr_.eval(i, i, p);
            } else {
                dest[i * xpr_.cols() + i] = xpr_.eval(i, i, p);
            }
        }
        return;
    }
    constexpr int rows() const { return xpr_.rows(); }
    constexpr int cols() const { return xpr_.cols(); }
    constexpr int input_size() const { return xpr_.input_size(); }
    constexpr int size() const { return xpr_.size(); }
    template <typename... Args> constexpr MatrixSymmetricView<Derived, ViewMode>& forward(Args&&... args) {
        xpr_.forward(std::forward<Args>(args)...);
        return *this;
    }
    constexpr auto operator()(int i, int j) const {
        if constexpr (ViewMode == fdapde::Lower) return i < j ? xpr_(j, i) : xpr_(i, j);
        if constexpr (ViewMode == fdapde::Upper) return i > j ? xpr_(j, i) : xpr_(i, j);
    }
    constexpr Scalar eval(int i, int j, const InputType& p) const {
        if constexpr (ViewMode == fdapde::Lower) return i < j ? xpr_.eval(j, i, p) : xpr_.eval(i, j, p);
        if constexpr (ViewMode == fdapde::Upper) return i > j ? xpr_.eval(j, i, p) : xpr_.eval(i, j, p);
    }
   protected:
    typename internals::ref_select<Derived>::type xpr_;
};

// base class for matrix expressions
template <int StaticInputSize, typename Derived> struct MatrixBase {
    constexpr MatrixBase() = default;

    constexpr const Derived& derived() const { return static_cast<const Derived&>(*this); }
    constexpr Derived& derived() { return static_cast<Derived&>(*this); }
    // evaluate the expression at point p storing result in dest
    template <typename InputType, typename Dest> constexpr void eval_at(const InputType& p, Dest& dest) const {
        fdapde_static_assert(
          std::is_invocable_v<Dest FDAPDE_COMMA int FDAPDE_COMMA int> ||
            fdapde::is_subscriptable<Dest FDAPDE_COMMA int>,
          DESTINATION_TYPE_MUST_EITHER_EXPOSE_A_MATRIX_LIKE_ACCESS_OPERATOR_OR_A_SUBSCRIPT_OPERATOR);
        for (int i = 0; i < derived().rows(); ++i) {
 	    for (int j = 0; j < derived().cols(); ++j) {
                if constexpr (std::is_invocable_v<Dest, int, int>) {
                    dest(i, j) = derived().eval(i, j, p);
                } else {
                    dest[i * derived().cols() + j] = derived().eval(i, j, p);
                }
            }
        }
        return;
    }
    // evaluate the expression at point p
    template <typename InputType> constexpr auto operator()(InputType&& p) const {
        fdapde_static_assert(
          std::is_convertible_v<InputType FDAPDE_COMMA typename Derived::InputType>,
          CANNOT_INVOKE_EXPRESSION_WITH_THIS_INPUT_TYPE);
        using Scalar = typename Derived::Scalar;
        typename std::conditional<
          Derived::Rows == Dynamic || Derived::Cols == Dynamic, DMatrix<Scalar>,
          cexpr::Matrix<Scalar, Derived::Rows, Derived::Cols>>::type out;
        if constexpr (StaticInputSize == Dynamic) {
            fdapde_assert(p.size() == derived().input_size());
            out.resize(derived().rows(), derived().cols());
        }
        eval_at(p, out);
        return out;
    }
    template <typename... Args> constexpr Derived& forward([[maybe_unused]] Args&&... args) { return derived(); }

    // transpose
    constexpr MatrixTranspose<Derived> transpose() const { return MatrixTranspose<Derived>(derived()); }
    // block operations
    template <int BlockRows, int BlockCols>   // static sized block
    constexpr MatrixBlock<BlockRows, BlockCols, Derived> block(int i, int j) {
        return MatrixBlock<BlockRows, BlockCols, Derived>(derived(), i, j);
    }
    MatrixBlock<Dynamic, Dynamic, Derived>   // dynamic sized block
    block(int start_row, int start_col, int block_rows, int block_cols) {
        return MatrixBlock<Dynamic, Dynamic, Derived>(derived(), start_row, start_col, block_rows, block_cols);
    }
    constexpr auto col(int i) { return MatrixBlock<Derived::Rows, 1, Derived>(derived(), i); }
    constexpr auto col(int i) const { return MatrixBlock<Derived::Rows, 1, const Derived>(derived(), i); }
    constexpr auto row(int i) { return MatrixBlock<1, Derived::Cols, Derived>(derived(), i); }
    constexpr auto row(int i) const { return MatrixBlock<1, Derived::Cols, const Derived>(derived(), i); }
    // other block-type accessors
    MatrixBlock<Dynamic, Dynamic, Derived> top_rows(int n) { return block(0, 0, n, derived().cols()); }
    MatrixBlock<Dynamic, Dynamic, Derived> bottom_rows(int n) {
        return block(derived().rows() - n, 0, n, derived().cols());
    }
    MatrixBlock<Dynamic, Dynamic, Derived> left_cols(int n) { return block(0, 0, derived().rows(), n); }
    MatrixBlock<Dynamic, Dynamic, Derived> right_cols(int n) {
        return block(0, derived().cols() - n, derived().rows(), n);
    }
    // unary minus
    constexpr MatrixCoeffWiseOp<Derived, int, std::multiplies<>> operator-() const { return -1 * derived(); }
    // matrix norm
    constexpr MatrixNorm<2, Derived, 0> norm() const { return MatrixNorm<2, Derived, 0>(derived()); }
    constexpr MatrixNorm<2, Derived, 1> squared_norm() const { return MatrixNorm<2, Derived, 1>(derived()); }
    template <int Order> constexpr MatrixNorm<Order, Derived, 0> lp_norm() const {
        return MatrixNorm<Order, Derived, 0>(derived());
    }
    // vector field divergence
    constexpr Divergence<Derived> divergence() const {
        fdapde_static_assert(Derived::Cols == 1, THIS_METHOD_IS_FOR_VECTOR_FIELDS_ONLY);
        return Divergence<Derived>(derived());
    }
    // dot product
    template <int RhsStaticInputSize, typename Rhs>
    constexpr DotProduct<Derived, Rhs> dot(const MatrixBase<RhsStaticInputSize, Rhs>& rhs) const {
        return DotProduct<Derived, Rhs>(derived(), rhs.derived());
    }
    // diagonal
    constexpr MatrixDiagonalBlock<Derived> diagonal() const { return MatrixDiagonalBlock<Derived>(derived()); }
    // symmetric view
    template <int ViewMode> constexpr MatrixSymmetricView<Derived, ViewMode> symmetric_view() const {
        return MatrixSymmetricView<Derived, ViewMode>(derived());
    }
};

// triangular views (?)
// if vector, take jacobian
// multiplication by fixed matrices, vectors (could be arrays, cexpr::Matrix, SMatrix, ...)
// rowwise, colwise iterators (?)
// inverse (?)
  
}   // namespace fdapde

#endif   // __MATRIX_FIELD_H__
