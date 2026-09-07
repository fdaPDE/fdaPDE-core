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

#ifndef __FDAPDE_LINALG_SKEW_H__
#define __FDAPDE_LINALG_SKEW_H__

#include "header_check.h"

namespace fdapde {

template <typename XprType> struct SkewSymmetricMatrixExpr;
template <typename Scalar_, int Rows_, int Cols_, int StorageOrder_> class SkewSymmetricMatrixView;

namespace internals {

template <typename Scalar, int Rows, int Cols, int StorageOrder>
struct is_mutable_matrix_view<SkewSymmetricMatrixView<Scalar, Rows, Cols, StorageOrder>> :
    std::bool_constant<!std::is_const_v<Scalar>> { };

}   // namespace internals

namespace internals {

constexpr int compute_skew_symmetric_shape(int storage_size) {
    if (storage_size < 0) return -1;
    int size = 0;
    while (static_cast<long long>(size) * (size - 1) / 2 < storage_size) ++size;
    return static_cast<long long>(size) * (size - 1) / 2 == storage_size ? size : -1;
}

constexpr int checked_skew_symmetric_shape(std::size_t size) {
    const int input_size = checked_matrix_data_size(size);
    const int shape = compute_skew_symmetric_shape(input_size);
    if (shape < 0) { throw std::invalid_argument("packed skew-symmetric input has an invalid length"); }
    return shape;
}

constexpr int checked_skew_symmetric_storage_size(int rows) {
    (void)checked_matrix_size(rows, rows);
    return static_cast<int>(static_cast<long long>(rows) * (rows - 1) / 2);
}

struct skew_symmetric_assignment_executor {
    template <typename DstXprType, typename SrcXprType, typename AssignmentOp>
    static constexpr void run(DstXprType& dst, const SrcXprType& src, AssignmentOp&& op) {
        fdapde_static_assert(DstXprType::ReadOnly == 0, ASSIGNMENT_TO_READ_ONLY_LOCATION);
        if constexpr (!std::is_arithmetic_v<SrcXprType>) {
            fdapde_static_assert(
              internals::is_dynamic_sized_v<DstXprType> || internals::is_dynamic_sized_v<SrcXprType> ||
                internals::same_static_shape_v<DstXprType FDAPDE_COMMA SrcXprType>,
              INVALID_ASSIGNMENT__DIFFERENT_LHS_AND_RHS_STATIC_SHAPES);
            if (dst.rows() != src.rows() || dst.cols() != src.cols()) {
                throw std::invalid_argument("skew-symmetric assignment requires matching dimensions");
            }
        }
        auto fetch = [](const SrcXprType& source, [[maybe_unused]] int i, [[maybe_unused]] int j) -> decltype(auto) {
            if constexpr (std::is_arithmetic_v<SrcXprType>) {
                return source;
            } else {
                return source(i, j);
            }
        };
        for (int i = 0, size = dst.rows(); i < size; ++i) {
            for (int j = i + 1; j < size; ++j) {
                auto& target = dst.data()[i * (2 * size - i - 1) / 2 + (j - i - 1)];
                op(target, fetch(src, i, j));
            }
        }
    }
};

template <int ViewMode_, typename SkewXprType_>
class skew_symmetric_wrapper : public SkewSymmetricMatrixExpr<skew_symmetric_wrapper<ViewMode_, SkewXprType_>> {
   private:
    fdapde_static_assert(ViewMode_ == Lower || ViewMode_ == Upper, VIEW_MODE_MUST_BE_EITHER_LOWER_OR_UPPER);
    using Base = SkewSymmetricMatrixExpr<skew_symmetric_wrapper<ViewMode_, SkewXprType_>>;
    using XprType = std::remove_reference_t<SkewXprType_>;
    using XprTypeClean = std::remove_cv_t<XprType>;
    using XprTypeNested = internals::ref_select_t<SkewXprType_>;
    static constexpr int ViewMode = ViewMode_;
   public:
    using Scalar = typename XprTypeClean::Scalar;
    static constexpr int Rows = XprTypeClean::Rows;
    static constexpr int Cols = XprTypeClean::Cols;
    static constexpr int StorageOrder = XprTypeClean::StorageOrder;
    static constexpr int NestAsRef = 0;
    static constexpr int ReadOnly = 1;

    constexpr skew_symmetric_wrapper(const skew_symmetric_wrapper&) = default;
    template <typename XprType__>
        requires(
          !std::same_as<std::remove_cvref_t<XprType__>, skew_symmetric_wrapper> &&
          internals::safely_nestable<XprTypeNested, XprType__>)
    constexpr explicit skew_symmetric_wrapper(XprType__&& xpr) :
        Base(), xpr_(std::forward<XprType__>(xpr)), size_(xpr_.rows() == xpr_.cols() ? xpr_.rows() : 0) {
        fdapde_static_assert(
          Rows == Dynamic || Cols == Dynamic || Rows == Cols, THIS_CLASS_IS_FOR_SQUARE_MATRICES_ONLY);
        if (xpr_.rows() < 0 || xpr_.rows() != xpr_.cols()) {
            throw std::invalid_argument("skew-symmetric view requires square dimensions");
        }
    }
    constexpr Scalar operator()(int i, int j) const {
        internals::validate_matrix_index(i, j, size_, size_);
        if (i == j) return Scalar(0);
        if constexpr (ViewMode == Upper) { return i < j ? xpr_(i, j) : -xpr_(j, i); }
        if constexpr (ViewMode == Lower) { return i > j ? xpr_(i, j) : -xpr_(j, i); }
    }
    constexpr int rows() const { return size_; }
    constexpr int cols() const { return size_; }
   private:
    XprTypeNested xpr_;
    int size_;
};

template <int ViewMode_, typename XprType_> constexpr auto skew_symmetric_cast(XprType_&& xpr) {
    return skew_symmetric_wrapper<ViewMode_, XprType_>(std::forward<XprType_>(xpr));
}

}   // namespace internals

template <typename XprType_> struct SkewSymmetricMatrixExpr : public MatrixExpr<XprType_> {
    using XprType = std::decay_t<XprType_>;
    using Base = MatrixExpr<XprType_>;
    using Base::derived;
    using Base::operator=;
    using Base::operator*=;

    template <internals::matrix_expression RhsXprType> constexpr void operator*=(const RhsXprType&) = delete;
};

template <typename Scalar_, int Rows_, int Cols_, int StorageOrder_, typename SkewMatrixType>
class SkewSymmetricMatrixBase : public SkewSymmetricMatrixExpr<SkewMatrixType> {
   protected:
    using Base = SkewSymmetricMatrixExpr<SkewMatrixType>;
    using Base::derived;
   public:
    using Scalar = Scalar_;
    static constexpr int Rows = Rows_;
    static constexpr int Cols = Cols_;
    static constexpr int StorageOrder = StorageOrder_;
    static constexpr int ReadOnly = std::is_const_v<Scalar_>;
    using Base::operator=;

    template <typename Scalar__>
        requires(std::is_same_v<std::remove_cv_t<Scalar>, std::remove_cv_t<Scalar__>>)
    class skew_symmetric_proxy {
       private:
        using Value = std::remove_const_t<Scalar__>;
       public:
        constexpr skew_symmetric_proxy(Scalar__* data, int i, int j, int size) :
            data_(data),
            index_(compute_linear_index_(i < j ? i : j, i < j ? j : i, size)),
            sign_flip_(i > j),
            diagonal_(i == j) { }
        template <typename T>
            requires(std::is_convertible_v<T, Value> && !std::is_const_v<Scalar__>)
        constexpr skew_symmetric_proxy& operator=(T value) {
            const Value converted = static_cast<Value>(value);
            if (diagonal_) {
                if (converted != Value(0)) {
                    throw std::invalid_argument("skew-symmetric diagonal coefficients must be zero");
                }
                return *this;
            }
            data_[index_] = sign_flip_ ? -converted : converted;
            return *this;
        }
        constexpr operator Value() const {
            if (diagonal_) return Value(0);
            const Value value = data_[index_];
            return value == Value(0) || !sign_flip_ ? value : -value;
        }
       private:
        static constexpr int compute_linear_index_(int i, int j, int size) {
            return i * (2 * size - i - 1) / 2 + (j - i - 1);
        }
        Scalar__* data_;
        int index_;
        bool sign_flip_;
        bool diagonal_;
    };
    using reference = skew_symmetric_proxy<Scalar>;
    using const_reference = skew_symmetric_proxy<const std::remove_const_t<Scalar>>;

    constexpr SkewSymmetricMatrixBase() : rows_(default_shape_()), cols_(default_shape_()) { }
    constexpr SkewSymmetricMatrixBase(int rows, int cols) : rows_(rows), cols_(cols) {
        internals::validate_matrix_shape<Rows, Cols>(rows, cols);
        (void)internals::checked_matrix_size(rows, cols);
        if (rows != cols) { throw std::invalid_argument("skew-symmetric matrix requires square dimensions"); }
    }
    constexpr const_reference operator()(int i, int j) const {
        internals::validate_matrix_index(i, j, rows_, cols_);
        return const_reference(derived().data(), i, j, rows_);
    }
    constexpr reference operator()(int i, int j) requires(ReadOnly == 0) {
        internals::validate_matrix_index(i, j, rows_, cols_);
        return reference(derived().data(), i, j, rows_);
    }
    constexpr int rows() const { return rows_; }
    constexpr int cols() const { return cols_; }
    constexpr int storage_size() const { return rows_ * (rows_ - 1) / 2; }
   protected:
    static constexpr int default_shape_() {
        if constexpr (Rows != Dynamic) return Rows;
        if constexpr (Cols != Dynamic) return Cols;
        return 0;
    }
    int rows_, cols_;
};

// Skew-symmetric matrices form a vector space. Products generally do not preserve the structure.
template <typename LhsXprType, typename RhsXprType>
constexpr auto
operator+(const SkewSymmetricMatrixExpr<LhsXprType>& lhs, const SkewSymmetricMatrixExpr<RhsXprType>& rhs) {
    return internals::skew_symmetric_cast<Upper>(
      MatrixBinOp<LhsXprType, RhsXprType, std::plus<>>(lhs.derived(), rhs.derived(), std::plus<>()));
}
template <typename LhsXprType, typename RhsXprType>
constexpr auto
operator-(const SkewSymmetricMatrixExpr<LhsXprType>& lhs, const SkewSymmetricMatrixExpr<RhsXprType>& rhs) {
    return internals::skew_symmetric_cast<Upper>(
      MatrixBinOp<LhsXprType, RhsXprType, std::minus<>>(lhs.derived(), rhs.derived(), std::minus<>()));
}
template <typename XprType, typename ScalarType>
    requires(std::is_arithmetic_v<ScalarType>)
constexpr auto operator*(const SkewSymmetricMatrixExpr<XprType>& lhs, ScalarType rhs) {
    return internals::skew_symmetric_cast<Upper>(
      MatrixScalarMultiplicationOp<XprType, ScalarType>(lhs.derived(), rhs));
}
template <typename XprType, typename ScalarType>
    requires(std::is_arithmetic_v<ScalarType>)
constexpr auto operator*(ScalarType lhs, const SkewSymmetricMatrixExpr<XprType>& rhs) {
    return rhs * lhs;
}
template <typename XprType, typename ScalarType>
    requires(std::is_arithmetic_v<ScalarType>)
constexpr auto operator/(const SkewSymmetricMatrixExpr<XprType>& lhs, ScalarType rhs) {
    return internals::skew_symmetric_cast<Upper>(
      static_cast<const MatrixExpr<XprType>&>(lhs) / rhs);
}

template <typename Scalar_, int Rows_, int Cols_ = Rows_, int StorageOrder_ = RowMajor>
class SkewSymmetricMatrix :
    public SkewSymmetricMatrixBase<
      Scalar_, Rows_, Cols_, StorageOrder_, SkewSymmetricMatrix<Scalar_, Rows_, Cols_, StorageOrder_>> {
   private:
    using Base = SkewSymmetricMatrixBase<
      Scalar_, Rows_, Cols_, StorageOrder_, SkewSymmetricMatrix<Scalar_, Rows_, Cols_, StorageOrder_>>;
    static constexpr int StaticShape = Rows_ != Dynamic ? Rows_ : Cols_;
    static constexpr bool HasSupportedStaticStorage =
      StaticShape == Dynamic ||
      static_cast<std::uint64_t>(StaticShape) * static_cast<std::uint64_t>(StaticShape) <=
        static_cast<std::uint64_t>(std::numeric_limits<int>::max());
   public:
    fdapde_static_assert(
      (Rows_ == Dynamic || Rows_ > 0) && (Cols_ == Dynamic || Cols_ > 0), INVALID_MATRIX_DIMENSIONS);
    fdapde_static_assert(
      Rows_ == Dynamic || Cols_ == Dynamic || Rows_ == Cols_, THIS_CLASS_IS_FOR_SQUARE_MATRICES_ONLY);
    fdapde_static_assert(StorageOrder_ == RowMajor, PACKED_COL_MAJOR_STRUCTURED_STORAGE_IS_NOT_SUPPORTED);
    fdapde_static_assert(HasSupportedStaticStorage, MATRIX_SIZE_EXCEEDS_SUPPORTED_RANGE);
    using Scalar = Scalar_;
    static constexpr int Rows = Rows_;
    static constexpr int Cols = Cols_;
    static constexpr int StorageOrder = StorageOrder_;
    static constexpr int StorageSize =
      StaticShape == Dynamic || !HasSupportedStaticStorage ? Dynamic : StaticShape * (StaticShape - 1) / 2;
    static constexpr int NestAsRef = 1;
    static constexpr int ReadOnly = std::is_const_v<Scalar_>;
    using assignment_executor = internals::skew_symmetric_assignment_executor;
    using StorageType = std::conditional_t<
      StorageSize == Dynamic, std::vector<Scalar_>,
      std::array<Scalar_, StorageSize < 0 ? 0 : static_cast<std::size_t>(StorageSize)>>;

    constexpr SkewSymmetricMatrix() : Base(), data_() { }
    constexpr SkewSymmetricMatrix(const SkewSymmetricMatrix& rhs) :
        Base(rhs.rows(), rhs.cols()), data_(rhs.data_) { }
    constexpr SkewSymmetricMatrix& operator=(const SkewSymmetricMatrix& rhs) & {
        if (this == std::addressof(rhs)) return *this;
        data_ = rhs.data_;
        if constexpr (Rows == Dynamic || Cols == Dynamic) {
            this->rows_ = rhs.rows();
            this->cols_ = rhs.cols();
        }
        return *this;
    }
    constexpr void operator=(const SkewSymmetricMatrix&) && = delete;
    using Base::operator=;

    constexpr explicit SkewSymmetricMatrix(int size) : SkewSymmetricMatrix(size, size) { }
    constexpr SkewSymmetricMatrix(int rows, int cols) : Base(rows, cols), data_() {
        fdapde_static_assert(Rows == Dynamic || Cols == Dynamic, THIS_METHOD_IS_FOR_DYNAMIC_SIZED_MATRICES_ONLY);
        const int storage_size = internals::checked_skew_symmetric_storage_size(this->rows_);
        if constexpr (StorageSize == Dynamic) { data_.resize(static_cast<std::size_t>(storage_size)); }
    }
    template <typename RhsXprType_>
    constexpr SkewSymmetricMatrix(const SkewSymmetricMatrixExpr<RhsXprType_>& rhs) :
        Base(rhs.rows(), rhs.cols()), data_() {
        if constexpr (StorageSize == Dynamic) {
            data_.resize(static_cast<std::size_t>(internals::checked_skew_symmetric_storage_size(this->rows_)));
        }
        assignment_executor::run(*this, rhs.derived(), [](auto& l, const auto& r) { l = r; });
    }
    template <typename RhsXprType_>
    constexpr SkewSymmetricMatrix& operator=(const SkewSymmetricMatrixExpr<RhsXprType_>& rhs) & {
        static_cast<MatrixExpr<SkewSymmetricMatrix>&>(*this).template operator=<RhsXprType_>(rhs);
        return *this;
    }
    template <typename Scalar__>
        requires(std::is_constructible_v<Scalar_, Scalar__>)
    constexpr explicit SkewSymmetricMatrix(const std::vector<Scalar__>& data) :
        Base(
          internals::checked_skew_symmetric_shape(data.size()),
          internals::checked_skew_symmetric_shape(data.size())),
        data_() {
        const int input_size = internals::checked_matrix_data_size(data.size());
        if constexpr (StorageSize == Dynamic) { data_.resize(data.size()); }
        if (!std::cmp_equal(data_.size(), input_size)) {
            throw std::invalid_argument("packed skew-symmetric input does not match its static size");
        }
        for (int i = 0; i < input_size; ++i) data_[static_cast<std::size_t>(i)] = data[static_cast<std::size_t>(i)];
    }
    template <typename Scalar__, std::size_t Size>
        requires(std::is_constructible_v<Scalar_, Scalar__>)
    constexpr explicit SkewSymmetricMatrix(const Scalar__ (&data)[Size]) : Base(), data_() {
        fdapde_static_assert(
          Rows_ != Dynamic && Cols_ != Dynamic && StorageSize == static_cast<int>(Size),
          THIS_METHOD_IS_FOR_STATIC_SIZED_MATRICES_ONLY);
        for (int i = 0; i < StorageSize; ++i) data_[static_cast<std::size_t>(i)] = data[static_cast<std::size_t>(i)];
    }
    void resize(int size) { resize(size, size); }
    void resize(int rows, int cols) {
        fdapde_static_assert(Rows == Dynamic || Cols == Dynamic, THIS_METHOD_IS_FOR_DYNAMIC_SIZED_MATRICES_ONLY);
        internals::validate_matrix_shape<Rows, Cols>(rows, cols);
        const int effective_rows = Rows == Dynamic ? rows : Rows;
        const int effective_cols = Cols == Dynamic ? cols : Cols;
        if (effective_rows != effective_cols) {
            throw std::invalid_argument("skew-symmetric matrix resize requires square dimensions");
        }
        const int storage_size = internals::checked_skew_symmetric_storage_size(effective_rows);
        if (this->rows_ == effective_rows && this->cols_ == effective_cols &&
            std::cmp_equal(data_.size(), storage_size)) {
            return;
        }
        if constexpr (StorageSize == Dynamic) { data_.resize(static_cast<std::size_t>(storage_size)); }
        this->rows_ = effective_rows;
        this->cols_ = effective_cols;
    }
    constexpr const std::remove_const_t<Scalar_>* data() const { return data_.data(); }
    constexpr Scalar_* data() { return data_.data(); }
   private:
    StorageType data_;
};

template <typename Scalar_, int Rows_, int Cols_ = Rows_, int StorageOrder_ = RowMajor>
class SkewSymmetricMatrixView :
    public SkewSymmetricMatrixBase<
      Scalar_, Rows_, Cols_, StorageOrder_, SkewSymmetricMatrixView<Scalar_, Rows_, Cols_, StorageOrder_>> {
   private:
    using Base = SkewSymmetricMatrixBase<
      Scalar_, Rows_, Cols_, StorageOrder_, SkewSymmetricMatrixView<Scalar_, Rows_, Cols_, StorageOrder_>>;
    static constexpr int StaticShape = Rows_ != Dynamic ? Rows_ : Cols_;
    static constexpr bool HasSupportedStaticStorage =
      StaticShape == Dynamic ||
      static_cast<std::uint64_t>(StaticShape) * static_cast<std::uint64_t>(StaticShape) <=
        static_cast<std::uint64_t>(std::numeric_limits<int>::max());
   public:
    fdapde_static_assert(
      (Rows_ == Dynamic || Rows_ > 0) && (Cols_ == Dynamic || Cols_ > 0), INVALID_MATRIX_DIMENSIONS);
    fdapde_static_assert(
      Rows_ == Dynamic || Cols_ == Dynamic || Rows_ == Cols_, THIS_CLASS_IS_FOR_SQUARE_MATRICES_ONLY);
    fdapde_static_assert(StorageOrder_ == RowMajor, PACKED_COL_MAJOR_STRUCTURED_STORAGE_IS_NOT_SUPPORTED);
    fdapde_static_assert(HasSupportedStaticStorage, MATRIX_SIZE_EXCEEDS_SUPPORTED_RANGE);
    using Scalar = Scalar_;
    static constexpr int Rows = Rows_;
    static constexpr int Cols = Cols_;
    static constexpr int StorageOrder = StorageOrder_;
    static constexpr int StorageSize =
      StaticShape == Dynamic || !HasSupportedStaticStorage ? Dynamic : StaticShape * (StaticShape - 1) / 2;
    static constexpr int NestAsRef = 0;
    static constexpr int ReadOnly = std::is_const_v<Scalar_>;
    using assignment_executor = internals::skew_symmetric_assignment_executor;

    constexpr SkewSymmetricMatrixView(const SkewSymmetricMatrixView&) = default;
    constexpr SkewSymmetricMatrixView() requires(Rows_ == Dynamic && Cols_ == Dynamic) : Base(), data_(nullptr) { }
    constexpr SkewSymmetricMatrixView() requires(Rows_ != Dynamic || Cols_ != Dynamic) = delete;
    template <typename Scalar__>
        requires(std::is_convertible_v<Scalar__*, Scalar_*>)
    constexpr explicit SkewSymmetricMatrixView(Scalar__* data) : Base(), data_(data) {
        fdapde_static_assert(Rows != Dynamic && Cols != Dynamic, THIS_METHOD_IS_FOR_STATIC_SIZED_MATRICES_ONLY);
        if (StorageSize > 0 && data == nullptr) {
            throw std::invalid_argument("nonempty skew-symmetric view requires storage");
        }
    }
    template <typename Scalar__>
        requires(std::is_convertible_v<Scalar__*, Scalar_*>)
    constexpr SkewSymmetricMatrixView(Scalar__* data, int size) : SkewSymmetricMatrixView(data, size, size) { }
    template <typename Scalar__>
        requires(std::is_convertible_v<Scalar__*, Scalar_*>)
    constexpr SkewSymmetricMatrixView(Scalar__* data, int rows, int cols) : Base(rows, cols), data_(data) {
        if (this->storage_size() > 0 && data == nullptr) {
            throw std::invalid_argument("nonempty skew-symmetric view requires storage");
        }
    }
    using Base::operator=;
    constexpr SkewSymmetricMatrixView& operator=(const SkewSymmetricMatrixView& other) & requires(ReadOnly == 0) {
        static_cast<Base&>(*this).template operator=<SkewSymmetricMatrixView>(other);
        return *this;
    }
    constexpr SkewSymmetricMatrixView operator=(const SkewSymmetricMatrixView& other) && requires(ReadOnly == 0) {
        static_cast<Base&>(*this).template operator=<SkewSymmetricMatrixView>(other);
        return *this;
    }
    constexpr const std::remove_const_t<Scalar_>* data() const { return data_; }
    constexpr Scalar_* data() requires(!std::is_const_v<Scalar_>) { return data_; }
   private:
    Scalar_* data_;
};

template <typename XprType> struct is_skew_symmetric_matrix {
    using Type = std::remove_cvref_t<XprType>;
    static constexpr bool value = std::is_base_of_v<SkewSymmetricMatrixExpr<Type>, Type>;
};
template <typename XprType>
static constexpr bool is_skew_symmetric_matrix_v = is_skew_symmetric_matrix<XprType>::value;

}   // namespace fdapde

#endif   // __FDAPDE_LINALG_SKEW_H__
