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

#ifndef __FDAPDE_LINALG_SPARSE_H__
#define __FDAPDE_LINALG_SPARSE_H__

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "header_check.h"

namespace fdapde {

/// @brief stores a row, column and mutable coefficient for sparse construction
template <typename Scalar_> class Triplet {
   public:
    using Index = int;
    using Scalar = std::remove_cvref_t<Scalar_>;

    /// @brief constructs a zero coefficient at position zero, zero
    constexpr Triplet() = default;
    /// @brief records the supplied indices and coefficient without matrix-specific validation
    constexpr Triplet(Index row, Index col, const Scalar& value) : row_(row), col_(col), value_(value) { }

    /// @brief returns the recorded row index
    constexpr Index row() const { return row_; }
    /// @brief returns the recorded column index
    constexpr Index col() const { return col_; }
    /// @brief borrows the recorded coefficient for read access
    constexpr const Scalar& value() const { return value_; }
    /// @brief borrows the recorded coefficient for mutation
    constexpr Scalar& value() { return value_; }
   private:
    Index row_ = 0;
    Index col_ = 0;
    Scalar value_ {};
};

/// @brief owns rectangular CSR storage with sorted columns, input-ordered duplicate sums and zero elision
template <typename Scalar_> class SparseMatrix {
   public:
    using Index = int;
    using Scalar = std::remove_cvref_t<Scalar_>;
    using triplet_type = Triplet<Scalar>;

    static_assert(!std::is_same_v<Scalar, bool>, "SparseMatrix<bool> is not supported");

    /// @brief provides read-only access to one stored row coefficient
    class ConstEntry {
       public:
        /// @brief returns the column index of this stored coefficient
        constexpr Index column() const { return column_; }
        /// @brief borrows the coefficient from its matrix owner
        constexpr const Scalar& value() const { return *value_; }
       private:
        friend class SparseMatrix;
        /// @brief binds a stored column index to its coefficient address
        constexpr ConstEntry(Index column, const Scalar* value) : column_(column), value_(value) { }

        Index column_ = 0;
        const Scalar* value_ = nullptr;
    };

    /// @brief borrows the stored entries of a matrix row in increasing column order
    class ConstRowView {
       public:
        /// @brief traverses stored row entries as read-only proxy values
        class const_iterator {
           public:
            using iterator_category = std::forward_iterator_tag;
            using difference_type = std::ptrdiff_t;
            using value_type = ConstEntry;
            using reference = ConstEntry;

            /// @brief constructs a singular iterator that can be assigned before use
            constexpr const_iterator() = default;

            /// @brief returns the current stored entry without copying its coefficient
            constexpr reference operator*() const { return ConstEntry(columns_[index_], values_ + index_); }
            /// @brief advances to the next stored coefficient
            constexpr const_iterator& operator++() {
                ++index_;
                return *this;
            }
            /// @brief advances while returning the previous iterator position
            constexpr const_iterator operator++(int) {
                const_iterator result(*this);
                ++(*this);
                return result;
            }
            /// @brief compares the storage identity and position of two iterators
            friend constexpr bool operator==(const const_iterator&, const const_iterator&) = default;
           private:
            friend class ConstRowView;
            /// @brief binds an iterator to matrix storage at the given position
            constexpr const_iterator(const Index* columns, const Scalar* values, Index index) :
                columns_(columns), values_(values), index_(index) { }

            const Index* columns_ = nullptr;
            const Scalar* values_ = nullptr;
            Index index_ = 0;
        };

        /// @brief returns an iterator to the first stored row entry
        constexpr const_iterator begin() const { return const_iterator(columns_, values_, begin_); }
        /// @brief returns the iterator immediately after the stored row entries
        constexpr const_iterator end() const { return const_iterator(columns_, values_, end_); }
        /// @brief returns the number of stored coefficients in this row
        constexpr Index size() const { return end_ - begin_; }
        /// @brief reports whether the row contains no stored coefficients
        constexpr bool empty() const { return begin_ == end_; }
       private:
        friend class SparseMatrix;
        /// @brief borrows the half-open storage interval of one row
        constexpr ConstRowView(const Index* columns, const Scalar* values, Index begin, Index end) :
            columns_(columns), values_(values), begin_(begin), end_(end) { }

        const Index* columns_ = nullptr;
        const Scalar* values_ = nullptr;
        Index begin_ = 0;
        Index end_ = 0;
    };

    // resize, rebuild, assignment, move and swap invalidate borrowed rows and iterators

    /// @brief constructs an empty zero-by-zero matrix
    SparseMatrix() : row_offsets_(1, 0) { }
    /// @brief constructs a checked rectangular shape without stored coefficients
    SparseMatrix(Index rows, Index cols) { reset_shape_(rows, cols); }
    /// @brief compresses checked triplets with stable duplicate summation and exact-zero removal
    SparseMatrix(Index rows, Index cols, const std::vector<triplet_type>& triplets) {
        validate_shape_(rows, cols);
        rows_ = rows;
        cols_ = cols;
        build_(triplets);
    }
    /// @brief compresses an initializer list using the same triplet construction rules
    SparseMatrix(Index rows, Index cols, std::initializer_list<triplet_type> triplets) :
        SparseMatrix(rows, cols, std::vector<triplet_type>(triplets)) { }

    /// @brief copies dimensions, pattern and coefficients into independent storage
    SparseMatrix(const SparseMatrix&) = default;
    /// @brief replaces the matrix with an independent copy, preserving it if copying fails
    SparseMatrix& operator=(const SparseMatrix& other) {
        if (this == &other) return *this;
        SparseMatrix replacement(other);
        swap(replacement);
        return *this;
    }
    /// @brief takes ownership of the source storage and clears its dimensions
    SparseMatrix(SparseMatrix&& other) noexcept :
        rows_(std::exchange(other.rows_, 0)),
        cols_(std::exchange(other.cols_, 0)),
        row_offsets_(std::move(other.row_offsets_)),
        column_indices_(std::move(other.column_indices_)),
        values_(std::move(other.values_)) { }
    /// @brief transfers storage and dimensions while tolerating self-move
    SparseMatrix& operator=(SparseMatrix&& other) noexcept {
        if (this == &other) return *this;
        rows_ = std::exchange(other.rows_, 0);
        cols_ = std::exchange(other.cols_, 0);
        row_offsets_ = std::move(other.row_offsets_);
        column_indices_ = std::move(other.column_indices_);
        values_ = std::move(other.values_);
        return *this;
    }

    /// @brief returns the matrix row count
    constexpr Index rows() const { return rows_; }
    /// @brief returns the matrix column count
    constexpr Index cols() const { return cols_; }
    /// @brief returns the stored entry count, including zeros introduced through value_ref
    Index non_zeros() const { return static_cast<Index>(values_.size()); }

    /// @brief returns a checked coefficient by value, or zero if absent from the pattern
    Scalar coeff(Index row, Index col) const {
        validate_index_(row, col);
        const Index position = find_position_(row, col);
        return position == missing_ ? Scalar {} : values_[position];
    }
    /// @brief reports whether a checked position belongs to the stored pattern
    bool contains(Index row, Index col) const {
        validate_index_(row, col);
        return find_position_(row, col) != missing_;
    }
    /// @brief borrows an existing coefficient from an lvalue owner without inserting a new entry
    Scalar& value_ref(Index row, Index col) & {
        validate_index_(row, col);
        const Index position = find_position_(row, col);
        fdapde_strong_assert(
          position != missing_, std::out_of_range, "SparseMatrix value_ref requires an existing stored coefficient");
        return values_[position];
    }
    /// @brief borrows a checked row from an lvalue matrix
    ConstRowView row(Index row_index) const& {
        validate_row_(row_index);
        return ConstRowView(
          column_indices_.data(), values_.data(), row_offsets_[row_index], row_offsets_[row_index + 1]);
    }

    /// @brief rejects a borrowed row whose temporary owner would immediately expire
    ConstRowView row(Index) const&& = delete;

    // value_ref preserves stored zeros; resize clears the pattern and rebuild elides exact zeros
    /// @brief replaces the shape and clears all stored entries, preserving the matrix on failure
    void resize(Index rows, Index cols) {
        SparseMatrix replacement(rows, cols);
        swap(replacement);
    }
    /// @brief replaces the pattern at the current shape, preserving the matrix on failure
    void rebuild(const std::vector<triplet_type>& triplets) {
        SparseMatrix replacement(rows_, cols_, triplets);
        swap(replacement);
    }
    /// @brief rebuilds the current shape from an initializer list
    void rebuild(std::initializer_list<triplet_type> triplets) { rebuild(std::vector<triplet_type>(triplets)); }

    /// @brief returns an owning transpose with sorted rows and explicit zeros removed
    SparseMatrix transpose() const {
        SparseMatrix result(cols_, rows_);
        for (Index row = 0; row < rows_; ++row) {
            for (Index current = row_offsets_[row]; current < row_offsets_[row + 1]; ++current) {
                if (values_[current] != Scalar {}) { ++result.row_offsets_[column_indices_[current] + 1]; }
            }
        }
        for (Index row = 0; row < result.rows_; ++row) { result.row_offsets_[row + 1] += result.row_offsets_[row]; }
        result.column_indices_.resize(static_cast<std::size_t>(result.row_offsets_.back()));
        result.values_.resize(static_cast<std::size_t>(result.row_offsets_.back()));
        std::vector<Index> next(result.row_offsets_);
        for (Index row = 0; row < rows_; ++row) {
            for (Index current = row_offsets_[row]; current < row_offsets_[row + 1]; ++current) {
                if (values_[current] == Scalar {}) continue;
                const Index transposed_row = column_indices_[current];
                const Index position = next[transposed_row]++;
                result.column_indices_[position] = row;
                result.values_[position] = values_[current];
            }
        }
        return result;
    }

    /// @brief mirrors a checked stored triangle into a full symmetric owner, omitting exact zeros
    SparseMatrix symmetric_expanded(int triangle) const {
        fdapde_strong_assert(
          triangle == Upper || triangle == Lower, std::invalid_argument,
          "symmetric expansion requires an upper or lower triangle");
        fdapde_strong_assert(rows_ == cols_, std::invalid_argument, "symmetric expansion requires a square matrix");

        std::size_t output_size = 0;
        for (Index row = 0; row < rows_; ++row) {
            for (Index current = row_offsets_[row]; current < row_offsets_[row + 1]; ++current) {
                const Index col = column_indices_[current];
                fdapde_strong_assert(
                  triangle == Upper ? row <= col : row >= col, std::invalid_argument,
                  "stored coefficient lies outside the declared triangle");
                if (values_[current] != Scalar {}) {
                    const std::size_t increment = row == col ? 1 : 2;
                    fdapde_strong_assert(
                      output_size <= static_cast<std::size_t>(std::numeric_limits<Index>::max()) - increment,
                      std::length_error, "symmetric expansion exceeds the supported int range");
                    output_size += increment;
                }
            }
        }

        std::vector<triplet_type> triplets;
        triplets.reserve(output_size);
        for (Index row = 0; row < rows_; ++row) {
            for (Index current = row_offsets_[row]; current < row_offsets_[row + 1]; ++current) {
                const Index col = column_indices_[current];
                const Scalar& value = values_[current];
                if (value == Scalar {}) continue;
                triplets.emplace_back(row, col, value);
                if (row != col) triplets.emplace_back(col, row, value);
            }
        }
        return SparseMatrix(rows_, cols_, triplets);
    }

    /// @brief multiplies a matching column-vector expression into an independent dense vector
    template <internals::matrix_expression RhsXprType>
        requires(std::remove_cvref_t<RhsXprType>::Cols == 1)
    auto operator*(const RhsXprType& rhs) const {
        using RhsScalar = std::remove_cv_t<typename std::remove_cvref_t<RhsXprType>::Scalar>;
        using ResultScalar = std::common_type_t<Scalar, RhsScalar>;
        fdapde_strong_assert(
          rhs.size() == cols_, std::invalid_argument, "sparse-vector product requires matching inner dimensions");
        Vector<ResultScalar, Dynamic> result(rows_);
        ResultScalar* result_data = result.data();
        for (Index row = 0; row < rows_; ++row) {
            ResultScalar value {};
            for (Index current = row_offsets_[row]; current < row_offsets_[row + 1]; ++current) {
                if constexpr (has_plain_dense_storage_<RhsXprType>) {
                    value = add_(
                      value, multiply_(
                               static_cast<ResultScalar>(values_[current]),
                               static_cast<ResultScalar>(rhs.data()[column_indices_[current]])));
                } else {
                    value = add_(
                      value, multiply_(
                               static_cast<ResultScalar>(values_[current]),
                               static_cast<ResultScalar>(rhs[column_indices_[current]])));
                }
            }
            result_data[row] = value;
        }
        return result;
    }

    /// @brief multiplies matching dense expressions into an independent row-major matrix
    template <internals::matrix_expression RhsXprType>
        requires(std::remove_cvref_t<RhsXprType>::Cols != 1)
    auto operator*(const RhsXprType& rhs) const {
        using RhsScalar = std::remove_cv_t<typename std::remove_cvref_t<RhsXprType>::Scalar>;
        using ResultScalar = std::common_type_t<Scalar, RhsScalar>;
        fdapde_strong_assert(
          rhs.rows() == cols_, std::invalid_argument, "sparse-dense product requires matching inner dimensions");
        const Index result_cols = rhs.cols();
        Matrix<ResultScalar, Dynamic, Dynamic> result(rows_, result_cols);
        ResultScalar* result_data = result.data();
        for (Index row = 0; row < rows_; ++row) {
            for (Index current = row_offsets_[row]; current < row_offsets_[row + 1]; ++current) {
                const Index inner = column_indices_[current];
                const ResultScalar lhs = static_cast<ResultScalar>(values_[current]);
                for (Index col = 0; col < result_cols; ++col) {
                    if constexpr (has_plain_dense_storage_<RhsXprType>) {
                        constexpr int RhsStorageOrder = std::remove_cvref_t<RhsXprType>::StorageOrder;
                        const Index rhs_index =
                          RhsStorageOrder == RowMajor ? inner * result_cols + col : col * rhs.rows() + inner;
                        auto& value = result_data[row * result_cols + col];
                        value = add_(value, multiply_(lhs, static_cast<ResultScalar>(rhs.data()[rhs_index])));
                    } else {
                        auto& value = result_data[row * result_cols + col];
                        value = add_(value, multiply_(lhs, static_cast<ResultScalar>(rhs(inner, col))));
                    }
                }
            }
        }
        return result;
    }

    /// @brief sums each stored row into a dense vector, rejecting integral overflow
    Vector<Scalar, Dynamic> row_sums() const {
        Vector<Scalar, Dynamic> result(rows_);
        for (Index row = 0; row < rows_; ++row) {
            Scalar value {};
            for (Index current = row_offsets_[row]; current < row_offsets_[row + 1]; ++current) {
                value = add_(value, values_[current]);
            }
            result[row] = value;
        }
        return result;
    }

    /// @brief extracts the diagonal of a square matrix, using zero for absent entries
    Vector<Scalar, Dynamic> diagonal() const {
        fdapde_strong_assert(rows_ == cols_, std::invalid_argument, "sparse diagonal requires a square matrix");
        Vector<Scalar, Dynamic> result(rows_);
        for (Index row = 0; row < rows_; ++row) {
            const Index position = find_position_(row, row);
            if (position != missing_) result[row] = values_[position];
        }
        return result;
    }

    /// @brief builds a square diagonal owner from a vector expression with checked integral conversion
    template <internals::matrix_expression DiagonalXprType>
        requires(std::remove_cvref_t<DiagonalXprType>::Rows == 1 || std::remove_cvref_t<DiagonalXprType>::Cols == 1)
    static SparseMatrix from_diagonal(const DiagonalXprType& diagonal) {
        validate_shape_(diagonal.size(), diagonal.size());
        std::vector<triplet_type> triplets;
        triplets.reserve(static_cast<std::size_t>(diagonal.size()));
        for (Index i = 0; i < diagonal.size(); ++i) {
            const Scalar value = diagonal_cast_(diagonal[i]);
            if (value != Scalar {}) triplets.emplace_back(i, i, value);
        }
        return SparseMatrix(diagonal.size(), diagonal.size(), triplets);
    }

    /// @brief evaluates the bilinear form x-transpose A x for a matching row or column vector
    template <internals::matrix_expression VectorXprType>
        requires(std::remove_cvref_t<VectorXprType>::Rows == 1 || std::remove_cvref_t<VectorXprType>::Cols == 1)
    auto quadratic_form(const VectorXprType& vector) const {
        using VectorScalar = std::remove_cv_t<typename std::remove_cvref_t<VectorXprType>::Scalar>;
        using ResultScalar = std::common_type_t<Scalar, VectorScalar>;
        fdapde_strong_assert(rows_ == cols_, std::invalid_argument, "sparse quadratic form requires a square matrix");
        fdapde_strong_assert(
          vector.size() == cols_, std::invalid_argument, "sparse quadratic form requires a matching vector");
        ResultScalar result {};
        for (Index row = 0; row < rows_; ++row) {
            const ResultScalar lhs = static_cast<ResultScalar>(vector[row]);
            for (Index current = row_offsets_[row]; current < row_offsets_[row + 1]; ++current) {
                result = add_(
                  result, multiply_(
                            multiply_(lhs, static_cast<ResultScalar>(values_[current])),
                            static_cast<ResultScalar>(vector[column_indices_[current]])));
            }
        }
        return result;
    }

    /// @brief exchanges dimensions and storage without allocating
    void swap(SparseMatrix& other) noexcept {
        using std::swap;
        swap(rows_, other.rows_);
        swap(cols_, other.cols_);
        row_offsets_.swap(other.row_offsets_);
        column_indices_.swap(other.column_indices_);
        values_.swap(other.values_);
    }
    /// @brief exchanges two matrices through their storage swap
    friend void swap(SparseMatrix& lhs, SparseMatrix& rhs) noexcept { lhs.swap(rhs); }
   private:
    static constexpr Index missing_ = -1;

    template <typename XprType_>
    static constexpr bool has_plain_dense_storage_ = [] {
        using XprType = std::remove_cvref_t<XprType_>;
        return std::same_as<
                 XprType, Matrix<typename XprType::Scalar, XprType::Rows, XprType::Cols, XprType::StorageOrder>> ||
               std::same_as<
                 XprType, MatrixView<typename XprType::Scalar, XprType::Rows, XprType::Cols, XprType::StorageOrder>>;
    }();

    /// @brief adds coefficients and rejects integral overflow before evaluating the sum
    template <typename Value> static Value add_(const Value& lhs, const Value& rhs) {
        if constexpr (std::is_integral_v<Value>) {
            if constexpr (std::is_signed_v<Value>) {
                fdapde_strong_assert(
                  (rhs <= 0 || lhs <= std::numeric_limits<Value>::max() - rhs) &&
                    (rhs >= 0 || lhs >= std::numeric_limits<Value>::min() - rhs),
                  std::overflow_error, "SparseMatrix sum exceeds the scalar range");
            } else {
                fdapde_strong_assert(
                  lhs <= std::numeric_limits<Value>::max() - rhs, std::overflow_error,
                  "SparseMatrix sum exceeds the scalar range");
            }
        }
        return lhs + rhs;
    }

    /// @brief multiplies coefficients after checking the integral result range
    template <typename Value> static Value multiply_(Value lhs, Value rhs) {
        if constexpr (std::is_integral_v<Value>) {
            if (lhs == 0 || rhs == 0) return 0;
            constexpr Value max = std::numeric_limits<Value>::max();
            if constexpr (std::is_signed_v<Value>) {
                constexpr Value min = std::numeric_limits<Value>::min();
                const bool fits = lhs > 0 ? (rhs > 0 ? lhs <= max / rhs : rhs >= min / lhs) :
                                            (rhs > 0 ? lhs >= min / rhs : rhs >= max / lhs);
                fdapde_strong_assert(fits, std::overflow_error, "SparseMatrix product exceeds the scalar range");
            } else {
                fdapde_strong_assert(
                  lhs <= max / rhs, std::overflow_error, "SparseMatrix product exceeds the scalar range");
            }
        }
        return lhs * rhs;
    }

    /// @brief converts a diagonal coefficient without out-of-range integral narrowing
    template <typename Value> static Scalar diagonal_cast_(Value value) {
        if constexpr (std::is_integral_v<Scalar>) {
            if constexpr (std::is_integral_v<Value>) {
                fdapde_strong_assert(
                  std::cmp_greater_equal(+value, +std::numeric_limits<Scalar>::min()) &&
                    std::cmp_less_equal(+value, +std::numeric_limits<Scalar>::max()),
                  std::overflow_error, "SparseMatrix diagonal coefficient exceeds the scalar range");
            } else if constexpr (std::is_floating_point_v<Value>) {
                const long double truncated = std::trunc(static_cast<long double>(value));
                const long double limit = std::ldexp(1.0L, std::numeric_limits<Scalar>::digits);
                fdapde_strong_assert(
                  truncated >= (std::is_signed_v<Scalar> ? -limit : 0.0L) && truncated < limit, std::overflow_error,
                  "SparseMatrix diagonal coefficient exceeds the scalar range");
            }
        }
        return static_cast<Scalar>(value);
    }

    /// @brief checks nonnegative dimensions and room for the terminal CSR row offset
    static void validate_shape_(Index rows, Index cols) {
        fdapde_strong_assert(
          rows >= 0 && cols >= 0, std::invalid_argument, "SparseMatrix dimensions must be nonnegative");
        fdapde_strong_assert(
          rows < std::numeric_limits<Index>::max(), std::length_error,
          "SparseMatrix row-offset storage exceeds the supported int range");
    }
    /// @brief checks a public row index against the current shape
    void validate_row_(Index row) const {
        fdapde_strong_assert(row >= 0 && row < rows_, std::out_of_range, "SparseMatrix row index is out of range");
    }
    /// @brief checks coefficient and incoming-triplet indices against the current shape
    void validate_index_(Index row, Index col) const {
        fdapde_strong_assert(
          row >= 0 && row < rows_ && col >= 0 && col < cols_, std::out_of_range,
          "SparseMatrix coefficient index is out of range");
    }
    /// @brief allocates empty row offsets before replacing the shape and clearing entries
    void reset_shape_(Index rows, Index cols) {
        validate_shape_(rows, cols);
        std::vector<Index> offsets(static_cast<std::size_t>(rows) + 1, 0);
        rows_ = rows;
        cols_ = cols;
        row_offsets_.swap(offsets);
        column_indices_.clear();
        values_.clear();
    }
    /// @brief compresses checked triplets into local buffers before publishing CSR storage
    void build_(const std::vector<triplet_type>& triplets) {
        fdapde_strong_assert(
          triplets.size() <= static_cast<std::size_t>(std::numeric_limits<Index>::max()), std::length_error,
          "SparseMatrix triplet count exceeds the supported int range");
        for (const auto& triplet : triplets) validate_index_(triplet.row(), triplet.col());

        std::vector<Index> offsets(static_cast<std::size_t>(rows_) + 1, 0);
        std::vector<Index> columns;
        std::vector<Scalar> values;
        if (static_cast<std::size_t>(cols_) <= triplets.size()) {
            // group by column to combine duplicates and emit each CSR row in sorted column order
            std::vector<Index> column_offsets(static_cast<std::size_t>(cols_) + 1, 0);
            for (const auto& triplet : triplets) ++column_offsets[triplet.col() + 1];
            for (Index col = 0; col < cols_; ++col) column_offsets[col + 1] += column_offsets[col];

            std::vector<Index> next(column_offsets);
            std::vector<Index> temporary_rows(triplets.size());
            std::vector<Scalar> temporary_values(triplets.size());
            for (const auto& triplet : triplets) {
                const Index position = next[triplet.col()]++;
                temporary_rows[position] = triplet.row();
                temporary_values[position] = triplet.value();
            }

            std::vector<Index> marker(static_cast<std::size_t>(rows_), missing_);
            Index count = 0;
            for (Index col = 0; col < cols_; ++col) {
                const Index start = count;
                const Index old_end = column_offsets[col + 1];
                for (Index current = column_offsets[col]; current < old_end; ++current) {
                    const Index row = temporary_rows[current];
                    if (marker[row] >= start) {
                        temporary_values[marker[row]] = add_(temporary_values[marker[row]], temporary_values[current]);
                    } else {
                        temporary_rows[count] = row;
                        temporary_values[count] = temporary_values[current];
                        marker[row] = count;
                        ++count;
                    }
                }
                column_offsets[col] = start;
            }
            column_offsets[cols_] = count;

            for (Index col = 0; col < cols_; ++col) {
                for (Index current = column_offsets[col]; current < column_offsets[col + 1]; ++current) {
                    if (temporary_values[current] != Scalar {}) ++offsets[temporary_rows[current] + 1];
                }
            }
            for (Index row = 0; row < rows_; ++row) offsets[row + 1] += offsets[row];

            columns.resize(static_cast<std::size_t>(offsets.back()));
            values.resize(static_cast<std::size_t>(offsets.back()));
            next = offsets;
            for (Index col = 0; col < cols_; ++col) {
                for (Index current = column_offsets[col]; current < column_offsets[col + 1]; ++current) {
                    if (temporary_values[current] == Scalar {}) continue;
                    const Index position = next[temporary_rows[current]]++;
                    columns[position] = col;
                    values[position] = std::move(temporary_values[current]);
                }
            }
        } else {
            // sort within rows when a column-sized workspace would exceed the number of triplets
            std::vector<Index> input_offsets(static_cast<std::size_t>(rows_) + 1, 0);
            for (const auto& triplet : triplets) ++input_offsets[triplet.row() + 1];
            for (Index row = 0; row < rows_; ++row) input_offsets[row + 1] += input_offsets[row];

            std::vector<Index> next(input_offsets);
            std::vector<triplet_type> grouped(triplets.size());
            for (const auto& triplet : triplets) grouped[next[triplet.row()]++] = triplet;

            columns.reserve(triplets.size());
            values.reserve(triplets.size());
            for (Index row = 0; row < rows_; ++row) {
                const auto first = grouped.begin() + input_offsets[row];
                const auto last = grouped.begin() + input_offsets[row + 1];
                std::stable_sort(
                  first, last, [](const triplet_type& lhs, const triplet_type& rhs) { return lhs.col() < rhs.col(); });
                auto current = first;
                while (current != last) {
                    const Index col = current->col();
                    Scalar value = current->value();
                    ++current;
                    while (current != last && current->col() == col) {
                        value = add_(value, current->value());
                        ++current;
                    }
                    if (value != Scalar {}) {
                        columns.push_back(col);
                        values.push_back(std::move(value));
                    }
                }
                offsets[row + 1] = static_cast<Index>(values.size());
            }
        }

        row_offsets_.swap(offsets);
        column_indices_.swap(columns);
        values_.swap(values);
    }
    /// @brief binary-searches a previously validated row for its column or the missing sentinel
    Index find_position_(Index row, Index col) const {
        const Index begin = row_offsets_[row];
        const Index end = row_offsets_[row + 1];
        const auto first = column_indices_.begin() + begin;
        const auto last = column_indices_.begin() + end;
        const auto found = std::lower_bound(first, last, col);
        if (found == last || *found != col) return missing_;
        return static_cast<Index>(found - column_indices_.begin());
    }

    Index rows_ = 0;
    Index cols_ = 0;
    std::vector<Index> row_offsets_;
    std::vector<Index> column_indices_;
    std::vector<Scalar> values_;
};

}   // namespace fdapde

#endif   // __FDAPDE_LINALG_SPARSE_H__
