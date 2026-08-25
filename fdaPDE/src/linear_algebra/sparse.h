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

template <typename Scalar_> class Triplet {
   public:
    using Index = int;
    using Scalar = std::remove_cvref_t<Scalar_>;

    constexpr Triplet() = default;
    constexpr Triplet(Index row, Index col, const Scalar& value) : row_(row), col_(col), value_(value) { }

    constexpr Index row() const { return row_; }
    constexpr Index col() const { return col_; }
    constexpr const Scalar& value() const { return value_; }
    constexpr Scalar& value() { return value_; }
   private:
    Index row_ = 0;
    Index col_ = 0;
    Scalar value_ {};
};

// Owning dynamic rectangular sparse matrix in compressed-row form. Construction
// and rebuild canonicalize the pattern: columns are sorted within each row,
// duplicates are summed in input order, and exact zero sums are omitted.
template <typename Scalar_> class SparseMatrix {
   public:
    using Index = int;
    using Scalar = std::remove_cvref_t<Scalar_>;
    using triplet_type = Triplet<Scalar>;

    static_assert(!std::is_same_v<Scalar, bool>, "SparseMatrix<bool> is not supported");

    class ConstEntry {
       public:
        constexpr Index column() const { return column_; }
        constexpr const Scalar& value() const { return *value_; }
       private:
        friend class SparseMatrix;
        constexpr ConstEntry(Index column, const Scalar* value) : column_(column), value_(value) { }

        Index column_ = 0;
        const Scalar* value_ = nullptr;
    };

    class ConstRowView {
       public:
        class const_iterator {
           public:
            using iterator_category = std::forward_iterator_tag;
            using difference_type = std::ptrdiff_t;
            using value_type = ConstEntry;
            using reference = ConstEntry;

            constexpr reference operator*() const { return ConstEntry(columns_[index_], values_ + index_); }
            constexpr const_iterator& operator++() {
                ++index_;
                return *this;
            }
            constexpr const_iterator operator++(int) {
                const_iterator result(*this);
                ++(*this);
                return result;
            }
            friend constexpr bool operator==(const const_iterator&, const const_iterator&) = default;
           private:
            friend class ConstRowView;
            constexpr const_iterator(const Index* columns, const Scalar* values, Index index) :
                columns_(columns), values_(values), index_(index) { }

            const Index* columns_ = nullptr;
            const Scalar* values_ = nullptr;
            Index index_ = 0;
        };

        constexpr const_iterator begin() const { return const_iterator(columns_, values_, begin_); }
        constexpr const_iterator end() const { return const_iterator(columns_, values_, end_); }
        constexpr Index size() const { return end_ - begin_; }
        constexpr bool empty() const { return begin_ == end_; }
       private:
        friend class SparseMatrix;
        constexpr ConstRowView(const Index* columns, const Scalar* values, Index begin, Index end) :
            columns_(columns), values_(values), begin_(begin), end_(end) { }

        const Index* columns_ = nullptr;
        const Scalar* values_ = nullptr;
        Index begin_ = 0;
        Index end_ = 0;
    };

    // Row views and their iterators follow vector-style invalidation: any
    // resize, rebuild, assignment, move, or swap of the matrix invalidates them.

    SparseMatrix() : row_offsets_(1, 0) { }
    SparseMatrix(Index rows, Index cols) { reset_shape_(rows, cols); }
    SparseMatrix(Index rows, Index cols, const std::vector<triplet_type>& triplets) {
        validate_shape_(rows, cols);
        rows_ = rows;
        cols_ = cols;
        build_(triplets);
    }
    SparseMatrix(Index rows, Index cols, std::initializer_list<triplet_type> triplets) :
        SparseMatrix(rows, cols, std::vector<triplet_type>(triplets)) { }

    SparseMatrix(const SparseMatrix&) = default;
    SparseMatrix& operator=(const SparseMatrix& other) {
        if (this == &other) return *this;
        SparseMatrix replacement(other);
        swap(replacement);
        return *this;
    }
    SparseMatrix(SparseMatrix&& other) noexcept :
        rows_(std::exchange(other.rows_, 0)),
        cols_(std::exchange(other.cols_, 0)),
        row_offsets_(std::move(other.row_offsets_)),
        column_indices_(std::move(other.column_indices_)),
        values_(std::move(other.values_)) { }
    SparseMatrix& operator=(SparseMatrix&& other) noexcept {
        if (this == &other) return *this;
        rows_ = std::exchange(other.rows_, 0);
        cols_ = std::exchange(other.cols_, 0);
        row_offsets_ = std::move(other.row_offsets_);
        column_indices_ = std::move(other.column_indices_);
        values_ = std::move(other.values_);
        return *this;
    }

    constexpr Index rows() const { return rows_; }
    constexpr Index cols() const { return cols_; }
    Index non_zeros() const { return static_cast<Index>(values_.size()); }

    Scalar coeff(Index row, Index col) const {
        validate_index_(row, col);
        const Index position = find_position_(row, col);
        return position == missing_ ? Scalar {} : values_[position];
    }
    bool contains(Index row, Index col) const {
        validate_index_(row, col);
        return find_position_(row, col) != missing_;
    }
    Scalar& value_ref(Index row, Index col) {
        validate_index_(row, col);
        const Index position = find_position_(row, col);
        if (position == missing_) {
            throw std::out_of_range("SparseMatrix value_ref requires an existing stored coefficient");
        }
        return values_[position];
    }
    ConstRowView row(Index row_index) const {
        validate_row_(row_index);
        return ConstRowView(
          column_indices_.data(), values_.data(), row_offsets_[row_index], row_offsets_[row_index + 1]);
    }

    // Structural changes are explicit and failure-atomic. value_ref preserves
    // the existing pattern even when a stored value becomes zero; resize
    // discards the pattern, while rebuild replaces it and elides exact zeros.
    void resize(Index rows, Index cols) {
        SparseMatrix replacement(rows, cols);
        swap(replacement);
    }
    void rebuild(const std::vector<triplet_type>& triplets) {
        SparseMatrix replacement(rows_, cols_, triplets);
        swap(replacement);
    }
    void rebuild(std::initializer_list<triplet_type> triplets) { rebuild(std::vector<triplet_type>(triplets)); }

    void swap(SparseMatrix& other) noexcept {
        using std::swap;
        swap(rows_, other.rows_);
        swap(cols_, other.cols_);
        row_offsets_.swap(other.row_offsets_);
        column_indices_.swap(other.column_indices_);
        values_.swap(other.values_);
    }
    friend void swap(SparseMatrix& lhs, SparseMatrix& rhs) noexcept { lhs.swap(rhs); }
   private:
    static constexpr Index missing_ = -1;

    static Scalar add_(const Scalar& lhs, const Scalar& rhs) {
        if constexpr (std::is_integral_v<Scalar>) {
            if constexpr (std::is_signed_v<Scalar>) {
                if (
                  (rhs > 0 && lhs > std::numeric_limits<Scalar>::max() - rhs) ||
                  (rhs < 0 && lhs < std::numeric_limits<Scalar>::min() - rhs)) {
                    throw std::overflow_error("SparseMatrix duplicate sum exceeds the scalar range");
                }
            } else if (lhs > std::numeric_limits<Scalar>::max() - rhs) {
                throw std::overflow_error("SparseMatrix duplicate sum exceeds the scalar range");
            }
        }
        return lhs + rhs;
    }

    static void validate_shape_(Index rows, Index cols) {
        if (rows < 0 || cols < 0) { throw std::invalid_argument("SparseMatrix dimensions must be nonnegative"); }
        if (rows == std::numeric_limits<Index>::max()) {
            throw std::length_error("SparseMatrix row-offset storage exceeds the supported int range");
        }
    }
    void validate_row_(Index row) const {
        if (row < 0 || row >= rows_) { throw std::out_of_range("SparseMatrix row index is out of range"); }
    }
    void validate_index_(Index row, Index col) const {
        if (row < 0 || row >= rows_ || col < 0 || col >= cols_) {
            throw std::out_of_range("SparseMatrix coefficient index is out of range");
        }
    }
    void reset_shape_(Index rows, Index cols) {
        validate_shape_(rows, cols);
        std::vector<Index> offsets(static_cast<std::size_t>(rows) + 1, 0);
        rows_ = rows;
        cols_ = cols;
        row_offsets_.swap(offsets);
        column_indices_.clear();
        values_.clear();
    }
    void build_(const std::vector<triplet_type>& triplets) {
        if (triplets.size() > static_cast<std::size_t>(std::numeric_limits<Index>::max())) {
            throw std::length_error("SparseMatrix triplet count exceeds the supported int range");
        }
        for (const auto& triplet : triplets) validate_index_(triplet.row(), triplet.col());

        std::vector<Index> offsets(static_cast<std::size_t>(rows_) + 1, 0);
        std::vector<Index> columns;
        std::vector<Scalar> values;
        if (static_cast<std::size_t>(cols_) <= triplets.size()) {
            // Linear-time construction through a column-grouped temporary.
            // Traversing its columns in order yields sorted CSR rows directly.
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
            // Avoid a column-sized workspace for extremely wide sparse shapes.
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
