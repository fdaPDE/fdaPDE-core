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

#ifndef __FDAPDE_SPARSE_BLOCK_MATRIX_H__
#define __FDAPDE_SPARSE_BLOCK_MATRIX_H__

#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "header_check.h"

namespace fdapde {
namespace internals {

/// @brief identifies types that are not native CSR owners
template <typename T> struct is_native_sparse_matrix : std::false_type { };
/// @brief identifies native CSR owners for eager block materialization
template <typename Scalar> struct is_native_sparse_matrix<SparseMatrix<Scalar>> : std::true_type { };
template <typename T>
inline constexpr bool is_native_sparse_matrix_v = is_native_sparse_matrix<std::remove_cvref_t<T>>::value;

}   // namespace internals

/// @brief owns a fixed grid of native CSR blocks with checked runtime row and column partitions
template <typename Scalar_, int BlockRows_, int BlockCols_> class SparseBlockMatrix {
   public:
    using Index = int;
    using Scalar = std::remove_cvref_t<Scalar_>;
    using block_type = SparseMatrix<Scalar>;
    using triplet_type = Triplet<Scalar>;

    fdapde_static_assert(BlockRows_ > 0 && BlockCols_ > 0, INVALID_SPARSE_BLOCK_GRID_DIMENSIONS);
    fdapde_static_assert(BlockRows_ > 1 || BlockCols_ > 1, SPARSE_BLOCK_MATRIX_REQUIRES_MULTIPLE_BLOCKS);
   private:
    static constexpr std::uint64_t block_count_64_ =
      static_cast<std::uint64_t>(BlockRows_) * static_cast<std::uint64_t>(BlockCols_);
    static constexpr bool has_supported_block_count_ =
      block_count_64_ <= static_cast<std::uint64_t>(std::numeric_limits<Index>::max());
    fdapde_static_assert(has_supported_block_count_, MATRIX_SIZE_EXCEEDS_SUPPORTED_RANGE);
    static constexpr std::size_t block_count_ =
      has_supported_block_count_ ? static_cast<std::size_t>(block_count_64_) : 0;

    template <typename T>
    static constexpr bool is_matrix_block_ = [] {
        using Block = std::remove_cvref_t<T>;
        if constexpr (
          internals::matrix_expression<Block> || internals::is_native_sparse_matrix_v<Block> ||
          requires(const Block& value) {
              requires internals::is_native_sparse_matrix_v<decltype(value.to_sparse())>;
          }) {
            return std::convertible_to<typename Block::Scalar, Scalar>;
        } else {
            return false;
        }
    }();

    template <typename Extents>
    static constexpr bool is_extent_container_ = requires(const Extents& extents, Index i) {
        { extents.size() } -> std::convertible_to<std::size_t>;
        requires std::integral<std::remove_cvref_t<decltype(extents[i])>>;
    };
   public:
    static constexpr Index BlockRows = BlockRows_;
    static constexpr Index BlockCols = BlockCols_;

    /// @brief constructs an empty partition with zero global dimensions
    SparseBlockMatrix() = default;

    /// @brief constructs an empty block grid from checked per-row and per-column extents
    template <typename RowExtents, typename ColExtents>
        requires(is_extent_container_<RowExtents> && is_extent_container_<ColExtents>)
    SparseBlockMatrix(const RowExtents& row_extents, const ColExtents& col_extents) {
        fdapde_strong_assert(
          !(static_cast<std::size_t>(row_extents.size()) != static_cast<std::size_t>(BlockRows_) ||
            static_cast<std::size_t>(col_extents.size()) != static_cast<std::size_t>(BlockCols_)),
          std::invalid_argument, "SparseBlockMatrix extent counts must match the block grid");
        std::array<Index, BlockRows_> rows {};
        std::array<Index, BlockCols_> cols {};
        for (Index i = 0; i < BlockRows_; ++i) rows[i] = checked_extent_(row_extents[i]);
        for (Index i = 0; i < BlockCols_; ++i) cols[i] = checked_extent_(col_extents[i]);
        configure_(rows, cols);
    }

    /// @brief constructs an empty grid with uniform block extents
    SparseBlockMatrix(Index block_rows, Index block_cols) {
        fdapde_strong_assert(
          !(block_rows < 0 || block_cols < 0), std::invalid_argument,
          "SparseBlockMatrix block dimensions must be nonnegative");
        std::array<Index, BlockRows_> rows {};
        std::array<Index, BlockCols_> cols {};
        rows.fill(block_rows);
        cols.fill(block_cols);
        configure_(rows, cols);
    }

    /// @brief materializes row-major block arguments, inferring zero-placeholder extents from neighboring blocks
    template <typename... Blocks>
        requires(
          sizeof...(Blocks) == block_count_ &&
          ((is_matrix_block_<Blocks> || std::convertible_to<Blocks, Scalar>) && ...))
    explicit SparseBlockMatrix(Blocks&&... blocks) {
        std::array<Index, block_count_> row_dimensions {};
        std::array<Index, block_count_> col_dimensions {};
        row_dimensions.fill(-1);
        col_dimensions.fill(-1);

        internals::for_each_index_and_args<static_cast<int>(block_count_)>(
          [&]<int I, typename Block>(Block&& block) {
              if constexpr (is_matrix_block_<Block>) {
                  row_dimensions[I] = checked_extent_(block.rows());
                  col_dimensions[I] = checked_extent_(block.cols());
              } else
                  fdapde_strong_assert(
                    !(block_type::checked_scalar_cast_(block) != Scalar {}), std::invalid_argument,
                    "SparseBlockMatrix scalar placeholders must be zero");
          },
          std::forward<Blocks>(blocks)...);

        std::array<Index, BlockRows_> rows {};
        std::array<Index, BlockCols_> cols {};
        for (Index block_row = 0; block_row < BlockRows_; ++block_row) {
            Index extent = -1;
            for (Index block_col = 0; block_col < BlockCols_; ++block_col) {
                const Index value = row_dimensions[flat_index_(block_row, block_col)];
                if (value < 0) continue;
                fdapde_strong_assert(
                  !(extent >= 0 && value != extent), std::invalid_argument,
                  "SparseBlockMatrix blocks in a block row must have equal rows");
                extent = value;
            }
            rows[block_row] = extent < 0 ? 1 : extent;
        }
        for (Index block_col = 0; block_col < BlockCols_; ++block_col) {
            Index extent = -1;
            for (Index block_row = 0; block_row < BlockRows_; ++block_row) {
                const Index value = col_dimensions[flat_index_(block_row, block_col)];
                if (value < 0) continue;
                fdapde_strong_assert(
                  !(extent >= 0 && value != extent), std::invalid_argument,
                  "SparseBlockMatrix blocks in a block column must have equal columns");
                extent = value;
            }
            cols[block_col] = extent < 0 ? 1 : extent;
        }

        configure_(rows, cols);
        internals::for_each_index_and_args<static_cast<int>(block_count_)>(
          [&]<int I, typename Block>(Block&& block) {
              if constexpr (is_matrix_block_<Block>) blocks_[I] = materialize_(block);
          },
          std::forward<Blocks>(blocks)...);
    }

    /// @brief copies the partition and coefficients into independent storage
    SparseBlockMatrix(const SparseBlockMatrix&) = default;
    /// @brief replaces the grid with an independent copy only after copying succeeds
    SparseBlockMatrix& operator=(const SparseBlockMatrix& other) {
        if (this == &other) return *this;
        SparseBlockMatrix replacement(other);
        swap(replacement);
        return *this;
    }
    /// @brief takes the partition and coefficients, leaving an empty source
    SparseBlockMatrix(SparseBlockMatrix&& other) noexcept { swap(other); }
    /// @brief takes the partition and coefficients while tolerating self-move
    SparseBlockMatrix& operator=(SparseBlockMatrix&& other) noexcept {
        if (this == &other) return *this;
        SparseBlockMatrix replacement(std::move(other));
        swap(replacement);
        return *this;
    }

    /// @brief returns the global row count
    constexpr Index rows() const { return rows_; }
    /// @brief returns the global column count
    constexpr Index cols() const { return cols_; }
    /// @brief returns the compile-time block-row count
    static constexpr Index block_rows() { return BlockRows_; }
    /// @brief returns the compile-time block-column count
    static constexpr Index block_cols() { return BlockCols_; }

    /// @brief borrows a const block from a checked position of an lvalue grid
    const block_type& block(Index row, Index col) const& {
        validate_block_index_(row, col);
        return blocks_[flat_index_(row, col)];
    }
    /// @brief borrows a mutable block whose shape must continue matching the partition
    block_type& block(Index row, Index col) & {
        validate_block_index_(row, col);
        return blocks_[flat_index_(row, col)];
    }

    /// @brief rejects borrowing a block from a temporary owner
    void block(Index, Index) const&& = delete;

    /// @brief counts all stored block entries, including explicitly stored zeros
    std::size_t non_zeros() const {
        validate_block_shapes_();
        std::size_t result = 0;
        for (const auto& current : blocks_) {
            const std::size_t increment = static_cast<std::size_t>(current.non_zeros());
            fdapde_strong_assert(
              !(result > std::numeric_limits<std::size_t>::max() - increment), std::length_error,
              "SparseBlockMatrix nonzero count exceeds the supported range");
            result += increment;
        }
        return result;
    }
    /// @brief maps a checked global row to its block row, skipping empty extents
    Index row_block(Index row) const {
        validate_global_row_(row);
        return block_index_(row_offsets_, row);
    }
    /// @brief maps a checked global column to its block column, skipping empty extents
    Index col_block(Index col) const {
        validate_global_col_(col);
        return block_index_(col_offsets_, col);
    }
    /// @brief maps a checked global row to its local coordinate within a block
    Index local_row(Index row) const {
        const Index block_row = row_block(row);
        return row - row_offsets_[block_row];
    }
    /// @brief maps a checked global column to its local coordinate within a block
    Index local_col(Index col) const {
        const Index block_col = col_block(col);
        return col - col_offsets_[block_col];
    }

    /// @brief returns a checked global coefficient by value, using zero for absent entries
    Scalar coeff(Index row, Index col) const {
        validate_block_shapes_();
        const Index block_row = row_block(row);
        const Index block_col = col_block(col);
        return blocks_[flat_index_(block_row, block_col)].coeff(
          row - row_offsets_[block_row], col - col_offsets_[block_col]);
    }
    /// @brief reports whether a checked global position has a stored coefficient
    bool contains(Index row, Index col) const {
        validate_block_shapes_();
        const Index block_row = row_block(row);
        const Index block_col = col_block(col);
        return blocks_[flat_index_(block_row, block_col)].contains(
          row - row_offsets_[block_row], col - col_offsets_[block_col]);
    }
    /// @brief borrows an existing global coefficient without changing the stored pattern
    Scalar& value_ref(Index row, Index col) & {
        validate_block_shapes_();
        const Index block_row = row_block(row);
        const Index block_col = col_block(col);
        return blocks_[flat_index_(block_row, block_col)].value_ref(
          row - row_offsets_[block_row], col - col_offsets_[block_col]);
    }
    /// @brief borrows a global coefficient, atomically inserting a stored zero if absent
    Scalar& coeff_ref(Index row, Index col) & {
        validate_block_shapes_();
        const Index block_row = row_block(row);
        const Index block_col = col_block(col);
        return blocks_[flat_index_(block_row, block_col)].coeff_ref(
          row - row_offsets_[block_row], col - col_offsets_[block_col]);
    }

    /// @brief copies the partition into sorted owning CSR storage, preserving stored zeros
    SparseMatrix<Scalar> to_sparse() const {
        validate_block_shapes_();
        const std::size_t count = non_zeros();
        fdapde_strong_assert(
          count <= static_cast<std::size_t>(std::numeric_limits<Index>::max()), std::length_error,
          "SparseBlockMatrix CSR storage exceeds the supported int range");
        block_type result(rows_, cols_);
        result.column_indices_.reserve(count);
        result.values_.reserve(count);
        // visit local rows across increasing block columns to emit already sorted CSR storage
        for (Index block_row = 0; block_row < BlockRows_; ++block_row) {
            for (Index row = 0; row < row_extents_[block_row]; ++row) {
                for (Index block_col = 0; block_col < BlockCols_; ++block_col) {
                    for (const auto entry : blocks_[flat_index_(block_row, block_col)].row(row)) {
                        result.column_indices_.push_back(col_offsets_[block_col] + entry.column());
                        result.values_.push_back(entry.value());
                    }
                }
                result.row_offsets_[row_offsets_[block_row] + row + 1] = static_cast<Index>(result.values_.size());
            }
        }
        return result;
    }

    /// @brief materializes all block coefficients into an independent dense matrix
    Matrix<Scalar, Dynamic, Dynamic> to_dense() const {
        validate_block_shapes_();
        Matrix<Scalar, Dynamic, Dynamic> result(rows_, cols_);
        result.set_zero();
        for (Index block_row = 0; block_row < BlockRows_; ++block_row) {
            for (Index block_col = 0; block_col < BlockCols_; ++block_col) {
                const auto& current = blocks_[flat_index_(block_row, block_col)];
                for (Index row = 0; row < current.rows(); ++row) {
                    for (const auto entry : current.row(row)) {
                        result(row_offsets_[block_row] + row, col_offsets_[block_col] + entry.column()) = entry.value();
                    }
                }
            }
        }
        return result;
    }

    /// @brief atomically replaces selected global rows and columns with unit diagonals
    void rebuild_with_constraints(const std::vector<Index>& dofs) {
        if (dofs.empty()) return;
        fdapde_strong_assert(
          !(rows_ != cols_), std::invalid_argument, "SparseBlockMatrix constraints require a square matrix");
        SparseMatrix<Scalar> constrained = to_sparse();
        constrained.rebuild_with_constraints(dofs);

        std::vector<triplet_type> triplets;
        triplets.reserve(static_cast<std::size_t>(constrained.non_zeros()));
        for (Index row = 0; row < constrained.rows(); ++row) {
            for (const auto entry : constrained.row(row)) { triplets.emplace_back(row, entry.column(), entry.value()); }
        }
        SparseBlockMatrix replacement(row_extents_, col_extents_);
        replacement.rebuild(triplets);
        swap(replacement);
    }

    /// @brief atomically rebuilds every block from checked global-coordinate triplets
    template <typename TripletList> void rebuild(const TripletList& triplets) {
        std::vector<std::vector<triplet_type>> block_triplets(block_count_);
        for (const auto& triplet : triplets) {
            const Index block_row = checked_triplet_block_row_(triplet.row());
            const Index block_col = checked_triplet_block_col_(triplet.col());
            block_triplets[flat_index_(block_row, block_col)].emplace_back(
              triplet.row() - row_offsets_[block_row], triplet.col() - col_offsets_[block_col],
              block_type::checked_scalar_cast_(triplet.value()));
        }

        SparseBlockMatrix replacement(row_extents_, col_extents_);
        for (std::size_t i = 0; i < block_count_; ++i) replacement.blocks_[i].rebuild(block_triplets[i]);
        swap(replacement);
    }
    /// @brief rebuilds the grid from a global-coordinate triplet initializer list
    void rebuild(std::initializer_list<triplet_type> triplets) {
        rebuild<std::initializer_list<triplet_type>>(triplets);
    }

    /// @brief atomically replaces one block from checked local-coordinate triplets
    template <typename TripletList> void rebuild_block(Index row, Index col, const TripletList& triplets) {
        validate_block_index_(row, col);
        std::vector<triplet_type> local_triplets;
        for (const auto& triplet : triplets) {
            local_triplets.emplace_back(
              triplet.row(), triplet.col(), block_type::checked_scalar_cast_(triplet.value()));
        }
        block_type replacement(row_extents_[row], col_extents_[col], local_triplets);
        blocks_[flat_index_(row, col)].swap(replacement);
    }
    /// @brief replaces one block from a local-coordinate triplet initializer list
    void rebuild_block(Index row, Index col, std::initializer_list<triplet_type> triplets) {
        rebuild_block<std::initializer_list<triplet_type>>(row, col, triplets);
    }

    /// @brief exchanges partitions and block storage without allocation
    void swap(SparseBlockMatrix& other) noexcept {
        using std::swap;
        for (std::size_t i = 0; i < block_count_; ++i) blocks_[i].swap(other.blocks_[i]);
        swap(row_extents_, other.row_extents_);
        swap(col_extents_, other.col_extents_);
        swap(row_offsets_, other.row_offsets_);
        swap(col_offsets_, other.col_offsets_);
        swap(rows_, other.rows_);
        swap(cols_, other.cols_);
    }
    /// @brief exchanges two grids through their storage swap
    friend void swap(SparseBlockMatrix& lhs, SparseBlockMatrix& rhs) noexcept { lhs.swap(rhs); }
   private:
    /// @brief converts a public extent to a nonnegative supported int value
    template <std::integral Extent> static Index checked_extent_(Extent value) {
        fdapde_strong_assert(
          std::in_range<Index>(value), std::length_error, "SparseBlockMatrix extent exceeds the supported int range");
        const Index result = static_cast<Index>(value);
        fdapde_strong_assert(!(result < 0), std::invalid_argument, "SparseBlockMatrix extents must be nonnegative");
        return result;
    }

    /// @brief maps validated block coordinates to the row-major storage array
    static constexpr std::size_t flat_index_(Index row, Index col) {
        return static_cast<std::size_t>(row) * static_cast<std::size_t>(BlockCols_) + static_cast<std::size_t>(col);
    }

    /// @brief builds partition prefix sums while rejecting global dimension overflow
    template <std::size_t N>
    static Index fill_offsets_(const std::array<Index, N>& extents, std::array<Index, N + 1>& offsets) {
        Index total = 0;
        offsets[0] = 0;
        for (std::size_t i = 0; i < N; ++i) {
            fdapde_strong_assert(
              !(extents[i] > std::numeric_limits<Index>::max() - total), std::length_error,
              "SparseBlockMatrix dimensions exceed the supported int range");
            total += extents[i];
            offsets[i + 1] = total;
        }
        return total;
    }

    /// @brief allocates empty blocks before publishing a validated partition
    void
    configure_(const std::array<Index, BlockRows_>& row_extents, const std::array<Index, BlockCols_>& col_extents) {
        std::array<Index, BlockRows_ + 1> row_offsets {};
        std::array<Index, BlockCols_ + 1> col_offsets {};
        const Index rows = fill_offsets_(row_extents, row_offsets);
        const Index cols = fill_offsets_(col_extents, col_offsets);

        std::array<block_type, block_count_> blocks {};
        for (Index row = 0; row < BlockRows_; ++row) {
            for (Index col = 0; col < BlockCols_; ++col) {
                blocks[flat_index_(row, col)] = block_type(row_extents[row], col_extents[col]);
            }
        }
        row_extents_ = row_extents;
        col_extents_ = col_extents;
        row_offsets_ = row_offsets;
        col_offsets_ = col_offsets;
        rows_ = rows;
        cols_ = cols;
        blocks_.swap(blocks);
    }

    /// @brief copies native CSR storage with checked scalar conversion and preserves its pattern
    template <typename SparseBlock> block_type materialize_sparse_(const SparseBlock& block) const {
        if constexpr (std::is_same_v<typename SparseBlock::Scalar, Scalar>) return block;
        block_type result(block.rows(), block.cols());
        result.row_offsets_ = block.row_offsets_;
        result.column_indices_ = block.column_indices_;
        result.values_.reserve(static_cast<std::size_t>(block.non_zeros()));
        for (const auto& value : block.values_) result.values_.push_back(block_type::checked_scalar_cast_(value));
        return result;
    }

    /// @brief materializes native sparse, nested block or dense expressions into owning CSR coefficients
    template <typename Block> block_type materialize_(const Block& block) const {
        if constexpr (internals::is_native_sparse_matrix_v<Block>) {
            return materialize_sparse_(block);
        } else if constexpr (requires(const Block& value) {
                                 requires internals::is_native_sparse_matrix_v<decltype(value.to_sparse())>;
                             }) {
            const auto sparse = block.to_sparse();
            return materialize_sparse_(sparse);
        } else {
            const Index rows = checked_extent_(block.rows());
            const Index cols = checked_extent_(block.cols());
            std::vector<triplet_type> triplets;
            for (Index row = 0; row < rows; ++row) {
                for (Index col = 0; col < cols; ++col) {
                    const Scalar value = block_type::checked_scalar_cast_(block(row, col));
                    if (value != Scalar {}) triplets.emplace_back(row, col, value);
                }
            }
            return block_type(rows, cols, triplets);
        }
    }

    /// @brief checks a public block coordinate against the fixed grid
    void validate_block_index_(Index row, Index col) const {
        fdapde_strong_assert(
          !(row < 0 || row >= BlockRows_ || col < 0 || col >= BlockCols_), std::out_of_range,
          "SparseBlockMatrix block index is out of range");
    }
    /// @brief checks a public row coordinate against the global shape
    void validate_global_row_(Index row) const {
        fdapde_strong_assert(
          !(row < 0 || row >= rows_), std::out_of_range, "SparseBlockMatrix row index is out of range");
    }
    /// @brief checks a public column coordinate against the global shape
    void validate_global_col_(Index col) const {
        fdapde_strong_assert(
          !(col < 0 || col >= cols_), std::out_of_range, "SparseBlockMatrix column index is out of range");
    }
    /// @brief rejects borrowed blocks whose structural mutation violates the partition
    void validate_block_shapes_() const {
        for (Index row = 0; row < BlockRows_; ++row) {
            for (Index col = 0; col < BlockCols_; ++col) {
                const auto& current = blocks_[flat_index_(row, col)];
                fdapde_strong_assert(
                  !(current.rows() != row_extents_[row] || current.cols() != col_extents_[col]), std::invalid_argument,
                  "SparseBlockMatrix mutable block no longer matches its partition");
            }
        }
    }

    /// @brief finds the nonempty partition interval containing a validated global coordinate
    template <std::size_t N> static Index block_index_(const std::array<Index, N>& offsets, Index index) {
        return static_cast<Index>(std::upper_bound(offsets.begin(), offsets.end(), index) - offsets.begin() - 1);
    }
    /// @brief validates an incoming triplet row and locates its block row
    Index checked_triplet_block_row_(Index row) const {
        validate_global_row_(row);
        return block_index_(row_offsets_, row);
    }
    /// @brief validates an incoming triplet column and locates its block column
    Index checked_triplet_block_col_(Index col) const {
        validate_global_col_(col);
        return block_index_(col_offsets_, col);
    }

    std::array<block_type, block_count_> blocks_ {};
    std::array<Index, BlockRows_> row_extents_ {};
    std::array<Index, BlockCols_> col_extents_ {};
    std::array<Index, BlockRows_ + 1> row_offsets_ {};
    std::array<Index, BlockCols_ + 1> col_offsets_ {};
    Index rows_ = 0;
    Index cols_ = 0;
};

}   // namespace fdapde

#endif   // __FDAPDE_SPARSE_BLOCK_MATRIX_H__
