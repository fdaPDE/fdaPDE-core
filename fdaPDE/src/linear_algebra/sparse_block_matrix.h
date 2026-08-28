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

template <typename T> struct is_native_sparse_matrix : std::false_type { };
template <typename Scalar> struct is_native_sparse_matrix<SparseMatrix<Scalar>> : std::true_type { };
template <typename T>
inline constexpr bool is_native_sparse_matrix_v = is_native_sparse_matrix<std::remove_cvref_t<T>>::value;

}   // namespace internals

// Owning fixed-grid composition of native CSR blocks. The grid dimensions are
// static; block and global extents are runtime values. Supplied matrix blocks
// are materialized immediately, so temporaries and copies are independent.
//
// The public storage contract is column-major traversal over owning CSR
// blocks. Generic sparse expressions are materialized explicitly.
template <typename Scalar_, int BlockRows_, int BlockCols_, int Options_ = ColMajor, typename StorageIndex_ = int>
    requires(Options_ == ColMajor && std::signed_integral<StorageIndex_>)
class SparseBlockMatrix {
   public:
    using Index = int;
    using Scalar = std::remove_cvref_t<Scalar_>;
    using StorageIndex = StorageIndex_;
    using Nested = const SparseBlockMatrix&;
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
    static constexpr int StorageOrder = Options_;

    // Iterators borrow an lvalue matrix and follow its blocks' vector-style
    // invalidation rules. Insert-on-miss, rebuild, mutable-block structural
    // edits, assignment, move, or swap invalidate them.
    class InnerIterator {
       public:
        InnerIterator() = default;
        InnerIterator(SparseBlockMatrix& matrix, Index outer) : matrix_(&matrix), mutable_matrix_(&matrix) {
            initialize_(outer);
        }
        InnerIterator(const SparseBlockMatrix& matrix, Index outer) : matrix_(&matrix) { initialize_(outer); }
        InnerIterator(SparseBlockMatrix&&, Index) = delete;
        InnerIterator(const SparseBlockMatrix&&, Index) = delete;

        InnerIterator& operator++() {
            validate_current_();
            ++row_;
            seek_();
            return *this;
        }
        operator bool() const { return value_ != nullptr; }
        const Scalar& value() const {
            validate_current_();
            return *value_;
        }
        Scalar& valueRef() {
            validate_current_();
            if (mutable_matrix_ == nullptr) {
                throw std::logic_error("SparseBlockMatrix const iterator has no mutable coefficient");
            }
            return mutable_matrix_->value_ref(row_, outer_);
        }
        Index row() const {
            validate_current_();
            return row_;
        }
        Index col() const {
            validate_current_();
            return outer_;
        }
        Index outer() const {
            validate_current_();
            return outer_;
        }
        StorageIndex index() const {
            validate_current_();
            return static_cast<StorageIndex>(row_);
        }
       private:
        void initialize_(Index outer) {
            matrix_->validate_block_shapes_();
            matrix_->validate_global_col_(outer);
            outer_ = outer;
            block_col_ = matrix_->outerBlockIndex(outer);
            local_col_ = outer - matrix_->col_offsets_[block_col_];
            seek_();
        }
        void seek_() {
            value_ = nullptr;
            while (row_ < matrix_->rows_) {
                const Index block_row = matrix_->innerBlockIndex(row_);
                const Index local_row = row_ - matrix_->row_offsets_[block_row];
                const auto& current = matrix_->blocks_[flat_index_(block_row, block_col_)];
                for (const auto entry : current.row(local_row)) {
                    if (entry.column() == local_col_) {
                        value_ = std::addressof(entry.value());
                        return;
                    }
                    if (entry.column() > local_col_) break;
                }
                ++row_;
            }
        }
        void validate_current_() const {
            if (value_ == nullptr) { throw std::logic_error("SparseBlockMatrix iterator is not dereferenceable"); }
        }

        const SparseBlockMatrix* matrix_ = nullptr;
        SparseBlockMatrix* mutable_matrix_ = nullptr;
        const Scalar* value_ = nullptr;
        Index outer_ = 0;
        Index row_ = 0;
        Index block_col_ = 0;
        Index local_col_ = 0;
    };

    SparseBlockMatrix() = default;

    template <typename RowExtents, typename ColExtents>
        requires(is_extent_container_<RowExtents> && is_extent_container_<ColExtents>)
    SparseBlockMatrix(const RowExtents& row_extents, const ColExtents& col_extents) {
        if (
          static_cast<std::size_t>(row_extents.size()) != static_cast<std::size_t>(BlockRows_) ||
          static_cast<std::size_t>(col_extents.size()) != static_cast<std::size_t>(BlockCols_)) {
            throw std::invalid_argument("SparseBlockMatrix extent counts must match the block grid");
        }
        std::array<Index, BlockRows_> rows {};
        std::array<Index, BlockCols_> cols {};
        for (Index i = 0; i < BlockRows_; ++i) rows[i] = checked_extent_(row_extents[i]);
        for (Index i = 0; i < BlockCols_; ++i) cols[i] = checked_extent_(col_extents[i]);
        configure_(rows, cols);
    }

    SparseBlockMatrix(Index block_rows, Index block_cols) {
        if (block_rows < 0 || block_cols < 0) {
            throw std::invalid_argument("SparseBlockMatrix block dimensions must be nonnegative");
        }
        std::array<Index, BlockRows_> rows {};
        std::array<Index, BlockCols_> cols {};
        rows.fill(block_rows);
        cols.fill(block_cols);
        configure_(rows, cols);
    }

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
                  row_dimensions[I] = checked_block_dimension_(block.rows());
                  col_dimensions[I] = checked_block_dimension_(block.cols());
              } else if (static_cast<Scalar>(block) != Scalar {}) {
                  throw std::invalid_argument("SparseBlockMatrix scalar placeholders must be zero");
              }
          },
          std::forward<Blocks>(blocks)...);

        std::array<Index, BlockRows_> rows {};
        std::array<Index, BlockCols_> cols {};
        for (Index block_row = 0; block_row < BlockRows_; ++block_row) {
            Index extent = -1;
            for (Index block_col = 0; block_col < BlockCols_; ++block_col) {
                const Index value = row_dimensions[flat_index_(block_row, block_col)];
                if (value < 0) continue;
                if (extent >= 0 && value != extent) {
                    throw std::invalid_argument("SparseBlockMatrix blocks in a block row must have equal rows");
                }
                extent = value;
            }
            rows[block_row] = extent < 0 ? 1 : extent;
        }
        for (Index block_col = 0; block_col < BlockCols_; ++block_col) {
            Index extent = -1;
            for (Index block_row = 0; block_row < BlockRows_; ++block_row) {
                const Index value = col_dimensions[flat_index_(block_row, block_col)];
                if (value < 0) continue;
                if (extent >= 0 && value != extent) {
                    throw std::invalid_argument("SparseBlockMatrix blocks in a block column must have equal columns");
                }
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

    SparseBlockMatrix(const SparseBlockMatrix&) = default;
    SparseBlockMatrix& operator=(const SparseBlockMatrix& other) {
        if (this == &other) return *this;
        SparseBlockMatrix replacement(other);
        swap(replacement);
        return *this;
    }
    SparseBlockMatrix(SparseBlockMatrix&& other) noexcept { swap(other); }
    SparseBlockMatrix& operator=(SparseBlockMatrix&& other) noexcept {
        if (this == &other) return *this;
        SparseBlockMatrix replacement(std::move(other));
        swap(replacement);
        return *this;
    }

    constexpr Index rows() const { return rows_; }
    constexpr Index cols() const { return cols_; }
    static constexpr Index block_rows() { return BlockRows_; }
    static constexpr Index block_cols() { return BlockCols_; }
    static constexpr Index blockRows() { return BlockRows_; }
    static constexpr Index blockCols() { return BlockCols_; }
    constexpr Index innerSize() const { return rows_; }
    constexpr Index outerSize() const { return cols_; }

    const block_type& block(Index row, Index col) const {
        validate_block_index_(row, col);
        return blocks_[flat_index_(row, col)];
    }
    block_type& block(Index row, Index col) {
        validate_block_index_(row, col);
        return blocks_[flat_index_(row, col)];
    }

    std::size_t non_zeros() const {
        validate_block_shapes_();
        std::size_t result = 0;
        for (const auto& current : blocks_) {
            const std::size_t increment = static_cast<std::size_t>(current.non_zeros());
            if (result > std::numeric_limits<std::size_t>::max() - increment) {
                throw std::length_error("SparseBlockMatrix nonzero count exceeds the supported range");
            }
            result += increment;
        }
        return result;
    }
    Index nonZerosEstimate() const {
        const std::size_t result = non_zeros();
        if (result > static_cast<std::size_t>(std::numeric_limits<Index>::max())) {
            throw std::length_error("SparseBlockMatrix nonzero estimate exceeds the supported int range");
        }
        return static_cast<Index>(result);
    }
    bool isCompressed() const {
        validate_block_shapes_();
        return true;
    }
    void makeCompressed() const { validate_block_shapes_(); }

    Index innerBlockIndex(Index row) const {
        validate_global_row_(row);
        return block_index_(row_offsets_, row);
    }
    Index outerBlockIndex(Index col) const {
        validate_global_col_(col);
        return block_index_(col_offsets_, col);
    }
    Index indexToBlockInner(Index row) const {
        const Index block_row = innerBlockIndex(row);
        return row - row_offsets_[block_row];
    }
    Index indexToBlockOuter(Index col) const {
        const Index block_col = outerBlockIndex(col);
        return col - col_offsets_[block_col];
    }

    Scalar coeff(Index row, Index col) const {
        validate_block_shapes_();
        const Index block_row = innerBlockIndex(row);
        const Index block_col = outerBlockIndex(col);
        return blocks_[flat_index_(block_row, block_col)].coeff(
          row - row_offsets_[block_row], col - col_offsets_[block_col]);
    }
    bool contains(Index row, Index col) const {
        validate_block_shapes_();
        const Index block_row = innerBlockIndex(row);
        const Index block_col = outerBlockIndex(col);
        return blocks_[flat_index_(block_row, block_col)].contains(
          row - row_offsets_[block_row], col - col_offsets_[block_col]);
    }
    Scalar& value_ref(Index row, Index col) {
        validate_block_shapes_();
        const Index block_row = innerBlockIndex(row);
        const Index block_col = outerBlockIndex(col);
        return blocks_[flat_index_(block_row, block_col)].value_ref(
          row - row_offsets_[block_row], col - col_offsets_[block_col]);
    }
    Scalar& coeffRef(Index row, Index col) {
        validate_block_shapes_();
        const Index block_row = innerBlockIndex(row);
        const Index block_col = outerBlockIndex(col);
        return blocks_[flat_index_(block_row, block_col)].coeffRef(
          row - row_offsets_[block_row], col - col_offsets_[block_col]);
    }

    SparseMatrix<Scalar> to_sparse() const {
        validate_block_shapes_();
        std::vector<triplet_type> triplets;
        std::vector<std::pair<Index, Index>> stored_zeros;
        triplets.reserve(non_zeros());
        for (Index block_row = 0; block_row < BlockRows_; ++block_row) {
            for (Index block_col = 0; block_col < BlockCols_; ++block_col) {
                const auto& current = blocks_[flat_index_(block_row, block_col)];
                for (Index row = 0; row < current.rows(); ++row) {
                    for (const auto entry : current.row(row)) {
                        const Index global_row = row_offsets_[block_row] + row;
                        const Index global_col = col_offsets_[block_col] + entry.column();
                        if (entry.value() == Scalar {}) {
                            stored_zeros.emplace_back(global_row, global_col);
                        } else {
                            triplets.emplace_back(global_row, global_col, entry.value());
                        }
                    }
                }
            }
        }
        SparseMatrix<Scalar> result(rows_, cols_, triplets);
        for (const auto [row, col] : stored_zeros) result.coeffRef(row, col);
        return result;
    }

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

    void rebuild_with_constraints(const std::vector<Index>& dofs) {
        if (dofs.empty()) return;
        if (rows_ != cols_) { throw std::invalid_argument("SparseBlockMatrix constraints require a square matrix"); }
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

    template <typename TripletList> void rebuild(const TripletList& triplets) {
        std::vector<std::vector<triplet_type>> block_triplets(block_count_);
        for (const auto& triplet : triplets) {
            const Index block_row = checked_triplet_block_row_(triplet.row());
            const Index block_col = checked_triplet_block_col_(triplet.col());
            block_triplets[flat_index_(block_row, block_col)].emplace_back(
              triplet.row() - row_offsets_[block_row], triplet.col() - col_offsets_[block_col],
              static_cast<Scalar>(triplet.value()));
        }

        SparseBlockMatrix replacement(row_extents_, col_extents_);
        for (std::size_t i = 0; i < block_count_; ++i) replacement.blocks_[i].rebuild(block_triplets[i]);
        swap(replacement);
    }
    void rebuild(std::initializer_list<triplet_type> triplets) {
        rebuild<std::initializer_list<triplet_type>>(triplets);
    }

    template <typename TripletList> void rebuild_block(Index row, Index col, const TripletList& triplets) {
        validate_block_index_(row, col);
        std::vector<triplet_type> local_triplets;
        for (const auto& triplet : triplets) {
            local_triplets.emplace_back(triplet.row(), triplet.col(), static_cast<Scalar>(triplet.value()));
        }
        block_type replacement(row_extents_[row], col_extents_[col], local_triplets);
        blocks_[flat_index_(row, col)].swap(replacement);
    }
    void rebuild_block(Index row, Index col, std::initializer_list<triplet_type> triplets) {
        rebuild_block<std::initializer_list<triplet_type>>(row, col, triplets);
    }

    template <typename TripletList> void setFromTriplets(const TripletList& triplets) { rebuild(triplets); }
    void setFromTriplets(std::initializer_list<triplet_type> triplets) { rebuild(triplets); }

    template <Index Row, Index Col, typename TripletList>
        requires(Row >= 0 && Row < BlockRows_ && Col >= 0 && Col < BlockCols_)
    void setBlockFromTriplets(const TripletList& triplets) {
        rebuild_block(Row, Col, triplets);
    }
    template <typename TripletList> void setBlockFromTriplets(Index row, Index col, const TripletList& triplets) {
        rebuild_block(row, col, triplets);
    }

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
    friend void swap(SparseBlockMatrix& lhs, SparseBlockMatrix& rhs) noexcept { lhs.swap(rhs); }
   private:
    template <std::integral Extent> static Index checked_extent_(Extent value) {
        if (!std::in_range<Index>(value)) {
            throw std::length_error("SparseBlockMatrix extent exceeds the supported int range");
        }
        const Index result = static_cast<Index>(value);
        if (result < 0) { throw std::invalid_argument("SparseBlockMatrix extents must be nonnegative"); }
        return result;
    }

    template <typename Dimension> static Index checked_block_dimension_(Dimension value) {
        if (!std::in_range<Index>(value)) {
            throw std::length_error("SparseBlockMatrix block dimension exceeds the supported int range");
        }
        const Index result = static_cast<Index>(value);
        if (result < 0) { throw std::invalid_argument("SparseBlockMatrix block dimensions must be nonnegative"); }
        return result;
    }

    static constexpr std::size_t flat_index_(Index row, Index col) {
        return static_cast<std::size_t>(row) * static_cast<std::size_t>(BlockCols_) + static_cast<std::size_t>(col);
    }

    template <std::size_t N>
    static Index fill_offsets_(const std::array<Index, N>& extents, std::array<Index, N + 1>& offsets) {
        Index total = 0;
        offsets[0] = 0;
        for (std::size_t i = 0; i < N; ++i) {
            if (extents[i] > std::numeric_limits<Index>::max() - total) {
                throw std::length_error("SparseBlockMatrix dimensions exceed the supported int range");
            }
            total += extents[i];
            offsets[i + 1] = total;
        }
        return total;
    }

    void
    configure_(const std::array<Index, BlockRows_>& row_extents, const std::array<Index, BlockCols_>& col_extents) {
        std::array<Index, BlockRows_ + 1> row_offsets {};
        std::array<Index, BlockCols_ + 1> col_offsets {};
        const Index rows = fill_offsets_(row_extents, row_offsets);
        const Index cols = fill_offsets_(col_extents, col_offsets);
        if (rows > 0) checked_storage_index_(rows - 1);

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

    template <typename SparseBlock> block_type materialize_sparse_(const SparseBlock& block) const {
        const Index rows = checked_block_dimension_(block.rows());
        const Index cols = checked_block_dimension_(block.cols());
        std::vector<triplet_type> triplets;
        std::vector<std::pair<Index, Index>> stored_zeros;
        triplets.reserve(static_cast<std::size_t>(block.non_zeros()));
        for (Index row = 0; row < rows; ++row) {
            for (const auto entry : block.row(row)) {
                const Scalar value = static_cast<Scalar>(entry.value());
                if (value == Scalar {}) {
                    stored_zeros.emplace_back(row, entry.column());
                } else {
                    triplets.emplace_back(row, entry.column(), value);
                }
            }
        }
        block_type result(rows, cols, triplets);
        for (const auto [row, col] : stored_zeros) result.coeffRef(row, col);
        return result;
    }

    template <typename Block> block_type materialize_(const Block& block) const {
        if constexpr (internals::is_native_sparse_matrix_v<Block>) {
            return materialize_sparse_(block);
        } else if constexpr (requires(const Block& value) {
                                 requires internals::is_native_sparse_matrix_v<decltype(value.to_sparse())>;
                             }) {
            const auto sparse = block.to_sparse();
            return materialize_sparse_(sparse);
        } else {
            const Index rows = checked_block_dimension_(block.rows());
            const Index cols = checked_block_dimension_(block.cols());
            std::vector<triplet_type> triplets;
            for (Index row = 0; row < rows; ++row) {
                for (Index col = 0; col < cols; ++col) {
                    const Scalar value = static_cast<Scalar>(block(row, col));
                    if (value != Scalar {}) triplets.emplace_back(row, col, value);
                }
            }
            return block_type(rows, cols, triplets);
        }
    }

    static StorageIndex checked_storage_index_(Index value) {
        if (!std::in_range<StorageIndex>(value)) {
            throw std::length_error("SparseBlockMatrix dimension exceeds the storage-index range");
        }
        return static_cast<StorageIndex>(value);
    }

    void validate_block_index_(Index row, Index col) const {
        if (row < 0 || row >= BlockRows_ || col < 0 || col >= BlockCols_) {
            throw std::out_of_range("SparseBlockMatrix block index is out of range");
        }
    }
    void validate_global_row_(Index row) const {
        if (row < 0 || row >= rows_) { throw std::out_of_range("SparseBlockMatrix row index is out of range"); }
    }
    void validate_global_col_(Index col) const {
        if (col < 0 || col >= cols_) { throw std::out_of_range("SparseBlockMatrix column index is out of range"); }
    }
    void validate_block_shapes_() const {
        for (Index row = 0; row < BlockRows_; ++row) {
            for (Index col = 0; col < BlockCols_; ++col) {
                const auto& current = blocks_[flat_index_(row, col)];
                if (current.rows() != row_extents_[row] || current.cols() != col_extents_[col]) {
                    throw std::invalid_argument("SparseBlockMatrix mutable block no longer matches its partition");
                }
            }
        }
    }

    template <std::size_t N> static Index block_index_(const std::array<Index, N>& offsets, Index index) {
        return static_cast<Index>(std::upper_bound(offsets.begin(), offsets.end(), index) - offsets.begin() - 1);
    }
    Index checked_triplet_block_row_(Index row) const {
        validate_global_row_(row);
        return block_index_(row_offsets_, row);
    }
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
