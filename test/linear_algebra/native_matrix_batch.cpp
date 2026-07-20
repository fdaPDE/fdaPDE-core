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

#include <fdaPDE/linear_algebra.h>
#include <gtest/gtest.h>

#include <array>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

namespace native = fdapde::linalg;

using MutableBatch = native::MatrixBatchView<double, 2, 3>;
using ConstBatch = native::MatrixBatchView<const double, 2, 3>;
using MutableRow = native::MatrixView<double, 2, 3, native::RowMajor>;
using ConstRow = native::MatrixView<const double, 2, 3, native::RowMajor>;

template <typename Batch>
concept permits_mutable_row_write = requires(Batch& batch) { batch[0](0, 0) = 1.0; };

template <typename Batch>
concept permits_const_batch_row_write = requires(const Batch& batch) { batch[0](0, 0) = 1.0; };

template <typename Batch>
concept permits_rvalue_indexing = requires(Batch& batch) { std::move(batch)[0]; };

static_assert(!std::is_default_constructible_v<MutableBatch>);
static_assert(std::is_constructible_v<MutableBatch, std::span<double>>);
static_assert(!std::is_constructible_v<MutableBatch, std::span<const double>>);
static_assert(std::is_constructible_v<ConstBatch, std::span<double>>);
static_assert(std::is_constructible_v<ConstBatch, std::span<const double>>);
static_assert(!std::is_constructible_v<MutableBatch, std::vector<double>&>);
static_assert(!std::is_constructible_v<ConstBatch, std::vector<double>&&>);
static_assert(!std::is_constructible_v<ConstBatch, std::array<double, 6>&&>);
static_assert(!std::is_constructible_v<MutableBatch, double*, std::size_t>);
static_assert(std::is_same_v<decltype(std::declval<MutableBatch&>()[0]), MutableRow>);
static_assert(std::is_same_v<decltype(std::declval<const MutableBatch&>()[0]), ConstRow>);
static_assert(std::is_same_v<decltype(std::declval<ConstBatch&>()[0]), ConstRow>);
static_assert(!std::is_reference_v<decltype(std::declval<MutableBatch&>()[0])>);
static_assert(permits_mutable_row_write<MutableBatch>);
static_assert(!permits_mutable_row_write<ConstBatch>);
static_assert(!permits_const_batch_row_write<MutableBatch>);
static_assert(!permits_rvalue_indexing<MutableBatch>);

TEST(NativeMatrixBatchView, MapsContiguousRowsAndPropagatesMutation) {
    std::array<double, 12> storage {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0};
    MutableBatch batch {std::span<double>(storage)};

    ASSERT_EQ(batch.size(), 2);
    EXPECT_FALSE(batch.empty());

    auto first = batch[0];
    auto second = batch[1];
    EXPECT_EQ(first.rows(), 2);
    EXPECT_EQ(first.cols(), 3);
    EXPECT_EQ(first.data(), storage.data());
    EXPECT_EQ(second.data(), storage.data() + MutableBatch::MatrixSize);
    EXPECT_DOUBLE_EQ(first(1, 2), 6.0);
    EXPECT_DOUBLE_EQ(second(0, 1), 8.0);

    second(1, 2) = 42.0;
    EXPECT_DOUBLE_EQ(storage[11], 42.0);
    storage[7] = -8.0;
    EXPECT_DOUBLE_EQ(second(0, 1), -8.0);

    const native::Matrix<double, 2, 3> replacement({-1.0, -2.0, -3.0, -4.0, -5.0, -6.0});
    batch[0] = replacement;
    EXPECT_EQ((native::Matrix<double, 2, 3>(first)), replacement);
    EXPECT_DOUBLE_EQ(storage[6], 7.0);
}

TEST(NativeMatrixBatchView, ConstStorageAndConstHandlesExposeReadOnlyRows) {
    std::array<double, 6> mutable_storage {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    const MutableBatch mutable_batch {std::span<double>(mutable_storage)};
    const auto row_from_const_handle = mutable_batch[0];
    EXPECT_EQ(row_from_const_handle.data(), mutable_storage.data());
    EXPECT_DOUBLE_EQ(row_from_const_handle(1, 2), 6.0);

    const std::array<double, 6> const_storage {6.0, 5.0, 4.0, 3.0, 2.0, 1.0};
    ConstBatch const_batch {std::span<const double>(const_storage)};
    const auto row_from_const_storage = const_batch[0];
    EXPECT_EQ(row_from_const_storage.data(), const_storage.data());
    EXPECT_DOUBLE_EQ(row_from_const_storage(1, 2), 1.0);
}

TEST(NativeMatrixBatchView, ValidatesBufferShapeAndEverySiteIndex) {
    // The historical MatrixMap accepted this by checking the truncated row count (6) instead of 37 % 6.
    std::array<double, 37> malformed {};
    EXPECT_THROW((MutableBatch {std::span<double>(malformed)}), std::invalid_argument);

    std::array<double, 12> storage {};
    MutableBatch batch {std::span<double>(storage)};
    EXPECT_NO_THROW(static_cast<void>(batch[0]));
    EXPECT_NO_THROW(static_cast<void>(batch[1]));
    EXPECT_THROW(static_cast<void>(batch[-1]), std::out_of_range);
    EXPECT_THROW(static_cast<void>(batch[2]), std::out_of_range);

    std::span<double> empty_storage;
    MutableBatch empty {empty_storage};
    EXPECT_TRUE(empty.empty());
    EXPECT_EQ(empty.size(), 0);
    EXPECT_THROW(static_cast<void>(empty[0]), std::out_of_range);
}

TEST(NativeMatrixBatchView, RowProxyDependsOnTheBufferNotTheBatchHandle) {
    std::array<double, 12> storage {};
    auto row = [&storage]() {
        MutableBatch batch {std::span<double>(storage)};
        return batch[1];
    }();

    row(1, 2) = 9.0;
    EXPECT_EQ(row.data(), storage.data() + MutableBatch::MatrixSize);
    EXPECT_DOUBLE_EQ(storage[11], 9.0);
}

}   // namespace
