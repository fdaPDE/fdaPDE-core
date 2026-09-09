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
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

template <typename T>
concept rvalue_md_blockable = requires(T value) { std::move(value).block(fdapde::full_extent, fdapde::full_extent); };

template <typename T>
concept rvalue_md_sliceable = requires(T value) { std::move(value).template slice<0>(0); };

template <typename T>
concept rvalue_md_iterable = requires(T value) { std::move(value).begin(); };

template <typename T>
concept writable_md_coefficient = requires(T& value) { value(0, 0) = 1; };

template <typename Destination, typename Source>
concept md_assignable_from = requires(Destination& destination, const Source& source) { destination = source; };

using fixed_md_array = fdapde::MdArray<int, fdapde::MdExtents<2, 3>>;
using fixed_md_map = fdapde::MdMap<int, fdapde::MdExtents<2, 3>>;
using dynamic_md_map = fdapde::MdMap<int, fdapde::MdExtents<fdapde::Dynamic, 3>>;
using const_md_map = fdapde::MdMap<const int, fdapde::MdExtents<2, 3>>;
using fixed_bool_md_array = fdapde::MdArray<bool, fdapde::MdExtents<2, 2>>;
using fixed_md_view = decltype(std::declval<fixed_md_array&>().block(fdapde::full_extent, fdapde::full_extent));
using const_owner_md_view =
  decltype(std::declval<const fixed_md_array&>().block(fdapde::full_extent, fdapde::full_extent));
using const_map_md_view = decltype(std::declval<const_md_map&>().block(fdapde::full_extent, fdapde::full_extent));

// checks at compile time: !rvalue_md_blockable<fixed_md_array>
static_assert(!rvalue_md_blockable<fixed_md_array>);
// checks at compile time: !rvalue_md_sliceable<fixed_md_array>
static_assert(!rvalue_md_sliceable<fixed_md_array>);
// checks at compile time: !rvalue_md_iterable<fixed_md_array>
static_assert(!rvalue_md_iterable<fixed_md_array>);
// checks at compile time: !rvalue_md_blockable<fixed_md_view>
static_assert(!rvalue_md_blockable<fixed_md_view>);
// checks at compile time: !rvalue_md_sliceable<fixed_md_view>
static_assert(!rvalue_md_sliceable<fixed_md_view>);
// checks at compile time: !rvalue_md_iterable<fixed_md_view>
static_assert(!rvalue_md_iterable<fixed_md_view>);
// checks at compile time: !std::is_default_constructible_v<fixed_md_map>
static_assert(!std::is_default_constructible_v<fixed_md_map>);
// checks at compile time: std::is_default_constructible_v<dynamic_md_map>
static_assert(std::is_default_constructible_v<dynamic_md_map>);
// checks at compile time: !writable_md_coefficient<const_md_map>
static_assert(!writable_md_coefficient<const_md_map>);
// checks at compile time: const_owner_md_view::ReadOnly == 1
static_assert(const_owner_md_view::ReadOnly == 1);
// checks at compile time: const_map_md_view::ReadOnly == 1
static_assert(const_map_md_view::ReadOnly == 1);
// checks at compile time: !writable_md_coefficient<const_owner_md_view>
static_assert(!writable_md_coefficient<const_owner_md_view>);
// checks at compile time: !writable_md_coefficient<const_map_md_view>
static_assert(!writable_md_coefficient<const_map_md_view>);
// checks at compile time: !md_assignable_from<const_owner_md_view, fixed_md_array>
static_assert(!md_assignable_from<const_owner_md_view, fixed_md_array>);
// checks at compile time: !md_assignable_from<const_map_md_view, fixed_md_array>
static_assert(!md_assignable_from<const_map_md_view, fixed_md_array>);
// checks at compile time: !writable_md_coefficient<const fixed_bool_md_array>
static_assert(!writable_md_coefficient<const fixed_bool_md_array>);

template <int StorageOrder> void check_mdarray_shapes_layouts_and_iteration() {
    fdapde::MdArray<int, fdapde::MdExtents<2, 3, 4>, StorageOrder> fixed;
    int value = 0;
    for (auto& coefficient : fixed) coefficient = ++value;
    // checks fixed.valid()
    EXPECT_TRUE(fixed.valid());
    // compares fixed.size(), 24 using eq semantics
    EXPECT_EQ(fixed.size(), 24);
    // compares fixed.extent(0), 2 using eq semantics
    EXPECT_EQ(fixed.extent(0), 2);
    // compares fixed.extent(1), 3 using eq semantics
    EXPECT_EQ(fixed.extent(1), 3);
    // compares fixed.extent(2), 4 using eq semantics
    EXPECT_EQ(fixed.extent(2), 4);
    // compares fixed(1, 2, 3), 24 using eq semantics
    EXPECT_EQ(fixed(1, 2, 3), 24);
    // compares fixed(std::array {1, 1, 2}), fixed(1, 1, 2) using eq semantics
    EXPECT_EQ(fixed(std::array {1, 1, 2}), fixed(1, 1, 2));
    // compares fixed(std::vector<int> {1, 1, 2}), fixed(1, 1, 2) using eq semantics
    EXPECT_EQ(fixed(std::vector<int> {1, 1, 2}), fixed(1, 1, 2));

    fdapde::MdArray<int, fdapde::MdExtents<2, 3>, StorageOrder> ordered;
    for (int i = 0; i < ordered.size(); ++i) ordered[i] = i + 1;
    if constexpr (StorageOrder == fdapde::RowMajor) {
        // compares ordered.mapping().stride(0), 3 using eq semantics
        EXPECT_EQ(ordered.mapping().stride(0), 3);
        // compares ordered.mapping().stride(1), 1 using eq semantics
        EXPECT_EQ(ordered.mapping().stride(1), 1);
        // compares std::vector<int>(ordered.begin(), ordered.end()), (std::vector<int> {1, 2, 3, 4, 5, 6})
        // using eq semantics
        EXPECT_EQ(std::vector<int>(ordered.begin(), ordered.end()), (std::vector<int> {1, 2, 3, 4, 5, 6}));
    } else {
        // compares ordered.mapping().stride(0), 1 using eq semantics
        EXPECT_EQ(ordered.mapping().stride(0), 1);
        // compares ordered.mapping().stride(1), 2 using eq semantics
        EXPECT_EQ(ordered.mapping().stride(1), 2);
        // compares std::vector<int>(ordered.begin(), ordered.end()), (std::vector<int> {1, 3, 5, 2, 4, 6})
        // using eq semantics
        EXPECT_EQ(std::vector<int>(ordered.begin(), ordered.end()), (std::vector<int> {1, 3, 5, 2, 4, 6}));
    }

    fdapde::MdArray<int, fdapde::MdExtents<fdapde::Dynamic, 2, fdapde::Dynamic>, StorageOrder> mixed(3, 4);
    // compares mixed.size(), 24 using eq semantics
    EXPECT_EQ(mixed.size(), 24);
    // checks mixed.resize(2, 6)
    EXPECT_TRUE(mixed.resize(2, 6));
    // compares mixed.extent(0), 2 using eq semantics
    EXPECT_EQ(mixed.extent(0), 2);
    // compares mixed.extent(1), 2 using eq semantics
    EXPECT_EQ(mixed.extent(1), 2);
    // compares mixed.extent(2), 6 using eq semantics
    EXPECT_EQ(mixed.extent(2), 6);

    auto copy = fixed;
    copy(0, 0, 0) = -1;
    // compares fixed(0, 0, 0), 1 using eq semantics
    EXPECT_EQ(fixed(0, 0, 0), 1);

    using dynamic_array = fdapde::MdArray<int, fdapde::full_dynamic_extent_t<2>, StorageOrder>;
    dynamic_array copy_source(2, 3);
    for (int i = 0; i < copy_source.extent(0); ++i) {
        for (int j = 0; j < copy_source.extent(1); ++j) copy_source(i, j) = i * 3 + j + 1;
    }
    dynamic_array copy_destination(1, 2);
    copy_destination = copy_source;
    // compares copy_destination.extent(0), 2 using eq semantics
    EXPECT_EQ(copy_destination.extent(0), 2);
    // compares copy_destination.extent(1), 3 using eq semantics
    EXPECT_EQ(copy_destination.extent(1), 3);
    for (int i = 0; i < copy_destination.extent(0); ++i) {
        // compares copy_destination(i, j), copy_source(i, j) using eq semantics
        for (int j = 0; j < copy_destination.extent(1); ++j) EXPECT_EQ(copy_destination(i, j), copy_source(i, j));
    }
    copy_source(0, 0) = -1;
    // compares copy_destination(0, 0), 1 using eq semantics
    EXPECT_EQ(copy_destination(0, 0), 1);
}

template <int StorageOrder> void check_mdarray_empty_and_huge_empty_shapes() {
    for (const auto& extents : {
           std::array {0, 2, 3},
            std::array {2, 0, 3},
            std::array {2, 3, 0}
    }) {
        fdapde::MdArray<int, fdapde::full_dynamic_extent_t<3>, StorageOrder> array(extents[0], extents[1], extents[2]);
        // checks array.valid()
        EXPECT_TRUE(array.valid());
        // compares array.size(), 0 using eq semantics
        EXPECT_EQ(array.size(), 0);
        // compares array.begin(), array.end() using eq semantics
        EXPECT_EQ(array.begin(), array.end());
    }

    constexpr int Max = std::numeric_limits<int>::max();
    if constexpr (StorageOrder == fdapde::RowMajor) {
        fdapde::MdArray<int, fdapde::full_dynamic_extent_t<3>, StorageOrder> huge_empty(0, Max, Max);
        // checks huge_empty.valid()
        EXPECT_TRUE(huge_empty.valid());
        // compares huge_empty.size(), 0 using eq semantics
        EXPECT_EQ(huge_empty.size(), 0);
        auto block = huge_empty.block(fdapde::full_extent, Max - 1, fdapde::full_extent);
        // checks block.valid()
        EXPECT_TRUE(block.valid());
        // compares block.size(), 0 using eq semantics
        EXPECT_EQ(block.size(), 0);
        auto slice = huge_empty.template slice<1>(Max - 1);
        // checks slice.valid()
        EXPECT_TRUE(slice.valid());
        // compares slice.size(), 0 using eq semantics
        EXPECT_EQ(slice.size(), 0);
    } else {
        fdapde::MdArray<int, fdapde::full_dynamic_extent_t<3>, StorageOrder> huge_empty(Max, Max, 0);
        // checks huge_empty.valid()
        EXPECT_TRUE(huge_empty.valid());
        // compares huge_empty.size(), 0 using eq semantics
        EXPECT_EQ(huge_empty.size(), 0);
        auto block = huge_empty.block(fdapde::full_extent, Max - 1, fdapde::full_extent);
        // checks block.valid()
        EXPECT_TRUE(block.valid());
        // compares block.size(), 0 using eq semantics
        EXPECT_EQ(block.size(), 0);
        auto slice = huge_empty.template slice<1>(Max - 1);
        // checks slice.valid()
        EXPECT_TRUE(slice.valid());
        // compares slice.size(), 0 using eq semantics
        EXPECT_EQ(slice.size(), 0);
    }
}

template <int StorageOrder> void check_mdarray_views_maps_and_aliasing() {
    constexpr int OtherStorageOrder = StorageOrder == fdapde::RowMajor ? fdapde::ColMajor : fdapde::RowMajor;
    fdapde::MdArray<int, fdapde::MdExtents<2, 4>, StorageOrder> array;
    int coefficient = 0;
    for (auto& entry : array) entry = ++coefficient;
    auto destination = array.block(fdapde::full_extent, std::pair {1, 3});
    auto source = array.block(fdapde::full_extent, std::pair {0, 2});
    destination = source;
    // compares array(0, 1), 1 using eq semantics
    EXPECT_EQ(array(0, 1), 1);
    // compares array(0, 2), 2 using eq semantics
    EXPECT_EQ(array(0, 2), 2);
    // compares array(0, 3), 3 using eq semantics
    EXPECT_EQ(array(0, 3), 3);
    // compares array(1, 1), 5 using eq semantics
    EXPECT_EQ(array(1, 1), 5);
    // compares array(1, 2), 6 using eq semantics
    EXPECT_EQ(array(1, 2), 6);
    // compares array(1, 3), 7 using eq semantics
    EXPECT_EQ(array(1, 3), 7);

    fdapde::MdArray<int, fdapde::full_dynamic_extent_t<2>, StorageOrder> independent(destination);
    destination(0, 0) = 20;
    // compares independent(0, 0), 1 using eq semantics
    EXPECT_EQ(independent(0, 0), 1);

    auto row = array.row(1);
    // compares row.extent(0), 1 using eq semantics
    EXPECT_EQ(row.extent(0), 1);
    // compares row.extent(1), 4 using eq semantics
    EXPECT_EQ(row.extent(1), 4);
    // compares row(0, 3), 7 using eq semantics
    EXPECT_EQ(row(0, 3), 7);
    fdapde::MdArray<int, fdapde::MdExtents<2, 4>, StorageOrder> slice_owner;
    coefficient = 0;
    for (auto& entry : slice_owner) entry = ++coefficient;
    auto strided = slice_owner.template slice<0>(1);
    fdapde::MdArray<int, fdapde::MdExtents<4>, StorageOrder> slice_copy(strided);
    // compares slice_copy(0), 5 using eq semantics
    EXPECT_EQ(slice_copy(0), 5);
    // compares slice_copy(3), 8 using eq semantics
    EXPECT_EQ(slice_copy(3), 8);
    auto column_slice = slice_owner.template slice<1>(2);
    // checks at compile time: !rvalue_md_iterable<decltype(column_slice)>
    static_assert(!rvalue_md_iterable<decltype(column_slice)>);
    // compares column_slice.extent(0), 2 using eq semantics
    EXPECT_EQ(column_slice.extent(0), 2);
    // compares column_slice(0), 3 using eq semantics
    EXPECT_EQ(column_slice(0), 3);
    // compares column_slice(1), 7 using eq semantics
    EXPECT_EQ(column_slice(1), 7);
    auto column = slice_owner.col(2);
    // compares column.extent(0), 2 using eq semantics
    EXPECT_EQ(column.extent(0), 2);
    // compares column.extent(1), 1 using eq semantics
    EXPECT_EQ(column.extent(1), 1);
    // compares column(0, 0), 3 using eq semantics
    EXPECT_EQ(column(0, 0), 3);
    // compares column(1, 0), 7 using eq semantics
    EXPECT_EQ(column(1, 0), 7);

    const auto& const_slice_owner = slice_owner;
    auto read_only_view = const_slice_owner.block(fdapde::full_extent, fdapde::full_extent);
    // checks at compile time: decltype(read_only_view)::ReadOnly == 1
    static_assert(decltype(read_only_view)::ReadOnly == 1);
    // compares read_only_view(1, 3), 8 using eq semantics
    EXPECT_EQ(read_only_view(1, 3), 8);

    fdapde::MdArray<int, fdapde::MdExtents<2, 3>, OtherStorageOrder> other_order;
    for (int i = 0; i < other_order.extent(0); ++i) {
        for (int j = 0; j < other_order.extent(1); ++j) other_order(i, j) = i * 3 + j + 1;
    }
    fdapde::MdArray<int, fdapde::MdExtents<2, 3>, StorageOrder> cross_order_owner;
    cross_order_owner = other_order;
    // compares cross_order_owner(0, 2), 3 using eq semantics
    EXPECT_EQ(cross_order_owner(0, 2), 3);
    // compares cross_order_owner(1, 0), 4 using eq semantics
    EXPECT_EQ(cross_order_owner(1, 0), 4);
    cross_order_owner(0, 0) = -1;
    auto cross_order_destination = cross_order_owner.block(fdapde::full_extent, fdapde::full_extent);
    auto cross_order_source = other_order.block(fdapde::full_extent, fdapde::full_extent);
    cross_order_destination = cross_order_source;
    // compares cross_order_owner(0, 0), 1 using eq semantics
    EXPECT_EQ(cross_order_owner(0, 0), 1);
    // compares cross_order_owner(1, 2), 6 using eq semantics
    EXPECT_EQ(cross_order_owner(1, 2), 6);

    fdapde::MdArray<int, fdapde::full_dynamic_extent_t<2>, StorageOrder> reshaped(2, 3);
    fdapde::MdArray<int, fdapde::full_dynamic_extent_t<2>, StorageOrder> other_shape(3, 2);
    coefficient = 9;
    for (auto& entry : other_shape) entry = ++coefficient;
    reshaped = other_shape.block(fdapde::full_extent, fdapde::full_extent);
    // compares reshaped.extent(0), 3 using eq semantics
    EXPECT_EQ(reshaped.extent(0), 3);
    // compares reshaped.extent(1), 2 using eq semantics
    EXPECT_EQ(reshaped.extent(1), 2);
    // compares reshaped(2, 1), 15 using eq semantics
    EXPECT_EQ(reshaped(2, 1), 15);

    std::array<double, 6> storage {};
    fdapde::MdMap<double, fdapde::MdExtents<2, 3>, StorageOrder> map(storage.data());
    map(1, 2) = 4.0;
    // compares storage[static_cast<std::size_t>(map.mapping()(1, 2))], 4.0 using double_eq semantics
    EXPECT_DOUBLE_EQ(storage[static_cast<std::size_t>(map.mapping()(1, 2))], 4.0);
    const auto& const_map_object = map;
    // checks at compile time: std::is_same_v<decltype(const_map_object(0, 0)), const double&>
    static_assert(std::is_same_v<decltype(const_map_object(0, 0)), const double&>);
    fdapde::MdMap<const double, fdapde::MdExtents<2, 3>, StorageOrder> read_only(storage.data());
    // checks at compile time: std::is_same_v<decltype(read_only(0, 0)), const double&>
    static_assert(std::is_same_v<decltype(read_only(0, 0)), const double&>);
    // compares read_only(1, 2), 4.0 using double_eq semantics
    EXPECT_DOUBLE_EQ(read_only(1, 2), 4.0);

    std::array<double, 6> dynamic_storage {};
    fdapde::MdMap<double, fdapde::full_dynamic_extent_t<2>, StorageOrder> dynamic_map(dynamic_storage.data(), 2, 3);
    // compares dynamic_map.size(), 6 using eq semantics
    EXPECT_EQ(dynamic_map.size(), 6);
    // compares dynamic_map.extent(0), 2 using eq semantics
    EXPECT_EQ(dynamic_map.extent(0), 2);
    // compares dynamic_map.extent(1), 3 using eq semantics
    EXPECT_EQ(dynamic_map.extent(1), 3);
    dynamic_map(0, 1) = 5.0;
    constexpr std::size_t DynamicOffset = StorageOrder == fdapde::RowMajor ? 1 : 2;
    // compares dynamic_storage[DynamicOffset], 5.0 using double_eq semantics
    EXPECT_DOUBLE_EQ(dynamic_storage[DynamicOffset], 5.0);

    std::array<double, 6> partial_storage {};
    fdapde::MdMap<double, fdapde::MdExtents<fdapde::Dynamic, 3>, StorageOrder> partial_map(partial_storage.data(), 2);
    // compares partial_map.size(), 6 using eq semantics
    EXPECT_EQ(partial_map.size(), 6);
    // compares partial_map.extent(0), 2 using eq semantics
    EXPECT_EQ(partial_map.extent(0), 2);
    // compares partial_map.extent(1), 3 using eq semantics
    EXPECT_EQ(partial_map.extent(1), 3);
    partial_map(1, 0) = 6.0;
    constexpr std::size_t PartialOffset = StorageOrder == fdapde::RowMajor ? 3 : 1;
    // compares partial_storage[PartialOffset], 6.0 using double_eq semantics
    EXPECT_DOUBLE_EQ(partial_storage[PartialOffset], 6.0);

    fdapde::MdMap<double, fdapde::MdExtents<fdapde::Dynamic, 3>, StorageOrder> empty_map;
    // checks empty_map.valid()
    EXPECT_TRUE(empty_map.valid());
    // compares empty_map.size(), 0 using eq semantics
    EXPECT_EQ(empty_map.size(), 0);
    // compares empty_map.begin(), empty_map.end() using eq semantics
    EXPECT_EQ(empty_map.begin(), empty_map.end());
}

template <int StorageOrder> void check_mdarray_boolean_storage() {
    constexpr int OtherStorageOrder = StorageOrder == fdapde::RowMajor ? fdapde::ColMajor : fdapde::RowMajor;
    fdapde::MdArray<bool, fdapde::MdExtents<2, 2>, StorageOrder> small;
    small(0, 0) = true;
    small(0, 1) = small(0, 0);
    // checks small(0, 1)
    EXPECT_TRUE(small(0, 1));
    // compares small.bitpacks(), 1 using eq semantics
    EXPECT_EQ(small.bitpacks(), 1);

    fdapde::MdArray<bool, fdapde::MdExtents<2, 32>, StorageOrder> exact;
    exact.set();
    // compares exact.bitpacks(), 1 using eq semantics
    EXPECT_EQ(exact.bitpacks(), 1);
    // checks exact(0, 31)
    EXPECT_TRUE(exact(0, 31));
    // checks exact(1, 0)
    EXPECT_TRUE(exact(1, 0));

    fdapde::MdArray<bool, fdapde::MdExtents<2, 33>, StorageOrder> cross_pack;
    cross_pack(0, 32) = true;
    cross_pack(1, 0) = true;
    cross_pack(1, 32) = true;
    // compares cross_pack.bitpacks(), 2 using eq semantics
    EXPECT_EQ(cross_pack.bitpacks(), 2);
    auto tail = cross_pack.block(fdapde::full_extent, std::pair {31, 32});
    fdapde::MdArray<bool, fdapde::MdExtents<2, 2>, StorageOrder> copy(tail);
    // checks copy(0, 0)
    EXPECT_FALSE(copy(0, 0));
    // checks copy(0, 1)
    EXPECT_TRUE(copy(0, 1));
    // checks copy(1, 0)
    EXPECT_FALSE(copy(1, 0));
    // checks copy(1, 1)
    EXPECT_TRUE(copy(1, 1));
    copy.clear();
    // checks copy(1, 1)
    EXPECT_FALSE(copy(1, 1));

    fdapde::MdArray<bool, fdapde::MdExtents<2, 33>, OtherStorageOrder> source;
    source(0, 0) = true;
    source(0, 32) = true;
    source(1, 1) = true;
    source(1, 31) = true;
    fdapde::MdArray<bool, fdapde::MdExtents<2, 33>, StorageOrder> converted(source);
    for (int i = 0; i < converted.extent(0); ++i) {
        // compares converted(i, j), source(i, j) using eq semantics
        for (int j = 0; j < converted.extent(1); ++j) EXPECT_EQ(converted(i, j), source(i, j));
    }

    fdapde::MdArray<bool, fdapde::MdExtents<2, 33>, StorageOrder> destination;
    destination.set();
    auto destination_view = destination.block(fdapde::full_extent, fdapde::full_extent);
    destination_view = source.block(fdapde::full_extent, fdapde::full_extent);
    for (int i = 0; i < destination.extent(0); ++i) {
        // compares destination(i, j), source(i, j) using eq semantics
        for (int j = 0; j < destination.extent(1); ++j) EXPECT_EQ(destination(i, j), source(i, j));
    }

    fdapde::MdArray<bool, fdapde::MdExtents<2, 34>, StorageOrder> aliasing;
    aliasing(0, 0) = true;
    aliasing(0, 33) = true;
    auto alias_destination = aliasing.block(fdapde::full_extent, std::pair {1, 33});
    auto alias_source = aliasing.block(fdapde::full_extent, std::pair {0, 32});
    alias_destination = alias_source;
    // checks aliasing(0, 0)
    EXPECT_TRUE(aliasing(0, 0));
    // checks aliasing(0, 1)
    EXPECT_TRUE(aliasing(0, 1));
    // checks aliasing(0, 2)
    EXPECT_FALSE(aliasing(0, 2));
    // checks aliasing(0, 33)
    EXPECT_FALSE(aliasing(0, 33));
    // checks aliasing(1, 1)
    EXPECT_FALSE(aliasing(1, 1));
}

template <int StorageOrder> void check_mdarray_runtime_failures() {
    using dynamic_array = fdapde::MdArray<int, fdapde::full_dynamic_extent_t<2>, StorageOrder>;
    constexpr int Max = std::numeric_limits<int>::max();

    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(dynamic_array(-1, 3)), std::invalid_argument);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(dynamic_array(Max, 2)), std::length_error);
    using partial_array = fdapde::MdArray<int, fdapde::MdExtents<fdapde::Dynamic, 2>, StorageOrder>;
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(partial_array(3, 3)), std::invalid_argument);

    dynamic_array preserved(2, 3);
    for (int i = 0; i < preserved.size(); ++i) preserved[i] = i + 1;
    const std::vector<int> preserved_values(preserved.begin(), preserved.end());
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(preserved.resize(-1, 3)), std::invalid_argument);
    // compares preserved.extent(0), 2 using eq semantics
    EXPECT_EQ(preserved.extent(0), 2);
    // compares preserved.extent(1), 3 using eq semantics
    EXPECT_EQ(preserved.extent(1), 3);
    // compares std::vector<int>(preserved.begin(), preserved.end()), preserved_values using eq semantics
    EXPECT_EQ(std::vector<int>(preserved.begin(), preserved.end()), preserved_values);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(preserved.resize(Max, 2)), std::length_error);
    // compares preserved.extent(0), 2 using eq semantics
    EXPECT_EQ(preserved.extent(0), 2);
    // compares preserved.extent(1), 3 using eq semantics
    EXPECT_EQ(preserved.extent(1), 3);
    // compares std::vector<int>(preserved.begin(), preserved.end()), preserved_values using eq semantics
    EXPECT_EQ(std::vector<int>(preserved.begin(), preserved.end()), preserved_values);

    fixed_md_array fixed;
    for (int i = 0; i < fixed.size(); ++i) fixed[i] = i + 1;
    const std::vector<int> fixed_values(fixed.begin(), fixed.end());
    dynamic_array wrong_shape(3, 2);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(fixed = wrong_shape, std::invalid_argument);
    // compares std::vector<int>(fixed.begin(), fixed.end()), fixed_values using eq semantics
    EXPECT_EQ(std::vector<int>(fixed.begin(), fixed.end()), fixed_values);

    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(fixed_md_map(nullptr)), std::invalid_argument);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(
      static_cast<void>(fdapde::MdMap<int, fdapde::full_dynamic_extent_t<2>, StorageOrder>(nullptr, 2, 3)),
      std::invalid_argument);

    auto block_source = dynamic_array(2, 3);
    for (int i = 0; i < block_source.size(); ++i) block_source[i] = i + 1;
    const std::vector<int> block_values(block_source.begin(), block_source.end());
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(block_source.block(std::pair {-1, 0}, fdapde::full_extent)), std::out_of_range);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(block_source.block(std::pair {0, 2}, fdapde::full_extent)), std::out_of_range);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(block_source.template slice<0>(-1)), std::out_of_range);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(block_source.template slice<1>(3)), std::out_of_range);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(block_source(-1, 0)), std::out_of_range);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(block_source(2, 0)), std::out_of_range);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(block_source(std::array {0})), std::invalid_argument);

    auto row = block_source.row(0);
    dynamic_array incompatible(3, 1);
    incompatible(0, 0) = 7;
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(row.assign_inplace_from(incompatible), std::invalid_argument);
    // compares std::vector<int>(block_source.begin(), block_source.end()), block_values using eq semantics
    EXPECT_EQ(std::vector<int>(block_source.begin(), block_source.end()), block_values);
}

// verifies mdarray through the public algebra API
TEST(linear_algebra, mdarray) {
    check_mdarray_shapes_layouts_and_iteration<fdapde::RowMajor>();
    check_mdarray_shapes_layouts_and_iteration<fdapde::ColMajor>();
    check_mdarray_empty_and_huge_empty_shapes<fdapde::RowMajor>();
    check_mdarray_empty_and_huge_empty_shapes<fdapde::ColMajor>();
    check_mdarray_views_maps_and_aliasing<fdapde::RowMajor>();
    check_mdarray_views_maps_and_aliasing<fdapde::ColMajor>();
    check_mdarray_boolean_storage<fdapde::RowMajor>();
    check_mdarray_boolean_storage<fdapde::ColMajor>();
    check_mdarray_runtime_failures<fdapde::RowMajor>();
    check_mdarray_runtime_failures<fdapde::ColMajor>();
}

}   // namespace
