// This file is part of fdaPDE, a C++ library for physics-informed
// spatial and functional data analysis.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

#include <fdaPDE/linear_algebra.h>
#include <gtest/gtest.h>

#include <array>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

namespace native = fdapde::linalg;

template <typename T>
concept rvalue_blockable = requires(T value) { std::move(value).block(native::full_extent, native::full_extent); };

template <typename T>
concept rvalue_sliceable = requires(T value) { std::move(value).template slice<0>(0); };

template <typename T>
concept rvalue_iterable = requires(T value) { std::move(value).begin(); };

template <typename T>
concept writable_matrix_coefficient = requires(T& value) { value(0, 0) = 1; };

using fixed_array = native::MdArray<int, native::MdExtents<2, 3>>;
using fixed_map = native::MdMap<int, native::MdExtents<2, 3>>;
using dynamic_map = native::MdMap<int, native::MdExtents<fdapde::Dynamic, 3>>;
using const_map = native::MdMap<const int, native::MdExtents<2, 3>>;
using fixed_bool_array = native::MdArray<bool, native::MdExtents<2, 2>>;

static_assert(!rvalue_blockable<fixed_array>);
static_assert(!rvalue_sliceable<fixed_array>);
static_assert(!std::is_default_constructible_v<fixed_map>);
static_assert(std::is_default_constructible_v<dynamic_map>);
static_assert(!writable_matrix_coefficient<const_map>);
static_assert(!writable_matrix_coefficient<const fixed_bool_array>);

TEST(NativeMdArray, ShapesLayoutsIndexingAndIteration) {
    native::MdArray<int, native::MdExtents<2, 3, 4>> fixed;
    int value = 0;
    for (auto& coefficient : fixed) coefficient = ++value;
    EXPECT_EQ(fixed.size(), 24);
    EXPECT_EQ(fixed.extent(0), 2);
    EXPECT_EQ(fixed.extent(1), 3);
    EXPECT_EQ(fixed.extent(2), 4);
    EXPECT_EQ(fixed(1, 2, 3), 24);
    const std::array index {1, 1, 2};
    EXPECT_EQ(fixed(index), fixed(1, 1, 2));
    const std::vector<int> vector_index {1, 1, 2};
    EXPECT_EQ(fixed(vector_index), fixed(1, 1, 2));

    native::MdArray<int, native::MdExtents<2, 3>, native::ColMajor> column_major;
    for (int i = 0; i < column_major.size(); ++i) column_major[i] = i + 1;
    EXPECT_EQ(column_major.mapping().stride(0), 1);
    EXPECT_EQ(column_major.mapping().stride(1), 2);
    EXPECT_EQ(column_major(1, 0), 2);
    EXPECT_EQ(column_major(0, 1), 3);
    std::vector<int> logical_order;
    for (int coefficient : column_major) logical_order.push_back(coefficient);
    EXPECT_EQ(logical_order, (std::vector<int> {1, 3, 5, 2, 4, 6}));

    native::MdArray<int, native::MdExtents<fdapde::Dynamic, 2, fdapde::Dynamic>> mixed(3, 4);
    EXPECT_EQ(mixed.size(), 24);
    EXPECT_TRUE(mixed.resize(2, 2, 6));
    EXPECT_EQ(mixed.extent(0), 2);
    EXPECT_EQ(mixed.extent(1), 2);
    EXPECT_EQ(mixed.extent(2), 6);
}

TEST(NativeMdArray, EmptyAxesHaveEmptyRanges) {
    for (const auto& extents : {
           std::array {0, 2, 3},
            std::array {2, 0, 3},
            std::array {2, 3, 0}
    }) {
        native::MdArray<int, native::full_dynamic_extent_t<3>> array(extents[0], extents[1], extents[2]);
        EXPECT_EQ(array.size(), 0);
        EXPECT_EQ(array.begin(), array.end());
    }
}

TEST(NativeMdArray, BlocksSlicesAndOwningCopiesUseLogicalIndices) {
    fixed_array array;
    for (int i = 0; i < array.size(); ++i) array[i] = i + 1;

    auto block = array.block(std::pair {0, 1}, std::pair {1, 2});
    EXPECT_EQ(block.extent(0), 2);
    EXPECT_EQ(block.extent(1), 2);
    EXPECT_EQ(block(0, 0), 2);
    EXPECT_EQ(block(1, 1), 6);
    block(0, 0) = 20;
    EXPECT_EQ(array(0, 1), 20);

    native::MdArray<int, native::full_dynamic_extent_t<2>> block_copy(block);
    EXPECT_EQ(block_copy.extent(0), 2);
    EXPECT_EQ(block_copy.extent(1), 2);
    EXPECT_EQ(block_copy(0, 0), 20);
    EXPECT_EQ(block_copy(1, 1), 6);
    block(0, 0) = 2;
    EXPECT_EQ(block_copy(0, 0), 20);

    auto row = array.row(1);
    EXPECT_EQ(row.extent(0), 1);
    EXPECT_EQ(row.extent(1), 3);
    EXPECT_EQ(row(0, 2), 6);
    auto column_slice = array.template slice<1>(1);
    EXPECT_EQ(column_slice.extent(0), 2);
    EXPECT_EQ(column_slice(0), 2);
    EXPECT_EQ(column_slice(1), 5);
    static_assert(!rvalue_iterable<decltype(column_slice)>);

    native::MdArray<int, native::MdExtents<fdapde::Dynamic>> slice_copy(column_slice);
    EXPECT_EQ(slice_copy.size(), 2);
    EXPECT_EQ(slice_copy(0), 2);
    EXPECT_EQ(slice_copy(1), 5);

    native::MdArray<int, native::MdExtents<2, 3>, native::ColMajor> column_major;
    int coefficient = 0;
    for (auto& entry : column_major) entry = ++coefficient;
    auto strided = column_major.template slice<1>(1);
    native::MdArray<int, native::MdExtents<2>> strided_copy(strided);
    EXPECT_EQ(strided_copy(0), 2);
    EXPECT_EQ(strided_copy(1), 5);
}

TEST(NativeMdArray, MapsPropagateMutationAndConstness) {
    std::array<double, 6> storage {};
    native::MdMap<double, native::MdExtents<2, 3>> map(storage.data());
    map(1, 2) = 4.0;
    EXPECT_DOUBLE_EQ(storage[5], 4.0);
    EXPECT_TRUE(map.valid());

    const auto& const_object = map;
    static_assert(std::is_same_v<decltype(const_object(0, 0)), const double&>);
    native::MdMap<const double, native::MdExtents<2, 3>> read_only(storage.data());
    static_assert(std::is_same_v<decltype(read_only(0, 0)), const double&>);
    EXPECT_DOUBLE_EQ(read_only(1, 2), 4.0);

    native::MdMap<double, native::MdExtents<fdapde::Dynamic, 3>> empty;
    EXPECT_TRUE(empty.valid());
    EXPECT_EQ(empty.size(), 0);
    EXPECT_EQ(empty.begin(), empty.end());
}

TEST(NativeMdArray, AssignmentClosesAliasingAndUpdatesShape) {
    native::MdArray<int, native::MdExtents<2, 4>> array;
    for (int i = 0; i < array.size(); ++i) array[i] = i + 1;
    auto destination = array.block(native::full_extent, std::pair {1, 3});
    auto source = array.block(native::full_extent, std::pair {0, 2});
    destination = source;
    EXPECT_EQ(array(0, 1), 1);
    EXPECT_EQ(array(0, 2), 2);
    EXPECT_EQ(array(0, 3), 3);
    EXPECT_EQ(array(1, 1), 5);
    EXPECT_EQ(array(1, 2), 6);
    EXPECT_EQ(array(1, 3), 7);

    native::MdArray<int, native::full_dynamic_extent_t<2>> reshaped(2, 3);
    native::MdArray<int, native::full_dynamic_extent_t<2>> other_shape(3, 2);
    for (int i = 0; i < other_shape.size(); ++i) other_shape[i] = 10 + i;
    reshaped = other_shape.block(native::full_extent, native::full_extent);
    EXPECT_EQ(reshaped.extent(0), 3);
    EXPECT_EQ(reshaped.extent(1), 2);
    EXPECT_EQ(reshaped(2, 1), 15);

    native::MdArray<int, native::MdExtents<2, 3>, native::ColMajor> column_major;
    int coefficient = 0;
    for (auto& entry : column_major) entry = ++coefficient;
    fixed_array row_major;
    row_major = column_major.block(native::full_extent, native::full_extent);
    EXPECT_EQ(row_major(0, 0), 1);
    EXPECT_EQ(row_major(0, 2), 3);
    EXPECT_EQ(row_major(1, 0), 4);
    EXPECT_EQ(row_major(1, 2), 6);
}

TEST(NativeMdArray, BooleanStorageUsesNativeBitPacking) {
    native::MdArray<bool, native::MdExtents<2, 2>> small;
    small(0, 0) = true;
    small(0, 1) = small(0, 0);
    EXPECT_TRUE(small(0, 1));
    EXPECT_EQ(small.bitpacks(), 1);

    native::MdArray<bool, native::MdExtents<1, 64>> exact;
    exact.set();
    EXPECT_EQ(exact.bitpacks(), 1);
    EXPECT_TRUE(exact(0, 63));

    native::MdArray<bool, native::MdExtents<1, 65>> cross_pack;
    cross_pack(0, 64) = true;
    EXPECT_EQ(cross_pack.bitpacks(), 2);
    auto tail = cross_pack.block(native::full_extent, std::pair {63, 64});
    native::MdArray<bool, native::MdExtents<1, 2>> copy(tail);
    EXPECT_FALSE(copy(0, 0));
    EXPECT_TRUE(copy(0, 1));
    copy.clear();
    EXPECT_FALSE(copy(0, 1));

    fixed_bool_array destination;
    destination.set();
    fixed_bool_array source;
    source(1, 1) = true;
    auto destination_view = destination.block(native::full_extent, native::full_extent);
    destination_view = source.block(native::full_extent, native::full_extent);
    EXPECT_FALSE(destination(0, 0));
    EXPECT_FALSE(destination(0, 1));
    EXPECT_FALSE(destination(1, 0));
    EXPECT_TRUE(destination(1, 1));
}

}   // namespace
