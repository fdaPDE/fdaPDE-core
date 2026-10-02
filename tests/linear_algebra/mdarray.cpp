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

#include <fdaPDE/dense_linear_algebra.h>
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

// a temporary owner cannot lend a block that would outlive its storage
static_assert(!rvalue_md_blockable<fixed_md_array>);
// a temporary owner cannot lend a slice that would outlive its storage
static_assert(!rvalue_md_sliceable<fixed_md_array>);
// a temporary owner cannot expose iterators into destroyed storage
static_assert(!rvalue_md_iterable<fixed_md_array>);
// a temporary view cannot lend a block that would retain its expired mapping
static_assert(!rvalue_md_blockable<fixed_md_view>);
// a temporary view cannot lend a slice that would retain its expired mapping
static_assert(!rvalue_md_sliceable<fixed_md_view>);
// a temporary view cannot expose iterators tied to its expired mapping
static_assert(!rvalue_md_iterable<fixed_md_view>);
// a fixed nonempty map requires an explicit storage binding
static_assert(!std::is_default_constructible_v<fixed_md_map>);
// a dynamic map permits an initially empty binding
static_assert(std::is_default_constructible_v<dynamic_md_map>);
// a map of const scalars cannot write coefficients
static_assert(!writable_md_coefficient<const_md_map>);
// a view obtained from a const owner advertises read-only access
static_assert(const_owner_md_view::ReadOnly == 1);
// a view obtained from a const map advertises read-only access
static_assert(const_map_md_view::ReadOnly == 1);
// a const owner's view rejects coefficient writes
static_assert(!writable_md_coefficient<const_owner_md_view>);
// a const map's view rejects coefficient writes
static_assert(!writable_md_coefficient<const_map_md_view>);
// a const owner's view rejects assignment from an owner
static_assert(!md_assignable_from<const_owner_md_view, fixed_md_array>);
// a const map's view rejects assignment from an owner
static_assert(!md_assignable_from<const_map_md_view, fixed_md_array>);
// a const packed Boolean owner cannot expose a writable bit proxy
static_assert(!writable_md_coefficient<const fixed_bool_md_array>);

template <int StorageOrder> void check_mdarray_shapes_layouts_and_iteration() {
    fdapde::MdArray<int, fdapde::MdExtents<2, 3, 4>, StorageOrder> fixed;
    int value = 0;
    for (auto& coefficient : fixed) coefficient = ++value;
    // a fixed three-dimensional owner starts with a valid mapping
    EXPECT_TRUE(fixed.valid());
    // the 2-by-3-by-4 shape allocates 24 coefficients
    EXPECT_EQ(fixed.size(), 24);
    // the first fixed axis retains extent two
    EXPECT_EQ(fixed.extent(0), 2);
    // the second fixed axis retains extent three
    EXPECT_EQ(fixed.extent(1), 3);
    // the third fixed axis retains extent four
    EXPECT_EQ(fixed.extent(2), 4);
    // logical iteration reaches the final coordinate with value 24
    EXPECT_EQ(fixed(1, 2, 3), 24);
    // array indices address the same coefficient as separate indices
    EXPECT_EQ(fixed(std::array {1, 1, 2}), fixed(1, 1, 2));
    // vector indices address the same coefficient as separate indices
    EXPECT_EQ(fixed(std::vector<int> {1, 1, 2}), fixed(1, 1, 2));

    fdapde::MdArray<int, fdapde::MdExtents<2, 3>, StorageOrder> ordered;
    for (int i = 0; i < ordered.size(); ++i) ordered[i] = i + 1;
    if constexpr (StorageOrder == fdapde::RowMajor) {
        // a row-major 2-by-3 mapping advances three coefficients between rows
        EXPECT_EQ(ordered.mapping().stride(0), 3);
        // a row-major mapping stores adjacent columns contiguously
        EXPECT_EQ(ordered.mapping().stride(1), 1);
        // row-major iteration visits the physical sequence in logical row order
        EXPECT_EQ(std::vector<int>(ordered.begin(), ordered.end()), (std::vector<int> {1, 2, 3, 4, 5, 6}));
    } else {
        // a column-major mapping stores adjacent rows contiguously
        EXPECT_EQ(ordered.mapping().stride(0), 1);
        // a column-major 2-by-3 mapping advances two coefficients between columns
        EXPECT_EQ(ordered.mapping().stride(1), 2);
        // column-major iteration reorders physical storage into logical row order
        EXPECT_EQ(std::vector<int>(ordered.begin(), ordered.end()), (std::vector<int> {1, 3, 5, 2, 4, 6}));
    }

    fdapde::MdArray<int, fdapde::MdExtents<fdapde::Dynamic, 2, fdapde::Dynamic>, StorageOrder> mixed(3, 4);
    // mixed static and dynamic extents produce the expected 24 coefficients
    EXPECT_EQ(mixed.size(), 24);
    // resizing the dynamic axes reports a successful change
    EXPECT_TRUE(mixed.resize(2, 6));
    // the first dynamic axis takes the requested extent two
    EXPECT_EQ(mixed.extent(0), 2);
    // resizing leaves the fixed middle axis at extent two
    EXPECT_EQ(mixed.extent(1), 2);
    // the last dynamic axis takes the requested extent six
    EXPECT_EQ(mixed.extent(2), 6);

    auto copy = fixed;
    copy(0, 0, 0) = -1;
    // writing a copied owner leaves the source coefficient unchanged
    EXPECT_EQ(fixed(0, 0, 0), 1);

    using dynamic_array = fdapde::MdArray<int, fdapde::full_dynamic_extent_t<2>, StorageOrder>;
    dynamic_array copy_source(2, 3);
    for (int i = 0; i < copy_source.extent(0); ++i) {
        for (int j = 0; j < copy_source.extent(1); ++j) copy_source(i, j) = i * 3 + j + 1;
    }
    dynamic_array copy_destination(1, 2);
    copy_destination = copy_source;
    // copy assignment adopts the source's first dynamic extent
    EXPECT_EQ(copy_destination.extent(0), 2);
    // copy assignment adopts the source's second dynamic extent
    EXPECT_EQ(copy_destination.extent(1), 3);
    for (int i = 0; i < copy_destination.extent(0); ++i) {
        // copy assignment preserves every logical coefficient
        for (int j = 0; j < copy_destination.extent(1); ++j) EXPECT_EQ(copy_destination(i, j), copy_source(i, j));
    }
    copy_source(0, 0) = -1;
    // mutating the source after assignment leaves the destination independent
    EXPECT_EQ(copy_destination(0, 0), 1);
}

template <int StorageOrder> void check_mdarray_empty_and_huge_empty_shapes() {
    for (const auto& extents : {
           std::array {0, 2, 3},
            std::array {2, 0, 3},
            std::array {2, 3, 0}
    }) {
        fdapde::MdArray<int, fdapde::full_dynamic_extent_t<3>, StorageOrder> array(extents[0], extents[1], extents[2]);
        // a shape with an empty axis still has a valid mapping
        EXPECT_TRUE(array.valid());
        // an empty axis makes the total coefficient count zero
        EXPECT_EQ(array.size(), 0);
        // an empty owner exposes an empty iterator range
        EXPECT_EQ(array.begin(), array.end());
    }

    constexpr int Max = std::numeric_limits<int>::max();
    if constexpr (StorageOrder == fdapde::RowMajor) {
        fdapde::MdArray<int, fdapde::full_dynamic_extent_t<3>, StorageOrder> huge_empty(0, Max, Max);
        // a leading zero extent keeps huge row-major extents representable
        EXPECT_TRUE(huge_empty.valid());
        // the huge row-major shape has no coefficients despite its nonzero extents
        EXPECT_EQ(huge_empty.size(), 0);
        auto block = huge_empty.block(fdapde::full_extent, Max - 1, fdapde::full_extent);
        // a block of the huge empty row-major shape has a valid mapping
        EXPECT_TRUE(block.valid());
        // selecting a nonempty axis does not remove the block's empty axis
        EXPECT_EQ(block.size(), 0);
        auto slice = huge_empty.template slice<1>(Max - 1);
        // a slice of the huge empty row-major shape has a valid mapping
        EXPECT_TRUE(slice.valid());
        // removing the selected axis leaves the row-major slice empty
        EXPECT_EQ(slice.size(), 0);
    } else {
        fdapde::MdArray<int, fdapde::full_dynamic_extent_t<3>, StorageOrder> huge_empty(Max, Max, 0);
        // a trailing zero extent keeps huge column-major extents representable
        EXPECT_TRUE(huge_empty.valid());
        // the huge column-major shape has no coefficients despite its nonzero extents
        EXPECT_EQ(huge_empty.size(), 0);
        auto block = huge_empty.block(fdapde::full_extent, Max - 1, fdapde::full_extent);
        // a block of the huge empty column-major shape has a valid mapping
        EXPECT_TRUE(block.valid());
        // selecting a nonempty axis does not remove the block's empty axis
        EXPECT_EQ(block.size(), 0);
        auto slice = huge_empty.template slice<1>(Max - 1);
        // a slice of the huge empty column-major shape has a valid mapping
        EXPECT_TRUE(slice.valid());
        // removing the selected axis leaves the column-major slice empty
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
    // overlapping assignment copies the original first coefficient one column right
    EXPECT_EQ(array(0, 1), 1);
    // overlapping assignment reads the original second coefficient before it is overwritten
    EXPECT_EQ(array(0, 2), 2);
    // overlapping assignment preserves the original third coefficient in the shifted row
    EXPECT_EQ(array(0, 3), 3);
    // the overlap-safe copy also shifts the second row's first coefficient
    EXPECT_EQ(array(1, 1), 5);
    // the second row's middle destination receives the original value six
    EXPECT_EQ(array(1, 2), 6);
    // the second row's last destination receives the original value seven
    EXPECT_EQ(array(1, 3), 7);

    fdapde::MdArray<int, fdapde::full_dynamic_extent_t<2>, StorageOrder> independent(destination);
    destination(0, 0) = 20;
    // an owner constructed from a block keeps its copy after the block changes
    EXPECT_EQ(independent(0, 0), 1);

    auto row = array.row(1);
    // a row view retains a singleton row axis
    EXPECT_EQ(row.extent(0), 1);
    // a row view spans all four columns
    EXPECT_EQ(row.extent(1), 4);
    // the row view resolves its last coordinate in the original owner
    EXPECT_EQ(row(0, 3), 7);
    fdapde::MdArray<int, fdapde::MdExtents<2, 4>, StorageOrder> slice_owner;
    coefficient = 0;
    for (auto& entry : slice_owner) entry = ++coefficient;
    auto strided = slice_owner.template slice<0>(1);
    fdapde::MdArray<int, fdapde::MdExtents<4>, StorageOrder> slice_copy(strided);
    // copying a slice starts with the selected row's first coefficient
    EXPECT_EQ(slice_copy(0), 5);
    // copying a slice preserves the selected row's last coefficient
    EXPECT_EQ(slice_copy(3), 8);
    auto column_slice = slice_owner.template slice<1>(2);
    // a temporary slice cannot expose iterators tied to its mapping
    static_assert(!rvalue_md_iterable<decltype(column_slice)>);
    // slicing away the column axis leaves a one-dimensional extent of two
    EXPECT_EQ(column_slice.extent(0), 2);
    // the column slice starts at the selected column in the first row
    EXPECT_EQ(column_slice(0), 3);
    // the column slice advances to the same column in the second row
    EXPECT_EQ(column_slice(1), 7);
    auto column = slice_owner.col(2);
    // a column view retains the owner's two rows
    EXPECT_EQ(column.extent(0), 2);
    // a column view retains a singleton column axis
    EXPECT_EQ(column.extent(1), 1);
    // the two-dimensional column view reads the selected first-row coefficient
    EXPECT_EQ(column(0, 0), 3);
    // the two-dimensional column view reads the selected second-row coefficient
    EXPECT_EQ(column(1, 0), 7);

    const auto& const_slice_owner = slice_owner;
    auto read_only_view = const_slice_owner.block(fdapde::full_extent, fdapde::full_extent);
    // a block borrowed from a const owner is read-only
    static_assert(decltype(read_only_view)::ReadOnly == 1);
    // a read-only block still exposes the original final coefficient
    EXPECT_EQ(read_only_view(1, 3), 8);

    fdapde::MdArray<int, fdapde::MdExtents<2, 3>, OtherStorageOrder> other_order;
    for (int i = 0; i < other_order.extent(0); ++i) {
        for (int j = 0; j < other_order.extent(1); ++j) other_order(i, j) = i * 3 + j + 1;
    }
    fdapde::MdArray<int, fdapde::MdExtents<2, 3>, StorageOrder> cross_order_owner;
    cross_order_owner = other_order;
    // cross-order owner assignment preserves the last column of the first row
    EXPECT_EQ(cross_order_owner(0, 2), 3);
    // cross-order owner assignment preserves the first column of the second row
    EXPECT_EQ(cross_order_owner(1, 0), 4);
    cross_order_owner(0, 0) = -1;
    auto cross_order_destination = cross_order_owner.block(fdapde::full_extent, fdapde::full_extent);
    auto cross_order_source = other_order.block(fdapde::full_extent, fdapde::full_extent);
    cross_order_destination = cross_order_source;
    // cross-order block assignment overwrites the modified destination coefficient
    EXPECT_EQ(cross_order_owner(0, 0), 1);
    // cross-order block assignment preserves the final logical coefficient
    EXPECT_EQ(cross_order_owner(1, 2), 6);

    fdapde::MdArray<int, fdapde::full_dynamic_extent_t<2>, StorageOrder> reshaped(2, 3);
    fdapde::MdArray<int, fdapde::full_dynamic_extent_t<2>, StorageOrder> other_shape(3, 2);
    coefficient = 9;
    for (auto& entry : other_shape) entry = ++coefficient;
    reshaped = other_shape.block(fdapde::full_extent, fdapde::full_extent);
    // assignment resizes the destination's first dynamic axis to three
    EXPECT_EQ(reshaped.extent(0), 3);
    // assignment resizes the destination's second dynamic axis to two
    EXPECT_EQ(reshaped.extent(1), 2);
    // resized assignment copies the source's final logical coefficient
    EXPECT_EQ(reshaped(2, 1), 15);

    std::array<double, 6> storage {};
    fdapde::MdMap<double, fdapde::MdExtents<2, 3>, StorageOrder> map(storage.data());
    map(1, 2) = 4.0;
    // writing through a map updates the offset computed by its storage mapping
    EXPECT_DOUBLE_EQ(storage[static_cast<std::size_t>(map.mapping()(1, 2))], 4.0);
    const auto& const_map_object = map;
    // a const map object returns a const scalar reference
    static_assert(std::is_same_v<decltype(const_map_object(0, 0)), const double&>);
    fdapde::MdMap<const double, fdapde::MdExtents<2, 3>, StorageOrder> read_only(storage.data());
    // a map bound to const storage returns a const scalar reference
    static_assert(std::is_same_v<decltype(read_only(0, 0)), const double&>);
    // the const map reads the value previously written through the mutable map
    EXPECT_DOUBLE_EQ(read_only(1, 2), 4.0);

    std::array<double, 6> dynamic_storage {};
    fdapde::MdMap<double, fdapde::full_dynamic_extent_t<2>, StorageOrder> dynamic_map(dynamic_storage.data(), 2, 3);
    // a fully dynamic 2-by-3 map exposes six coefficients
    EXPECT_EQ(dynamic_map.size(), 6);
    // a fully dynamic map retains the supplied row extent
    EXPECT_EQ(dynamic_map.extent(0), 2);
    // a fully dynamic map retains the supplied column extent
    EXPECT_EQ(dynamic_map.extent(1), 3);
    dynamic_map(0, 1) = 5.0;
    constexpr std::size_t DynamicOffset = StorageOrder == fdapde::RowMajor ? 1 : 2;
    // a dynamic map writes the expected physical offset for its storage order
    EXPECT_DOUBLE_EQ(dynamic_storage[DynamicOffset], 5.0);

    std::array<double, 6> partial_storage {};
    fdapde::MdMap<double, fdapde::MdExtents<fdapde::Dynamic, 3>, StorageOrder> partial_map(partial_storage.data(), 2);
    // a partially dynamic 2-by-3 map exposes six coefficients
    EXPECT_EQ(partial_map.size(), 6);
    // a partially dynamic map retains its first extent
    EXPECT_EQ(partial_map.extent(0), 2);
    // a partially dynamic map retains its second extent
    EXPECT_EQ(partial_map.extent(1), 3);
    partial_map(1, 0) = 6.0;
    constexpr std::size_t PartialOffset = StorageOrder == fdapde::RowMajor ? 3 : 1;
    // a partially dynamic map writes the expected physical offset for its storage order
    EXPECT_DOUBLE_EQ(partial_storage[PartialOffset], 6.0);

    fdapde::MdMap<double, fdapde::MdExtents<fdapde::Dynamic, 3>, StorageOrder> empty_map;
    // an empty map permits a null storage binding
    EXPECT_TRUE(empty_map.valid());
    // an empty map has zero coefficients
    EXPECT_EQ(empty_map.size(), 0);
    // an empty map has equal begin and end iterators
    EXPECT_EQ(empty_map.begin(), empty_map.end());
}

template <int StorageOrder> void check_mdarray_boolean_storage() {
    constexpr int OtherStorageOrder = StorageOrder == fdapde::RowMajor ? fdapde::ColMajor : fdapde::RowMajor;
    fdapde::MdArray<bool, fdapde::MdExtents<2, 2>, StorageOrder> small;
    small(0, 0) = true;
    small(0, 1) = small(0, 0);
    // writing a Boolean coefficient sets the addressed bit
    EXPECT_TRUE(small(0, 1));
    // a small Boolean owner needs one storage word
    EXPECT_EQ(small.bitpacks(), 1);

    fdapde::MdArray<bool, fdapde::MdExtents<2, 32>, StorageOrder> exact;
    exact.set();
    // an exactly full Boolean owner needs one storage word
    EXPECT_EQ(exact.bitpacks(), 1);
    // the final bit of the first logical row remains readable
    EXPECT_TRUE(exact(0, 31));
    // the first bit of the next logical row remains readable
    EXPECT_TRUE(exact(1, 0));

    fdapde::MdArray<bool, fdapde::MdExtents<2, 33>, StorageOrder> cross_pack;
    cross_pack(0, 32) = true;
    cross_pack(1, 0) = true;
    cross_pack(1, 32) = true;
    // crossing the storage-word boundary allocates a second word
    EXPECT_EQ(cross_pack.bitpacks(), 2);
    auto tail = cross_pack.block(fdapde::full_extent, std::pair {31, 32});
    fdapde::MdArray<bool, fdapde::MdExtents<2, 2>, StorageOrder> copy(tail);
    // copy construction preserves a false coefficient at the first coordinate
    EXPECT_FALSE(copy(0, 0));
    // copy construction preserves a true coefficient in the first row
    EXPECT_TRUE(copy(0, 1));
    // copy construction preserves a false coefficient in the second row
    EXPECT_FALSE(copy(1, 0));
    // copy construction preserves a true coefficient at the last coordinate
    EXPECT_TRUE(copy(1, 1));
    copy.clear();
    // clearing the copied bit updates that coefficient to false
    EXPECT_FALSE(copy(1, 1));

    fdapde::MdArray<bool, fdapde::MdExtents<2, 33>, OtherStorageOrder> source;
    source(0, 0) = true;
    source(0, 32) = true;
    source(1, 1) = true;
    source(1, 31) = true;
    fdapde::MdArray<bool, fdapde::MdExtents<2, 33>, StorageOrder> converted(source);
    for (int i = 0; i < converted.extent(0); ++i) {
        // cross-order Boolean construction preserves every logical bit
        for (int j = 0; j < converted.extent(1); ++j) EXPECT_EQ(converted(i, j), source(i, j));
    }

    fdapde::MdArray<bool, fdapde::MdExtents<2, 33>, StorageOrder> destination;
    destination.set();
    auto destination_view = destination.block(fdapde::full_extent, fdapde::full_extent);
    destination_view = source.block(fdapde::full_extent, fdapde::full_extent);
    for (int i = 0; i < destination.extent(0); ++i) {
        // cross-order Boolean assignment preserves every logical bit
        for (int j = 0; j < destination.extent(1); ++j) EXPECT_EQ(destination(i, j), source(i, j));
    }

    fdapde::MdArray<bool, fdapde::MdExtents<2, 34>, StorageOrder> aliasing;
    aliasing(0, 0) = true;
    aliasing(0, 33) = true;
    auto alias_destination = aliasing.block(fdapde::full_extent, std::pair {1, 33});
    auto alias_source = aliasing.block(fdapde::full_extent, std::pair {0, 32});
    alias_destination = alias_source;
    // an overlapping Boolean shift leaves the source's leading bit intact
    EXPECT_TRUE(aliasing(0, 0));
    // the shifted destination receives the original leading true bit
    EXPECT_TRUE(aliasing(0, 1));
    // the shifted destination receives the original second false bit
    EXPECT_FALSE(aliasing(0, 2));
    // the overlap-safe shift preserves a false bit across the word boundary
    EXPECT_FALSE(aliasing(0, 33));
    // the overlap-safe shift preserves the expected false bit in the second row
    EXPECT_FALSE(aliasing(1, 1));
}

template <int StorageOrder> void check_mdarray_runtime_failures() {
    using dynamic_array = fdapde::MdArray<int, fdapde::full_dynamic_extent_t<2>, StorageOrder>;
    constexpr int Max = std::numeric_limits<int>::max();

    // negative dynamic extents are rejected before allocation
    EXPECT_THROW(static_cast<void>(dynamic_array(-1, 3)), std::invalid_argument);
    // a nonempty extent product beyond the supported size raises length_error
    EXPECT_THROW(static_cast<void>(dynamic_array(Max, 2)), std::length_error);
    using partial_array = fdapde::MdArray<int, fdapde::MdExtents<fdapde::Dynamic, 2>, StorageOrder>;
    // runtime extents that conflict with a static axis are rejected
    EXPECT_THROW(static_cast<void>(partial_array(3, 3)), std::invalid_argument);

    dynamic_array preserved(2, 3);
    for (int i = 0; i < preserved.size(); ++i) preserved[i] = i + 1;
    const std::vector<int> preserved_values(preserved.begin(), preserved.end());
    // resize rejects a negative extent
    EXPECT_THROW(static_cast<void>(preserved.resize(-1, 3)), std::invalid_argument);
    // failed negative resize preserves the first extent
    EXPECT_EQ(preserved.extent(0), 2);
    // failed negative resize preserves the second extent
    EXPECT_EQ(preserved.extent(1), 3);
    // failed negative resize preserves all coefficients in iteration order
    EXPECT_EQ(std::vector<int>(preserved.begin(), preserved.end()), preserved_values);
    // resize rejects an overflowing nonempty extent product
    EXPECT_THROW(static_cast<void>(preserved.resize(Max, 2)), std::length_error);
    // failed overflowing resize preserves the first extent
    EXPECT_EQ(preserved.extent(0), 2);
    // failed overflowing resize preserves the second extent
    EXPECT_EQ(preserved.extent(1), 3);
    // failed overflowing resize preserves all coefficients in iteration order
    EXPECT_EQ(std::vector<int>(preserved.begin(), preserved.end()), preserved_values);

    fixed_md_array fixed;
    for (int i = 0; i < fixed.size(); ++i) fixed[i] = i + 1;
    const std::vector<int> fixed_values(fixed.begin(), fixed.end());
    dynamic_array wrong_shape(3, 2);
    // assignment to a fixed owner rejects an incompatible shape
    EXPECT_THROW(fixed = wrong_shape, std::invalid_argument);
    // failed fixed-shape assignment preserves all destination coefficients
    EXPECT_EQ(std::vector<int>(fixed.begin(), fixed.end()), fixed_values);

    // a nonempty fixed map rejects a null storage pointer
    EXPECT_THROW(static_cast<void>(fixed_md_map(nullptr)), std::invalid_argument);
    // a nonempty dynamic map rejects a null storage pointer
    EXPECT_THROW(
      static_cast<void>(fdapde::MdMap<int, fdapde::full_dynamic_extent_t<2>, StorageOrder>(nullptr, 2, 3)),
      std::invalid_argument);

    auto block_source = dynamic_array(2, 3);
    for (int i = 0; i < block_source.size(); ++i) block_source[i] = i + 1;
    const std::vector<int> block_values(block_source.begin(), block_source.end());
    // a block rejects a negative range endpoint
    EXPECT_THROW(static_cast<void>(block_source.block(std::pair {-1, 0}, fdapde::full_extent)), std::out_of_range);
    // a block rejects a range endpoint beyond the owner's first axis
    EXPECT_THROW(static_cast<void>(block_source.block(std::pair {0, 2}, fdapde::full_extent)), std::out_of_range);
    // a slice rejects a negative coordinate
    EXPECT_THROW(static_cast<void>(block_source.template slice<0>(-1)), std::out_of_range);
    // a slice rejects a coordinate equal to the selected axis extent
    EXPECT_THROW(static_cast<void>(block_source.template slice<1>(3)), std::out_of_range);
    // coefficient access rejects a negative row
    EXPECT_THROW(static_cast<void>(block_source(-1, 0)), std::out_of_range);
    // coefficient access rejects a row equal to the row extent
    EXPECT_THROW(static_cast<void>(block_source(2, 0)), std::out_of_range);
    // index-container access rejects the wrong number of dimensions
    EXPECT_THROW(static_cast<void>(block_source(std::array {0})), std::invalid_argument);

    auto row = block_source.row(0);
    dynamic_array incompatible(3, 1);
    incompatible(0, 0) = 7;
    // in-place row assignment rejects an incompatible source shape
    EXPECT_THROW(row.assign_inplace_from(incompatible), std::invalid_argument);
    // failed in-place assignment preserves every coefficient of the underlying owner
    EXPECT_EQ(std::vector<int>(block_source.begin(), block_source.end()), block_values);
}

// exercise multidimensional storage, maps, slices, aliasing and invalid shapes in both storage orders
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
