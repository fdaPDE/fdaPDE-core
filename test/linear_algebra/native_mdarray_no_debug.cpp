// This file is part of fdaPDE, a C++ library for physics-informed
// spatial and functional data analysis.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

#include <fdaPDE/linear_algebra.h>

#include <array>
#include <limits>

int main() {
    namespace native = fdapde::linalg;
    using dynamic_array = native::MdArray<int, native::full_dynamic_extent_t<2>>;

    dynamic_array negative(-1, 3);
    if (negative.valid() || negative.size() != 0 || negative.extent(0) != 0 || negative.extent(1) != 0) return 1;

    dynamic_array overflow(std::numeric_limits<int>::max(), 2);
    if (overflow.size() != 0) return 2;

    native::MdArray<int, native::full_dynamic_extent_t<3>> large_empty(
      0, std::numeric_limits<int>::max(), std::numeric_limits<int>::max());
    if (!large_empty.valid() || large_empty.size() != 0) return 15;
    auto large_empty_block =
      large_empty.block(native::full_extent, std::numeric_limits<int>::max() - 1, native::full_extent);
    if (!large_empty_block.valid() || large_empty_block.size() != 0) return 19;
    auto large_empty_slice = large_empty.template slice<1>(std::numeric_limits<int>::max() - 1);
    if (!large_empty_slice.valid() || large_empty_slice.size() != 0) return 20;
    native::MdArray<int, native::full_dynamic_extent_t<3>, native::ColMajor> large_empty_column_major(
      std::numeric_limits<int>::max(), std::numeric_limits<int>::max(), 0);
    if (!large_empty_column_major.valid() || large_empty_column_major.size() != 0) return 16;
    auto large_empty_column_slice = large_empty_column_major.template slice<1>(std::numeric_limits<int>::max() - 1);
    if (!large_empty_column_slice.valid() || large_empty_column_slice.size() != 0) return 21;

    native::MdArray<int, native::MdExtents<fdapde::Dynamic, 2, fdapde::Dynamic>> mixed(3, 4);
    if (mixed.resize(3, 99, 4)) return 3;
    if (mixed.extent(0) != 3 || mixed.extent(1) != 2 || mixed.extent(2) != 4) return 4;
    if (mixed.resize(-1, 4)) return 5;
    if (mixed.extent(0) != 3 || mixed.extent(2) != 4) return 6;

    dynamic_array destination(2, 3);
    destination[0] = 9;
    destination = negative;
    if (!destination.valid() || destination.extent(0) != 2 || destination.extent(1) != 3 || destination[0] != 9)
        return 17;
    const std::array<int, 1> short_index {1};
    if (destination(short_index) != 9) return 18;

    native::MdMap<int, native::MdExtents<2, 3>> null_fixed(nullptr);
    if (null_fixed.valid() || null_fixed.size() != 0) return 7;
    native::MdMap<int, native::full_dynamic_extent_t<2>> null_dynamic(nullptr, 2, 3);
    if (null_dynamic.valid() || null_dynamic.size() != 0) return 8;
    int storage = 0;
    native::MdMap<int, native::full_dynamic_extent_t<2>> invalid_shape(&storage, -1, 3);
    if (invalid_shape.valid() || invalid_shape.size() != 0) return 14;

    dynamic_array array(2, 3);
    auto negative_block = array.block(std::pair {-1, 0}, native::full_extent);
    if (negative_block.valid() || negative_block.size() != 0) return 9;
    auto wide_block = array.block(std::pair {0, 2}, std::pair {0, 3});
    if (wide_block.valid() || wide_block.size() != 0) return 10;
    auto negative_slice = array.template slice<0>(-1);
    if (negative_slice.valid() || negative_slice.size() != 0) return 11;
    auto wide_slice = array.template slice<1>(3);
    if (wide_slice.valid() || wide_slice.size() != 0) return 12;

    auto row = array.row(0);
    dynamic_array wrong_shape(3, 1);
    wrong_shape[0] = 9;
    row.assign_inplace_from(wrong_shape);
    if (array(0, 0) != 0 || array(0, 1) != 0 || array(0, 2) != 0) return 13;
    return 0;
}
