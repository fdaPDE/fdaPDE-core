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

namespace {

static_assert(fdapde::internals::bitpack_count(0, 64) == 0);
static_assert(fdapde::internals::bitpack_count(1, 64) == 1);
static_assert(fdapde::internals::bitpack_count(63, 64) == 1);
static_assert(fdapde::internals::bitpack_count(64, 64) == 1);
static_assert(fdapde::internals::bitpack_count(65, 64) == 2);
static_assert(
  fdapde::internals::bitpack_count(std::numeric_limits<int>::max(), 64) == 33'554'432);

template <int StorageOrder> void check_exact_boolean_pack_accounting() {
    using exact_pack = fdapde::Matrix<bool, 8, 8, StorageOrder>;
    using partial_pack = fdapde::Matrix<bool, 5, 13, StorageOrder>;
    using dynamic_column = fdapde::Matrix<bool, fdapde::Dynamic, 1, StorageOrder>;
    using partial_owner = fdapde::Matrix<bool, 2, fdapde::Dynamic, StorageOrder>;

    static_assert(exact_pack::StorageSize == 1);
    static_assert(partial_pack::StorageSize == 2);

    exact_pack bits;
    EXPECT_EQ(bits.bitpacks(), 1);
    EXPECT_EQ(bits.bitpack(0), 0u);
    bits(1, 6) = true;
    constexpr int expected_bit = StorageOrder == fdapde::RowMajor ? 14 : 49;
    EXPECT_NE(bits.bitpack(0) & (typename exact_pack::bitpack_t(1) << expected_bit), 0u);
    const exact_pack copied = bits;
    EXPECT_TRUE(copied == bits);

    const partial_pack partial;
    EXPECT_EQ(partial.bitpacks(), 2);
    EXPECT_EQ(partial.bitpack(0), 0u);
    EXPECT_EQ(partial.bitpack(1), 0u);

    partial_owner normalized_owner(2, 64);
    EXPECT_EQ(normalized_owner.rows(), 2);
    EXPECT_EQ(normalized_owner.cols(), 64);
    EXPECT_EQ(normalized_owner.bitpacks(), 2);
    normalized_owner(1, 63) = true;
    EXPECT_TRUE(normalized_owner(1, 63));

    constexpr std::array<int, 6> sizes {0, 1, 63, 64, 65, 130};
    constexpr std::array<int, 6> expected_counts {0, 1, 1, 1, 2, 3};
    for (std::size_t i = 0; i < sizes.size(); ++i) {
        const dynamic_column dynamic(sizes[i]);
        EXPECT_EQ(dynamic.bitpacks(), expected_counts[i]);
    }

    exact_pack block_owner;
    auto whole = block_owner.template block<8, 8>(0, 0);
    EXPECT_EQ(whole.bitpacks(), 1);
    whole.set();
    for (int i = 0; i < block_owner.rows(); ++i) {
        for (int j = 0; j < block_owner.cols(); ++j) EXPECT_TRUE(block_owner(i, j));
    }
    whole.clear();

    auto trailing = block_owner.template block<2, 3>(6, 5);
    trailing.set();
    for (int i = 0; i < block_owner.rows(); ++i) {
        for (int j = 0; j < block_owner.cols(); ++j) {
            EXPECT_EQ(bool(block_owner(i, j)), i >= 6 && j >= 5);
        }
    }
    trailing.clear();
    for (int i = 0; i < block_owner.rows(); ++i) {
        for (int j = 0; j < block_owner.cols(); ++j) EXPECT_FALSE(block_owner(i, j));
    }

    using view_type = fdapde::MatrixView<bool, 8, 8, StorageOrder>;
    using bitpack_t = typename view_type::bitpack_t;
    constexpr bitpack_t canary = bitpack_t(0x5a5a);
    std::array<bitpack_t, 2> storage {bitpack_t(0), canary};
    view_type view(storage.data());
    EXPECT_EQ(view.bitpacks(), 1);
    view.set();
    EXPECT_EQ(storage[0], std::numeric_limits<bitpack_t>::max());
    EXPECT_EQ(storage[1], canary);
    EXPECT_TRUE(view == exact_pack(true));
    view.clear();
    EXPECT_EQ(storage[0], bitpack_t(0));
    EXPECT_EQ(storage[1], canary);

    using partial_view = fdapde::MatrixView<bool, 2, fdapde::Dynamic, StorageOrder>;
    std::array<bitpack_t, 3> partial_storage {bitpack_t(0), bitpack_t(0), canary};
    partial_view normalized_view(partial_storage.data(), 2, 64);
    EXPECT_EQ(normalized_view.rows(), 2);
    EXPECT_EQ(normalized_view.cols(), 64);
    EXPECT_EQ(normalized_view.bitpacks(), 2);
    normalized_view.set(1, 63);
    EXPECT_TRUE(normalized_view(1, 63));
    EXPECT_EQ(partial_storage[2], canary);
}

}   // namespace

TEST(linear_algebra, boolean) {
    fdapde::Matrix<bool, 2, 3> fixed({true, false, true, false, true, false});
    EXPECT_EQ(fixed.rows(), 2);
    EXPECT_EQ(fixed.cols(), 3);
    EXPECT_TRUE(fixed(0, 0));
    EXPECT_FALSE(fixed(0, 1));

    fixed(1, 0).set();
    fixed(0, 2).clear();
    EXPECT_TRUE(fixed(1, 0));
    EXPECT_FALSE(fixed(0, 2));

    using dynamic_bool_matrix = fdapde::Matrix<bool, fdapde::Dynamic, fdapde::Dynamic>;
    constexpr int pack_size = static_cast<int>(dynamic_bool_matrix::PackSize);
    dynamic_bool_matrix dynamic(1, pack_size + 1);
    dynamic(0, pack_size - 1) = true;
    dynamic(0, pack_size) = true;

    dynamic_bool_matrix copy = dynamic;
    EXPECT_TRUE(copy(0, pack_size - 1));
    EXPECT_TRUE(copy(0, pack_size));
    copy(0, pack_size - 1).clear();
    EXPECT_FALSE(copy(0, pack_size - 1));
    EXPECT_TRUE(copy(0, pack_size));
    EXPECT_TRUE(dynamic(0, pack_size - 1));

    check_exact_boolean_pack_accounting<fdapde::RowMajor>();
    check_exact_boolean_pack_accounting<fdapde::ColMajor>();
}

// Current regression adapted from 86ff6d12:tests/linear_algebra/bool.cpp.
// Stable source: a2a9c88:test/src/binary_matrix_test.cpp.
// Stable declarations (9): static_sized_matrix, dynamic_sized_matrix, binary_vector, block_operations,
// binary_expresssions, visitors, block_repeat, eigen_assignment_and_construct, and reshaped.
// TODO(P4-B): cover exact pack sizing, resize, views, aliasing, Boolean vectors, row/column/block access,
// expressions and reductions, repeat, reshape, and select.
// Replace the historical Eigen assignment/construct assertion with native numeric-matrix conversion; do not
// restore an implicit Eigen bridge.
