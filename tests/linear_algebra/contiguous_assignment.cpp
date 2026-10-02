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
#include <cstdint>

namespace fdapde {
namespace {

/// @brief checks short dense assignments against scalar coordinate references in both layouts
template <typename Scalar, int StorageOrder> void check_dense_assignment() {
    constexpr int OtherOrder = StorageOrder == RowMajor ? ColMajor : RowMajor;
    for (int rows : {0, 1, 3, 5, 17}) {
        Matrix<Scalar, Dynamic, Dynamic, StorageOrder> source(rows, 3), result(rows, 3);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < 3; ++j) source(i, j) = Scalar(10 * i + j + 1);
        }
        result.cwise() = Scalar(3);
        result.cwise() += Scalar(2);
        result.cwise() -= Scalar(1);
        result.cwise() *= Scalar(3);
        result.cwise() /= Scalar(2);
        for (int k = 0; k < result.size(); ++k) {
            // the scalar operations produce six at every physical offset, including odd tails
            EXPECT_EQ(result.data()[k], Scalar(6));
        }
        // zero rows perform no accesses and preserve the empty shape
        EXPECT_EQ(result.size(), rows * 3);
        result = source;
        result += source;
        for (int k = 0; k < result.size(); ++k) {
            // same-layout addition doubles the saved source before subsequent subtraction
            EXPECT_EQ(result.data()[k], Scalar(2) * source.data()[k]);
        }
        result -= source;
        result *= Scalar(2);
        result /= Scalar(2);
        const Matrix<Scalar, Dynamic, Dynamic, OtherOrder> reordered(source);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < 3; ++j) {
                // same-layout copy and compound assignments reproduce the coordinate oracle
                EXPECT_EQ(result(i, j), Scalar(10 * i + j + 1));
                // cross-layout construction keeps logical coordinates rather than raw offsets
                EXPECT_EQ(reordered(i, j), Scalar(10 * i + j + 1));
            }
        }
    }
}

/// @brief checks snapshot assignments between overlapping external dense storage ranges
template <int StorageOrder> void check_overlapping_views() {
    std::array<double, 17> storage;
    for (int k = 0; k < 17; ++k) storage[k] = k + 1.0;
    const auto original = storage;
    MatrixView<const double, 3, 5, StorageOrder> source(storage.data());
    MatrixView<double, 3, 5, StorageOrder> destination(storage.data() + 1);
    destination = source;
    for (int k = 0; k < 15; ++k) {
        // overlapping copy reads the saved source before writing the next physical coefficient
        EXPECT_DOUBLE_EQ(storage[k + 1], original[k]);
    }
    storage = original;
    destination = 2.0 * source + destination;
    for (int k = 0; k < 15; ++k) {
        // the affine assignment reads both operands from their original overlapping buffers
        EXPECT_DOUBLE_EQ(storage[k + 1], 2.0 * original[k] + original[k + 1]);
    }
    // the value preceding the destination remains outside every assignment
    EXPECT_DOUBLE_EQ(storage.front(), original.front());
    // the value after the destination remains outside every assignment
    EXPECT_DOUBLE_EQ(storage.back(), original.back());
}

}   // namespace

// checks empty, short and odd dense assignments for floating-point and integral scalar types
TEST(ContiguousAssignment, DenseLayoutsAndScalarOperations) {
    check_dense_assignment<double, RowMajor>();
    check_dense_assignment<double, ColMajor>();
    check_dense_assignment<float, RowMajor>();
    check_dense_assignment<float, ColMajor>();
    check_dense_assignment<int, RowMajor>();
    check_dense_assignment<int, ColMajor>();
}

// checks vector orientation, scalar broadcast and affine expressions against indexed values
TEST(ContiguousAssignment, VectorOrientationAndAffineExpressions) {
    for (int size : {0, 1, 3, 5, 17}) {
        Matrix<float, 1, Dynamic, ColMajor> row(size);
        for (int k = 0; k < size; ++k) row[k] = float(k + 1);
        Vector<float, Dynamic> column(row), offset(size);
        offset.cwise() = 3.0f;
        column = 2.0f * column + offset;
        for (int k = 0; k < size; ++k) {
            // row-to-column construction and affine assignment retain the indexed source order
            EXPECT_FLOAT_EQ(column[k], 2.0f * float(k + 1) + 3.0f);
        }
        // empty and nonempty row sources retain their size after column construction
        EXPECT_EQ(column.size(), size);
    }
    const Matrix<double, 1, 5, ColMajor> row({1, 2, 3, 4, 5});
    Vector<double, 5> column;
    column = row;
    for (int k = 0; k < 5; ++k) {
        // fixed row-to-column assignment uses vector index order despite differing layouts
        EXPECT_DOUBLE_EQ(column[k], k + 1.0);
    }
    const Vector<double, Dynamic> short_source(4);
    // a runtime-sized source with four coefficients cannot overwrite a fixed five-entry vector
    EXPECT_THROW(column = short_source, std::invalid_argument);

    Matrix<double, Dynamic, Dynamic> matrix(5, 3);
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 3; ++j) matrix(i, j) = 10.0 * i + j;
    }
    const Vector<double, Dynamic> from_column_block(matrix.block(0, 1, 5, 1));
    for (int i = 0; i < 5; ++i) {
        // a runtime-shaped column block follows its strided coordinate oracle without vector indexing
        EXPECT_DOUBLE_EQ(from_column_block[i], 10.0 * i + 1.0);
    }
    const Matrix<double, 1, Dynamic> from_row_block(matrix.block(2, 0, 1, 3));
    for (int j = 0; j < 3; ++j) {
        // a runtime-shaped row block follows its coordinate oracle without static vector dimensions
        EXPECT_DOUBLE_EQ(from_row_block[j], 20.0 + j);
    }
    Matrix<bool, Dynamic, Dynamic> condition(5, 1);
    Vector<double, Dynamic> lhs(5), rhs(5);
    for (int k = 0; k < 5; ++k) {
        condition(k, 0) = k % 2 == 0;
        lhs[k] = 10.0 + k;
        rhs[k] = 20.0 + k;
    }
    const Vector<double, Dynamic> selected(condition.select(lhs, rhs));
    for (int k = 0; k < 5; ++k) {
        // a dynamic matrix mask selects the coordinate oracle without requiring vector-shaped mask types
        EXPECT_DOUBLE_EQ(selected[k], (k % 2 == 0 ? 10.0 : 20.0) + k);
    }
}

// checks unaligned external buffers and verifies sentinels around a fifteen-coefficient assignment
TEST(ContiguousAssignment, UnalignedViewsPreserveSentinels) {
    alignas(64) std::array<double, 17> source_storage, destination_storage;
    source_storage.fill(-71.0);
    destination_storage.fill(-83.0);
    for (int k = 0; k < 15; ++k) source_storage[k + 1] = k + 1.0;
    MatrixView<const double, 3, 5, ColMajor> source(source_storage.data() + 1);
    MatrixView<double, 3, 5, ColMajor> destination(destination_storage.data() + 1);
    // adding one double to a cache-line aligned buffer prevents sixteen-byte alignment
    ASSERT_EQ(reinterpret_cast<std::uintptr_t>(destination.data()) % 16, sizeof(double));
    const MatrixView<const double, Dynamic, Dynamic, ColMajor> wrong_shape(source.data(), 1, 15);
    // equal coefficient counts cannot bypass the debug check for incompatible matrix shapes
    EXPECT_THROW(destination = wrong_shape, std::invalid_argument);
    destination = source;
    destination *= 2.0;
    destination.cwise() += 3.0;
    for (int k = 0; k < 15; ++k) {
        // each unaligned coefficient matches the scalar arithmetic oracle, including the tail
        EXPECT_DOUBLE_EQ(destination_storage[k + 1], 2.0 * (k + 1.0) + 3.0);
    }
    // the prefix sentinel detects writes before the bound range
    EXPECT_DOUBLE_EQ(destination_storage.front(), -83.0);
    // the suffix sentinel detects writes beyond the odd tail
    EXPECT_DOUBLE_EQ(destination_storage.back(), -83.0);
}

// checks alias-safe copies and affine expressions in both physical storage orders
TEST(ContiguousAssignment, OverlappingViewsMaterializeOriginalCoefficients) {
    check_overlapping_views<RowMajor>();
    check_overlapping_views<ColMajor>();
}

// checks that optimized assignment paths retain compile-time evaluation of fixed storage
TEST(ContiguousAssignment, FixedAssignmentsRemainConstexpr) {
    constexpr auto result = [] {
        const Matrix<float, 2, 3, ColMajor> source({1, 2, 3, 4, 5, 6});
        Matrix<float, 2, 3, ColMajor> destination(source);
        destination.cwise() += 3.0f;
        destination *= 2.0f;
        return destination;
    }();
    // fixed copy, scalar broadcast and self scaling match the first and last coordinate oracles
    static_assert(result(0, 0) == 8.0f && result(1, 2) == 18.0f);
}

}   // namespace fdapde
