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

using namespace fdapde;

using full_cache =
  Cache::Union<Cache::Spectral, Cache::Log, Cache::Sqrt, Cache::InverseSqrt, Cache::LogDividedDifferences>;
using cached_point = SPDMatrix<double, 2, 2, full_cache>;

/// @brief returns the leading coefficient of a batch element for lifetime contract checks
struct leading_coefficient {
    /// @brief reads the first coefficient from one batch element
    template <typename Value> auto operator()(const Value& value) const { return value(0, 0); }
};

template <typename Batch>
concept permits_const_rvalue_map = requires(const Batch& batch) { std::move(batch).map(leading_coefficient {}); };
template <typename Batch>
concept permits_const_rvalue_select = requires(const Batch& batch) { std::move(batch).select(std::array<int, 1> {0}); };
template <typename Batch>
concept permits_const_rvalue_derived = requires(const Batch& batch) { std::move(batch).derived(); };

using scalar_batch = MatrixBatch<Matrix<double, 1, 1>>;
using scalar_map = decltype(std::declval<scalar_batch&>().map(leading_coefficient {}));

// owning batches cannot be borrowed by a deferred map after a const temporary expires
static_assert(!permits_const_rvalue_map<scalar_batch>);
// expression nodes cannot be borrowed by a deferred map after a const temporary expires
static_assert(!permits_const_rvalue_map<scalar_map>);
// owning batches cannot be borrowed by a selection after a const temporary expires
static_assert(!permits_const_rvalue_select<scalar_batch>);
// expression nodes cannot be borrowed by a selection after a const temporary expires
static_assert(!permits_const_rvalue_select<scalar_map>);
// the CRTP base cannot expose a derived reference from a const temporary
static_assert(!permits_const_rvalue_derived<scalar_map>);

template <typename Actual, typename Expected> void expect_matrix_eq(const Actual& actual, const Expected& expected) {
    // matching row counts make each comparison coordinate valid
    ASSERT_EQ(actual.rows(), expected.rows());
    // matching column counts make each comparison coordinate valid
    ASSERT_EQ(actual.cols(), expected.cols());
    for (int i = 0; i < actual.rows(); ++i) {
        for (int j = 0; j < actual.cols(); ++j) {
            // each visible coefficient retains its source value
            EXPECT_DOUBLE_EQ(static_cast<double>(actual(i, j)), static_cast<double>(expected(i, j)));
        }
    }
}

// coefficient rows use native packed layouts for every supported matrix structure
TEST(MatrixBatch, UsesCompactRowsForStaticAndDynamicElements) {
    MatrixBatch<Matrix<double, 2, 2>> dense(2);
    auto dense_coefficients = dense.coefficients();
    for (std::size_t i = 0; i < dense_coefficients.size(); ++i) dense_coefficients[i] = static_cast<double>(i + 1);
    // dense rows contain all four coefficients
    EXPECT_EQ(dense.coefficient_stride(), std::size_t {4});
    // dense coefficients are contiguous in row-major element rows
    EXPECT_EQ(dense_coefficients[4], 5);
    // the second dense view starts at the second coefficient row
    EXPECT_DOUBLE_EQ(dense[1](0, 0), 5);

    MatrixBatch<Matrix<double, Dynamic, Dynamic>> dynamic_dense(2, 2, 3);
    // dynamic dense rows retain their runtime six-coefficient extent
    EXPECT_EQ(dynamic_dense.coefficient_stride(), std::size_t {6});
    // dynamic dense batches preserve the supplied row extent
    EXPECT_EQ(dynamic_dense.rows(), 2);
    // dynamic dense batches preserve the supplied column extent
    EXPECT_EQ(dynamic_dense.cols(), 3);

    MatrixBatch<SymmetricMatrix<double, 3, 3>> symmetric(2);
    // symmetric rows retain only the six native packed lower coefficients
    EXPECT_EQ(symmetric.coefficient_stride(), std::size_t {6});
    // the second symmetric row follows the first packed row directly
    EXPECT_EQ(symmetric.coefficients().data() + 6, symmetric[1].data());

    MatrixBatch<SymmetricMatrix<double, Dynamic, Dynamic>> dynamic_symmetric(2, 3, 3);
    // dynamic symmetric rows retain the runtime packed lower triangle
    EXPECT_EQ(dynamic_symmetric.coefficient_stride(), std::size_t {6});

    MatrixBatch<DiagonalMatrix<double, 3>> diagonal(2);
    // diagonal rows retain only their three diagonal coefficients
    EXPECT_EQ(diagonal.coefficient_stride(), std::size_t {3});
    // the second diagonal view begins at the following compact row
    EXPECT_EQ(diagonal.coefficients().data() + 3, diagonal[1].data());

    MatrixBatch<DiagonalMatrix<double, Dynamic>> dynamic_diagonal(2, 3, 3);
    // dynamic diagonal rows retain their runtime diagonal length
    EXPECT_EQ(dynamic_diagonal.coefficient_stride(), std::size_t {3});

    MatrixBatch<SPDMatrix<double, Dynamic, Dynamic, Cache::Log>> dynamic_spd(2, 2, 2);
    // dynamic SPD rows retain the runtime packed lower triangle
    EXPECT_EQ(dynamic_spd.coefficient_stride(), std::size_t {3});
    // dynamic SPD points retain their requested runtime order
    EXPECT_EQ(dynamic_spd[1].rows(), 2);
}

// cached SPD rows initialize identity coefficients and aggregate cache slots without per-point owners
TEST(MatrixBatch, InitializesContiguousCachedSPDIdentitySlots) {
    MatrixBatch<cached_point> points(3);
    const auto pointers = points.cache_pointers();
    const auto stride = cached_point::CacheSlot::scalar_count(2);

    // SPD rows retain their three native packed coefficients
    EXPECT_EQ(points.coefficient_stride(), std::size_t {3});
    // one cache slot pointer is exposed for each SPD point
    ASSERT_EQ(pointers.size(), points.size());
    // adjacent cache slots refer to adjacent aggregate scalar ranges
    EXPECT_EQ(pointers[1]->data() - pointers[0]->data(), static_cast<std::ptrdiff_t>(stride));
    // the final cache slot stays in the same aggregate scalar range
    EXPECT_EQ(pointers[2]->data() - pointers[1]->data(), static_cast<std::ptrdiff_t>(stride));
    for (std::size_t k = 0; k < points.size(); ++k) {
        const auto point = points[k];
        for (int i = 0; i < point.rows(); ++i) {
            for (int j = 0; j < point.cols(); ++j) {
                // every SPD point begins with identity coefficients
                EXPECT_EQ(point(i, j), i == j ? 1 : 0);
                // identity divided differences use the repeated-spectrum limit one
                EXPECT_EQ(point.cache().log_divided_differences()(i, j), 1);
            }
        }
    }
}

// copies and moves retain independent cache storage while preserving coefficient values
TEST(MatrixBatch, CopiesAndMovesRebuildCachedSlotBindings) {
    MatrixBatch<cached_point> source(2);
    const Matrix<double, 2, 2> replacement({4, 0, 0, 9});
    source[0].assign(replacement);
    const MatrixBatch<cached_point> copy(source);
    const auto source_pointers = source.cache_pointers();
    const auto copy_pointers = copy.cache_pointers();

    // copied batches do not share the first cache scalar buffer
    EXPECT_NE(source_pointers[0]->data(), copy_pointers[0]->data());
    // copied batches retain the assigned source coefficients
    expect_matrix_eq(copy[0], replacement);

    source[0].assign(Matrix<double, 2, 2>({16, 0, 0, 25}));
    // replacing source cache data cannot mutate an independent copied slot
    expect_matrix_eq(copy[0], replacement);

    MatrixBatch<cached_point> moved(std::move(source));
    // moved batches retain the assigned source coefficients
    expect_matrix_eq(moved[0], Matrix<double, 2, 2>({16, 0, 0, 25}));
    // moved batches retain valid cache slot bindings
    EXPECT_NE(moved.cache_pointers()[0]->data(), nullptr);
    // moved-from batches are empty after ownership transfer
    EXPECT_TRUE(source.empty());
    // moved-from batches expose no residual coefficient storage
    EXPECT_TRUE(source.coefficients().empty());
    // moved-from batches expose no cache pointers into the moved-to owner
    EXPECT_TRUE(source.cache_pointers().empty());
}

// cache-free batches have no cache-bearing layout state and ordinary entries remain writable
TEST(MatrixBatch, UsesNoCachePolicyForOrdinaryMatrices) {
    using dense_batch = MatrixBatch<Matrix<double, 2, 2>>;
    // ordinary batch elements select the no-cache policy
    static_assert(dense_batch::CachePolicy::Flags == Cache::None::Flags);
    // no-cache SPD batches match coefficient-equivalent symmetric batches on this toolchain
    static_assert(sizeof(MatrixBatch<SPDMatrix<double, 2, 2>>) == sizeof(MatrixBatch<SymmetricMatrix<double, 2, 2>>));

    dense_batch values(1);
    values.coefficients()[0] = 7;
    // ordinary coefficient spans permit controlled batch-level writes
    EXPECT_EQ(values[0](0, 0), 7);
}

// deferred maps evaluate only requested elements and normalize scalar results to one-by-one matrices
TEST(MatrixBatch, MapsLazilyAndPreservesResultStructure) {
    MatrixBatch<Matrix<double, 2, 2>> values(3);
    for (std::size_t i = 0; i < values.size(); ++i) values.coefficients()[i * 4] = static_cast<double>(i + 1);
    int calls = 0;
    const auto squares = values.map([&calls](const auto& value) {
        ++calls;
        return value(0, 0) * value(0, 0);
    });

    // map creation does not invoke the deferred callable
    EXPECT_EQ(calls, 0);
    const auto first = squares[0];
    // one indexed access invokes the callable for only that source element
    EXPECT_EQ(calls, 1);
    // scalar map results materialize as one-by-one matrices
    EXPECT_EQ(first.rows(), 1);
    // scalar map results retain their one output column
    EXPECT_EQ(first.cols(), 1);
    // scalar output stores the mapped source value
    EXPECT_EQ(first(0, 0), 1);

    MatrixBatch<DiagonalMatrix<double, 2>> diagonal(2);
    diagonal.coefficients()[0] = 2;
    diagonal.coefficients()[1] = 3;
    const auto diagonal_map = diagonal.map([](const auto& value) { return DiagonalMatrix<double, 2>(value); });
    MatrixBatch<DiagonalMatrix<double, 2>> diagonal_result(diagonal_map);
    // a diagonal map materializes a diagonal batch with compact rows
    EXPECT_EQ(diagonal_result.coefficient_stride(), std::size_t {2});
    // the diagonal map retains the first diagonal coefficient
    EXPECT_EQ(diagonal_result[0](0, 0), 2);
    // the diagonal map preserves implicit off-diagonal zeros
    EXPECT_EQ(diagonal_result[0](0, 1), 0);

    const auto chained =
      values.map([](const auto& value) { return value + value; }).map([](const auto& value) { return value(0, 0); });
    MatrixBatch<Matrix<double, 1, 1>> chained_result(chained);
    // nested temporary map nodes survive until chained materialization
    EXPECT_EQ(chained_result[2](0, 0), 6);

    const auto stateful = values.map([offset = 0](const auto& value) mutable { return value(0, 0) + offset++; });
    // a callable stored by value may update its own state between deferred element evaluations
    EXPECT_EQ(stateful[0](0, 0), 1);
    // later evaluations observe the mutable callable state retained by the map expression
    EXPECT_EQ(stateful[1](0, 0), 3);
}

// reductions run in index order and preserve the supplied initializer for empty batches
TEST(MatrixBatch, ReducesInOrderAndHandlesEmptyBatches) {
    MatrixBatch<Matrix<int, 1, 1>> values(3);
    values.coefficients()[0] = 1;
    values.coefficients()[1] = 2;
    values.coefficients()[2] = 3;
    const auto ordered =
      values.redux(0, [](int accumulator, const auto& value) { return accumulator * 10 + value(0, 0); });
    const MatrixBatch<Matrix<int, 1, 1>> empty;
    const auto untouched = empty.redux(17, [](int accumulator, const auto&) { return accumulator + 1; });

    // redux visits source elements in increasing index order
    EXPECT_EQ(ordered, 123);
    // redux returns init when no source elements exist
    EXPECT_EQ(untouched, 17);
}

// selections copy index values while borrowing original coefficient and cache storage in the requested order
TEST(MatrixBatch, SelectsSharedEntriesWithCopiedIndices) {
    MatrixBatch<cached_point> points(3);
    points[0].assign(Matrix<double, 2, 2>({4, 0, 0, 9}));
    points[2].assign(Matrix<double, 2, 2>({16, 0, 0, 25}));
    std::vector<int> indices {2, 0, 2};
    const auto selected = points.select(indices);
    indices[0] = 1;

    // selection size retains repeated user-supplied indices
    ASSERT_EQ(selected.size(), std::size_t {3});
    // copied indices preserve the original first selected position
    EXPECT_EQ(selected[0](0, 0), 16);
    // selection preserves the requested second source position
    EXPECT_EQ(selected[1](1, 1), 9);
    // repeated selections borrow the same original coefficient storage
    EXPECT_EQ(selected[0].data(), points[2].data());
    // selections borrow the original cache slots without copying them
    EXPECT_EQ(selected[0].cache().data(), points[2].cache().data());

    const auto nested = points.select(std::array<int, 2> {0, 2}).map([](const auto& value) { return value(0, 0); });
    MatrixBatch<Matrix<double, 1, 1>> nested_result(nested);
    // temporary selections remain alive while their native map nodes materialize
    EXPECT_EQ(nested_result[1](0, 0), 16);
}

// checked SPD replacements leave coefficients and cache quantities unchanged when validation rejects a candidate
TEST(MatrixBatch, PreservesAssignedSPDValueAfterFailure) {
    MatrixBatch<cached_point> points(1);
    points[0].assign(Matrix<double, 2, 2>({4, 0, 0, 9}));
    const auto before = points[0];
    const auto cache_before = before.cache().template matrix<Cache::Log>();
    const Matrix<double, 2, 2> indefinite({-1, 0, 0, 1});

    // checked element assignment rejects a nonpositive-definite candidate
    EXPECT_THROW(points[0].assign(indefinite), std::domain_error);
    // failed assignment retains the previous verified coefficients
    expect_matrix_eq(points[0], before);
    // failed assignment retains the previous prepared logarithm
    expect_matrix_eq(points[0].cache().template matrix<Cache::Log>(), cache_before);
}

// public boundaries reject invalid indices and dimension or coefficient allocation overflows
TEST(MatrixBatch, RejectsInvalidIndicesAndOversizedRequests) {
    MatrixBatch<Matrix<double, 1, 1>> values(1);
    // element access rejects an index equal to the batch size
    EXPECT_THROW(static_cast<void>(values[1]), std::out_of_range);
    // map expression access rejects an index equal to its source size
    EXPECT_THROW(static_cast<void>(values.map([](const auto& value) { return value; })[1]), std::out_of_range);
    // selection rejects a negative signed index
    EXPECT_THROW(static_cast<void>(values.select(std::array<int, 1> {-1})), std::out_of_range);
    // selection rejects an index beyond the source range
    EXPECT_THROW(static_cast<void>(values.select(std::array<unsigned, 1> {1})), std::out_of_range);
    // dynamic dimensions reject dense workspace products beyond the supported int range
    EXPECT_THROW((MatrixBatch<Matrix<double, Dynamic, Dynamic>>(1, 50000, 50000)), std::length_error);
    // coefficient allocation rejects a count that cannot fit the backing vector
    EXPECT_THROW((MatrixBatch<Matrix<double, 1, 1>>(std::numeric_limits<std::size_t>::max())), std::length_error);
}

// a throwing map aborts construction in index order and leaves the assigned batch unchanged
TEST(MatrixBatch, FailedMaterializationPreservesDestinationAndStopsEvaluation) {
    MatrixBatch<SPDMatrix<double, 2, 2, full_cache>> source(3);
    auto destination = source;
    const auto* original_cache = destination[0].cache().data();
    int calls = 0;
    const auto expression = source.map([&](const auto& value) {
        ++calls;
        // the callback injects a failure on its second visit to exercise partially built candidate cleanup
        fdapde_strong_assert(calls != 2, std::runtime_error, "injected map failure");
        return value;
    });
    // the second callback failure propagates before the candidate replaces destination
    EXPECT_THROW(destination = expression, std::runtime_error);
    // increasing-index evaluation stops at the failing second element
    EXPECT_EQ(calls, 2);
    // failed whole-batch assignment retains the destination's original storage binding
    EXPECT_EQ(destination[0].cache().data(), original_cache);
    // the original identity value remains readable with its ready logarithm after cleanup
    EXPECT_DOUBLE_EQ(destination[0](0, 0), 1);
    // the retained cache is still the logarithm of the unchanged identity
    EXPECT_DOUBLE_EQ(destination[0].cache().template matrix<Cache::Log>()(0, 0), 0);
}

}   // namespace
