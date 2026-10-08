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
#include <cmath>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using namespace fdapde;

using full_cache =
  Cache::Union<Cache::Spectral, Cache::Log, Cache::Sqrt, Cache::InverseSqrt, Cache::LogDividedDifferences>;
using cached_point = SPDMatrix<double, 2, full_cache>;

/// @brief bounds the standalone batch suite's runtime before any parallel transform begins
class BatchWorkerEnvironment : public ::testing::Environment {
    /// @brief selects four workers for deterministic resource use in parallel batch tests
    void SetUp() override {
        parallel_set_num_threads(4);
        // the standalone batch executable initializes exactly the requested four workers
        ASSERT_EQ(parallel_get_num_threads(), 4);
    }
    /// @brief completes outstanding tasks before the test process exits
    void TearDown() override { parallel_join(); }
};

[[maybe_unused]] ::testing::Environment* const batch_execution_environment =
  ::testing::AddGlobalTestEnvironment(new BatchWorkerEnvironment);

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
template <typename Batch>
concept permits_batch_log = requires(const Batch& batch) { batch.log(); };
template <typename Batch>
concept permits_batch_eigenvalues = requires(const Batch& batch) { batch.eigenvalues(); };
template <typename Batch>
concept permits_batch_trace = requires(const Batch& batch) { batch.trace(); };
template <typename Batch>
concept permits_batch_determinant = requires(const Batch& batch) { batch.determinant(); };
template <typename Batch>
concept permits_batch_diagonal = requires(const Batch& batch) { batch.diagonal(); };
template <typename Batch>
concept permits_batch_eigenvectors = requires(const Batch& batch) { batch.eigenvectors(); };
template <typename Batch>
concept permits_batch_exp = requires(const Batch& batch) { batch.exp(); };
template <typename Batch>
concept permits_batch_sqrt = requires(const Batch& batch) { batch.sqrt(); };
template <typename Batch>
concept permits_batch_inverse_sqrt = requires(const Batch& batch) { batch.inv_sqrt(); };
template <typename Batch>
concept permits_batch_norm = requires(const Batch& batch) { batch.norm(); };

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

    MatrixBatch<SymmetricMatrix<double, 3>> symmetric(2);
    // symmetric rows retain only the six native packed lower coefficients
    EXPECT_EQ(symmetric.coefficient_stride(), std::size_t {6});
    // the second symmetric row follows the first packed row directly
    EXPECT_EQ(symmetric.coefficients().data() + 6, symmetric[1].data());

    MatrixBatch<SymmetricMatrix<double, Dynamic>> dynamic_symmetric(2, 3, 3);
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

    MatrixBatch<SPDMatrix<double, Dynamic, Cache::Log>> dynamic_spd(2, 2, 2);
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

    // packed SPD rows retain their three native coefficients
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

// copies, moves and swaps preserve coefficient values and their independent aggregate cache bindings
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

    using dynamic_cached_point = SPDMatrix<double, Dynamic, full_cache>;
    MatrixBatch<dynamic_cached_point> dynamic_source(2, 2, 2);
    dynamic_source[0].assign(Matrix<double, 2, 2>({4, 0, 0, 9}));
    dynamic_source[1].assign(Matrix<double, 2, 2>({16, 0, 0, 25}));
    const MatrixBatch<dynamic_cached_point> dynamic_copy(dynamic_source);
    // dynamic copies bind their first cache slot to an independent aggregate buffer
    EXPECT_NE(dynamic_copy.cache_pointers()[0]->data(), dynamic_source.cache_pointers()[0]->data());
    // dynamic copies retain the first coefficient row with its matching cache slot
    EXPECT_EQ(dynamic_copy[0].cache().data(), dynamic_copy.cache_pointers()[0]->data());
    // the copied first cache logarithm matches the retained coefficient nine
    EXPECT_DOUBLE_EQ(dynamic_copy[0].cache().template matrix<Cache::Log>()(1, 1), std::log(9));
    // dynamic copies retain the second coefficient row with its matching cache slot
    EXPECT_EQ(dynamic_copy[1].cache().data(), dynamic_copy.cache_pointers()[1]->data());
    // dynamic copies preserve their two-row runtime shape
    EXPECT_EQ(dynamic_copy[1].rows(), 2);

    MatrixBatch<dynamic_cached_point> dynamic_moved(std::move(dynamic_source));
    // dynamic moves preserve the first coefficient and cache association
    EXPECT_EQ(dynamic_moved[0].cache().data(), dynamic_moved.cache_pointers()[0]->data());
    // the moved first cache logarithm remains associated with coefficient nine
    EXPECT_DOUBLE_EQ(dynamic_moved[0].cache().template matrix<Cache::Log>()(1, 1), std::log(9));
    // dynamic moves preserve the second coefficient and cache association
    EXPECT_EQ(dynamic_moved[1].cache().data(), dynamic_moved.cache_pointers()[1]->data());
    MatrixBatch<dynamic_cached_point> dynamic_other(1, 3, 3);
    dynamic_moved.swap(dynamic_other);
    // swapping dynamic batches transfers the three-row identity shape metadata
    EXPECT_EQ(dynamic_moved[0].rows(), 3);
    // swapped three-row identities keep the known zero logarithm in their transferred cache slot
    EXPECT_DOUBLE_EQ(dynamic_moved[0].cache().template matrix<Cache::Log>()(2, 2), 0);
    // swapped dynamic batches rebind the transferred three-row cache slot to its new owner
    EXPECT_EQ(dynamic_moved[0].cache().data(), dynamic_moved.cache_pointers()[0]->data());
    // swapped-from dynamic batches retain the original two-row first coefficient row
    EXPECT_EQ(dynamic_other[0](0, 0), 4);
    // swapped-from dynamic batches retain the logarithm associated with coefficient nine
    EXPECT_DOUBLE_EQ(dynamic_other[0].cache().template matrix<Cache::Log>()(1, 1), std::log(9));
    // swapped-from dynamic batches rebind the original first cache slot to their new owner
    EXPECT_EQ(dynamic_other[0].cache().data(), dynamic_other.cache_pointers()[0]->data());
}

// cache-free batches have no cache-bearing layout state and ordinary entries remain writable
TEST(MatrixBatch, UsesNoCachePolicyForOrdinaryMatrices) {
    using dense_batch = MatrixBatch<Matrix<double, 2, 2>>;
    // ordinary batch elements select the no-cache policy
    static_assert(dense_batch::CachePolicy::Flags == Cache::None::Flags);
    // no-cache SPD batches match coefficient-equivalent symmetric batches on this toolchain
    static_assert(sizeof(MatrixBatch<SPDMatrix<double, 2>>) == sizeof(MatrixBatch<SymmetricMatrix<double, 2>>));

    dense_batch values(1);
    values.coefficients()[0] = 7;
    // ordinary coefficient spans permit controlled batch-level writes
    EXPECT_EQ(values[0](0, 0), 7);
}

// deferred maps preserve result structure and laziness, including known and unknown empty-result shapes
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

    const MatrixBatch<Matrix<double, 2, 2>> empty;
    int empty_calls = 0;
    const auto empty_map = empty.map([&empty_calls](const auto&) {
        ++empty_calls;
        return Matrix<double, 1, 1>(0);
    });
    MatrixBatch<Matrix<double, 1, 1>> empty_result(empty_map);
    // an empty fixed-shape map materializes an empty result without calling its callable
    EXPECT_TRUE(empty_result.empty());
    // fixed scalar map results preserve their one-row shape when no source value exists
    EXPECT_EQ(empty_result.rows(), 1);
    // fixed scalar map results preserve their one-column shape when no source value exists
    EXPECT_EQ(empty_result.cols(), 1);
    // empty materialization does not evaluate the deferred callable
    EXPECT_EQ(empty_calls, 0);

    const MatrixBatch<Matrix<double, Dynamic, Dynamic>> dynamic_empty(0, 2, 2);
    const auto dynamic_map = dynamic_empty.map([](const auto&) { return Matrix<double, Dynamic, Dynamic>(1, 1); });
    // empty dynamic map materialization rejects the result shape that no source value can establish
    EXPECT_THROW((MatrixBatch<Matrix<double, Dynamic, Dynamic>>(dynamic_map)), std::invalid_argument);
}

// reductions preserve index order and empty initializers while fused maps observe current values without memoization
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

    int calls = 0;
    const auto fused = values.map([&calls](const auto& value) {
        ++calls;
        return value(0, 0);
    });
    values.coefficients()[1] = 20;
    const auto observed = fused.redux(0, [](int accumulator, const auto& value) { return accumulator + value(0, 0); });
    // fused redux invokes the mapped callable once for each source element
    EXPECT_EQ(calls, 3);
    // fused redux observes source coefficients changed after map creation and before evaluation
    EXPECT_EQ(observed, 24);
    // a later indexed map evaluation invokes the callable once more without memoization
    EXPECT_EQ(fused[1](0, 0), 20);
    // the indexed reevaluation increments the same callable visit counter
    EXPECT_EQ(calls, 4);
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
    const cached_point before(points[0]);
    const SymmetricMatrix<double, 2> cache_before(points[0].cache().template matrix<Cache::Log>());
    const Matrix<double, 2, 2> indefinite({-1, 0, 0, 1});

    // checked element assignment rejects a nonpositive-definite candidate
    EXPECT_THROW(points[0].assign(indefinite), std::domain_error);
    // failed assignment retains the previous verified coefficients
    expect_matrix_eq(points[0], before);
    // failed assignment retains the previous prepared logarithm
    expect_matrix_eq(points[0].cache().template matrix<Cache::Log>(), cache_before);
}

// assignment between batch SPD views transfers a verified value without rebinding destination storage
TEST(MatrixBatch, AssignsAliasedSPDViewsByValue) {
    MatrixBatch<cached_point> points(2);
    points[0].assign(Matrix<double, 2, 2>({4, 0, 0, 9}));
    points[1].assign(Matrix<double, 2, 2>({16, 0, 0, 25}));
    const auto* const destination_coefficients = points[0].data();
    const auto* const destination_cache = points[0].cache().data();
    const cached_point source_value(points[1]);

    points[0] = points[1];
    // view-to-view assignment retains the destination coefficient row binding
    EXPECT_EQ(points[0].data(), destination_coefficients);
    // view-to-view assignment retains the destination cache slot binding
    EXPECT_EQ(points[0].cache().data(), destination_cache);
    // view-to-view assignment copies all verified source coefficients by value
    expect_matrix_eq(points[0], source_value);
    // view-to-view assignment refreshes logarithms in the destination cache slot
    EXPECT_DOUBLE_EQ(points[0].cache().template matrix<Cache::Log>()(1, 1), std::log(25));
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
    MatrixBatch<SPDMatrix<double, 2, full_cache>> source(3);
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

/// @brief checks single-order SPD storage and independently owned logarithms for either cache policy
template <int Order, typename Policy> void check_square_order_log_batch() {
    using Point = SPDMatrix<double, Order, Policy>;
    using Sym = SymmetricMatrix<double, Order>;
    // both generic matrix extents follow the sole template order
    static_assert(Point::Rows == Order && Point::Cols == Order && Sym::Rows == Order && Sym::Cols == Order);
    MatrixBatch<Point> points(2, 2, 2);
    points[0] = Point(Vector<double, 3> {2., 0., 3.});
    points[1] = Point(Vector<double, 3> {4., 0., 5.});
    const MatrixBatch<Sym> logs(points.map([](const auto& point) { return matrix_log(point); }));
    // mapping the logarithm preserves the number of SPD factors
    ASSERT_EQ(logs.size(), 2);
    for (std::size_t i = 0; i < logs.size(); ++i) {
        // diagonal eigenvalues provide an independent scalar logarithm oracle
        EXPECT_NEAR(logs[i](0, 0), std::log(2. + 2. * i), 1e-14);
        // the second diagonal coefficient follows its own eigenvalue
        EXPECT_NEAR(logs[i](1, 1), std::log(3. + 2. * i), 1e-14);
        // a diagonal SPD input has no off-diagonal logarithm coefficient
        EXPECT_DOUBLE_EQ(logs[i](1, 0), 0.);
    }
    points[0] = Point::Identity(2);
    // replacing the input leaves the materialized symmetric logarithm independent
    EXPECT_NEAR(logs[0](0, 0), std::log(2.), 1e-14);
}

// fixed and dynamic orders support the same batch logarithm with and without retained caches
TEST(MatrixBatch, SingleOrderSPDLogarithms) {
    // fixed owners exercise the uncached logarithm and the cached read path
    check_square_order_log_batch<2, Cache::None>();
    check_square_order_log_batch<2, Cache::Log>();
    // runtime orders preserve shape and ownership across the same two paths
    check_square_order_log_batch<Dynamic, Cache::None>();
    check_square_order_log_batch<Dynamic, Cache::Log>();
}

// direct transforms preserve order and owning result types while parallel evaluation matches sequential kernels
TEST(MatrixBatch, DirectLogarithmsAndEigenvaluesMatchSequentialAndParallelResults) {
    using Point = SPDMatrix<double, 2, Cache::Log>;
    using Sym = SymmetricMatrix<double, 2, Cache::Spectral>;
    MatrixBatch<Point> points(128);
    for (std::size_t i = 0; i < points.size(); ++i)
        points[i] = Point(Vector<double, 3> {2. + i * .01, .2, 3. + i * .02});
    const auto logs = points.log<Cache::Spectral>();
    const auto sequential_logs = points.log<Cache::Spectral>(execution_seq);
    const auto parallel_logs = points.log<Cache::Spectral>(execution_par);
    const auto values = logs.eigenvalues();
    const auto parallel_values = parallel_logs.eigenvalues(execution_par);
    const auto spd_values = points.eigenvalues(execution_par);
    // the logarithm uses the explicitly requested symmetric cache policy rather than the input SPD policy
    static_assert(std::same_as<std::remove_cvref_t<decltype(logs)>, MatrixBatch<Sym>>);
    // eigenvalues materialize as native owning two-entry column vectors
    static_assert(std::same_as<std::remove_cvref_t<decltype(values)>, MatrixBatch<Vector<double, 2>>>);
    // direct logarithms retain one result for each original SPD point
    ASSERT_EQ(logs.size(), points.size());
    // direct eigenvalues retain the complete source ordering and cardinality
    ASSERT_EQ(values.size(), points.size());
    for (std::size_t i = 0; i < points.size(); ++i) {
        // the default execution policy produces the same coefficients as explicit sequential evaluation
        expect_matrix_eq(logs[i], sequential_logs[i]);
        // independent worker slots preserve exactly the sequential logarithm coefficients and ordering
        expect_matrix_eq(parallel_logs[i], logs[i]);
        // cached symmetric spectra are independent of whether slot evaluation runs on workers
        expect_matrix_eq(parallel_values[i], values[i]);
        // the logarithmic spectrum sums to the log determinant of the supplied two-by-two SPD coefficients
        EXPECT_NEAR(values[i].sum(), std::log((2. + i * .01) * (3. + i * .02) - .04), 2e-12);
        // direct SPD spectra retain the independently known diagonal trace at every index
        EXPECT_NEAR(spd_values[i].sum(), 5. + i * .03, 2e-12);
    }
    const auto plain_logs = points.log();
    // omitting the output policy preserves the ordinary cache-free symmetric owner type
    static_assert(std::same_as<std::remove_cvref_t<decltype(plain_logs)>, MatrixBatch<SymmetricMatrix<double, 2>>>);
    // allocating a different output policy does not change the principal logarithm coefficients
    expect_matrix_eq(plain_logs[37], logs[37]);
}

// original symmetric cache slots are prepared in place and retained raw aliases cannot leave batch spectra stale
TEST(MatrixBatch, DirectEigenvaluesRefreshOriginalSlotsAfterTrackedAndRawWrites) {
    using Sym = SymmetricMatrix<double, 2, Cache::Spectral>;
    MatrixBatch<Sym> matrices(64);
    for (std::size_t i = 0; i < matrices.size(); ++i) matrices[i] = Sym(Vector<double, 3> {2., .5, 3.});
    matrices[17](0, 0) = 4.;
    const auto slots = matrices.cache_pointers();
    // a tracked coefficient replacement invalidates the actual batch slot before spectral evaluation
    EXPECT_FALSE(slots[17]->valid());
    const auto before = matrices.eigenvalues(execution_par);
    // direct batch evaluation prepares the original slot instead of only a detached temporary owner
    EXPECT_TRUE(slots[17]->valid());
    // the prepared result reflects the tracked diagonal replacement through its exact trace
    EXPECT_NEAR(before[17].sum(), 7., 1e-13);
    auto* writable = matrices[17].data();
    writable[0] = 7.;
    const auto changed = matrices.eigenvalues(execution_seq);
    // sequential evaluation refreshes factors after the first write through the retained packed pointer
    EXPECT_NEAR(changed[17].sum(), 10., 1e-13);
    writable[0] = 11.;
    const auto changed_again = matrices.eigenvalues(execution_par);
    // parallel evaluation refreshes again even though no new mutable data request preceded the second write
    EXPECT_NEAR(changed_again[17].sum(), 14., 1e-13);
    // a prior eager result owns its spectrum independently of later writes to the source slot
    EXPECT_NEAR(changed[17].sum(), 10., 1e-13);
    // exposure of one slot leaves the independently cached neighboring matrix and its spectrum unchanged
    EXPECT_NEAR(changed_again[16].sum(), 5., 1e-13);
}

// direct transforms retain runtime dimensions and scalar precision even when a batch has no entries
TEST(MatrixBatch, DirectTransformsPreserveDynamicFloatAndEmptyShapes) {
    using Point = SPDMatrix<float, Dynamic, Cache::Log>;
    using Sym = SymmetricMatrix<float, Dynamic, Cache::Spectral>;
    MatrixBatch<Point> points(3, 3, 3);
    points[1] = Point(Matrix<float, 3, 3>({4, 0, 0, 0, 9, 0, 0, 0, 16}));
    const auto logs = points.log<Cache::Spectral>(execution_par);
    const auto values = logs.eigenvalues(execution_par);
    // dynamic float logarithms keep their requested symmetric spectral-cache policy
    static_assert(std::same_as<std::remove_cvref_t<decltype(logs)>, MatrixBatch<Sym>>);
    // dynamic eigenvalues keep float coefficients and a compile-time column-vector shape
    static_assert(std::same_as<std::remove_cvref_t<decltype(values)>, MatrixBatch<Vector<float, Dynamic>>>);
    // the runtime SPD order becomes the eigenvalue vector length
    EXPECT_EQ(values.rows(), 3);
    // eigenvalues have one column independently of the square input's column count
    EXPECT_EQ(values.cols(), 1);
    // the diagonal example supplies an independent float trace of the principal logarithm
    EXPECT_NEAR(values[1].sum(), std::log(4.f * 9.f * 16.f), 2e-6f);
    const MatrixBatch<Point> empty(0, 3, 3);
    const auto empty_logs = empty.log<Cache::Spectral>(execution_par);
    const auto empty_values = empty_logs.eigenvalues(execution_par);
    const auto empty_spd_values = empty.eigenvalues(execution_seq);
    // an empty dynamic logarithm needs no source entry to establish its known runtime shape
    EXPECT_TRUE(empty_logs.empty());
    // empty logarithms retain the supplied row order
    EXPECT_EQ(empty_logs.rows(), 3);
    // empty logarithms retain the supplied square column extent
    EXPECT_EQ(empty_logs.cols(), 3);
    // empty symmetric spectra remain empty without invoking an eigensolver
    EXPECT_TRUE(empty_values.empty());
    // an empty spectrum still retains the known eigenvalue vector length
    EXPECT_EQ(empty_values.rows(), 3);
    // an empty spectrum still advertises its single output column
    EXPECT_EQ(empty_values.cols(), 1);
    // the direct SPD empty path uses the same known vector length as the symmetric path
    EXPECT_EQ(empty_spd_values.rows(), 3);
    // the direct SPD empty path does not preserve the input's square column extent
    EXPECT_EQ(empty_spd_values.cols(), 1);
}

// eager transforms own temporary results while lazy maps and repeated selections retain their existing API
TEST(MatrixBatch, DirectTransformsOwnTemporariesAndStayOnOwningBatches) {
    using Points = MatrixBatch<SPDMatrix<double, 2>>;
    const auto logs = Points(64).log<Cache::Spectral>(execution_par);
    const auto values = Points(64).log<Cache::Spectral>(execution_par).eigenvalues(execution_par);
    // destroying the source temporary leaves all logarithm entries available
    ASSERT_EQ(logs.size(), 64);
    // destroying the intermediate temporary leaves all eigenvalue entries available
    ASSERT_EQ(values.size(), 64);
    // the retained temporary-derived identity logarithm has the independently known zero diagonal
    EXPECT_DOUBLE_EQ(logs[63](1, 1), 0.);
    // the retained zero logarithm has two exactly zero eigenvalues
    EXPECT_DOUBLE_EQ(values[63].sum(), 0.);
    using Selection = decltype(std::declval<const Points&>().select(std::array<int, 2> {0, 0}));
    // borrowed repeated-index selections do not acquire an eager parallel logarithm entry point
    static_assert(!permits_batch_log<Selection>);
    // borrowed repeated-index selections do not acquire an eager parallel eigensolver entry point
    static_assert(!permits_batch_eigenvalues<Selection>);
    // deferred map nodes retain their lazy callable interface without a direct logarithm operation
    static_assert(!permits_batch_log<scalar_map>);
    // deferred map nodes do not expose eager spectral evaluation of potentially repeated borrowed slots
    static_assert(!permits_batch_eigenvalues<scalar_map>);
    // symmetric owners cannot request an SPD logarithm without a positive-definite input contract
    static_assert(!permits_batch_log<MatrixBatch<SymmetricMatrix<double, 2>>>);
    // generic dense owners cannot request the SPD principal logarithm without an SPD contract
    static_assert(!permits_batch_log<scalar_batch>);
    // generic dense owners cannot request the native symmetric eigensolver without a symmetric contract
    static_assert(!permits_batch_eigenvalues<scalar_batch>);
}

// worker failures reach the caller after batch work completes and do not poison subsequent parallel transforms
TEST(MatrixBatch, ParallelEigenvalueFailurePropagatesAndRuntimeRemainsUsable) {
    using Sym = SymmetricMatrix<double, 2, Cache::Spectral>;
    MatrixBatch<Sym> matrices(128);
    for (std::size_t i = 0; i < matrices.size(); ++i) matrices[i] = Sym(Vector<double, 3> {2., .5, 3.});
    auto* writable = matrices[87].data();
    writable[0] = std::numeric_limits<double>::quiet_NaN();
    // invalid finite-coefficient validation escapes worker evaluation as the original public exception
    EXPECT_THROW(matrices.eigenvalues(execution_par), std::invalid_argument);
    writable[0] = 4.;
    const auto recovered = matrices.eigenvalues(execution_par);
    // a later parallel transform still completes every requested output after the failed operation has joined
    ASSERT_EQ(recovered.size(), matrices.size());
    // repairing the invalid coefficient restores the independent diagonal-trace oracle in the failed slot
    EXPECT_NEAR(recovered[87].sum(), 7., 1e-13);
    // an unrelated slot also remains usable after partial work from the failed transform is discarded
    EXPECT_NEAR(recovered[86].sum(), 5., 1e-13);
}

// scalar transforms retain index order and support indefinite and singular symmetric matrices under either policy
TEST(MatrixBatch, DirectScalarTransformsMatchAnalyticSymmetricValues) {
    using Sym = SymmetricMatrix<double, 2, Cache::Spectral>;
    MatrixBatch<Sym> matrices(4);
    matrices[0] = Sym(Vector<double, 3> {2., .5, -1.});
    matrices[1] = Sym(Vector<double, 3> {1., 2., 4.});
    matrices[2] = Sym(Vector<double, 3> {0., 1., 0.});
    matrices[3] = Sym(Vector<double, 3> {3., -2., 5.});
    const auto traces = matrices.trace();
    const auto parallel_traces = matrices.trace(execution_par);
    // summing diagonals under either policy leaves the source's initially invalid spectral cache unprepared
    EXPECT_FALSE(matrices.cache_pointers()[0]->valid());
    const auto determinants = matrices.determinant(execution_seq);
    const auto parallel_determinants = matrices.determinant(execution_par);
    const std::array<double, 4> expected_traces {1., 5., 0., 8.};
    const std::array<double, 4> expected_determinants {-2.25, 0., -1., 11.};
    // trace results follow the existing scalar-map convention of owning one-by-one native matrices
    static_assert(std::same_as<std::remove_cvref_t<decltype(traces)>, MatrixBatch<Matrix<double, 1, 1>>>);
    // determinant results use the same scalar owner type even when values are negative or zero
    static_assert(std::same_as<std::remove_cvref_t<decltype(determinants)>, MatrixBatch<Matrix<double, 1, 1>>>);
    // one trace is retained for each input matrix without merging results through a reduction
    ASSERT_EQ(traces.size(), expected_traces.size());
    // one determinant is retained for every corresponding input position
    ASSERT_EQ(determinants.size(), expected_determinants.size());
    for (std::size_t i = 0; i < matrices.size(); ++i) {
        // adding the two supplied diagonal coefficients gives each independent trace oracle
        EXPECT_DOUBLE_EQ(traces[i](0, 0), expected_traces[i]);
        // the two-by-two formula a*d-b*b covers negative, singular and pivoted determinants
        EXPECT_NEAR(determinants[i](0, 0), expected_determinants[i], 1e-13);
        // worker evaluation retains the same trace value and source position as sequential evaluation
        expect_matrix_eq(parallel_traces[i], traces[i]);
        // worker evaluation retains the same signed determinant and source position as sequential evaluation
        expect_matrix_eq(parallel_determinants[i], determinants[i]);
    }
    matrices[0](0, 0) = 10.;
    // eager trace outputs keep their original value after a later source coefficient write
    EXPECT_DOUBLE_EQ(traces[0](0, 0), 1.);
    // eager determinant outputs own their original value after the same source mutation
    EXPECT_NEAR(determinants[0](0, 0), -2.25, 1e-13);
}

// scalar transforms preserve native scalar types and delegate to dense, diagonal and cached SPD element operations
TEST(MatrixBatch, DirectScalarTransformsSupportNativeMatrixFamilies) {
    MatrixBatch<Matrix<float, Dynamic, Dynamic>> dense(1, 2, 2);
    dense[0] = Matrix<float, 2, 2>({1, 2, 3, 4});
    const auto dense_traces = dense.trace(execution_seq);
    const auto dense_determinants = dense.determinant();
    // a dynamic float source produces fixed one-by-one float results without promoting its scalar type
    static_assert(std::same_as<std::remove_cvref_t<decltype(dense_traces)>, MatrixBatch<Matrix<float, 1, 1>>>);
    // dense trace ignores both unequal off-diagonal coefficients
    EXPECT_FLOAT_EQ(dense_traces[0](0, 0), 5.f);
    // the nonsymmetric two-by-two determinant follows the independent formula 1*4-2*3
    EXPECT_NEAR(dense_determinants[0](0, 0), -2.f, 1e-6f);
    MatrixBatch<DiagonalMatrix<int, 2>> diagonal(1);
    diagonal.coefficients()[0] = 2;
    diagonal.coefficients()[1] = -3;
    const auto diagonal_traces = diagonal.trace(execution_par);
    const auto diagonal_determinants = diagonal.determinant(execution_par);
    // diagonal integer trace remains an integer scalar result rather than requiring a floating eigensolver
    static_assert(std::same_as<std::remove_cvref_t<decltype(diagonal_traces)>, MatrixBatch<Matrix<int, 1, 1>>>);
    // the diagonal trace adds the signed stored entries
    EXPECT_EQ(diagonal_traces[0](0, 0), -1);
    // the diagonal determinant multiplies the signed stored entries without dense factorization
    EXPECT_EQ(diagonal_determinants[0](0, 0), -6);
    using Point = SPDMatrix<double, 2, Cache::Spectral>;
    MatrixBatch<Point> points(1);
    points[0] = Point(Vector<double, 3> {3., 1., 2.});
    const auto spd_determinants = points.determinant(execution_par);
    // the cached SPD spectrum has product 3*2-1*1 for the supplied non-diagonal point
    EXPECT_NEAR(spd_determinants[0](0, 0), 5., 1e-13);
    const auto temporary_determinants = MatrixBatch<Point>(2).determinant(execution_par);
    // destroying a temporary identity batch leaves independently owned unit determinants
    EXPECT_DOUBLE_EQ(temporary_determinants[1](0, 0), 1.);
    const Matrix<int, 2, 2> integer({2, 7, -3, 5});
    // the new standalone matrix trace retains the exact sum of the two integer diagonal coefficients
    EXPECT_EQ(integer.trace(), 7);
    // lazy dense expressions inherit trace and evaluate only their diagonal entries
    EXPECT_EQ((integer + integer).trace(), 14);
}

// empty square batches retain scalar output shape while rectangular matrices fail before element evaluation
TEST(MatrixBatch, DirectScalarTransformsValidateSquareShapesAndPreserveEmptyResults) {
    using DynamicMatrix = Matrix<double, Dynamic, Dynamic>;
    const MatrixBatch<DynamicMatrix> empty(0, 3, 3);
    const auto traces = empty.trace(execution_par);
    const auto determinants = empty.determinant();
    // empty square inputs produce no trace entries without needing a first matrix for shape inference
    EXPECT_TRUE(traces.empty());
    // empty square inputs produce no determinant entries without invoking a factorization
    EXPECT_TRUE(determinants.empty());
    // empty trace entries still advertise the scalar-map row extent
    EXPECT_EQ(traces.rows(), 1);
    // empty trace entries still advertise the scalar-map column extent
    EXPECT_EQ(traces.cols(), 1);
    // empty determinant entries retain the same scalar row extent
    EXPECT_EQ(determinants.rows(), 1);
    // empty determinant entries retain the same scalar column extent
    EXPECT_EQ(determinants.cols(), 1);
    for (std::size_t count : {std::size_t {0}, std::size_t {1}}) {
        const MatrixBatch<DynamicMatrix> rectangular(count, 2, 3);
        // sequential trace validates square input shape even when no entries would be visited
        EXPECT_THROW(rectangular.trace(execution_seq), std::invalid_argument);
        // parallel trace performs the same square-shape validation before scheduling any entries
        EXPECT_THROW(rectangular.trace(execution_par), std::invalid_argument);
        // sequential determinant rejects nonsquare shape independently of the number of entries
        EXPECT_THROW(rectangular.determinant(execution_seq), std::invalid_argument);
        // parallel determinant cannot bypass nonsquare validation for an empty batch
        EXPECT_THROW(rectangular.determinant(execution_par), std::invalid_argument);
    }
    using RectangularBatch = MatrixBatch<Matrix<double, 2, 3>>;
    // a statically rectangular element type has no viable batch trace operation
    static_assert(!permits_batch_trace<RectangularBatch>);
    // a statically rectangular element type has no viable batch determinant operation
    static_assert(!permits_batch_determinant<RectangularBatch>);
    const DynamicMatrix empty_matrix(0, 0);
    // the diagonal sum of a standalone zero-order square matrix is the additive identity
    EXPECT_DOUBLE_EQ(empty_matrix.trace(), 0.);
    const DynamicMatrix rectangular_matrix(2, 3);
    // the standalone trace rejects the same runtime rectangular shape as the batch entry point
    EXPECT_THROW(rectangular_matrix.trace(), std::invalid_argument);
}

// logical Frobenius norms count both symmetric halves and diagonal extraction returns independent column owners
TEST(MatrixBatch, DirectNormsAndDiagonalsUseLogicalCoefficients) {
    using Sym = SymmetricMatrix<double, 2, Cache::Spectral>;
    MatrixBatch<Sym> matrices(1);
    matrices[0] = Sym(Vector<double, 3> {2., 3., -4.});
    const auto norms = matrices.norm();
    const auto parallel_norms = matrices.norm(execution_par);
    const auto squared = matrices.squared_norm(execution_par);
    const auto diagonals = matrices.diagonal(execution_par);
    // the logical square sum includes both off-diagonal threes in addition to the stored diagonal squares
    EXPECT_DOUBLE_EQ(squared[0](0, 0), 38.);
    // the Frobenius norm is the square root of the independently enumerated logical coefficient sum
    EXPECT_NEAR(norms[0](0, 0), std::sqrt(38.), 1e-14);
    // parallel norm evaluation produces the same stable scalar result as sequential evaluation
    expect_matrix_eq(parallel_norms[0], norms[0]);
    // extracting norms and diagonals does not prepare an unused spectral decomposition
    EXPECT_FALSE(matrices.cache_pointers()[0]->valid());
    // diagonal extraction owns a fixed-length native column vector with the original scalar type
    static_assert(std::same_as<std::remove_cvref_t<decltype(diagonals)>, MatrixBatch<Vector<double, 2>>>);
    matrices[0](0, 0) = 11.;
    // the extracted diagonal keeps its previous first value after the source changes
    EXPECT_DOUBLE_EQ(diagonals[0](0, 0), 2.);
    // the extracted diagonal excludes packed off-diagonal storage and retains the signed second diagonal
    EXPECT_DOUBLE_EQ(diagonals[0](1, 0), -4.);
    MatrixBatch<Matrix<float, 1, 2>> rectangular(1);
    rectangular[0] = Matrix<float, 1, 2>({3.f, 4.f});
    const auto rectangular_norms = rectangular.norm(execution_par);
    // nonsquare dense matrices still support the independently known three-four-five Frobenius norm
    EXPECT_FLOAT_EQ(rectangular_norms[0](0, 0), 5.f);
    MatrixBatch<SymmetricMatrix<int, 2>> integers(1);
    integers[0] = SymmetricMatrix<int, 2>(Vector<int, 3> {1, 2, 3});
    const auto integer_squared = integers.squared_norm();
    // squared norms preserve integral scalar results without requiring a floating square root
    static_assert(std::same_as<std::remove_cvref_t<decltype(integer_squared)>, MatrixBatch<Matrix<int, 1, 1>>>);
    // integer symmetric storage also counts both reflected off-diagonal coefficients
    EXPECT_EQ(integer_squared[0](0, 0), 18);
    // the norm follows the native floating-point-only contract while squared_norm supports integers
    static_assert(!permits_batch_norm<decltype(integers)>);
}

// spectral transforms reuse original slots and preserve eigenspaces and principal matrix-function identities
TEST(MatrixBatch, DirectSpectralTransformsReuseSlotsAndMatchUncachedFunctions) {
    using Sym = SymmetricMatrix<double, 2, Cache::Spectral>;
    using Point = SPDMatrix<double, 2, full_cache>;
    MatrixBatch<Sym> matrices(4);
    for (std::size_t i = 0; i < matrices.size(); ++i)
        matrices[i] = Sym(Vector<double, 3> {.2 + i * .1, i == 0 ? 0. : .05, .2 + i * .2});
    const auto vectors = matrices.eigenvectors();
    // eigenvector extraction prepares the original previously invalid slot rather than a detached owner
    EXPECT_TRUE(matrices.cache_pointers()[1]->valid());
    const auto parallel_vectors = matrices.eigenvectors(execution_par);
    const auto values = matrices.eigenvalues();
    const Matrix<double, 2, 2> identity({1, 0, 0, 1});
    for (std::size_t i = 0; i < matrices.size(); ++i) {
        const auto basis = vectors[i], parallel_basis = parallel_vectors[i];
        const auto spectrum = values[i];
        // eigenvector columns satisfy paired eigenvalue equations, including the first entry's repeated spectrum
        EXPECT_LT((matrices[i] * basis - basis * spectrum.as_diagonal()).norm(), 1e-12);
        // the returned basis is orthonormal without imposing arbitrary eigenvector signs
        EXPECT_LT((basis.transpose() * basis - identity).norm(), 1e-12);
        // parallel columns satisfy the same eigenvalue equations without requiring a particular repeated basis
        EXPECT_LT((matrices[i] * parallel_basis - parallel_basis * spectrum.as_diagonal()).norm(), 1e-12);
    }
    matrices[2](0, 0) += .1;
    const auto exponentials = matrices.exp<full_cache>(execution_par);
    // exponentiation refreshes the original slot invalidated after its earlier eigenvector extraction
    EXPECT_TRUE(matrices.cache_pointers()[2]->valid());
    // the chosen exponential output cache becomes the exact native SPD element policy
    static_assert(std::same_as<std::remove_cvref_t<decltype(exponentials)>, MatrixBatch<Point>>);
    const auto roots = exponentials.sqrt<Cache::Log>(execution_par);
    const auto inverse_roots = exponentials.inv_sqrt(execution_seq);
    const auto spd_vectors = exponentials.eigenvectors(execution_par);
    const auto spd_values = exponentials.eigenvalues();
    // principal roots retain their separately selected output policy
    static_assert(std::same_as<std::remove_cvref_t<decltype(roots)>, MatrixBatch<SPDMatrix<double, 2, Cache::Log>>>);
    // an omitted inverse-root output policy produces ordinary uncached SPD owners
    static_assert(std::same_as<std::remove_cvref_t<decltype(inverse_roots)>, MatrixBatch<SPDMatrix<double, 2>>>);
    for (std::size_t i = 0; i < matrices.size(); ++i) {
        const auto expected_exp = matrix_exp(SymmetricMatrix<double, 2>(matrices[i]));
        const SPDMatrix<double, 2> uncached(exponentials[i]);
        const auto expected_root = matrix_sqrt(uncached);
        const auto expected_inverse_root = matrix_inv_sqrt(uncached);
        const auto recovered_log = matrix_log(exponentials[i]);
        const auto spectrum = spd_values[i];
        // verified SPD batches expose eigenvector columns paired with their own cached eigenvalues
        EXPECT_LT((exponentials[i] * spd_vectors[i] - spd_vectors[i] * spectrum.as_diagonal()).norm(), 1e-12);
        // the cached exponential matches a decomposition of independent uncached source coefficients
        EXPECT_LT((exponentials[i] - expected_exp).norm(), 1e-12);
        // prepared SPD square-root factors reproduce the independently decomposed principal root
        EXPECT_LT((roots[i] - expected_root).norm(), 1e-12);
        // prepared inverse-root factors reproduce the independent uncached inverse square root
        EXPECT_LT((inverse_roots[i] - expected_inverse_root).norm(), 1e-12);
        // the checked exponential and its principal logarithm round-trip the original finite symmetric value
        EXPECT_LT((recovered_log - matrices[i]).norm(), 1e-12);
    }
}

/// @brief checks owned batch cardinality and runtime matrix extents against a supplied shape
template <typename Batch> void expect_batch_shape(const Batch& batch, std::size_t count, int rows, int cols) {
    // materialization preserves the explicitly expected number of independent result entries
    EXPECT_EQ(batch.size(), count);
    // the result retains the row extent dictated by its operation rather than its first entry
    EXPECT_EQ(batch.rows(), rows);
    // the result retains the corresponding scalar, vector or matrix column extent
    EXPECT_EQ(batch.cols(), cols);
}

// dynamic float transforms retain their scalar and empty-result extents without inventing input entries
TEST(MatrixBatch, DirectOperationsPreserveDynamicFloatShapesAndConstraints) {
    using Sym = SymmetricMatrix<float, Dynamic, Cache::Spectral>;
    const MatrixBatch<Sym> empty(0, 3, 3);
    // empty norms have scalar output shape
    expect_batch_shape(empty.norm(execution_par), 0, 1, 1);
    // empty squared norms retain the same scalar output shape
    expect_batch_shape(empty.squared_norm(), 0, 1, 1);
    // empty diagonal extraction retains the known runtime vector length
    expect_batch_shape(empty.diagonal(execution_par), 0, 3, 1);
    // empty eigenvector extraction retains the full runtime square basis shape
    expect_batch_shape(empty.eigenvectors(execution_par), 0, 3, 3);
    const auto empty_exp = empty.exp<Cache::Sqrt>(execution_par);
    // empty exponentials retain float precision, dynamic order and the requested output cache
    static_assert(
      std::same_as<std::remove_cvref_t<decltype(empty_exp)>, MatrixBatch<SPDMatrix<float, Dynamic, Cache::Sqrt>>>);
    // empty exponential results retain the runtime matrix order
    expect_batch_shape(empty_exp, 0, 3, 3);
    // empty principal roots need no element from which to infer output dimensions
    expect_batch_shape(empty_exp.sqrt(execution_par), 0, 3, 3);
    // empty inverse roots also preserve the known runtime order
    expect_batch_shape(empty_exp.inv_sqrt(), 0, 3, 3);
    MatrixBatch<Sym> matrices(1, 3, 3);
    matrices[0] = Sym(Matrix<float, 3, 3>({0, 0, 0, 0, 1, 0, 0, 0, 2}));
    const auto exponentials = matrices.exp();
    const auto roots = exponentials.sqrt(execution_par);
    // dynamic float roots retain the analytic exponential half-value on the final diagonal
    EXPECT_NEAR(roots[0](2, 2), std::exp(1.f), 2e-6f);
    // a positive-looking symmetric input still needs an SPD contract to request a principal root
    static_assert(!permits_batch_sqrt<MatrixBatch<Sym>>);
    // inverse roots require the same verified positive-definite input contract
    static_assert(!permits_batch_inverse_sqrt<MatrixBatch<Sym>>);
    // a generic dense owner has no symmetric matrix-exponential entry point
    static_assert(!permits_batch_exp<scalar_batch>);
    // generic dense storage has no native symmetric eigenspace operation
    static_assert(!permits_batch_eigenvectors<scalar_batch>);
    // static rectangular inputs do not expose the square-matrix diagonal batch operation
    static_assert(!permits_batch_diagonal<MatrixBatch<Matrix<double, 2, 3>>>);
    const MatrixBatch<Matrix<double, Dynamic, Dynamic>> rectangular(0, 2, 3);
    // runtime rectangular shape is rejected even when no diagonal entry would be evaluated
    EXPECT_THROW(rectangular.diagonal(execution_par), std::invalid_argument);
}

// checked exponential failures propagate from workers and a subsequent call can reuse the runtime safely
TEST(MatrixBatch, ParallelExponentialRejectsOverflowAndRecovers) {
    using Sym = SymmetricMatrix<double, 2, Cache::Spectral>;
    MatrixBatch<Sym> matrices(4);
    matrices[2] = Sym(Vector<double, 3> {1000., 0., 1000.});
    // exponentials outside the finite SPD domain reject the whole eager result at the calling thread
    EXPECT_THROW(matrices.exp(execution_par), std::domain_error);
    matrices[2] = Sym(Vector<double, 3> {0., 0., 0.});
    const auto recovered = matrices.exp(execution_par);
    // the repaired zero symmetric matrix returns the analytic identity after the failed parallel operation
    EXPECT_DOUBLE_EQ(recovered[2](1, 1), 1.);
    const auto temporary = MatrixBatch<SPDMatrix<double, 2>>(4).inv_sqrt(execution_par);
    // a temporary source leaves independently owned inverse-root values after its destruction
    EXPECT_DOUBLE_EQ(temporary[3](0, 0), 1.);
}

// orthogonal batches preserve reflected bases, independent owners and structured results of lazy maps
TEST(MatrixBatch, OrthogonalEigenvectorsRetainStructureAndControlledAssignment) {
    using Basis = OrthogonalMatrix<double, 2, 2>;
    using Dense = Matrix<double, 2, 2>;
    MatrixBatch<Basis> bases(2);
    const Dense identity({1., 0., 0., 1.});
    // new orthogonal slots start at identity rather than an invalid zero matrix
    expect_matrix_eq(bases[0], identity);
    const Basis reflection({1., 0., 0., -1.}, checked);
    bases[0] = reflection;
    // the full orthogonal group admits reflected bases with determinant minus one
    EXPECT_DOUBLE_EQ(bases[0].determinant(), -1.);
    // direct coefficient access cannot break an orthogonal batch's invariant
    static_assert(std::same_as<decltype(bases.coefficients()), std::span<const double>>);
    // unstructured matrices cannot be assigned through a controlled orthogonal view
    static_assert(!std::is_assignable_v<decltype(bases[0]), Dense>);
    auto independent = bases;
    bases[0] = bases[1];
    // copying a batch preserves the reflection independently of later source assignment
    EXPECT_DOUBLE_EQ(independent[0].determinant(), -1.);
    const auto inverse_map = independent.map([](const auto& basis) { return basis.inv(); });
    // mapped orthogonal expressions keep their structure when materialized
    static_assert(std::same_as<typename decltype(inverse_map)::MatrixType, Basis>);
    const MatrixBatch<Basis> inverses(inverse_map);
    // the inverse uses the orthogonal transpose and composes with the reflected source to identity
    EXPECT_LT((independent[0] * inverses[0] - identity).norm(), 1e-14);
    const auto mapped_again = inverse_map.map([](const auto& basis) { return basis.inv(); });
    const MatrixBatch<Basis> twice(mapped_again);
    // nested maps borrow a read-only orthogonal view of their intermediate owner without dangling references
    expect_matrix_eq(twice[0], independent[0]);
    MatrixBatch<SymmetricMatrix<float, Dynamic, Cache::Spectral>> input(1, 3, 3);
    input[0] = SymmetricMatrix<float, Dynamic>(Matrix<float, 3, 3>({1, .2f, 0, .2f, 2, 0, 0, 0, 3}));
    const auto vectors = input.eigenvectors(execution_par);
    // dynamic eigenspaces retain float precision, runtime order and an orthogonal owner type
    static_assert(
      std::same_as<std::remove_cvref_t<decltype(vectors)>, MatrixBatch<OrthogonalMatrix<float, Dynamic, Dynamic>>>);
    const auto basis = vectors[0];
    const Matrix<float, 3, 3> identity3({1, 0, 0, 0, 1, 0, 0, 0, 1});
    // the structural inverse agrees with the transpose and composes to identity within float precision
    EXPECT_LT((basis * basis.inv() - identity3).norm(), 2e-5f);
    // runtime nonsquare dimensions cannot create even an empty orthogonal batch
    EXPECT_THROW((MatrixBatch<OrthogonalMatrix<double, Dynamic, Dynamic>>(0, 2, 3)), std::invalid_argument);
}

// eager inverses retain structure, result order and shape while dispatching to each element's native algorithm
TEST(MatrixBatch, InversePreservesStructureAndParallelOrder) {
    using Point = SPDMatrix<double, 2, Cache::Cholesky>;
    MatrixBatch<Point> points(2);
    points[0] = Point(Vector<double, 3> {4., 1., 3.});
    const auto sequential = points.inv<Cache::Log>();
    const auto parallel = points.inv<Cache::Log>(execution_par);
    // explicit output caching retains verified SPD owners in the batch
    static_assert(std::same_as<std::remove_cvref_t<decltype(parallel)>, MatrixBatch<SPDMatrix<double, 2, Cache::Log>>>);
    const Matrix<double, 2, 2> identity({1., 0., 0., 1.});
    for (std::size_t i = 0; i < points.size(); ++i) {
        // parallel evaluation leaves each inverse at the index of its corresponding input
        EXPECT_LT((points[i] * parallel[i] - identity).norm(), 1e-12);
        // both policies use the same retained input factors and agree numerically
        expect_matrix_eq(parallel[i], sequential[i]);
    }
    const auto bases = points.eigenvectors(execution_par);
    const auto inverses = bases.inv(execution_par);
    // orthogonal batch inverses retain orthogonal owners without promotion to dense matrices
    static_assert(std::same_as<std::remove_cvref_t<decltype(inverses)>, MatrixBatch<OrthogonalMatrix<double, 2, 2>>>);
    // structural inversion is the transpose coefficient-for-coefficient
    expect_matrix_eq(inverses[0], bases[0].transpose());
    MatrixBatch<DiagonalMatrix<double, 2>> diagonals(1);
    diagonals[0] = DiagonalMatrix<double, 2>(2., 4.);
    const auto reciprocal = diagonals.inv(execution_par);
    // diagonal inversion keeps compact diagonal storage in the output batch
    static_assert(std::same_as<std::remove_cvref_t<decltype(reciprocal)>, MatrixBatch<DiagonalMatrix<double, 2>>>);
    // diagonal entries use their scalar reciprocals rather than a dense factorization
    EXPECT_DOUBLE_EQ(reciprocal[0](1, 1), .25);
    MatrixBatch<Matrix<double, 2, 2>> dense(1);
    dense[0] = Matrix<double, 2, 2>({2., 1., 0., 4.});
    const auto dense_inverse = dense.inv(execution_par);
    // a nonsymmetric dense input follows the pivoted solve and satisfies the inverse identity
    EXPECT_LT((dense[0] * dense_inverse[0] - identity).norm(), 1e-12);
    const MatrixBatch<SymmetricMatrix<double, Dynamic, Cache::Spectral>> empty(0, 3, 3);
    // empty structured inverses preserve the runtime matrix order without reading an element
    expect_batch_shape(empty.inv(execution_par), 0, 3, 3);
    const MatrixBatch<Matrix<double, Dynamic, Dynamic>> rectangular(0, 2, 3);
    // an empty batch cannot bypass square-shape validation for inversion
    EXPECT_THROW(rectangular.inv(execution_par), std::invalid_argument);
    MatrixBatch<SymmetricMatrix<double, 2, Cache::Spectral>> singular(2);
    // singular worker inputs fail at the caller rather than returning a partial batch
    EXPECT_THROW(singular.inv(execution_par), std::domain_error);
    singular[0] = SymmetricMatrix<double, 2>(identity);
    singular[1] = SymmetricMatrix<double, 2>(identity);
    const auto recovered = singular.inv(execution_par);
    // the executor and complete result initialization remain usable after the failed inversion
    expect_matrix_eq(recovered[1], identity);
}

}   // namespace
