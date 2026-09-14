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

#include <fdaPDE/manifold_optimization.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace benchmark_memory {

/// @brief enables accounting only for a selected sequential measurement interval
bool probe_active = false;
/// @brief counts allocations made during the selected sequential measurement interval
std::size_t total_allocations = 0;
/// @brief counts live allocations made during the selected sequential measurement interval
std::size_t live_allocations = 0;

/// @brief obtains storage and records one live allocation
void* allocate(std::size_t size) {
    if (void* pointer = std::malloc(size == 0 ? 1 : size)) {
        if (probe_active) {
            ++total_allocations;
            ++live_allocations;
        }
        return pointer;
    }
    throw std::bad_alloc {};
}

/// @brief obtains aligned storage and records one live allocation
void* allocate_aligned(std::size_t size, std::size_t alignment) {
    void* pointer = nullptr;
    if (posix_memalign(&pointer, alignment, size == 0 ? 1 : size) == 0) {
        if (probe_active) {
            ++total_allocations;
            ++live_allocations;
        }
        return pointer;
    }
    throw std::bad_alloc {};
}

/// @brief releases recorded storage when a nonnull allocation is destroyed
void release(void* pointer) noexcept {
    if (pointer != nullptr) {
        if (probe_active) --live_allocations;
        std::free(pointer);
    }
}

/// @brief captures allocation counters within sequential benchmark work
struct Snapshot {
    std::size_t total;
    std::size_t live;
};

/// @brief returns a stable allocation counter snapshot for one measurement boundary
Snapshot snapshot() { return {total_allocations, live_allocations}; }

/// @brief limits accounting to one construction and destruction interval
class ProbeGuard {
   public:
    /// @brief begins an isolated sequential allocation measurement
    ProbeGuard() { probe_active = true; }
    /// @brief ends the isolated sequential allocation measurement
    ~ProbeGuard() { probe_active = false; }
    /// @brief prevents copying an active process-wide measurement scope
    ProbeGuard(const ProbeGuard&) = delete;
    /// @brief prevents replacing an active process-wide measurement scope
    ProbeGuard& operator=(const ProbeGuard&) = delete;
};

}   // namespace benchmark_memory

/// @brief records scalar allocation deltas while a constructed value remains alive
struct ConstructionMeasurement {
    std::size_t persistent_allocations;
    std::size_t temporary_allocations;
    std::size_t total_allocations;
};

/// @brief defines ordinary global allocation for this benchmark executable only
void* operator new(std::size_t size) { return benchmark_memory::allocate(size); }
/// @brief defines ordinary global array allocation for this benchmark executable only
void* operator new[](std::size_t size) { return benchmark_memory::allocate(size); }
/// @brief defines aligned global allocation for this benchmark executable only
void* operator new(std::size_t size, std::align_val_t alignment) {
    return benchmark_memory::allocate_aligned(size, static_cast<std::size_t>(alignment));
}
/// @brief defines aligned global array allocation for this benchmark executable only
void* operator new[](std::size_t size, std::align_val_t alignment) {
    return benchmark_memory::allocate_aligned(size, static_cast<std::size_t>(alignment));
}
/// @brief defines ordinary global deallocation for this benchmark executable only
void operator delete(void* pointer) noexcept { benchmark_memory::release(pointer); }
/// @brief defines ordinary global array deallocation for this benchmark executable only
void operator delete[](void* pointer) noexcept { benchmark_memory::release(pointer); }
/// @brief defines sized global deallocation for this benchmark executable only
void operator delete(void* pointer, std::size_t) noexcept { benchmark_memory::release(pointer); }
/// @brief defines sized global array deallocation for this benchmark executable only
void operator delete[](void* pointer, std::size_t) noexcept { benchmark_memory::release(pointer); }
/// @brief defines aligned global deallocation for this benchmark executable only
void operator delete(void* pointer, std::align_val_t) noexcept { benchmark_memory::release(pointer); }
/// @brief defines aligned global array deallocation for this benchmark executable only
void operator delete[](void* pointer, std::align_val_t) noexcept { benchmark_memory::release(pointer); }
/// @brief defines sized aligned global deallocation for this benchmark executable only
void operator delete(void* pointer, std::size_t, std::align_val_t) noexcept { benchmark_memory::release(pointer); }
/// @brief defines sized aligned global array deallocation for this benchmark executable only
void operator delete[](void* pointer, std::size_t, std::align_val_t) noexcept { benchmark_memory::release(pointer); }

namespace {

using namespace fdapde;
using namespace fdapde::manifold;

using full_cache =
  Cache::Union<Cache::Spectral, Cache::Log, Cache::Sqrt, Cache::InverseSqrt, Cache::LogDividedDifferences>;
using static_none_owner = SPDMatrix<double, 2, 2>;
using static_full_owner = SPDMatrix<double, 2, 2, full_cache>;
using dynamic_none_owner = SPDMatrix<double, Dynamic, Dynamic>;
using dynamic_full_owner = SPDMatrix<double, Dynamic, Dynamic, full_cache>;
using no_cache_geometry = LogEuclideanSPDGeometry<double, 2, Usage::None>;
using cached_geometry = LogEuclideanSPDGeometry<double, 2, Usage::Distance | Usage::BasePointMaps>;

// no-cache owners retain exactly the compact symmetric representation
static_assert(sizeof(static_none_owner) == sizeof(SymmetricMatrix<double, 2, 2>));
// no-cache owners preserve compact symmetric alignment
static_assert(alignof(static_none_owner) == alignof(SymmetricMatrix<double, 2, 2>));
// geometry without uses stores no cache quantities in its canonical point
static_assert(no_cache_geometry::CachePolicy::Flags == Cache::None::Flags);
// distance and base-point uses request retained log and spectral quantities
static_assert(cached_geometry::CachePolicy::Flags != Cache::None::Flags);

volatile double benchmark_checksum = 0;

/// @brief prevents timed numeric results from being discarded by optimization
void consume(double value) { benchmark_checksum = benchmark_checksum + value; }

/// @brief measures persistent and total allocations while a factory result remains alive
template <typename Factory> ConstructionMeasurement measure_construction(Factory&& factory) {
    const auto before = benchmark_memory::snapshot();
    ConstructionMeasurement measurement {};
    double observed_rows = 0;
    {
        benchmark_memory::ProbeGuard guard;
        auto value = std::forward<Factory>(factory)();
        const auto constructed = benchmark_memory::snapshot();
        observed_rows = static_cast<double>(value.rows());
        const auto persistent = constructed.live - before.live;
        const auto total = constructed.total - before.total;
        measurement = {persistent, total - persistent, total};
    }
    const auto cleaned = benchmark_memory::snapshot();
    // construction and destruction restore the live allocation snapshot
    if (cleaned.live != before.live) std::abort();
    consume(observed_rows);
    return measurement;
}

/// @brief returns the median nanoseconds per invocation over fixed sequential runs
template <typename Operation> double median_nanoseconds(std::size_t iterations, Operation&& operation) {
    std::array<double, 7> samples {};
    for (double& sample : samples) {
        const auto start = std::chrono::steady_clock::now();
        for (std::size_t i = 0; i < iterations; ++i) consume(std::forward<Operation>(operation)());
        const auto stop = std::chrono::steady_clock::now();
        sample = std::chrono::duration<double, std::nano>(stop - start).count() / static_cast<double>(iterations);
    }
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

/// @brief creates distinct two-by-two SPD values in a batch without timing setup work
template <typename Batch> void populate_points(Batch& points) {
    for (std::size_t i = 0; i < points.size(); ++i) {
        const double first = 2 + static_cast<double>(i % 5);
        const double second = 5 + static_cast<double>(i % 7);
        points[i].assign(Matrix<double, 2, 2>({first, 0.25, 0.25, second}));
    }
}

/// @brief prints one construction allocation measurement as a compact csv record
void print_allocation(const char* name, std::size_t count, ConstructionMeasurement measurement) {
    std::cout << "allocation," << name << ',' << count << ',' << measurement.persistent_allocations << ','
              << measurement.temporary_allocations << ',' << measurement.total_allocations << '\n';
}

/// @brief prints one sequential timing median as a compact csv record
void print_timing(const char* name, double nanoseconds) {
    std::cout << "timing," << name << ',' << std::fixed << std::setprecision(2) << nanoseconds << '\n';
}

/// @brief checks aggregate identity allocation contracts without constraining allocation byte sizes
void verify_identity_batches(
  ConstructionMeasurement static_none, ConstructionMeasurement static_full, ConstructionMeasurement dynamic_none,
  ConstructionMeasurement dynamic_full) {
    // nonempty static cache-free batches retain only their coefficient vector allocation
    fdapde_strong_assert(
      static_none.persistent_allocations == 1, std::logic_error,
      "benchmark: static cache-free batch allocation count changed");
    // nonempty dynamic cache-free batches retain only their coefficient vector allocation
    fdapde_strong_assert(
      dynamic_none.persistent_allocations == 1, std::logic_error,
      "benchmark: dynamic cache-free batch allocation count changed");
    // static cached batches retain coefficients plus values slots and pointer table allocations
    fdapde_strong_assert(
      static_full.persistent_allocations == 4, std::logic_error,
      "benchmark: static cached batch allocation count changed");
    // dynamic cached batches retain the same aggregate cache allocations as static batches
    fdapde_strong_assert(
      dynamic_full.persistent_allocations == 4, std::logic_error,
      "benchmark: dynamic cached batch allocation count changed");
    // identity construction introduces no transient allocation for any batch storage mode
    fdapde_strong_assert(
      static_none.temporary_allocations == 0 && static_full.temporary_allocations == 0 &&
        dynamic_none.temporary_allocations == 0 && dynamic_full.temporary_allocations == 0,
      std::logic_error, "benchmark: identity batch construction allocated a temporary workspace");
}

/// @brief measures static and dynamic batch construction at a fixed count
void measure_batches(std::size_t count) {
    using static_none_batch = MatrixBatch<static_none_owner>;
    using static_full_batch = MatrixBatch<static_full_owner>;
    using dynamic_none_batch = MatrixBatch<dynamic_none_owner>;
    using dynamic_full_batch = MatrixBatch<dynamic_full_owner>;

    const auto static_none = measure_construction([&] { return static_none_batch(count); });
    const auto static_full = measure_construction([&] { return static_full_batch(count); });
    const auto dynamic_none = measure_construction([&] { return dynamic_none_batch(count, 2, 2); });
    const auto dynamic_full = measure_construction([&] { return dynamic_full_batch(count, 2, 2); });
    verify_identity_batches(static_none, static_full, dynamic_none, dynamic_full);
    // static cache-free identity batches establish the coefficient-only allocation baseline
    print_allocation("batch_static_none_identity", count, static_none);
    // static cached identity batches retain aggregate cache storage independent of point count
    print_allocation("batch_static_full_identity", count, static_full);
    // dynamic cache-free identity batches establish runtime-shape coefficient storage
    print_allocation("batch_dynamic_none_identity", count, dynamic_none);
    // dynamic cached identity batches retain runtime aggregate cache storage independent of point count
    print_allocation("batch_dynamic_full_identity", count, dynamic_full);
}

/// @brief measures candidate preparation allocations separately from persistent aggregate batch storage
void measure_batch_from_values() {
    MatrixBatch<static_none_owner> source(64);
    populate_points(source);
    // verified source construction prepares missing cache quantities without recertifying input coefficients
    print_allocation("batch_static_full_from_values", source.size(), measure_construction([&] {
                         return MatrixBatch<static_full_owner>(source);
                     }));

    MatrixBatch<dynamic_none_owner> dynamic_source(64, 2, 2);
    populate_points(dynamic_source);
    // dynamic source-value construction separates runtime workspace from aggregate cache storage
    print_allocation("batch_dynamic_full_from_values", dynamic_source.size(), measure_construction([&] {
                         return MatrixBatch<dynamic_full_owner>(dynamic_source);
                     }));

    MatrixBatch<Matrix<double, Dynamic, Dynamic>> dense_source(64, 2, 2);
    for (std::size_t i = 0; i < dense_source.size(); ++i) dense_source[i] = dynamic_source[i];
    const auto dense_none = measure_construction([&] { return MatrixBatch<dynamic_none_owner>(dense_source); });
    const auto dense_full = measure_construction([&] { return MatrixBatch<dynamic_full_owner>(dense_source); });
    // unverified dense values retain one aggregate coefficient allocation after SPD validation
    fdapde_strong_assert(
      dense_none.persistent_allocations == 1, std::logic_error,
      "benchmark: dense source cache-free batch allocation count changed");
    // unverified dense values retain aggregate coefficients values slots and pointer table after cache preparation
    fdapde_strong_assert(
      dense_full.persistent_allocations == 4, std::logic_error,
      "benchmark: dense source cached batch allocation count changed");
    // raw dense coefficient views force validation work without copying an owning SPD input
    print_allocation("batch_dynamic_none_from_unverified_dense", dense_source.size(), dense_none);
    // raw dense coefficient views add cache preparation to the same validation path
    print_allocation("batch_dynamic_full_from_unverified_dense", dense_source.size(), dense_full);
}

/// @brief measures distance, deferred weighted mean and discarded candidates for one cache policy
template <typename Geometry> void measure_geometry(const char* suffix) {
    using Point = typename Geometry::Point;
    constexpr std::size_t point_count = 16;
    Geometry geometry;
    MatrixBatch<Point> points(point_count);
    populate_points(points);
    Vector<double, static_cast<int>(point_count)> weights;
    for (std::size_t i = 0; i < point_count; ++i) weights[static_cast<int>(i)] = 1.0 / static_cast<double>(point_count);

    print_timing((std::string("le_distance_") + suffix).c_str(), median_nanoseconds(4000, [&] {
                     return geometry.distance(points[0], points[1]);
                 }));
    print_timing((std::string("le_weighted_mean_") + suffix).c_str(), median_nanoseconds(100, [&] {
                     Point value(geometry.weighted_mean(points, weights));
                     return value(0, 0);
                 }));

    const Matrix<double, 2, 2> candidate_value({4, 0.25, 0.25, 9});
    print_timing((std::string("discarded_candidate_") + suffix).c_str(), median_nanoseconds(1000, [&] {
                     Point candidate(candidate_value);
                     return candidate(0, 0);
                 }));
}

}   // namespace

// measures cache layout, allocation lifetime and sequential geometry costs without test framework dependencies
int main() {
    std::cout << "metric,name,count,persistent_allocations,temporary_allocations,total_allocations\n";
    std::cout << "layout,symmetric_owner_size," << sizeof(SymmetricMatrix<double, 2, 2>) << ",align,"
              << alignof(SymmetricMatrix<double, 2, 2>) << '\n';
    std::cout << "layout,static_none_owner_size," << sizeof(static_none_owner) << ",align,"
              << alignof(static_none_owner) << '\n';
    std::cout << "layout,static_full_owner_size," << sizeof(static_full_owner) << ",align,"
              << alignof(static_full_owner) << '\n';

    const auto static_none = measure_construction([] { return static_none_owner::Identity(); });
    const auto static_full = measure_construction([] { return static_full_owner::Identity(); });
    const auto dynamic_none = measure_construction([] { return dynamic_none_owner::Identity(2); });
    const auto dynamic_full = measure_construction([] { return dynamic_full_owner::Identity(2); });
    // static cache-free identity owners allocate neither cache storage nor temporary workspaces
    fdapde_strong_assert(
      static_none.persistent_allocations == 0 && static_none.total_allocations == 0, std::logic_error,
      "benchmark: static cache-free owner allocation count changed");
    // dynamic cache-free identities retain exactly one coefficient allocation
    fdapde_strong_assert(
      dynamic_none.persistent_allocations == 1, std::logic_error,
      "benchmark: dynamic cache-free owner allocation count changed");
    // dynamic cached identities add exactly the cache owner and its selected scalar buffer
    fdapde_strong_assert(
      dynamic_full.persistent_allocations == dynamic_none.persistent_allocations + 2, std::logic_error,
      "benchmark: dynamic cached owner allocation count changed");
    // cache-free and fully cached static owners isolate conditional owner storage
    print_allocation("owner_static_none_identity", 1, static_none);
    // fully cached static identity owners initialize selected intermediates without an eigendecomposition
    print_allocation("owner_static_full_identity", 1, static_full);
    // cache-free dynamic identities retain only runtime symmetric coefficient storage
    print_allocation("owner_dynamic_none_identity", 1, dynamic_none);
    // fully cached dynamic identities initialize selected intermediates without an eigendecomposition
    print_allocation("owner_dynamic_full_identity", 1, dynamic_full);

    for (const std::size_t count : {std::size_t {1}, std::size_t {64}, std::size_t {1024}}) measure_batches(count);
    measure_batch_from_values();
    measure_geometry<no_cache_geometry>("none");
    measure_geometry<cached_geometry>("distance_base_maps");

    std::cout << "checksum," << std::setprecision(17) << static_cast<double>(benchmark_checksum) << '\n';
}
