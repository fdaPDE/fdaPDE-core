# Incremental integration tests

Configure and run the integration suite with:

```sh
cmake -S tests -B build/integration -DCMAKE_BUILD_TYPE=Debug
cmake --build build/integration --parallel 4
ctest --test-dir build/integration --output-on-failure
```

Requirements: C++20 compiler, CMake 3.20+, Eigen 3.4+ for stable core checks,
and GoogleTest 1.14 (pinned source downloaded by CMake). An existing source
checkout can be supplied through `FETCHCONTENT_SOURCE_DIR_GOOGLETEST`.
The historical `test/` directory is retained; its disabled test includes are
not counted as verification of this pilot.

`fdapde_assert(condition, exception_type, message)` throws in debug mode;
`FDAPDE_NO_DEBUG` removes both condition and message evaluation.
`fdapde_strong_assert` always checks its condition. Messages are evaluated only
on failure. `NDEBUG` alone does not disable fdaPDE assertions, so Release builds
still exercise debug contracts unless `FDAPDE_NO_DEBUG` is explicitly set.
The former one-argument runtime assertion is replaced. Callers use
`std::invalid_argument` for invalid dimensions, values, and descriptors,
`std::out_of_range` for index and name lookup failures, `std::domain_error` for
mathematical domain violations, and `std::logic_error` for invalid object states
and internal invariants. Independent conditions have separate diagnostics.
The typed macro also supports constant evaluation; the one-argument constexpr
helper remains available for compatibility.

The assertion header is self-contained so execution can use the shared macros
without pulling in the dense algebra implementation.
The single `AssertionsDisabled` case is a focused exception to debug-only
verification: argument erasure cannot be demonstrated with debug enabled.
No ordinary suite is duplicated with `FDAPDE_NO_DEBUG`.

The core header check and the grid-search/interval cases guard the migration
of stable callers. A pre-existing unused `NaN` constant in GridSearch was
removed to permit the strict header build; no numerical implementation changed.


The Linux integration workflow uses GCC 14 and Clang with debug assertions and
warnings as errors. GCC 13.3 on the Ubuntu 24.04 runner crashes inside
`add_alignment_attribute` while emitting DWARF for the stable GeoFrame templates.
Compiler jobs run independently so a failure in one does not cancel the other.

## Dense algebra

The dense target exercises matrices, expressions, packed Boolean storage,
multidimensional arrays, structured types, and LU/QR/EVD. The caller target checks
P1/P2 cardinality in 1D/2D/3D, exact P1 and P2 triangle assembly, mixed extent types
GeoFrame column storage and grid-search storage order. Header checks compile each public aggregate alone.

Configuration also compiles negative programs for each registered rejection contract and requires each to emit
its specific static-assert diagnostic. Logs are written beneath
`compile_fail/` in the build directory. Debug assertions remain enabled in these
programs; none duplicates the suite in NoDebug mode.

Eight cases from `test/src/binary_matrix_test.cpp` are executed in
`linear_algebra/historical_boolean.cpp`. Their original locations retain precise
replacement pointers. The historical Eigen conversion case remains in place,
as do historical FEM and other cases whose complete coverage is not replaced.

## Native algebra

The dense tests include `dense_linear_algebra.h` directly without an Eigen target
dependency. Run the native lane with Eigen unavailable to the compiler:

```sh
cmake -S tests -B build/native -DFDAPDE_NATIVE_ONLY=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build/native --parallel 4
ctest --test-dir build/native --output-on-failure
```

The default integration lane retains Eigen-dependent core callers and verifies
both dense/full aggregate inclusion orders in linked translation units. Native
and integration lanes both keep debug assertions enabled in ordinary tests.

## SPD geometry

The geometry targets exercise log-Euclidean and affine-invariant SPD operations,
public error contracts and owning results. The integration lane additionally
checks both manifold/full aggregate include orders in linked translation units.


## SPD caches and matrix batches

The cache and batch targets cover selective static/dynamic storage, identity,
policy conversion, checked view assignment, deferred map/redux/select and
lifetime rejection. Geometry tests cover mixed policies, canonical result types,
non-normalized LE means, global materialization counts and cached metric/map
agreement on noncommuting inputs. All use default debug assertions. The only
NoDebug target is the focused assertion-argument-erasure test.

For a standalone allocation probe and sequential microbenchmark:

```sh
cmake -S tests -B build/native-bench -DFDAPDE_NATIVE_ONLY=ON \
  -DCMAKE_BUILD_TYPE=Release -DFDAPDE_BUILD_SPD_BENCHMARK=ON
cmake --build build/native-bench --target fdapde_spd_cache_batch_benchmark --parallel 4
build/native-bench/fdapde_spd_cache_batch_benchmark
```

The probe also compiles directly on macOS/Linux without GoogleTest or Eigen:

```sh
c++ -std=c++20 -O2 -I. tests/benchmarks/spd_cache_batch.cpp -o /tmp/spd-cache-bench
/tmp/spd-cache-bench
```

Allocation instrumentation lives only in this executable and is disabled during
timing. Persistent allocations are live immediately after construction;
temporary allocations are the total allocation count minus that live increase.
The probe checks NoCache layout and allocations against the native storage
control, and checks constant aggregate allocation counts for 1, 64 and 1024
static/dynamic SPD elements. It separately measures construction from supplied
values, including transient validation/cache preparation work. Timing reports
medians of seven sequential runs and consumes results through a checksum.
Policy-expansion probes compare sources with and without a retained logarithm,
both with and without spectral factors, to detect repeated reconstruction work.
Results are workload/toolchain dependent; the discarded-candidate measurement
includes cache preparation whose benefit is never used.
