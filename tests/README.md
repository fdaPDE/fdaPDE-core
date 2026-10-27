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

Configuration also compiles the registered negative programs and requires each to emit
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

## Execution

Execution tests check task ownership, accounting, queue bounds, nested algorithms, reduction identity and configuration validation. Configuration and saturation use separate processes. See [the execution contract](../docs/execution.md).

## Optimized algebra integration (27 October 2026)

The existing dense target retains the contiguous-assignment, contiguous-product and
preallocated-product regressions and now covers Cholesky, quaternions, structured
inverse/trace APIs and symmetric cache invalidation. The SPD target additionally
checks specialized 2x2/3x3 kernels against dynamic EVD, independent high-precision
reference values and finite-difference derivatives. Cache and batch suites verify
policy reuse, deferred arithmetic, aliases, failed assignments, sequential/parallel
ordering and exception recovery. Native sparse products and Eigen oracle suites
remain registered.

Configure the same tests with `-DCMAKE_CXX_FLAGS=-DFDAPDE_ENABLE_SIMD=1` to exercise
the optimized dense paths. Ordinary tests keep debug assertions enabled in Release.
The two focused disabled-assertion targets check argument erasure and mandatory
orthogonalization, rather than duplicating the normal suites.

With `-DFDAPDE_BUILD_SPD_BENCHMARK=ON`, run
`build/integration/fdapde_spd_cache_batch_benchmark` for native allocation checks,
LU timings and complete SPD3 logarithm batches at 4096 and 65536 entries. Batch
timing uses four configured executor threads, seven samples of three complete calls,
and excludes input setup; every sequential/parallel output coefficient is checked
before timing. This module probe does not require manifold geometry. Allocation
probes finish before worker startup. Wall-clock values are diagnostic, without
performance pass/fail thresholds.

The report and archived historical evidence are in
[`docs/reports/linear-algebra-2026-10-27/`](../docs/reports/linear-algebra-2026-10-27/).
