# Incremental integration tests

Configure and run this stable-compatible pilot with:

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
without pulling in the stable utility matrix implementation.
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

The execution suite contains 23 cases. With the 16 assertion and caller-contract
cases and one multi-TU runtime test, CTest runs 40 cases.
Two standalone header checks plus eight include-order units compile in the same build.
Execution configuration and one-worker saturation run in separate processes so singleton
configuration is never changed after initialization. All execution tests use debug assertions.

The new suite checks ownership, accounting, bounded queues, dependency ordering,
nested parallel execution, reduction order/identity and configuration validation.
The multi-TU test checks shared configuration, singleton identity and real execution.
The CI workflow targets this incremental suite; native-only gates are intentionally
absent while the stable numerical modules still depend on Eigen.

See [the execution contract](../docs/execution.md) for the
API requirements and runtime limits.
