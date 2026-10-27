# Linear algebra optimization report - 27 October 2026

The Italian report is saved at
[`output/pdf/linear-algebra-optimizations-2026-10-27.pdf`](../../../output/pdf/linear-algebra-optimizations-2026-10-27.pdf).
It separates historical SIMD/Eigen measurements from validation and timing of the
algebra integration on `develop-SIMD`. Exact code and source revisions are recorded
in `evidence/validation.json`.

The algebra port comes from `develop-GFA@f146ddf27cd28de333e134db252ae15eaaf918f4`.
Execution and native SPD spectral foundations preserve their original history.
The port excludes GFA manifold/GFE implementations and uncommitted C-LE changes.
SIMD and native sparse preallocated APIs remain present. The source remains
available on GFA; this operation consolidates its reusable algebra on the thematic
branch without removing its application dependencies.

## Validation

- AppleClang 21, Release, SIMD OFF, integration: 313/313 tests
- GCC 15, Release, SIMD ON, integration: 313/313 tests
- AppleClang 21, Release, SIMD ON, native-only: 277/277 tests, Eigen unavailable
- 28 compile-fail diagnostics per configuration, plus public-header checks
- All 11 numbered algebra tutorials compiled and executed

Ordinary tests keep fdaPDE debug assertions enabled. The two narrow NoDebug cases
check assertion-argument erasure and required orthogonalization. The macOS GCC lane
uses `-D_Static_assert=static_assert` for SDK compatibility. Tutorial 11 treats Eigen
as a system include, matching CMake's imported Eigen target.

```sh
cmake -S tests -B build/integration -DCMAKE_BUILD_TYPE=Release -DFDAPDE_BUILD_SPD_BENCHMARK=ON
cmake --build build/integration --parallel 4
ctest --test-dir build/integration --output-on-failure --parallel 4
build/integration/fdapde_spd_cache_batch_benchmark
```

For ON add `-DCMAKE_CXX_FLAGS=-DFDAPDE_ENABLE_SIMD=1`; for a native-only build also add
`-DFDAPDE_NATIVE_ONLY=ON`. CMake downloads pinned GoogleTest by default. An existing
checkout can be selected with `FETCHCONTENT_SOURCE_DIR_GOOGLETEST`, as recorded in
the exact local configuration metadata.

## Evidence and limits

`evidence/` contains current JUnit XML, configuration/source hashes, allocation and
batch timings, and three alternating process pairs for the current SIMD controls.
Timing has no pass/fail threshold. The batch probe compares every output coefficient
before measuring seven samples of three complete calls with four executor threads.
Input preparation is excluded; result allocation is included. It runs allocation
probes before executor startup and keeps accounting disabled during parallel work.

Historical JSON files are gzip copies of the original bytes. The archive manifest
records source paths and SHA-256 values. Full CSV summaries contain the 575 verified
Eigen comparisons (143 cases) and 410 V1 sweep rows. Native-product results retain
separate AppleClang 21 and Clang 17 measurements. V2 has only 31 rows in two product
cases and one stop record; the large-assignment and Clang17-v2 summaries are absent.
No completed campaign or plateau is inferred from those incomplete files.

Historical sanitizer results refer to the earlier preallocated implementation;
no sanitizer, Linux/Kami or end-to-end GFA run was added in this integration.
Background workloads, compiler versions, cache residency and measurement protocol
limit performance conclusions. SIMD remains opt-in.

## Regenerate the PDF

Use Python with ReportLab and the Arial/Arial Bold fonts installed on macOS:

```sh
python3 docs/reports/linear-algebra-2026-10-27/generate_report.py
```

The script reads only archived evidence. It does not rerun or overwrite benchmark
measurements. On another platform, replace the two explicit font paths with local
fonts before generating. Render with Poppler and inspect all pages after changes.

## Review scope

The final integration checks covered matrix materialization and SIMD dispatch,
structured storage boundaries, public headers, test registrations, comments and
Doxygen on recovered module changes. Changed C++ sources follow the repository
clang-format configuration. Existing task/runtime limits are documented in
`docs/execution.md`; no general performance or memory-model guarantee is claimed.
The current native/Eigen boundary and batch API are documented by the runnable
numbered tutorials and module docs; these are the source-backed updates relevant
to the hub's `fdapde/modules/matrix-type-system.md` evidence cache.
