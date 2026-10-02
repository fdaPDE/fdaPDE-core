# SIMD in native dense algebra

## Compile-time selection

The native loop optimizations are **off by default**. Enable both before including
any fdaPDE header, or pass the compiler definition:

```sh
-DFDAPDE_ENABLE_SIMD=1
```

The two paths can also be selected independently:

```sh
-DFDAPDE_ENABLE_SIMD_ASSIGNMENT=1  # contiguous assignments, broadcasts and updates
-DFDAPDE_ENABLE_SIMD_PRODUCT=1     # contiguous dense product materialization
```

Each individual switch defaults to the umbrella switch. An explicit individual
`=0` overrides an enabled umbrella switch. Keep these definitions consistent across
all translation units using fdaPDE: they change inline/template definitions.
For CMake consumers use `target_compile_definitions(your_target PRIVATE
FDAPDE_ENABLE_SIMD=1)`. The core requires no library, architecture option or new
dependency. These switches select C++ loops; actual SIMD generation remains the
compiler's decision. Disabled loops can still be auto-vectorized at `-O3`.

The existing pilot benchmark targets explicitly enable both paths. The four sweep
targets explicitly select their own assignment/product combination.

## Contiguous assignment

With the assignment switch enabled, native assignment uses the compiler's loop vectorizer within the C++20 API.
Ordinary `Matrix` and `MatrixView` storage can use a flat coefficient loop:

- scalar broadcasting uses contiguous destination storage;
- same-layout dense matrix copies and updates use contiguous source and destination storage;
- vector copies use physical coefficient order independently of row/column orientation.

The executor captures storage pointers before the loop, avoiding repeated stride
loads and making short matrix axes independent of SIMD lane width. Fixed storage
still supports constant evaluation. External storage requires only the scalar's
normal alignment. Compiler-generated remainder loops handle odd coefficient counts.

Shape checks retain their existing assertion policy. Expression assignments still
materialize a source snapshot before writing the destination, preserving overlapping
views and self assignment. Blocks, mixed matrix layouts, structured storage and
arbitrary expressions retain coordinate evaluation where contiguous access does not
apply. Dense and sparse kernels share the exact owner/view storage classification;
packed Boolean and structured storage do not qualify for the flat path.

## Reproduce the pilot

The benchmark times public scaling for owning vectors, row-major matrices with
three columns, column-major matrices with three rows, and external views. Affine
assignments cover vectors, row-major matrices and external views. It reports the
median of five rounds in nanoseconds per call.
Every output coefficient is checked against scalar arithmetic, including in builds
with `NDEBUG`. Affine inputs occupy distinct allocations and public source snapshots
remain part of the measured time.

From the repository root, build the opt-in CMake target in Release:

```sh
cmake -S tests -B build/simd -DCMAKE_BUILD_TYPE=Release -DFDAPDE_NATIVE_ONLY=ON
cmake --build build/simd --target fdapde_simd_benchmark
./build/simd/fdapde_simd_benchmark
```

The target disables internal debug assertions for timing and keeps its own driver
checks active. System-header treatment avoids warnings about assertion-erased
parameters while keeping warnings enabled for the benchmark source. Floating-point
contraction is disabled for GCC/Clang to make scalar reference comparisons exact.

A standalone build requires only a C++20 compiler and the native headers:

```sh
c++ -std=c++20 -O3 -DNDEBUG -DFDAPDE_NO_DEBUG -DFDAPDE_ENABLE_SIMD=1 -ffp-contract=off \
  -I. tests/benchmarks/simd.cpp -o /tmp/fdapde-simd-benchmark
/tmp/fdapde-simd-benchmark
```

Inspect generated vector loops using Clang's `-Rpass=loop-vectorize` and
`-Rpass-missed=loop-vectorize`, or GCC's `-fopt-info-vec-optimized` and
`-fopt-info-vec-missed`. Clang's `-fno-vectorize -fno-slp-vectorize` or GCC's
`-fno-tree-vectorize` can help distinguish compiler vectorization from other
changes. Compare the same benchmark source and flags against the merged dependency
baseline; public snapshot costs make raw-pointer loops a different workload.

## Pilot observations

Measured on an Apple M3 Max (ARM64) against dependency baseline `9f83f60`,
using the same benchmark source and `-O3 -DNDEBUG -DFDAPDE_NO_DEBUG
-ffp-contract=off`. Two sequential comparison batches used opposite execution
orders after compiler jobs finished. Ranges span the two paired median ratios,
not a statistical confidence interval. A ratio above one is faster.

| Compiler | Public workload | Coefficients | Baseline / contiguous path |
| --- | --- | ---: | ---: |
| Clang 17.0.6 | vector scaling | 12,288 | 2.26–3.04× |
| Clang 17.0.6 | three-column matrix scaling | 12,288 | 4.66–6.08× |
| Clang 17.0.6 | matrix affine assignment | 393,216 | 1.41–1.43× |
| AppleClang 21.0.0 | three-column matrix scaling | 12,288 | 1.97–2.05× |
| AppleClang 21.0.0 | matrix affine assignment | 393,216 | 1.38–1.40× |
| AppleClang 21.0.0 | vector scaling | 393,216 | 0.84–0.97× |

Clang 17 emits packed double arithmetic for vector-reference scaling with the
contiguous path; the baseline missed that vectorization. AppleClang 21 already
vectorizes baseline vectors. Its large-vector scaling timings were slower in
these runs, while the main SIMD load/multiply/store loop is unchanged. The
results establish a benefit for short-axis matrices on this machine and do not
establish a universal improvement across workloads or CPUs.

Both Clang and GCC 15 passed 108 registered tests, all public-header targets,
and 16 expected compile failures. Tests keep debug assertions enabled; the
benchmark verifies every coefficient with explicit release checks. GCC's macOS
GoogleTest build used the temporary SDK compatibility flag
`-D_Static_assert=static_assert` without changing library sources.

Timing logs, compiler remarks, LLVM IR and validation logs are retained locally
under `output/simd/`. The same benchmark can be compiled against headers from
`9f83f60` to reproduce the comparison with the merged PR dependencies.

## Native product materialization

Runtime construction of an ordinary dense owner from a direct product can evaluate
its complete output with a contiguous inner loop. The generic product executor
uses `i-k-j` accumulation for row-major output with row-major right operands, or
`j-k-i` accumulation for column-major output with column-major left operands.
Each output coefficient still accumulates in increasing `k` order. There is no
horizontal reduction, fast-math requirement, explicit ISA code or external backend.

The private product hook is available only to `Matrix` materialization. Public
assignment continues to evaluate into an independent snapshot before writing the
destination, including overlapping views, self products and compound assignments.
The output is explicitly initialized to zero, and empty products avoid pointer
offsets on absent input storage.

This path requires ordinary `Matrix` or `MatrixView` operands, homogeneous `float`
or `double`, and matching output scalar/layout eligibility. Constant evaluation,
volatile input, other scalar types, mixed precision, unfavorable layouts, nested
expressions, blocks and structured executors retain coefficient evaluation.
Vector orientation changes also retain their existing physical coefficient order.
The loops are currently unblocked; cache tiling needs separate measurements on
larger products before adding another kernel.

The sparse multi-RHS inner loop already vectorizes on both tested Clang compilers.
Reductions and solver algorithms are unchanged.

Build the public product benchmark without an external algebra library:

```sh
cmake -S tests -B build/simd -DCMAKE_BUILD_TYPE=Release -DFDAPDE_NATIVE_ONLY=ON
cmake --build build/simd --target fdapde_native_product_benchmark
./build/simd/fdapde_native_product_benchmark
```

The benchmark includes static FEM-sized products, odd rectangular shapes, both
storage orders, mixed layouts, `float`, `double` and unaligned external views.
Every result is checked against an ordered scalar oracle even with `NDEBUG`;
public assignment snapshots and allocation costs are included in the timings.
Compare against `91be5d5` with the same benchmark source and flags. Timing logs,
compiler remarks, sanitizer checks and review notes are retained locally under
`output/simd/native-product/`.

The same driver also builds directly against the native headers:

```sh
c++ -std=c++20 -O3 -DNDEBUG -DFDAPDE_NO_DEBUG -DFDAPDE_ENABLE_SIMD=1 -ffp-contract=off \
  -isystem . tests/benchmarks/native_product.cpp -o /tmp/fdapde-native-product-benchmark
/tmp/fdapde-native-product-benchmark
```

## Product observations

Measured on the same Apple M3 Max (ARM64), against `91be5d5`, with
`-O3 -DNDEBUG -DFDAPDE_NO_DEBUG -ffp-contract=off` and no ISA-specific options.
The same benchmark source was used for both revisions. Each compiler ran three
sequential baseline/current pairs in alternating process order after all build
jobs completed. Entries are ratios of the median of three five-round process
medians, not confidence intervals. A ratio above one is faster.

| Public workload | Clang 17.0.6 | AppleClang 21.0.0 |
| --- | ---: | ---: |
| static double 3 × 3 | 1.090× | 1.034× |
| static double 4 × 4 | 1.110× | 0.996× |
| static double 10 × 10 | 1.179× | 1.119× |
| double 128 × 128, row-major | 4.117× | 4.091× |
| double 128 × 128, column-major | 4.113× | 4.078× |
| float 128 × 128, row-major | 8.503× | 8.520× |
| double 65 × 97 times 97 × 33 | 2.713× | 2.797× |
| double 96 × 257 times 257 × 17 | 2.248× | 2.212× |
| double 128 × 128, RCR fallback | 0.995× | 0.984× |
| double 128 × 128, CRC assignment | 3.556× | 3.531× |
| unaligned double views 65 × 97 times 97 × 33 | 2.674× | 2.778× |

RCR and CRC list left/right/destination layouts. Public assignment materializes
in the product layout before copying to the destination. The RCR case retains
coefficient evaluation; no speedup is claimed for that fallback. Static FEM
products show no material regression in these runs, including the AppleClang
4 × 4 ratio of 0.996. These results characterize this machine and these warm
workloads, not all platforms or cache regimes.

Both AppleClang 21 and GCC 15 passed 114 registered tests and public-header
checks; 16 intended compile failures remain verified. The six new product tests
also passed ASan and UBSan with debug assertions enabled. GCC used the same
temporary macOS SDK compatibility flag as the assignment pilot.

## Controlled size sweep

`tests/benchmarks/simd_sweep.cpp` measures public operations and verifies every
coefficient outside the timed region, including release builds. The standard-library
runner builds the **same source** four ways: neither optimization, assignments only,
products only, and both. Compilation finishes before timing starts. No external
BLAS or benchmarking package is used.

```sh
python3 tests/benchmarks/run_simd_sweep.py --self-test
python3 tests/benchmarks/run_simd_sweep.py \
  --compiler /usr/bin/clang++ --label appleclang --output output/simd/sweep/appleclang \
  --build-only
python3 tests/benchmarks/run_simd_sweep.py \
  --compiler /usr/bin/clang++ --label appleclang --output output/simd/sweep/appleclang \
  --reuse-binaries --factorial
Rscript tests/benchmarks/plot_simd_sweep.R \
  output/simd/sweep/appleclang/summary.csv output/simd/sweep/appleclang/plots
```

Assignment cases isolate `off:assignment`. Product cases isolate `assignment:all`,
keeping the final copy path enabled on both sides. `--factorial` also compares
`off:product` and `off:all` at the smallest, nearest-to-128 and largest actually
measured product sizes. These extra ratios include the corresponding copy costs
and must not be pooled with the isolated product ratios.

Assignment `--size` means total output coefficients, with odd tails and equal
volumes for vectors, three-column row-major matrices, three-row column-major
matrices and unaligned external views. Product `--size` is the shape parameter;
use reported `rows`, `inner` and `cols` for rectangular cases. Static FEM cases
retain sizes 3, 4 and 10. Orientation and cross-layout assignment controls use
three anchor sizes. Inputs are nonconstant binary fractions; scaling alternates
powers of two to avoid drift. Public snapshots, copies and allocation are timed.

Each point has three process pairs with alternating order; each process reports
five timed rounds after warmup. One shared calibrated repetition count targets
25 ms for the faster variant, caps the slower variant at 250 ms and caps repetitions
at five million. A single large call can exceed that target. The reported ratio
is the median of the **three paired ratios**. The minimum/maximum range is observed
process dispersion, not a confidence interval. Raw rounds and exact build flags,
source hashes and binary hashes are retained. `--reuse-binaries` verifies the
manifest; `--resume` preserves completed cases and points after interruption.

A confirmed local plateau requires four increasing sizes: three points plus a
larger confirmation, all output buffers at least `--cache-bytes` (default 16 MiB),
central ratios within 5% and pair spreads within 10%. This is a reproducible local
criterion, not an assertion of asymptotic or hardware-independent performance.
The runner records an explicit stop reason when the schedule, predicted/measured
per-call limit or process timeout is reached without confirmation. Cache regimes,
short rounds and regressions must remain visible in the report. Only operand
storage is included in `working_set_bytes`; internal temporaries and the untimed
oracle can consume additional memory.
