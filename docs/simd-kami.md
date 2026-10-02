# SIMD validation on Kami

`tests/benchmarks/kami_simd.sh` prepares and submits a PBS Pro job that runs the
native test suite and the complete SIMD size sweep. It compares the four loop
configurations from the same source and compiler, then produces a summary.
This guide describes the workflow; it contains no measurements from Kami.

## Prerequisites

Use the Kami login endpoint:

```sh
ssh donelli@kami.inside.mate.polimi.it
```

The environment needs a C++ compiler and standard library supporting the current
C++20 code, CMake 3.20 or later, Python 3.9 or later, Git, and Linux `taskset`.
PBS commands `qsub`, `qstat`, and `qdel` are needed for submission and monitoring. Choose the compiler through `CXX`;
use the compiler setup already available on Kami rather than assuming a module
name. `CXX` must identify an executable, not a compiler command containing flags.
The launcher first sources `~/kami-vars.sh`, then `~/kami-load.sh`, when readable,
for preparation, submission and execution inside PBS. This initializes the site
toolchain before Python, `CXX`, CMake or other programs are resolved. The profiles
receive no launcher arguments; optional unset variables are permitted while they
load. `SIMD_KAMI_ENV_DIR` can select another directory containing these two files.
The script then resolves and records the compiler path and version. Submission
checks `cmake`, `ctest`, `taskset` and `git` before calling `qsub`; a missing program
is named explicitly both on the login and in the compute-node log.

The native lane does not require BLAS or Eigen. It builds every test registered
by `tests/CMakeLists.txt` with `FDAPDE_NATIVE_ONLY=ON`; Eigen-dependent caller and
aggregate-header checks are excluded. This lane also rejects an environment in
which `Eigen/Eigen` is discoverable. The test dependency is the existing pinned
GoogleTest revision `f8d7d77c06936315286eb55f8de22cd23c188571`, prepared on the login
node before submission so the compute job does not need to download it.

`Rscript` is optional. If it is already installed and its graphics devices are
available, the job can produce plots using base R. It does not install R or any
packages, and missing R does not prevent tests, measurements, or the text/CSV
summary.

## Checkout and compiler

Publish the local `develop-SIMD` changes before updating the Kami checkout:

```sh
git push origin develop-SIMD
```

For a new checkout:

```sh
git clone --branch develop-SIMD https://github.com/fdaPDE/fdaPDE-core.git
cd fdaPDE-core
```

For an existing checkout, preserve local work and update the branch with a
fast-forward:

```sh
git fetch origin develop-SIMD
git switch develop-SIMD
git pull --ff-only origin develop-SIMD
git status --short
```

If the local branch does not exist yet, use
`git switch --track origin/develop-SIMD` instead of the `git switch` command above.
The default compiler is `g++`, resolved after loading the Kami profiles. To select
another installed compiler, set its executable name or absolute path before
submission:

```sh
export CXX=g++
```

For interactive version checks, first source `~/kami-vars.sh` and `~/kami-load.sh`
in the same order. The launcher performs this loading automatically.

## Prepare and submit

From the repository root on the login node:

```sh
bash tests/benchmarks/kami_simd.sh prepare
bash tests/benchmarks/kami_simd.sh submit
```

`prepare` fetches the pinned GoogleTest source into
`output/simd/kami/deps`; it does not compile or run benchmarks. If that source
is already available, set `FDAPDE_GTEST_SOURCE` to its absolute source directory
before preparing and submitting. Use the same pinned revision. The compute job
passes it to CMake through `FETCHCONTENT_SOURCE_DIR_GOOGLETEST` and builds offline.

`submit` creates a unique output directory named with UTC time and the short Git
revision below `output/simd/kami/`, generates the PBS job, and prints the submitted
job identifier and output path. Keep those values for monitoring and retrieval.
The worker uses the same checkout when the job starts and rejects a `HEAD` that
differs from the commit recorded at submission. Keep benchmark sources and
headers unchanged while the job is queued or running, including uncommitted
changes: the build manifest checks their hashes before timing but does not freeze
the checkout. Generated output directories can be written normally.
An explicit, unused output directory can be supplied:

```sh
bash tests/benchmarks/kami_simd.sh submit output/simd/kami/my-run
```

The worker entry point is `run OUTPUT_DIR`. It is intended for the generated
PBS job and checks `PBS_JOBID`; do not launch it on the login node or set a fake
job identifier to bypass that guard.

Resource and sweep settings can be exported before submission:

| Variable | Default / role |
| --- | --- |
| `CXX` | compiler executable selected after loading the profiles; `g++` if unset |
| `SIMD_KAMI_ENV_DIR` | directory containing `kami-vars.sh` and `kami-load.sh`; user home by default |
| `FDAPDE_GTEST_SOURCE` | optional existing source directory for the pinned GoogleTest revision |
| `SIMD_QUEUE` | `test` |
| `SIMD_CPUS` | `4` allocated CPUs |
| `SIMD_MEM` | `32gb` |
| `SIMD_WALLTIME` | `72:00:00` |
| `SIMD_PAIRS` | `3` process pairs |
| `SIMD_ROUNDS` | `5` timed rounds per ordinary process |
| `SIMD_LARGE_ROUNDS` | `1` timed round for large calls |
| `SIMD_MAX_CALL_SECONDS` | `60` per-call stopping limit |
| `SIMD_TIMEOUT` | `900` seconds per benchmark process |
| `SIMD_CACHE_BYTES` | largest cache size reported by CPU sysfs, or 16 MiB when unavailable |
| `SIMD_ASSIGNMENT_SIZES` | comma-separated assignment coefficient schedule, from 9 to 201326595 by default |
| `SIMD_PRODUCT_SIZES` | comma-separated product shape-parameter schedule, from 3 to 8193 by default |

For example, preserve the compiler choice and make the requested resource policy
explicit:

```sh
export SIMD_QUEUE=test SIMD_CPUS=4 SIMD_MEM=32gb SIMD_WALLTIME=72:00:00
bash tests/benchmarks/kami_simd.sh submit
```

Changing a size schedule or limit produces a different campaign. Keep those
settings with the output metadata when comparing runs.

## Execution policy

The submission uses the group PBS Pro resource convention:

```text
select=1:ncpus=4:mem=32gb
place=excl
queue=test
walltime=72:00:00
```

These are PBS Pro `select`/`place` requests. Do not replace them with Torque's
`nodes=...:ppn=...` syntax. Four CPUs are allocated for the job, while benchmark
processes run one at a time and use one thread. All compilation finishes before
timing starts. On Linux, `taskset` pins timing to the first CPU in the job's
allowed CPU set; the allocation is not inferred from the number of lines in
`PBS_NODEFILE`, which lists nodes on Kami.

The script reserves the node exclusively, but the observations still depend on
that node's CPU, compiler, cache hierarchy, operating-system activity, and input
shapes. Do not combine timings from different jobs or machines into one paired
ratio.

## Validation before measurement

The launcher bootstrap can be checked locally without downloads, compilation or
PBS submission:

```sh
python3 tests/benchmarks/check_kami_environment.py
```

This checks the profile order, argument isolation, tool resolution and explicit
missing-tool diagnostics using a temporary source cache and scheduler stubs.

The worker builds and runs the complete native CTest lane in four configurations:
`A0P0`, `A1P0`, `A0P1`, and `A1P1`. Debug assertions remain enabled for these tests.
It also runs the 11 focused assignment/product tests with AddressSanitizer and
UndefinedBehaviorSanitizer in `A1P1`. Benchmark binaries use optimized release
flags and disable debug assertions; their independent coefficient checks remain
active.

Four binaries are built from the same sweep source before timing begins. All 37
published cases receive a correctness smoke check. The complete size sweeps then
run serially, with no compiler or test jobs running concurrently. No ISA-specific
compiler flags, fast-math options, new BLAS backend, or benchmark dependency are
added by default. Compiler auto-vectorization remains enabled in every variant.

## Reading the results

Results are stored below `output/simd/kami/<run-tag>/` in the checkout. Keep the
recorded status, test logs, build metadata, raw measurements, CSV files, summary,
and any generated plots together. `hardware.json`, `job-metadata.json`, and
`job.log` describe the job; metadata records the current stage while the job is
running and its final completion or failure status. `assignment/` and `product/`
contain sweep datasets,
and `summary.md` reports the test status and benchmark comparisons. The recorded
Git revision and source/binary hashes identify the measured code; the plot or summary alone is not sufficient
provenance.

The primary assignment comparison is `A0P0 / A1P0`; the primary product comparison
is `A1P0 / A1P1`, keeping the final assignment path fixed. Factorial anchor points
also compare `A0P0 / A0P1` and `A0P0 / A1P1` in the product sweep, at the
smallest, nearest-to-128, and largest actually measured sizes. A ratio above one means the second
configuration was faster. OFF preserves compiler auto-vectorization; it disables
the corresponding native traversal choices.

With the default settings, each point uses three alternating process pairs and
each ordinary process has five timed rounds; large calls use one timed round when the slower calibration
call exceeds 250 ms. The CSV records the actual count. The central statistic is
the median of the three paired ratios; their minimum and maximum describe observed
dispersion, not a confidence interval. A missing or incomplete factorial anchor
must remain visible.

The sweep covers all published assignment and product cases, including float,
layout controls, unaligned views, rectangular products, and fixed FEM sizes.
Assignment size is the number of output coefficients; product size is a shape
parameter, so consult the reported `rows`, `inner`, and `cols`.

A confirmed plateau is a local criterion across four increasing sizes with
sufficiently large output buffers and stable ratios. The worker uses the largest
cache size reported by CPU sysfs for the output-buffer threshold, unless `SIMD_CACHE_BYTES` overrides it. The fallback is
16 MiB; this is a protocol parameter and does not establish which cache or CPU
frequency a benchmark actually used. A schedule, per-call limit, or process timeout can stop a case without observing a plateau.
The Kami wrapper defaults to a 60-second per-call limit and a 900-second process
timeout; these override the shorter standalone runner defaults. Read the recorded
stop reason rather than assuming every family reached a
plateau. The reported operand-storage estimate excludes internal temporaries and
the untimed product oracle. See [the sweep protocol](simd.md#controlled-size-sweep)
for the exact stability and verification rules.

## Regenerating the summary

The worker generates `summary.md` at normal completion and when a trapped error
or termination permits cleanup. It includes every measured size for each case
and comparison, test status, shape/type, timings in milliseconds, paired ratio,
min/max dispersion, actual rounds, stop reasons, and provenance problems. Missing
results remain missing; a partial job does not imply unexecuted tests passed.

If the scheduler terminates the worker before cleanup, or a report must be
regenerated after copying results, use the recorded datasets and job metadata:

```sh
kami_run=output/simd/kami/my-run
python3 tests/benchmarks/summarize_simd_sweep.py \
  "$kami_run/assignment" "$kami_run/product" \
  --job-metadata "$kami_run/job-metadata.json" --output "$kami_run/summary.md"
```

This reads existing files and does not rerun tests or benchmarks. The helper
records missing or malformed datasets in the report, rather than inventing
cluster hardware or measurements. Check `pbs.log`, `job.log`, and the last
recorded stage when completion metadata is absent or still says `running`.

## Monitoring and copying results

Inspect a submitted job with:

```sh
qstat -f <job-id>
qstat -Qf test
```

`qdel <job-id>` cancels a submitted job. Queue eligibility and wait time are
scheduler decisions; a node marked `free` need not have zero assigned CPUs.

From the local checkout, copy the entire run directory without deleting local
files. Replace the remote repository path and run tag with the actual values:

```sh
rsync -avh --partial --info=progress2 \
  donelli@kami.inside.mate.polimi.it:/absolute/path/to/fdaPDE-core/output/simd/kami/<run-tag>/ \
  output/simd/kami/<run-tag>/
```

Shell placeholders such as `<run-tag>` and `<job-id>` must be replaced before
running these commands.
