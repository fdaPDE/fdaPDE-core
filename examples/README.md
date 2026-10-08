# Runnable examples

Each `.cpp` is a standalone C++20 program using public fdaPDE headers. Read the
source alongside its output: the algebra walkthroughs introduce a small problem,
explain the relevant types and operations, then print their results.

The numbers restart in each folder and suggest a reading order. Begin with
`linear_algebra/01`, continue through the structured matrices and decompositions,
then explore batches, manifold geometry and spatial fields. Each program can
also be compiled and run independently.

## Build and run

From the repository root, build the native examples without Eigen or GoogleTest:

```sh
cmake -S examples -B build/examples -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS='-Wall -Wextra -Wpedantic -Werror'
cmake --build build/examples --parallel 4
./build/examples/01_dense_storage_and_views
```

CMake targets and executables use the complete source basename, including its
number. To build just the first tutorial:

```sh
cmake --build build/examples --target 01_dense_storage_and_views
```

The sparse algebra example and two mesh/FEM examples require Eigen 3.4. Enable
both groups to build every program:

```sh
cmake -S examples -B build/examples -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS='-Wall -Wextra -Wpedantic -Werror' \
  -DFDAPDE_EXAMPLES_WITH_EIGEN=ON -DFDAPDE_EXAMPLES_WITH_FEM=ON \
  -DEigen3_DIR=/path/to/eigen3/cmake
cmake --build build/examples --parallel 4
./build/examples/11_eigen_sparse_blocks_and_products
```

Use only `FDAPDE_EXAMPLES_WITH_EIGEN=ON` for sparse algebra, or only
`FDAPDE_EXAMPLES_WITH_FEM=ON` for mesh/FEM. Both default to `OFF`. Omit
`Eigen3_DIR` when CMake already locates Eigen. No dependency is downloaded.

The compile command at the top of each source also works without CMake:

```sh
c++ -std=c++20 -O2 -pthread -I. \
  examples/linear_algebra/01_dense_storage_and_views.cpp -o /tmp/01_dense_storage_and_views
/tmp/01_dense_storage_and_views
```

To run every enabled program, use
`ctest --test-dir build/examples --output-on-failure`. With multi-configuration
generators, add `--config Release` to the build command and `-C Release` to
CTest; executables are in the `Release` subdirectory.

## Choose an example

### Linear algebra

Examples 01–10 use native fdaPDE types and need no Eigen. They include
`<fdaPDE/dense_linear_algebra.h>`. Example 11 uses `<fdaPDE/linear_algebra.h>`,
which adds Eigen sparse types and helpers.

| Order | Tutorial | What you learn |
|---|---|---|
| 01 | [Dense storage and views](linear_algebra/01_dense_storage_and_views.cpp) | Create a table with fixed or runtime dimensions, edit rows and blocks, borrow external memory and reshape storage |
| 02 | [Expressions and products](linear_algebra/02_dense_expressions_and_products.cpp) | Materialize lazy expressions, distinguish matrix and coefficient products, center observations and build Kronecker products |
| 03 | [Diagonal and triangular solves](linear_algebra/03_diagonal_and_triangular_solve.cpp) | Use packed storage and solve one or several loads with diagonal and triangular operators |
| 04 | [Symmetric and skew storage](linear_algebra/04_symmetric_and_skew_storage.cpp) | Supply independent coordinates, edit structured views and split a general matrix into symmetric and skew parts |
| 05 | [Orthogonal matrices, permutations and rotations](linear_algebra/05_orthogonal_permutation_and_rotation.cpp) | Change basis, reorder observations and compose planar rotations |
| 06 | [Dense factorizations](linear_algebra/06_dense_factorizations.cpp) | Use LU, QR, EVD and Cholesky algorithm classes, solve systems and inspect factor types |
| 07 | [SPD spectral functions and caches](linear_algebra/07_spd_spectral_and_caches.cpp) | Access structured factors through methods, reuse caches and compute matrix logarithms, roots and inverses |
| 08 | [Boolean and multidimensional storage](linear_algebra/08_boolean_and_multidimensional_storage.cpp) | Select readings with packed masks, slice experiment/sensor/time data and map an external buffer |
| 09 | [Matrix batch arithmetic](linear_algebra/09_matrix_batch_arithmetic.cpp) | Apply lazy arithmetic, broadcasting, maps, selections and reductions to collections of matrices |
| 10 | [SPD batch operations](linear_algebra/10_spd_batch_operations.cpp) | Apply spectral functions and cache policies to entire SPD batches |
| 11 | [Eigen sparse blocks and products](linear_algebra/11_eigen_sparse_blocks_and_products.cpp) | Assemble sparse block operators, add couplings, form Kronecker products, lump rows and solve with Eigen |

### Manifold geometry

Continue here to use structured matrices as points on a manifold. All three
programs use native fdaPDE types.

| Order | Tutorial | What you learn |
|---|---|---|
| 01 | [SPD metrics and means](manifold/01_spd_metrics_and_means.cpp) | Choose LE, AIRM, BW, log-Cholesky or Cheeger geometry and compute maps, transport and weighted means |
| 02 | [SO maps and branches](manifold/02_so_maps_and_branches.cpp) | Work with rotation tangents, exp/log maps, transport and explicit logarithm branches |
| 03 | [SO product optimization](manifold/03_so_product_optimization.cpp) | Optimize several rotations together with product geometry and trust-region methods |

### Geometric finite elements

Start with interpolation at one point, then move to spatial fields and fitting.
Examples 04 and 05 require `FDAPDE_EXAMPLES_WITH_FEM=ON` and Eigen.

| Order | Tutorial | What you learn |
|---|---|---|
| 01 | [SO P1 interpolation](geometric_finite_elements/01_so_p1_interpolation.cpp) | Interpolate nodal rotations and use spatial and nodal differentials |
| 02 | [Cheeger P1 interpolation](geometric_finite_elements/02_cheeger_p1_interpolation.cpp) | Interpolate SPD values with Cheeger geometry and differentiate with respect to its parameter |
| 03 | [SPD discrete tension](geometric_finite_elements/03_spd_discrete_tension.cpp) | Evaluate spatial smoothing energies and their gradients |
| 04 | [SPD P1 field](geometric_finite_elements/04_spd_p1_field.cpp) | Build a mesh-based SPD field, evaluate it and update its nodal values |
| 05 | [SPD field estimation](geometric_finite_elements/05_spd_field_estimation.cpp) | Fit symmetric and SPD fields with trust-region optimization |

### Geometric splines

The same `GeometricSpace` and `GeometricFunction` interface accepts `BsSpace`.
This tutorial requires `FDAPDE_EXAMPLES_WITH_FEM=ON` and Eigen.

| Order | Tutorial | What you learn |
|---|---|---|
| 01 | [SPD spline field](geometric_discretization/01_spd_spline_field.cpp) | Build geometric control coefficients, reuse prepared basis weights, differentiate and update an SPD-valued B-spline field |

### Timing comparisons

After the relevant tutorials, use these programs to explore cache and prepared
interpolation choices. Run them individually when collecting timings; the
reported times depend on the compiler and machine.

| Order | Program | What it compares |
|---|---|---|
| 01 | [Matrix function caches](benchmarks/01_matrix_function_caches.cpp) | Rotation logarithms, symmetric spectral caches and prepared rotation curves |
| 02 | [Log-Euclidean interpolation](benchmarks/02_spd_log_euclidean_interpolation.cpp) | Direct and prepared LE interpolation with different input/output caches |
| 03 | [Affine-invariant interpolation](benchmarks/03_spd_airm_interpolation.cpp) | Direct and prepared AIRM interpolation with different input/output caches |

## Keep in mind while experimenting

- Views and lazy expressions borrow their operands. Keep owners alive and avoid
  resizing their storage while using views. Construct or assign a `Matrix` to
  save a dense expression; batch expressions also provide `eval()` and `assign()`.
- Structured types encode different mathematical constraints. Symmetric writes
  mirror, skew writes preserve signs, and orthogonal/SPD/rotation views replace complete
  valid values. Packed triangular, symmetric, skew and SPD storage is row-major.
- QR solves full-column-rank square or overdetermined systems; LU solves square systems. Eigenvalues are
  paired with eigenvector columns and are not sorted automatically.
- Cache policies choose retained quantities. Cached matrices borrow their owner;
  request them again after replacing its value. Symmetric proxy/view edits
  invalidate cached eigenpairs.

For interface details, see [dense algebra](../docs/dense-algebra.md),
[spectral caches](../docs/symmetric-cache.md) and
[matrix batches](../docs/matrix-batch.md). Regression coverage belongs to the
[test suite](../tests/README.md).

The Eigen aggregate also exposes Woodbury solvers, FSPAI and randomized
factorizations; these need separate walkthroughs and are not covered here.
