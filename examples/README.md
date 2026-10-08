# Runnable examples

Each `.cpp` is a standalone C++20 program using public fdaPDE headers. The
examples use small deterministic data, describe the mathematical coordinates of
their inputs, and keep the computation in the source file. Numerical tutorials
check their results against known values, adjoint identities or finite
differences; the short batch API walkthroughs show the available operations.

Sources are grouped by their main purpose:

- `linear_algebra/`: matrix batch arithmetic and spectral operations
- `manifold/`: metrics, maps, weighted means and product optimization
- `geometric_finite_elements/`: P1 interpolation, tensor fields and spatial energies
- `benchmarks/`: cache and prepared-interpolation timing comparisons

File names describe the demonstrated operation. CMake targets and executables
use the source basename, without its directory or `.cpp` extension. For example,
`manifold/so_maps_and_branches.cpp` builds the `so_maps_and_branches` target.

## Build and run

From the repository root, build the native examples without Eigen or GoogleTest:

```sh
cmake -S examples -B build/examples -DCMAKE_BUILD_TYPE=Release
cmake --build build/examples --parallel 4
./build/examples/so_product_optimization
ctest --test-dir build/examples --output-on-failure
```

To build one program, use `cmake --build build/examples --target so_maps_and_branches`.
On multi-configuration generators, add `--config Release` to the build command
and `-C Release` to CTest; executables are in the `Release` subdirectory.

The two mesh/FEM examples also require Eigen 3.4:

```sh
cmake -S examples -B build/examples -DCMAKE_BUILD_TYPE=Release \
  -DFDAPDE_EXAMPLES_WITH_FEM=ON -DEigen3_DIR=/path/to/eigen3/cmake
cmake --build build/examples --parallel 4
ctest --test-dir build/examples --output-on-failure
```

Omit `Eigen3_DIR` when CMake can already locate Eigen. No dependency is downloaded.
Programs can also be compiled directly, as described at the top of each source:

```sh
c++ -std=c++20 -O2 -pthread -I. examples/manifold/so_maps_and_branches.cpp -o /tmp/so_maps_and_branches
/tmp/so_maps_and_branches
```

## Choose an example

The table maps the public workflows to runnable code. It covers the main use
cases rather than every overload, cache policy or invalid-input combination;
those contracts are exercised by the [test suite](../tests/README.md).

| Example | Public API and workflow | Dependencies |
|---|---|---|
| [linear_algebra/matrix_batch_arithmetic.cpp](linear_algebra/matrix_batch_arithmetic.cpp) | Lazy batch arithmetic, broadcasting, `eval`, `assign`, `map`, `select` and reductions | Native |
| [linear_algebra/spd_batch_operations.cpp](linear_algebra/spd_batch_operations.cpp) | SPD batch logarithms, exponentials, roots, inverses, eigenpairs and cache policies | Native |
| [manifold/spd_metrics_and_means.cpp](manifold/spd_metrics_and_means.cpp) | LE, AIRM, BW, log-Cholesky and Cheeger metrics, log/exp, transport and weighted means | Native |
| [manifold/so_maps_and_branches.cpp](manifold/so_maps_and_branches.cpp) | Body and ambient tangents, SO exp/log, metric, transport and explicit logarithm branches | Native |
| [manifold/so_product_optimization.cpp](manifold/so_product_optimization.cpp) | Packed skew batches, coupled rotation objective, ambient gradient/Hessian, steepest descent and trust region | Native |
| [geometric_finite_elements/so_p1_interpolation.cpp](geometric_finite_elements/so_p1_interpolation.cpp) | Native P1 rotation values, weight/nodal and mixed covariant derivatives, metric adjoints | Native |
| [geometric_finite_elements/cheeger_p1_interpolation.cpp](geometric_finite_elements/cheeger_p1_interpolation.cpp) | Native Cheeger P1 interpolation, nodal rho, scalar sensitivity and convergence diagnostics | Native |
| [geometric_finite_elements/spd_discrete_tension.cpp](geometric_finite_elements/spd_discrete_tension.cpp) | Prepared discrete-tension stencils, values and gradients, sequential/parallel evaluation and Cheeger rho covectors | Native |
| [geometric_finite_elements/spd_p1_field.cpp](geometric_finite_elements/spd_p1_field.cpp) | Mesh, `GeometricSpace`/`GeometricFunction`, prepared evaluation, coefficient updates, spatial/nodal derivatives and Dirichlet energy | Eigen |
| [geometric_finite_elements/spd_field_estimation.cpp](geometric_finite_elements/spd_field_estimation.cpp) | Product optimization of a tensor field in symmetric log coordinates and SPD coordinates, then spatial prediction | Eigen |
| [geometric_discretization/01_spd_spline_field.cpp](geometric_discretization/01_spd_spline_field.cpp) | `BsSpace`, shared `GeometricSpace`/`GeometricFunction`, prepared evaluation, spatial derivatives and control coefficient updates | Eigen |
| [benchmarks/spd_airm_interpolation.cpp](benchmarks/spd_airm_interpolation.cpp) | AIRM geodesic sampling, prepared interpolation and input/output cache choices | Native, timing comparison |
| [benchmarks/spd_log_euclidean_interpolation.cpp](benchmarks/spd_log_euclidean_interpolation.cpp) | LE geodesic sampling, prepared interpolation and input/output cache choices | Native, timing comparison |
| [benchmarks/matrix_function_caches.cpp](benchmarks/matrix_function_caches.cpp) | Rotation logarithm caches, symmetric spectral caches and prepared rotation curves | Native, timing comparison |

Start with the two batch walkthroughs, then the geometry tutorials. Continue with
product optimization or interpolation and smoothing according to the intended
application. The mesh examples combine these components into a spatial field.
Timing comparisons report observations for the current compiler and machine;
their numerical checks, rather than speed ratios, determine success.
Run timing programs individually when collecting performance measurements.

## Related contracts

- [Geometry, solver and mean public headers](../docs/manifold-modules.md)
- [Batch storage, arithmetic, ownership and execution](../docs/matrix-batch.md)
- [SPD geometry and objective derivatives](../docs/spd-geometry.md)
- [SO geometry, tangent conventions and branches](../docs/so-geometry.md)
- [SPD spatial interpolation and derivatives](../docs/spd-interpolation.md)
- [Rotation interpolation and derivative certificates](../docs/rotation-interpolation.md)
- [Cheeger interpolation and rho derivatives](../docs/cheeger-interpolation.md)
- [Smoothing energies, prepared stencils and gradients](../docs/spd-smoothing.md)
