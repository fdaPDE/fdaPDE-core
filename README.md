<div align="center"> <h1> fdaPDE </h1>

<h5> Physics-Informed Spatial and Functional Data Analysis </h5> </div>

![test-linux-gcc](https://img.shields.io/github/actions/workflow/status/fdaPDE/fdaPDE-core/test-linux-gcc.yml?branch=stable&label=test-linux-gcc)
![test-linux-clang](https://img.shields.io/github/actions/workflow/status/fdaPDE/fdaPDE-core/test-linux-clang.yml?branch=stable&label=test-linux-clang)
![test-macos-clang](https://img.shields.io/github/actions/workflow/status/fdaPDE/fdaPDE-core/test-macos-clang.yml?branch=stable&label=test-macos-clang)

This repository contains the C++, header-only, core library system for the fdaPDE project, providing basic functionalities like a finite element solver for second-order linear elliptic boundary value problems, nonlinear unconstrained optimization algorithms, linear and non-linear system solvers, multithreading support, and more.

## Documentation
Documentation can be found on our [documentation site](https://fdapde.github.io/)

The [runnable examples](examples/README.md) cover native dense and structured matrices,
decompositions, spectral caches, matrix batches, Eigen sparse blocks, SPD and SO
geometries, manifold optimization, interpolation and spatial tensor fields.
Scripts are numbered within each folder to guide reading; the index explains
the learning path and standalone build commands.

## Dependencies
fdaPDE-core is a header-only library and does not require installation. Add the
repository root to your include path. To compile code using the library you need:
* A C++20 compliant compiler. Supported versions are:
     * Linux: `gcc` 11 (or higher), `clang` 15 (or higher)
	 * macOS: `apple-clang` (the XCode version of `clang`, AppleClang 15 or higher).
* [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page) 3.4 for
  `core.h`, `linear_algebra.h` and the spatial mesh/FEM modules.

The native `dense_linear_algebra.h`, [manifold aggregates](docs/manifold-modules.md) and
`geometric_finite_elements.h` aggregates do not require Eigen. The example build
enables sparse algebra with `FDAPDE_EXAMPLES_WITH_EIGEN=ON` and mesh/FEM
programs with `FDAPDE_EXAMPLES_WITH_FEM=ON`. Both options default to `OFF`.

If you wish to run the test suite contained in the `test/` folder, be sure to have [Google Test](http://google.github.io/googletest/) installed. 
