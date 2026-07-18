<div align="center"> <h1> fdaPDE </h1>

<h5> Physics-Informed Spatial and Functional Data Analysis </h5> </div>

![test-linux-gcc](https://img.shields.io/github/actions/workflow/status/fdaPDE/fdaPDE-core/test-linux-gcc.yml?branch=stable&label=test-linux-gcc)
![test-linux-clang](https://img.shields.io/github/actions/workflow/status/fdaPDE/fdaPDE-core/test-linux-clang.yml?branch=stable&label=test-linux-clang)
![test-macos-clang](https://img.shields.io/github/actions/workflow/status/fdaPDE/fdaPDE-core/test-macos-clang.yml?branch=stable&label=test-macos-clang)

This repository contains the C++, header-only, core library system for the fdaPDE project, providing basic functionalities like a finite element solver for second-order linear elliptic boundary value problems, nonlinear unconstrained optimization algorithms, linear and non-linear system solvers, multithreading support, and more.

## Documentation
Documentation can be found on our [documentation site](https://fdapde.github.io/)

## Discontinuous Galerkin

Scalar symmetric interior-penalty forms on planar triangular meshes can be written directly in the finite-element DSL:

```cpp
auto mesh = Triangulation<2, 2>::UnitSquare(17);
FeSpace space(mesh, DG<1, 1>);
TrialFunction u(space);
TestFunction v(space);

auto n = facet_normal(mesh);
auto h = facet_size(mesh);
constexpr double penalty = 10.0;

auto A = integral(mesh)(
  dot(grad(u), grad(v))
  - dot(avg(grad(u)), n) * jump(v)
  - dot(avg(grad(v)), n) * jump(u)
  + (penalty / h) * jump(u) * jump(v)
).assemble();
```

On an interior facet, `avg(q)` is `(q+ + q-) / 2`, `jump(q)` is `q+ - q-`, `n` points from the plus cell to the minus cell, and `h` is the facet length. The current interior-facet assembly path targets scalar forms over a complete two-dimensional triangular mesh. Boundary-facet and filtered-cell DG forms are not yet implemented; the C++ benchmark imposes homogeneous boundary trace degrees of freedom strongly. See the [manufactured Poisson benchmarks](test/benchmarks/README.md) for a discontinuous-source example and a FEniCSx reference.

## Dependencies
fdaPDE-core is an header-only library, therefore it does not require any installation. Just make sure to have it in your include path. Neverthless, to compile code including this library you need:
* A C++20 compliant compiler. Supported versions are:
     * Linux: `gcc` 11 (or higher), `clang` 15 (or higher)
	 * macOS: `apple-clang` (the XCode version of `clang`, AppleClang 15 or higher).
* The [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page) linear algebra library, version 3.4.0.

If you wish to run the test suite contained in the `test/` folder, be sure to have [Google Test](http://google.github.io/googletest/) installed. 
