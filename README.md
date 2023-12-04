<div align="center"> <h1> fdaPDE </h1>

<h5> Physics-Informed Spatial and Functional Data Analysis </h5> </div>

![test-linux-gcc](https://img.shields.io/github/actions/workflow/status/fdaPDE/fdaPDE-core/test-linux-gcc.yml?branch=stable&label=test-linux-gcc)
![test-linux-clang](https://img.shields.io/github/actions/workflow/status/fdaPDE/fdaPDE-core/test-linux-clang.yml?branch=stable&label=test-linux-clang)

This repository contains the C++, header-only, core library system for the fdaPDE project, providing basic functionalities like a finite element solver for second-order linear elliptic boundary value problems, nonlinear unconstrained optimization algorithms, linear and non-linear system solvers, multithreading support, and more.

## Documentation
Documentation can be found on our [documentation site](https://fdapde.github.io/)

## Dependencies
fdaPDE-core is an header-only library, therefore it does not require any installation. Just make sure to have it in your include path. Neverthless, compiled code including this library must satisfy the following dependencies:
* C++17 compiler. Supported versions are:
     * Linux: `gcc` 11.2.1 (or higher), `clang` 13.0.1 (or higher)
* [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page) (3.3.9) library

If you wish to run the test suite contained in the `test/` folder, be sure to have [Google Test](http://google.github.io/googletest/) installed. 
