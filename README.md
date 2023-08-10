<div align="center"> <h1> fdaPDE </h1>

<h5> Physics-Informed Spatial and Functional Data Analysis </h5> </div>

![test-linux-gcc](https://img.shields.io/github/actions/workflow/status/fdaPDE/fdaPDE-core/test-workflow.yml?branch=stable&label=test-linux-gcc)

This repository contains the C++, header-only, core library system for the fdaPDE project, providing basic functionalities like a finite element solver for second-order linear elliptic boundary value problems, nonlinear unconstrained optimization algorithms, linear and non-linear system solvers, multithreading support, and more.

## Documentation
Documentation can be found on our [documentation site](https://fdapde.github.io/)

## Dependencies
fdaPDE-core is an header-only library, therefore it does not require any installation. Just make sure to have it in your include path. Neverthless, compiled code including the core library must satisfy the following dependencies:
* code must be compiled with C++17 support enabled
* [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page) linear algebra library (version 3.3.9)

If you wish to run the test suite contained in the `test/` folder, be sure to have [Google Test](http://google.github.io/googletest/) installed. 
