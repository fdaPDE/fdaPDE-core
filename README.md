<div align="center"> <h1> fdaPDE </h1>

<h5> Physics-Informed Spatial and Functional Data Analysis </h5> </div>

![test-linux-gcc](https://img.shields.io/github/actions/workflow/status/fdaPDE/fdaPDE-core/test-workflow.yml?branch=develop&label=test-linux-gcc)


This repository contains the C++, header-only, core library system for the fdaPDE library, providing basic functionalities like a finite element solver for second-order linear elliptic boundary value problems, nonlinear unconstrained optimization algorithms, linear and non-linear system solvers, multithreading support, and more.

## Documentation
Documentation for both end-users and developers can be found on our [documentation site](https://alepalu.github.io/fdaPDE/)

## Dependencies
fdaPDE-core is an header-only library, therefore it does not require any installation. Neverthless, compiled code including the core library must satisfy the following dependencies:
* C++17 compliant compiler
* [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page) linear algebra library (version 3.3.9)
