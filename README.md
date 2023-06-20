<div align="center"> <h1> fdaPDE </h1>

<h5> Bring the expressiveness power of PDE into statistics </h5> </div>

<div align="center">

![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/AlePalu/fdaPDE/test-workflow.yml?branch=develop&label=test-status)
[![Downloads](https://cranlogs.r-pkg.org/badges/fdaPDE)](https://cran.r-project.org/package=fdaPDE)

</div>


This is the rewriting repo of the fdaPDE project. Code here will be merged into mainstream when it is ready.

## Documentation
Documentation for both end-users and developers can be found on the [documentation site](https://alepalu.github.io/fdaPDE/)

## Dependencies
In order to be able to compile the C++ core of the library the following dependencies have to be installed on your system:
* a C++17 compliant compiler. (`gcc` versions higher than 7 should be fine)
* make
* CMake
* Eigen linear algebra library (version higher than 3.3)

## Installation (R development)
fdaPDE makes use of `Rcpp` and `RcppEigen` to interface the C++ core library. To install the package for development purposes execute the following from an R console 

```r
# install dependencies (execute this only once, if you not have Rcpp installed yet)
install.packages("Rcpp")
install.packages("RcppEigen")

# load Rcpp library
library(Rcpp)

setwd("/fdaPDE_root_folder/wrappers/R/")
# update RCppExports.cpp
compileAttributes(".")
# install fdaPDE
install.packages(".", type="source", repos=NULL)

# you might be required to exit from the current R session to load the new 
# installed version of the library in case you have executed library(fdaPDE) before
```

Compilation under `RcppEigen` produces a lot of annoying warnings. Those are kept hidden by `-Wno-ignored-attributes` flag in `wrappers/R/Makevars` file. Recall that CRAN policies doesn't allow to hide such kind of warnings.

> Compiling the package using roxigen2 or devtools causes any compiler optimization to be turned off. If you are compiling manually from source compile the package as described above to not incur in a huge performance loss. 

## C++ development
The C++ core of fdaPDE is extremely modular, this means that you can focus on single modules during development and work in isolation with respect to other modules. This avoid the need to recompile the whole codebase, e.g. from the R layer, at each change in the C++ core library, opening for faster development, better debugging and a direct interaction with the C++ layer.
To work on single modules is enought to put a `Main.cpp` file containing what you want to execute togheter with the following minimal `CMakeLists.txt` file in the module's folder:

```CMake
cmake_minimum_required(VERSION 3.14)

project(__MODULE_NAME__)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_FLAGS "-O3 -march=native -std=c++17")

find_package (Eigen3 3.3 REQUIRED NO_MODULE)

include_directories(.)

add_executable(Main Main.cpp)
target_link_libraries (Main Eigen3::Eigen)
```

Then just execute (from a terminal)

```Shell
cmake CMakeLists.txt
make
./Main
```

`Main.cpp` files as well as `CMakeLists.txt` files and any `cmake` releated file are all filtered out from the repo to keep the codebase clean.
