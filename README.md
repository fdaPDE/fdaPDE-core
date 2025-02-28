<div align="center"> <h1> fdaPDE </h1>

<h5> Physics-Informed Spatial and Functional Data Analysis </h5> </div>

![test-linux-gcc](https://img.shields.io/github/actions/workflow/status/fdaPDE/fdaPDE-core/test-linux-gcc.yml?branch=stable&label=test-linux-gcc)
![test-linux-clang](https://img.shields.io/github/actions/workflow/status/fdaPDE/fdaPDE-core/test-linux-clang.yml?branch=stable&label=test-linux-clang)
![test-macos-clang](https://img.shields.io/github/actions/workflow/status/fdaPDE/fdaPDE-core/test-macos-clang.yml?branch=stable&label=test-macos-clang)

This repository contains the C++, header-only, core library system for the fdaPDE project, providing basic functionalities like a finite element solver for second-order linear elliptic boundary value problems, nonlinear unconstrained optimization algorithms, linear and non-linear system solvers, multithreading support, and more.

## Documentation
Documentation can be found on our [documentation site](https://fdapde.github.io/)

## Dependencies

To compile and run the \texttt{fdapde} library and the full workflow defined in the provided \texttt{Makefile}, the following dependencies must be installed on the system:
- a C++20-compliant compiler, 
- make, 
- the Eigen library (version 3.4),
- the nlohmann/json header,
- Python 3, together with the following Python packages: numpy, pandas, matplotlib,
- R, in case Docker is not used, along with the \texttt{fmesher} package, installable via

         install.packages("fmesher", repos = "https://cloud.r-project.org")
- alternatively, a working installation of Docker, to run R-based scripts in an isolated container.

---

## Usage Instructions

Different functionalities can be accessed by running the corresponding commands from the directory created after cloning the project repository.

To enable this, it is necessary to edit the initial Makefile variables to reflect the correct local paths on your system.


### Plot different domains' meshes

Run the command:

      make delaunay


To change the domain, modify the call in Meshes/Test_fdaPDEmesher/main.cpp with a shape in domains.h:

      Delaunay<2, 2> del(std::vector<Eigen::Matrix<double, Eigen::Dynamic,2>> {ext_bd, region1, ...}, min_angle, max_area, N, holes);



### Evaluate computational cost of Delaunay's conflict graph algorithm and refinement algorithm

      make conflict
      make refinement


### Plot domains' meshes through triangle.c, and evaluate its refinement algorithm

      make triangle


To change the domains, in Makefile change the name of $(TRIANGLE_NAME) into star, letter_A or skyline. It’s also possible to evaluate other shapes by creating the appropriate .poly file.


### Plot domains' meshes through fmesher, and evaluate its refinement algorithm

      make fmesher


To change the domains, in Makefile change the name of $(FMESHER_SCRIPT) to the appropriate R file.

The fmesher target runs an R environment inside a Docker container (rocker/geospatial image), allowing the installation of the fmesher package and execution of R scripts in a clean and reproducible way, since fmesher may have system-level dependencies that are hard to set up manually.

If R is installed locally, the user can bypass the Docker container and execute the R scripts directly:

      install.packages("fmesher", repos = "https://cloud.r-project.org")
      Rscript Meshes/Test_fmesher/star.r
      Rscript Meshes/Test_fmesher/rectangle_timing.r


###  To solve a PDE predicting the density of wildifire spread in Southern Italy, utilizing all the features of the fdaPDE-cpp library:

      make solve_PDE

 A Docker environment is used to ensure that all the correct dependencies required to run the entire fdaPDE library are properly installed. Otherwise, it is possible to run directly the commands for the simulation by running:

      cd workingdir/
      cpp=$(pwd)/fdaPDE-cpp
      core=$cpp/fdaPDE/core
      cd $(pwd)/test
      g++ -w -o script depde_fire.cpp -I$cpp -I$core -I/usr/include/eigen3 -O2 -std=c++20 -march=native -s
      ./script


### Clean the directories from object files and csv files 

      make clean