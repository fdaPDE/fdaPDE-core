<div align="center"> <h1> fdaPDE: High-Order PDEs on Surfaces via Isogeometric Analysis </h1>

<h5> fdaPDE - Physics-Informed Spatial and Functional Data Analysis </h5> </div>

This repository is a fork of the fdaPDE-core C++ header only library system for the fdaPDE project. The main purpose of this fork is the developement of the final project for the PACS course, held by professor Luca Formaggia at Politecnico di Milano. This project was developed by Massimo Rizzuto under the supervision of professors Laura M. Sangalli, Eleonora Arnone and doctor Alessandro Palummo.
The main contribution to the repository is the implementation of the IsoGeometric Analysis discretization methods. Most of the code is located in the following folders:
- `fdaPDE/src/fields`: contains the implementation of the `spline.h` and `nurbs.h`, which implement single B-Spline and NURBS fields, respectively. 
- `fdaPDE/src/splines`: contains the implementation of Spline utilities.
- `fdaPDE-core/src/nurbs`: contains the implementation of NURBS utilities.
- `fdaPDE-core/src/geometry`: contains the implementation of the `isomesh.h`, the main class for the IGA discretization method.
- `fdaPDE-core/src/isogeometric`: contains the implementation of the PDE solver for the IGA method.

## Documentation
The official documentation of fdaPDE, still under development, can be found on [documentation site](https://fdapde.github.io/).

## Dependencies
**fdaPDE** is an header-only library, therefore it does not require any installation. Just make sure to have it in your include path. Nevertheless, the following dependencies are required to compile the code and run the simulations:

- **GCC/G++-compliant compiler**  
  Use `g++14` or newer. On Ubuntu or Debian-based systems, you can install it with: `sudo apt install g++`. For other platforms, refer to the official [GCC installation guide](https://gcc.gnu.org/install/).

- **make**  
  A build automation tool. Install it on Ubuntu/Debian with: `sudo apt install make`. For other platforms, refer to the official [GNU Make installation guide](https://www.gnu.org/software/make/).

- **CMake**
  A cross-platform build system generator. Install it on Ubuntu/Debian with: `sudo apt install cmake`. For other platforms, refer to the official [CMake installation guide](https://cmake.org/install/).

- **Eigen3** linear algebra library  
  Version 3.3 or newer (we are using 3.3.9). It can be installed via `sudo apt install libeigen3-dev` on Ubuntu or Debian-based systems. For other platforms, refer to the official [Eigen3 installation guide](https://eigen.tuxfamily.org/dox/GettingStarted.html).

- **gtest** (*optional*)  
  Only needed to run the library tests. For Ubuntu or Debian-based systems, it can be installed via `sudo apt install libgtest-dev`. For other platforms, refer to the official [GoogleTest repository](https://github.com/google/googletest).

- **asymptote** (*optional*)  
  Used for vector-based visualization of simulation results. It can be installed via `sudo apt install asymptote` on Ubuntu or Debian-based systems. For other systems, please refer to the [Asymptote installation guide](https://asymptote.sourceforge.io/).

## How to run the code

- **Library tests**
  All the tests for our IGA framework can be run directly from shell:
    
  ```bash
  #!/bin/bash
  cd iso_tests
  ./run_tests.sh
  ```
  Note that maybe the compiler should be set properly in the CMake file or the shell file if you are not using the default one. 
- **Simulations**
    All simulations presented in Chapter 4 of our report can be run directly from shell, taking as an example the square laplacian simulation, with the following commands:
    
    ```bash
    #!/bin/bash
    cd simulations
    cd quarter_ring_diff/
    ./build.sh
    ```

    To run other simulations, simply substitute line 3 with the directory associated with the desired simulation. The possible options are:
    * quarter_ring_diff,
    * quarter_ring_advdiff,
    * square_mixed_bc,
    * sphere_diff,
    * torus_diff,
    * quarter_ring_biharmonic,
    * sphere_biharmonic,
    

    To visualize plots, execute the following command in the simulation folder. With `asymptote`, we generate three kinds of plots: the computational geometry, the solution, and the error. All plots are saved as high-quality `.png` files. Here an example of how to plot the results once the simulation is completed:
    
    ```bash
    #!/bin/bash
    asy plot_solution.asy -u 3
    asy plot_mesh.asy -u 3
    asy plot_error.asy 
    ```
    The `-u <ref>` option is used to set the refinement level of the mesh. The default value is 3, but it can be changed to any integer value to increase or decrease the resolution of the plot. Note that the corresponding refinement must be set in the source files.

