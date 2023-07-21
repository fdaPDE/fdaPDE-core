### v 2.0 (dd mm 2023; first independent release)
---
First release
* Release numbering started at release number of `fdaPDE`
* Released as independent submodule after restructuring of `fdaPDE` version 1.1-9

**Major stable features**

1. Expression-template based arithmetic support for multivariate scalar, vector and matrix fields
2. Basic managment of triangulated domains 
    * import of third-party generated triangulations from .csv or .mtx files
	* basic interfacing with mesh elements
	* basic iterators
	* solution to point location problem (barycentric walking, alternating digital tree)
3. Finite Element discretization of linear second order elliptic differential equation
    * basic FEM infrastructure (pde interface, assembly loop, operators)
    * support for Lagrangian basis of any order on 1D, 2D and 3D spaces
	* various quadrature rules, exact for order 1 and 2 finite elements, and for 1D, 2D and 3D spaces
	* basic solver based on direct SparseLU factorization of system matrix
4. Data structures: BlockFrame, BinaryTree, BlockVector, SparseBlockMatrix
5. Linear algebra module: Sherman-Morrison-Woodbury based system solver, support for Sparse-Sparse and Dense-Dense Kronecker product as Eigen expression nodes
6. Multithreading support by work-stealing thread pool implementation
