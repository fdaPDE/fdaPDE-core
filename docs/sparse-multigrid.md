# Fixed symmetric sparse multigrid

`fdapde::SparseMultigrid`, exposed by `<fdaPDE/linear_algebra.h>`, supplies a
fixed linear approximate inverse for a real sparse SPD scalar elliptic matrix.
It uses the Eigen-backed sparse module, independently of geometric-field models.

The constructor takes an `Eigen::SparseMatrix<double>` and optionally the fixed
cycle count (default 4, permitted 1–16) and coarse threshold (default 256, at
least 2). `solve(VectorXd)` starts from zero each time; it does not solve to a
requested residual tolerance and does not retain previous right-hand sides.
`level_sizes()` and `nonzeros()` expose hierarchy complexity; the latter counts
Galerkin and transfer coefficients but excludes coarse-factor fill.

Strong-neighbor greedy aggregates are deterministic. Stranded singletons attach
to their strongest established neighboring aggregate, avoiding coarse levels
that barely shrink. One subcritical Jacobi step smooths the tentative transfer; Galerkin products define the next level.
Two forward Gauss–Seidel sweeps, a recursive coarse correction, then two
backward sweeps form a symmetric cycle. The final level uses sparse Cholesky.
Repeating a fixed number of these cycles yields a linear symmetric positive
approximate inverse for SPD input. Work vectors are local to each call.

The caller must supply SPD input. Public checks reject malformed, nonfinite or
asymmetric matrices, nonpositive diagonals while coarsening, and failed coarse
Cholesky. They do not certify positive definiteness of the complete fine matrix.
Disconnected components are allowed if the matrix remains SPD; an unscreened
Neumann Laplacian is singular and is outside the contract.

This is a deliberately small scalar-elliptic implementation, not a universal
algebraic multigrid framework. Strong anisotropy and unusual sparsity can cause
poor coarsening or excessive fill. If aggregation cannot reduce a level, that
level is factored directly, even above the requested threshold. Review hierarchy
complexity on new workloads. No near-nullspace basis for vector systems is
inferred. Fixed cycles are important for optimizer charts that require a
constant map and its transpose: a residual-stopped inner Krylov solve generally
does not preserve that linearity.

`tests/linear_algebra/sparse_multigrid.cpp` verifies multilevel manufactured
solutions, linearity, symmetry, positive spectrum, scaling and invalid inputs.
The test target is `fdapde_sparse_multigrid_test`.
