# Dense iterative solvers

`<fdaPDE/dense_linear_algebra.h>` exposes restarted, left-preconditioned
`fdapde::GMRES`, `IdentityPreconditioner` and `DiagonalPreconditioner` without
Eigen. They are also available through `<fdaPDE/linear_algebra.h>`.

```cpp
using Matrix = fdapde::Matrix<double, 3, 3>;
using Vector = fdapde::Vector<double, 3>;
Matrix a({4, 1, -1, 0, 3, 1, 2, 0, 5});
Vector b(std::vector<double>{1, 2, 3});
fdapde::GMRES solver(a, fdapde::DiagonalPreconditioner<Matrix>{}, 100, 3, 1e-12);
auto x = solver.solve(b);
bool accepted = solver.converged();
```

The solver copies the dense operator and owns its preconditioner and workspaces.
Fixed, dynamic and partially dynamic square matrices, both dense storage orders,
const views and temporary expressions are supported. `compute(a)` replaces the
operator and recomputes the preconditioner. A failed computation leaves the
solver unavailable until a successful `compute`.

`solve(b)` starts from zero; `solve(b, x0)` accepts a finite initial guess.
Both return an owning column vector. A solve resets diagnostics, including when
its input is rejected. Invalid solve input does not invalidate the stored operator.
Multiple right-hand-side columns are supported by the preconditioners, but GMRES
solves one column vector at a time.

The stopping rule uses `||P^-1 (b - A x)|| <= tolerance * ||P^-1 b||`.
When `P^-1 b` is zero, it uses absolute tolerance instead. Estimated convergence
is checked against a freshly recomputed preconditioned residual. `residual()`
reports its **absolute** norm; `iterations()` counts completed Arnoldi steps.
Scaled norm comparisons can remain meaningful even when a norm is too large to
represent as a scalar, in which case the observer returns infinity.

Reaching the iteration limit, singular breakdown or unrepresentable iterative
arithmetic returns the last valid iterate with `converged() == false`. Callers
must inspect that flag before accepting a result. Default settings are 500
iterations, restart length 50 and tolerance `1e-6`; the Krylov dimension is capped
at the matrix order. Restarting can stagnate, and convergence in a preconditioned
norm alone does not guarantee small forward error on ill-conditioned systems.

Public shape, state, finite-input and configuration checks use typed exceptions.
Identity preconditioning checks shape and copies its input; it does not inspect
unused operator coefficients. Diagonal preconditioning owns the reciprocals of
finite, nonzero diagonal entries and rejects unrepresentable reciprocals. It
ignores off-diagonal entries and otherwise applies ordinary scalar arithmetic to
right-hand-side coefficients.

The implementation materializes a dense matrix: operator storage is quadratic,
and each Arnoldi product is dense. Workspaces use O(n m + m^2) storage for order
n and restart dimension m. Sparse matrices and matrix-free operators are not
accepted by this API.

Tests cover known solutions, nonsymmetric float/double systems, restart cycles,
exact and zero initial residuals, iteration limits, breakdown, extreme residual
scales, copying and expression ownership, invalid inputs and recomputation.
The integration test lane additionally compares against Eigen pivoted LU; the
native lane builds and runs with Eigen unavailable to the compiler.
