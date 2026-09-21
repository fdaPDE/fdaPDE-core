# Native factorized sparse approximate inverse

`FSPAI<Scalar>` computes an owning lower CSR factor `L` for a symmetric positive-definite native sparse matrix. `lower_factor()` borrows this factor from an lvalue; `upper_factor()` returns its owning transpose, and `inverse()` evaluates `L * L.transpose()`. Dense and sparse `solve(rhs)` apply that approximate inverse; `solve_in_place(rhs)` writes a dense result back after successful evaluation.

Only unqualified floating coefficient types are supported. `alpha` bounds pattern-expansion steps, `beta` bounds candidates added per step, and finite nonnegative `epsilon` stops expansion when the largest candidate score is too small; candidates below the mean score are discarded. Successful `compute(matrix, alpha, beta, epsilon)` stores these settings for subsequent `compute(matrix)` calls. Failed recomputation preserves the previous factor and settings.

Shape, finite coefficients, symmetry and necessary positive-definiteness conditions are checked. The caller supplies an SPD matrix: these inexpensive checks are not a global SPD certificate. Detected local Cholesky breakdown throws `std::domain_error`; malformed inputs and settings throw `std::invalid_argument`.

The supporting CSR-by-CSR product accepts matching coefficient types, checks shape and integral arithmetic, combines contributions in each row, sorts columns and removes exact zeros. A dense accumulator handles ordinary widths; a hash accumulator avoids workspace proportional to extremely wide, lightly populated matrices. Floating arithmetic follows the coefficient type and may produce nonfinite values outside FSPAI's validated numerical path.

The implementation uses the Eigen-free sparse aggregate and does not require the SPD geometry or GMRES branches. The historical 264-by-264 FSPAI fixture is covered by active native tests, including the expected factor and independently accumulated approximate-inverse coefficients.
