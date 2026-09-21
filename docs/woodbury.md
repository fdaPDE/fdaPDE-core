# Native Woodbury solver

`Woodbury(base_solver, U, inverse_c, V)` solves `(A + U C V) x = b`. The base solver supplies `A^{-1}b` through `solve` on a native double column vector; `inverse_c` is `C^{-1}`. Inputs have shapes `U(n,q)`, `inverse_c(q,q)` and `V(q,n)` with positive `n,q`.

The decomposition owns its base solver and materializes update expressions. It caches `A^{-1}U` and a pivoted-LU factorization of `C^{-1} + V A^{-1}U`. Each solve accepts one or more right-hand sides and returns owning double coefficients with the input's static dimensions and storage order. Integer inputs therefore retain fractional solutions; no conversion back to an integer coefficient type occurs.

Invalid shapes or nonfinite inputs throw `std::invalid_argument`. Uninitialized or moved-from state, singular correction matrices and invalid backend results throw `std::domain_error`. Copies use the supplied solver's value semantics. Moving clears the source decomposition. Backend accuracy and convergence remain the backend's responsibility.

The API is available through the Eigen-free dense aggregate. There is no separate one-shot helper or sparse-backend compatibility layer.
