# Randomly pivoted native Cholesky

`RpChol(matrix, tolerance, block_size, max_iterations, seed)` approximates a symmetric positive-semidefinite dense native matrix by `L * L.transpose()`. The relative Frobenius residual controls early stopping; the rank cannot exceed `min(order, block_size * max_iterations)`. Hitting this capacity returns the computed approximation, so the requested tolerance is not guaranteed when capacity is insufficient.

`factor()` borrows the owning rectangular factor and `pivots()` borrows distinct indices in selection order. Both reject temporary owners. The coefficient type and dense storage order follow the input. A fixed seed gives reproducible sampling on a fixed standard-library implementation. The zero matrix returns an order-by-zero factor.

Inputs are materialized and scaled before sampling. Shape, finite-value, symmetry and necessary PSD conditions are checked. These checks do not certify global positive semidefiniteness; PSD is an input precondition, and detected numerical breakdown throws `std::domain_error`. Configuration, shape and finite-input violations throw `std::invalid_argument`. Failed recomputation preserves previous results.

The Eigen-free dense aggregate supplies the implementation. No SPD geometry, sparse module or general public Cholesky factorization is required. There are no duplicate factor-accessor names or stored unordered pivot set.
