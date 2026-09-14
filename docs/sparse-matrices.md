# Native CSR matrices

`<fdaPDE/sparse_linear_algebra.h>` exposes `fdapde::SparseMatrix<Scalar>` and
`fdapde::Triplet<Scalar>` together with the native dense expression API, without Eigen.
The full `<fdaPDE/linear_algebra.h>` aggregate also exposes these types and
retains its other Eigen-dependent helpers.

```cpp
using Sparse = fdapde::SparseMatrix<double>;
Sparse matrix(2, 3, {{0, 2, 1.0}, {0, 2, 2.0}, {1, 0, 4.0}});
double value = matrix.coeff(0, 2); // three after duplicate compression
matrix.value_ref(1, 0) = 5.0;
for (const auto entry : matrix.row(0)) {
    int column = entry.column();
    double coefficient = entry.value();
}
```

The owner uses compressed sparse rows with dynamic rectangular dimensions and
`int` indices. Rows contain strictly increasing column indices. Construction
and `rebuild` combine duplicate coordinates in their input order and remove
exact zero sums. Thus floating-point results depend on duplicate input order;
there is no tolerance-based pruning. Floating coefficients follow ordinary
scalar arithmetic, including nonfinite values. Integral overflow in compression,
products and reductions throws `std::overflow_error` before performing the overflowing operation.
`bool` coefficients are unsupported.

`coeff(row, col)` returns a coefficient by value, with zero for an absent entry.
`contains(row, col)` distinguishes a stored entry from an implicit zero.
`value_ref(row, col)` requires an existing entry and never inserts. Assigning
zero through this reference preserves the pattern, so `non_zeros()` counts
**stored entries**, including explicitly assigned zeros.

`row(i)` returns a borrowed read-only range. Its iterators satisfy the C++20
forward-iterator requirements and yield proxies containing a column and a
reference to the owner's coefficient. Row ranges and mutable coefficient
references can only be obtained from lvalue matrices. The owner must outlive
all borrowed ranges, entries, iterators and references. Structural mutation,
assignment, move and swap invalidate these borrows.

`resize(rows, cols)` replaces the shape and clears its pattern. `rebuild`
replaces the pattern at the current shape. Both construct replacement storage
before publishing it, preserving the original matrix if validation, allocation
or coefficient construction fails. Copies own independent buffers. Moves
transfer storage and leave the source with zero dimensions; it can be resized
and rebuilt again.

Negative dimensions and out-of-range indices throw typed exceptions. Zero-row
and zero-column shapes are supported. The row count must be less than `INT_MAX`
to leave room for the terminal CSR offset; the triplet and stored-entry counts
must fit in `int`. An extremely wide matrix can have `INT_MAX` columns without
allocating a column-sized workspace when only a few entries are supplied.

Compression groups by columns when the number of columns does not exceed the
triplet count. Otherwise it groups by rows and stable-sorts within each row,
avoiding memory proportional to a very large column dimension. Coefficient lookup
uses binary search within a row. Storage is O(rows + stored entries); construction
also uses temporary buffers proportional to the input and the selected grouping.

Sparse operations evaluate eagerly into independent owners. `transpose()` swaps
rectangular dimensions and removes stored zeros. `symmetric_expanded(Upper)` or
`Lower` mirrors an authoritative triangle of a square matrix; any stored entry
in the opposite triangle is rejected, including an explicitly stored zero.

`matrix * rhs` accepts native dense owners, views and expressions with matching
inner dimensions. A compile-time column vector yields a dynamic native vector;
other dense expressions yield a dynamic row-major matrix. Coefficients use the
common scalar type of both operands. Empty inner dimensions yield zero results.
`row_sums()` returns one total per row. `diagonal()` requires a square matrix and
returns zero for absent diagonal entries. `SparseMatrix<Scalar>::from_diagonal(v)`
accepts row or column vector expressions and elides exact zeros after conversion.
Conversions to integral coefficients reject out-of-range and nonfinite values;
representable fractional values truncate toward zero.

`quadratic_form(v)` evaluates `vᵀ A v` for a square matrix and a matching row or
column vector. It uses the stored matrix as supplied, without implicit symmetry
expansion or complex conjugation. Integral products and partial sums must each
fit their result scalar type, even if later cancellation could make the final
result representable. Dimension, triangle and integral range checks remain active
in all builds. Constraint rebuilding is not exposed yet.

The native tests cover both compression paths, duplicate order, cancellation,
integer overflow, bounds, empty/wide shapes, mutation, ownership and failure
preservation. An integration-only Eigen oracle compares complete compressed
patterns and coefficients. The optional `fdapde_sparse_benchmark` target measures
construction, transpose and sparse-vector/dense products, including result
allocation, checksum traversal and destruction on identical triangle-assembly
inputs with alternating execution order and warmups. Run it from a Release build without competing workloads; it rejects mismatched observations or a
median native/Eigen time ratio above 1.25.
