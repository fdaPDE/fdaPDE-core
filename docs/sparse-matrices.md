# Native CSR matrices

`<fdaPDE/sparse_linear_algebra.h>` exposes `fdapde::SparseMatrix<Scalar>` and
`fdapde::Triplet<Scalar>` without Eigen or the dense matrix implementation.
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
scalar arithmetic, including nonfinite values. Integral duplicate overflow
throws `std::overflow_error` before performing the overflowing operation.
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

This aggregate currently provides storage, construction, lookup and row traversal.
Transpose, products and constraint rebuilding belong to the sparse operations
layer and are not yet exposed here.

The native tests cover both compression paths, duplicate order, cancellation,
integer overflow, bounds, empty/wide shapes, mutation, ownership and failure
preservation. An integration-only Eigen oracle compares complete compressed
patterns and coefficients. The optional `fdapde_sparse_benchmark` target measures
construction, checksum traversal and destruction on identical triangle-assembly
triplets with alternating execution order and warmups. Run it from a Release
build without competing workloads; it rejects mismatched observations or a
median native/Eigen time ratio above 1.25.
