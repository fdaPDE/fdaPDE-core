# Native matrix lumping

`fdapde::lump(matrix)` constructs a diagonal whose entries are the row sums of
a square input. It is available through `<fdaPDE/sparse_linear_algebra.h>`
without Eigen, and through the full `<fdaPDE/linear_algebra.h>` aggregate.

Dense owners, views and expressions produce an independent
`DiagonalMatrix<Scalar, Dynamic>`. Sparse owners produce an independent
`SparseMatrix<Scalar>`. Coefficient types are retained, including integer,
float and complex types. Lumping an expression evaluates it immediately;
subsequent input changes do not affect the result.

The sparse output stores one entry per diagonal position, including zero sums
and empty input rows. This allows updating any diagonal entry through
`value_ref`. It deliberately differs from `SparseMatrix::from_diagonal`, which
elides zero entries. Subsequent sparse transformations follow their own zero
pruning rules.

```cpp
fdapde::SparseMatrix<double> matrix(2, 2, {{0, 0, 2.0}, {0, 1, -2.0}, {1, 1, 3.0}});
auto diagonal = fdapde::lump(matrix);
// both diagonal positions are stored, with coefficients zero and three
diagonal.value_ref(0, 0) = 1.0;
```

Statically rectangular dense inputs are rejected at compile time. Runtime
rectangular shapes throw `std::invalid_argument`; zero-by-zero shapes are valid.
Floating and complex inputs must have finite components. A nonfinite input
throws `std::invalid_argument`, while a finite row whose partial sum becomes
nonfinite throws `std::overflow_error`. Signed and unsigned integral partial
sums are checked before addition and throw `std::overflow_error` when they
exceed the scalar range. All these public checks remain active in every build.
Custom scalar types are accumulated with `operator+=` under their own arithmetic
semantics. Failure leaves the input unchanged.

Dense accumulation follows logical column order within each row, independently
of storage layout. Sparse accumulation follows sorted CSR column order. Floating
results can depend on that summation order; no tolerance or compensation is
applied. Dense lumping visits every logical coefficient, while sparse lumping
runs in O(rows + stored entries), using O(rows) result storage.

The optional `fdapde_sparse_benchmark` target compares sparse lumping with Eigen
sparse multiplication by a vector of ones followed by diagonal CSR construction.
It includes result allocation, checksum traversal and destruction on both sides.
This benchmark does not measure dense-expression lumping.
