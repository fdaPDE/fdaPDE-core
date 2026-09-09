# Dense algebra

Include `<fdaPDE/linear_algebra.h>` for matrices, expression operations,
multidimensional arrays and dense decompositions. `<fdaPDE/utility.h>` contains
shared utilities and numeric functions; it no longer declares matrix types.

`fdapde::Matrix<Scalar, Rows, Cols, StorageOrder>` is the owning dense type.
Dimensions may be positive compile-time constants or `fdapde::Dynamic`.
The fourth parameter selects `RowMajor` (the default) or `ColMajor`.
`Vector<Scalar, Rows>` is a column matrix. Packed Boolean storage uses
`Matrix<bool, Rows, Cols, StorageOrder>` and `Vector<bool, Rows>`.

```cpp
fdapde::Matrix<double, 2, 2> a({1, 2, 3, 4});
fdapde::Matrix<double, 2, 2> b = a / 2;
a = a.transpose();
a.row(1) = a.row(0);
```

The initializer above specifies coefficients in logical row order for either
storage order. Scalar division divides each coefficient by the scalar;
`double` coefficients divided by an integer retain floating-point arithmetic.
Assignments materialize their source before overwriting aliased storage.
Dynamic owner assignment can resize; fixed-shape assignments must match.
Resizing an owner discards its old coefficients, including packed Boolean bits.

`MatrixExpr` supplies the expression API. `MatrixBase` supplies shared storage
shape and coefficient access to owners and external-storage views; its template
parameters are `Scalar, Rows, Cols, StorageOrder, MatrixType`. Code accepting
arbitrary expressions should use `MatrixExpr` or the expression concepts rather
than depending on the storage base.

## Views and expression lifetime

`MatrixView<Scalar, Rows, Cols, StorageOrder>` maps contiguous external memory.
Use a const scalar to make the view read-only. Its storage must outlive every
view and expression that refers to it. Assignment writes coefficients through
a view; it does not rebind the pointer.

Blocks, rows, columns, reshapes, coefficient-wise adaptors and vector-wise
adaptors borrow their input. Owning rvalues are rejected where borrowing would
leave a dangling expression. Give an owner a name before taking a view or
building a lazy expression from it. Destruction or resizing of the owner
invalidates its views. Nested non-owning expression nodes may be held by value;
this does not extend the lifetime of their underlying storage.

`cwise()` supplies coefficient-wise arithmetic and comparisons; `mwise()` returns
to matrix algebra. `rowwise()` and `colwise()` reduce or broadcast along the
selected axis. Reductions check their domains in debug builds; mean, minimum
and maximum require nonempty input. Floating norms use scale-safe accumulation.

## Multidimensional and structured storage

`MdArray<Scalar, MdExtents<...>, StorageOrder>` owns multidimensional storage.
`MdMap` maps external storage. `block`, `slice`, `row` and `col` produce `MdView`
objects whose parent must remain alive. Pairs passed to `block` specify inclusive
endpoints. `slice<Axes...>` removes the selected axes. Copying between arrays or
views follows logical coordinates, including across storage orders, and handles
aliasing. Assignment requires compatible static extents.

Dense structured types include diagonal, triangular, symmetric, skew-symmetric,
orthogonal and permutation matrices and their applicable views. Packed
triangular, symmetric and skew-symmetric storage currently supports `RowMajor`;
unsupported packed `ColMajor` instantiations fail at compile time. This restriction
does not apply to ordinary dense `Matrix` or packed Boolean storage.

`PartialPivLU` factors square matrices and solves nonsingular systems.
`HouseholderQR` factors rectangular matrices. `EVD` computes a symmetric
Jacobi eigendecomposition. Check the decomposition's reported status where
provided; singularity, invalid input and convergence failures remain distinct.
Dense inverse and determinant operations use the corresponding complete
factorization implementation.

Debug preconditions use the typed `fdapde_assert` interface. Storage-capacity
limits, null external storage and finite factorization inputs have selected
permanent checks. Compile-time shape, size and unsupported-storage checks remain
independent of debug configuration.

## Core integration

The current geometry, fields, FEM, splines, optimization and GeoFrame modules
consume the new types. Existing Eigen-based sparse algorithms and applicable
core endpoints still use Eigen; including the algebra aggregate currently
requires Eigen. There are no implicit conversions between Eigen and the new
matrix expression system. Callers copy coefficients or use a view with an
explicit storage order at those boundaries.

FEM interpolation still builds and solves the same Vandermonde system.
Quadrature coefficients, basis definitions and assembly formulas retain their
current implementation. The gradient packet uses explicit `(embedding dimension,
component count)` extents to match the Jacobian accessors; copies between matrix
and multidimensional storage specify logical coordinates.

The active tests exercise dense and structured contracts, static rejection
programs, historical Boolean behavior, and P1/P2 interpolation and assembly.
Historical cases are retained in `test/` until equivalent active tests have been
verified; migrated cases carry precise replacement pointers.
