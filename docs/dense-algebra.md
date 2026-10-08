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
Numeric `Matrix::resize` preserves the retained prefix of physical storage; it does not preserve
logical coordinates when strides change. Boolean matrices with a compile-time row or column
extent of one also preserve their retained vector prefix and clear newly exposed bits.
Other Boolean matrix shapes clear all logical bits when either dimension changes, even when
the total size is unchanged. Resizing to the same shape preserves existing coefficients.

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

## Symmetric and SPD coordinates

Square owners take one template order: `SymmetricMatrix<Scalar, Order, Policy = Cache::None, StorageOrder = RowMajor>`
and `SPDMatrix<Scalar, Order, Policy = Cache::None, StorageOrder = RowMajor>`.
Their views also take one order and the same cache policy. `Rows` and `Cols`
remain equal compile-time properties for generic matrix algorithms.
Use `Dynamic` for runtime order; a dynamic symmetric owner can be constructed
and resized with one order. The two-dimension overloads remain for generic
matrix code and reject unequal dimensions.

`SymmetricMatrix` and `SPDMatrix` accept a native row or column vector (including a
native vector expression) containing the independent matrix coefficients. The order
is the row-wise lower triangle: `(s11, s21, s22)` for order two, and
`(s11, s21, s22, s31, s32, s33)` for order three. Off-diagonal entries are unscaled.
Dynamic matrix order is inferred from the triangular coordinate count; fixed order
must agree with that count. The vector is copied into independently owned storage.

```cpp
const fdapde::Vector<double, 3> coordinates {2., .1, 1.};
const fdapde::SymmetricMatrix<double, 2> symmetric(coordinates);
const fdapde::SPDMatrix<double, 2> spd(coordinates);
```

Both types and their views expose `eigenvalues()`, which returns an independent
native vector and automatically reuses a spectral cache when available. Use
`evd()` when eigenvectors are also needed. See the [spectral access and cache
contract](symmetric-cache.md#spectral-operations-and-batches), including the effect
of mutable raw `data()` access.

SPD owners, views and SPD expressions also provide
`log<OutputCachePolicy = Cache::None>()`. It immediately returns an independent
`SymmetricMatrix<Scalar, Order, OutputCachePolicy>`, using the input cache when
available. The output policy is separate from the input SPD policy:

```cpp
auto logarithm = spd.log(); // owns symmetric coefficients without a cache
auto cached_logarithm = spd.log<fdapde::Cache::Spectral>();
auto values = cached_logarithm.eigenvalues(); // owns the output eigenvalues
```

The free function `matrix_log(spd)` remains available. Owning native batches
provide corresponding eager logarithm and eigenvalue operations with optional
parallel execution; see [MatrixBatch](matrix-batch.md#owning-spectral-operations).

SPD construction treats the vector as coefficients of `S`, checks finite values and
numerical positive definiteness, and prepares the selected caches. It does not interpret
them as logarithms. For logarithmic coordinates, construct a symmetric matrix and call
`matrix_exp` explicitly. Empty, nontriangular or mismatched native coordinate vectors
are rejected before coefficient reads or allocation.
