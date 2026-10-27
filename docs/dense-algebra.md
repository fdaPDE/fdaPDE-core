# Dense algebra

Include `<fdaPDE/dense_linear_algebra.h>` for native matrices, expression
operations, multidimensional arrays and dense decompositions without Eigen.
`<fdaPDE/linear_algebra.h>` adds Eigen integration and sparse algorithms.
`<fdaPDE/utility.h>` contains shared utilities and numeric functions; it no longer declares matrix types.

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
objects whose parent must remain alive. `MdView` supports access and assignment,
but does not expose nested block/slice factories; select regions from an owner
or map. Assign through a full block view to copy coefficients into `MdMap`.
Pairs passed to `block` specify inclusive endpoints. `slice<Axes...>` removes the selected axes. Copying between arrays or
views follows logical coordinates, including across storage orders, and handles
aliasing. Assignment requires compatible static extents.

Dense structured types include diagonal, triangular, symmetric, skew-symmetric,
orthogonal and permutation matrices and their applicable views. Packed
triangular, symmetric and skew-symmetric storage currently supports `RowMajor`;
unsupported packed `ColMajor` instantiations fail at compile time. This restriction
does not apply to ordinary dense `Matrix` or packed Boolean storage.

`PartialPivLU` factors square matrices and solves nonsingular systems.
For usable finite factors, its determinant reuses the pivot signs and normalized
pivots with binary exponent tracking, avoiding another elimination and intermediate
scale overflow. Unusable numerical pivots or factor growth retain the independent
complete-pivot determinant path; the algebraic determinant remains available even
when solving is rejected. Constant evaluation retains its existing implementation.
`HouseholderQR` factors rectangular matrices. `Q()` borrows a validated
`OrthogonalMatrix<Scalar, Rows>` and `R()` borrows the dense upper-trapezoidal
factor with the input shape. `solve(B)` reuses these factors to solve square
systems or minimize the residual norm of overdetermined systems with full
numerical column rank. Vector and matrix right-hand sides return independent
solutions with one row per input column. Underdetermined systems, deficient
rank, invalid right-hand sides and nonrepresentable results are rejected.
Factor references require the decomposition to
remain alive and must not outlive a recomputation. `EVD` computes a symmetric
Jacobi eigendecomposition from generic matrices or views, checking finite coefficients
and symmetry, and exposes eigenvectors as a const orthogonal owner reference.
Its `solve(B)` uses the retained eigenpairs for one or several right-hand sides,
including indefinite symmetric matrices. Eigenvalue magnitudes at or below
`dimension*epsilon*max(abs(eigenvalues))` are rejected as numerically singular.
Invalid right-hand sides and nonfinite solutions are rejected, and returned
solutions own their coefficients independently of the decomposition.
`Cholesky` accepts generic matrices or views with finite symmetric positive-definite
coefficients, exposes an owning `LowerTriangularMatrix` through `L()` and solves
vector or matrix right-hand sides. Invalid shape, asymmetry and nonfinite inputs
are rejected; nonpositive or unrepresentable pivots raise a numerical error.
Symmetric matrices provide `.evd()`, and SPD matrices provide `.cholesky()` even
with `Cache::None`. Both return independent owning decomposition objects and reuse
compatible native input caches when present. These methods use the guarantees of
the structured types; the algorithm classes remain available for generic input.
Symmetric matrices also provide direct owning `eigenvalues()` and
`eigenvectors()` results; `Cache::Spectral` reuses the decomposition across these
requests. Check the decomposition's reported status where
provided; singularity, invalid input and convergence failures remain distinct.
SPD matrices also provide `.cholesky_factor()` for an independent owning
`LowerTriangularMatrix`, reusing `Cache::Cholesky` when present and computing the
factor otherwise.
Dense inverse and determinant operations use the corresponding complete
factorization implementation.

Debug preconditions use the typed `fdapde_assert` interface. Storage-capacity
limits, null external storage and finite factorization inputs have selected
permanent checks. Compile-time shape, size and unsupported-storage checks remain
independent of debug configuration.

Use `matrix.orthogonalize()` to orthonormalize the columns of a finite full-rank
square matrix. It evaluates immediately and returns an owning
`OrthogonalMatrix` with the input's scalar type, dimensions and storage order;
the source is unchanged. Rank deficiency, nonfinite coefficients and invalid
shapes are rejected even with `FDAPDE_NO_DEBUG`. It also accepts native views and expressions, including
temporary inputs, because the result owns its coefficients.

```cpp
const fdapde::Matrix<double, 2, 2> directions({1., 1., 1., 0.});
const auto basis = directions.orthogonalize(); // OrthogonalMatrix<double, 2>
```

Construct `OrthogonalMatrix` directly from an already orthogonal matrix or
row-ordered coefficients. Construction and replacement always validate shape,
finiteness and orthogonality; no public `checked` or `unchecked` overload is
available. `owner.view()` borrows verified storage from a persistent owner.
External-buffer views also validate their coefficients at construction, and
view assignment validates a complete candidate before changing the buffer.
A view cannot change its bound shape. The buffer must stay alive and orthogonal
while borrowed; failed replacement preserves the previous coefficients.

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

## Square matrix interfaces

Matrix families that are square by construction take one template order:

| Family | Template parameters |
| --- | --- |
| Permutation | `PermutationMatrix<Order>` |
| Orthogonal | `OrthogonalMatrix<Scalar, Order, StorageOrder = RowMajor>` |
| Rotation | `RotationMatrix<Scalar, Order, Policy = RotationCache::None>` |
| Skew-symmetric | `SkewSymmetricMatrix<Scalar, Order, StorageOrder = RowMajor>` |
| Triangular | `TriangularMatrix<Scalar, Order, ViewMode, StorageOrder = RowMajor>` |
| Upper or lower triangular | `UpperTriangularMatrix<Scalar, Order>` or `LowerTriangularMatrix<Scalar, Order>` |

Their views use the same single order. `Dynamic` selects a runtime order;
`Rows` and `Cols` remain equal properties for generic algorithms. Dense owners
and the QR input and `R` factor retain separate row and column dimensions
because they can be rectangular. Existing two-dimension construction overloads for generic code
still require equal dimensions.

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

## Runnable tutorials

The [example index](../examples/README.md#choose-an-example) maps native matrix
families and Eigen sparse helpers to standalone C++20 programs. Begin with
[dense storage and views](../examples/linear_algebra/01_dense_storage_and_views.cpp),
then [expressions and products](../examples/linear_algebra/02_dense_expressions_and_products.cpp),
[structured solves](../examples/linear_algebra/03_diagonal_and_triangular_solve.cpp) and
[dense factorizations](../examples/linear_algebra/06_dense_factorizations.cpp).
[SPD spectral caches](../examples/linear_algebra/07_spd_spectral_and_caches.cpp) shows
how to transform and whiten a covariance, reuse cached quantities and replace
its value. The numbered index provides a reading order and build commands.
