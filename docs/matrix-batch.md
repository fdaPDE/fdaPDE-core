# MatrixBatch

Include `<fdaPDE/dense_linear_algebra.h>`. `MatrixBatch<MatrixType, RowMajor>` owns
uniformly shaped native `Matrix`, `SymmetricMatrix`, `DiagonalMatrix`,
`SPDMatrix` or `OrthogonalMatrix` elements. Both the batch and its element storage
must be RowMajor; ColMajor is rejected at compilation.

```cpp
using Point = fdapde::SPDMatrix<double, 2, fdapde::Cache::Log>;
fdapde::MatrixBatch<Point> points(10);  // identities with ready logarithms
points[1] = 2.0 * points[0];
auto determinants = points.map([](const auto& point) { return point.determinant(); });
fdapde::MatrixBatch<fdapde::Matrix<double, 1, 1>> values(determinants);
auto total = determinants.redux(0.0, [](double acc, const auto& value) {
    return acc + value(0, 0);
});
```

`MatrixBatch(count)` uses fixed element dimensions. `MatrixBatch(count, rows,
cols)` specifies uniform runtime dimensions (and checks any fixed axes).
SPD and orthogonal elements start as identity; ordinary matrices start at zero. Construction
from a collection or batch expression initializes directly from its values,
without intermediate identities. Each source element is evaluated once. Values
must have a uniform positive shape; square structures require square elements.

`size()` counts matrices; `rows()`/`cols()` describe each matrix, including an
empty batch. `coefficient_stride()` is the physical count per matrix: r*c dense,
n diagonal, n*(n+1)/2 symmetric/SPD. `coefficients()` borrows a contiguous span of
row-packed coefficients, preserving native lower-triangular order. SPD, orthogonal and cached
symmetric spans are read-only; `Cache::None` symmetric batches permit writable spans.
A cached symmetric element exposes writable packed coefficients through its view
`data()`. Calling `batch[i].data()` immediately invalidates that slot and permanently
disables its reuse, even if no write follows. Subsequent `cache()` or `evd()` requests
through any alias recompute its eigenpairs; other elements retain normal caching.
Same-slot assignment or destruction of the view does not restore reuse. For reads,
use `std::as_const(batch)[i].data()`; assigning `batch[i].data()` to a pointer-to-const
still invokes mutable access. See the [raw-access contract and example](symmetric-cache.md#raw-coefficient-access-and-cache-reuse).

`operator[]` returns the element's native `View` or `ConstView`;
SPD value assignment validates an independent candidate before committing it.
Bounds and allocation-size overflow checks are always active.

Selected SPD cache quantities or symmetric eigenpairs occupy one scalar buffer for the entire batch,
including dynamic element dimensions. Descriptors occupy one contiguous block,
and one table points to those descriptors; no descriptor owns another dynamic
matrix. The number of persistent cache allocations is independent of element
count for fixed shape and policy. `cache_pointers()` provides read-only layout
inspection. `Cache::None` removes cache allocation state and pointer tables via
a conditional empty member. Ordinary matrices have no SPD cache.

Copies own independent coefficient/cache buffers and rebuild bindings. Moves
transfer all buffers and leave an empty source with its previous element shape.
Assignment prepares a whole candidate and swaps it only after success. Treat
all views, coefficient spans and cache borrows as invalid after owner replacement,
move or swap. There is no resize API. Element assignment preserves shape and
updates the existing coefficient/cache slot. Owners must outlive their views.

## Owning spectral operations

Owning native batches provide the following elementwise operations. Every result
is an independent `MatrixBatch` of the indicated element type and is evaluated
immediately, including when stored with `auto`. `Scalar` and `Order` come from
the input elements; `P` is an explicit output cache policy defaulting to
`Cache::None`.

| Operation | Input elements | Output element type |
|---|---|---|
| `log<P>()` | SPD | `SymmetricMatrix<Scalar, Order, P>` |
| `exp<P>()` | symmetric or SPD | `SPDMatrix<Scalar, Order, P>` |
| `sqrt<P>()`, `inv_sqrt<P>()` | SPD | `SPDMatrix<Scalar, Order, P>` |
| `inv<P>()` | SPD | `SPDMatrix<Scalar, Order, P>` |
| `inv<P>()` | symmetric | `SymmetricMatrix<Scalar, Order, P>` |
| `inv()` | dense, diagonal or orthogonal | corresponding native inverse owner |
| `eigenvalues()` | symmetric or SPD | `Vector<Scalar, Order>` |
| `eigenvectors()` | symmetric or SPD | `OrthogonalMatrix<Scalar, Order, Order>` |
| `diagonal()` | square matrices | `Vector<Scalar, Order>` |
| `trace()`, `determinant()` | square matrices | `Matrix<Scalar, 1, 1>` |
| `norm()`, `squared_norm()` | matrices, including rectangular shapes | `Matrix<Scalar, 1, 1>` |

```cpp
auto logs = points.log<fdapde::Cache::Spectral>(fdapde::execution_par);
auto values = logs.eigenvalues(fdapde::execution_par);
auto traces = logs.trace(fdapde::execution_par);
auto determinants = logs.determinant(fdapde::execution_par);
```

Read scalar results with `traces[i](0, 0)`, `determinants[i](0, 0)` or the
corresponding norm batch entry. Statically rectangular element types do not
expose `trace()`, `determinant()`, `diagonal()` or `inv()`. Dynamic rectangular shapes
throw `std::invalid_argument` for these operations, even for an empty batch.

`norm()` requires floating-point scalars and returns the Frobenius norm.
`squared_norm()` sums squared logical coefficients: a packed symmetric
off-diagonal coefficient contributes twice. These norms, `trace()` and
`diagonal()` read coefficients without preparing an eigendecomposition or a
spectral cache. Batch determinants use each element's existing determinant
kernel: an SPD spectral cache can supply eigenvalues, while symmetric matrices
retain their LU-based determinant, including with `Cache::Spectral`.

`eigenvalues()` and `eigenvectors()` reuse coherent input spectral caches when
available and otherwise run the native eigensolver. Eigenvectors occupy columns
in the corresponding eigenvalue order. Eigenvalues are not guaranteed to be
sorted; eigenvector signs and bases within repeated eigenspaces are not unique.
The returned vectors and matrices remain valid after source mutation or
destruction. Orthogonal bases retain `inv()` as the transpose, may have
determinant -1, and need not be rotations. Orthogonal element views permit
whole-value assignment from orthogonal expressions, not arbitrary coefficient
writes; lazy maps preserve orthogonal results when materialized. For a single matrix, `evd()` returns both factors from one
decomposition.

`inv()` requires floating-point square inputs and preserves matrix structure.
Orthogonal bases use their transpose; rotations also retain their rotation type.
Diagonal inputs use coefficient reciprocals and dense inputs use pivoted LU.
SPD inputs prefer retained eigenpairs, then a retained Cholesky factor, then a
retained inverse root; otherwise they compute a local eigendecomposition.
Symmetric inputs use their native spectral cache when present and otherwise use
pivoted LU, retaining symmetry even for indefinite matrices. Requested SPD or
symmetric output caches are independent of input caches; other inverse outputs
use their native result type without an output policy override. SPD inverses
still certify their rounded coefficients before publication. The same input
cache invalidation rules apply to `inv()` as to other spectral operations.

```cpp
auto inverses = points.inv<fdapde::Cache::Spectral>(fdapde::execution_par);
auto inverse_bases = eigenvectors.inv(fdapde::execution_par);
auto inverse_roots = points.inv_sqrt(fdapde::execution_par);
```

The native dense-algebra spelling is now `inv()` (formerly `inverse()`) and
`inv_sqrt()` (formerly `inverse_sqrt()`). The free SPD functions are
`matrix_inv()` and `matrix_inv_sqrt()`. Cache tags such as `Cache::InverseSqrt`
retain their existing names.

`log()` uses a retained input logarithm or spectral factors when available.
`exp()` reuses available input eigenpairs. `sqrt()` and `inv_sqrt()` first
use the corresponding retained matrix factor, otherwise reuse input eigenpairs
when available or compute them. The output cache policy is independent of the
input policy. The exponential and roots certify their reconstructed SPD outputs;
requesting an output cache does not bypass finite-value or positive-definiteness
checks. The corresponding single-matrix members use the same kernels and return
owning matrices.

Omitting the execution argument selects `execution_seq`; each operation also
accepts it explicitly. `execution_par` uses the existing executor, processes
independent elements concurrently and preserves their index order in the result.
All calls are synchronous: they return complete independent batches. Worker
exceptions are rethrown to the caller after the outstanding work has joined.
The execution tags are available through `<fdaPDE/dense_linear_algebra.h>`.
To configure the executor's worker count, call `parallel_set_num_threads(count)`
before its first use.

The operations leave input coefficients unchanged but can prepare their cache
slots. Do not write to the inputs concurrently. Other operations on the same
unprepared or invalidated caches require external synchronization; this also
applies to caches whose reuse was disabled by mutable `data()` access. Empty
batches preserve each operation's result shape, including dynamic input order:
matrix functions and eigenvectors have shape `order x order`, eigenvalues and
diagonals have shape `order x 1`, and scalar results have shape `1 x 1`.
They contain zero entries and submit no worker tasks; shape restrictions still
apply before returning an empty result.

These methods belong to owning native batches. To apply them to a selection or
map, first materialize that expression into a `MatrixBatch`; the independent
element storage then supports parallel evaluation. The expression operations
below retain their existing sequential evaluation rules.

## Expressions

`map(callable)` retains a callable by value and borrows its persistent source.
It does no work until an element is requested or materialized. Access to i
evaluates only i; repeated access repeats computation and sees current source
values. The callable receives a const view. Scalar results become native
`Matrix<T,1,1>` owners. Matrix results retain their dense, symmetric, diagonal or
verified SPD structure; expressions and views are materialized while the source
proxy remains alive. Native nesting rules still apply: a lambda must not return
an expression borrowing its destroyed local owner. Return that local owner by
value or materialize the local expression before returning it.

Temporary expression nodes in chains are retained by value; persistent nodes
are borrowed. Temporary owning batches cannot be borrowed by map or select.
`redux(init, callable)` immediately folds in increasing index order; its
accumulator type is the type of init. Empty sources return init. A mapped
reduction evaluates and accumulates one element at a time, without a batch
intermediate. Exceptions propagate after local candidates are destroyed.

`select(indices)` owns a copy of the integral indices and preserves their order,
duplicates, source matrix storage and cache bindings. Invalid indices are
rejected. A temporary selection can be retained within a deferred mean or another
batch expression. Persistent selections can be reused with changing quadrature
weights. Sources must remain alive and must not be modified concurrently with
any evaluation.

Empty collections support map and redux. A materialized empty map needs a known
result shape: fixed result extents work; dynamic shape-changing callables cannot
supply an unknown result shape without evaluation and are rejected. An empty
selection retains its source shape. There is no strided batch storage or general
parallel map/redux API; the owning spectral operations above provide explicit
parallel evaluation.

Allocation and timing measurements are available through the optional benchmark
in `tests/benchmarks/spd_cache_batch.cpp`; see [test instructions](../tests/README.md).
