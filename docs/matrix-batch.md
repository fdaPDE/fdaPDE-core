# MatrixBatch

Include `<fdaPDE/dense_linear_algebra.h>`. `MatrixBatch<MatrixType, RowMajor>` owns
uniformly shaped native `Matrix`, `SymmetricMatrix`, `SkewSymmetricMatrix`,
`DiagonalMatrix`, `SPDMatrix`, `OrthogonalMatrix` or `RotationMatrix` elements.
Both the batch and its element storage must be RowMajor; ColMajor is rejected
at compilation.

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
n diagonal, n*(n+1)/2 symmetric/SPD, and n*(n-1)/2 skew-symmetric.
`coefficients()` borrows a contiguous span of row-packed coefficients: lower
triangle for symmetric/SPD and strict upper triangle for skew-symmetric elements.
Order-one skew elements have stride zero and no stored coefficients, including
in a nonempty batch; their only logical coefficient is zero.
SPD, orthogonal and cached
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
Skew views reflect upper coefficients with the opposite sign below the diagonal.
As with standalone skew views, element assignment reads only the strict upper
triangle of its source; whole-batch skew construction and assignment require
skew-symmetric expressions. Use `as_skew_symmetric<Upper>()` or
`as_skew_symmetric<Lower>()` to explicitly interpret a dense source triangle.
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
`squared_norm()` sums squared logical coefficients: a packed symmetric or skew-symmetric
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
element storage then supports parallel evaluation. Expression materialization
also accepts an execution policy through `eval()` or `assign()`.

## Batch arithmetic and evaluation

Native arithmetic operators lift to batch entries. Two batch operands must
have equal lengths: result entry `i` combines the two entries at `i`, with no
Cartesian product or implicit length-one batch broadcasting. A single matrix
operand is broadcast to every entry. Matrix addition and subtraction require
matching shapes; multiplication is the native matrix product and requires
compatible inner dimensions. Scalars support multiplication on either side
and division on the right. Unary minus negates each entry.

```cpp
using Sym = fdapde::SymmetricMatrix<double, 2>;
fdapde::MatrixBatch<Sym> a(10), b(10), sum;
const Sym shift(fdapde::Vector<double, 3>{0.2, 0.05, 0.1});

auto expression = a + b;                       // deferred MatrixBatchExpr
auto result = expression.eval(fdapde::execution_par);  // owning MatrixBatch<Sym>
sum = expression;                             // sequential owning assignment
sum.assign(expression, fdapde::execution_par); // preserves destination type and cache policy
auto combined = (2.0 * a + b - shift).eval();   // sequential by default
a += shift;                                   // alias-safe sequential update
```

`auto expression = a + b` stores an expression node, not a batch of per-entry
matrix expressions. It borrows persistent matrix and batch operands and owns
temporary expression nodes. Scalars are copied by value. No entries are
evaluated until accessed or materialized; repeated evaluation sees current
operand values. Keep borrowed operands alive. Temporary owning matrices and
batches are rejected as deferred arithmetic operands, following native matrix
and batch nesting rules.

`expression.eval(policy)` returns an independent owning `MatrixBatch` with the
element type inferred from the native operation. `destination.assign(expression,
policy)` instead uses the destination's element type and cache policy, applying
its native constructor conversions and validation. `destination = expression`
selects sequential assignment. Each method defaults to `execution_seq`, accepts
`execution_par`, and finishes before returning. Parallel evaluation preserves
entry order and joins submitted work before propagating an exception. The
complete candidate replaces the destination only after success, so expressions
may read the destination or selections that alias it. Successful replacement
invalidates previously borrowed destination views.

Expressions are composed before materialization: `(2.0 * a + b - shift).eval(policy)`
evaluates the whole formula for each index without allocating intermediate
batches. Coefficientwise sums, differences and scalar operations fuse across
arithmetic nodes, also avoiding intermediate dynamic matrix owners. Full matrix
products and mapped results retain their materialization boundaries so products
are not recomputed for every downstream coefficient and returned temporaries
stay alive. The same evaluation methods apply to maps and selections. Nonempty
dynamic results infer their shape from the first entry, which is evaluated only
once. Empty expressions retain known operand/result shapes; an empty dynamic map
with unknown output shape remains invalid.

Native structure determines the result owner:

| Element operation | Inferred result element |
|---|---|
| symmetric `+`, `-`, scalar `*`, scalar `/`, unary `-` | symmetric |
| skew-symmetric `+`, `-`, scalar `*`, scalar `/`, unary `-` | skew-symmetric |
| SPD `+` or `-`, scalar scaling | symmetric, following native arithmetic |
| symmetric matrix product | dense |
| skew-symmetric matrix product | dense |
| diagonal arithmetic and diagonal matrix product | diagonal |
| orthogonal matrix product | orthogonal |
| rotation matrix product | rotation |
| general mixed matrix arithmetic | native dense result |

Ordinary symmetric arithmetic expressions use `Cache::None`; caches cannot be
copied from changed operand values. A cached destination retains its selected
policy through `assign()` and receives independent cache slots for the new
values. Arithmetic does not populate or invalidate input caches. Compound
`+=`, `-=`, `*=`, and `/=` evaluate sequentially through the same candidate
replacement path. They require a result structure compatible with the
destination; for example, multiplying a symmetric batch by a general matrix
cannot silently store only one triangle. SPD destination updates additionally
certify the resulting values through native SPD construction. For a parallel
update, use `a.assign(a + shift, execution_par)`.

See the standalone [arithmetic tutorial](../examples/linear_algebra/matrix_batch_arithmetic.cpp),
kept separate from the [spectral operations tutorial](../examples/linear_algebra/spd_batch_operations.cpp).

## Expressions

`map(callable)` retains a callable by value and borrows its persistent source.
It does no work until an element is requested or materialized. Access to i
evaluates only i; repeated access repeats computation and sees current source
values. The callable receives a const view. During one evaluation, the callable
must not modify any operand storage, including through captured aliases; future
evaluations may observe changes made between calls. Scalar results become native
`Matrix<T,1,1>` owners. Matrix results retain their dense, symmetric, skew-symmetric, diagonal or
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
selection retains its source shape. There is no strided batch storage or parallel
reduction API. Maps can be materialized with `eval(execution_par)` when their
callable and captures are thread-safe. The callable is shared between workers;
stateful captures require synchronization. Selections with repeated indices can
share input cache slots, so mapped operations that prepare those same slots
concurrently also require external synchronization. Plain coefficient arithmetic
does not prepare input caches.

Allocation and timing measurements are available through the optional benchmark
in `tests/benchmarks/spd_cache_batch.cpp`; see [test instructions](../tests/README.md).
