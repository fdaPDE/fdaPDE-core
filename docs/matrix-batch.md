# MatrixBatch

Include `<fdaPDE/dense_linear_algebra.h>`. `MatrixBatch<MatrixType, RowMajor>` owns
uniformly shaped native `Matrix`, `SymmetricMatrix`, `DiagonalMatrix` or
`SPDMatrix` elements. Both the batch and its element storage must be RowMajor;
ColMajor is rejected at compilation.

```cpp
using Point = fdapde::SPDMatrix<double, 2, 2, fdapde::Cache::Log>;
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
SPD elements start as identity; ordinary matrices start at zero. Construction
from a collection or batch expression initializes directly from its values,
without intermediate identities. Each source element is evaluated once. Values
must have a uniform positive shape; square structures require square elements.

`size()` counts matrices; `rows()`/`cols()` describe each matrix, including an
empty batch. `coefficient_stride()` is the physical count per matrix: r*c dense,
n diagonal, n*(n+1)/2 symmetric/SPD. `coefficients()` borrows a contiguous span of
row-packed coefficients, preserving native lower-triangular order. SPD spans are
read-only. `operator[]` returns the element's native `View` or `ConstView`;
SPD value assignment validates an independent candidate before committing it.
Bounds and allocation-size overflow checks are always active.

Selected SPD cache quantities occupy one scalar buffer for the entire batch,
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
selection retains its source shape. No stride or parallel/executor backend is
provided; all operations in this increment are sequential.

Allocation and timing measurements are available through the optional benchmark
in `tests/benchmarks/spd_cache_batch.cpp`; see [test instructions](../tests/README.md).
