# Symmetric matrix caching

`SymmetricMatrix<Scalar, Order, Policy = Cache::None, StorageOrder = RowMajor>`
is the public symmetric owner available through `<fdaPDE/dense_linear_algebra.h>`.
The same type supports uncached coefficients and optional lazily refreshed
spectral data. `Cache::None`, `Cache::Spectral` and their unions are supported;
SPD-only logarithm or square-root policies are not. `SymmetricMatrixView` takes
the same parameters and shares its owner's invalidation state.

```cpp
using Sym = fdapde::SymmetricMatrix<double, 2, fdapde::Cache::Spectral>;
Sym matrix(fdapde::Vector<double, 3> {2., 0.3, 4.});
auto coefficient = matrix(0, 0);
auto first = fdapde::expm(matrix);  // prepares and reuses spectral data
coefficient = 5;                   // invalidates even this previously saved alias
auto second = fdapde::expm(matrix); // refreshes the decomposition
```

Coefficient writes, whole-value assignment and compound arithmetic invalidate
cached factors. This includes writes through a symmetric view or a saved
coefficient proxy. The next `cache()` access or spectral operation computes fresh
eigenpairs. Packed symmetric writes update both reflected coordinates. Both
policies support packed construction and runtime-order resize; `Cache::None`
allocates no spectral buffer.

## Raw coefficient access and cache reuse

**Calling non-const `data()` immediately invalidates the cache and permanently
disables its reuse for the exposed storage binding, even if the pointer is only
used to read or is never used at all.** The library cannot track later writes
through a retained raw pointer. Every subsequent `cache()` or `evd()` request
therefore computes fresh eigenpairs, including repeated requests with no
intervening writes. Spectral operations that use this cache incur the same
recomputation cost. `Cache::None` has no cache state to invalidate.

| Access | Effect on spectral caching |
|---|---|
| `std::as_const(matrix).data()` | preserves current state; does not restore previously disabled reuse |
| `matrix.data()` on a mutable owner | invalidates immediately and disables subsequent reuse |
| `const double* p = matrix.data()` on a mutable owner | selects the mutable overload and also disables reuse |
| `matrix(i, j) = value` or tracked view assignment | invalidates on the write; the next request refreshes and can be reused |

The object's constness selects the overload; the destination pointer's type
does not. Include `<utility>` when using `std::as_const`.

```cpp
#include <fdaPDE/dense_linear_algebra.h>
#include <utility>

using Sym = fdapde::SymmetricMatrix<double, 2, fdapde::Cache::Spectral>;
Sym matrix(fdapde::Vector<double, 3> {2., 0., 3.});
matrix.cache();
const double* read = std::as_const(matrix).data(); // preserves the prepared cache

double* write = matrix.data(); // invalidates now and disables subsequent reuse
write[0] = 5.;                // packed order is s00, s10, s11
auto first = matrix.evd();    // computes eigenvalues 5 and 3
write[0] = 7.;                // the same pointer is still usable after that computation
auto second = matrix.evd();   // computes eigenvalues 7 and 3 instead of reusing first

Sym independent(matrix);      // copies coefficients into separate storage
independent.cache();          // prepares an independently reusable cache
independent(0, 0) = 9.;        // a tracked write only invalidates that cache
independent.cache();          // refreshes it; later requests can reuse it
```

The restriction belongs to the storage and its shared cache slot. Access through
`matrix.view().data()`, `matrix.rep().data()` or `batch[i].data()` has the same
effect for that owner or batch element and all its aliases. Other batch elements
are unaffected. Destroying the view, discarding the pointer, calling the const
overload, or assigning new coefficients at the same shape does **not** restore
reuse. There is no explicit operation to re-enable it on the exposed binding.
An independent owning copy has its own reusable cache and does not inherit
potentially stale factors from the exposed source. Prefer coefficient proxies
or tracked assignment when modifying coefficients while retaining cache reuse.

## Borrowed state and lifetime

Cache references and borrowed eigenvector/eigenvalue views must not be retained
across mutation. A retained slot detects controlled writes; previously obtained
raw factor views are invalidated by mutation. Raw-pointer writes cannot be
observed by an already borrowed slot, so request `cache()` again after such a
write. In particular, `slot.valid()` reports prepared state and cannot certify
freshness after an untracked raw write. Owner shape changes, destruction, and batch replacement/move/swap invalidate
value views, raw pointers and coefficient proxies as well. Same-shape owner
replacement preserves value and invalidation bindings. Owner and batch copies
have independent coefficients and reusable cache state; exposed source factors
are not copied as ready data. Concurrent access that can populate the same cold
cache requires external synchronization, as do writes; prepare unexposed caches
before parallel read-only use.

## Spectral operations and batches

`matrix.eigenvalues()` returns an independent native `Vector<Scalar, Order>`
for symmetric and SPD owners, their views and symmetric expressions. A native
spectral cache is selected at compile time: a reusable cache copies only the
eigenvalues, without constructing an EVD result or copying its eigenvectors.
An invalidated cache is refreshed first; a source without a spectral cache uses
the native eigensolver. The result preserves scalar precision and remains valid
after source mutation or destruction. Eigenvalues are not guaranteed to be sorted.
`matrix.eigenvectors()` similarly returns an independent
`OrthogonalMatrix<Scalar, Order, Order>`, copying a reusable cache's eigenvectors or running
the native eigensolver. Its columns follow the corresponding eigenvalue order;
signs and bases within repeated eigenspaces are not unique. After mutable
`data()` access, every `eigenvalues()` or `eigenvectors()` request refreshes the
spectrum, just like `cache()` and `evd()`. Use `evd()` when both factors are
required from one decomposition.

Symmetric and SPD owners, views and expressions provide `exp<Policy>()`, which
returns a checked owning SPD exponential; SPD inputs also provide
`sqrt<Policy>()` and `inv_sqrt<Policy>()`. The output policy defaults to
`Cache::None` and is independent of the source policy. The exponential reuses
available source eigenpairs. The roots use their operation-specific retained
factor when available, otherwise reuse spectral factors or compute them.
Reconstructed outputs still undergo finite-value and SPD certification.

The native EVD constructor reuses coherent cached factors. `expm`, `logm`,
`sqrtm`, `powm`, checked `matrix_exp` and symmetric Frechet operations thereby
avoid recomputing a ready decomposition at matching scalar precision. Legacy
functions that promote float inputs to double recompute in double rather than
silently reusing lower-precision factors. Domain checks still run: an indefinite
symmetric owner is valid, while its real SPD logarithm is rejected. Matrix
functions of expressions such as `matrix + other` decompose the resulting matrix;
the operands' eigenpairs do not in general determine that sum's eigenpairs.

`MatrixBatch<SymmetricMatrix<..., Cache::Spectral>>` uses packed contiguous
coefficient rows and aggregate spectral buffers. Its mutable views share slot
invalidation with the batch. Selection and map preserve the source policy when
their result is a symmetric owner or view. A batch of SPD logarithms can therefore
materialize symmetric caches and then extract independent eigenvalue vectors.
The owning batch methods evaluate immediately and accept `execution_seq` (the
default) or `execution_par`:

```cpp
#include <fdaPDE/manifold_optimization.h>

using namespace fdapde;
using namespace fdapde::manifold;
using SPD = SPDMatrix<double, 2, Cache::Log>;

const SPD A(Vector<double, 3> {2., 0.3, 1.});
const SPD B(Vector<double, 3> {1., 0.2, 3.});
const LogEuclideanGeometry<SPD> geometry;
auto points = geometry.geodesic<Cache::Log>(A, B, 10, execution_par);

auto logs = points.log<Cache::Spectral>(execution_par);
auto eigenvalues = logs.eigenvalues(execution_par);
auto traces = logs.trace(execution_par);
auto determinants = logs.determinant(execution_par);
```

The overload returns an owning `MatrixBatch<SPD>` using the geometry's point
type and cache policy by default. `geometry.geodesic<Cache::Spectral>(A, B, 10)`
selects a different output cache while preserving scalar and order. The fourth
argument selects `execution_seq` (default) or `execution_par`, preserving output
order and joining all submitted work before returning or rethrowing. It prepares
the geodesic once and evaluates ten uniformly spaced parameters from zero to one, including both endpoints up to floating-point
reconstruction. The sample count must be at least two; smaller counts throw
`std::invalid_argument`. The prepared LE curve reuses endpoint logarithms.
The logarithm reads each SPD element's `Cache::Log`; `logs` owns a separate
`Cache::Spectral` for each symmetric output. The explicit output policy matters:
`points.log()` defaults to `Cache::None`. `eigenvalues` owns independent vectors,
and all batch operations finish before returning. `traces` and `determinants`
each own a `MatrixBatch<Matrix<double, 1, 1>>`; read their entries with
`traces[i](0, 0)` and `determinants[i](0, 0)`. Trace sums diagonal coefficients
without preparing a spectral cache. The determinant of each symmetric logarithm
uses its existing LU-based kernel, even though `logs` has `Cache::Spectral`.
A single SPD owner, view or expression also supports
`point.log<Cache::Spectral>()` with the same output policy selection.

The same batches support coefficient summaries, eigenvectors and matrix functions:

```cpp
auto norms = logs.norm(execution_par);
auto diagonals = logs.diagonal(execution_par);
auto eigenvectors = logs.eigenvectors(execution_par);
auto reconstructed = logs.exp<Cache::Log>(execution_par);
auto roots = points.sqrt(execution_par);
auto inverse_roots = points.inv_sqrt(execution_par);
```

`norms[i](0, 0)` is the Frobenius norm; `logs.squared_norm()` returns its squared
counterpart, counting both reflected off-diagonal coefficients. Norms and diagonal
extraction read coefficients without preparing spectral data. `diagonals` owns
vectors and `eigenvectors` owns orthogonal matrices (including bases with
determinant -1). The exponential reuses the
symmetric spectral cache and reconstructs the original SPD points up to
floating-point error, with the explicitly selected `Cache::Log` output policy.
The roots act on the SPD `points` and default to uncached checked SPD outputs.

Parallel execution preserves element order and joins outstanding work before
propagating worker exceptions. Configure the worker count with
`parallel_set_num_threads(count)` before the executor is first used. Inputs must
not be modified concurrently; other accesses that prepare the same cache slots
require external synchronization. These eager methods are available on owning
batches; materialize selections or maps before using them. See
[owning spectral operations](matrix-batch.md#owning-spectral-operations) for the
execution and empty-batch contracts.

The example `examples/so_cache.cpp` compares repeated exponentials with and
without spectral caching.
