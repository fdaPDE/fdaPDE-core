# Cached symmetric matrices

`CachedSymmetricMatrix<Scalar, Rows, Cols, Policy = Cache::Spectral>` is an opt-in
symmetric owner available through `<fdaPDE/dense_linear_algebra.h>`. It supports
`Cache::None`, `Cache::Spectral` and their unions. Spectral eigenpairs are the
reusable quantities required by the supported matrix functions; no SPD-only
logarithm or square root is assumed to exist. Existing `SymmetricMatrix` keeps its
ordinary mutable storage interface.

```cpp
using Cached = fdapde::CachedSymmetricMatrix<double, 2, 2>;
Cached matrix(fdapde::Matrix<double, 2, 2>({2, 0.3, 0.3, 4}));
auto coefficient = matrix(0, 0);
auto first = fdapde::expm(matrix); // prepares and reuses spectral data
coefficient = 5;                 // invalidates even this previously saved alias
auto second = fdapde::expm(matrix); // refreshes the decomposition
```

Every coefficient write, including one through `CachedSymmetricMatrixView` or a
saved proxy, invalidates the shared slot. The next `cache()` access or spectral
operation computes fresh eigenpairs. Whole-value assignments validate a finite
symmetric candidate before committing it. Packed symmetric writes update both
reflected coordinates. Raw mutable `data()`, representation views, and batch
coefficient spans are deliberately unavailable: they would allow later writes
without invalidation. Use coefficient proxies or whole-value assignment.

Cache references and borrowed eigenvector/eigenvalue views must not be retained
across mutation. A retained slot detects stale access; previously obtained raw
factor views are invalidated by mutation. Owner shape changes, destruction, and
batch replacement/move/swap invalidate value views and coefficient proxies as
well. Same-shape owner replacement preserves value and invalidation bindings. Owner and batch copies have independent coefficients and cache state.
Concurrent access that can populate the same cold cache requires external
synchronization, as do writes; prepare caches before parallel read-only use.

The native EVD constructor reuses coherent cached factors. `expm`, `logm`, `sqrtm`,
`powm`, checked `matrix_exp` and symmetric Frechet operations thereby avoid
recomputing a ready decomposition at matching scalar precision. Legacy functions
that promote float inputs to double recompute in double rather than silently
reusing lower-precision factors. Domain checks still run: an indefinite
symmetric owner is valid, while its real SPD logarithm is rejected. Matrix
functions of expressions such as `matrix + other` decompose the resulting matrix;
the operands' eigenpairs do not in general determine that sum's eigenpairs.

`MatrixBatch<CachedSymmetricMatrix<...>>` uses packed contiguous coefficient rows
and aggregate spectral buffers. Its mutable views share slot invalidation with
the batch. Selection and map preserve cached symmetric result ownership. The
example `examples/so_cache.cpp` compares repeated exponentials with and without
spectral caching.
