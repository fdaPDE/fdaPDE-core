# SPD matrices and symmetric spectral functions

Include `<fdaPDE/dense_linear_algebra.h>` (or the Eigen-integrating `<fdaPDE/linear_algebra.h>`). These operations use the native dense matrix
and symmetric eigendecomposition APIs.

## Checked SPD ownership

`SPDMatrix<Scalar, Order>` stores the matrix coefficients in packed symmetric
storage. Its default `Cache::None` retains no intermediates; an optional third
type parameter selects algebraic cache quantities without selecting a geometric metric. `Scalar`
must be an unqualified floating-point type. The order must be positive or `Dynamic`; packed storage supports `RowMajor` only.

```cpp
const fdapde::Matrix<double, 2, 2> a({4.0, 1.0, 1.0, 3.0});
fdapde::SPDMatrix<double, 2> point(a);
const auto logarithm = fdapde::matrix_log(point);
const auto root = fdapde::matrix_sqrt(point);
point.assign(a * 2.0);
```

Construction and `assign(...)` require nonempty square input, compatible
fixed dimensions, finite coefficients, numerical symmetry and numerical positive
definiteness. The symmetry tolerance is
`32 * dimension * epsilon<Scalar> * max_abs_coefficient`; the lower triangle is
stored. Positive definiteness requires
`min_eigenvalue > 64 * dimension * epsilon<Scalar> * max_eigenvalue` and
`max_eigenvalue > 0`, with finite eigenvalues. These are floating-point acceptance
criteria: a mathematically positive-definite matrix can be rejected when too
ill-conditioned numerically.
The dense workspace must fit the supported `int` index range.

Coefficients are read-only. `rep()` borrows a const symmetric representation and
`data()` borrows a const pointer to lower-triangular coefficients packed row by
row. These borrows require a living owner and may be invalidated by replacement;
borrowing from a temporary is disabled. Copying a validated owner is permitted.
Construction from arbitrary matrix data is explicit and always validates, without a public
validation tag. Native verified owners and views reuse their checked coefficients
and compatible retained quantities. Copies with a different scalar type are revalidated. Default construction, unchecked construction and individual
coefficient writes are unavailable. Failed validation during checked assignment leaves the existing
coefficients and dimensions unchanged.

## Typed operations

| Function | Input | Owned result |
| --- | --- | --- |
| `matrix_log` | SPD expression | symmetric matrix |
| `matrix_exp` | symmetric expression | checked SPD matrix |
| `matrix_inv` | SPD expression | checked SPD inverse |
| `matrix_sqrt` | SPD expression | checked SPD principal square root |
| `matrix_inv_sqrt` | SPD expression | checked SPD inverse principal square root |
| `matrix_log_frechet` | SPD point, symmetric direction | symmetric logarithm differential |
| `matrix_exp_frechet` | symmetric point, symmetric direction | symmetric exponential differential |

Results preserve the point's unqualified scalar type and static shape. Directions
must have matching dimensions and finite coefficients. Logarithm and exponential
differentials use divided differences in the eigenvector basis, including the
analytic limits at repeated eigenvalues and stable formulas for nearby values.
Results own their coefficients and can outlive input views and temporary owners.
The inputs must remain valid throughout the call.

Typed exponential, square-root and inverse-square-root results must satisfy the
same checked SPD invariant as direct construction. After checking finite
reconstructed coefficients, the three primitives use a private factory to certify
the rounded result's shape and actual eigenspectrum before returning an owner.
The certification decomposition also prepares the selected result cache. Neither
the trusted tag nor the factory is accessible to callers. This
avoids repeating coefficient validation in the public constructor, while retaining
the final numerical positivity check on the reconstructed matrix. Exponentiation
can therefore report a domain error after underflow or loss of numerical positive definiteness,
even if the mathematical result would be positive definite.

## Functions on ordinary native matrices

`logm`, `expm`, `powm(matrix, int)` and `sqrtm` accept real arithmetic native
matrix expressions and return owned `Matrix<double, Rows, Cols>` results. Fixed,
fully dynamic and partially dynamic shapes retain their compile-time dimensions.
Input is materialized as `double`, then checked for nonempty square shape, finite
coefficients and symmetry using the same symmetry formula with double epsilon.
These are functions of real symmetric matrices, not general nonsymmetric matrix
functions.

- `logm` requires eigenvalues above the relative spectral tolerance
- `expm` accepts indefinite symmetric input and rejects nonfinite results
- `powm` accepts an integer exponent; negative powers require every eigenvalue's
  magnitude above the relative spectral tolerance
- `sqrtm` computes the principal positive-semidefinite square root, allowing zero
  eigenvalues and clamping small negative eigenvalues within the tolerance to zero

The relative spectral tolerance is
`64 * dimension * epsilon<double> * max_abs_eigenvalue`. `sqrtm` rejects negative
eigenvalues beyond this tolerance. It evaluates square roots directly rather than
passing a fractional value to the integer-power API.

The former Eigen overloads of these four functions are replaced by this native
API. Pass native matrices or expressions directly; no compatibility overload is
provided. Existing Eigen-based randomized and sparse helpers keep their current
interfaces.

## Error reporting

Public validation and numerical failure checks remain active when debug assertions
are disabled. Invalid dimensions, asymmetry and nonfinite input produce
`std::invalid_argument`; unsupported workspace sizes produce `std::length_error`;
invalid spectra, eigensolver failure and nonfinite numerical results produce
`std::domain_error`. Invalid static SPD template parameters fail at compilation.
Usual native matrix coefficient bounds remain subject to their existing debug
assertion contracts.

## Selective cache and views

`SPDMatrix<Scalar, Order, Policy = Cache::None, StorageOrder = RowMajor>`
takes a cache policy as its third argument and an optional storage order as its
fourth. `ColMajor` remains rejected.
`Cache::Union<...>` combines `Spectral`, `Log`, `Sqrt`, `InverseSqrt` and
`LogDividedDifferences`. Unknown policy bits are rejected. Cache fields retain
only the selected scalar data, separately from packed coefficients. The owner
uses a conditional empty member for `None`, with `[[no_unique_address]]`.

`Point::Identity()` (fixed size) and `Point::Identity(order)` initialize checked
coefficients and known cache values without EVD. The logarithm is zero, root
factors and spectral eigenvectors are identity, eigenvalues are one, and every
logarithmic divided difference is one. Successful construction leaves the cache
ready; readers do not initialize or mutate it. Nonrepresentable selected
intermediates raise a numerical error during preparation.

`point.view()` returns `Point::View` or `Point::ConstView`. Copy construction of a
view preserves its binding; assignment changes the verified value in the bound
storage, with destination policy preserved. Raw SPD coefficients remain
read-only, including through generic expression assignment. `SPDLike` recognizes
only native verified owners and views; `is_spd_matrix_v` retains its broader
historical expression-contract meaning and does not authorize trusted bypasses.

Assignment prepares coefficients and cache before commit. Copies own independent
cache data. Policy expansion copies common quantities and computes only missing
ones. When expansion newly pairs spectral factors with divided differences,
the latter are rebuilt in the new basis. Same-shape assignment reuses the
destination cache buffer; shape changes invalidate views and borrowed cache
references. Owner move operations
currently preserve both values by independent copying. Batch moves transfer
aggregate buffers, as described in [MatrixBatch](matrix-batch.md).

`matrix_log` and logarithmic differentials consume compatible retained data.
`matrix_sqrt<Policy>`, `matrix_inv_sqrt<Policy>` and `matrix_exp<Policy>` return
owners with the requested destination policy (`None` by default), retaining
certification of rounded result coefficients. Input EVD and result certification
are decompositions of different matrices. A cached determinant multiplies stored
eigenvalues only when spectral factors already exist; otherwise it uses native LU.

A `LogDividedDifferences`-only policy stores L without silently retaining E.
Differential evaluation reuses L only when the same cache also retains its
spectral basis; otherwise a local EVD and its own divided differences are used
together. Cross-policy copies preserve this association, including repeated
eigenspaces. Cache inspection is read-only via `cache().eigenvalues()`,
`eigenvectors()`, `matrix<Cache::Log>()` (likewise root factors), and
`log_divided_differences()` when the corresponding quantity is selected.

### Structured inverse

`point.inv<Policy>()` and `matrix_inv<Policy>(point)` return an owning SPD inverse.
They reuse native input eigenpairs, or a retained Cholesky factor or inverse root
when eigenpairs are absent. Without useful retained factors they compute a local
symmetric eigendecomposition. Reciprocal eigenvalues and reconstructed outputs
must remain finite and numerically positive definite; output certification and
its selected cache use the rounded inverse coefficients.

`matrix.inv<Policy>()` on a symmetric matrix returns a symmetric owner, including
for indefinite invertible inputs. It refreshes and reuses native spectral caches;
uncached symmetric inputs use pivoted LU. Orthogonal `inv()` is a transpose
expression with the existing borrowing/lifetime rules. Dense, triangular,
diagonal, permutation and rotation native methods use the same `inv()` spelling
and their structure-specific algorithms. The inverse-root API is `inv_sqrt()` /
`matrix_inv_sqrt()`; cache tags retain their previous names.
