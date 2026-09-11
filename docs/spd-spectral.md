# SPD matrices and symmetric spectral functions

Include `<fdaPDE/linear_algebra.h>`. These operations use the native dense matrix
and symmetric eigendecomposition APIs.

## Checked SPD ownership

`SPDMatrix<Scalar, Rows, Cols>` stores the matrix coefficients in packed symmetric
storage. It does not store a logarithm or select a geometric metric. `Scalar`
must be an unqualified floating-point type. Dimensions must be either positive,
fixed and equal, or both `Dynamic`; packed storage supports `RowMajor` only.

```cpp
const fdapde::Matrix<double, 2, 2> a({4.0, 1.0, 1.0, 3.0});
fdapde::SPDMatrix<double, 2, 2> point(a, fdapde::checked);
const auto logarithm = fdapde::matrix_log(point);
const auto root = fdapde::matrix_sqrt(point);
point.assign(a * 2.0, fdapde::checked);
```

Construction and `assign(..., checked)` require nonempty square input, compatible
fixed dimensions, finite coefficients, numerical symmetry and numerical positive
definiteness. The symmetry tolerance is
`32 * dimension * epsilon<Scalar> * max_abs_coefficient`; the lower triangle is
stored. Positive definiteness requires
`min_eigenvalue > 64 * dimension * epsilon<Scalar> * max_eigenvalue` and
`max_eigenvalue > 0`, with finite eigenvalues. These are floating-point acceptance criteria: a mathematically
positive-definite matrix can be rejected when too ill-conditioned numerically.
The dense workspace must fit the supported `int` index range.

Coefficients are read-only. `rep()` borrows a const symmetric representation and
`data()` borrows a const pointer to lower-triangular coefficients packed row by
row. These borrows require a living owner and may be invalidated by replacement;
borrowing from a temporary is disabled. Copying a validated owner is permitted.
Default construction, unchecked construction and individual coefficient writes
are unavailable. Failed validation during checked assignment leaves the existing
coefficients and dimensions unchanged.

## Typed operations

| Function | Input | Owned result |
| --- | --- | --- |
| `matrix_log` | SPD expression | symmetric matrix |
| `matrix_exp` | symmetric expression | checked SPD matrix |
| `matrix_sqrt` | SPD expression | checked SPD principal square root |
| `matrix_inverse_sqrt` | SPD expression | checked SPD inverse principal square root |
| `matrix_log_frechet` | SPD point, symmetric direction | symmetric logarithm differential |
| `matrix_exp_frechet` | symmetric point, symmetric direction | symmetric exponential differential |

Results preserve the point's unqualified scalar type and static shape. Directions
must have matching dimensions and finite coefficients. Logarithm and exponential
differentials use divided differences in the eigenvector basis, including the
analytic limits at repeated eigenvalues and stable formulas for nearby values.
Results own their coefficients and can outlive input views and temporary owners.
The inputs must remain valid throughout the call.

Typed exponential, square-root and inverse-square-root results must satisfy the
same checked SPD invariant as direct construction. Exponentiation can therefore
report a domain error after underflow or loss of numerical positive definiteness,
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
