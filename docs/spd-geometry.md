# SPD geometry

Include `<fdaPDE/manifold_optimization.h>` to use
`fdapde::manifold::LogEuclideanSPDGeometry<Scalar, Order>` and
`fdapde::manifold::AffineInvariantSPDGeometry<Scalar, Order>`. This aggregate is
opt-in and is not included by `core.h`.

Both geometries use root native `fdapde::SPDMatrix<Scalar, Order, Order>` points
and `fdapde::SymmetricMatrix<Scalar, Order, Order>` tangents. A tangent is an
ambient symmetric matrix at its base point, not a vector of logarithmic
coordinates. `SPDMatrix` remains independent of the metric and has no geometry
cache. Every returned point or tangent owns its coefficients; geometry objects
store only their order and retain no references to arguments.

A positive fixed order is default-constructed. For `Order = fdapde::Dynamic`,
construct the geometry with a positive order. `order()` returns the matrix
order; `dimension()` returns `order * (order + 1) / 2`. The same dense workspace
bound as SPD spectral operations applies. Packed storage is row-major.

```cpp
#include <fdaPDE/manifold_optimization.h>

fdapde::Matrix<double, 2, 2> coefficients;
coefficients.set_zero();
coefficients(0, 0) = 1;
coefficients(1, 1) = 1;
const fdapde::SPDMatrix<double, 2, 2> point(coefficients);
const fdapde::manifold::LogEuclideanSPDGeometry<double, 2> geometry;
auto tangent = geometry.zero_tangent(point);
tangent(0, 0) = 0.2;
const auto next = geometry.exponential(point, tangent, 0.5);
// next = diag(exp(0.1), 1), independently owned
```

## Metric and maps

Write `L_P(U) = D log(P)[U]`, `E_A(U) = D exp(A)[U]`, and
`<U,V>_F = tr(U V)` for symmetric matrices. Off-diagonal entries contribute twice
in this inner product, including when stored only once in packed storage.

| Operation | Log-Euclidean | Affine-invariant |
|---|---|---|
| `inner_product(P,U,V)` | `<L_P(U),L_P(V)>_F` | `tr(P^-1 U P^-1 V)` |
| `exponential(P,U,t)` | `exp(log(P) + t L_P(U))` | `P^1/2 exp(t P^-1/2 U P^-1/2) P^1/2` |
| `logarithm(P,Q)` | `E_log(P)(log(Q) - log(P))` | `P^1/2 log(P^-1/2 Q P^-1/2) P^1/2` |
| `distance(P,Q)` | `||log(Q) - log(P)||_F` | `||log(P^-1/2 Q P^-1/2)||_F` |
| `euclidean_to_riemannian_gradient(P,G)` | `E_log(P)(E_log(P)(G))` | `P G P` |

`project(P,U)` copies an already symmetric ambient tangent. It does not accept
an arbitrary nonsymmetric matrix. `zero_tangent` and `linear_combination` return
owning ambient tangents. `norm` uses scale-safe Frobenius accumulation after the
metric's chart or whitening map. `transport(P,Q,U)` is parallel transport along
the unique geodesic: LE preserves `L_P(U)` in logarithmic coordinates; AIRM
uses the relative principal square root and congruence transformations.

`logarithm(P,Q)` is the negative Riemannian gradient at `P` of
`distance(P,Q)^2 / 2`. The exposed `FirstOrderGeometry`, `GeodesicGeometry`, and
`VectorTransportGeometry` concepts describe the operations needed by subsequent
optimization and interpolation slices; they do not implement an optimizer.

LE `retract` equals its exponential. AIRM deliberately retains the second-order
retraction `P^1/2 (I + W + W^2/2) P^1/2`, where
`W = t P^-1/2 U P^-1/2`. It is distinct from the exact exponential; this preserves
the retraction intended for Armijo optimization.

## Validation and numerical limits

Shape, finite input, and public scalar-coefficient checks are always active,
including under `FDAPDE_NO_DEBUG`. Invalid inputs throw `std::invalid_argument`;
oversized dynamic orders throw `std::length_error`. Unrepresentable arithmetic
results throw `std::domain_error`. Public double steps and combination weights
must be representable in the geometry's scalar type before conversion.

Spectral operations retain the positivity, conditioning, eigensolver, and
representability limits documented in [SPD spectral operations](spd-spectral.md).
A mathematically positive point or AIRM relative point can be rejected at these
numerical limits. The geometry does not promise arbitrary-condition-number or
arbitrary-scale arithmetic. Operations build results in local owning storage;
a failed operation leaves all input values unchanged, including in assignments
such as `point = geometry.retract(point, tangent, step)`.

## Native dependency boundary

`<fdaPDE/dense_linear_algebra.h>` exposes the native dense, structured, SPD and
spectral layer shared by the geometry and the existing linear algebra aggregate.
It has no Eigen includes or Eigen build dependency. The geometry and native test
targets use this header directly. `FDAPDE_NATIVE_ONLY=ON` builds them without
finding or linking Eigen and rejects a configuration where `<Eigen/Eigen>` is
compiler-discoverable. CI also removes the Eigen development package in this lane.

The older `<fdaPDE/linear_algebra.h>` still includes Eigen and the existing
Eigen-backed sparse/randomized helpers. The complete core is therefore **not**
Eigen-free on this incremental branch. This split supplies the minimal native
prerequisite for SPD geometry without importing the sparse/FEM migration from
`develop-resurrection`. Both aggregate inclusion orders are checked, including
Eigen matrix/vector classification and a multi-translation-unit link.

Weighted means, GFE interpolation and linearizations, nodal derivatives,
second Fréchet derivatives, optimizer implementations, C-LE, prepared-point
caches and global smoothing objectives are outside this slice.
