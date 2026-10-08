# Rotations and SO(n)

The standalone [geometry tutorial](../examples/manifold/02_so_maps_and_branches.cpp) demonstrates
tangent coordinates and branch selection; the
[product optimization tutorial](../examples/manifold/03_so_product_optimization.cpp) solves
a coupled rotation problem with both optimizers. Build them using the
[example instructions](../examples/README.md).

Include `<fdaPDE/manifold_geometry.h>` for geometric operations and
`<fdaPDE/manifold_means.h>` for Karcher means. Add
`<fdaPDE/manifold_solvers.h>` for optimization; see the
[module boundaries](manifold-modules.md). `RotationMatrix<Scalar, Order, Policy>`
supports positive fixed orders and `Dynamic` for runtime order.
SO(1) is the trivial group. It uses the native orthogonal expression interface,
finite coefficient checks, the existing orthogonality tolerance
`64 * order * epsilon`, and a positive determinant check. Reflections are rejected.
There is no public unchecked rotation constructor. Coefficients are read-only;
owner and view replacement prepare a complete candidate before publishing it.
`inv()` returns the checked transpose; composition returns a checked rotation.

```cpp
using Geometry = fdapde::manifold::SOGeometry<double, 3>;
Geometry geometry;
auto first = Geometry::Point::Identity();
auto omega = geometry.zero_tangent(first);
omega(0, 1) = -0.4;
auto last = geometry.exponential(first, omega);
auto curve = geometry.geodesic(first, last);
Geometry::Point midpoint(curve(0.5));
fdapde::Matrix<double, 3, 3> coefficients(curve(0.5));
```

Tangents are **body-coordinate skew matrices**: `omega` denotes the ambient
velocity `Q * omega`. The bi-invariant metric is the full Frobenius product,
`g_Q(omega, xi) = tr(omega^T xi)`, without a factor of one half. Thus a single
rotation plane of angle `theta` has distance `sqrt(2) * abs(theta)` from identity.
`from_ambient(Q, A)` and `euclidean_to_riemannian_gradient(Q, A)` return
`skew(Q^T A)`; `to_ambient` returns `Q omega`. `project(Q, omega)` preserves an
already represented skew tangent, while its generic matrix overload projects an
ambient matrix. The exact exponential is also used as the retraction. Transport
along a regular shortest branch uses `exp(-omega/2) xi exp(omega/2)` in body
coordinates. The geometry satisfies the existing first-order, geodesic and
vector-transport optimizer concepts.

## Products and joint optimization

`ProductGeometry(SOGeometry<Scalar, Order>{}, count)` operates on a native
`MatrixBatch<RotationMatrix<...>>` point and a
`MatrixBatch<SkewSymmetricMatrix<...>>` tangent. Fixed and dynamic orders are
supported. The point cache policy comes from the supplied SO geometry; tangent
rows store only their strict upper triangles. SO(1) products have dimension zero
and valid empty tangent rows.

```cpp
using Factor = fdapde::manifold::SOGeometry<double, 3>;
const fdapde::manifold::ProductGeometry geometry(Factor {}, 2);
using Product = decltype(geometry);
Product::Point points(2);  // two identities
auto direction = geometry.zero_tangent(points);
direction[0](0, 1) = -0.2;
direction[1](0, 2) = 0.1;
auto next = geometry.retract(points, direction, 0.5);
auto ambient_velocity = geometry.to_ambient(points, direction);
```

For runtime order, construct the factor as `SOGeometry<double, Dynamic>(order)`
and the point batch as `Product::Point(count, order, order)`. The product metric
sums full Frobenius body pairings. Its norm and distance combine factor values
with `hypot`; retraction, exponential, logarithm and transport act on corresponding
factors using the same tangent conventions and branch restrictions as a single
rotation. A skew owner, skew view or typed skew expression passed to `project`
already represents body coordinates and is copied unchanged. A general dense
matrix is interpreted as ambient input and projected by `skew(Q^T A)`.

The optimization problem receives the complete batch and may couple different
factors. Intrinsic `grad` and `hess` return packed skew batches. For ambient
objectives, `egrad` returns a batch of full matrices, while `ehess` receives full
ambient velocities `Q_i Omega_i` and returns their coupled ambient Hessian action.
The product converts each resulting component back to body coordinates, including
the SO connection correction; it never discards mixed-factor derivatives. Both
`RiemannianSteepestDescent` and `RiemannianTrustRegion` accept these problems through
the ordinary [objective contracts](spd-geometry.md#objective-derivatives).

## Branches and numerical limits

The relative rotation is `Q0^T Q1`; endpoint logarithms are never subtracted.
`distance` remains defined at relative eigenvalue -1. Ordinary `rotation_log`,
`geometry.logarithm` and implicit `geometry.geodesic` reject ambiguous or
numerically unresolved branches. `minimum_rotation_log` and
`geometry.minimum_logarithm` explicitly select a minimum skew logarithm and return
it together with diagnostics. Pass that result to `geometry.geodesic(Q0, Q1,
branch)` to retain the selected branch; the endpoint and minimum length are
validated. A different minimum tangent may be supplied in that result. No chosen
cut-locus branch is silently stored by the Log cache.

`RotationLogStatus` distinguishes:

- `Regular`: a resolved unique branch separated numerically from the cut locus
- `NearCut`: a resolved unique branch with a small principal-angle gap from pi
- `Ambiguous`: a negative real subspace with exactly zero skew part in the computed representation
- `Unresolved`: a nonzero skew part too small to resolve near the negative real axis, or a failed numerical plane decomposition

These are diagnostics of floating-point coefficients, not an oracle for the
exact real matrix that generated them. `cut_gap` and reconstruction `residual`
are also reported. The resolution threshold is `64 * order * epsilon`; the
near-cut threshold is its square root. `rotation_penalty_gradient` requires
`Regular`; projecting a Euclidean gradient does not require a logarithm branch.
The penalty is half the squared identity distance and its body gradient is log(Q).

The native real plane decomposition uses symmetric EVDs to jointly resolve
`(Q+Q^T)/2` and `(Q-Q^T)/2`. Near the real axis it resolves sine rather than
recovering small angles from cosine. The real symmetric lift of the skew part
avoids squaring small angles. A final orthogonality and reconstruction check
rejects unresolved decompositions. Such a failure marks a selected cache as
`Unresolved` without rejecting the verified rotation. Distance falls back to the
cosine spectrum; this fallback loses relative precision for very small angles,
and unavailable Schur factors or logarithms cannot be accessed. If even the
fallback EVD fails, the point is still constructible but distance access reports
a numerical failure. Unavailable diagnostics use NaN for the gap and infinity
for the residual. This is specialized to rotations, not a
public general nonsymmetric Schur solver. Exponentiation uses a scaled convergent
Taylor series followed by squaring and checked rotation construction. Extreme
finite tangents or parameters may be rejected when rounding destroys the
required invariants. General orders share this path; no separate 2D/3D formulas
are maintained.

For the distinction between the principal logarithm and logarithms at negative
real eigenvalues, see [Higham, What Is the Matrix Logarithm?](https://nhigham.com/2020/11/17/what-is-the-matrix-logarithm/).

## Caching and lifetimes

`RotationCache::None`, `Schur`, `Log` and `Union<...>` select per-matrix quantities.
Schur stores a real orthogonal basis and adjacent signed plane angles. Log stores
only the compact strict-upper skew logarithm when uniquely resolved, plus branch
diagnostics and identity distance. Selecting Log does not retain Schur factors;
a union prepares both from one decomposition. At a cut-locus point, construction
remains valid and Log access rejects the unresolved/nonunique branch while the
cached distance and diagnostics remain available. SO(1)'s empty logarithm reserves
one padding scalar for aggregate storage.

`RotationUsage::{IdentityDistance, IdentityLog, RotationPenalty}` selects Log
for `SOGeometry`'s point type. Metric, projection and updates need no cached
quantity. Repeated pairwise interpolation and distances require decomposition of
the relative rotation and cannot generally benefit from endpoint Log caches.

`RotationMatrixView` and `MatrixBatch<RotationMatrix<...>>` preserve verified
coefficients and caches together. Batches retain contiguous coefficient rows and
aggregate cache buffers, with independent copies. Views borrow their owners;
owner shape changes, destruction, or batch replacement/move/swap invalidate them.
Same-shape owner replacement updates existing value views and their cache binding.
A view assignment updates its existing binding. Mutable raw coefficient spans
are unavailable for rotations. Prepared curves own endpoint snapshots; `curve(t)`
borrows a persistent curve or owns a temporary one, and copies `t`. Borrowed curves
must outlive their expressions without replacement or moving. Ordinary dense
materialization reconstructs coefficients; rotation destinations additionally
certify the rounded matrix and prepare their requested cache.

The standalone `examples/benchmarks/01_matrix_function_caches.cpp` reports cache preparation separately from
repeated per-point operations and compares prepared interpolation with repeated
exponential evaluation. It does not claim reuse of endpoint logarithms for a
relative rotation. C-LE geometry and logarithm Frechet derivatives are not included.
