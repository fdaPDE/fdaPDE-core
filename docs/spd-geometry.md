# SPD geometry

Include `<fdaPDE/manifold_optimization.h>` to use
`fdapde::manifold::LogEuclideanSPDGeometry<Scalar, Order, Uses = Usage::None>` and
`fdapde::manifold::AffineInvariantSPDGeometry<Scalar, Order, Uses = Usage::None>`. This aggregate is
opt-in and is not included by `core.h`.

Both geometries expose a canonical `Point` owner with their `CachePolicy`, and
`fdapde::SymmetricMatrix<Scalar, Order, Order>` tangents. Inputs accept `SPDLike`
owners and views with different cache policies. `retract` and `exponential`
always return the geometry's canonical `Point`. A tangent is an ambient
symmetric matrix at its base point, not a vector of logarithmic coordinates.
`SPDMatrix` remains independent of the metric. Geometry objects store only their
order; deferred means borrow their geometry and operands until evaluation.

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
`VectorTransportGeometry` concepts describe the operations available to
optimization and interpolation consumers.

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
Eigen-free on this incremental branch. Both aggregate inclusion orders are checked, including
Eigen matrix/vector classification and a multi-translation-unit link.

## Usage and destination policies

Uses combine with `|`; their cache requirements are unioned without duplicated
quantities. The default `Usage::None` preserves the uncached API.

| Usage | Log-Euclidean | Affine-invariant |
|---|---|---|
| `Distance` | Log | InverseSqrt |
| `InterpolationNodes` | Log | None |
| `TangentMetric` | Spectral, LogDividedDifferences | InverseSqrt |
| `LogExpDifferentials` | Spectral, LogDividedDifferences | Spectral, LogDividedDifferences |
| `BasePointMaps` | Log, Spectral, LogDividedDifferences | Sqrt, InverseSqrt |

`LogExpDifferentials` refers to algebraic logarithm/exponential differentials;
it does not advertise additional AIRM derivative APIs. AIRM interpolation nodes
are distinct from iterative base points, whose factors require `BasePointMaps`.
LE applies the exponential differential at log(P) using reciprocal logarithmic
divided differences when a coherent spectral basis is retained. AIRM borrows
cached root factors as internal symmetric intermediates and still certifies each
returned SPD owner. Ambient coordinates, transport and the second-order AIRM
retraction remain unchanged.

## Deferred log-Euclidean weighted mean

```cpp
using Geometry = fdapde::manifold::LogEuclideanSPDGeometry<
    double, 2, fdapde::Usage::InterpolationNodes>;
Geometry geometry;
fdapde::MatrixBatch<Geometry::Point> points(3);
fdapde::Vector<double, 3> weights(1.0, -0.5, 2.0);
auto expression = geometry.weighted_mean(points, weights);
fdapde::SPDMatrix<double, 2, 2> result(expression);
```

The expression computes exactly `exp(sum_i weights[i] * log(points[i]))`, with
finite real weights of either sign and any sum. It never normalizes. Zero weights
are accepted; an empty batch/selection with a known element shape, and all-zero
weights, produce identity. Weight count and point/geometry shape must agree.
Overflow or numerical loss of positivity raises the native numerical error.

Construction does not evaluate the mean. Each conversion or assignment evaluates
the complete combination once, using the destination SPD cache policy. Native
`Matrix` construction/assignment shares this global evaluation path. A
`GeometryExpr` is not a verified SPD value and has no coefficient evaluator.

Persistent points, weights and geometry are borrowed, so their current values
are observed when evaluated. Temporary selections and expression nodes are owned
by the containing expression; temporary owning batches, native weight vectors
and geometry objects are rejected. Owners must outlive every borrowing
expression. Sources must not be modified concurrently with evaluation.

See [MatrixBatch](matrix-batch.md) for element views, selections and map/redux.
AIRM iterative means, C-LE, interpolation derivatives/adjoints, optimizer
integration, FEM assembly and parallel execution are not provided by this API.
