# SPD geometry

Include `<fdaPDE/manifold_optimization.h>` to use the point-typed
`fdapde::manifold::LogEuclideanGeometry<Point>` API and the compatible legacy forms
`fdapde::manifold::LogEuclideanSPDGeometry<Scalar, Order, Uses = Usage::None>` and
`fdapde::manifold::AffineInvariantSPDGeometry<Scalar, Order, Uses = Usage::None>`. This aggregate is
opt-in and is not included by `core.h`.

Both geometries expose a canonical `Point` owner with their `CachePolicy`, and
`fdapde::SymmetricMatrix<Scalar, Order>` tangents. Inputs accept `SPDLike`
owners and views with different cache policies. `retract` and `exponential`
always return the geometry's canonical `Point`. A tangent is an ambient
symmetric matrix at its base point, not a vector of logarithmic coordinates.
`SPDMatrix` remains independent of the metric. Geometry objects store their matrix
order and metric parameters; deferred means borrow their geometry and operands until
evaluation.

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
const fdapde::SPDMatrix<double, 2> point(coefficients);
const fdapde::manifold::LogEuclideanSPDGeometry<double, 2> geometry;
auto tangent = geometry.zero_tangent(point);
tangent(0, 0) = 0.2;
const auto next = geometry.exponential(point, tangent, 0.5);
// next = diag(exp(0.1), 1), independently owned
```

## Matrix element types and product geometry

Geometry templates identify one native matrix element. The point-typed APIs are
`EuclideanGeometry<Matrix>`, `LogEuclideanGeometry<SPD>`,
`AffineInvariantGeometry<SPD>`, `BuresWassersteinGeometry<SPD>`,
`LogCholeskyGeometry<SPD>` and `CheegerLogEuclideanGeometry<SPD>`. SPD aliases retain
the exact requested owner and cache policy. Metric parameters belong to the element
geometry; for example, `CheegerLogEuclideanGeometry<SPD>(epsilon)` selects its
deformation parameter. Dynamic SPD geometries take their matrix order, while dynamic
Euclidean geometries take their element shape.

`ProductGeometry(geometry, count)` owns a copy of the element metric and constructs
its product for a positive number of uniformly shaped matrices. Its `Point` is
`MatrixBatch<Geometry::Point>` and its `Tangent` is `MatrixBatch<Geometry::Tangent>`.
The matrix shape comes from the element geometry, independently of the factor count.
The product dimension is `count * geometry.dimension()`.

```cpp
using SPD = fdapde::SPDMatrix<double, 2, fdapde::Cache::Log>;
using Sym = fdapde::SymmetricMatrix<double, 2>;
const fdapde::manifold::EuclideanGeometry<Sym> symmetric_geometry;
const fdapde::manifold::ProductGeometry symmetric_product(symmetric_geometry, count);
const fdapde::manifold::CheegerLogEuclideanGeometry<SPD> cheeger_geometry(0.1);
const fdapde::manifold::ProductGeometry spd_product(cheeger_geometry, count);
```

`RiemannianTrustRegion` and `RiemannianSteepestDescent` optimize native matrix batches
using this explicit product geometry. They validate the initial count and shape
before evaluating the problem and return an owning batch. The problem supplies one
scalar cost and batch gradient/Hessian, including any cross-matrix coupling. The
steepest-descent workspace overload uses `evaluation_context_t<Problem, Geometry>`,
which also retains the ambient gradient when the objective supplies one.

Product inner products sum factor metrics; norms and distances combine factor
norms and distances with `hypot`. Retractions, logarithms, exponentials, transport
and derivative conversions apply the element geometry to corresponding factors.
Spatial GFE uses the element geometry with `MatrixBatch<SPD>` coefficients. Existing
legacy `*SPDGeometry<Scalar, Order, Uses>` calls retain their original constructors.

## Objective derivatives

An objective supplies a default-constructible, movable `Workspace` and
`cost(point, workspace)`. Derivative dispatch prefers an explicit Riemannian
`grad(point, workspace)` and `hess(point, tangent_direction, workspace)`.
The former objective names `gradient` and `hessian_vector` are no longer accepted.
`cost_gradient` can supply a joint scalar cost and intrinsic gradient when no
separate intrinsic gradient is provided.

If an intrinsic derivative is absent, supply its ambient counterpart:

- `egrad(point, workspace)` returns an owning Euclidean gradient
- `ehess(point, ambient_direction, workspace)` returns the Euclidean Hessian
  applied to that direction, rather than a dense Hessian matrix

The geometry converts these through
`euclidean_to_riemannian_gradient(point, egrad)` and
`euclidean_to_riemannian_hessian(point, egrad, ehess, direction)`. Hessian
conversion also requires `egrad`, even when an intrinsic gradient is supplied,
because the connection correction depends on it. The core supports either
mixed derivative combination, with the intrinsic derivative taking precedence
independently for each order. A geometry without the required conversion cannot
use that ambient route.

Use `egrad` and `ehess` for an ambient objective independent of the optimizer's
metric: the objective then has no geometry member or metric conversion code.
Objectives built from metric distances, GFE interpolation or intrinsic penalties
may provide `grad` and `hess` directly. These outputs are already Riemannian and
are not converted again. A change of derivative interface does not change the
metric defining the objective or its interpolation.

Euclidean, LE, AIRM, BW, LC, C-LE and SO geometries provide these conversions.
For SPD points both ambient vectors and tangents are native symmetric matrices
with the full Frobenius inner product. Derivatives are with respect to SPD
coefficients, not logarithmic coordinates. If starting from derivatives of
packed coordinates, divide off-diagonal gradient entries by two. For SO,
`to_ambient(point, direction)` converts body skew tangents to full matrix
velocities before calling `ehess`; ambient gradients and Hessian actions use
full matrices, while intrinsic outputs remain skew body tangents.

An objective on `ProductGeometry` evaluates the complete batch at once, including
all mixed-node Hessian terms. The product converts the resulting ambient
components factor by factor; it does not assume a separable cost. Its ambient
gradient cache is bound to a candidate generation, reused across truncated-CG
directions, promoted with accepted trials and cleared when the slot is reset.

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
fdapde::SPDMatrix<double, 2> result(expression);
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

## Prepared two-point geodesics

Both geometries provide `geodesic(from, to)`. Preparation accepts verified owners
or views with independent cache policies and owns a snapshot of the data needed
for the entire curve. Endpoint or geometry destruction, and subsequent endpoint
updates, do not change that snapshot.

```cpp
auto curve = geometry.geodesic(first, last);
Point midpoint(curve(0.5)); // certifies SPD and prepares the destination cache
SymmetricMatrix<double, 2> value(curve(0.5)); // reconstructs without SPD certification
Point checked(value); // certifies the stored coefficients later
points[i] = curve(t); // follows the destination type, including batch views
```

For uniformly spaced samples on `[0,1]`, `geometry.interpolate(from, to, int count)`
returns an owning `MatrixBatch<typename Geometry::Point>`. This method is
available for LE, AIRM, BW, LC and Cheeger LE geometries. The geometry's point
type determines the output scalar and order. An optional leading template
argument selects the output cache: `geometry.interpolate<OutputPolicy>(from, to, count)`.
It defaults to `Geometry::Point::CachePolicy`, independently of the endpoint
cache policies; explicit `Cache::None` requests uncached samples. A count below two throws `std::invalid_argument`.
The method prepares the two-point curve once, then evaluates each sample once
at `t = i / (count - 1)` for `i = 0, ..., count - 1`. The first and last samples
include the endpoints up to floating-point reconstruction.

A fourth argument selects `execution_seq` (the default) or `execution_par`:
`geometry.interpolate<OutputPolicy>(from, to, count, execution_par)`. Preparation
runs once before sampling; parallel evaluation shares the immutable prepared
curve and writes independent result slots directly, preserving index order and
the requested output caches. Both policies return a complete owning batch.
Worker exceptions reach the caller after all submitted work has joined.
Endpoint coefficients and caches must not be modified concurrently with
preparation. Configure workers with `parallel_set_num_threads(n)` before the
executor's first use, as for the [batch operations](matrix-batch.md).


```cpp
using SPD = fdapde::SPDMatrix<double, 2, fdapde::Cache::Log>;
const SPD A(fdapde::Vector<double, 3> {2., 0.3, 1.});
const SPD B(fdapde::Vector<double, 3> {1., 0.2, 3.});
const fdapde::manifold::LogEuclideanGeometry<SPD> geometry;
// MatrixBatch<SPD>
auto points = geometry.interpolate(A, B, 10);
// MatrixBatch<SPDMatrix<double, 2, Cache::Spectral>>
auto spectral_points = geometry.interpolate<fdapde::Cache::Spectral>(A, B, 10, fdapde::execution_par);
```

For AIRM, preparation computes `A^(-1/2) B A^(-1/2) = U Lambda U^T` and retains
`F = A^(1/2) U` and `ell = log(diag(Lambda))`. Evaluation reconstructs
`F diag(exp(t * ell)) F^T`. Only scalar exponentials and reconstruction are needed
to evaluate the expression; storage is quadratic in matrix order.

For LE, preparation retains `X = log(A)` and `D = log(B) - X`. Evaluation computes
`exp(X + t D)`. Noncommuting endpoints generally require a new decomposition of
that chart at each parameter; the endpoint logarithms are reused.

`curve(t)` is a `GeometryExpr`, evaluated globally once per materialization.
It copies the parameter, borrows a persistent curve, and owns a temporary curve.
A const temporary curve is rejected. A borrowed curve must outlive its expressions
and must not be replaced or moved while they are being used. Curve copies own
independent data. Any finite real parameter is permitted, including extrapolation
outside `[0,1]`; nonfinite parameters and nonfinite results are rejected
at materialization. Dense and symmetric destinations receive the reconstructed
coefficients without SPD certification. This includes finite results that have
become singular or numerically ill-conditioned. Only an SPD destination certifies
positive definiteness and selects its cache policy; the certification EVD also
prepares the requested cache.

The standalone examples `examples/spd_batch_interpolation.cpp` (AIRM) and
`examples/spd_batch_interpolation_le.cpp` (LE) compare cached inputs, output cache
costs, and prepared curves in a single `main` with separate scopes. Cases 4 and 5
use identical preparation and `points[i] = curve(t)` evaluation. Only the batch
element type changes: SPD in case 4, symmetric in case 5. Case 5 defers SPD
certification until after timing. Setup and allocation are timed, and every sample
is checked after timing. The ratio includes destination storage and assignment
costs as well as certification work.

## Native Euclidean parameter spaces

`EuclideanGeometry<Point>` uses native dense matrices, vectors or symmetric owners as
both points and tangents. It supplies the Frobenius inner product and affine retraction.
A symmetric off-diagonal entry contributes twice to the metric, while `dimension()`
counts only independent entries. Fixed extents can be default-constructed; dynamic
extents are supplied as `(rows, cols)` (a dynamic column vector only needs `rows`).

`ProductGeometry(EuclideanGeometry<Point>{}, count)` supplies the product Frobenius
metric for a joint native batch optimizer call.

```cpp
using Sym = fdapde::SymmetricMatrix<double, 2>;
using Batch = fdapde::MatrixBatch<Sym>;
const fdapde::manifold::ProductGeometry geometry(
    fdapde::manifold::EuclideanGeometry<Sym> {}, initial.size());
const auto solution = fdapde::manifold::RiemannianTrustRegion().optimize(problem, geometry, initial);
const auto tensors = solution.point.exp(fdapde::execution_par);
```

The problem supplies `Workspace`, `cost`, `grad` and `hess`. Gradients
satisfy `dF(L)[V] = sum_i <gradient_i, V_i>_F`; Hessian actions differentiate that same
gradient. For unscaled packed coordinates, each off-diagonal gradient entry is half
the corresponding coordinate derivative, because the Frobenius product includes
both mirrored entries. If the application evaluates SPD tensors through `S_i = exp(L_i)`, its
objective and derivatives include this map and any subsequent GFE interpolation.
This optimizes symmetric logarithms; the selected SPD geometry still defines the field
interpolation and metric-dependent objective terms. Application whitening remains a
separate coordinate change rather than an implicit property of this geometry.
