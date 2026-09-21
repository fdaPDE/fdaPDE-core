# Native SPD P1 interpolation

Include `fdaPDE/geometric_finite_elements.h` for LE/AIRM P1 values, linearizations
and prepared spatial interpolants. This header and its algebra/solver paths do
not require Eigen. The existing `Simplex` and mesh APIs still use Eigen spatial
coordinates; include `fdaPDE/geometry.h` when using them.

```cpp
using namespace fdapde;
using Node = SPDMatrix<double, 3, 3, Cache::Log>;
MatrixBatch<Node> nodal_values(source_values);
auto local_values = nodal_values.select(vertex_ids);

manifold::AffineInvariantSPDGeometry<double, 3, Usage::BasePointMaps> geometry;
gfe::P1GeodesicLinearizationOptions options;
options.mean.solver.gradient_tolerance = 1e-10;
auto interpolant = geometry.interpolant(element, local_values, options);
auto expression = interpolant(x);
SPDMatrix<double, 3, 3> value(expression);
```

`element` is a simplex or simplex mesh cell; `x` has its `NodeType`. The
interpolant owns the cell coordinates and computes barycentric weights with
`Simplex::barycentric_coords`. Local values must follow the element's vertex
order. Points outside the simplex, including off-plane points of embedded
simplices, are rejected. The P1 contract requires finite nonnegative weights
summing to one within its rounding tolerance; it does not extrapolate.

Use `LogEuclideanSPDGeometry<double, 3, Usage::InterpolationNodes>` for LE.
`Cache::Log` on the batch avoids recomputing node logarithms in LE and in the
LE initializer of AIRM. Add `Cache::Spectral` and
`Cache::LogDividedDifferences` through `Cache::Union` when nodal derivatives
are frequent. `Usage::BasePointMaps` caches AIRM candidate roots. Cache
selection remains explicit; caches require storage and allocate memory.

The batch is borrowed and must remain alive and unchanged while using the
interpolant, a borrowed linearization, or their expressions. Temporary batch
owners are rejected. Temporary selections are retained by value; their source
batch is still borrowed. A selection built from an lvalue expression can itself
borrow that expression, according to `MatrixBatch::select` nesting rules.
An expression from an lvalue interpolant borrows it; an expression from a
temporary interpolant owns it. Rebuild the interpolant after replacing nodal
data. Concurrent evaluations of an immutable interpolant have separate solver
workspaces and can share the ready node caches through `parallel_async`.

Vertices return their certified nodal value. Each simplex edge has a prepared
geodesic, evaluated directly when exactly two weights are active. Interior LE
uses its compensated logarithmic mean. AIRM with at least three active nodes
uses the existing weighted Karcher mean with Armijo steepest descent, followed
by at most six local stationarity polishing steps when the residual is below
`1e-5`. The polishing step uses the existing tangent CG Hessian solve and
requires a residual contraction, accommodating the cost roundoff floor.
All SPD values and relative matrices remain certified `SPDMatrix` objects.

`interpolant.result(x)` exposes the candidate, normalized weights, iteration
count, residual, stop reason and uniqueness certificate. Materializing
`interpolant(x)` throws `std::runtime_error` with stop reason, iterations and
residual if the mean does not converge. Invalid input and failed SPD
certification retain their `invalid_argument`/`domain_error` exceptions.

```cpp
auto linearization = interpolant.linearization(x);
auto spatial_action = linearization.weight_jvp(shape_direction);
auto nodal_action = linearization.nodal_jvp(nodal_directions);
auto nodal_pullback = linearization.nodal_vjp(value_metric_dual);
```

`shape_direction` is the finite zero-sum derivative of the P1 shape weights in
the desired spatial direction. Derivatives use the represented normalized
weight total. LE actions return tangents directly. AIRM actions return a
`P1DerivativeResult`; inspect `converged()` before using `derivative`.
`covariant_mixed_nodal_jvp` and `covariant_mixed_nodal_vjp` are also retained;
AIRM mixed results contain three solve certificates. The adjoints use the
geometry metric, not the ambient Frobenius pairing. An edge linearization
retains every node: a transverse weight variation can activate a third node.

AIRM retains the scaled relative spectrum, logarithm and logarithm divided
differences for each active node. `EvaluationContext` owns separate current
and trial slots: an accepted trial is promoted, a changed candidate invalidates
its frames. The converged workspace moves into the linearization, which also
prepares inactive nodes needed by transverse derivatives. Cache reuse is local
to that candidate; it does not carry across unrelated spatial points.

The lower-level `gfe::p1_geodesic_value` and `gfe::p1_geodesic_linearization`
accept explicit barycentric weights. Legacy contiguous spans remain supported;
a span-based linearization owns a `MatrixBatch` snapshot, while a batch-based
linearization borrows the batch. Solver diagnostics remain available through
`manifold::weighted_karcher_mean` and the bounded optimization solvers.
