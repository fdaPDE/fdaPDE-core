# Native SPD P1 interpolation

Include `fdaPDE/geometric_finite_elements.h` for LE/AIRM P1 values, linearizations
and prepared spatial interpolants. This header and its algebra/solver paths do
not require Eigen. The existing `Simplex` and mesh APIs still use Eigen spatial
coordinates; include `fdaPDE/geometry.h` when using them.

The finite element interface pairs a scalar reference basis with the target
matrix geometry. Include `fdaPDE/finite_elements.h` as well for `FeSpace`, mesh
types and the `P1` descriptor; either include order is supported.

```cpp
using namespace fdapde;
using Node = SPDMatrix<double, 3, 3, Cache::Log>;
MatrixBatch<Node> values_batch(source_values);
manifold::AffineInvariantSPDGeometry<double, 3, Usage::BasePointMaps> geometry;
GeometricFeSpace W(mesh, P1<1>, geometry);
GeometricFeFunction U(W, values_batch);  // copies the batch
// use GeometricFeFunction U(W, std::move(values_batch)) to transfer its storage
auto expression = U(x);
SPDMatrix<double, 3, 3> value(expression);
```

`GeometricFeSpace` owns an existing scalar `FeSpace` and a copy of the target
geometry. `triangulation()`, `dof_handler()`, `n_dofs()`,
`eval_shape_value(i, reference_point)` and `geometry()` expose its immutable
bindings. Each scalar DOF weights a complete matrix: the SPD order is set by
`geometry`, not by scalar vector components. Only `P1<1>` is accepted;
`P2<1>`, other degrees and vector descriptors such as `P1<3>` fail compilation.

`GeometricFeFunction` owns its `MatrixBatch`, preserving the batch's element
cache policy. An lvalue is copied; an rvalue transfers its buffers.
`function_space()` returns the borrowed space and `coeff()` returns a const
batch reference. It uses the `DofHandler` table in local DOF order and evaluates
weights through the scalar reference basis, reusing the prepared P1 engine.
Optional solver tolerances are the third constructor argument, as in
`GeometricFeFunction U(W, values_batch, options)`.

`U.set_coeff(new_values)` validates the count and matrix shape, discards
coefficient-dependent prepared cells, and copies or transfers the replacement
batch. It retains the spatial index. A failed validation preserves the previous
coefficients, cache and expressions. A successful replacement invalidates all
previously obtained expressions, linearizations and coefficient views: recreate
them before evaluation. External changes to the original lvalue batch have no
effect on `U`.

The immutable mesh must outlive `W`, and `W` must outlive `U`. Spaces and
functions are not copyable or movable, preserving internal and borrowed
bindings. Expressions and linearizations must not outlive `U`; temporary mesh
and space bindings and deferred evaluation on temporary functions are rejected.
Concurrent evaluations are supported; `set_coeff` and destruction require
exclusive access, including exclusion of evaluations through existing borrowed
expressions or linearizations.

`U.result(x)`, `U.linearization(x)` and `U.prepared_cells()` have the local
solver diagnostics, derivative ordering and lazy cell preparation contracts
described below. No global derivative assembly or higher-order interpolation is
provided.

## Prepared interpolation on one simplex

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
For a single simplex, an expression from an lvalue interpolant borrows it; an
expression from a temporary interpolant owns it. Rebuild the interpolant after
replacing nodal data. Concurrent evaluations of an immutable interpolant have separate solver
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

## Interpolation over a mesh

```cpp
auto interpolant = geometry.interpolant(mesh, nodal_values, options);
auto expression = interpolant(x);
SPDMatrix<double, 3, 3> value(expression);
```

The same member dispatches LE and AIRM to `gfe::P1FieldInterpolant`. The batch
contains one value per global mesh vertex. For each query, an independent
`TreeSearch` finds a containing cell; its connectivity supplies the indices for
`MatrixBatch::select` in local vertex order. The selected data feed the same
prepared P1 simplex interpolant used by the cell API. Intervals, planar and
embedded triangular meshes, linear networks and tetrahedral meshes use this
path. No interpolation algorithm is duplicated.

Construction builds the spatial index but prepares no local interpolation data.
The first visit to a cell prepares its edge curves; subsequent visits reuse the
same cell object. `prepared_cells()` reports this count. Storage grows with the
number of visited cells and is retained until field destruction; the spatial
index covers the full mesh. In particular, intervals currently use this common
tree index rather than a specialized interval locator.

The mesh, nodal batch and any borrowed source expressions must remain alive and
unchanged. The field is not copyable or movable, keeping its cached references
stable. Expressions and linearizations must not outlive the field; deferred
calls on a temporary field and construction from a temporary mesh are rejected.
Temporary batch selections are retained by value with their usual source
lifetime contract. Rebuild the field after modifying mesh or nodal data.

Queries on one field or independent fields sharing an immutable mesh can run
concurrently. Their spatial indices read mesh data without using the mesh's
mutable cell scratch or lazy locator. A mutex protects cache lookup and first
cell preparation; interpolation and derivative solves run outside that lock
with separate workspaces. Destruction and mutation require external exclusion.
This does not change the concurrency contract of other mesh methods.

`result(x)` and `linearization(x)` expose the existing local diagnostics and
derivatives. Derivative arrays follow the located cell's local vertex order;
they are not assembled into a global nodal vector. On shared faces, any
containing cell may be chosen: P1 values agree, while derivatives are those of
the chosen cell. Points outside the mesh, including off-surface points, and
nonfinite coordinates are rejected.
