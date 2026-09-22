# Native rotation P1 interpolation

Include `fdaPDE/geometric_finite_elements.h` for rotation values and derivatives.
Include `fdaPDE/finite_elements.h` as well when using meshes or finite element
spaces. The native value and derivative kernels do not require Eigen.

```cpp
using Geometry = fdapde::manifold::SOGeometry<double, 3>;
using Rotation = fdapde::RotationMatrix<double, 3, 3>;
Geometry geometry;
fdapde::MatrixBatch<Rotation> coefficients(mesh.n_nodes());
// assign certified nodal rotations in the scalar P1 DOF order
fdapde::GeometricFeSpace W(mesh, fdapde::P1<1>, geometry);
fdapde::GeometricFeFunction U(W, std::move(coefficients));
Rotation value(U(x));
auto local = U.linearization(x);

auto weight = local.weight_jvp(weight_direction);
auto nodal = local.nodal_jvp(nodal_directions);
auto pullback = local.nodal_vjp(value_dual);
auto mixed = local.covariant_mixed_nodal_jvp(weight_direction, nodal_directions);
auto mixed_pullback = local.covariant_mixed_nodal_vjp(weight_direction, value_dual);
```

The derivative results carry a `converged()` certificate and a `derivative`
candidate, exactly as for AIRM. Mixed results retain the three dependent solve
statuses. Directions and metric duals are **body-coordinate skew matrices**:
a tangent `Omega` at `Q` represents ambient velocity `Q * Omega`. The metric
is the full Frobenius product, including both triangles. Weight directions
are finite, sum to zero, and follow cell-local DOF order.

`geometry.interpolant(simplex, batch)` and
`geometry.interpolant(mesh, batch)` also support rotations. The lower-level
`gfe::p1_geodesic_value` and `gfe::p1_geodesic_linearization` accept native
batches/selections; the span linearization overload takes an owning snapshot.
Fixed and dynamic rotation orders use the same implementation.

```cpp
fdapde::MatrixBatch<fdapde::Vector<double, 2>> locations;
// fill locations with spatial coordinates
const auto evaluation = W.prepare_evaluation(locations, fdapde::execution_par);
auto values = evaluation(U, fdapde::execution_par); // MatrixBatch<Geometry::Point>
U.set_coeff(updated_coefficients);
auto updated_values = evaluation(U); // reuses spatial preparation
```

Ownership, `set_coeff()` invalidation, independent execution policies, and
lowest-index parallel failure propagation follow [the shared P1 contracts](spd-interpolation.md).
Plans borrow their space, while outputs own their rotation storage. Concurrent
evaluations are supported; coefficient replacement requires exclusive access.

## Values and branch policy

Vertices reproduce the selected rotation exactly. Edges use the existing
prepared shortest rotation geodesic. An ambiguous or unresolved edge is
recorded during cell preparation and rejected only when that edge is used;
it does not prevent evaluating a different vertex or regular edge.

Interior values use the existing weighted Karcher solver. The initializer is
the node with the largest barycentric weight, taking the first in local node
order on ties. The generic solver retains candidate-relative logarithms so
cost and gradient share one rotation decomposition per active node. Its local
roundoff refinement is shared with AIRM and accepts only small Newton steps
that contract the stationarity residual. Refinement can prepare derivative
operators when a tight tolerance reaches the cost roundoff floor.

A needed ambiguous/unresolved relative logarithm raises `std::domain_error`;
P1 never silently selects a branch at the cut locus. Convergence in the
interior certifies stationarity, **not global uniqueness or a global
minimum**. Widely separated data can have different local means; switching
initializers can switch branches. Applications requiring a continuous field
must keep the nodal data in a suitable common convex neighborhood.

## Derivative preparation and limitations

Linearization retains relative logarithms, tangent-space eigensystems and
Hessian divided differences at the converged mean. Repeated JVP/VJP calls
reuse them. No finite differences are used in production. Local mean
Hessians are checked for positive definiteness before using the existing CG
solver. Singular, indefinite or numerically unresolved Hessians raise
`std::domain_error`; failed CG solves retain their explicit diagnostics.

All nodal relative branches must be regular when building a linearization,
including zero-weight nodes, because weight and mixed derivatives can activate
them. Numerically near-cut branches are rejected for derivatives even when a
unique value logarithm is still available. A failed mean can be inspected
through `result()` but cannot be differentiated.

For `m = n(n-1)/2`, each relative derivative preparation stores O(m^2) scalars
and uses a native O(m^3) symmetric eigensolve. A cached Hessian or target-log
JVP/VJP costs O(m^2); a mixed action/pullback costs O(m^3). This is compact for
SO(2)/SO(3) and supports general SO(n), but is not an asymptotically optimized
large-order rotation kernel. Edge values and ordinary mean iterations do not
construct these derivative spectra. Native matrix index bounds are checked
before allocating tangent-space operators.

## Analytic operators

The implementation uses an orthonormal skew basis with coordinates
`sqrt(2) * Omega(i,j)`, `i < j`. Let `L = log(Q.transpose() * R)`,
`K = ad_L`, `A = -K^2` and `H = f(A)`, where

```
f(a) = sqrt(a)/2 * cot(sqrt(a)/2),  f(0) = 1
D_target log[v] = (H + K/2) v
Hess_Q (distance(Q,R)^2/2)[u] = H u
```

The body-coordinate logarithm variation for endpoint directions `(u,v)` is
`delta_L = -H*u + K*u/2 + H*v + K*v/2`.
Differentiating `A` gives
`delta_A = -(ad_delta_L*K + K*ad_delta_L)`. The covariant Hessian variation
on a parallel input `w` is

```
Df(A)[delta_A] w + (ad_u H - H ad_u) w / 2
```

`Df` uses analytic spectral divided differences with continuous limits at
zero and repeated frequencies. The mixed pullback reverses this same
self-adjoint Frechet operator and the sparse commutator construction; it
does not evaluate a forward action once per basis vector. The shared implicit
P1 implementation then differentiates the mean stationarity equation.

The body-coordinate logarithm Jacobian convention agrees with the inverse
right Jacobian in [Sola, Deray and Atchuthan, *A micro Lie theory for state
estimation in robotics*](https://arxiv.org/abs/1812.01537).
The general-order commutator/Hessian construction above is documented here
and checked independently by endpoint perturbations and parallel transport.

Active tests are `tests/geometric_finite_elements/rotations.cpp` and
`tests/geometric_finite_elements/rotation_spatial.cpp`. They cover commuting
angle oracles, noncommuting endpoint and mean derivatives, metric adjoints,
repeated frequencies, branch/curvature failures, native materialization,
cache replacement and all sequential/parallel policy pairs.
