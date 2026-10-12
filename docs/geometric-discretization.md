# Geometric discretization

`GeometricSpace(scalar_space, geometry)` borrows an existing scalar discretization
and owns the target geometry. `GeometricFunction(space, controls)` owns a native
`MatrixBatch` with one geometric coefficient per scalar DOF. The same interface
supports scalar continuous P1 `FeSpace` and interval `BsSpace` of arbitrary
nonnegative degree. Vector finite elements and higher-order Lagrange bases are
outside the current support.

Include `geometric_discretization.h` for the shared facade and weighted geometric
kernels, together with `finite_elements.h` when constructing scalar P1 spaces.
`geometric_splines.h` includes both the shared facade and `splines.h`. The shared
header itself does not require Eigen; current scalar meshes and spline spaces do.
`geometric_finite_elements.h` includes the scalar FEM module and its quadrature
adapters. There is no separate `_fem` aggregate header.

```cpp
const BsSpace scalar_space(interval, 3);
const manifold::LogEuclideanGeometry<SPD> geometry;
const GeometricSpace space(scalar_space, geometry);
GeometricFunction field(space, std::move(controls));
const auto Psi = space.prepare_evaluation(grid, execution_par);
const auto values = Psi(field, execution_par);
```

## Geometry and domain compatibility

The common facade supports the log-Euclidean, log-Cholesky, affine-invariant,
Bures-Wasserstein and Cheeger SPD geometries, and special orthogonal (SO)
geometry. The target matrix order is independent of the physical domain's local
and embedding dimensions. The following list covers every triangulation type
implemented in the core:

| Scalar discretization | Triangulation | Physical domain | Target geometries |
| --- | --- | --- | --- |
| P1 `FeSpace` | `Triangulation<1, 1>` | Interval | All six families |
| P1 `FeSpace` | `Triangulation<1, 2>` | Planar linear network, including junctions | All six families |
| P1 `FeSpace` | `Triangulation<2, 2>` | Planar triangular mesh | All six families |
| P1 `FeSpace` | `Triangulation<2, 3>` | Triangulated surface in three dimensions | All six families |
| P1 `FeSpace` | `Triangulation<3, 3>` | Tetrahedral volume | All six families |
| `BsSpace` | `Triangulation<1, 1>` | Interval | All six families |

`BsSpace` itself accepts only interval triangulations. Spline fields on networks,
surfaces or volumes require a new scalar spline discretization before they can
use this facade. Euclidean vector targets remain deferred, and product geometries
used by optimization are not single matrix-valued field targets.

The registered [compatibility tests](../tests/geometric_discretization/compatibility.cpp)
exercise each family on all five P1 mesh types and on spline degrees 0, 1, 3 and
5, with and without `cache_cells`. They use nonuniform knots, branching networks,
a folded surface and adjacent tetrahedra. Values are checked against independently
assembled geometric means; spatial derivatives are checked by centered finite
differences. Coverage also includes vertices and spline endpoints, cold parallel
cell preparation, all preparation/evaluation policy combinations, native output
point/cache types, coefficient replacement and retained spatial preparations.
Both specialized 2-by-2 and general 3-by-3 Cheeger kernels are exercised.

Intrinsic SPD and SO means retain their numerical convergence and local branch
conditions on every domain. Supporting a mesh type does not guarantee a unique,
converged mean for arbitrary control values; inspect `result()` diagnostics or let
materialization report the failure.

The scalar space and its mesh must remain alive and unchanged while the geometric
space, fields, prepared evaluations, expressions or derivatives depend on them.
Temporary scalar spaces and temporary geometric spaces are rejected by the public
constructors. Fields expose coefficients only through const access. `set_coeff`
validates the new batch before replacing it and clears coefficient-dependent
local data. Obtain fresh expressions and linearizations after an update; retained
spatial preparations and previously returned owning value batches remain valid.
Updating or destroying a field requires exclusive access.

For each location, preparation retains the active control indices and scalar
basis weights. FEM uses the native P1 reference basis and retains its existing
prepared edge curves. Splines use only the active physical B-spline basis on the
located knot span. Interior knots select the right-hand span; the right endpoint
selects the final span. Both clamped endpoints are supported, including degree
zero. Nonfinite locations and locations outside the scalar mesh are rejected.
Empty evaluation retains the output matrix shape. Preparation and evaluation
accept independent `execution_seq` or `execution_par` policies; `field.eval_at`
accepts one policy for both phases.

Spline coefficients are control matrices, generally not values at the physical
knots. Log-Euclidean interpolation evaluates
`exp(sum_i B_i(x) * log(C_i))`. Other supported SPD and SO geometries use the same
weighted mean solvers and convergence diagnostics as P1 interpolation. Numerical
materialization rejects failed means instead of returning uncertified values.
Cache policies on the control matrices remain available to these shared kernels.

`field.linearization(x)` retains derivatives in active control order. Compose its
`weight_jvp` with derivatives of the physical scalar basis for spatial derivatives;
use `nodal_jvp` and `nodal_vjp` for control perturbations and their metric adjoints.
The existing low-level weighted result and differential types retain their `P1`
names; no duplicate spline mean or derivative implementation is introduced.
Cheeger fields also accept positive scalar controls through the space constructor
or `field.set_rho`, interpolated with the same basis weights as the matrix field.
P1-specific quadrature packets, lumped stencils and smoothing energies are not
implicitly generalized to splines.

See the runnable [spline field tutorial](../examples/geometric_discretization/01_spd_spline_field.cpp)
and [P1 field tutorial](../examples/geometric_finite_elements/04_spd_p1_field.cpp).
