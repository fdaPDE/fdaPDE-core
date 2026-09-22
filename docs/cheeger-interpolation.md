# Native C-LE SPD(2) interpolation

`manifold::CheegerLogEuclideanSPDGeometry<double, 2>` implements the recovered
Cheeger log-Euclidean geometry for SPD(2). Include
`fdaPDE/geometric_finite_elements.h` for native values and derivatives; also
include `fdaPDE/finite_elements.h` for spatial meshes and spaces. The native
kernels work without Eigen.

```cpp
using Geometry = fdapde::manifold::CheegerLogEuclideanSPDGeometry<
    double, 2, fdapde::Usage::InterpolationNodes>;
Geometry geometry;                         // epsilon = .5, rho = .25
Geometry local = Geometry::from_rho(.27);   // accepts rho directly
fdapde::MatrixBatch<Geometry::Point> coefficients(mesh.n_nodes());
// assign the SPD matrices in scalar P1 DOF order
std::vector<double> rho(mesh.n_nodes(), .25);
// assign strictly positive nodal rho values in the same DOF order
fdapde::GeometricFeSpace W(mesh, fdapde::P1<1>, geometry,
                         std::span<const double>(rho));
fdapde::GeometricFeFunction U(W, std::move(coefficients));
Geometry::Point value(U(x));
auto fit = U.linearization(x);
auto spatial = fit.weight_jvp(weight_direction);
auto scalar = fit.rho_jvp();
auto scalar_nodes = fit.nodal_rho_jvp(rho_direction);
auto tensors = fit.nodal_jvp(ambient_tangents);
```

The space **owns an immutable snapshot** of the scalar coefficients. It uses
the existing scalar P1 DOFs and shape functions, so
`rho(x) = sum_i w_i(x) rho_i`. Positive finite nodal coefficients give a
positive continuous P1 field. The geometry owns only a local scalar `rho`,
without a mesh dependency. Omitting the fourth constructor argument uses
constant `geometry.rho()` with no nodal scalar allocation. The three-argument
constructor and the legacy epsilon geometry constructor remain available.

The scalar field stays fixed during a fit. To change it, construct a new
space and rebind its functions and spatial evaluation plans. Matrix updates
use the existing `U.set_coeff()` cache invalidation. Plans, source lifetimes,
and sequential/parallel policies follow the [shared P1 contracts](spd-interpolation.md).
`geometry.interpolant(simplex, batch)` and `geometry.interpolant(mesh, batch)`
use a constant rho; the space supplies the variable field.

## Preparation and derivatives

Matrix coefficients use `MatrixBatch` and native SPD logarithm caches.
Fixed-rho cells prepare unique edge alignments once. Variable-rho cells
reuse spatial metadata and cached nodal logarithms, but solve the alignment
using rho at the evaluation point. A fixed-metric geodesic cannot be reused
along an edge whose metric varies.

The imported lifted multistart solver operates on the three independent
symmetric logarithmic coordinates. Value-only evaluation does not prepare
a derivative factorization. A linearization factors its positive lifted
Hessian once and reuses it for every first derivative action. The mean log
matrix also retains its native spectral cache for repeated exponential
differentials.

- `weight_jvp(dw)` includes both the interpolation weight derivative and
  `D_rho I * sum_i rho_i dw_i` when a nodal rho field is bound. Directions
  sum to zero and follow cell-local DOF order.
- `rho_jvp(drho)` changes the local scalar at fixed weights and tensors.
- `nodal_rho_jvp(d)` uses `drho = sum_i w_i d_i`.
- `nodal_jvp(d)` accepts ambient symmetric matrix tangents at the nodes.
- The recovered `nodal_log_jvp(d)` accepts symmetric **log-coordinate**
  perturbations and returns the ambient derivative directly, throwing if
  its linear solve fails. The other actions return `P1DerivativeResult`;
  inspect `converged()` before using `derivative`.

The lower-level `gfe::p1_geodesic_linearization(geometry, batch, weights,
options, rho_nodes)` binds the same local scalar field contract. Native
batch owners are borrowed, selections retain their bindings, and the span
adapter takes an owning batch snapshot. Keep borrowed coefficients unchanged
throughout a linearization's lifetime.

This import retains the existing restriction to **interior weights** for
implicit derivatives. Nonconverged means, detected ties and nonpositive or
singular lifted Hessians reject derivative evaluation. Values remain
available at vertices and edges. Mixed derivatives and adjoint operators
are not part of this C-LE import.

## Uniqueness and the future rho optimization

`result()` retains convergence, objective and `detected_ambiguity` diagnostics.
Matrix materialization rejects nonconvergence and detected ambiguity.
Only stationary competing multistart candidates count toward a detected
tie. A positive local Hessian or the absence of detected ties **does not
certify global uniqueness**, and the result keeps `not_certified` for
nonvertex means. Branch switches can also prevent global continuity even
though rho itself is continuous.

The future target is a rho field as close as possible to `1/4`, subject to
an appropriate uniqueness guarantee. No optimizer or uniqueness certificate
is implemented here. Before adding one, define the norm measuring field
deviation, the scope of the uniqueness constraint (evaluation sites or the
whole mesh), and a certifiable condition. The two-point strict-convexity
bound is not a certificate for every multinode mean or for the complete
smoothing objective. The variable-rho smoothing penalty/discrete tension
also requires its own definition; this change covers interpolation.

Native tests retain the broad-data branch discontinuity and the captured
TSPDE epsilon probe: increasing epsilon from `.5` to `.505` resolves the
observed local near-tie without claiming uniqueness. The captured
unfinished-candidate regression is exercised on its original mesh and sites.
