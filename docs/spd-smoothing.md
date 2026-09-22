# Native LE/AIRM smoothing kernels

Include `fdaPDE/geometric_finite_elements.h` for the Eigen-free objective kernels
and trust-region solver. Include `fdaPDE/geometric_finite_elements_fem.h` to
assemble their spatial data from the existing FEM module, which uses Eigen.

## Nodal storage and spatial assembly

```cpp
using namespace fdapde;
using Geometry = manifold::LogEuclideanSPDGeometry<
  double, 2, Usage::InterpolationNodes | Usage::LogExpDifferentials>;
Geometry geometry;
GeometricFeSpace space(mesh, P1<1>, geometry);
MatrixBatch<Geometry::Point> nodes(source_tensors); // one SPD tensor per scalar DOF
const auto stencil = gfe::p1_lumped_laplacian_stencil(space, QS2DP4);
const auto cell = gfe::p1_fem_cell_quadrature(space, 0, QS2DP4);
const auto local_nodes = nodes.select(cell.dofs); // borrows the batch and its caches
```

For AIRM, use `manifold::AffineInvariantSPDGeometry<double, 2,
Usage::BasePointMaps>`. Fixed and dynamic SPD orders are supported. The stencil
can also be assembled from a scalar `FeSpace`, or directly from a span of
`P1FEMCellQuadrature` packets without Eigen. Assembly covers the full continuous
P1 space, with the natural homogeneous Neumann convention and no boundary DOF
elimination. It supports simplices of local dimension 1–3, including embedded
meshes. Quadrature weights must be nonnegative and sum to one.

Nodal arguments accept `MatrixBatch`, its selections, and legacy spans. The
batch's element policy controls persistent SPD caches. Selections borrow those
caches; keep their owner alive and avoid mutation during evaluation. Temporary
residuals and gradients use vectors of native symmetric tangents. Cell kernels
select the original batch; the compatibility path for spans builds a local
batch. No Eigen conversion is used inside the matrix objective kernels.

## Data term and smoothing penalty

For the LE/AIRM tensor smoothing objective, use geodesic interpolation in both
the data term and prediction:

```cpp
Geometry::Tangent observation(observed_tensor);
const auto data = gfe::p1_frobenius_data_site_contribution(
  geometry, local_nodes, barycentric_weights, observation);
const auto penalty = gfe::p1_discrete_tension_contribution(geometry, nodes, stencil);
if (!data.converged() || !penalty.converged()) {
    // reject this evaluation before passing it to the optimizer
}
```

The data contribution is `0.5 * ||I(P, w) - D||_F^2`, where `I` is the LE mean or
AIRM Karcher mean. The observation is a symmetric native tangent containing the
coefficients of `D`. Returned gradients are **Riemannian metric gradients**,
not derivatives with respect to packed coefficients. Data gradients follow the
supplied local node order; the caller scatters them into the global DOFs and
applies observation weights or division by the observation count.

Let `K` be the scalar stiffness matrix and `m_i` the lumped mass. The discrete
penalty is

\[
 r_i = \sum_{j\ne i} K_{ij}\operatorname{Log}_{P_i}(P_j),
 \qquad E(P)=\frac12\sum_i\frac{\|r_i\|_{P_i}^{2}}{m_i}.
\]

`p1_discrete_tension_contribution` returns this value and gradients in global
node order. Apply the smoothing parameter `lambda` in the caller. In LE,
`E = 0.5 * ||M^{-1/2} K log(P)||_F^2`, so the implementation works directly in
log coordinates. In AIRM, each directed edge's relative frame is computed once
and reused for its logarithm and both gradient actions. The value-only path
avoids retaining these frames. Signed stiffness edges, including positive
off-diagonals from obtuse cells, are valid for discrete tension.

The imported alternatives remain explicitly named:

- `p1_log_coordinate_data_site_*` (LE): residual against a precomputed `log(D)`
- `p1_ambient_frobenius_data_site_*`: Frobenius loss of arithmetic P1 interpolation
- `p1_dirichlet_cell_*`: quadrature of half the squared spatial metric derivative;
  gradients follow `packet.dofs`, requiring global scatter
- `p1_squared_distance_edge_dirichlet_*`: half the weighted sum of squared
  geodesic edge distances, requiring strictly negative off-diagonal stiffness

Each pair exposes `_value` and `_contribution`. These penalties describe
different objectives; the TSPDE fit uses discrete tension. AIRM geodesic data
contributions and cell Dirichlet kernels accept `P1GeodesicLinearizationOptions`.
A failed barycenter or derivative solve sets `first_failure`; partial values
and gradients must not be consumed as valid objective evaluations. Malformed
inputs and nonfinite numerical results throw through the core assertions.

## Existing optimizer and TSPDE fit verification

`manifold::RiemannianTrustRegion` and `SteihaugTruncatedCG` use the existing
geometry, problem and evaluation-workspace contracts. The outer solver keeps
accepted and trial workspaces separate and rejects nonfinite trial costs.
Hessian-vector products are supplied by the problem. See
[the runnable solver checks](../tests/manifold_optimization/trust_region.cpp).

The TSPDE fit driver was adapted in a separate verification probe to use these
kernels and `MatrixBatch`, retaining its whitening of nodal log coordinates and
central finite differences of the analytic gradient for Hessian-vector products.
Its LE data term now uses LE interpolation, matching prediction. Whitening
remains a change of optimizer coordinates and is independent of that correction.
The probe converged for LE and AIRM with observations inside cells and passed a
finite-difference check of the assembled objective gradient.

A unified library fit API, the whitening wrapper, model selection and CV remain
in the application layer for now. This import supplies the reusable core pieces;
it does not introduce a new smoothing model facade or C-LE/rho optimization.
