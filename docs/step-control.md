# Objective increments and lumped stationarity

`<fdaPDE/optimization.h>` exposes `armijo_step(oracle, slope, initial_step,
contraction, sufficient_decrease, max_trials)`. The oracle returns
`f(x + h*d) - f(x)` directly. A compensated reduction of two already rounded
full costs cannot recover a sub-ulp change; callers need a model-specific
increment formula. Nonfinite increments are rejected. The direction must have
finite negative slope; search parameters are checked at the public boundary.
The result reports acceptance, evaluated step, increment, Armijo bound and
oracle evaluation count. A failed search does not update the caller's point.

`lumped_gradient_summary(g, masses, components)` accepts flat interleaved nodal
covectors and positive finite lumped masses. It returns both infinity norms,
their controlling coefficient indices and total volume. Density means
`max(abs(g_ia)/m_i)` in the caller's spatial volume units; it is not an intrinsic
matrix norm. Coordinate changes require consistent units for tolerances.

`<fdaPDE/dense_linear_algebra.h>` exposes
`matrix_exp_increment(A,H)` for symmetric native expressions. It computes the
owned symmetric exponential difference using the cross spectral divided
identity and the supplied increment. It avoids cancellation when `H` is below
the ulp of `A`. Shapes and finite inputs/results are checked; SPD-only overflow
or conditioning guarantees of `matrix_exp` are not claimed for a difference.

`<fdaPDE/geometric_discretization.h>` and the FE aggregate expose
`gfe::p1_lumped_stiffness_at(prepared, field, node, component, components)`.
It applies a row of the signed P1 stiffness operator to flat coordinates using
compensated edge differences. It does not apply inverse mass or change boundary
conditions. The prepared stencil owns validated masses, signed edges and nodal
incidence; row/index preconditions are internal debug assertions.

`<fdaPDE/utility.h>` exposes double-precision `CompensatedSum`, with `add` and
`value`. Callers provide finite summands and check their own finite result.

## Validation

The new target `fdapde_step_control_test` runs with assertions enabled and
covers hidden increases/descent under a large constant offset, finite search
budgets, invalid domains/directions, density units/indexing, compensated
cancellation, tiny and noncommuting exponential increments, dynamic shape,
and constant-annihilating signed stiffness rows. Assertions include their
numeric or structural oracle in comments. Historical tests were not removed.

```bash
cmake -S tests -B /private/tmp/gfa-core-fe-clang
cmake --build /private/tmp/gfa-core-fe-clang --target \
  fdapde_step_control_test fdapde_spd_test fdapde_numeric_test \
  fdapde_geometric_discretization_test fdapde_module_header_checks -j 4
ctest --test-dir /private/tmp/gfa-core-fe-clang -R '^StepControl\.' --output-on-failure
```

Validated locally with Apple Clang: 7 targeted tests, 39 SPD regressions,
6 numeric regressions, 17 geometric-discretization regressions and module
public-header compilation. No blanket NoDebug duplicates were introduced.
Application validation is documented in `GFA-application/docs/mgh-le-dwi-step-control.md`.
The generic Armijo increment interface is used by the LE-DWI application solver;
the legacy `BacktrackingLineSearch` callback is not migrated by this change.

Proposed research-hub update: record the new increment-oracle API under the
optimization module and the cross-spectral exponential increment under native
SPD algebra, citing the public aggregates, implementation and
`tests/optimization/step_control.cpp`. Existing hub architecture entries are
not modified by this local implementation task.

## Roundoff-aware Wolfe search

`approximate_wolfe_step` adds bounded derivative-aware search when a direct
objective-increment oracle is unavailable. The oracle returns full cost and
physical directional derivative. Outside the roundoff window it requires
ordinary Armijo decrease and weak Wolfe curvature. Inside
`64*epsilon*max(1,abs(f0))`, it instead requires
`curvature*slope0 <= slope1 <= (2*sufficient_decrease-1)*slope0`.
The result identifies approximate acceptance and labels the quadratic energy
change as an estimate. This is not an exact increment or a stationarity test.

The implementation follows the acceptance conditions in Hager and Zhang,
ACM TOMS 32(1), 2006, equations 22, 23 and 27:
https://people.clas.ufl.edu/hager/files/cg_code.pdf
It uses safeguarded secant/bisection and capped expansion, not the full
CG_DESCENT search. Non-descent inputs, invalid domains, visible increases,
flat unchanged slopes, step caps and budgets have explicit tests. Approximate
acceptance permits only the stated roundoff-sized cost increase; convergence
must be assessed separately in the caller's original gradient coordinates.

The same compensated reduction is now used for serial and parallel Cheeger
energy values, preserving their exact parity. No Cheeger gradient, rho
pullback, deterministic alignment start or inner tolerance was changed.

Verified scope for this extension: 12 step-control tests, 43 geometric
smoothing regressions, aggregate public-header checks and application native /
Python checks. Core comments and every new assertion were reviewed separately
from clang-format. No historical test implementation or pointer was removed.

Hub proposal: extend `fdapde/modules/optimization.md` with the public
`approximate_wolfe_step` acceptance contract and its separate stationarity
requirement; source `fdaPDE/src/optimization/step_control.h` and
`tests/optimization/step_control.cpp`, confidence high. Add exact compensated
serial/parallel energy parity as a review heuristic under `fdapde/patterns/`,
source `cheeger_smoothing.h` and `tension_execution.cpp`, confidence high.
