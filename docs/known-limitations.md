# Known limitations

## Vector Lagrange finite element evaluation

The current `FeP<Order, NComponents>` basis and DOF handler contain componentwise
vector support. However, instantiating `FeFunction::operator()` with `P1<2>` fails
to compile: `fe_objects.h` returns a scalar NaN where the output is a vector, and
its reference-coordinate and shape-value path is inconsistent with vector basis
evaluation. Relevant sources are
[`fe_objects.h`](../fdaPDE/src/finite_elements/fe_objects.h) and
[`fe_space.h`](../fdaPDE/src/finite_elements/fe_space.h).

Before extending `GeometricSpace` to Euclidean vector targets, repair this path
and verify component ordering, vector basis counts, inside/outside evaluation,
and equivalence to scalar P1 interpolation with grouped vector coefficients.
This is deferred work; the current geometric facade supports scalar P1 and
B-spline discretizations with SPD or rotation-valued coefficients.

## Legacy scalar spline function evaluation

`BsSpace` and its physical B-spline basis are used directly by the new geometric
facade. The separate scalar `BsFunction` evaluation path still references a
removed interval `range()` accessor, loses constness in its space pointer, and
uses a constructor initializer order that fails strict warning builds. Repair
that path before using `BsFunction` as an independent scalar evaluation oracle;
the geometric spline tests instead sum the complete physical basis explicitly.
These issues are in [`sp_objects.h`](../fdaPDE/src/splines/sp_objects.h).
