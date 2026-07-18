#!/usr/bin/env python3
"""FEniCSx reference for the aligned discontinuous-force SIPG benchmark."""

import math

try:
    from mpi4py import MPI
    import numpy as np
    import ufl
    from dolfinx import fem, mesh
    from dolfinx.fem.petsc import LinearProblem
except ModuleNotFoundError as error:
    raise SystemExit(
        f"Missing FEniCSx dependency {error.name!r}. Run this from a DOLFINx environment, or use:\n"
        '  docker run --rm -v "$PWD":/work -w /work dolfinx/dolfinx:stable '
        "python3 test/benchmarks/fenicsx_dg_poisson.py"
    ) from None


ALPHA = 1.0
PENALTY = 10.0
NODE_COUNTS = (5, 9, 17)


def function_space(domain):
    """Cover the public constructor rename in older DOLFINx releases."""
    factory = getattr(fem, "functionspace", None)
    return factory(domain, ("DG", 1)) if factory else fem.FunctionSpace(domain, ("DG", 1))


def linear_problem(a, linear_form, prefix):
    options = {"ksp_type": "preonly", "pc_type": "lu", "ksp_error_if_not_converged": True}
    try:
        return LinearProblem(
            a, linear_form, petsc_options_prefix=prefix, petsc_options=options
        )
    except TypeError as error:
        if "petsc_options_prefix" not in str(error):
            raise
        return LinearProblem(a, linear_form, petsc_options=options)


def solve(node_count):
    subdivisions = node_count - 1
    domain = mesh.create_unit_square(
        MPI.COMM_WORLD,
        subdivisions,
        subdivisions,
        cell_type=mesh.CellType.triangle,
    )
    space = function_space(domain)
    trial = ufl.TrialFunction(space)
    test = ufl.TestFunction(space)
    x = ufl.SpatialCoordinate(domain)
    normal = ufl.FacetNormal(domain)
    facet_size = ufl.FacetArea(domain)

    shifted = x[0] - 0.5
    on_right = ufl.gt(x[0], 0.5)
    correction = ufl.conditional(on_right, shifted**2 * (1.0 - x[0]), 0.0)
    g = x[0] * (1.0 - x[0]) + ALPHA * correction
    exact = g * ufl.sin(np.pi * x[1])
    g_second_correction = ufl.conditional(on_right, ALPHA * (4.0 - 6.0 * x[0]), 0.0)
    force = (np.pi**2 * g + 2.0 - g_second_correction) * ufl.sin(np.pi * x[1])

    quadrature = {"quadrature_degree": 8}
    dx = ufl.Measure("dx", domain=domain, metadata=quadrature)
    dS = ufl.Measure("dS", domain=domain, metadata=quadrature)
    ds = ufl.Measure("ds", domain=domain, metadata=quadrature)

    a = ufl.inner(ufl.grad(trial), ufl.grad(test)) * dx
    a += -ufl.inner(ufl.avg(ufl.grad(trial)), ufl.jump(test, normal)) * dS
    a += -ufl.inner(ufl.jump(trial, normal), ufl.avg(ufl.grad(test))) * dS
    a += PENALTY / ufl.avg(facet_size) * ufl.inner(
        ufl.jump(trial, normal), ufl.jump(test, normal)
    ) * dS

    # Symmetric Nitsche terms impose the homogeneous boundary condition weakly.
    a += -ufl.inner(ufl.dot(ufl.grad(trial), normal), test) * ds
    a += -ufl.inner(trial, ufl.dot(ufl.grad(test), normal)) * ds
    a += PENALTY / facet_size * ufl.inner(trial, test) * ds
    linear_form = ufl.inner(force, test) * dx

    problem = linear_problem(a, linear_form, f"fenicsx_dg_{node_count}_")
    approximate = problem.solve()
    error = approximate - exact
    local_l2 = fem.assemble_scalar(fem.form(ufl.inner(error, error) * dx))
    local_h1 = fem.assemble_scalar(fem.form(ufl.inner(ufl.grad(error), ufl.grad(error)) * dx))
    l2 = math.sqrt(float(np.real(domain.comm.allreduce(local_l2, op=MPI.SUM))))
    broken_h1 = math.sqrt(float(np.real(domain.comm.allreduce(local_h1, op=MPI.SUM))))
    return l2, broken_h1


def rate(previous, current):
    return math.log(previous / current, 2.0)


def main():
    results = []
    for node_count in NODE_COUNTS:
        l2, broken_h1 = solve(node_count)
        results.append((node_count, l2, broken_h1))

    if MPI.COMM_WORLD.rank == 0:
        print("nodes       L2 error   rate    broken H1   rate")
        for index, (nodes, l2, broken_h1) in enumerate(results):
            l2_rate = "-" if index == 0 else f"{rate(results[index - 1][1], l2):.3f}"
            h1_rate = "-" if index == 0 else f"{rate(results[index - 1][2], broken_h1):.3f}"
            print(f"{nodes:5d}  {l2:13.6e}  {l2_rate:>5}  {broken_h1:13.6e}  {h1_rate:>5}")


if __name__ == "__main__":
    main()
