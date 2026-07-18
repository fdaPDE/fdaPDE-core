# FEniCSx DG reference

`fenicsx_dg_poisson.py` solves the same aligned discontinuous-force manufactured problem as the C++ DG benchmark on `UnitSquare(5)`, `UnitSquare(9)`, and `UnitSquare(17)`. It uses scalar discontinuous P1 elements, SIPG penalty `10`, and prints the L2 and broken-H1 errors and refinement rates.

Run it from the `core` directory in an existing DOLFINx environment:

```sh
python3 test/benchmarks/fenicsx_dg_poisson.py
```

Or use the official stable container:

```sh
docker run --rm -v "$PWD":/work -w /work dolfinx/dolfinx:stable \
  python3 test/benchmarks/fenicsx_dg_poisson.py
```

The reference uses the standard weak SIPG/Nitsche boundary terms; the C++ benchmark currently constrains homogeneous boundary degrees of freedom strongly. Compare convergence orders first, and expect small differences in error constants.
