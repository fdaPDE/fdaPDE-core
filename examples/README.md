# Linear algebra tutorials

## Choose an example

The numbered programs under `linear_algebra/` cover storage/views (01), lazy
expressions/products (02), structured solves (03), symmetric/skew storage (04),
orthogonal/rotation matrices (05), factorizations (06), SPD caches (07), Boolean
and multidimensional storage (08), batch arithmetic (09), batch spectral transforms
(10), and Eigen sparse blocks/products (11).

Compile a native tutorial from the repository root, for example:

```sh
c++ -std=c++20 -O2 -pthread -I. examples/linear_algebra/09_matrix_batch_arithmetic.cpp -o /tmp/fdapde-batch
/tmp/fdapde-batch
```

Tutorial 11 additionally needs Eigen 3.4 on the include path. SIMD remains opt-in
through `-DFDAPDE_ENABLE_SIMD=1`; parallel batch calls explicitly use `execution_par`.
