# Native sparse Kronecker products

`kron(A, B)` accepts native CSR matrices and returns independent CSR storage with dimensions `(A.rows() * B.rows(), A.cols() * B.cols())`. Coefficients use the common type of the two input scalars. The dense expression overload remains available through the dense aggregate; CSR overloads are exposed by `<fdaPDE/sparse_linear_algebra.h>` and the full algebra aggregate.

The result removes exact zero products and permits temporary sparse owners because evaluation is eager. Inputs are unchanged. Empty axes retain the other dimension products. Dimensions and the possible entry count must fit the supported index range; invalid sizes throw `std::length_error`. Integral coefficient overflow throws `std::overflow_error` before multiplication. Floating-point products follow ordinary scalar arithmetic, including underflow and nonfinite values.

Use `kron` directly. The former Eigen Kronecker implementation and alternate `kronecker` spelling are not retained.
