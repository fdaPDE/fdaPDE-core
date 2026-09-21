# Native sparse block matrices

`SparseBlockMatrix<Scalar, BlockRows, BlockCols>` owns a fixed grid of native CSR blocks. Construct an empty grid from row/column extent containers or uniform block dimensions, or supply dense, sparse and nested block inputs in row-major block order. Scalar zero placeholders inherit their partition extents from neighboring matrix inputs; an entirely unspecified block row or column has extent one. Temporary inputs are materialized immediately.

`block(i,j)` borrows block storage from an lvalue owner. Its dimensions must continue matching the partition; global operations detect an incompatible resize. `coeff` and `contains` address global positions. `value_ref` modifies existing entries; `coeff_ref` inserts a stored zero when necessary. CSR insertion copies O(nnz) storage for atomic replacement, so use `rebuild` or `rebuild_block` for bulk assembly. Structural changes invalidate borrowed coefficient references and row iterators; accessing an already stored coefficient does not.

`to_sparse()` emits sorted owning CSR in O(rows + nnz) traversal for a fixed block grid and preserves stored zeros. `to_dense()` materializes coefficients into an independent dense owner. Integral scalar conversions reject out-of-range values before narrowing; representable floating-to-integral conversions truncate toward zero.

Global triplet rebuilds and row/column unit constraints publish replacement blocks only after success. An empty constraint list is a no-op. Right-hand-side adjustments remain the caller's responsibility.

The API is available through the Eigen-free sparse aggregate. It has no Eigen storage options, alternate index types, compressed-state stubs or column-oriented compatibility iterator. Traverse the native blocks or materialize CSR when a global row traversal is needed.
