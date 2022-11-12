// call operator
template <int N, int M, int K, typename F>
SMatrix<M,K> MatrixField<N,M,K,F>::operator()(const SVector<N>& point) const {
  SMatrix<M, K> result;
  for(std::size_t i = 0; i < M; ++i){
    for(std::size_t j = 0; j < K; ++j){
      result(i,j) = field_[i][j](point);
    }
  }
  return result;
}
// constant access operator
template <int N, int M, int K, typename F>
const ScalarField<N,F>& MatrixField<N,M,K,F>::operator()(std::size_t i, std::size_t j) const{
  return field_[i][j];
}
// non constant access operator
template <int N, int M, int K, typename F>
ScalarField<N,F>& MatrixField<N,M,K,F>::operator()(std::size_t i, std::size_t j) {
  return field_[i][j];
}
// required by FieldExpr to work correctly
template <int N, int M, int K, typename F>
const ScalarField<N,F>& MatrixField<N,M,K,F>::coeff(std::size_t i, std::size_t j) const { return field_[i][j]; }

// out of class definitions of MatrixField arithmetic
// rhs multiplication by SVector
template <int N, int M, int K, typename F>
MatrixVectorProduct<N,M,K, MatrixField<N,M,K,F>, VectorConst<N,K>>
operator*(const MatrixField<N,M,K,F>& op1, const SVector<K>& op2){
  return MatrixVectorProduct<N,M,K, MatrixField<N,M,K,F>, VectorConst<N,K>>(op1, VectorConst<N,K>(op2));
}
// rhs multiplication by VectorField
template <int N, int M, int K, typename F1, typename F2>
MatrixVectorProduct<N,M,K, MatrixField<N,M,K,F1>, VectorField<N,K,F2>>
operator*(const MatrixField<N,M,K,F1>& op1, const VectorField<N,K,F2>& op2){
  return MatrixVectorProduct<N,M,K, MatrixField<N,M,K,F1>, VectorField<N,K,F2>>(op1, op2);
}
