// call operator
template <int M, int N, typename F>
SVector<N> VectorField<M,N,F>::operator()(const SVector<M>& point) const {
  SVector<N> result;
  for(size_t i = 0; i < N; ++i){
    // call callable for each dimension of the vector field
    result[i] = field_[i](point);
  }
  return result;
}
// subscript operator, constant and not constant version
template <int M, int N, typename F>
const ScalarField<M,F>& VectorField<M,N,F>::operator[](size_t i) const { return field_[i]; }
template <int M, int N, typename F>
ScalarField<M,F>& VectorField<M,N,F>::operator[](size_t i) { return field_[i]; }

// out of class definitions of VectorField arithmetic
// forward declaration
template <unsigned int N, unsigned int M, unsigned int K> class MatrixConst;
// VectorField-VectorField inner product
template <int M, int N, typename F>
DotProduct<VectorField<M,N,F>, VectorConst<M,N>>
VectorField<M,N,F>::dot(const SVector<N>& rhs) const {  
  return DotProduct<VectorField<M,N,F>, VectorConst<M,N>>(*this, VectorConst<M,N>(rhs));
}
// VectorField-VectorExpr inner product
template <int M, int N, typename F>
template <typename E>
DotProduct<VectorField<M,N,F>, E>
VectorField<M,N,F>::dot(const VectorExpr<M,N,E> &rhs) const {  
  return DotProduct<VectorField<M,N,F>, E>(*this, rhs.get());
}    
