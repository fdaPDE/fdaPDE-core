// call operator
template <int N>
SVector<N> VectorField<N>::operator()(const SVector<N> &point) const {
  SVector<N> result;
  for(size_t i = 0; i < N; ++i){
    // call lambda for each dimension of the vector field
    result[i] = field_[i](point);
  }
  return result;
}

// subscript operator, constant and not constant version
template <int N>
const ScalarField<N>& VectorField<N>::operator[](size_t i) const { return field_[i]; }
template <int N>
ScalarField<N>& VectorField<N>::operator[](size_t i) { return field_[i]; }

// out of class definition for VectorField

// matrix-VectorField product operator. Returns a new vector field where each component of the field represent
// the action of the multiplication between a given matrix and a vector field
template <int N, int M>
VectorField<M> operator*(const Eigen::Matrix<double,N,M>& op1, const VectorField<N>& op2) {
  VectorField<M> result;
  for(size_t i = 0; i < M; ++i){
    std::function<double(SVector<N>)> f;

    // create lambda expression to represent the multiplication between a matrix and the field
    f = [i, op1, op2](const SVector<N>& p) -> double {
      double product = 0;
      // matrix row times vector field
      for(size_t j = 0; j < N; ++j){
	product += op1(i,j)*op2[j](p);
      }
      return product;
    };
    result[i] = f;
  }
  return result;
}  

// double-VectorField product operator. Multiply the scalar to each dimension of the VectorField
template <int N>
VectorField<N> operator*(double scalar, const VectorField<N>& op){
  VectorField<N> result;
  for(size_t i = 0; i < N; ++i){
    std::function<double(SVector<N>)> f;

    // create lambda expression to represent the multiplication between a matrix and the field
    f = [scalar, op](const SVector<N>& p) -> double {
      double product = 0;
      // matrix row times vector field
      for(size_t j = 0; j < N; ++j){
	product += scalar*op[j](p);
      }
      return product;
    };
    result[i] = f;
  }
  return result;  
}

template <int N>
VectorField<N> operator*(const VectorField<N>& op, double scalar){
  return scalar*op;
}

// return a functor representing an inner product.
template <int N>
DotProduct<VectorField<N>, VectorField<N>> VectorField<N>::dot(const VectorField<N> &rhs) const {  
  return DotProduct<VectorField<N>, VectorField<N>>(*this, rhs);
}

template <int N>
ScalarField<N> VectorField<N>::dot(const SVector<N>& op) const {
  std::function<double(SVector<N>)> result;
  // build lambda expressing inner product
  result = [this, op](const SVector<N>& x) -> double{
    double y = 0;
    for(size_t i = 0; i < N; ++i)
      y += op[i]*operator[](i)(x);
      
    return y;
  };
  return ScalarField<N>(result);
}
