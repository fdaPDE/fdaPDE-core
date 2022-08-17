// constructors
template <int M, int N>
VectorField<M,N>::VectorField(std::array<std::function<double(SVector<M>)>, N> field) {
  for(std::size_t i = 0; i < N; ++i){
    field_[i] = ScalarField<M>(field[i]);
  }
}
// allow braced-list initialization
template <int M, int N>
VectorField<M,N>::VectorField(std::initializer_list<std::function<double(SVector<M>)>> field) {
  std::size_t i = 0;
  for(auto it = field.begin(); it != field.end(); ++it){
    field_[i] = ScalarField<M>(*it);
    i++;
  }
}    
// allow for the construction of a VectorField from a single vectorial lambda. Observe that by doing so evaluating the
// field along direction i only still requires the evaluation of all other dimesions. Use with care.
template <int M, int N>
VectorField<M,N>::VectorField(const std::function<SVector<N>(SVector<M>)>& field) {
  for(std::size_t i = 0; i < N; ++i){
    auto fieldExpr = [=](SVector<M> x) -> double { return field(x)[i]; };
    field_[i] = ScalarField<M>(fieldExpr);
  }
}

// wrap a VectExpr into a valid VectorField
template <int M, int N>
template <typename E>
VectorField<M,N>::VectorField(const VectExpr<M, N, E>& expr) {
  for(std::size_t i = 0; i < N; ++i){
    field_[i] = expr[i];
  }
};

// call operator
template <int M, int N>
SVector<N> VectorField<M,N>::operator()(const SVector<M> &point) const {
  SVector<N> result;
  for(size_t i = 0; i < N; ++i){
    // call lambda for each dimension of the vector field
    result[i] = field_[i](point);
  }
  return result;
}

// subscript operator, constant and not constant version
template <int M, int N>
const ScalarField<M>& VectorField<M,N>::operator[](size_t i) const { return field_[i]; }
template <int M, int N>
ScalarField<M>& VectorField<M,N>::operator[](size_t i) { return field_[i]; }

// out of class definition for VectorField

// matrix-VectorField product operator. Returns a new vector field where each component of the field represent
// the action of the multiplication between a given matrix and a vector field
template <int M, int N, int K>
VectorField<M,K> operator*(const Eigen::Matrix<double,K,N>& op1, const VectorField<M,N>& op2) {
  VectorField<M,K> result;
  for(size_t i = 0; i < K; ++i){
    std::function<double(SVector<M>)> f;

    // create lambda expression to represent the multiplication between a matrix and the field
    f = [i, op1, op2](const SVector<M>& p) -> double {
      double product = 0;
      // j-th matrix row - vector field product
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
template <int M, int N>
VectorField<M,N> operator*(double scalar, const VectorField<M,N>& op){
  VectorField<M,N> result;
  for(size_t i = 0; i < N; ++i){
    std::function<double(SVector<M>)> f;

    // create lambda expression to represent the multiplication between a matrix and the field
    f = [scalar, op, i](const SVector<M>& p) -> double {
      return scalar*op[i](p);
    };
    result[i] = f;
  }
  return result;  
}

template <int M, int N>
VectorField<M,N> operator*(const VectorField<M,N>& op, double scalar){
  return scalar*op;
}

// return a functor representing an inner product.
template <int M, int N>
DotProduct<VectorField<M,N>, VectorField<M,N>> VectorField<M,N>::dot(const VectorField<M,N> &rhs) const {  
  return DotProduct<VectorField<M,N>, VectorField<M,N>>(*this, rhs);
}

template <int M, int N>
ScalarField<M> VectorField<M, N>::dot(const SVector<N>& op) const {
  std::function<double(SVector<M>)> result;
  // build lambda expressing inner product
  result = [this, op](const SVector<M>& x) -> double{
    double y = 0;
    for(std::size_t i = 0; i < N; ++i)
      y += op[i]*operator[](i)(x);
      
    return y;
  };
  return ScalarField<M>(result);
}
