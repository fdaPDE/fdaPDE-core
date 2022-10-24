// direct construction from array of std::functions
template <int N, int M, int K>
MatrixField<N,M,K>::MatrixField(std::array<std::array<std::function<double(SVector<N>)>, K>, M> field){
  for(std::size_t i = 0; i < M; ++i){
    for(std::size_t j = 0; j < K; ++j){
      field_[i][j] = ScalarField<N>(field[i][j]);
    }
  }
}
  
// braced-list constructor
template <int N, int M, int K>
MatrixField<N,M,K>::MatrixField
(std::initializer_list<std::initializer_list<std::function<double(SVector<N>)>>> field) {
  std::size_t i,j;
  // cycle over each row list
  for(auto it_row = field.begin(); it_row != field.end(); ++it_row){
    j = 0;
    // for each row, scan the elements column-wise
    for(auto it_col = it_row->begin(); it_col != it_row->end(); ++it_col){
      field_[i][j] = ScalarField<M>(*it_col);
      j++;
    }
    i++;
  }
}

// call operator
template <int N, int M, int K>
SMatrix<M, K> MatrixField<N,M,K>::operator()(const SVector<N>& point) const {
  SMatrix<M, K> result;
  for(std::size_t i = 0; i < M; ++i){
    for(std::size_t j = 0; j < K; ++j){
      result(i,j) = field_[i][j](point);
    }
  }
  return result;
}

// constant access operator
template <int N, int M, int K>
const ScalarField<N>& MatrixField<N,M,K>::operator()(std::size_t i, std::size_t j) const{
  return field_[i][j];
}
// non constant access operator
template <int N, int M, int K>
ScalarField<N>& MatrixField<N,M,K>::operator()(std::size_t i, std::size_t j) {
  return field_[i][j];
}
// required by FieldExpr to work correctly
template <int N, int M, int K>
const ScalarField<N>& MatrixField<N,M,K>::coeff(std::size_t i, std::size_t j) const{ return field_[i][j]; }

// return VectorField built from i-th row of the MatrixField
template <int N, int M, int K>
VectorField<N, K> MatrixField<N,M,K>::row(std::size_t i) const {
  VectorField<N, K> result;
  for(std::size_t j = 0; j < K; ++j){
    result[j] = field_[i][j];
  }
  return result;
}
// return VectorField built from i-th column of the MatrixField
template <int N, int M, int K>
VectorField<N, M> MatrixField<N,M,K>::col(std::size_t i) const {
  VectorField<N, M> result;
  for(std::size_t j = 0; j < M; ++j){
    result[j] = field_[j][i];
  }
  return result;
}

// out of class definitions of MatrixField arithmetic

// multiplication by scalar. Multiply each element of the field by scalar
template <int N, int M, int K>
MatrixField<N,M,K> operator*(double scalar, const MatrixField<N,M,K>& op){
  MatrixField<N,M,K> result;
  for(std::size_t i = 0; i < M; i++){
    for(std::size_t j = 0; j < K; ++j){
      std::function<double(SVector<N>)> f;

      // callable representing the multiplication of the (i,j)-th element by scalar
      f = [scalar, op, i, j](const SVector<N>& p) -> double {
	return scalar*op(i,j)(p);
      };
      result(i,j) = f; // converting constructor called
    }
  }
  return result;
}
template <int N, int M, int K>
MatrixField<N,M,K> operator*(const MatrixField<N,M,K>& op, double scalar){
  return scalar*op;
}

// rhs multiplication by SVector
template <int N, int M, int K>
VectorField<N,M> operator*(const MatrixField<N,M,K>& op1, const SVector<K>& op2){
  VectorField<N,M> result;
  for(std::size_t i = 0; i < M; ++i){
    // store the dot product of i-th row of op1 with the constant vector given in input
    result[i] = op1.row(i).dot(op2);
  }
  return result;
}
// rhs multiplication by VectorField
template <int N, int M, int K>
VectorField<N,M> operator*(const MatrixField<N,M,K>& op1, const VectorField<N,K>& op2){
  VectorField<N,M> result;
  for(std::size_t i = 0; i < M; ++i){
    // store the dot product of i-th row of op1 with the vector field given in input
    result[i] = op1.row(i).dot(op2);
  }
  return result;
}
// MatrixField - MatrixField product
template <int N, int M, int K, int H>
MatrixField<N,M,H> operator*(const MatrixField<N,M,K>& op1, const MatrixField<N,K,H>& op2){
  MatrixField<N,M,H> result;
  // implementation of simple schoolbook O(n^3) matrix multiplication
  for(std::size_t i = 0; i < M; ++i){
    for(std::size_t j = 0; j < H; ++j){
      // store the dot product of i-th row of op1 with j-th column of op2
      result(i,j) = op1.row(i).dot(op2.col(j)); 
    }
  }
  return result;
}
