// evaluate the form on two different points (this is like evaluating one scalar field at one point,
// the second one at another point and then perform the inner product between the two numerical results)
template <typename T1, typename T2>
template <int N>
double DotProduct<T1,T2>::operator()(const SVector<N>& x, const SVector<N>& y) const{
  static_assert(T1::size == T2::size);
  // implementation of the scalar product operation
  double result = 0;
  for(std::size_t i = 0; i < T1::size; ++i){
    result += lhs_[i](x)*rhs_[i](y);
  }
  return result;
}

// consider the analytical expression of the inner product as a scalar field, evaluate it at point x
template <typename T1, typename T2>
template <int N>
double DotProduct<T1,T2>::operator()(const SVector<N>& x) const{
  static_assert(T1::size == T2::size);
  // implementation of the scalar product operation
  double result = 0;
  for(size_t i = 0; i < T1::size; ++i){
    result += lhs_[i](x)*rhs_[i](x);
  }
  return result;
}
