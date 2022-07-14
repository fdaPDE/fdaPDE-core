// approximation of first derivative of field along direction i
template <int N>
double ScalarField<N>::approxFirstDerivative(const SVector<N>& x, std::size_t i, double step) const {
  // variation around point x along direction i
  SVector<N> h = SVector<N>::Zero();
  h[i] = step;
  
  // approximation of i-th partial derivative
  return (f_(x + h) - f_(x - h))/(2*step);
}

// approximation of second derivative of field along direction i, j
template <int N>
double ScalarField<N>::approxSecondDerivative(const SVector<N>& x, std::size_t i, std::size_t j, double step) const {
  SVector<N> h_i = SVector<N>::Zero();
  h_i[i] = step; // variation along i-th direction
  if(i == j)
    return (-f_(x + 2*h_i) + 16*f_(x + h_i) - 30*f_(x) + 16*f_(x - h_i) - f_(x - 2*h_i))/(12*pow(step,2));
  else{ 
    SVector<N> h_j = SVector<N>::Zero();
    h_j[j] = step; // variation along j-th direction
    
    // approximation of (i,j).th second derivative
    return (f_(x + h_i + h_j) - f_(x + h_i - h_j) - f_(x - h_i + h_j) + f_(x - h_i - h_j))/(4*pow(step,2));	
  }
}

// approximate computation of gradient via central finite differences
template <int N>
SVector<N> ScalarField<N>::approxGradient(const SVector<N>& x, double step) const {
  SVector<N> gradient;
  for(size_t i = 0; i < N; ++i){
    // approximation of i-th partial derivative at point x
    gradient[i] = approxFirstDerivative(x, i, step);
  }
  return gradient;
}

// approximate computation of hessian matrix via central finite differences
template <int N>
SMatrix<N> ScalarField<N>::approxHessian(const SVector<N>& x, double step) const {
  SMatrix<N> hessian = SMatrix<N>::Zero();

  // hessian matrix is symmetric, compute just the lower triangular part
  for(size_t i = 0; i<N; ++i){
    for(size_t j = 0; j<=i; ++j){
      // approximation of (i,j)-th partial derivative at point x
      hessian(i,j) = approxSecondDerivative(x, i, j, step);
      hessian(j,i) = hessian(i,j); // exploit symmetry of hessian matrix
    }
  }
  return hessian;
}

template <int N>
VectorField<N> ScalarField<N>::derive(double step) const{
  std::array<std::function<double(SVector<N>)>, N> components;

  for(std::size_t i = 0; i < N; ++i){
    // functor wrapping the gradient approximation along direction i
    std::function<double(SVector<N>)> gradientApprox = [=](SVector<N> x) -> double {
      return approxFirstDerivative(x, i, step);
    };
    components[i] = gradientApprox;
  }
  return VectorField<N>(components);
}

// use a standard step value for the approximation
template <int N>
VectorField<N>  ScalarField<N>::derive() const{
  std::array<std::function<double(SVector<N>)>, N> components;

  for(std::size_t i = 0; i < N; ++i){
    // functor wrapping the gradient approximation along direction i using fixed step
    std::function<double(SVector<N>)> gradientApprox = [=](SVector<N> x) -> double {
      return approxFirstDerivative(x, i, step_);
    };
    components[i] = gradientApprox;
  }
  return VectorField<N>(components);

}

template <int N>
std::function<SMatrix<N>(SVector<N>)> ScalarField<N>::deriveTwice(double step) const{
  // lambda wrapping the hessian approximation method
  std::function<SMatrix<N>(SVector<N>)> hessianApprox = [=](SVector<N> x) -> SMatrix<N> {
    return approxHessian(x, step);
  };
  return hessianApprox;
}
    
// use a standard step value for the approximation
template <int N>
std::function<SMatrix<N>(SVector<N>)> ScalarField<N>::deriveTwice() const{
  // lambda wrapping the hessian approximation method using fixed step
  std::function<SMatrix<N>(SVector<N>)> hessianApprox = [=](SVector<N> x) -> SMatrix<N> {
    return approxHessian(x, step_);
  };
  return hessianApprox;
}
