// approximation of first derivative of field along direction i
template <int N, typename F>
double ScalarField<N,F>::approxFirstDerivative(const SVector<N>& x, std::size_t i, double step) {
  // variation around point x along direction i
  SVector<N> h = SVector<N>::Zero();
  h[i] = step;  
  // approximation of i-th partial derivative
  return (f_(x + h) - f_(x - h))/(2*step);
}

// approximation of second derivative of field along direction i, j
template <int N, typename F>
double ScalarField<N,F>::approxSecondDerivative(const SVector<N>& x, std::size_t i, std::size_t j, double step) {
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

// gradient vector computation
// approximate computation of gradient via central finite differences
template <int N, typename F>
SVector<N> ScalarField<N,F>::approxGradient(const SVector<N>& x, double step) {
  SVector<N> gradient;
  for(size_t i = 0; i < N; ++i){
    // approximation of i-th partial derivative at point x
    gradient[i] = approxFirstDerivative(x, i, step);
  }
  return gradient;
}
template <int N, typename F>
VectorField<N,N,std::function<double(SVector<N>)>> ScalarField<N,F>::derive(double step) {
  std::array<std::function<double(SVector<N>)>, N> components;
  for(std::size_t i = 0; i < N; ++i){
    // functor wrapping the gradient approximation along direction i
    std::function<double(SVector<N>)> gradientApprox = [=](SVector<N> x) -> double {
      return approxFirstDerivative(x, i, step);
    };
    components[i] = gradientApprox;
  }
  return VectorField<N,N,std::function<double(SVector<N>)>>(components);
}
// use a standard step value for the approximation
template <int N, typename F>
VectorField<N,N,std::function<double(SVector<N>)>> ScalarField<N,F>::derive() {
  return derive(step_);
}

// hessian matrix computation
// approximate computation of hessian matrix via central finite differences
template <int N, typename F>
SMatrix<N> ScalarField<N,F>::approxHessian(const SVector<N>& x, double step) {
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
template <int N, typename F>
std::function<SMatrix<N>(SVector<N>)> ScalarField<N,F>::deriveTwice(double step) {
  // lambda wrapping the hessian approximation method
  std::function<SMatrix<N>(SVector<N>)> hessianApprox = [=](SVector<N> x) -> SMatrix<N> {
    return approxHessian(x, step);
  };
  return hessianApprox;
}    
// use a standard step value for the approximation
template <int N, typename F>
std::function<SMatrix<N>(SVector<N>)> ScalarField<N,F>::deriveTwice() {
  return deriveTwice(step_);
}
