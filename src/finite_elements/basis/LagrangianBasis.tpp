// compute coefficients via Vandermonde matrix
template <unsigned int M, unsigned int N, unsigned int R>
void LagrangianBasis<M,N,R>::computeCoefficients(const std::array<std::array<double, M>, ct_binomial_coefficient(M+R,R)>& nodes){
  // build vandermonde matrix
  constexpr unsigned int n_basis = ct_binomial_coefficient(M+R,R);
  constexpr std::array<std::array<unsigned, M>, n_basis> expTable_ = MultivariatePolynomial<M,R>::expTable_;

  // Vandermonde matrix construction
  SMatrix<n_basis> V = Eigen::Matrix<double, n_basis, n_basis>::Ones();
  for(size_t i = 0; i < n_basis; ++i){
    for(size_t j = 1; j < n_basis; ++j){
      V(i,j) = MonomialProduct<M-1, std::array<double, M>, std::array<unsigned, M>>::unfold(nodes_[i], expTable_[j]);
    }
  }
 
  // solve system V*a = b with b vector having 1 at position i and 0 everywhere else.
  // Its solution gives the vector of coefficients of the i-th Lagrange polynomial
  Eigen::PartialPivLU<SMatrix<n_basis>> LUdecomposition(V);
  for(size_t i = 0; i < n_basis; ++i){
    // build rhs vector
    SVector<n_basis> b = Eigen::Matrix<double, n_basis, 1>::Zero();
    b[i] = 1;
    // solve linear system V*a = b
    SVector<n_basis> coeff = LUdecomposition.solve(b);
    // cast to array
    std::array<double, n_basis> coeff_array;
    for(size_t j = 0; j < n_basis; ++j) coeff_array[j] = coeff[j];
    // store basis
    basis_[i] = MultivariatePolynomial<M, R>(coeff_array);
  }
  return;
}

// expose iterators
template <unsigned int M, unsigned int N, unsigned int R>
typename std::array<MultivariatePolynomial<M,R>, ct_binomial_coefficient(M+R,R)>::const_iterator
LagrangianBasis<M,N,R>::begin() const {
  return basis_.cbegin();
}
template <unsigned int M, unsigned int N, unsigned int R>
typename std::array<MultivariatePolynomial<M,R>, ct_binomial_coefficient(M+R,R)>::const_iterator
LagrangianBasis<M,N,R>::end() const {
  return basis_.cend();
}
