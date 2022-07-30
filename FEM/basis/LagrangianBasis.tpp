// constructor from mesh element
template <unsigned int M, unsigned int R>
template <unsigned int N>
LagrangianBasis<M,R>::LagrangianBasis(const Element<M,N,R>& e){
  // collect nodes from mesh element .
  std::array<std::array<double, M>, ct_nnodes(M, R)> elementNodes{};
  for(std::size_t i = 0; i < ct_nnodes(M, R); ++i){
    for(std::size_t j = 0; j < M; ++j){
      elementNodes[i][j] = e.coords()[i][j];
    }
  }
  nodes_ = elementNodes; // store nodes on which the base is defined
  computeCoefficients(elementNodes); // compute coefficients
  return;
}

// compute coefficients via Vandermonde matrix
template <unsigned int M, unsigned int R>
void LagrangianBasis<M, R>::computeCoefficients(const std::array<std::array<double, M>, ct_binomial_coefficient(M+R,R)>& nodes){
    // build vandermonde matrix
    constexpr unsigned int N = ct_binomial_coefficient(M+R,R);
    constexpr std::array<std::array<unsigned, M>, N> expTable_ = MultivariatePolynomial<M,R>::expTable_;

    // Vandermonde matrix construction
    SMatrix<N> V = Eigen::Matrix<double, N, N>::Ones();
    for(size_t i = 0; i < N; ++i){
      for(size_t j = 1; j < N; ++j){
	V(i,j) = MonomialProduct<M-1, std::array<double, M>, std::array<unsigned, M>>::unfold(nodes_[i], expTable_[j]);
      }
    }
 
    // solve system V*a = b with b vector having 1 at position i and 0 everywhere else.
    // Its solution gives the vector of coefficients of the i-th Lagrange polynomial
    Eigen::ColPivHouseholderQR<SMatrix<N>> QRdecomposition(V);
    for(size_t i = 0; i < N; ++i){
      // build rhs vector
      SVector<N> b = Eigen::Matrix<double, N, 1>::Zero();
      b[i] = 1;
      // solve linear system V*a = b
      SVector<N> coeff = QRdecomposition.solve(b);
      // cast to array
      std::array<double, N> coeff_array;
      for(size_t j = 0; j < N; ++j) coeff_array[j] = coeff[j];
      // store basis
      basis_[i] = MultivariatePolynomial<M, R>(coeff_array);
    }
    return;
}

// expose iterators
template <unsigned int M, unsigned int R>
typename std::array<MultivariatePolynomial<M,R>, ct_binomial_coefficient(M+R,R)>::const_iterator
LagrangianBasis<M, R>::begin() const {
  return basis_.cbegin();
}
template <unsigned int M, unsigned int R>
typename std::array<MultivariatePolynomial<M,R>, ct_binomial_coefficient(M+R,R)>::const_iterator
LagrangianBasis<M, R>::end() const {
  return basis_.cend();
}
