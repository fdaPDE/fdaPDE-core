// integrate a BilinearFormExpr over mesh element e using basis B. The pair (i,j) indicates the element position of the produced
// value in the matrix discretizing the form. This method is used as part of the assembly loop in the computation of the
// discretization matrix of the differential operator L
template <unsigned int M, unsigned int K>
template <unsigned int N, unsigned int R, typename B, typename F>
double Integrator<M, K>::integrate(const B& basis, const Element<M, N, R>& e, int i , int j, const F& bilinearForm) const{
  // apply quadrature rule
  double value = 0;
  // builds the callable to integrate here once from the bilinear form passed as argument
  ScalarField<M> f = bilinearForm.integrate(basis, e, i, j);
  for(size_t iq = 0; iq < integrationTable_.num_nodes; ++iq){
    SVector<M> p = SVector<M>(integrationTable_.nodes[iq].data());
    value += f(p) * integrationTable_.weights[iq];
  }
  // correct for measure of domain (element e)
  return value * e.measure();
}

// integrate a callable F over a mesh element e.
template <unsigned int M, unsigned int K>
template <unsigned int N, unsigned int R, typename F>
double Integrator<M, K>::integrate(const Element<M, N, R>& e, F &f) const {
  double value = 0;
  // execute quadrature rule
  for(size_t iq = 0; iq < integrationTable_.num_nodes; ++iq){
    // map quadrature point to current element e
    SVector<N> p = e.barycentricMatrix()*SVector<M>(integrationTable_.nodes[iq].data()) + e.coords()[0];
    value += f(p)*integrationTable_.weights[iq];
  }
  // correct for measure of domain (element e)
  return value * e.measure();
}

// integrate a callable F over the entire mesh m.
// Just exploit linearity of the integral operation to sum the result of the integral of F over each mesh element
template <unsigned int M, unsigned int K>
template <unsigned int N, unsigned int R, typename F>
double Integrator<M, K>::integrate(const Mesh<M, N, R>& m, const F &f) const {
  double value = 0;
  // cycle over all mesh elements
  for(const auto& e : m)
    value += integrate(*e, f);
  return value;
}
