// basis table cache initialization
template <unsigned int M, unsigned int N, unsigned int R, typename E, typename B, typename Solver>
void PDE<M, N, R, E, B, Solver>::buildBasis_() {
  // preallocate memory for functional basis
  basis_.resize(domain_.elements());
  for(const auto& e : domain_){ // fill basisTable_
    // build base over the element and store pointer to it. Observe that the basis is built as function of the reference basis
    // (i.e. no explicit construction on element e is performed)
    basis_[e->ID()].reserve(ct_binomial_coefficient(M+R,R)); // reserve space for basis elements
    for(std::size_t i = 0; i < ct_binomial_coefficient(M+R,R); ++i){
      basis_[e->ID()].emplace_back(
	 [this, e, i](SVector<N> x) -> double {
	   // map x into reference element
	   SVector<N> p = e->invBarycentricMatrix()*(x - e->coords()[0]);
	   return referenceBasis_[i](p); // evaluate reference basis at p
	 });
    }
  }
  return;
}

// constructors
template <unsigned int M, unsigned int N, unsigned int R, typename E, typename B, typename Solver>
PDE<M, N, R, E, B, Solver>::PDE(const Mesh<M,N,R>& domain, E bilinearForm,  const DMatrix<double>& forcingData) :
  domain_(domain), bilinearForm_(bilinearForm), forcingData_(forcingData) {
  // prepare basis cache
  buildBasis_();
}
template <unsigned int M, unsigned int N, unsigned int R, typename E, typename B, typename Solver>
PDE<M, N, R, E, B, Solver>::PDE(const Mesh<M,N,R>& domain, E bilinearForm, const B& basis, const DMatrix<double>& forcingData) :
  domain_(domain), referenceBasis_(basis), bilinearForm_(bilinearForm), forcingData_(forcingData) {
  // prepare basis cache
  buildBasis_();
}

// store in the format (boundaryID, { ... }) the dirichlet boundary conditions, where { ... } is the time series of the
// data at boundary for boundary node boundaryID
template <unsigned int M, unsigned int N, unsigned int R, typename E, typename B, typename Solver>
void PDE<M, N, R, E, B, Solver>::setDirichletBC(const DMatrix<double>& data){
 for(size_t j = 0; j < domain_.nodes(); ++j){
    // if j is a node on the domain boundary store the pair (node ID - boundary value)
    if(domain_.isOnBoundary(j)){
      boundaryData_[j] = data.row(j); // O(1) complexity
    }
  }
  return;
}

template <unsigned int M, unsigned int N, unsigned int R, typename E, typename B, typename Solver>
template <typename I, typename... Args>
void PDE<M, N, R, E, B, Solver>::init(const I &integrator, Args... args) {
  // precomputes some quantites of interest for high level users of FEM. Do not solve the PDE (which means no linear system is solved)
  solver_.init(*this, integrator, args...);
  return;
}

template <unsigned int M, unsigned int N, unsigned int R, typename E, typename B, typename Solver>
template <typename I, typename... Args>
void PDE<M, N, R, E, B, Solver>::solve(const I &integrator, Args... args) {
  // define solver and call solve method on it
  solver_.solve(*this, integrator, args...);
  return;
}
