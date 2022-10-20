// basis table cache initialization
template <unsigned int M, unsigned int N, unsigned int R, typename E, typename F, typename B, typename I, typename S>
void PDE<M,N,R,E,F,B,I,S>::buildBasis_() {
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
template <unsigned int M, unsigned int N, unsigned int R, typename E, typename F, typename B, typename I, typename S>
PDE<M,N,R,E,F,B,I,S>::PDE(const Mesh<M,N,R>& domain, E bilinearForm, const F& forcingData) :
  domain_(domain), bilinearForm_(bilinearForm), forcingData_(forcingData) {
  // prepare basis cache
  buildBasis_();
}
template <unsigned int M, unsigned int N, unsigned int R, typename E, typename F, typename B, typename I, typename S>
PDE<M,N,R,E,F,B,I,S>::PDE(const Mesh<M,N,R>& domain, E bilinearForm, const F& forcingData, const B& basis, const I& integrator) :
  domain_(domain), bilinearForm_(bilinearForm), forcingData_(forcingData), referenceBasis_(basis), integrator_(integrator) {
  // prepare basis cache
  buildBasis_();
}

// store in the format (boundaryID, { ... }) the dirichlet boundary conditions, where { ... } is the time series of the
// data at boundary for boundary node boundaryID
template <unsigned int M, unsigned int N, unsigned int R, typename E, typename F, typename B, typename I, typename S>
void PDE<M,N,R,E,F,B,I,S>::setDirichletBC(const DMatrix<double>& data){
 for(size_t j = 0; j < domain_.nodes(); ++j){
    // if j is a node on the domain boundary store the pair (node ID - boundary value)
    if(domain_.isOnBoundary(j)){
      boundaryData_[j] = data.row(j); // O(1) complexity
    }
  }
  return;
}

template <unsigned int M, unsigned int N, unsigned int R, typename E, typename F, typename B, typename I, typename S>
void PDE<M,N,R,E,F,B,I,S>::init() {
  // precomputes some quantites of interest for high level users of FEM.
  // Do not solve the PDE (which means no linear system is solved) for a lower computational cost.
  solver_.init(*this);
  return;
}

template <unsigned int M, unsigned int N, unsigned int R, typename E, typename F, typename B, typename I, typename S>
void PDE<M,N,R,E,F,B,I,S>::solve() {
  // define solver and call solve method on it
  solver_.solve(*this);
  return;
}
