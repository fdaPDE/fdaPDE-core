// basis table cache initialization
template <unsigned int M, unsigned int N, unsigned int R, typename E,
	  typename F, typename B, typename I, typename S>
void PDE<M,N,R,E,F,B,I,S>::buildBasis_() {
  // preallocate memory for functional basis
  basis_.resize(domain_.elements());
  for(std::size_t i = 0; i < domain_.elements(); ++i){ // fill basisTable_
    // build base over the element as function of the reference basis and store pointer to it.
    basis_[i].reserve(ct_nnodes(M,R)); // reserve space for basis elements
    for(std::size_t j = 0; j < ct_nnodes(M,R); ++j){
      // there must be a 1-1 correspondance between the j-th physical dof and the j-th dof on the reference element
      basis_[i].emplace_back(domain_.dof_table()(i,j), *domain_.element(i), referenceBasis_[j]);
    }
  }
  return;
}

// constructors
template <unsigned int M, unsigned int N, unsigned int R, typename E,
	  typename F, typename B, typename I, typename S>
PDE<M,N,R,E,F,B,I,S>::PDE(const Mesh<M,N,R>& domain, E bilinearForm, const F& forcingData) :
  domain_(domain), bilinearForm_(bilinearForm), forcingData_(forcingData) {
  // prepare basis cache
  buildBasis_();
}
template <unsigned int M, unsigned int N, unsigned int R, typename E,
	  typename F, typename B, typename I, typename S>
PDE<M,N,R,E,F,B,I,S>::PDE(const Mesh<M,N,R>& domain, E bilinearForm, const F& forcingData,
			  const B& basis, const I& integrator) :
  domain_(domain), bilinearForm_(bilinearForm), forcingData_(forcingData),
  referenceBasis_(basis), integrator_(integrator) {
  // prepare basis cache
  buildBasis_();
}

// store in the format (boundaryID, { ... }) the dirichlet boundary conditions, where { ... } is the time series of the
// data at boundary for boundary node boundaryID
template <unsigned int M, unsigned int N, unsigned int R, typename E,
	  typename F, typename B, typename I, typename S>
void PDE<M,N,R,E,F,B,I,S>::setDirichletBC(const DMatrix<double>& data){
 for(size_t j = 0; j < domain_.dof(); ++j){
    // if j is a node on the domain boundary store the pair (node ID - boundary value)
    if(domain_.isOnBoundary(j)){
      boundaryData_[j] = data.row(j); // O(1) complexity
    }
  }
  return;
}

template <unsigned int M, unsigned int N, unsigned int R, typename E,
	  typename F, typename B, typename I, typename S>
void PDE<M,N,R,E,F,B,I,S>::init() {
  // precomputes some quantites of interest for high level users of FEM.
  // Do not solve the PDE (which means no linear system is solved) for a lower computational cost.
  solver_.init(*this);
  return;
}

template <unsigned int M, unsigned int N, unsigned int R, typename E,
	  typename F, typename B, typename I, typename S>
void PDE<M,N,R,E,F,B,I,S>::solve() {
  // define solver and call solve method on it
  solver_.solve(*this);
  return;
}
