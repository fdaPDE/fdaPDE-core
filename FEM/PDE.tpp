// constructor
template <unsigned int M, unsigned int N, unsigned int R, typename E, typename B, typename Solver>
PDE<M, N, R, E, B, Solver>::PDE(const Mesh<M,N,R>& domain, E bilinearForm, const B& basis, const DMatrix<double>& forcingData) :
  domain_(domain), bilinearForm_(bilinearForm), forcingData_(forcingData) {
  // preallocate memory for functional basis
  basis_.resize(domain_.elements());
  for(const auto& e : domain_){ // fill basisTable_
    // build base over the element and store pointer to it
    basis_[e->ID()] = std::make_shared<B>(*e);
  }
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
  solver_.init(*this, basis_, integrator, args...);
  return;
}

template <unsigned int M, unsigned int N, unsigned int R, typename E, typename B, typename Solver>
template <typename I, typename... Args>
void PDE<M, N, R, E, B, Solver>::solve(const I &integrator, Args... args) {
  // define solver and call solve method on it
  solver_.solve(*this, basis_, integrator, args...);
  return;
}
