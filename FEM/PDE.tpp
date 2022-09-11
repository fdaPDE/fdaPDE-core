// store in the format (boundaryID, { ... }) the dirichlet boundary conditions, where { ... } is the time series of the
// data at boundary for boundary node boundaryID
template <unsigned int M, unsigned int N, unsigned int R, typename E, typename Solver>
void PDE<M, N, R, E, Solver>::setDirichletBC(const DMatrix<double>& data){
 for(size_t j = 0; j < domain_.nodes(); ++j){
    // if j is a node on the domain boundary store the pair (node ID - boundary value)
    if(domain_.isOnBoundary(j)){
      boundaryData_[j] = data.row(j); // O(1) complexity
    }
  }
  return;
}

template <unsigned int M, unsigned int N, unsigned int R, typename E, typename Solver>
template <typename B, typename I, typename... Args>
void PDE<M, N, R, E, Solver>::init(const B &base, const I &integrator, Args... args) {
  // precomputes some quantites of interest for high level users of FEM. Do not solve the PDE (which means no linear system is solved)
  solver_.init(*this, base, integrator, args...);
  return;
}

template <unsigned int M, unsigned int N, unsigned int R, typename E, typename Solver>
template <typename B, typename I, typename... Args>
void PDE<M, N, R, E, Solver>::solve(const B &base, const I &integrator, Args... args) {
  // define solver and call solve method on it
  solver_.solve(*this, base, integrator, args...);
  return;
}
