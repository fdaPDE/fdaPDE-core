// optimizes the objective on the supplied grid
template <int N>
template <typename F, typename... Args>
void GridOptimizer<N>::optimize
(ScalarField<N,F>& objective, const std::vector<SVector<N>>& grid, Args&... args) {
  // can be set true by some extension to cause a foced stop at any point of the execution
  bool customStop = false; 
  // algorithm initialization
  x_old_ = grid[0];
  value_ = objective(x_old_);
  optimum_ = x_old_;
  // optimize field over supplied grid
  for(std::size_t i = 1; i < grid.size() && !customStop; ++i){
    x_old_ = grid[i]; // update current point
    // evaluate function at current point
    double x = objective(x_old_);
    // update minimum if better optimum found
    if(x < value_){
      value_ = x;
      optimum_ = x_old_;
    }
  }
  return;
}

// computes a grid of points with the provided step sizes and optimizes the objective on it
template <int N>
template <typename F, typename... Args>
void GridOptimizer<N>::optimize
(ScalarField<N,F>& objective, const std::array<std::pair<double,double>, N>& domain,
 const std::array<double, N>& steps, Args&... args){
  // assemble grid of points
  std::vector<SVector<N>> grid;
  std::size_t n_points = 0;
  SVector<N> p; // auxiliary point used for grid construction
  // compute number of points in the grid and init p
  for(std::size_t i = 0; i < N; ++i){
    n_points += std::ceil((domain[i].second - domain[i].first)/steps[i]) + 1;
    p[i] = domain[i].first;
  }
  grid.reserve(n_points); // reserve space to avoid useless reallocations
  // compute grid
  for(std::size_t n = 0; n < n_points; ++n){
    grid.emplace_back(p);
    p += steps[0];
    // trigger increment in next elements of the counter if necessary
    for(int i = 0; p[i] >= domain[i].second && i < N-1; ++i){ 
      p[i] = domain[i].first;
      p[i+1] += steps[i+1];
    }
  }
  // call optimizer on built grid
  optimize(objective, grid, args...);
  return;
}

// computes a grid of equidistant points in all dimensions and optimizes the objective on it
template <int N>
template <typename F, typename... Args>
void GridOptimizer<N>::optimize
(ScalarField<N,F>& objective, const std::array<std::pair<double,double>, N>& domain, 
 double step, Args&... args){
  // build stepSizes vector
  std::array<double, N> steps;
  for(std::size_t i = 0; i < N ; ++i) steps[i] = step;
  // compute grid and optimize
  optimize(objective, domain, steps, args...);
  return;
}
