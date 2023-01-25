template <int N>
template <typename F, typename... Args>
void GridOptimizer<N>::findMinimum(ScalarField<N,F>& objective, const std::array<std::pair<double,double>, N>& domainLimits,
				   const std::array<double, N>& stepSizes, Args&... args){
  // can be set true by some extension to cause a foced stop at any point of the execution
  bool customStop = false; 
  customStop |= Extension::executeInitOptimization(*this, objective, args...);

  // number of iterations for each dimension
  std::array<int, N> numberIteration;
  // prepare data structure to execute the N-dimensional scan
  SVector<N> gridPoint;
  
  // init helper data structures
  for(size_t i = 0; i < N; ++i){
    numberIteration[i] = std::ceil((domainLimits[i].second - domainLimits[i].first)/stepSizes[i]) + 1;
    gridPoint[i] = domainLimits[i].first;
  }

  // initial minimum
  double minimum = objective(gridPoint);
  SVector<N> minPoint = gridPoint;
  x_old_ = gridPoint;
  
  // counter to keep track of iterations executed
  // this is a vector of N elements started at [0, 0, ..., 0]. The algorithms always increments the vector at
  // position 0 and when it reaches the maximum number of iterations for that dimension it is restarted from 0 incrementing
  // recursively the next positions following the same schema. Loop stops when currentIteration = numberIteration
  std::array<int, N> currentIteration = {};
  
  // search for minimum
  while(currentIteration[N-1] < numberIteration[N-1] && !customStop){ // loop until the last dimension has not been scan entirely
    customStop |= Extension::executeInitIteration(*this, objective, args...);

    // evaluate function at current point
    gridPoint[0] += stepSizes[0];
    double x = objective(gridPoint);

    x_old_ = gridPoint; // update current point
    // update minimum
    if(x < minimum){
      minimum = x;
      minPoint = gridPoint;
    }

    // increment counter at first dimension
    currentIteration[0]++; 
    
    // trigger increment in next elements of the counter if necessary
    for(int i = 0; currentIteration[i] == numberIteration[i] && i < N-1; ++i){ // dimension i has been spanned
      currentIteration[i] = 0;               // reset dimension i
      gridPoint[i] = domainLimits[i].first;
      
      currentIteration[i+1]++;               // increment counter for next dimension
      gridPoint[i+1] += stepSizes[i+1];
    }
    customStop |= Extension::executeEndIteration(*this, objective, args...);
  }
  customStop |= Extension::executeEndOptimization(*this, objective, args...);
  // finalize optimization
  minimumPoint_ = minPoint;
  objectiveValue_ = minimum;
  return;
}

template <int N>
template <typename F, typename... Args>
void GridOptimizer<N>::findMinimum(ScalarField<N,F>& objective, const std::array<std::pair<double,double>, N>& domainLimits, 
				     double stepSize, Args&... args){
  // build stepSizes vector
  std::array<double, N> stepSizes;
  for(std::size_t i = 0; i < N ; ++i){
    stepSizes[i] = stepSize;
  }
  // call general routine
  findMinimum(objective, domainLimits, stepSizes, args...);
  return;
}


template <int N>
template <typename F, typename... Args>
void GridOptimizer<N>::findMinimum(ScalarField<N,F>& objective, const std::vector<SVector<N>>& pointList, Args&... args) {
  // can be set true by some extension to cause a foced stop at any point of the execution
  bool customStop = false; 
  // prepare data structure to execute optimization
  SVector<N> gridPoint;
  gridPoint = pointList[0]; // init point
  x_old_ = gridPoint;

  customStop |= Extension::executeInitOptimization(*this, objective, args...);  
  // initial minimum
  double minimum = objective(gridPoint);
  SVector<N> minPoint = gridPoint;

  for(std::size_t i = 1; i < pointList.size() && !customStop; ++i){ // loop until all supplied points have not been used
    x_old_ = pointList[i]; // update current point
    customStop |= Extension::executeInitIteration(*this, objective, args...);

    // evaluate function at current point
    gridPoint = pointList[i];
    double x = objective(gridPoint);
    
    // update minimum if necessary
    if(x < minimum){
      minimum = x;
      minPoint = gridPoint;
    }
    customStop |= Extension::executeEndIteration(*this, objective, args...);
  }
  customStop |= Extension::executeEndOptimization(*this, objective, args...);
  // finalize optimization
  minimumPoint_ = minPoint;
  objectiveValue_ = minimum;
  return;
}
