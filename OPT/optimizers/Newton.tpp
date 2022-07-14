// newton optimization routine
template <unsigned int N>
template <typename... Args>
void NewtonOptimizer<N>::findMinimum(const ScalarField<N>& objective, // objective to optimize
				     const SVector<N>& x0, // starting point
				     const Args&... args) {
  // can be set true by some extension to cause a foced stop at any point of the execution
  bool customStop = false; 
  customStop |= Extension::executeInitOptimization(*this, objective, args...);

  // clean state of possibly previous execution
  numIter_ = 0;
  // algorithm initialization
  x_old_ = x0;
  x_new_ = x_old_; // always guarantee x_new_ in a valid state (if while loop skipped)
  error_ = objective.derive()(x0).norm();
  SVector<N> gradient{};
  SMatrix<N> hessian{};

  // start loop
  while (numIter_ < maxIter_ && error_ > tolerance_ && !customStop){
    customStop |= Extension::executeInitIteration(*this, objective, args...);

    // compute hessian and gradient either by resorting to a numerical approximation or to their analytical expression
    // depending on the type of the objective
    gradient = objective.derive()(x_old_);
    hessian  = objective.deriveTwice()(x_old_);

    // solve linear system by using an Householder QR decomposition with column-pivoting: A*P = Q*R
    Eigen::ColPivHouseholderQR<SMatrix<N>> QRdecomposition(hessian);
    update_ = QRdecomposition.solve(gradient);

    // update step
    x_new_ = x_old_ - h_*update_;
    // compute new error (L2 norm of gradient vector)
    error_ = objective.derive()(x_new_).norm();

    customStop |= Extension::executeEndIteration(*this, objective, args...);
    // prepare next iteration
    x_old_ = x_new_;    
    numIter_++;
  }
  
  Extension::executeEndOptimization(*this, objective, args...);
  // finalize optimization
  minimumPoint_ = x_new_;
  objectiveValue_ = objective(x_new_);
  return;
}
