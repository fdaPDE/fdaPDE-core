// gradient descent optimization routine
template <unsigned int N>
template <typename... Args>
void GradientDescentOptimizer<N>::findMinimum(const ScalarField<N>& objective, // objective to optimize
					      const SVector<N>& x0, // initial point
					      Args&... args){
  // can be set true by some extension to cause a foced stop at any point of the execution
  bool customStop = false; 
  customStop |= Extension::executeInitOptimization(*this, objective, args...);
  
  // clean state of possibly previous execution
  this->numIter_ = 0;
  
  // algorithm initialization
  x_old_ = x0;
  x_new_ = x_old_; // always guarantee x_new_ in a valid state (if while loop skipped)
  
  // standard termination criteria based on l^2 norm of the gradient
  grad_old_ = objective.derive()(x_old_);
  this->error_ = grad_old_.squaredNorm();
  update_ = grad_old_;
  
  while (this->numIter_ < this->maxIter_ && this->error_ > this->tolerance_ && !customStop){
    customStop |= Extension::executeInitIteration(*this, objective, args...);

    std::cout << "lambda-" << this->numIter_ << ": " << x_old_ << std::endl;
    std::cout << "error: " << this->error_ << std::endl;
    
    // compute gradient
    grad_old_ = objective.derive()(x_old_);
    update_ = grad_old_;
    // update step    
    x_new_ = x_old_ - this->h_*update_;
    // error update: standard termination criteria based on l^2 norm of the gradient
    this->error_ = grad_old_.squaredNorm();

    customStop |= Extension::executeEndIteration(*this, objective, args...);
    // prepare next iteration    
    x_old_ = x_new_;    
    this->numIter_++;
  }

  std::cout << "lambda-" << this->numIter_ << ": " << x_old_ << std::endl;
  std::cout << "error: " << this->error_ << std::endl;
  
  customStop |= Extension::executeEndOptimization(*this, objective, args...);
  // finalize optimization
  this->minimumPoint_ = x_new_;
  this->objectiveValue_ = objective(x_new_);
  // restore optimizer status if some extension caused a change in its initial configuration
  this->restore();
  return;  
}
