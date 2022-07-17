// BFGS optimization routine
template <unsigned int N>
template <typename... Args>
void BFGSOptimizer<N>::findMinimum(const ScalarField<N>& objective, // objective to optimize
				   const SVector<N>& x0, // starting point
				   Args&... args){
  // can be set true by some extension to cause a foced stop at any point of the execution
  bool customStop = false; 
  customStop |= Extension::executeInitOptimization(*this, objective, args...);

  // clean state of possibly previous execution
  this->numIter_ = 0;
  
  // algorithm initialization
  x_old_ = x0;
  x_new_ = x_old_; // always guarantee x_new_ in a valid state (if while loop skipped)

  // set the hessian approximation at first iteration equal to the identity matrix
  // this will make the first iteration of the algorithm to behave as a gradient descent
  // step. hessian will be corrected as the algorithm goes on.
  hessian_ = SMatrix<N>::Identity();

  grad_old_ = objective.derive()(x_old_);
  if (grad_old_.isApprox(SVector<N>::Zero())){ // gradient is zero, already at stationary point
    // finalize optimization
    this->minimumPoint_ = x_old_;
    this->objectiveValue_ = objective(x_old_);
    // restore optimizer status if some extension caused a change in its initial configuration
    this->restore();
    return;
  }
  this->error_ = grad_old_.squaredNorm();
  
  while(this->numIter_ < this->maxIter_ && this->error_ > this->tolerance_ && !customStop){
    update_ = hessian_*grad_old_;
    customStop |= Extension::executeInitIteration(*this, objective, args...);
    
    // update along descent direction
    x_new_ = x_old_ - this->h_*update_;
    // gradient update
    grad_new_ = objective.derive()(x_new_);
    if (grad_new_.isApprox(SVector<N>::Zero())){ // gradient is zero, already at stationary point
      // finalize optimization
      this->minimumPoint_ = x_new_;
      this->objectiveValue_ = objective(x_new_);
      // restore optimizer status if some extension caused a change in its initial configuration
      this->restore();
      return;
    }

    // update inverse hessian approximation
    SVector<N> deltaX    = x_new_ - x_old_;
    SVector<N> deltaGrad = grad_new_ - grad_old_;
    double xg = deltaX.dot(deltaGrad);     // inner product between deltaX and deltaGrad
    SVector<N> hx = hessian_*deltaGrad;     // product between hessian matrix and deltaGrad

    // see references for detailed derivation of the equations
    SMatrix<N> U = (1 + (deltaGrad.dot(hx))/xg)*((deltaX*deltaX.transpose())/xg);
    SMatrix<N> V = ((hx*deltaX.transpose() + deltaX*hx.transpose()))/xg;
    
    hessian_ += U - V; // hessian approximation update

    // prepare next iteration
    this->error_ = grad_new_.squaredNorm();
    customStop |= Extension::executeEndIteration(*this, objective, args...);
    // execute custom action before overwriting x_old_
    x_old_ = x_new_;
    grad_old_ = grad_new_;
    this->numIter_++;
  }

  customStop |= Extension::executeEndOptimization(*this, objective, args...);
  // finalize optimization
  this->minimumPoint_ = x_new_;
  this->objectiveValue_ = objective(x_new_);
  // restore optimizer status if some extension caused a change in its initial configuration
  this->restore();
  return;  
}
