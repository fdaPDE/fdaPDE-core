// gradient descent optimization routine
template <unsigned int N>
template <typename... Args>
std::pair<SVector<N>, double> GradientDescentOptimizer<N>::findMinimum(const DifferentiableScalarField<N>& objective,
								       const SVector<N>& x0,
								       const Args&... args){
  // can be set true by some extension to cause a foced stop at any point of the execution
  bool customStop = false; 
  customStop |= Extension::executeInitOptimization(*this, objective, args...);
  
  // algorithm initialization
  x_old = x0;
  
  // standard termination criteria based on l^2 norm of the gradient
  error = objective.derive()(x_old).squaredNorm();
  grad_old = objective.derive()(x_old);
  update = grad_old;
  
  while (numIt < maxIt && error > tolerance && !customStop){
    customStop |= Extension::executeInitIteration(*this, objective, args...);

    // compute exact gradient
    grad_old = objective.derive()(x_old);
    update = grad_old;
    // update step    
    x_new = x_old - step*update;
    // error update: standard termination criteria based on l^2 norm of the gradient
    error = grad_old.squaredNorm();

    customStop |= Extension::executeEndIteration(*this, objective, args...);
    // prepare next iteration    
    x_old = x_new;    
    numIt++;
  }
  
  customStop |= Extension::executeEndOptimization(*this, objective, args...);
  return std::pair<SVector<N>, double>(x_old, objective(x_old));
}
