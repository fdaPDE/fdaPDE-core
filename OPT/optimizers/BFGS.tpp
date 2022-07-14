// BFGS optimization routine
template <unsigned int N>
template <typename... Args>
std::pair<SVector<N>, double> BFGSOptimizer<N>::findMinimum(const DifferentiableScalarField<N>& objective,
							    const SVector<N>& x0,
							    const Args&... args){
  // can be set true by some extension to cause a foced stop at any point of the execution
  bool customStop = false; 
  customStop |= Extension::executeInitOptimization(*this, objective, args...);
  
  // algorithm initialization
  x_old = x0;

  // set the hessian approximation at first iteration equal to the identity matrix
  // this will make the first iteration of the algorithm to behave as a gradient descent
  // step. hessian will be corrected as the algorithm goes on.
  hessian = SMatrix<N>::Identity();

  grad_old = objective.derive()(x_old);
  if (grad_old.isApprox(SVector<N>::Zero())) // gradient is zero, already at stationary point
    return std::pair<SVector<N>, double>(x_old, objective(x_old));

  error = grad_old.squaredNorm();

  while(numIt < maxIt && error > tolerance && !customStop){
    update = hessian*grad_old;
    customStop |= Extension::executeInitIteration(*this, objective, args...);
    
    // update along descent direction
    x_new = x_old - step*update;
    // gradient update
    grad_new = objective.derive()(x_new);
    if (grad_new.isApprox(SVector<N>::Zero())){ // gradient is zero, already at stationary point
      return std::pair<SVector<N>, double>(x_new, objective(x_new));
    }
      
    // update inverse hessian approximation
    SVector<N> deltaX    = x_new - x_old;
    SVector<N> deltaGrad = grad_new - grad_old;
    double xg = deltaX.dot(deltaGrad);     // inner product between deltaX and deltaGrad
    SVector<N> hx = hessian*deltaGrad;     // product between hessian matrix and deltaGrad

    // see references for detailed derivation of the equations
    SMatrix<N> U = (1 + (deltaGrad.dot(hx))/xg)*((deltaX*deltaX.transpose())/xg);
    SMatrix<N> V = ((hx*deltaX.transpose() + deltaX*hx.transpose()))/xg;
    
    hessian += U - V; // hessian approximation update

    // prepare next iteration
    error = grad_new.squaredNorm();
    customStop |= Extension::executeEndIteration(*this, objective, args...);

    x_old = x_new;
    grad_old = grad_new;
    numIt++;
  }

  customStop |= Extension::executeEndOptimization(*this, objective, args...);
  return std::pair<SVector<N>, double>(x_old, objective(x_old));
}
