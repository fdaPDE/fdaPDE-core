#ifndef __BACKTRACKING_ADAPTIVE_STEP_H__
#define __BACKTRACKING_ADAPTIVE_STEP_H__

// an extension implementing the backtracking policy for step selection
#include "../../utils/Symbols.h"

class BacktrackingAdaptiveStep{

private:
  // backtracking parameter
  double alpha = 1;
  double beta  = 0.3;
  double gamma = 0.8;

public:
  // default constructor
  BacktrackingAdaptiveStep() = default;

  // set parameters for backtracking. Recommended if default not works.
  BacktrackingAdaptiveStep(double alpha_, double beta_, double gamma_) :
    alpha(alpha_), beta(beta_), gamma(gamma_) {};
  
  // backtracking based step search
  template <typename Optimizer, typename Objective>
  bool initIteration(Optimizer& opt, Objective& obj){
    // line search 
      do{ alpha *= beta; }
      while( obj(opt.getXold()) - (obj( opt.getXold() - alpha*opt.getGradientOld()))
	     + gamma*alpha*(opt.getGradientOld().dot(opt.getUpdate())) < 0 );
    
    opt.setStepSize(alpha);
    return false;
  }
  
};

#endif // __BACKTRACKING_ADAPTIVE_STEP_H__
