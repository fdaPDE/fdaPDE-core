#ifndef __BACKTRACKING_ADAPTIVE_STEP_H__
#define __BACKTRACKING_ADAPTIVE_STEP_H__

#include "../../utils/Symbols.h"

namespace fdaPDE{
namespace core{
namespace OPT{
  // an extension implementation of the backatracking line search method for step selection
  class BacktrackingAdaptiveStep{
  private:
    // backtracking parameter
    double alpha_ = 1;
    double beta_  = 0.3;
    double gamma_ = 0.8;

  public:
    // default constructor
    BacktrackingAdaptiveStep() = default;
    // set parameters for backtracking. Recommended if default not works.
    BacktrackingAdaptiveStep(double alpha, double beta, double gamma) :
      alpha_(alpha), beta_(beta), gamma_(gamma) {};
  
    // backtracking based step search
    template <typename Optimizer, typename Objective>
    bool initIteration(Optimizer& opt, Objective& obj){
      // line search. 
      do{ alpha_ *= beta_; }
      while( obj(opt.x_old()) - (obj( opt.x_old() - alpha_*opt.gradient_old()))
	     + gamma_*alpha_*(opt.gradient_old().dot(opt.update_vector())) < 0 );
    
      opt.setStepSize(alpha_);
      return false;
    }
  };
  
}}}
  
#endif // __BACKTRACKING_ADAPTIVE_STEP_H__
