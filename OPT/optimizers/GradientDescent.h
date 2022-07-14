#ifndef __GRADIENT_DESCENT_OPTIMIZER__
#define __GRADIENT_DESCENT_OPTIMIZER__

#include <tuple>
#include "../../utils/Symbols.h"
#include "../../utils/fields/ScalarField.h"
#include "../extensions/Extension.h"

namespace fdaPDE{
namespace core{
namespace OPT{
  // gradient descent method
  template <unsigned int N>
  class GradientDescentOptimizer{

  private:
    double step;                      // step employed by the optimization scheme.

    // internal status of the optimizer 
    SVector<N> x_old;                 // value of the optimization point before the update step
    SVector<N> x_new;                 // value of the optimization point after the update step
    SVector<N> update;                // update vector computed at each step
    double error;                     // squared l^2 norm of the gradient after the update step
    SVector<N> grad_old;              // value of the gradient before the update step
    
    // optimization problem data
    unsigned int maxIt;               // maximum number of iterations before forced stop
    double tolerance;                 // tolerance on error
    unsigned int numIt = 0;           // counter to keep track of the number of iterations executed
    
  public:
    // constructor
    GradientDescentOptimizer(unsigned int maxIt_, double tolerance_)
      : maxIt(maxIt_), tolerance(tolerance_) {};

    // set step size (use this if you want to employ a fixed step method. For adaptive step, use a proper extension)
    void setStepSize(double step_) { step = step_; }
  
    // getters to internal state
    unsigned int getNumIteration() const { return numIt;    }
    double getError()              const { return error;    }
    SVector<N> getXold()           const { return x_old;    }
    SVector<N> getXnew()           const { return x_new;    }
    SVector<N> getUpdate()         const { return update;   }
    double getStep()               const { return step;     }
    SVector<N> getGradientOld()    const { return grad_old; }
    
    // optimization routine
    template <typename... Args>
    std::pair<SVector<N>, double> findMinimum(const DifferentiableScalarField<N>& objective, const SVector<N>& x0, const Args&... args);

    const std::string description = "Gradient descent method";
  };

#include "GradientDescent.tpp"
}}}
  
#endif // __GRADIENT_DESCENT_OPTIMIZER__
