#ifndef __GRADIENT_DESCENT_OPTIMIZER__
#define __GRADIENT_DESCENT_OPTIMIZER__

#include "../../utils/Symbols.h"
#include "../../utils/fields/ScalarField.h"
#include "../extensions/Extension.h"
#include "IterativeOptimizer.h"

namespace fdaPDE{
namespace core{
namespace OPT{

  // gradient descent method
  template <unsigned int N>
  class GradientDescentOptimizer : public IterativeOptimizer<N>{
  private:

    // internal status of the optimizer 
    SVector<N> x_old_{};    // value of the optimization point before the update step
    SVector<N> x_new_{};    // value of the optimization point after the update step
    SVector<N> update_{};   // update vector computed at each step
    SVector<N> grad_old_{}; // value of the gradient before the update step
  public:
    // constructor
    GradientDescentOptimizer() = default;
    GradientDescentOptimizer(unsigned int maxIter, double tolerance, double h)
      : IterativeOptimizer<N>(maxIter, tolerance, h) {};
      
    // optimization routine, depending on the objective type the method employs an approximation or the exact expression
    // of gradient and/or hessian function.
    template <typename... Args>
    void findMinimum(const ScalarField<N>& objective, // objective to optimize
		     const SVector<N>& x0, // initial point
		     Args&... args);

    // getters to internal state
    SVector<N> x_old() const { return x_old_; }
    SVector<N> x_new() const { return x_new_; }
    SVector<N> update_vector() const { return update_; }
    SVector<N> gradient_old() const { return grad_old_; }
  };

#include "GradientDescent.tpp"
}}}
  
#endif // __GRADIENT_DESCENT_OPTIMIZER__
