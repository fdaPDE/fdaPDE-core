#ifndef __BFGS_H__
#define __BFGS_H__

#include "../../utils/Symbols.h"
#include "../../utils/fields/ScalarField.h"
#include "../extensions/Extension.h"
#include "IterativeOptimizer.h"

namespace fdaPDE{
namespace core{
namespace OPT{
  
  // implementation of the BFGS optimizer
  template <unsigned int N>
  class BFGSOptimizer : public IterativeOptimizer<N> {
  private:
    // internal status of the optimizer 
    SVector<N> x_old_{};    // value of the optimization point before the update step
    SVector<N> x_new_{};    // value of the optimization point after the update step
    SVector<N> update_{};   // update vector computed at each step
    SVector<N> grad_old_{}; // value of the field's gradient before the update step
    SVector<N> grad_new_{}; // value of the field's gradient after the update step
    SMatrix<N> hessian_{};  // value of the hessian matrix approximation at iteration i.
    
  public:
    // constructor
    BFGSOptimizer() = default;
    BFGSOptimizer(unsigned int maxIter, double tolerance, double h)
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
    SVector<N> gradient_new() const { return grad_new_; }
    SMatrix<N> hessian() const { return hessian_; }
  };

#include "BFGS.tpp"
}}}
#endif // __BFGS_H__
