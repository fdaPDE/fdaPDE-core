#ifndef __NEWTON_H__
#define __NEWTON_H__

#include <Eigen/Dense>
#include "../../utils/Symbols.h"
#include "../../utils/fields/ScalarField.h"
#include "../extensions/Extension.h"
#include "IterativeOptimizer.h"

namespace fdaPDE{
namespace core{
namespace OPT{  

  // newton method based on exact gradient and hessian computation. It requires second differentiabiliy
  // of the scalar field passed as objective
  template <unsigned int N_>
  class NewtonOptimizer : public IterativeOptimizer<N_> {
  private:  
    // internal status of the optimizer 
    SVector<N_> x_old_{};  // value of the optimization point before the update step
    SVector<N_> x_new_{};  // value of the optimization point after the update step
    SVector<N_> update_{}; // update vector computed at each step
    SVector<N_> grad_old_{}; // value of the field's gradient before the update step
    SMatrix<N_> hessian_{};  // value of the hessian matrix approximation at iteration i.
  public:
    // compile time informations
    static constexpr unsigned int N = N_;
    // constructor
    NewtonOptimizer() = default;
    NewtonOptimizer(unsigned int maxIter, double tolerance, double h)
      : IterativeOptimizer<N_>(maxIter, tolerance, h) {};
    
    // optimization routine, depending on the objective type the method employs an approximation or the exact expression
    // of gradient and/or hessian function.
    template <typename F, typename... Args>
    void findMinimum(F& objective, // objective to optimize
		     const SVector<N_>& x0, // initial point
		     Args&... args);

    // getters to internal state
    SVector<N_> x_old() const { return x_old_;  }
    SVector<N_> x_new() const { return x_new_;  }
    SVector<N_> update_vector() const { return update_; }
    SVector<N_> gradient_old() const { return grad_old_; }    
    SMatrix<N_> hessian() const { return hessian_; }
  };

#include "Newton.tpp"
}}}

#endif // __NEWTON_H__
