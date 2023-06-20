#ifndef __ITERATIVE_OPTIMIZER_H__
#define __ITERATIVE_OPTIMIZER_H__

#include "../../utils/Symbols.h"
#include <tuple>

// base class for iterative optimization methods
template <unsigned int N>
class IterativeOptimizer {
protected:
  // optimizer independent data (not bounded to the specific implementation)
  unsigned int maxIter_; // maximum number of iterations before forced stop
  double tolerance_; // tolerance on error
  unsigned int numIter_ = 0; // counter to keep track of the number of iterations executed
  double h_; // step employed by the optimization scheme.
  double error_ = 0; // squared l^2 norm of the gradient after the update step

  // results of the optimization
  SVector<N> minimumPoint_{};
  double objectiveValue_;

  // intial configuration of the optimizer, can be restored by extensions to bring the optimizer back to
  // its initial, user-defined, setup
  std::tuple<unsigned int, double, double> initConfiguration_{};
  
public:
  IterativeOptimizer() = default;
  IterativeOptimizer(unsigned int maxIter, double tolerance, double h)
    : maxIter_(maxIter), tolerance_(tolerance), h_(h), initConfiguration_(std::make_tuple(maxIter, tolerance, h)) {};

  // getters to optimizer configuration
  unsigned int iterations() const { return numIter_; }
  double error() const { return error_; }
  
  // getters to optimization solution
  SVector<N> getSolution() const { return minimumPoint_; }
  double getObjValue() const { return objectiveValue_; }

  // setters (allow extensions to change behaviour of the optimizer at run-time)
  void setStepSize(double h) { h_ = h; }

  // restore initial optimizer configuration
  void restore() {
    maxIter_ = std::get<0>(initConfiguration_);
    tolerance_ = std::get<1>(initConfiguration_);
    h_ = std::get<2>(initConfiguration_);
  }
};

#endif // __ITERATIVE_OPTIMIZER_H__
