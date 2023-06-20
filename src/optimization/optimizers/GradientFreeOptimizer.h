#ifndef __GRADIENT_FREE_OPTIMIZER_H__
#define __GRADIENT_FREE_OPTIMIZER_H__

namespace fdaPDE{
namespace core{
namespace OPT{

  // base class for any **gradient free** optimizer
  // gradient free optimizers typically explore the search space by keeping track of the current optimum only
  template <int N>
  class GradientFreeOptimizer {
  protected:
    // internal status of the optimizer
    SVector<N> x_old_{}; // visited point at current iteration
    SVector<N> optimum_; // current best optimum found
    double value_;       // objective value at current best optimum
  public:
    GradientFreeOptimizer() = default;
    // getters
    SVector<N> x_old() const { return x_old_; }
    SVector<N> optimum() const { return optimum_; }
    double value() const { return value_; }
  };

  // requires a type Base in the scope of this macro
#define IMPORT_GRADIENT_FREE_OPT_SYMBOLS			      \
  using Base::value_;   /* objective value at current best optimum */ \
  using Base::optimum_; /* current best optimum found */	      \
  using Base::x_old_;   /* visited point at current iteration */      \
  
}}}

#endif // __GRADIENT_FREE_OPTIMIZER_H__
