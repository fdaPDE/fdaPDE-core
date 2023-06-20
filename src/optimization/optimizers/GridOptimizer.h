#ifndef __GRID_OPTIMIZER_H__
#define __GRID_OPTIMIZER_H__

#include <array>
#include "../../utils/Symbols.h"
#include "../../utils/fields/ScalarField.h"
#include "../extensions/Extension.h"
#include "GradientFreeOptimizer.h"

namespace fdaPDE{
namespace core{
namespace OPT{
  
  // optimize ScalarField<N,F> over an N-dimensional grid of points
  template <int N>
  struct GridOptimizer : public GradientFreeOptimizer<N> {
    typedef GradientFreeOptimizer<N> Base;
    IMPORT_GRADIENT_FREE_OPT_SYMBOLS;
    // constructor
    GridOptimizer() = default;
  
    // perform the minimum search over an automatically built grid of N-dimensional points
    template <typename F, typename... Args>
    void optimize(ScalarField<N,F>& objective, // objective to optimize
		  const std::array<std::pair<double,double>, N>& domain, // search space
		  const std::array<double, N>& steps, // steps for each dimension of the grid
		  Args&... args);
    template <typename F, typename... Args>
    void optimize(ScalarField<N,F>& objective, // objective to optimize
		  const std::array<std::pair<double,double>, N>& domain, // search space
		  double step, // use the same step for all dimensions
		  Args&... args);
    
    // perform the minimum search over a vector of user defined values
    template <typename F, typename... Args>
    void optimize(ScalarField<N,F>& objective, // objective to optimize
		  const std::vector<SVector<N>>& grid, // search space
		  Args&... args);
  };

#include "GridOptimizer.tpp"

}}}

#endif // __GRID_OPTIMIZER_H__
