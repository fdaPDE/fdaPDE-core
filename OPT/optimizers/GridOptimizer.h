#ifndef __GRID_OPTIMIZER_H__
#define __GRID_OPTIMIZER_H__

#include <array>
#include "../../utils/Symbols.h"
#include "../../utils/fields/ScalarField.h"
#include "../extensions/Extension.h"

namespace fdaPDE{
namespace core{
namespace OPT{
  // optimize a given scalar field over an N-dimensional grid of equidistant points
  template <int N>
  class GridOptimizer{
  private:
    // internal status of the optimizer
    SVector<N> x_old_{}; // value of the optimization point at current iteration
    
    // results of the optimization
    SVector<N> minimumPoint_;
    double objectiveValue_;
  public:
    // constructor
    GridOptimizer() = default;
  
    // perform the minimum search over an automatically built grid of equidistant N-dimensional points
    template <typename F, typename... Args>
    void findMinimum(ScalarField<N,F>& objective, // objective to optimize
		     const std::array<std::pair<double,double>, N>& domainLimits, // domain where search for minimum
		     const std::array<double, N>& stepSizes, // stepSizes for each dimension of the grid
		     Args&... args);
    template <typename F, typename... Args>
    void findMinimum(ScalarField<N,F>& objective, // objective to optimize
		     const std::array<std::pair<double,double>, N>& domainLimits, // domain where search for minimum
		     double stepSize, // use the same step size along all dimensions
		     Args&... args);
    
    // perform the minimum search over a vector of user defined values
    template <typename F, typename... Args>
    void findMinimum(ScalarField<N,F>& objective, // objective to optimize
		     const std::vector<SVector<N>>& pointList, // set of pointw where to perform the search
		     Args&... args);

    // getters to internal state
    SVector<N> x_old() const { return x_old_; }
    
    SVector<N> getSolution() const { return minimumPoint_; }
    double getObjValue() const { return objectiveValue_; }
  };

#include "GridOptimizer.tpp"
}}}

#endif // __GRID_OPTIMIZER_H__
