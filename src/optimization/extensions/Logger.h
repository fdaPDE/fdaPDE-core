#ifndef __LOGGER_H__
#define __LOGGER_H__

namespace fdaPDE {
namespace core {
namespace OPT {

  // an extension recording the execution of optimizer O
  template <typename O>
  class Logger {
  private:
    std::vector<SVector<O::N>> x_vect_; // points explored during optimization process
    std::vector<double> y_vect_; // objective values found during optimization

    // store point-objective pair
    template <typename Optimizer, typename Objective>
    void store(Optimizer& opt, Objective& obj) {
      x_vect_.emplace_back(opt.x_old());
      y_vect_.emplace_back(obj(opt.x_old()));
    }
    
  public:
    // constructor
    Logger() = default;

    // injected logic
    template <typename Optimizer, typename Objective>
    bool initIteration  (Optimizer& opt, Objective& obj) { store(opt, obj); return false; }
    template <typename Optimizer, typename Objective>
    bool endOptimization(Optimizer& opt, Objective& obj) { store(opt, obj); return false; }

    // getters
    const std::vector<SVector<O::N>>& x_vect() const { return x_vect_; }
    const std::vector<double>& y_vect() const { return y_vect_; }
  };
  
}}}

#endif // __LOGGER_H_
