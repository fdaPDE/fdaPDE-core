#ifndef __SUMMARIZE_H__
#define __SUMMARIZE_H__

#include <cstddef>
#include <vector>
#include <string>
#include <iostream>

class Summarize {

private:
  // prints an SVector (a.k.a. eigen vector) by row instead of by column
  std::string rowPrint(const std::vector<double>& input){
    std::string output = "[" + std::to_string(input[0]);

    for(size_t j = 1; j < input.size(); ++j){
      output += ", " + std::to_string(input[j]);
    }

    output += "]";
    return output;
  }
  
public:
  Summarize() = default;
  
  template <typename Optimizer, typename Objective>
  bool endOptimization(Optimizer& opt, Objective& obj){
    // transform SVector in standard vector for easy of printing
    unsigned int I = opt.getXnew().rows();
    std::vector<double> v(I, 0);
    for(size_t i = 0; i < I; ++i){
      v[i] = opt.getXnew()[i];
    }
    
    std::cout << "Summary of optimization routine" << std::endl;
    std::cout << "   type of optimization:     " << opt.description << std::endl; 
    std::cout << "   number of iterations:     " << opt.getNumIteration() << std::endl;
    std::cout << "   optimium point found:     " << rowPrint(v) << std::endl;
    std::cout << "   objective optimum value:  " << obj(opt.getXnew()) << std::endl;
    std::cout << "   l2 error:                 " << opt.getError() << std::endl;
    
    return false;
  }
};

#endif // __SUMMARIZE_H__
