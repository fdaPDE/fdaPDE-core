#ifndef __CONSTANTS_H__
#define __CONSTANTS_H__

#include <limits>

// a set of usefull constants
namespace fdapde {
namespace testing {

  // the treshold under which two doubles are considered equal ** DO NOT CHANGE THIS TOLERANCE **
  constexpr double DOUBLE_TOLERANCE = 1e-7;
  constexpr double MACHINE_EPSILON  = std::numeric_limits<double>::epsilon(); // approx 2.22*10^-16
  constexpr double pi = 3.14159265358979323846;
  
}}

#endif // __CONSTANTS_H__
