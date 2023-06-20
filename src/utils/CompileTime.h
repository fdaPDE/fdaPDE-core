#ifndef __COMPILE_TIME_H__
#define __COMPILE_TIME_H__

#include <array>

// this file collects a set of constexpr routines usefull in different part of the core library.
// some routines in this file require C++14 standard (any constexpr function which is not made by a single return statement)

// prefix the name of a compile time function by ct_ to indicate that it will be evaluated at compile time

// compile time evaluation of the factorial function
constexpr unsigned int ct_factorial(const unsigned int n) {
    return n ? (n * ct_factorial(n - 1)) : 1;
}

// compile time evaluation of binomial coefficient
constexpr unsigned int ct_binomial_coefficient(const unsigned int N, const unsigned int M) {
  return ct_factorial(N)/(ct_factorial(M)*ct_factorial(N - M));
}

// compute and return the sum of elements in array A at compile time
template <unsigned int N> constexpr int ct_array_sum(std::array<unsigned, N> A) {
  int result = 0;
  for(int j = 0; j < N; ++j) result += A[j];
  return result;
}


#endif // __COMPILE_TIME_H__
