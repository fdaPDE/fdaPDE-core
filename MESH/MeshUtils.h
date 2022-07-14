#ifndef __MESH_UTILS_H__
#define __MESH_UTILS_H__

// compile time evaluation of the factorial function
constexpr unsigned int FACTORIAL(const unsigned int n) {
    return n ? (n * FACTORIAL(n - 1)) : 1;
}

// compile time evaluation of binomial coefficient
constexpr unsigned int BINOMIAL_COEFFICIENT(const unsigned int N,
					   const unsigned int M) {
  return FACTORIAL(N)/(FACTORIAL(M)*FACTORIAL(N - M));
}

// evaluate at compile time the number of vertices of a given element
// knowing its order N and the dimension of the embedding space M
constexpr unsigned int N_VERTICES(const unsigned int M,
				  const unsigned int N) {
  return M+1;
}


#endif
