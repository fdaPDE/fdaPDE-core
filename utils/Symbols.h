#ifndef __SYMBOLS_H__
#define __SYMBOLS_H__

// Common symbols and data types used in the Core library
#include <Eigen/Core>
#include <Eigen/Sparse>

// static structures, allocated on stack at compile time.
template <unsigned int N> using SVector = Eigen::Matrix<double, N, 1>;
template <unsigned int N> using SMatrix = Eigen::Matrix<double, N, N>;

// dynamic size linear algebra structures. Observe that such structures are stored in the heap, always use
// these if you have to deal with very big matrices or vectors (using statically allocated object can lead to
// stack overflow). See Eigen documentation for more details.
using DMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
using DVector = Eigen::Matrix<double, Eigen::Dynamic, 1>;

// sparse structures
using SpMatrix = Eigen::SparseMatrix<double>;

#endif // __SYMBOLS_H__
