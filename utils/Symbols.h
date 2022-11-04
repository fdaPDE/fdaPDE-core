#ifndef __SYMBOLS_H__
#define __SYMBOLS_H__

// Common symbols and data types used in the Core library
#include <Eigen/Core>
#include <Eigen/Sparse>

// static structures, allocated on stack at compile time.
template <unsigned int N> using SVector = Eigen::Matrix<double, N, 1>;
template <unsigned int N, unsigned int M = N> using SMatrix = Eigen::Matrix<double, N, M>;

// dynamic size linear algebra structures. Observe that such structures are stored in the heap, always use
// these if you have to deal with very big matrices or vectors (using statically allocated object can lead to
// stack overflow). See Eigen documentation for more details.
template <typename T> using DMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
template <typename T> using DVector = Eigen::Matrix<T, Eigen::Dynamic, 1>;
template <typename T> using DiagMatrix = Eigen::DiagonalMatrix<double, Eigen::Dynamic, Eigen::Dynamic>;

// sparse structures
template <typename T> using SpMatrix = Eigen::SparseMatrix<T>;

namespace fdaPDE {
  // a Triplet type (almost identical with respect to Eigen::Triplet<T>) but allowing for non const access to stored value
  // this is compatible to Eigen::setFromTriplets() method used for the construction of sparse matrices
  template <typename T>
  class Triplet {
  private:
    Eigen::Index row_, col_;
    T value_;
  public:
    Triplet() = default;
    Triplet(const Eigen::Index& row, const Eigen::Index& col, const T& value)
      : row_(row), col_(col), value_(value) {};
    
    const Eigen::Index& row() const { return row_; }
    const Eigen::Index& col() const { return col_; }
    const T& value() const { return value_; }
    T& value() { return value_; } // allow for modifications of stored value, this not allowed by Eigen::Triplet
  };
}

#endif // __SYMBOLS_H__
