#ifndef __SYMBOLS_H__
#define __SYMBOLS_H__

// Common symbols and data types used in the Core library
#include <memory>
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

  // hash function for std::pair (allow pairs as key of unordered_map). inspired from boost::hash
  struct pair_hash{
    template <typename T1, typename T2>
    std::size_t operator()(const std::pair<T1,T2>& pair) const {
      std::size_t hash = 0;
      hash ^= std::hash<T1>()(pair.first)  + 0x9e3779b9 + (hash<<6) + (hash>>2);
      hash ^= std::hash<T2>()(pair.second) + 0x9e3779b9 + (hash<<6) + (hash>>2);
      return hash;
    }
  };
  // hash function for DMatrix<T>, allows DMatrix<T> as key in an unordered map.
  struct matrix_hash {
    template <typename T>
    std::size_t operator()(const DMatrix<T>& matrix) const {
      std::size_t hash = 0;
      for(std::size_t i = 0; i < matrix.rows(); ++i)
	for(std::size_t j = 0; j < matrix.cols(); ++j)
	  hash ^= std::hash<T>()(matrix(i,j))  + 0x9e3779b9 + (hash<<6) + (hash>>2);
      return hash;
    };
  };

  // oredering relation for SVector<N>, allows SVector<N> to be keys of std::map
  template <unsigned int N>
  struct s_vector_compare {
    bool operator()(const SVector<N>& lhs, const SVector<N>& rhs) const {
      return std::lexicographical_compare(lhs.begin(),lhs.end(), rhs.begin(),rhs.end());
    };
  };
  
  // a movable wrapper for Eigen::SparseLU (Eigen::SparseLU has a deleted copy and assignment operator)
  template <typename T>
  class SparseLU {
  private:
    typedef Eigen::SparseLU<T, Eigen::COLAMDOrdering<int>> SparseLU_;
    std::shared_ptr<SparseLU_> solver_; // wrap Eigen::SparseLU into a movable object
  public:
    // default constructor
    SparseLU() = default;
    // we expose only the compute and solve methods of Eigen::SparseLU
    void compute(const T& matrix){
      // initialize pointer
      solver_ = std::make_shared<SparseLU_>();
      solver_->compute(matrix);
    }
    // solve method, dense rhs operand
    template <typename Rhs>
    const Eigen::Solve<SparseLU_, Rhs>
    solve(const Eigen::MatrixBase<Rhs>& b) const { 
      return solver_->solve(b);
    }
    template <typename Rhs> // sparse rhs operand
    const Eigen::Solve<SparseLU_, Rhs>
    solve(const Eigen::SparseMatrixBase<Rhs>& b) const { 
      return solver_->solve(b);
    }
    // direct access to Eigen::SparseLU
    std::shared_ptr<SparseLU_> operator->() { return solver_; }
  };

  // test for floating point equality based on relative error.
  const double DOUBLE_TOLERANCE = 50*std::numeric_limits<double>::epsilon(); // approx 10^-14
  template <typename T>
  typename std::enable_if<!std::numeric_limits<T>::is_integer, bool>::type
  almost_equal(T a, T b, T epsilon){
    return std::fabs(a-b) < epsilon ||
      std::fabs(a-b) < ((std::fabs(a) < std::fabs(b) ? std::fabs(b) : std::fabs(a)) * epsilon);
  }
  // default to DOUBLE_TOLERANCE
  template <typename T>
  typename std::enable_if<!std::numeric_limits<T>::is_integer, bool>::type
  almost_equal(T a, T b){ return almost_equal(a,b, DOUBLE_TOLERANCE); }

}

#endif // __SYMBOLS_H__
