// This file is part of fdaPDE, a C++ library for physics-informed
// spatial and functional data analysis.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#ifndef __FDAPDE_SYMBOLS_H__
#define __FDAPDE_SYMBOLS_H__

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <memory>
#include "traits.h"

// static structures, allocated on stack at compile time.
template <int N, typename T = double> using SVector = Eigen::Matrix<T, N, 1>;
template <int N, int M = N, typename T = double> using SMatrix = Eigen::Matrix<T, N, M>;

// dynamic size, heap-appocated, structures.
template <typename T, int Options_ = Eigen::ColMajor>
using DMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Options_>;
template <typename T> using DVector    = Eigen::Matrix<T, Eigen::Dynamic, 1>;
template <typename T> using DiagMatrix = Eigen::DiagonalMatrix<T, Eigen::Dynamic, Eigen::Dynamic>;

// sparse matrix structures
template <typename T> using SpMatrix = Eigen::SparseMatrix<T>;

namespace fdapde {

constexpr int Dynamic = -1;       // used when the size of a vector or matrix is not known at compile time
constexpr int random_seed = -1;   // signals that a random seed is used somewhere

template <int N, typename T = double> struct static_dynamic_vector_selector {
    using type = typename std::conditional<N == Dynamic, DVector<T>, SVector<N, T>>::type;
};
template <int N, typename T = double>
using static_dynamic_vector_selector_t = typename static_dynamic_vector_selector<N, T>::type;

template <int N, int M, typename T = double> struct static_dynamic_matrix_selector {
    using type = typename switch_type<
      switch_type_case<N == Dynamic && M == Dynamic, DMatrix<T>>,
      switch_type_case<N == Dynamic && M != Dynamic, Eigen::Matrix<T, N, Eigen::Dynamic>>,
      switch_type_case<N != Dynamic && M == Dynamic, Eigen::Matrix<T, Eigen::Dynamic, M>>,
      switch_type_case<N != Dynamic && M != Dynamic, SMatrix<N, M, T>>>::type;
};
template <int N, int M, typename T = double>
using static_dynamic_matrix_selector_t = typename static_dynamic_matrix_selector<N, M, T>::type;
  
// a Triplet type (almost identical with respect to Eigen::Triplet<T>) but allowing for non-const access to stored value
// this is compatible to Eigen::setFromTriplets() method used for the sparse matrix construction
template <typename T> class Triplet {
   private:
    Eigen::Index row_, col_;
    T value_;
   public:
    Triplet() = default;
    Triplet(const Eigen::Index& row, const Eigen::Index& col, const T& value) : row_(row), col_(col), value_(value) {};

    const Eigen::Index& row() const { return row_; }
    const Eigen::Index& col() const { return col_; }
    const T& value() const { return value_; }
    T& value() { return value_; }   // allow for modifications of stored value, this not allowed by Eigen::Triplet
};

// hash function for std::pair (allow pairs as key of unordered_map). inspired from boost::hash
struct pair_hash {
    template <typename T1, typename T2> std::size_t operator()(const std::pair<T1, T2>& pair) const {
        std::size_t hash = 0;
        hash ^= std::hash<T1>()(pair.first) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        hash ^= std::hash<T2>()(pair.second) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        return hash;
    }
};
// hash function for DMatrix<T>, allows DMatrix<T> as key in an unordered map.
struct matrix_hash {
    template <typename T> std::size_t operator()(const DMatrix<T>& matrix) const {
        std::size_t hash = 0;
        for (std::size_t i = 0; i < matrix.rows(); ++i) {
            for (std::size_t j = 0; j < matrix.cols(); ++j) {
                hash ^= std::hash<T>()(matrix(i, j)) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
            }
        }
        return hash;
    };
};
  
// hash function for an standard container's iterator range
template <typename T>
struct std_container_hash {
  std::size_t operator()(const typename T::const_iterator& begin, const typename T::const_iterator& end) const {
    std::size_t hash = 0;
    for(auto it = begin; it != end; ++it) {
      hash ^= std::hash<typename T::value_type>()(*it) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    }
    return hash;
  }
};
// hash function for std::array<T, N>
template <typename T, int N> struct std_array_hash {
    std::size_t operator()(const std::array<T, N>& array) const {
      return std_container_hash<std::array<T, N>>()(array.begin(), array.end());
    };
};
  
// oredering relation for SVector<N>, allows SVector<N> to be keys of std::map
template <int N> struct s_vector_compare {
    bool operator()(const SVector<N>& lhs, const SVector<N>& rhs) const {
        return std::lexicographical_compare(lhs.data(), lhs.data() + lhs.size(), rhs.data(), rhs.data() + rhs.size());
    };
};
// ordering relation for DVector<T>
template <typename T> struct d_vector_compare {
    bool operator()(const DVector<T>& lhs, const DVector<T>& rhs) const {
        return std::lexicographical_compare(lhs.data(), lhs.data() + lhs.size(), rhs.data(), rhs.data() + rhs.size());
    }
};

// a movable wrapper for Eigen::SparseLU (Eigen::SparseLU has a deleted copy and assignment operator)
template <typename T> class SparseLU {
   private:
    typedef Eigen::SparseLU<T, Eigen::COLAMDOrdering<int>> SparseLU_;
    std::shared_ptr<SparseLU_> solver_;   // wrap Eigen::SparseLU into a movable object
    bool computed_ = false;               // asserted true if factorization is successfully computed
   public:
    // default constructor
    SparseLU() = default;
    // we expose only the compute and solve methods of Eigen::SparseLU
    void compute(const T& matrix) {
        solver_ = std::make_shared<SparseLU_>();
        solver_->compute(matrix);
	if (solver_->info() == Eigen::Success) { computed_ = true; }  
    }

    template <typename Rhs>   // solve method, dense rhs operand
    const Eigen::Solve<SparseLU_, Rhs> solve(const Eigen::MatrixBase<Rhs>& b) const {
        return solver_->solve(b);
    }
    template <typename Rhs>   // solve method, sparse rhs operand
    const Eigen::Solve<SparseLU_, Rhs> solve(const Eigen::SparseMatrixBase<Rhs>& b) const {
        return solver_->solve(b);
    }
    // direct access to Eigen::SparseLU
    std::shared_ptr<SparseLU_> operator->() { return solver_; }
    Eigen::ComputationInfo info() const { return solver_->info(); }
    operator bool() const { return computed_; }  
};

// test for floating point equality based on relative error.
constexpr double DOUBLE_TOLERANCE = 50 * std::numeric_limits<double>::epsilon();   // approx 10^-14
template <typename T>
typename std::enable_if<!std::numeric_limits<T>::is_integer, bool>::type almost_equal(T a, T b, T epsilon) {
    return std::fabs(a - b) < epsilon ||
           std::fabs(a - b) < ((std::fabs(a) < std::fabs(b) ? std::fabs(b) : std::fabs(a)) * epsilon);
}
// default to DOUBLE_TOLERANCE
template <typename T> typename std::enable_if<!std::numeric_limits<T>::is_integer, bool>::type almost_equal(T a, T b) {
    return almost_equal(a, b, DOUBLE_TOLERANCE);
}

// test if Eigen matrix is empty (a zero-sized matrix is considered empty)
template <typename Derived> bool is_empty(const Eigen::EigenBase<Derived>& matrix) { return matrix.size() == 0; }

// compute log(1 + exp(x)) in a numerical stable way (see Machler, M. (2012). Accurately computing log(1-exp(-|a|)))
constexpr double log1pexp(double x) {
    if (x <= -37.0) return std::exp(x);
    if (x <=  18.0) return std::log1p(std::exp(x));
    if (x >   33.3) return x;
    return x + std::exp(-x);
}

}   // namespace fdapde

#endif   // __FDAPDE_SYMBOLS_H__
