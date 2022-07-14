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

#ifndef __UTILS_H__
#define __UTILS_H__

#include <utility.h>
#include <unsupported/Eigen/SparseExtra>
#include <string>
#include "constants.h"
//using fdapde::core::CSVReader;

// a set of usefull utilities
namespace fdapde {
namespace isotesting {

  template <typename T> using DMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
  template <typename T> using SpMatrix = Eigen::SparseMatrix<T>;

  // this function is an implementation of the test for floating point equality based on relative error. There is
  // an huge literature about floating point comparison, refer to it for details
template <typename T>
typename std::enable_if<!std::numeric_limits<T>::is_integer, bool>::type almost_equal(T a, T b, T epsilon) {
    return std::fabs(a - b) < epsilon ||
           std::fabs(a - b) < ((std::fabs(a) < std::fabs(b) ? std::fabs(b) : std::fabs(a)) * epsilon);
}

// set default epsilon to DOUBLE_TOLERANCE
template <typename T>
typename std::enable_if<std::is_floating_point<T>::value, bool>::type
almost_equal(T a, T b) {
    return almost_equal(a, b, DOUBLE_TOLERANCE);
}

template <typename DerivedA, typename DerivedB>
inline bool almost_equal(const Eigen::MatrixBase<DerivedA>& op1,
                         const Eigen::MatrixBase<DerivedB>& op2,
                         double epsilon) {
    return (op1 - op2).template lpNorm<Eigen::Infinity>() < epsilon ||
           (op1 - op2).template lpNorm<Eigen::Infinity>() <
           (std::max(op1.template lpNorm<Eigen::Infinity>(), op2.template lpNorm<Eigen::Infinity>()) * epsilon);
}

template <typename DerivedA, typename DerivedB>
inline bool almost_equal(const Eigen::MatrixBase<DerivedA>& op1,
                         const Eigen::MatrixBase<DerivedB>& op2) {
    return almost_equal(op1, op2, DOUBLE_TOLERANCE);
}

// sparse operands
inline bool almost_equal(const SpMatrix<double>& op1, const SpMatrix<double>& op2, double epsilon) {
    const Eigen::MatrixXd dense1 = op1.toDense();
    const Eigen::MatrixXd dense2 = op2.toDense();
    return almost_equal(dense1, dense2, epsilon);
}
inline bool almost_equal(const SpMatrix<double>& op1, const SpMatrix<double>& op2) {
    const Eigen::MatrixXd dense1 = op1.toDense();
    const Eigen::MatrixXd dense2 = op2.toDense();
    return almost_equal(dense1, dense2);
}

/*
  // load rhs from file
  bool almost_equal(const SpMatrix<double>& op1, std::string op2) {
    SpMatrix<double> mem_buff;
    Eigen::loadMarket(mem_buff, op2);
    return almost_equal(op1, mem_buff);
  }
  bool almost_equal(const DMatrix<double>& op1, std::string op2) {
    SpMatrix<double> mem_buff;
    Eigen::loadMarket(mem_buff, op2);
    return almost_equal(op1, DMatrix<double>(mem_buff));
  }

  bool almost_equal(const std::vector<double>& op1, std::string op2) {
    DMatrix<double> m;
    m.resize(op1.size(), 1);
    for (std::size_t i = 0; i < op1.size(); ++i) m(i, 0) = op1[i];
    return almost_equal(m, op2);
  }

  template <int N> bool almost_equal(const std::array<double,N>& op1, const std::array<double,N>& op2) {
    bool equal = true;
    for (int i = 0; i < N; ++i) equal &= almost_equal(op1[i], op2[i]);
    return equal;
  }

  // utility to import .mtx files
  template <typename T>
  DMatrix<T> read_mtx(const std::string& file_name) {
    SpMatrix<double> buff;
    Eigen::loadMarket(buff, file_name);
    return buff;
  }
  // utility to import .csv files
  
  template <typename T>
  DMatrix<T> read_csv(const std::string& file_name) {
    CSVReader<T> reader {};
    return reader.template parse_file<Eigen::Dense>(file_name);
  }
  */ 
  
}}

#endif // __UTILS_H__
