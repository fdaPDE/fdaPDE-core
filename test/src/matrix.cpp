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

/*
#include <Eigen/Eigen>
#define __FDAPDE_HAS_EIGEN__
*/

#include <fdaPDE/linear_algebra.h>

#include <gtest/gtest.h>   // testing framework
#include <fdaPDE/utility.h>

using namespace fdapde;

TEST(matrix_test, Matrix) {

    using Scalar = double;

    // Default constructor
    Matrix<Scalar, 2, 2> m_default;
    m_default.setZero();
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            assert(m_default(i,j) == Scalar(0));
    std::cout << "setZero" << std::endl;
    std::cout << m_default << std::endl;
    std::cout << std::endl;

    // Constant value constructor
    Matrix<Scalar, 2, 2> m_const = Matrix<Scalar, 2, 2>::Constant(Scalar(3));
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            assert(m_const(i,j) == Scalar(3));
    std::cout << "Constant(3)" << std::endl;
    std::cout << m_const << std::endl;
    std::cout << std::endl;

    // Ones
    Matrix<Scalar, 2, 2> m_ones = Matrix<Scalar, 2, 2>::Ones();
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            assert(m_ones(i,j) == Scalar(1));
    std::cout << "Ones" << std::endl;
    std::cout << m_ones << std::endl;
    std::cout << std::endl;

    // Zero
    Matrix<Scalar, 2, 2> m_zero = Matrix<Scalar, 2, 2>::Zero();
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            assert(m_zero(i,j) == Scalar(0));
    std::cout << "Zero" << std::endl;
    std::cout << m_zero << std::endl;
    std::cout << std::endl;

    // NaN
    auto m_nan = Matrix<Scalar, 2, 2>::NaN();
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            assert(std::isnan(m_nan(i,j)));
    std::cout << "NaN" << std::endl;
    std::cout << m_nan << std::endl;
    std::cout << std::endl;

    // Constructor from std::array
    std::array<Scalar, 4> arr = {1, 2, 3, 4};
    Matrix<Scalar, 2, 2> m_arr(arr);
    assert(m_arr(0,0) == Scalar(1));
    assert(m_arr(1,1) == Scalar(4));
    std::cout << "Constructor from std::array = {1,2,3,4}" << std::endl;
    std::cout << m_arr << std::endl;
    std::cout << std::endl;

    // Constructor from std::vector
    std::vector<Scalar> vec = {5, 6, 7, 8};
    Matrix<Scalar, 2, 2> m_vec(vec);
    assert(m_vec(0,0) == Scalar(5));
    assert(m_vec(1,1) == Scalar(8));
    std::cout << "Constructor from std::vector = {5,6,7,8}" << std::endl;
    std::cout << m_vec << std::endl;
    std::cout << std::endl;

    // Scalar constructors for 1x1, 1x2, 1x3
    Vector<Scalar, 1> m1(Scalar(1));
    assert(m1[0] == Scalar(1));
    std::cout << "Scalar constructors for x1 = 1" << std::endl;
    std::cout << m1 << std::endl;
    std::cout << std::endl;
    Vector<Scalar, 2> m2(Scalar(1), Scalar(2));
    assert(m2[0] == Scalar(1));
    assert(m2[1] == Scalar(2));
    std::cout << "Scalar constructors for x1 = 1, x2 = 2" << std::endl;
    std::cout << m2 << std::endl;
    std::cout << std::endl;
    Vector<Scalar, 3> m3(Scalar(1), Scalar(2), Scalar(3));
    assert(m3[0] == Scalar(1));
    assert(m3[1] == Scalar(2));
    assert(m3[2] == Scalar(3));
    std::cout << "Scalar constructors for x1 = 1, x2 = 2, x3 = 3" << std::endl;
    std::cout << m3 << std::endl;
    std::cout << std::endl;

    // Callable constructor
    Matrix<Scalar, 2, 2> m_callable([]() {
        return std::array<Scalar, 4>{9, 8, 7, 6};
    });
    assert(m_callable(0,0) == Scalar(9));
    assert(m_callable(1,1) == Scalar(6));
    std::cout << "Callable constructor std::array = {9, 8, 7, 6}" << std::endl;
    std::cout << m_callable << std::endl;
    std::cout << std::endl;

    // Copy and assignment from another MatrixBase expression
    Matrix<Scalar, 2, 2> m_copy = m_arr;
    assert(m_copy(1,0) == Scalar(3));
    std::cout << "Copy from another MatrixBase expression" << std::endl;
    std::cout << m_copy << std::endl;
    std::cout << std::endl;
    Matrix<Scalar, 2, 2> m_assign;
    m_assign = m_vec;
    assert(m_assign(0,1) == Scalar(6));
    std::cout << "Assign from another MatrixBase expression" << std::endl;
    std::cout << m_assign << std::endl;
    std::cout << std::endl;

    // Assignment from std::array
    m_assign = arr;
    assert(m_assign(1,0) == Scalar(3));
    std::cout << "Assign from std::array" << std::endl;
    std::cout << m_assign << std::endl;
    std::cout << std::endl;

    // Operator[] for vector-shaped matrices
    auto v1 = Vector<Scalar, 2>(Scalar(10), Scalar(20));
    assert(v1[0] == Scalar(10));
    assert(v1[1] == Scalar(20));
    v1[0] = Scalar(30);
    assert(v1[0] == Scalar(30));
    std::cout << "Operator[] for vector-shaped matrices" << std::endl;
    std::cout << v1 << std::endl;
    std::cout << std::endl;

    // PermutationMatrix
    std::array<int, 2> perm = {1, 0};
    PermutationMatrix<2> P(perm);
    auto v =  Vector<Scalar, 2>(Scalar(5), Scalar(7));
    auto Pv = P * v;
    assert(Pv[0] == Scalar(7) && Pv[1] == Scalar(5));
    std::cout << "Permutation matrix" << std::endl;
    // std::cout << P << std::endl;
    // std::cout << "--" << std::endl;
    std::cout << v << std::endl;
    std::cout << "--" << std::endl;
    std::cout << Pv << std::endl;
    std::cout << std::endl;

    // Test LU factorization with solve
    Matrix<Scalar, 2, 2> A({2, 1, 4, 3});
    Vector<Scalar, 2> b(Scalar(5), Scalar(11));
    PartialPivLU<Matrix<Scalar, 2, 2>> lu(A);
    auto x = lu.solve(b);
    assert(std::abs(A(0,0)*x[0] + A(0,1)*x[1] - b[0]) < 1e-10);
    assert(std::abs(A(1,0)*x[0] + A(1,1)*x[1] - b[1]) < 1e-10);
    std::cout << "LU factorization with solve" << std::endl;
    std::cout << std::endl;


    #ifdef __FDAPDE_HAS_EIGEN__
        // Eigen constructor and assignment
        Eigen::Matrix<Scalar, 2, 2> emat;
        emat << 1, 2, 3, 4;
        Matrix<Scalar, 2, 2> m_eigen(emat);
        assert(m_eigen(1,0) == Scalar(3));
        std::cout << "Eigne constructor from emat << 1, 2, 3, 4;" << std::endl;
        std::cout << m_eigen << std::endl;
        std::cout << std::endl;

        Matrix<Scalar, 2, 2> m_eigen_assign;
        m_eigen_assign = emat;
        assert(m_eigen_assign(0,1) == Scalar(2));
        std::cout << "Eigne assignement from emat << 1, 2, 3, 4;" << std::endl;
        std::cout << m_eigen_assign << std::endl;
        std::cout << std::endl;
    #endif

}

TEST(matrix_test, MatrixBase) {

    Matrix<double, 3, 3> M({1,2,3,4,5,6,7,8,9});
    std::cout << "M" << std::endl;
    std::cout << M << std::endl;
    std::cout << std::endl;

    std::cout << "(Lower) Triangular view" << std::endl;
    std::cout << M.triangular_view<Lower>() << std::endl;
    std::cout << std::endl;
    std::cout << "(Upper) Triangular view" << std::endl;
    std::cout << M.triangular_view<Upper>() << std::endl;
    std::cout << std::endl;
    std::cout << "Diagonal view" << std::endl;
    std::cout << M.diagonal() << std::endl;
    std::cout << std::endl;
    std::cout << "Transpose" << std::endl;
    std::cout << M.transpose() << std::endl;
    std::cout << std::endl;

    Matrix<double, 3, 3> N({4,3,2,1,0,-1,-2,-3,-4});
    std::cout << "N" << std::endl;
    std::cout << N << std::endl;
    std::cout << std::endl;

    std::cout << "M+N" << std::endl;
    std::cout << M+N << std::endl;
    std::cout << std::endl;

    std::cout << "2M+N" << std::endl;
    std::cout << 2*M+N << std::endl;
    std::cout << std::endl;

    std::cout << "(2M+2N)/2" << std::endl;
    std::cout << (2*M+2*N)/2 << std::endl;
    std::cout << std::endl;

    std::cout << "Trace(N) = ";
    std::cout << N.trace() << std::endl;
    std::cout << "Sum(N) = ";
    std::cout << N.sum() << std::endl;
    std::cout << "Norm(N) = ";
    std::cout << N.norm() << std::endl;
    std::cout << "SqNorm(N) = ";
    std::cout << N.squared_norm() << std::endl;
    std::cout << "InfNorm(N) = ";
    std::cout << N.inf_norm() << std::endl;
    std::cout << "Min(N) = ";
    std::cout << N.min() << std::endl;
    std::cout << "Max(N) = ";
    std::cout << N.max() << std::endl;
    std::cout << "Mean(N) = ";
    std::cout << M.mean() << std::endl;
    std::cout << "Prod(M) = ";
    std::cout << M.prod() << std::endl;
    std::cout << std::endl;

}
