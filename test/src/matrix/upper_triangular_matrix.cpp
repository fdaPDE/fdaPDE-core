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

#include <fdaPDE/linear_algebra.h>

#include <gtest/gtest.h>   // testing framework
#include <fdaPDE/utility.h>

using namespace fdapde;

TEST(matrix_test, UpperTriangularMatrixView) {

    using Scalar = double;

    // data
    constexpr int n = 3;
    Scalar data[n * (n+1) / 2] = {1, 2, 3, 4, 5, 6};
    Scalar data_new[n * (n+1) / 2] = {0};

    // Default constructor
    TriangularMatrixView<Scalar, 3, Upper> m_raw(data);
    std::cout << "UpperTriangularMatrixView" << std::endl;
    std::cout << m_raw << std::endl;
    std::cout << std::endl;
    m_raw.setZero();
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            assert(m_raw(i,j) == Scalar((j >= i) ? 0 : 0)); // all zeros after setZero
    std::cout << "setZero" << std::endl;
    std::cout << m_raw << std::endl;
    std::cout << std::endl;

    // assignment from std::array
    std::array<Scalar, 6> arr = {1, 2, 3, 4, 5, 6};
    TriangularMatrixView<Scalar, 3, Upper> m_arr(arr);
    assert(m_arr(0,0) == Scalar(1));
    assert(m_arr(0,1) == Scalar(2));
    assert(m_arr(1,0) == Scalar(0));
    std::cout << "Assignment from std::array = {1, 2, 3, 4, 5, 6}" << std::endl;
    std::cout << m_arr << std::endl;
    std::cout << std::endl;

    // assignment from std::array
    m_arr = arr;
    assert(m_arr(0,0) == Scalar(1));
    assert(m_arr(0,1) == Scalar(2));
    std::cout << "Assignment from std::array = {1, 2, 3, 4, 5, 6}" << std::endl;
    std::cout << m_arr << std::endl;
    std::cout << std::endl;

    // Constructor from std::vector
    std::vector<Scalar> vec = {1, 2, 3, 4, 5, 6};
    TriangularMatrixView<Scalar, 3, Upper> m_vec(data_new);
    m_vec = vec;
    assert(m_vec(0,0) == Scalar(1));
    assert(m_vec(0,1) == Scalar(2));
    assert(m_vec(1,0) == Scalar(0));
    std::cout << "Constructor from std::vector = {1, 2, 3, 4, 5, 6}" << std::endl;
    std::cout << m_vec << std::endl;
    std::cout << std::endl;

    // Callable constructor
    auto callable = []() {return std::array<Scalar, 6>{6, 5, 4, 3, 2, 1};};
    TriangularMatrixView<Scalar, 3, Upper> m_callable(data_new);
    m_callable = callable;
    assert(m_callable(0,0) == Scalar(6));
    assert(m_callable(1,1) == Scalar(3));
    std::cout << "Callable constructor std::array = {6, 5, 4, 3, 2, 1}" << std::endl;
    std::cout << m_callable << std::endl;
    std::cout << std::endl;

    // Copy and assignment from another UpperTriangularMatrixView
    {
        TriangularMatrixView<Scalar, 3, Upper> m_copy(data_new);
        m_copy = m_arr;
        assert(m_arr(0,0) == Scalar(1));
        assert(m_arr(0,1) == Scalar(2));
        std::cout << "Assign from another UpperTriangularMatrixView" << std::endl;
        std::cout << m_copy << std::endl;
        std::cout << std::endl;
    }

    // Copy and assignment from a MatrixBase expression (it takes the upper triangular part)
    {
        Matrix<double, 3, 3> M({1, 2, 3, 4, 5, 6, 7, 8, 9});
        TriangularMatrixView<Scalar, 3, Upper> m_assign(data);
        m_assign = M;
        assert(m_assign(1,2) == Scalar(6));
        assert(m_assign(2,1) == Scalar(0));
        std::cout << "Assign from another MatrixBase expression (it takes the upper triangular part)" << std::endl;
        std::cout << m_assign << std::endl;
        std::cout << std::endl;
    }

    #ifdef __FDAPDE_HAS_EIGEN__
        // Eigen assignment
        Scalar raw_eigen[3] = {0};
        Eigen::Matrix<Scalar, 2, 2> emat;
        emat << 1, 2, 3, 4;
        TriangularMatrixView<Scalar, 2, Upper> m_eigen_assign(raw_eigen);
        m_eigen_assign = emat;
        assert(m_eigen_assign(0,1) == Scalar(2));
        assert(m_eigen_assign(1,0) == Scalar(0));
        std::cout << "Eigen assignment from emat << 1, 2, 3, 4;" << std::endl;
        std::cout << m_eigen_assign << std::endl;
        std::cout << std::endl;
    #endif

}

TEST(matrix_test, UpperTriangularMatrix) {

    using Scalar = double;

    // Default constructor
    TriangularMatrix<Scalar, 2, Upper> m_default;
    m_default.setZero();
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            assert(m_default(i,j) == Scalar(0));
    std::cout << "setZero" << std::endl;
    std::cout << m_default << std::endl;
    std::cout << std::endl;

    // Constant value constructor
    TriangularMatrix<Scalar, 2, Upper> m_const(TriangularMatrix<Scalar, 2, Upper>::Constant(Scalar(3)));
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            assert(m_const(i,j) == (j >= i ? Scalar(3) : Scalar(0)));
    std::cout << "Constant(3)" << std::endl;
    std::cout << m_const << std::endl;
    std::cout << std::endl;

    // Ones (take upper part of dense Ones)
    TriangularMatrix<Scalar, 2, Upper> m_ones(Matrix<Scalar, 2, 2>::Ones());
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            assert(m_ones(i,j) == (j >= i ? Scalar(1) : Scalar(0)));
    std::cout << "Ones (upper part)" << std::endl;
    std::cout << m_ones << std::endl;
    std::cout << std::endl;

    // Zero
    TriangularMatrix<Scalar, 2, Upper> m_zero(TriangularMatrix<Scalar, 2, Upper>::Zero());
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            assert(m_zero(i,j) == Scalar(0));
    std::cout << "Zero" << std::endl;
    std::cout << m_zero << std::endl;
    std::cout << std::endl;

    // NaN
    auto m_nan = TriangularMatrix<Scalar, 2, Upper>::NaN();
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            (j >= i) ? assert(std::isnan(m_nan(i,j))) : assert(m_nan(i,j) == Scalar(0));
    std::cout << "NaN (upper NaNs, below diagonal zeros)" << std::endl;
    std::cout << m_nan << std::endl;
    std::cout << std::endl;

    // Constructor from std::array
    std::array<Scalar, 6> arr = {1, 2, 3, 4, 5, 6};
    TriangularMatrix<Scalar, 3, Upper> m_arr(arr);
    assert(m_arr(0,0) == Scalar(1));
    assert(m_arr(0,1) == Scalar(2));
    assert(m_arr(1,0) == Scalar(0));
    std::cout << "Constructor from std::array = {1, 2, 3, 4, 5, 6}" << std::endl;
    std::cout << m_arr << std::endl;
    std::cout << std::endl;

    // Constructor from std::vector
    std::vector<Scalar> vec = {1, 2, 3, 4, 5, 6};
    TriangularMatrix<Scalar, 3, Upper> m_vec(vec);
    assert(m_vec(0,0) == Scalar(1));
    assert(m_vec(0,1) == Scalar(2));
    std::cout << "Constructor from std::vector = {1, 2, 3, 4, 5, 6}" << std::endl;
    std::cout << m_vec << std::endl;
    std::cout << std::endl;

    // Callable constructor
    TriangularMatrix<Scalar, 3, Upper> m_callable([]() {
        return std::array<Scalar, 6>{6, 5, 4, 3, 2, 1};
    });
    assert(m_callable(0,0) == Scalar(6));
    assert(m_callable(1,1) == Scalar(3));
    std::cout << "Callable constructor std::array = {6, 5, 4, 3, 2, 1}" << std::endl;
    std::cout << m_callable << std::endl;
    std::cout << std::endl;

    // Copy and assignment from another UpperTriangularMatrix
    {
        TriangularMatrix<Scalar, 3, Upper> m_copy(m_arr);
        assert(m_arr(0,0) == Scalar(1));
        assert(m_arr(0,1) == Scalar(2));
        std::cout << "Copy from another UpperTriangularMatrix" << std::endl;
        std::cout << m_copy << std::endl;
        std::cout << std::endl;
        TriangularMatrix<Scalar, 3, Upper> m_assign;
        m_assign = m_vec;
        assert(m_assign(0,0) == Scalar(1));
        assert(m_assign(0,1) == Scalar(2));
        std::cout << "Assign from another UpperTriangularMatrix" << std::endl;
        std::cout << m_assign << std::endl;
        std::cout << std::endl;
    }

    // Copy and assignment from a MatrixBase expression (it takes the upper triangular part)
    {
        Matrix<double, 3, 3> M({1, 2, 3, 4, 5, 6, 7, 8, 9});
        TriangularMatrix<Scalar, 3, Upper> m_copy(M);
        assert(m_copy(2,1) == Scalar(0));
        assert(m_copy(1,2) == Scalar(6));
        std::cout << "Copy from another MatrixBase expression (upper triangular part)" << std::endl;
        std::cout << m_copy << std::endl;
        std::cout << std::endl;
        TriangularMatrix<Scalar, 3, Upper> m_assign;
        m_assign = M;
        assert(m_assign(0,2) == Scalar(3));
        std::cout << "Assign from another MatrixBase expression (upper triangular part)" << std::endl;
        std::cout << m_assign << std::endl;
        std::cout << std::endl;
    }

    #ifdef __FDAPDE_HAS_EIGEN__
        // Eigen constructor and assignment
        Eigen::Matrix<Scalar, 2, 2> emat;
        emat << 1, 2, 3, 4;
        TriangularMatrix<Scalar, 2, Upper> m_eigen(emat);
        assert(m_eigen(0,1) == Scalar(2));
        assert(m_eigen(1,0) == Scalar(0));
        std::cout << "Eigen constructor from emat << 1, 2, 3, 4;" << std::endl;
        std::cout << m_eigen << std::endl;
        std::cout << std::endl;
        TriangularMatrix<Scalar, 2, Upper> m_eigen_assign;
        m_eigen_assign = emat;
        assert(m_eigen_assign(0,1) == Scalar(2));
        std::cout << "Eigen assignment from emat << 1, 2, 3, 4;" << std::endl;
        std::cout << m_eigen_assign << std::endl;
        std::cout << std::endl;
        // Conversion to Eigen
        Eigen::Matrix<Scalar, 2, 2> emat_conv = m_eigen.as_eigen();
        assert(emat_conv(1,0) == Scalar(0));
        std::cout << "Conversion to Eigen" << std::endl;
        std::cout << emat_conv << std::endl;
        std::cout << std::endl;
    #endif
}


TEST(matrix_test, UpperTriangularMatrixAlgebra) {
    TriangularMatrix<double, 3, Upper> U{{1,2,3,4,5,6}};
    std::cout << "U" << std::endl;
    std::cout << U << std::endl;
    std::cout << std::endl;

    std::cout << "2*U" << std::endl;
    std::cout << 2*U << std::endl;
    std::cout << std::endl;

    std::cout << "U + 2*U" << std::endl;
    std::cout << U + 2*U << std::endl;
    std::cout << std::endl;

    std::cout << "U + Matrix::Identity()" << std::endl;
    std::cout << U + Matrix<double, 3, 3>::Identity() << std::endl;
    std::cout << std::endl;

    std::cout << "U + TriangularMatrix::Identity()" << std::endl;
    std::cout << U + TriangularMatrix<double, 3, Upper>::Identity() << std::endl;
    std::cout << std::endl;

    std::cout << "as_matrix(U)" << std::endl;
    std::cout << U.as_matrix() << std::endl;
    std::cout << std::endl;

}