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

TEST(matrix_test, DiagonalMatrixView) {

    using Scalar = double;

    // data
    constexpr int n = 3;
    Scalar data[n] = {1, 2, 3};
    Scalar data_new[n] = {0};

    // Default constructor
    DiagonalMatrixView<Scalar, 3> m_raw(data);
    std::cout << "DiagonalMatrixView" << std::endl;
    std::cout << m_raw << std::endl;
    std::cout << std::endl;
    m_raw.setZero();
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            assert(m_raw(i,j) == Scalar(0));
    std::cout << "setZero" << std::endl;
    std::cout << m_raw << std::endl;
    std::cout << std::endl;

    // assignment from std::array
    std::array<Scalar, 3> arr = {1, 2, 3};
    DiagonalMatrixView<Scalar, 3> m_arr(arr);
    assert(m_arr(0,0) == Scalar(1));
    assert(m_arr(1,1) == Scalar(2));
    std::cout << "Assignment from std::array = {1, 2, 3}" << std::endl;
    std::cout << m_arr << std::endl;
    std::cout << std::endl;

    // assignment from std::array
    m_arr = arr;
    assert(m_arr(0,0) == Scalar(1));
    assert(m_arr(1,1) == Scalar(2));
    std::cout << "Assignment from std::array = {1, 2, 3}" << std::endl;
    std::cout << m_arr << std::endl;
    std::cout << std::endl;

    // Constructor from std::vector
    std::vector<Scalar> vec = {1, 2, 3};
    DiagonalMatrixView<Scalar, 3> m_vec(data_new);
    m_vec = vec;
    assert(m_vec(0,0) == Scalar(1));
    assert(m_vec(1,1) == Scalar(2));
    std::cout << "Constructor from std::vector = {1, 2, 3}" << std::endl;
    std::cout << m_vec << std::endl;
    std::cout << std::endl;

    // Callable constructor
    auto callable = []() {return std::array<Scalar, 3>{3, 2, 1};};
    DiagonalMatrixView<Scalar, 3> m_callable(data_new);
    m_callable = callable;
    assert(m_callable(0,0) == Scalar(3));
    assert(m_callable(1,1) == Scalar(2));
    std::cout << "Callable constructor std::array = {3, 2, 1}" << std::endl;
    std::cout << m_callable << std::endl;
    std::cout << std::endl;

    // Copy and assignment from another DiagonalMatrixView
    {
        DiagonalMatrixView<Scalar, 3> m_copy(data_new);
        m_copy = m_arr;
        assert(m_arr(0,0) == Scalar(1));
        assert(m_arr(1,1) == Scalar(2));
        std::cout << "Assign from another DiagonalMatrixView" << std::endl;
        std::cout << m_copy << std::endl;
        std::cout << std::endl;
    }

    // Copy and assignment from a MatrixBase expression (it takes the diagonal part)
    {
        Matrix<double, 3, 3> M({1, 2, 3, 4, 5, 6, 7, 8, 9});
        DiagonalMatrixView<Scalar, 3> m_assign(data);
        m_assign = M;
        assert(m_assign(1,1) == Scalar(5));
        std::cout << "Assign from another MatrixBase expression (it takes the diagonal part)" << std::endl;
        std::cout << m_assign << std::endl;
        std::cout << std::endl;
    }

    #ifdef __FDAPDE_HAS_EIGEN__
        // Eigen assignment
        Scalar raw_eigen[2] = {0};
        Eigen::Matrix<Scalar, 2, 2> emat;
        emat << 1, 2, 3, 4;
        DiagonalMatrixView<Scalar, 2> m_eigen_assign(raw_eigen);
        m_eigen_assign = emat;
        assert(m_eigen_assign(0,0) == Scalar(1));
        assert(m_eigen_assign(1,1) == Scalar(4));
        std::cout << "Eigen assignment from emat << 1, 2, 3, 4;" << std::endl;
        std::cout << m_eigen_assign << std::endl;
        std::cout << std::endl;
    #endif

}

TEST(matrix_test, DiagonalMatrix) {

    using Scalar = double;

    // Default constructor
    DiagonalMatrix<Scalar, 2> m_default;
    m_default.setZero();
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            assert(m_default(i,j) == Scalar(0));
    std::cout << "setZero" << std::endl;
    std::cout << m_default << std::endl;
    std::cout << std::endl;

    // Constant value constructor
    DiagonalMatrix<Scalar, 2> m_const(DiagonalMatrix<Scalar, 2>::Constant(Scalar(3)));
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            assert(m_const(i,j) == (i == j ? Scalar(3) : Scalar(0)));
    std::cout << "Constant(3)" << std::endl;
    std::cout << m_const << std::endl;
    std::cout << std::endl;

    // Ones
    DiagonalMatrix<Scalar, 2> m_ones(Matrix<Scalar, 2, 2>::Ones());
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            assert(m_ones(i,j) == (i == j ? Scalar(1) : Scalar(0)));
    std::cout << "Ones (diagonal from dense Ones())" << std::endl;
    std::cout << m_ones << std::endl;
    std::cout << std::endl;

    // Zero
    DiagonalMatrix<Scalar, 2> m_zero(DiagonalMatrix<Scalar, 2>::Zero());
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            assert(m_zero(i,j) == Scalar(0));
    std::cout << "Zero" << std::endl;
    std::cout << m_zero << std::endl;
    std::cout << std::endl;

    // NaN
    auto m_nan = DiagonalMatrix<Scalar, 2>::NaN();
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            (i == j) ? assert(std::isnan(m_nan(i,j))) : assert(m_nan(i,j) == Scalar(0));
    std::cout << "NaN (diagonal NaNs, off-diagonal zeros)" << std::endl;
    std::cout << m_nan << std::endl;
    std::cout << std::endl;

    // Constructor from std::array
    std::array<Scalar, 3> arr = {1, 2, 3};
    DiagonalMatrix<Scalar, 3> m_arr(arr);
    assert(m_arr(0,0) == Scalar(1));
    assert(m_arr(1,1) == Scalar(2));
    std::cout << "Constructor from std::array = {1, 2, 3}" << std::endl;
    std::cout << m_arr << std::endl;
    std::cout << std::endl;

    // Constructor from std::vector
    std::vector<Scalar> vec = {1, 2, 3};
    DiagonalMatrix<Scalar, 3> m_vec(vec);
    assert(m_vec(0,0) == Scalar(1));
    assert(m_vec(1,1) == Scalar(2));
    std::cout << "Constructor from std::vector = {1, 2, 3}" << std::endl;
    std::cout << m_vec << std::endl;
    std::cout << std::endl;

    // Callable constructor
    DiagonalMatrix<Scalar, 3> m_callable([]() {
        return std::array<Scalar, 3>{3, 2, 1};
    });
    assert(m_callable(0,0) == Scalar(3));
    assert(m_callable(1,1) == Scalar(2));
    std::cout << "Callable constructor std::array = {3, 2, 1}" << std::endl;
    std::cout << m_callable << std::endl;
    std::cout << std::endl;

    // Copy and assignment from another DiagonalMatrix
    {
        DiagonalMatrix<Scalar, 3> m_copy(m_arr);
        assert(m_arr(0,0) == Scalar(1));
        assert(m_arr(1,1) == Scalar(2));
        std::cout << "Copy from another DiagonalMatrix" << std::endl;
        std::cout << m_copy << std::endl;
        std::cout << std::endl;
        DiagonalMatrix<Scalar, 3> m_assign;
        m_assign = m_vec;
        assert(m_assign(0,0) == Scalar(1));
        assert(m_assign(1,1) == Scalar(2));
        std::cout << "Assign from another DiagonalMatrix" << std::endl;
        std::cout << m_assign << std::endl;
        std::cout << std::endl;
    }

    // Copy and assignment from a MatrixBase expression (it takes the diagonal part)
    {
        Matrix<double, 3, 3> M({1, 2, 3, 4, 5, 6, 7, 8, 9});
        DiagonalMatrix<Scalar, 3> m_copy(M);
        assert(m_copy(1,1) == Scalar(5));
        std::cout << "Copy from another MatrixBase expression (it takes the diagonal part)" << std::endl;
        std::cout << m_copy << std::endl;
        std::cout << std::endl;
        DiagonalMatrix<Scalar, 3> m_assign;
        m_assign = M;
        assert(m_assign(2,2) == Scalar(9));
        std::cout << "Assign from another MatrixBase expression (it takes the diagonal part)" << std::endl;
        std::cout << m_assign << std::endl;
        std::cout << std::endl;
    }

    #ifdef __FDAPDE_HAS_EIGEN__
        // Eigen constructor and assignment
        Eigen::Matrix<Scalar, 2, 2> emat;
        emat << 1, 2, 3, 4;
        DiagonalMatrix<Scalar, 2> m_eigen(emat);
        assert(m_eigen(1,1) == Scalar(4));
        std::cout << "Eigen constructor from emat << 1, 2, 3, 4;" << std::endl;
        std::cout << m_eigen << std::endl;
        std::cout << std::endl;
        DiagonalMatrix<Scalar, 2> m_eigen_assign;
        m_eigen_assign = emat;
        assert(m_eigen_assign(0,0) == Scalar(1));
        assert(m_eigen_assign(0,1) == Scalar(0));
        std::cout << "Eigen assignment from emat << 1, 2, 3, 4;" << std::endl;
        std::cout << m_eigen_assign << std::endl;
        std::cout << std::endl;
        // Conversion to Eigen
        Eigen::Matrix<Scalar, 2, 2> emat_conv = m_eigen.as_eigen();
        assert(emat_conv(0,1) == Scalar(0));
        assert(emat_conv(1,1) == Scalar(4));
        std::cout << "Conversion to Eigen" << std::endl;
        std::cout << emat_conv << std::endl;
        std::cout << std::endl;
    #endif
}


TEST(matrix_test, DiagonalMatrixAlgebra) {
    DiagonalMatrix<double, 3> D{{1,2,3}};
    std::cout << "D" << std::endl;
    std::cout << D << std::endl;
    std::cout << std::endl;

    std::cout << "2*D" << std::endl;
    std::cout << 2*D << std::endl;
    std::cout << std::endl;

    std::cout << "D + 2*D" << std::endl;
    std::cout << D + 2*D << std::endl;
    std::cout << std::endl;

    std::cout << "D + Matrix::Identity()" << std::endl;
    std::cout << D + Matrix<double, 3, 3>::Identity() << std::endl;
    std::cout << std::endl;

    std::cout << "D + DiagonalMatrix::Identity()" << std::endl;
    std::cout << D + DiagonalMatrix<double, 3>::Identity() << std::endl;
    std::cout << std::endl;

    std::cout << "as_matrix(D)" << std::endl;
    std::cout << D.as_matrix() << std::endl;
    std::cout << std::endl;

}