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

TEST(matrix_test, SymmetricMatrixView) {

    using Scalar = double;

    // data
    constexpr int n = 3;
    Scalar data[n * (n+1) / 2] = {1, 2, 3, 4, 5, 6};
    Scalar data_new[n * (n+1) / 2] = {0};

    // Default constructor
    SymmetricMatrixView<Scalar, 3> m_raw(data);
    std::cout << "SymmetricMatrixView" << std::endl;
    std::cout << m_raw << std::endl;
    std::cout << std::endl;
    m_raw.setZero();
    for (int i = 0; i < n; ++i)
        for (int j = i; j < n; ++j)
            assert(m_raw(i,j) == Scalar(0));
    std::cout << "setZero" << std::endl;
    std::cout << m_raw << std::endl;
    std::cout << std::endl;

    // assignment from std::array
    std::array<Scalar, 6> arr = {1, 2, 3, 4, 5, 6};
    SymmetricMatrixView<Scalar, 3> m_arr(arr);
    assert(m_arr(0,0) == Scalar(1));
    assert(m_arr(0,1) == Scalar(2));
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
    SymmetricMatrixView<Scalar, 3> m_vec(data_new);
    m_vec = vec;
    assert(m_vec(0,0) == Scalar(1));
    assert(m_vec(0,1) == Scalar(2));
    std::cout << "Constructor from std::vector = {1, 2, 3, 4, 5, 6}" << std::endl;
    std::cout << m_vec << std::endl;
    std::cout << std::endl;

    // Callable constructor
    auto callable = []() {return std::array<Scalar, 6>{6, 5, 4, 3, 2, 1};};
    SymmetricMatrixView<Scalar, 3> m_callable(data_new);
    m_callable = callable;
    assert(m_callable(0,0) == Scalar(6));
    assert(m_callable(1,1) == Scalar(3));
    std::cout << "Callable constructor std::array = {6, 5, 4, 3, 2, 1}" << std::endl;
    std::cout << m_callable << std::endl;
    std::cout << std::endl;

    // Copy and assignment from another SymmetricMatrix
    {
        SymmetricMatrixView<Scalar, 3> m_copy(data_new);
        m_copy = m_arr;
        assert(m_arr(0,0) == Scalar(1));
        assert(m_arr(0,1) == Scalar(2));
        std::cout << "Assign from another SymmetricMatrixView" << std::endl;
        std::cout << m_copy << std::endl;
        std::cout << std::endl;
    }

    // Copy and assignment from a MatrixBase expression (it takes the symmetric part)
    {
        Matrix<double, 3, 3> M({1, 2, 3, 4, 5, 6, 7, 8, 9});
        SymmetricMatrix<Scalar, 3> m_assign(data);
        m_assign = M;
        assert(m_assign(1,0) == Scalar(3));
        std::cout << "Assign from another MatrixBase expression (it takes the symmetric part)" << std::endl;
        std::cout << m_assign << std::endl;
        std::cout << std::endl;
    }

    #ifdef __FDAPDE_HAS_EIGEN__
        // Eigen assignment
        Scalar raw_eigen[3] = {0};
        Eigen::Matrix<Scalar, 2, 2> emat;
        emat << 1, 2, 3, 4;
        SymmetricMatrixView<Scalar, 2> m_eigen_assign(raw_eigen);
        m_eigen_assign = emat;
        assert(m_eigen_assign(0,1) == Scalar(2.5));
        std::cout << "Eigen assignment from emat << 1, 2, 3, 4;" << std::endl;
        std::cout << m_eigen_assign << std::endl;
        std::cout << std::endl;
    #endif

}

TEST(matrix_test, SymmetricMatrix) {

    using Scalar = double;

    // Default constructor
    SymmetricMatrix<Scalar, 2> m_default;
    m_default.setZero();
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            assert(m_default(i,j) == Scalar(0));
    std::cout << "setZero" << std::endl;
    std::cout << m_default << std::endl;
    std::cout << std::endl;

    // Constant value constructor
    SymmetricMatrix<Scalar, 2> m_const(SymmetricMatrix<Scalar, 2>::Constant(Scalar(3)));
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            assert(m_const(i,j) == Scalar(3));
    std::cout << "Constant(3)" << std::endl;
    std::cout << m_const << std::endl;
    std::cout << std::endl;

    // Ones
    SymmetricMatrix<Scalar, 2> m_ones(Matrix<Scalar, 2, 2>::Ones());
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            assert(m_ones(i,j) == Scalar(1));
    std::cout << "Ones" << std::endl;
    std::cout << m_ones << std::endl;
    std::cout << std::endl;

    // Zero
    SymmetricMatrix<Scalar, 2> m_zero(SymmetricMatrix<Scalar, 2>::Zero());
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            assert(m_zero(i,j) == Scalar(0));
    std::cout << "Zero" << std::endl;
    std::cout << m_zero << std::endl;
    std::cout << std::endl;

    // NaN
    auto m_nan = SymmetricMatrix<Scalar, 2>::NaN();
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            assert(std::isnan(m_nan(i,j)));
    std::cout << "NaN" << std::endl;
    std::cout << m_nan << std::endl;
    std::cout << std::endl;

    // Constructor from std::array
    std::array<Scalar, 6> arr = {1, 2, 3, 4, 5, 6};
    SymmetricMatrix<Scalar, 3> m_arr(arr);
    assert(m_arr(0,0) == Scalar(1));
    assert(m_arr(0,1) == Scalar(2));
    std::cout << "Constructor from std::array = {1, 2, 3, 4, 5, 6}" << std::endl;
    std::cout << m_arr << std::endl;
    std::cout << std::endl;

    // Constructor from std::vector
    std::vector<Scalar> vec = {1, 2, 3, 4, 5, 6};
    SymmetricMatrix<Scalar, 3> m_vec(vec);
    assert(m_vec(0,0) == Scalar(1));
    assert(m_vec(0,1) == Scalar(2));
    std::cout << "Constructor from std::vector = {1, 2, 3, 4, 5, 6}" << std::endl;
    std::cout << m_vec << std::endl;
    std::cout << std::endl;

    // Callable constructor
    SymmetricMatrix<Scalar, 3> m_callable([]() {
        return std::array<Scalar, 6>{6, 5, 4, 3, 2, 1};
    });
    assert(m_callable(0,0) == Scalar(6));
    assert(m_callable(1,1) == Scalar(3));
    std::cout << "Callable constructor std::array = {6, 5, 4, 3, 2, 1}" << std::endl;
    std::cout << m_callable << std::endl;
    std::cout << std::endl;

    // Copy and assignment from another SymmetricMatrix
    {
        SymmetricMatrix<Scalar, 3> m_copy(m_arr);
        assert(m_arr(0,0) == Scalar(1));
        assert(m_arr(0,1) == Scalar(2));
        std::cout << "Copy from another SymmetricMatrix" << std::endl;
        std::cout << m_copy << std::endl;
        std::cout << std::endl;
        SymmetricMatrix<Scalar, 3> m_assign;
        m_assign = m_vec;
        assert(m_assign(0,0) == Scalar(1));
        assert(m_assign(0,1) == Scalar(2));
        std::cout << "Assign from another SymmetricMatrix" << std::endl;
        std::cout << m_assign << std::endl;
        std::cout << std::endl;
    }

    // Copy and assignment from a MatrixBase expression (it takes the symmetric part)
    {
        Matrix<double, 3, 3> M({1, 2, 3, 4, 5, 6, 7, 8, 9});
        SymmetricMatrix<Scalar, 3> m_copy(M);
        assert(m_copy(1,0) == Scalar(3));
        std::cout << "Copy from another MatrixBase expression (it takes the symmetric part)" << std::endl;
        std::cout << m_copy << std::endl;
        std::cout << std::endl;
        SymmetricMatrix<Scalar, 3> m_assign;
        m_assign = M;
        assert(m_assign(1,0) == Scalar(3));
        std::cout << "Assign from another MatrixBase expression (it takes the symmetric part)" << std::endl;
        std::cout << m_assign << std::endl;
        std::cout << std::endl;
    }

    #ifdef __FDAPDE_HAS_EIGEN__
        // Eigen constructor and assignment
        Eigen::Matrix<Scalar, 2, 2> emat;
        emat << 1, 2, 3, 4;
        SymmetricMatrix<Scalar, 2> m_eigen(emat);
        assert(m_eigen(1,0) == Scalar(2.5));
        std::cout << "Eigen constructor from emat << 1, 2, 3, 4;" << std::endl;
        std::cout << m_eigen << std::endl;
        std::cout << std::endl;
        SymmetricMatrix<Scalar, 2> m_eigen_assign;
        m_eigen_assign = emat;
        assert(m_eigen_assign(0,1) == Scalar(2.5));
        std::cout << "Eigen assignment from emat << 1, 2, 3, 4;" << std::endl;
        std::cout << m_eigen_assign << std::endl;
        std::cout << std::endl;
        // Conversion to Eigen
        Eigen::Matrix<Scalar, 2, 2> emat_conv = m_eigen.as_eigen();
        assert(emat_conv(0,1) == Scalar(2.5));
        std::cout << "Conversion to Eigen" << std::endl;
        std::cout << emat_conv << std::endl;
        std::cout << std::endl;
    #endif
}


TEST(matrix_test, SymmetricMatrixAlgebra) {
    SymmetricMatrix<double, 3> S{{1,2,3,4,5,6}};
    std::cout << "S" << std::endl;
    std::cout << S << std::endl;
    std::cout << std::endl;

    std::cout << "2*S" << std::endl;
    std::cout << 2*S << std::endl;
    std::cout << std::endl;

    std::cout << "S + 2*S" << std::endl;
    std::cout << S + 2*S << std::endl;
    std::cout << std::endl;

    std::cout << "S + Matrix::Identity()" << std::endl;
    std::cout << S + Matrix<double, 3, 3>::Identity() << std::endl;
    std::cout << std::endl;

    std::cout << "S + SymmetricMatrix::Identity()" << std::endl;
    std::cout << S + SymmetricMatrix<double, 3>::Identity() << std::endl;
    std::cout << std::endl;

    std::cout << "Full(S)" << std::endl;
    std::cout << S.full() << std::endl;
    std::cout << std::endl;

}