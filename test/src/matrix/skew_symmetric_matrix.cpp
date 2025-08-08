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

TEST(matrix_test, SkewSymmetricMatrixView) {

    using Scalar = double;

    // data
    constexpr int n = 3;
    Scalar data[n * (n-1) / 2] = {1, 2, 3};
    Scalar data_new[n * (n-1) / 2] = {0};

    // Default constructor
    SkewSymmetricMatrixView<Scalar, 3> M_raw(data);
    std::cout << "SkewSymmetricMatrixView" << std::endl;
    std::cout << M_raw << std::endl;
    std::cout << std::endl;
    M_raw.setZero();
    for (int i = 0; i < n; ++i)
        for (int j = i; j < n; ++j)
            assert(M_raw(i,j) == Scalar(0));
    std::cout << "setZero" << std::endl;
    std::cout << M_raw << std::endl;
    std::cout << std::endl;

    // constructor from std::array
    std::array<Scalar, 3> arr = {1, 2, 3};
    SkewSymmetricMatrixView<Scalar, 3> M_arr(arr);
    assert(M_arr(0,0) == Scalar(0));
    assert(M_arr(0,1) == Scalar(1));
    assert(M_arr(1,0) == Scalar(-1));
    std::cout << "Constructor from std::array = {1, 2, 3}" << std::endl;
    std::cout << M_arr << std::endl;
    std::cout << std::endl;

    // assignment from std::array
    M_arr = arr;
    assert(M_arr(0,0) == Scalar(0));
    assert(M_arr(0,1) == Scalar(1));
    assert(M_arr(1,0) == Scalar(-1));
    std::cout << "Assignment from std::array = {1, 2, 3}" << std::endl;
    std::cout << M_arr << std::endl;
    std::cout << std::endl;

    // Constructor from std::vector
    std::vector<Scalar> vec = {1, 2, 3};
    SkewSymmetricMatrixView<Scalar, 3> M_vec(data_new);
    M_vec = vec;
    assert(M_vec(0,0) == Scalar(0));
    assert(M_vec(0,1) == Scalar(1));
    assert(M_vec(1,0) == Scalar(-1));
    std::cout << "Constructor from std::vector = {1, 2, 3}" << std::endl;
    std::cout << M_vec << std::endl;
    std::cout << std::endl;

    // Callable constructor
    auto callable = []() {return std::array<Scalar, 3>{3, 2, 1};};
    SkewSymmetricMatrixView<Scalar, 3> M_callable(data_new);
    M_callable = callable;
    assert(M_callable(0,0) == Scalar(0));
    assert(M_callable(0,1) == Scalar(3));
    assert(M_callable(1,0) == Scalar(-3));
    std::cout << "Callable constructor std::array = {3, 2, 1}" << std::endl;
    std::cout << M_callable << std::endl;
    std::cout << std::endl;

    // Copy and assignment from another SkewSymmetricMatrix
    {
        SkewSymmetricMatrixView<Scalar, 3> M_copy(data_new);
        M_copy = M_arr;
        assert(M_arr(0,0) == Scalar(0));
        assert(M_arr(0,1) == Scalar(1));
        assert(M_arr(1,0) == Scalar(-1));
        std::cout << "Assign from another SkewSymmetricMatrixView" << std::endl;
        std::cout << M_copy << std::endl;
        std::cout << std::endl;
    }

    // Copy and assignment from a MatrixBase expression (it takes the symmetric part)
    {
        Matrix<double, 3, 3> M({1, 2, 3, 4, 5, 6, 7, 8, 9});
        SkewSymmetricMatrixView<Scalar, 3> M_assign(data);
        M_assign = M;
        assert(M_assign(0,0) == Scalar(0));
        assert(M_assign(0,1) == Scalar(-1));
        assert(M_assign(1,0) == Scalar(1));
        std::cout << "Assign from another MatrixBase expression (it takes the symmetric part)" << std::endl;
        std::cout << M_assign << std::endl;
        std::cout << std::endl;
    }

    #ifdef __FDAPDE_HAS_EIGEN__
        // Eigen assignment
        Scalar raw_eigen[3] = {0};
        Eigen::Matrix<Scalar, 3, 3> emat;
        emat << 1, 2, 3, 4, 5, 6, 7, 8, 9;
        SkewSymmetricMatrixView<Scalar, 3> M_eigen_assign(raw_eigen);
        M_eigen_assign = emat;
        assert(M_eigen_assign(0,1) == Scalar(-1));
        std::cout << "Eigen assignment from emat << 1, 2, 3, 4, 5, 6" << std::endl;
        std::cout << M_eigen_assign << std::endl;
        std::cout << std::endl;
    #endif

}

TEST(matrix_test, SkewSymmetricMatrix) {

    using Scalar = double;

    // Default constructor
    SkewSymmetricMatrix<Scalar, 3> M_default;
    M_default.setZero();
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            assert(M_default(i,j) == Scalar(0));
    std::cout << "setZero" << std::endl;
    std::cout << M_default << std::endl;
    std::cout << std::endl;

    // Constant value constructor
    auto M_const = SkewSymmetricMatrix<Scalar, 3>::Constant(Scalar(3));
    for (int i = 0; i < 3; ++i){
        assert(M_const(i,i) == Scalar(0));
        for (int j = i+1; j < 3; ++j) {
            assert(M_const(i,j) == Scalar(3));
            assert(M_const(j,i) == Scalar(-3));
        }
    }
    std::cout << "Constant(3)" << std::endl;
    std::cout << M_const << std::endl;
    std::cout << std::endl;

    // Ones
    auto  M_ones = SkewSymmetricMatrix<Scalar, 3>::Ones();
    for (int i = 0; i < 3; ++i) {
        assert(M_const(i,i) == Scalar(0));
        for (int j = i+1; j < 3; ++j){
            assert(M_ones(i,j) == Scalar(1));
            assert(M_ones(j,i) == Scalar(-1));
        }
    }
    std::cout << "Ones" << std::endl;
    std::cout << M_ones << std::endl;
    std::cout << std::endl;

    // Zero
    auto M_zero = SkewSymmetricMatrix<Scalar, 3>::Zero();
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            assert(M_zero(i,j) == Scalar(0));
    std::cout << "Zero" << std::endl;
    std::cout << M_zero << std::endl;
    std::cout << std::endl;

    // NaN
    auto M_nan = SkewSymmetricMatrix<Scalar, 3>::NaN();
    for (int i = 0; i < 3; ++i){
        assert(M_nan(i,i) == Scalar(0));
        for (int j = i + 1; j < 3; ++j) {
            assert(std::isnan(M_nan(i,j)));
            assert(std::isnan(M_nan(j,i)));
        }
    }
    std::cout << "NaN" << std::endl;
    std::cout << M_nan << std::endl;
    std::cout << std::endl;

    // Constructor from std::array
    std::array<Scalar, 3> arr = {1, 2, 3};
    SkewSymmetricMatrix<Scalar, 3> M_arr(arr);
    assert(M_arr(0,0) == Scalar(0));
    assert(M_arr(0,1) == Scalar(1));
    assert(M_arr(1,0) == Scalar(-1));
    std::cout << "Constructor from std::array = {1, 2, 3}" << std::endl;
    std::cout << M_arr << std::endl;
    std::cout << std::endl;

    // Constructor from std::vector
    std::vector<Scalar> vec = {1, 2, 3};
    SkewSymmetricMatrix<Scalar, 3> M_vec(vec);
    assert(M_vec(0,0) == Scalar(0));
    assert(M_vec(0,1) == Scalar(1));
    assert(M_vec(1,0) == Scalar(-1));
    std::cout << "Constructor from std::vector = {1, 2, 3}" << std::endl;
    std::cout << M_vec << std::endl;
    std::cout << std::endl;

    // Callable constructor
    SkewSymmetricMatrix<Scalar, 3> M_callable([]() {
        return std::array<Scalar, 3>{3, 2, 1};
    });
    assert(M_callable(0,0) == Scalar(0));
    assert(M_callable(0,1) == Scalar(3));
    assert(M_callable(1,0) == Scalar(-3));
    std::cout << "Callable constructor std::array = {3, 2, 1}" << std::endl;
    std::cout << M_callable << std::endl;
    std::cout << std::endl;

    // Copy and assignment from another SkewSymmetricMatrix
    {
        SkewSymmetricMatrix<Scalar, 3> M_copy(M_arr);
        assert(M_arr(0,0) == Scalar(0));
        assert(M_arr(0,1) == Scalar(1));
        assert(M_arr(1,0) == Scalar(-1));
        std::cout << "Copy from another SkewSymmetricMatrix" << std::endl;
        std::cout << M_copy << std::endl;
        std::cout << std::endl;
        SkewSymmetricMatrix<Scalar, 3> M_assign;
        M_assign = M_vec;
        assert(M_assign(0,0) == Scalar(0));
        assert(M_assign(0,1) == Scalar(1));
        assert(M_assign(1,0) == Scalar(-1));
        std::cout << "Assign from another SkewSymmetricMatrix" << std::endl;
        std::cout << M_assign << std::endl;
        std::cout << std::endl;
    }

    // Copy and assignment from a MatrixBase expression (it takes the skew-symmetric part)
    {
        Matrix<double, 3, 3> M({1, 2, 3, 4, 5, 6, 7, 8, 9});
        SkewSymmetricMatrix<Scalar, 3> M_copy(M);
        assert(M_copy(1,0) == Scalar(1));
        std::cout << "Copy from another MatrixBase expression (it takes the skew-symmetric part)" << std::endl;
        std::cout << M_copy << std::endl;
        std::cout << std::endl;
        SkewSymmetricMatrix<Scalar, 3> M_assign;
        M_assign = M;
        assert(M_assign(1,0) == Scalar(1));
        std::cout << "Assign from another MatrixBase expression (it takes the skew-symmetric part)" << std::endl;
        std::cout << M_assign << std::endl;
        std::cout << std::endl;
    }

    #ifdef __FDAPDE_HAS_EIGEN__
        // Eigen constructor and assignment
        Eigen::Matrix<Scalar, 3, 3> emat;
        emat << 1, 2, 3, 4, 5, 6, 7, 8, 9;
        SkewSymmetricMatrix<Scalar, 3> M_eigen(emat);
        assert(M_eigen(1,0) == Scalar(1));
        std::cout << "Eigen constructor from emat << 1, 2, 3, 4, 5, 6, 7, 8, 9" << std::endl;
        std::cout << M_eigen << std::endl;
        std::cout << std::endl;
        SkewSymmetricMatrix<Scalar, 3> M_eigen_assign;
        M_eigen_assign = emat;
        assert(M_eigen_assign(1,0) == Scalar(1));
        std::cout << "Eigen assignment from emat << 1, 2, 3, 4, 5, 6, 7, 8, 9" << std::endl;
        std::cout << M_eigen_assign << std::endl;
        std::cout << std::endl;
        // Conversion to Eigen
        Eigen::Matrix<Scalar, 3, 3> emat_conv = M_eigen.as_eigen();
        assert(emat_conv(1,0) == Scalar(1));
        std::cout << "Conversion to Eigen" << std::endl;
        std::cout << emat_conv << std::endl;
        std::cout << std::endl;
    #endif

}

TEST(matrix_test, SkewSymmetricMatrixAlgebra) {
    SkewSymmetricMatrix<double, 3> S{{1,2,3}};
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

    std::cout << "as_matrix(S)" << std::endl;
    std::cout << S.as_matrix() << std::endl;
    std::cout << std::endl;

}
