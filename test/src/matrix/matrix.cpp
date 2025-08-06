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

TEST(matrix_test, MatrixView_Initialization) {

    // types and dimensions
    using Scalar = double;
    constexpr int rows = 2, cols = 3;

    // from C-style array
    Scalar data_raw[rows * cols] = {1, 2, 3, 4, 5, 6};
    MatrixView<Scalar, rows, cols, RowMajor> M_raw(data_raw);
    std::cout << "MatrixView at C-style array" << std::endl;
    std::cout << M_raw << std::endl;
    std::cout << std::endl;

    // from std::array
    std::array<Scalar, rows*cols>data_arr{{6, 5, 4, 3, 2, 1}};
    MatrixView<Scalar, rows, cols, RowMajor> M_arr(data_arr);
    std::cout << "MatrixView at std::array" << std::endl;
    std::cout << M_arr << std::endl;
    std::cout << std::endl;
}


TEST(matrix_test, MatrixView_StorageOrder) {

    // types and dimensions
    using Scalar = double;
    constexpr int rows = 2, cols = 3;

    // ColMajor MatrixView
    Scalar data_col[rows * cols] = {1, 2, 3, 4, 5, 6};
    MatrixView<Scalar, rows, cols, ColMajor> M_col(data_col);
    assert(M_col(0, 0) == 1);
    assert(M_col(1, 0) == 2);
    assert(M_col(0, 1) == 3);
    assert(M_col(1, 1) == 4);
    assert(M_col(0, 2) == 5);
    assert(M_col(1, 2) == 6);
    std::cout << "ColMajor MatrixView" << std::endl;
    std::cout << M_col << std::endl;
    std::cout << std::endl;

    // non-const access operator (ColMajor)
    M_col(1, 0) = 42;
    assert(data_col[1] == 42);
    std::cout << "non-const access operator (ColMajor)" << std::endl;
    std::cout << M_col << std::endl;
    std::cout << std::endl;

    // RowMajor MatrixView
    Scalar data_row[rows * cols] = {1, 2, 3, 4, 5, 6};
    MatrixView<Scalar, rows, cols, RowMajor> M_row(data_row);
    assert(M_row(0, 0) == 1);
    assert(M_row(0, 1) == 2);
    assert(M_row(0, 2) == 3);
    assert(M_row(1, 0) == 4);
    assert(M_row(1, 1) == 5);
    assert(M_row(1, 2) == 6);
    std::cout << "RowMajor MatrixView" << std::endl;
    std::cout << M_row << std::endl;
    std::cout << std::endl;

    // non-const access operator (ColMajor)
    M_row(1, 0) = 42;
    assert(data_row[3] == 42);
    std::cout << "non-const access operator (RowMajor)" << std::endl;
    std::cout << M_row << std::endl;
    std::cout << std::endl;
}

TEST(matrix_test, MatrixView_VectorAccess) {

    // data
    using Scalar = int;
    constexpr int len = 4;
    Scalar data[len] = {10, 20, 30, 40};

    MatrixView<Scalar, 1, len> row_vec(data);
    std::cout << "row-VectorView" << std::endl;
    std::cout << row_vec << std::endl;
    std::cout << std::endl;
    for (int i = 0; i < len; ++i) {
        assert(row_vec[i] == data[i]);
        row_vec[i] += 1;
        assert(row_vec[i] == data[i]);
    }
    std::cout << "row-VectorView" << std::endl;
    std::cout << row_vec << std::endl;
    std::cout << std::endl;

    VectorView<Scalar, len> col_vec(data);
    std::cout << "col-VectorView" << std::endl;
    std::cout << col_vec << std::endl;
    std::cout << std::endl;
    for (int i = 0; i < len; ++i) {
        assert(col_vec[i] == data[i]);
        col_vec[i] += 1;
        assert(col_vec[i] == data[i]);
    }
    std::cout << "col-VectorView" << std::endl;
    std::cout << col_vec << std::endl;
    std::cout << std::endl;
}

TEST(matrix_test, MatrixView_Assignment) {

    // data
    using Scalar = float;
    constexpr int rows = 2, cols = 3;
    Scalar data[rows * cols] = {0};

    // Matrix and MatrixView initialization
    Matrix<Scalar, rows, cols> M;
    M.setOnes();
    std::cout << "Matrix" << std::endl;
    std::cout << M << std::endl;
    std::cout << std::endl;
    std::cout << "Initialization of the MatrixView" << std::endl;
    MatrixView<Scalar, rows, cols> mat_view(data);
    std::cout << mat_view << std::endl;
    std::cout << std::endl;

    // assignment
    mat_view = M;
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            assert(mat_view(i, j) == M(i, j));
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            assert(data[i*cols + j] == M(i, j));
    std::cout << "Assignment" << std::endl;
    std::cout << mat_view << std::endl;
    std::cout << std::endl;
    std::cout << "Check that data has changed" << std::endl;
    std::cout << data[0] << " ..." << std::endl;
    std::cout << std::endl;
}

TEST(matrix_test, Matrix_Initialization) {
    using Scalar = double;

    // Default constructor
    Matrix<Scalar, 2, 2> M_default;
    M_default.setZero();
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            assert(M_default(i,j) == Scalar(0));
    std::cout << "setZero" << std::endl;
    std::cout << M_default << std::endl;
    std::cout << std::endl;

    // Constant value constructor
    Matrix<Scalar, 2, 2> M_const = Matrix<Scalar, 2, 2>::Constant(Scalar(3));
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            assert(M_const(i,j) == Scalar(3));
    std::cout << "Constant(3)" << std::endl;
    std::cout << M_const << std::endl;
    std::cout << std::endl;

    // Ones
    Matrix<Scalar, 2, 2> M_ones(Matrix<Scalar, 2, 2>::Ones());
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            assert(M_ones(i,j) == Scalar(1));
    std::cout << "Ones" << std::endl;
    std::cout << M_ones << std::endl;
    std::cout << std::endl;

    // Zero
    Matrix<Scalar, 2, 2> M_zero(Matrix<Scalar, 2, 2>::Zero());
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            assert(M_zero(i,j) == Scalar(0));
    std::cout << "Zero" << std::endl;
    std::cout << M_zero << std::endl;
    std::cout << std::endl;

    // NaN
    auto M_nan = Matrix<Scalar, 2, 2>::NaN();
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            assert(std::isnan(M_nan(i,j)));
    std::cout << "NaN" << std::endl;
    std::cout << M_nan << std::endl;
    std::cout << std::endl;

    // Constructor from std::array
    std::array<Scalar, 4> arr = {1, 2, 3, 4};
    Matrix<Scalar, 2, 2> M_arr(arr);
    assert(M_arr(0,0) == Scalar(1));
    assert(M_arr(1,1) == Scalar(4));
    std::cout << "Constructor from std::array = {1,2,3,4}" << std::endl;
    std::cout << M_arr << std::endl;
    std::cout << std::endl;

    // Constructor from std::vector
    std::vector<Scalar> vec = {5, 6, 7, 8};
    Matrix<Scalar, 2, 2> M_vec(vec);
    assert(M_vec(0,0) == Scalar(5));
    assert(M_vec(1,1) == Scalar(8));
    std::cout << "Constructor from std::vector = {5,6,7,8}" << std::endl;
    std::cout << M_vec << std::endl;
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
    Matrix<Scalar, 2, 2> M_callable([]() {
        return std::array<Scalar, 4>{9, 8, 7, 6};
    });
    assert(M_callable(0,0) == Scalar(9));
    assert(M_callable(1,1) == Scalar(6));
    std::cout << "Callable constructor std::array = {9, 8, 7, 6}" << std::endl;
    std::cout << M_callable << std::endl;
    std::cout << std::endl;

    // Copy constructor from another MatrixBase expression
    Matrix<Scalar, 2, 2> M({1,2,3,4});
    Matrix<Scalar, 2, 2> M_copy = M;
    assert(M_copy(1,0) == Scalar(3));
    std::cout << "Copy constructor from another MatrixBase expression" << std::endl;
    std::cout << M_copy << std::endl;
    std::cout << std::endl;

    #ifdef __FDAPDE_HAS_EIGEN__
        // Eigen constructor (ColMajor)
        Eigen::Matrix<Scalar, 2, 2, Eigen::ColMajor> emat_col;
        emat_col << 1, 2, 3, 4;
        Eigen::Matrix<Scalar, 2, 2, Eigen::RowMajor> emat_row;
        emat_row << 1, 2, 3, 4;
        {
            Matrix<Scalar, 2,2> M_eigen(emat_col);
            assert(M_eigen(1,0) == Scalar(3));
            std::cout << "emat_col << 1, 2, 3, 4;" << std::endl;
            std::cout << emat_col << std::endl;
            std::cout << "Eigen constructor from (ColMajor) emat_col" << std::endl;
            std::cout << M_eigen << std::endl;
            std::cout << std::endl;
        }
        {
            Matrix<Scalar, 2,2>  M_eigen(emat_row);
            assert(M_eigen(1,0) == Scalar(3));
            std::cout << "emat_row << 1, 2, 3, 4;" << std::endl;
            std::cout << emat_row << std::endl;
            std::cout << "Eigen constructor from (RowMajor) emat_row" << std::endl;
            std::cout << M_eigen << std::endl;
            std::cout << std::endl;
        }
    #endif
}

TEST(matrix_test, Matrix_Assignment) {

    using Scalar = double;

    // Assignment from another MatrixBase expression
    Matrix<Scalar, 2, 2> M({1,2,3,4});
    Matrix<Scalar, 2, 2> M_assign;
    M_assign = M;
    assert(M_assign(0,1) == Scalar(2));
    std::cout << "Assignment from another MatrixBase expression" << std::endl;
    std::cout << M_assign << std::endl;
    std::cout << std::endl;

    // Assignment from std::array
    std::array<Scalar, 4> arr = {1, 2, 3, 4};
    M_assign = arr;
    assert(M_assign(1,0) == Scalar(3));
    std::cout << "Assignment from std::array" << std::endl;
    std::cout << M_assign << std::endl;
    std::cout << std::endl;

    #ifdef __FDAPDE_HAS_EIGEN__
        // Eigen constructor (ColMajor)
        Eigen::Matrix<Scalar, 2, 2, Eigen::ColMajor> emat_col;
        emat_col << 1, 2, 3, 4;
        Eigen::Matrix<Scalar, 2, 2, Eigen::RowMajor> emat_row;
        emat_row << 1, 2, 3, 4;
        {
            Matrix<Scalar, 2,2> M_eigen;
            M_eigen = emat_col;
            assert(M_eigen(1,0) == Scalar(3));
            std::cout << "emat_col << 1, 2, 3, 4;" << std::endl;
            std::cout << emat_col << std::endl;
            std::cout << "Eigen constructor from (ColMajor) emat_col" << std::endl;
            std::cout << M_eigen << std::endl;
            std::cout << std::endl;
        }
        {
            Matrix<Scalar, 2,2>  M_eigen;
            M_eigen = emat_row;
            assert(M_eigen(1,0) == Scalar(3));
            std::cout << "emat_row << 1, 2, 3, 4;" << std::endl;
            std::cout << emat_row << std::endl;
            std::cout << "Eigen constructor from (RowMajor) emat_row" << std::endl;
            std::cout << M_eigen << std::endl;
            std::cout << std::endl;
        }
    #endif
}

TEST(matrix_test, Matrix_VectorAccess) {

    using Scalar = double;

    // Operator[] for vector-shaped matrices
    auto v1 = Vector<Scalar, 2>(Scalar(10), Scalar(20));
    assert(v1[0] == Scalar(10));
    assert(v1[1] == Scalar(20));
    v1[0] = Scalar(30);
    assert(v1[0] == Scalar(30));
    std::cout << "Operator[] for vector-shaped matrices" << std::endl;
    std::cout << v1 << std::endl;
    std::cout << std::endl;
}

TEST(matrix_test, Matrix_Conversion) {

    using Scalar = double;

    #ifdef __FDAPDE_HAS_EIGEN__
        // Conversion to Eigen
        Matrix<Scalar, 2, 2, ColMajor> M_col({1,2,3,4});
        auto emat_conv_col = M_col.as_eigen_map();
        assert(!decltype(emat_conv_col)::IsRowMajor);
        std::cout << "M_col" << std::endl;
        std::cout << M_col << std::endl;
        std::cout << "Conversion to Eigen (of the ColMajor matrix) M_col" << std::endl;
        std::cout << emat_conv_col << std::endl;
        std::cout << std::endl;
        // Conversion to Eigen
        Matrix<Scalar, 2, 2, RowMajor> M_row({1,2,3,4});
        auto emat_conv_row = M_row.as_eigen_map();
        assert(decltype(emat_conv_row)::IsRowMajor);
        std::cout << "M_row" << std::endl;
        std::cout << M_row << std::endl;
        std::cout << "Conversion to Eigen (of the RowMajor matrix) M_row" << std::endl;
        std::cout << emat_conv_row << std::endl;
        std::cout << std::endl;
    #endif
}