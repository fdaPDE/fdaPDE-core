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

TEST(matrix_test, MatrixView_Basic) {
    using Scalar = double;
    constexpr int rows = 2, cols = 3;

    // raw data
    Scalar raw[rows * cols] = {1, 2, 3, 4, 5, 6};

    // ColMajor MatrixView
    MatrixView<Scalar, rows, cols, ColMajor> map_col(raw);
    assert(map_col(0, 0) == 1);
    assert(map_col(1, 0) == 2);
    assert(map_col(0, 1) == 3);
    assert(map_col(1, 1) == 4);
    assert(map_col(0, 2) == 5);
    assert(map_col(1, 2) == 6);
    std::cout << "ColMajor MatrixView" << std::endl;
    std::cout << map_col << std::endl;
    std::cout << std::endl;

    // RowMajor MatrixView
    Scalar raw_row_major[rows * cols] = {1, 2, 3, 4, 5, 6};
    MatrixView<Scalar, rows, cols, RowMajor> map_row(raw_row_major);
    assert(map_row(0, 0) == 1);
    assert(map_row(0, 1) == 2);
    assert(map_row(0, 2) == 3);
    assert(map_row(1, 0) == 4);
    assert(map_row(1, 1) == 5);
    assert(map_row(1, 2) == 6);
    std::cout << "RowMajor MatrixView" << std::endl;
    std::cout << map_row << std::endl;
    std::cout << std::endl;

    // modify values
    map_col(0, 0) = 42;
    assert(raw[0] == 42);
    std::cout << "non-const access operator" << std::endl;
    std::cout << map_col << std::endl;
    std::cout << std::endl;
}

TEST(matrix_test, MatrixView_Assignement) {
    using Scalar = float;
    constexpr int rows = 2, cols = 3;

    Scalar raw[rows * cols] = {0};

    Matrix<Scalar, rows, cols> mat_expr(Matrix<Scalar, rows, cols>::Ones());
    std::cout << "Raw Matrix" << std::endl;
    std::cout << mat_expr << std::endl;
    std::cout << std::endl;
    std::cout << "Raw MatrixView" << std::endl;
    MatrixView<Scalar, rows, cols> MatrixView(raw);
    std::cout << MatrixView << std::endl;
    std::cout << std::endl;

    MatrixView = mat_expr;
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            assert(MatrixView(i, j) == mat_expr(i, j));
    std::cout << "Assignment" << std::endl;
    std::cout << MatrixView << std::endl;
    std::cout << std::endl;
}

TEST(matrix_test, MatrixView_VectorAccess) {
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


TEST(matrix_test, Matrix_Base) {

    Matrix<double, 3, 4> M({1,2,3,4, 5,6,7,8, 9,10,11,12});
    std::cout << "M" << std::endl;
    std::cout << M << std::endl;
    std::cout << std::endl;

    std::cout << "MatrixBlockView" << std::endl;
    std::cout << MatrixBlockView<3, 3, Matrix<double, 3, 4>>(M, 0, 0) << std::endl;
    std::cout << std::endl;

    Matrix<double, 3, 3> MSQ(M.block<3, 3>(0, 0));
    std::cout << "MatrixBlock" << std::endl;
    std::cout << MSQ << std::endl;
    std::cout << std::endl;

    std::cout << "(Lower) TriangularView" << std::endl;
    std::cout << MSQ.triangular_view<Lower>() << std::endl;
    std::cout << std::endl;
    std::cout << "(Upper) TriangularView" << std::endl;
    std::cout << MSQ.triangular_view<Upper>() << std::endl;
    std::cout << std::endl;
    std::cout << "DiagonalView" << std::endl;
    std::cout << MSQ.diagonal() << std::endl;
    std::cout << std::endl;
    std::cout << "Transpose" << std::endl;
    std::cout << M.transpose() << std::endl;
    std::cout << std::endl;

    Matrix<double, 3, 3> N({4,3,2,1,0,-1,-2,-3,-4});
    std::cout << "N" << std::endl;
    std::cout << N << std::endl;
    std::cout << std::endl;

    std::cout << "MSQ+N" << std::endl;
    std::cout << MSQ+N << std::endl;
    std::cout << std::endl;

    std::cout << "2MSQ+N" << std::endl;
    std::cout << 2*MSQ+N << std::endl;
    std::cout << std::endl;

    std::cout << "(2MSQ+2N)/2" << std::endl;
    std::cout << (2*MSQ+2*N)/2 << std::endl;
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

TEST(matrix_test, MatrixAndSquareMatrix) {

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
    Matrix<Scalar, 2, 2> m_ones(Matrix<Scalar, 2, 2>::Ones());
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            assert(m_ones(i,j) == Scalar(1));
    std::cout << "Ones" << std::endl;
    std::cout << m_ones << std::endl;
    std::cout << std::endl;

    // Zero
    Matrix<Scalar, 2, 2> m_zero(Matrix<Scalar, 2, 2>::Zero());
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

    // PermutationMatrix (applied to a vector)
    std::array<int, 2> perm = {1, 0};
    PermutationMatrix<2> P(perm);
    auto v =  Vector<Scalar, 2>(Scalar(5), Scalar(7));
    auto Pv = P * v;
    assert(Pv[0] == Scalar(7) && Pv[1] == Scalar(5));
    std::cout << "Permutation matrix (applied to a vector)" << std::endl;
    std::cout << P << std::endl;
    std::cout << "--" << std::endl;
    std::cout << v << std::endl;
    std::cout << "--" << std::endl;
    std::cout << Pv << std::endl;
    std::cout << std::endl;

    // PermutationMatrix (applied to a matrix)
    auto Pm = P * m_arr;
    assert(Pm(0,0) == Scalar(3) && Pm(1,0) == Scalar(1));
    std::cout << "Permutation matrix (applied to a matrix)" << std::endl;
    std::cout << P << std::endl;
    std::cout << "--" << std::endl;
    std::cout << m_arr << std::endl;
    std::cout << "--" << std::endl;
    std::cout << Pm << std::endl;
    std::cout << std::endl;

    // Test LU factorization with solve
    Matrix<Scalar, 2, 2> A({2, 1, 4, 3});
    Vector<Scalar, 2> b(Scalar(5), Scalar(11));
    auto lu = PartialPivLU<Matrix<double,2,2>>(A);
    auto x  = lu.solve(b);
    assert(std::abs(A(0,0)*x[0] + A(0,1)*x[1] - b[0]) < 1e-10);
    assert(std::abs(A(1,0)*x[0] + A(1,1)*x[1] - b[1]) < 1e-10);
    std::cout << "LU factorization with solve" << std::endl;
    std::cout << "A: " << A << std::endl;
    std::cout << "--" << std::endl;
    std::cout << "x: " << x << std::endl;
    std::cout << "--" << std::endl;
    std::cout << "Ax: " << A*x << std::endl;
    std::cout << "--" << std::endl;
    std::cout << "b: " << b << std::endl;
    std::cout << std::endl;

    #ifdef __FDAPDE_HAS_EIGEN__
        // Eigen constructor
        Eigen::Matrix<Scalar, 2, 2> emat;
        emat << 1, 2, 3, 4;
        Matrix<Scalar, 2, 2> m_eigen(emat);
        assert(m_eigen(1,0) == Scalar(3));
        std::cout << "Eigen constructor from emat << 1, 2, 3, 4;" << std::endl;
        std::cout << m_eigen << std::endl;
        std::cout << std::endl;
        // Eigen assignment
        Matrix<Scalar, 2, 2> m_eigen_assign;
        m_eigen_assign = emat;
        assert(m_eigen_assign(0,1) == Scalar(2));
        std::cout << "Eigen assignement from emat << 1, 2, 3, 4;" << std::endl;
        std::cout << m_eigen_assign << std::endl;
        std::cout << std::endl;
        // Conversion to Eigen
        Eigen::Matrix<Scalar, 2, 2> emat_conv = m_eigen.as_eigen_map();
        std::cout << "Conversion to Eigen" << std::endl;
        std::cout << emat_conv << std::endl;
        std::cout << std::endl;
    #endif

}


TEST(matrix_test, IdentityMatrix) {
    // The identity matrix is only available for square matrices
    // std::cout << "Identity Matrix" << std::endl;
    // std::cout << Matrix<double, 3, 4>::Identity() << std::endl;
    // std::cout << std::endl;
    std::cout << "Identity Matrix (Called from Matrix)" << std::endl;
    std::cout << Matrix<double, 3, 3>::Identity() << std::endl;
    std::cout << std::endl;
    std::cout << "Identity Matrix (Called from PermutationMatrix)" << std::endl;
    std::cout << PermutationMatrix<3>::Identity() << std::endl;
    std::cout << std::endl;
    std::cout << "Identity Matrix (Called from SymmetricMatrix)" << std::endl;
    std::cout << SymmetricMatrix<double, 3>::Identity() << std::endl;
    std::cout << std::endl;
    // But not for the skew-symmetric ones
    // std::cout << "Identity Matrix (Called from SkewSymmetricMatrix)" << std::endl;
    // std::cout << SkewSymmetricMatrix<double, 3>::Identity() << std::endl;
    // std::cout << std::endl;

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
        assert(m_arr(0,0) == Scalar(1));
        assert(m_arr(0,1) == Scalar(2));
        std::cout << "Assign from another SymmetricMatrix" << std::endl;
        std::cout << m_assign << std::endl;
        std::cout << std::endl;
    }

    // Copy and assignment from a MatrixBase expression (it takes the symmetric part)
    {
        Matrix<double, 3, 3> M({1, 2, 3, 4, 5, 6, 7, 8, 9});
        SymmetricMatrix<Scalar, 3> m_copy(M);
        assert(m_copy(1,0) == Scalar(3));
        std::cout << "Copy from another MatrixBase expression" << std::endl;
        std::cout << m_copy << std::endl;
        std::cout << std::endl;
        SymmetricMatrix<Scalar, 3> m_assign;
        m_assign = M;
        assert(m_assign(1,0) == Scalar(3));
        std::cout << "Assign from another MatrixBase expression" << std::endl;
        std::cout << m_assign << std::endl;
        std::cout << std::endl;
    }

    // Test LU factorization with solve for SymmetricMatrix
    SymmetricMatrix<Scalar, 2> A({1, 2, 2});
    Vector<Scalar, 2> b(Scalar(5), Scalar(11));
    PartialPivLU<Matrix<Scalar, 2, 2>> lu(A); // This MUST be a Matrix<Scalar, 2, 2>, it can not be SymmetricMatrix<Scalar, 2>
    auto x = lu.solve(b);
    assert(std::abs(A(0,0)*x[0] + A(0,1)*x[1] - b[0]) < 1e-10);
    assert(std::abs(A(1,0)*x[0] + A(1,1)*x[1] - b[1]) < 1e-10);
    std::cout << "LU factorization with solve" << std::endl;
    std::cout << "A: " << A << std::endl;
    std::cout << "--" << std::endl;
    std::cout << "x: " << x << std::endl;
    std::cout << "--" << std::endl;
    std::cout << "Ax: " << A*x << std::endl;
    std::cout << "--" << std::endl;
    std::cout << "b: " << b << std::endl;
    std::cout << std::endl;

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
        std::cout << "Eigen assignement from emat << 1, 2, 3, 4;" << std::endl;
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

TEST(matrix_test, SkewSymmetricMatrix) {

    using Scalar = double;

    // Default constructor
    SkewSymmetricMatrix<Scalar, 3> m_default;
    m_default.setZero();
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            assert(m_default(i,j) == Scalar(0));
    std::cout << "setZero" << std::endl;
    std::cout << m_default << std::endl;
    std::cout << std::endl;

    // Constant value constructor
    SkewSymmetricMatrix<Scalar, 3> m_const(SkewSymmetricMatrix<Scalar, 3>::Constant(Scalar(3)));
    for (int i = 0; i < 3; ++i)
        for (int j = i+1; j < 3; ++j)
            assert(m_const(i,j) == Scalar(3));
    std::cout << "Constant(3)" << std::endl;
    std::cout << m_const << std::endl;
    std::cout << std::endl;

    // Ones
    SkewSymmetricMatrix<Scalar, 3> m_ones(SkewSymmetricMatrix<Scalar, 3>::Ones());
    for (int i = 0; i < 3; ++i)
        for (int j = i+1; j < 3; ++j)
            assert(m_ones(i,j) == Scalar(1));
    std::cout << "Ones" << std::endl;
    std::cout << m_ones << std::endl;
    std::cout << std::endl;

    // Zero
    SkewSymmetricMatrix<Scalar, 3> m_zero(SkewSymmetricMatrix<Scalar, 3>::Zero());
    for (int i = 0; i < 3; ++i)
        for (int j = i+1; j < 3; ++j)
            assert(m_zero(i,j) == Scalar(0));
    std::cout << "Zero" << std::endl;
    std::cout << m_zero << std::endl;
    std::cout << std::endl;

    // NaN
    auto m_nan(SkewSymmetricMatrix<Scalar, 3>::NaN());
    for (int i = 0; i < 3; ++i)
        for (int j = i+1; j < 3; ++j)
            assert(std::isnan(m_nan(i,j)));
    std::cout << "NaN" << std::endl;
    std::cout << m_nan << std::endl;
    std::cout << std::endl;

    // Constructor from std::array
    std::array<Scalar, 3> arr = {1, 2, 3};
    SkewSymmetricMatrix<Scalar, 3> m_arr(arr);
    assert(m_arr(0,0) == Scalar(0));
    assert(m_arr(0,1) == Scalar(1));
    std::cout << "Constructor from std::array = {1, 2, 3}" << std::endl;
    std::cout << m_arr << std::endl;
    std::cout << std::endl;

    // Constructor from std::vector
    std::vector<Scalar> vec = {1, 2, 3};
    SkewSymmetricMatrix<Scalar, 3> m_vec(vec);
    assert(m_vec(0,0) == Scalar(0));
    assert(m_vec(0,1) == Scalar(1));
    std::cout << "Constructor from std::vector = {1, 2, 3}" << std::endl;
    std::cout << m_vec << std::endl;
    std::cout << std::endl;

    // Callable constructor
    SkewSymmetricMatrix<Scalar, 3> m_callable([]() {
        return std::array<Scalar, 3>{3, 2, 1};
    });
    assert(m_callable(0,0) == Scalar(0));
    assert(m_callable(0,1) == Scalar(3));
    std::cout << "Callable constructor std::array = {3, 2, 1}" << std::endl;
    std::cout << m_callable << std::endl;
    std::cout << std::endl;

    // Assignment from std::array
    SkewSymmetricMatrix<Scalar, 3> m_assign;
    m_assign = arr;
    assert(m_assign(0,1) == Scalar(1));
    std::cout << "Assign from std::array = {1, 2, 3}" << std::endl;
    std::cout << m_assign << std::endl;
    std::cout << std::endl;

    // Copy and assignment from a MatrixBase expression (it takes the skew-symmetric part)
    {
        Matrix<double, 3, 3> M({1, 2, 3, 4, 5, 6, 7, 8, 9});
        SkewSymmetricMatrix<Scalar, 3> m_copy(M);
        assert(m_arr(0,0) == Scalar(0));
        assert(m_arr(0,1) == Scalar(1));
        std::cout << "Copy from another SkewSymmetricMatrix" << std::endl;
        std::cout << m_copy << std::endl;
        std::cout << std::endl;
        SkewSymmetricMatrix<Scalar, 3> m_assign;
        m_assign = M;
        assert(m_arr(0,0) == Scalar(0));
        assert(m_arr(0,1) == Scalar(1));
        std::cout << "Assign from another SkewSymmetricMatrix" << std::endl;
        std::cout << m_assign << std::endl;
        std::cout << std::endl;
    }

    // Copy and assignment from a MatrixBase expression (it takes the skew-symmetric part)
    {
        Matrix<double, 3, 3> M({1, 2, 3, 4, 5, 6, 7, 8, 9});
        SkewSymmetricMatrix<Scalar, 3> m_copy(M);
        assert(m_copy(0,1) == Scalar(-1));
        std::cout << "Copy from another MatrixBase expression" << std::endl;
        std::cout << m_copy << std::endl;
        std::cout << std::endl;
        SkewSymmetricMatrix<Scalar, 3> m_assign;
        m_assign = M;
        assert(m_copy(0,1) == Scalar(-1));
        std::cout << "Assign from another MatrixBase expression" << std::endl;
        std::cout << m_assign << std::endl;
        std::cout << std::endl;
    }

    #ifdef __FDAPDE_HAS_EIGEN__
        // Eigen constructor and assignment
        Eigen::Matrix<Scalar, 2, 2> emat;
        emat << 1, 2, 3, 4;
        SkewSymmetricMatrix<Scalar, 2> m_eigen(emat);
        assert(m_eigen(0,1) == Scalar(-0.5));
        std::cout << "Eigen constructor from emat << 1, 2, 3, 4;" << std::endl;
        std::cout << m_eigen << std::endl;
        std::cout << std::endl;
        SkewSymmetricMatrix<Scalar, 2> m_eigen_assign;
        m_eigen_assign = emat;
        assert(m_eigen_assign(0,1) == Scalar(-0.5));
        std::cout << "Eigen assignement from emat << 1, 2, 3, 4;" << std::endl;
        std::cout << m_eigen_assign << std::endl;
        std::cout << std::endl;
        // Conversion to Eigen
        Eigen::Matrix<Scalar, 2, 2> emat_conv = m_eigen.as_eigen();
        assert(emat_conv(0,1) == Scalar(-0.5));
        std::cout << "Conversion to Eigen" << std::endl;
        std::cout << emat_conv << std::endl;
        std::cout << std::endl;
    #endif

}

TEST(matrix_test, SymmetricAndSkewSymmetric) {
    Matrix<double, 3, 3> M{{1,2,3, 4,5,6, 7,8,9}};
    std::cout << "M" << std::endl;
    std::cout << M << std::endl;
    std::cout << std::endl;

    std::cout << "Sym(M)" << std::endl;
    auto Sy_M = M.symmetric();
    std::cout << Sy_M << std::endl;
    std::cout << std::endl;

    std::cout << "SkewSym(M)" << std::endl;
    auto Sk_M = M.skew_symmetric();
    std::cout << Sk_M << std::endl;
    std::cout << std::endl;

    std::cout << "Sym(M) + SkewSym(M)" << std::endl;
    std::cout << Sy_M + Sk_M << std::endl;
    std::cout << std::endl;

    std::cout << "Sym(Sy_M)" << std::endl;
    std::cout << Sy_M.symmetric() << std::endl;
    std::cout << std::endl;

    std::cout << "SkewSym(Sk_M)" << std::endl;
    std::cout << Sk_M.skew_symmetric() << std::endl;
    std::cout << std::endl;

    std::cout << "Sym(Sk_M)" << std::endl;
    std::cout << Sk_M.symmetric() << std::endl;
    std::cout << std::endl;

    std::cout << "SkewSym(Sy_M)" << std::endl;
    std::cout << Sy_M.skew_symmetric() << std::endl;
    std::cout << std::endl;
}