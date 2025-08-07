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

TEST(matrix_test, MatrixBase) {

    Matrix<double, 3, 4> M({1,2,3,4, 5,6,7,8, 9,10,11,12});
    std::cout << "M" << std::endl;
    std::cout << M << std::endl;
    std::cout << std::endl;

    // views
    std::cout << "MatrixBlockView" << std::endl;
    std::cout << MatrixBlockView<2, 3, decltype(M)>(M, 0, 0) << std::endl;
    std::cout << std::endl;
    std::cout << "Transpose" << std::endl;
    std::cout << M.transpose() << std::endl;
    std::cout << std::endl;

    // summaries
    std::cout << "Sum(M) = ";
    std::cout << M.sum() << std::endl;
    std::cout << "Norm(M) = ";
    std::cout << M.norm() << std::endl;
    std::cout << "SqNorm(M) = ";
    std::cout << M.squared_norm() << std::endl;
    std::cout << "InfNorm(M) = ";
    std::cout << M.inf_norm() << std::endl;
    std::cout << "Min(M) = ";
    std::cout << M.min() << std::endl;
    std::cout << "Max(M) = ";
    std::cout << M.max() << std::endl;
    std::cout << "Mean(M) = ";
    std::cout << M.mean() << std::endl;
    std::cout << "Prod(M) = ";
    std::cout << M.prod() << std::endl;
    std::cout << std::endl;
}

TEST(matrix_test, expression_tempate) {

    Matrix<double, 3, 4> M({1,2,3,4, 5,6,7,8, 9,10,11,12});
    std::cout << "M" << std::endl;
    std::cout << M << std::endl;
    std::cout << std::endl;

    Matrix<double, 3, 4> N({5,4,3,2, 1,0,-1,-2, -3,-4,-5,-6});
    std::cout << "N" << std::endl;
    std::cout << N << std::endl;
    std::cout << std::endl;

    std::cout << "M+N" << std::endl;
    std::cout << M+N << std::endl;
    std::cout << std::endl;

    std::cout << "2M+N" << std::endl;
    std::cout << 2*M+N << std::endl;
    std::cout << std::endl;

    std::cout << "(2MSQ+2N)/2" << std::endl;
    std::cout << (2*M+2*N)/2 << std::endl;
    std::cout << std::endl;
}

TEST(matrix_test, SquareMatrixBase) {

    Matrix<double, 3, 3> M({1,2,3, 4,5,6, 7,8,9});
    std::cout << "M" << std::endl;
    std::cout << M << std::endl;
    std::cout << std::endl;

    // views
    std::cout << "(Lower) TriangularView" << std::endl;
    std::cout << M.triangular_view<Lower>() << std::endl;
    std::cout << std::endl;
    std::cout << "(Upper) TriangularView" << std::endl;
    std::cout << M.triangular_view<Upper>() << std::endl;
    std::cout << std::endl;
    std::cout << "DiagonalView" << std::endl;
    std::cout << M.diagonal() << std::endl;
    std::cout << std::endl;
    std::cout << "SymmetricPartView" << std::endl;
    std::cout << M.symmetric_part() << std::endl;
    std::cout << std::endl;
    std::cout << "SkewSymmetricPartView" << std::endl;
    std::cout << M.skew_symmetric_part() << std::endl;
    std::cout << std::endl;

    // summaries
    std::cout << "Trace(M) = ";
    std::cout << M.trace() << std::endl;
    std::cout << std::endl;

    // checks
    std::cout << "is_symmetric(M) = ";
    std::cout << M.is_symmetric() << std::endl;
    std::cout << "is_symmetric(Sym(M)) = ";
    std::cout << M.symmetric_part().is_symmetric() << std::endl;
    std::cout << std::endl;
    std::cout << "is_skew_symmetric(M) = ";
    std::cout << M.is_skew_symmetric() << std::endl;
    std::cout << "is_skew_symmetric(SkewSym(M)) = ";
    std::cout << M.skew_symmetric_part().is_skew_symmetric() << std::endl;
    std::cout << std::endl;
}

TEST(matrix_test, SymmetricAndSkewSymmetric) {
    Matrix<double, 3, 3> M{{1,2,3, 4,5,6, 7,8,9}};
    std::cout << "M" << std::endl;
    std::cout << M << std::endl;
    std::cout << std::endl;

    std::cout << "Sym(M)" << std::endl;
    auto Sy_M = M.symmetric_part();
    std::cout << Sy_M << std::endl;
    std::cout << std::endl;

    std::cout << "SkewSym(M)" << std::endl;
    auto Sk_M = M.skew_symmetric_part();
    std::cout << Sk_M << std::endl;
    std::cout << std::endl;

    std::cout << "Sym(M) + SkewSym(M)" << std::endl;
    std::cout << Sy_M + Sk_M << std::endl;
    std::cout << std::endl;

    std::cout << "Sym(Sy_M)" << std::endl;
    std::cout << Sy_M.symmetric_part() << std::endl;
    std::cout << std::endl;

    std::cout << "SkewSym(Sk_M)" << std::endl;
    std::cout << Sk_M.skew_symmetric_part() << std::endl;
    std::cout << std::endl;

    std::cout << "Sym(Sk_M)" << std::endl;
    std::cout << Sk_M.symmetric_part() << std::endl;
    std::cout << std::endl;

    std::cout << "SkewSym(Sy_M)" << std::endl;
    std::cout << Sy_M.skew_symmetric_part() << std::endl;
    std::cout << std::endl;
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

TEST(matrix_test, PermutationMatrix) {

    using Scalar = double;

    // PermutationMatrix (applied to a vector)
    std::array<int, 2> perm = {1, 0};
    PermutationMatrix<2> P(perm);
    auto v =  Vector<Scalar, 2>(Scalar(5), Scalar(7));
    auto Pv = P * v;
    assert(Pv[0] == Scalar(7) && Pv[1] == Scalar(5));
    std::cout << "Permutation matrix (applied to a vector)" << std::endl;
    std::cout << "P: " << P << std::endl;
    std::cout << "--" << std::endl;
    std::cout << "v: " << v << std::endl;
    std::cout << "--" << std::endl;
    std::cout << "P[v]: " << P*v << std::endl;
    std::cout << std::endl;

    // PermutationMatrix (applied to a matrix)
    Matrix<Scalar, 2, 2> M({1,2,3,4});
    auto Pm = P * M;
    assert(Pm(0,0) == Scalar(3) && Pm(1,0) == Scalar(1));
    std::cout << "Permutation matrix (applied to a matrix)" << std::endl;
    std::cout << "P: " <<P << std::endl;
    std::cout << "--" << std::endl;
    std::cout << "M: " << M << std::endl;
    std::cout << "--" << std::endl;
    std::cout << "P[M]: " << P*M << std::endl;
    std::cout << std::endl;
    std::cout << "[M]P: " << M*P << std::endl;
    std::cout << std::endl;
}


TEST(matrix_test, LU) {

    using Scalar = double;

    // test LU decomposition
    // of a MatrixView (direct initialization)
    {
        std::cout << "LU decomposition of a MatrixView (direct initialization)" << std::endl;
        Scalar data[4] = {2, 1, 4, 3};
        MatrixView<Scalar, 2, 2> A(data);
        Vector<Scalar, 2> b(Scalar(5), Scalar(11));
        PartialPivLU<decltype(A)> lu(A);
        auto x  = lu.solve(b);
        assert(std::abs(A(0,0)*x[0] + A(0,1)*x[1] - b[0]) < 1e-10);
        assert(std::abs(A(1,0)*x[0] + A(1,1)*x[1] - b[1]) < 1e-10);
        std::cout << "A: " << A << std::endl;
        std::cout << "--" << std::endl;
        std::cout << "x: " << x << std::endl;
        std::cout << "--" << std::endl;
        std::cout << "Ax: " << A*x << std::endl;
        std::cout << "--" << std::endl;
        std::cout << "b: " << b << std::endl;
        std::cout << std::endl;
    }

    // of a Matrix (direct initialization)
    {
        std::cout << "LU decomposition of a Matrix (direct initialization)" << std::endl;
        Matrix<Scalar, 2, 2> A({2, 1, 4, 3});
        Vector<Scalar, 2> b(Scalar(5), Scalar(11));
        PartialPivLU<decltype(A)> lu(A);
        auto x  = lu.solve(b);
        assert(std::abs(A(0,0)*x[0] + A(0,1)*x[1] - b[0]) < 1e-10);
        assert(std::abs(A(1,0)*x[0] + A(1,1)*x[1] - b[1]) < 1e-10);
        std::cout << "A: " << A << std::endl;
        std::cout << "--" << std::endl;
        std::cout << "x: " << x << std::endl;
        std::cout << "--" << std::endl;
        std::cout << "Ax: " << A*x << std::endl;
        std::cout << "--" << std::endl;
        std::cout << "b: " << b << std::endl;
        std::cout << std::endl;
    }

    // of a Matrix (using compute)
    {
        std::cout << "LU decomposition of a Matrix (using compute)" << std::endl;
        Matrix<Scalar, 2, 2> A({2, 1, 4, 3});
        Vector<Scalar, 2> b(Scalar(5), Scalar(11));
        PartialPivLU<decltype(A)> lu;
        lu.compute(A);
        auto x  = lu.solve(b);
        assert(std::abs(A(0,0)*x[0] + A(0,1)*x[1] - b[0]) < 1e-10);
        assert(std::abs(A(1,0)*x[0] + A(1,1)*x[1] - b[1]) < 1e-10);
        std::cout << "A: " << A << std::endl;
        std::cout << "--" << std::endl;
        std::cout << "x: " << x << std::endl;
        std::cout << "--" << std::endl;
        std::cout << "Ax: " << A*x << std::endl;
        std::cout << "--" << std::endl;
        std::cout << "b: " << b << std::endl;
        std::cout << std::endl;
    }

    // of a MatrixView (direct initialization)
    {
        std::cout << "LU decomposition of a SymmetricMatrixView (using compute)" << std::endl;
        Scalar data[4] = {1, 2, 2};
        SymmetricMatrixView<Scalar, 2> A(data);
        Vector<Scalar, 2> b(Scalar(5), Scalar(11));
        PartialPivLU<decltype(A)> lu(A);
        auto x  = lu.solve(b);
        assert(std::abs(A(0,0)*x[0] + A(0,1)*x[1] - b[0]) < 1e-10);
        assert(std::abs(A(1,0)*x[0] + A(1,1)*x[1] - b[1]) < 1e-10);
        std::cout << "A: " << A << std::endl;
        std::cout << "--" << std::endl;
        std::cout << "x: " << x << std::endl;
        std::cout << "--" << std::endl;
        std::cout << "Ax: " << A*x << std::endl;
        std::cout << "--" << std::endl;
        std::cout << "b: " << b << std::endl;
        std::cout << std::endl;
    }

    // of a SymmetricMatrix
    {
        std::cout << "LU decomposition of a SymmetricMatrix (using compute)" << std::endl;
        SymmetricMatrix<Scalar, 2> A({1, 2, 2});
        Vector<Scalar, 2> b(Scalar(5), Scalar(11));
        PartialPivLU<decltype(A)> lu;
        lu.compute(A);
        auto x  = lu.solve(b);
        assert(std::abs(A(0,0)*x[0] + A(0,1)*x[1] - b[0]) < 1e-10);
        assert(std::abs(A(1,0)*x[0] + A(1,1)*x[1] - b[1]) < 1e-10);
        std::cout << "A: " << A << std::endl;
        std::cout << "--" << std::endl;
        std::cout << "x: " << x << std::endl;
        std::cout << "--" << std::endl;
        std::cout << "Ax: " << A*x << std::endl;
        std::cout << "--" << std::endl;
        std::cout << "b: " << b << std::endl;
        std::cout << std::endl;
    }

    // check info in case of singular matrix
    {
        std::cout << "Try LU decomposition of a (singular) Matrix (using compute)" << std::endl;
        Matrix<Scalar, 2, 2> A({2, 1, 2, 1});
        PartialPivLU<decltype(A)> lu;
        lu.compute(A);
        if (lu.info() == Eigen::Success) {
            assert(1==0); // the matrix is actually singular
        } else {
            std::cout << "LU decomposition failed -> The matrix is singular\n" << std::endl;
        }
    }
}