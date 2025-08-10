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
/*
TEST(MatrixTest, ForwardBackwardSubstitution) {

    // Lower-triangular solve Lx=b
    {
        LowerTriangularMatrix<double, 3> L({2, 3, 5,  1, 4,  6});
        Vector<double,3> b({4, 21, 38});
        auto x = forward_sub(L, b);
        EXPECT_TRUE(almost_equal(x[0], 2.));
        EXPECT_TRUE(almost_equal(x[1], 3.));
        EXPECT_TRUE(almost_equal(x[2], 4.));
        std::cout << "L: " << L << std::endl;
        std::cout << "b: " << b << std::endl;
        std::cout << "x: " << x << std::endl;
        std::cout << std::endl;
    }

    // Upper-triangular solve Ux=b
    {
        UpperTriangularMatrix<double, 3> U({2,  3, 1,  5, 4, 6});
        Vector<double,3> b{11, 32, 18};
        auto x = backward_sub(U.triangular_view<Upper>(), b);
        EXPECT_TRUE(almost_equal(x[2], 3.));
        EXPECT_TRUE(almost_equal(x[1], 4.));
        EXPECT_TRUE(almost_equal(x[0], -2.));
        std::cout << "U: " << U << std::endl;
        std::cout << "b: " << b << std::endl;
        std::cout << "x: " << x << std::endl;
        std::cout << std::endl;
    }
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
        EXPECT_TRUE(lu.info());
        auto x  = lu.solve(b);
        EXPECT_TRUE(std::abs(A(0,0)*x[0] + A(0,1)*x[1] - b[0]) < 1e-10);
        EXPECT_TRUE(std::abs(A(1,0)*x[0] + A(1,1)*x[1] - b[1]) < 1e-10);
        std::cout << "A: " << A << std::endl;
        std::cout << "--" << std::endl;
        std::cout << "x: " << x << std::endl;
        std::cout << "--" << std::endl;
        std::cout << "Ax: " << A*x << std::endl;
        std::cout << "--" << std::endl;
        std::cout << "b: " << b << std::endl;
        std::cout << std::endl;
        std::cout << "P: " <<lu.P() << std::endl;
        std::cout << "U: "<< lu.U() << std::endl;
        std::cout << "L: "<< lu.L() << std::endl;
        std::cout << std::endl;
    }

    // of a Matrix (direct initialization)
    {
        std::cout << "LU decomposition of a Matrix (direct initialization)" << std::endl;
        Matrix<Scalar, 2, 2> A({2, 1, 4, 3});
        Vector<Scalar, 2> b(Scalar(5), Scalar(11));
        PartialPivLU<decltype(A)> lu(A);
        EXPECT_TRUE(lu.info());
        auto x  = lu.solve(b);
        EXPECT_TRUE(std::abs(A(0,0)*x[0] + A(0,1)*x[1] - b[0]) < 1e-10);
        EXPECT_TRUE(std::abs(A(1,0)*x[0] + A(1,1)*x[1] - b[1]) < 1e-10);
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
        EXPECT_TRUE(lu.info());
        lu.compute(A);
        auto x  = lu.solve(b);
        EXPECT_TRUE(std::abs(A(0,0)*x[0] + A(0,1)*x[1] - b[0]) < 1e-10);
        EXPECT_TRUE(std::abs(A(1,0)*x[0] + A(1,1)*x[1] - b[1]) < 1e-10);
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
        EXPECT_TRUE(lu.info());
        auto x  = lu.solve(b);
        EXPECT_TRUE(std::abs(A(0,0)*x[0] + A(0,1)*x[1] - b[0]) < 1e-10);
        EXPECT_TRUE(std::abs(A(1,0)*x[0] + A(1,1)*x[1] - b[1]) < 1e-10);
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
        EXPECT_TRUE(std::abs(A(0,0)*x[0] + A(0,1)*x[1] - b[0]) < 1e-10);
        EXPECT_TRUE(std::abs(A(1,0)*x[0] + A(1,1)*x[1] - b[1]) < 1e-10);
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
        EXPECT_TRUE(!lu.info());
        if (!lu.info()) {
            std::cout << "LU decomposition failed -> The matrix is singular\n" << std::endl;
        }
    }
    {
        std::cout << "Try LU decomposition of a (singular) Matrix (using compute)" << std::endl;
        Matrix<Scalar, 3, 3> A({2, 1, 0, 2, 1, 0, 2, 1, 0});
        PartialPivLU<decltype(A)> lu;
        lu.compute(A);
        EXPECT_TRUE(!lu.info());
        if (!lu.info()) {
            std::cout << "LU decomposition failed -> The matrix is singular\n" << std::endl;
        }
    }
}
*/
TEST(MatrixTest, QRDecomposition) {
    Matrix<double,3,3> M({1, 2, 3,  0, 1, 4,  5, 6, 0});
    QRDecomposition<Matrix<double,3,3>> qr(M);
    auto Q = qr.Q();
    auto R = qr.R();
    // Q*R ≈ M
    std::cout << "M" << std::endl;
    std::cout << M << std::endl;
    std::cout << std::endl;
    std::cout << "Q" << std::endl;
    std::cout << Q << std::endl;
    std::cout << std::endl;
    std::cout << "R" << std::endl;
    std::cout << R << std::endl;
    std::cout << std::endl;
    std::cout << "Q*R" << std::endl;
    std::cout << Q*R << std::endl;
    EXPECT_TRUE(almost_equal(Q * R, M));
    // Q^T * Q ≈ I
    Matrix<double,3,3> I_check(Q.transpose() * Q);
    auto I = Matrix<double, 3, 3>::Identity();
    EXPECT_TRUE(almost_equal(I, I_check));
}

TEST(MatrixTest, EigenDecomposition2x2) {

    // matrix to be decomposed
    Matrix<double,2,2> A({3, 1, 1, -1});

    // EVD (QR solver)
    {
        std::cout << "EVD (QR solver)\n" << std::endl;
        EigenDecomposition<decltype(A), QR> evd(A.symmetric_part());
        auto lambdas = evd.eigenvalues();
        auto Q = evd.eigenvectors();

        // check eigenvalues
        EXPECT_TRUE(almost_equal(lambdas.sum(), A.trace()));

        // check eigenvectors orthonormality
        auto I = Matrix<double,2,2>::Identity();
        EXPECT_TRUE(almost_equal(Q.transpose()*Q, I));

        // check reconstruction
        auto D = Matrix<double,2,2>::Zero();
        D(0,0) = lambdas[0];
        D(1,1) = lambdas[1];
        auto A_rec = Q * D * Q.transpose();
        EXPECT_TRUE(almost_equal(A, A_rec));

        // print
        std::cout << "A" << std::endl;
        std::cout << A << std::endl;
        std::cout << std::endl;
        std::cout << "Q" << std::endl;
        std::cout << Q << std::endl;
        std::cout << std::endl;
        std::cout << "D" << std::endl;
        std::cout << D << std::endl;
        std::cout << std::endl;
        std::cout << "A_rec" << std::endl;
        std::cout << A_rec << std::endl;
        std::cout << std::endl;
    }

}

TEST(MatrixTest, EigenDecomposition3x3) {

    // matrix to be decomposed
    Matrix<double,3,3> A({6, 2, 1,  2, 3, 1,  1, 1, 1});

    // EVD (QR solver)
    {
        std::cout << "EVD (QR solver)\n" << std::endl;
        EigenDecomposition<decltype(A), QR> evd(A.symmetric_part());
        auto lambdas = evd.eigenvalues();
        auto Q = evd.eigenvectors();

        // check eigenvalues
        EXPECT_TRUE(almost_equal(lambdas.sum(), A.trace()));

        // check eigenvectors orthonormality
        auto I = Matrix<double, 3, 3>::Identity();
        EXPECT_TRUE(almost_equal(Q.transpose()*Q, I));

        // check reconstruction
        auto D = Matrix<double, 3, 3>::Zero();
        D(0,0) = lambdas[0];
        D(1,1) = lambdas[1];
        D(2,2) = lambdas[2];
        auto A_rec = Q * D * Q.transpose();
        EXPECT_TRUE(almost_equal(A, A_rec));

        // print
        std::cout << "A" << std::endl;
        std::cout << A << std::endl;
        std::cout << std::endl;
        std::cout << "Q" << std::endl;
        std::cout << Q << std::endl;
        std::cout << std::endl;
        std::cout << "D" << std::endl;
        std::cout << D << std::endl;
        std::cout << std::endl;
        std::cout << "A_rec" << std::endl;
        std::cout << A_rec << std::endl;
        std::cout << std::endl;
    }

}


TEST(MatrixTest, EigenDecompositionWithMultiplicity2x2) {
    // double multiplicity
    SymmetricMatrix<double,2> A({2, 0, 2});
    std::cout << "A: " << A << std::endl;
    EigenDecomposition<decltype(A)> evd(A);
    auto lambdas = evd.eigenvalues();
    auto Q = evd.eigenvectors();
    std::cout << "eigenvalues: " << lambdas << std::endl;
    std::cout << "eigenvectors: " << Q << std::endl;
    std::cout << std::endl;
}

TEST(MatrixTest, EigenDecompositionWithMultiplicity3x3) {
    // triple multiplicity
    {
        SymmetricMatrix<double,3> A({2,0,0, 2,0, 2});
        std::cout << "A: " << A << std::endl;
        EigenDecomposition<decltype(A)> evd(A);
        auto lambdas = evd.eigenvalues();
        auto Q = evd.eigenvectors();
        std::cout << "eigenvalues: " << lambdas << std::endl;
        std::cout << "eigenvectors: " << Q << std::endl;
        std::cout << std::endl;
    }
    // double multiplicity
    {
        SymmetricMatrix<double,3> A({2,0,0, 2,0, 1});
        std::cout << "A: " << A << std::endl;
        EigenDecomposition<decltype(A)> evd(A);
        auto lambdas = evd.eigenvalues();
        auto Q = evd.eigenvectors();
        std::cout << "eigenvalues: " << lambdas << std::endl;
        std::cout << "eigenvectors: " << Q << std::endl;
        std::cout << std::endl;
    }
}

TEST(MatrixTest, EigenDecompositionRankDeficient2x2) {
    // double multiplicity
    SymmetricMatrix<double,2> A({1, 1, 1});
    std::cout << "A: " << A << std::endl;
    EigenDecomposition<decltype(A)> evd(A);
    auto lambdas = evd.eigenvalues();
    auto Q = evd.eigenvectors();
    std::cout << "eigenvalues: " << lambdas << std::endl;
    std::cout << "eigenvectors: " <<  Q << std::endl;
    std::cout << std::endl;
}
// rank deficient
TEST(MatrixTest, EigenDecompositionRankDeficient3x3) {
    // double multiplicity
    SymmetricMatrix<double,3> A({1, 1, 0,  1, 0,  0});
    std::cout << "A: " << A << std::endl;
    EigenDecomposition<decltype(A)> evd(A);
    auto lambdas = evd.eigenvalues();
    auto Q = evd.eigenvectors();
    std::cout << "eigenvalues: " << lambdas << std::endl;
    std::cout << "eigenvectors: " <<  Q << std::endl;
    std::cout << std::endl;
}