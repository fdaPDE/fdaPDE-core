#include <gtest/gtest.h>

#include <unsupported/Eigen/SparseExtra>

#include "../../fdaPDE/linear_algebra.h"
#include "utils/rand_indices.h"

using namespace fdapde::mumps;
using namespace Eigen;

#include "../../test/src/utils/constants.h"
using fdapde::testing::DOUBLE_TOLERANCE;

int schur_size = 10;

TEST(MumpsSchur_test, split_analyze_factorize) {
    SparseMatrix<double> A;
    Eigen::loadMarket(A, "../data/matrix_fullrank.mtx");

    MumpsSchur<SparseMatrix<double>> solver;
    EXPECT_TRUE(solver.rows() == 0);
    EXPECT_TRUE(solver.cols() == 0);

    solver.setSchurSize(schur_size);

    solver.analyzePattern(A);
    EXPECT_TRUE(solver.info() == Success);
    EXPECT_TRUE(solver.rows() == A.rows());
    EXPECT_TRUE(solver.cols() == A.cols());

    solver.factorize(A);
    EXPECT_TRUE(solver.info() == Success);
    double det = solver.determinant();

    VectorXd x, b;
    b = VectorXd::Ones(A.rows());
    x = solver.solve(b);

    MatrixXd X, B;
    B = MatrixXd::Ones(A.rows(), 3);
    X = solver.solve(B);

    MatrixXd complement = solver.complement();
    EXPECT_TRUE(complement.rows() == schur_size);
    EXPECT_TRUE(complement.cols() == schur_size);
}

TEST(MumpsSchur_test, compute) {
    SparseMatrix<double> A;
    Eigen::loadMarket(A, "../data/matrix_fullrank.mtx");

    MumpsSchur<SparseMatrix<double>> solver;
    EXPECT_TRUE(solver.rows() == 0);
    EXPECT_TRUE(solver.cols() == 0);

    // std::vector<int> schur_size;

    solver.setSchurSize(schur_size);

    solver.compute(A);
    EXPECT_TRUE(solver.info() == Success);
    EXPECT_TRUE(solver.rows() == A.rows());
    EXPECT_TRUE(solver.cols() == A.cols());
    double det = solver.determinant();

    VectorXd x, b;
    b = VectorXd::Ones(A.rows());
    x = solver.solve(b);

    MatrixXd X, B;
    B = MatrixXd::Ones(A.rows(), 3);
    X = solver.solve(B);

    MatrixXd complement = solver.complement();
    EXPECT_TRUE(complement.rows() == schur_size);
    EXPECT_TRUE(complement.cols() == schur_size);
}

TEST(MumpsSchur_test, type_deduction) {
    SparseMatrix<double> A;
    Eigen::loadMarket(A, "../data/matrix_fullrank.mtx");

    MumpsSchur solver(A, schur_size);
    EXPECT_TRUE(solver.info() == Success);
    EXPECT_TRUE(solver.rows() == A.rows());
    EXPECT_TRUE(solver.cols() == A.cols());
    double det = solver.determinant();

    VectorXd x, b;
    b = VectorXd::Ones(A.rows());
    x = solver.solve(b);

    MatrixXd X, B;
    B = MatrixXd::Ones(A.rows(), 3);
    X = solver.solve(B);

    MatrixXd complement = solver.complement();
    EXPECT_TRUE(complement.rows() == schur_size);
    EXPECT_TRUE(complement.cols() == schur_size);
}

TEST(MumpsSchur_test, flags) {
    MumpsSchur<SparseMatrix<double>> solver(NoDeterminant);
    EXPECT_TRUE(solver.mumpsIcntl()[0] == -1);
    EXPECT_TRUE(solver.mumpsIcntl()[1] == -1);
    EXPECT_TRUE(solver.mumpsIcntl()[2] == -1);
    EXPECT_TRUE(solver.mumpsIcntl()[3] == 0);
    EXPECT_TRUE(solver.mumpsIcntl()[32] == 0);
    EXPECT_TRUE(solver.mumpsRawStruct().par == 0);

    MumpsSchur<SparseMatrix<double>> solver2(Verbose);
    if (solver2.getProcessRank() == 0) {
        EXPECT_TRUE(solver2.mumpsIcntl()[0] == 6);
        EXPECT_TRUE(solver2.mumpsIcntl()[1] == 6);
        EXPECT_TRUE(solver2.mumpsIcntl()[2] == 6);
        EXPECT_TRUE(solver2.mumpsIcntl()[3] == 4);
    }
    EXPECT_TRUE(solver2.mumpsIcntl()[32] == 1);
    EXPECT_TRUE(solver2.mumpsRawStruct().par == 0);

    MumpsSchur<SparseMatrix<double>> solver3(WorkingHost);
    EXPECT_TRUE(solver3.mumpsIcntl()[0] == -1);
    EXPECT_TRUE(solver3.mumpsIcntl()[1] == -1);
    EXPECT_TRUE(solver3.mumpsIcntl()[2] == -1);
    EXPECT_TRUE(solver3.mumpsIcntl()[3] == 0);
    EXPECT_TRUE(solver3.mumpsIcntl()[32] == 1);
    EXPECT_TRUE(solver3.mumpsRawStruct().par == 1);

    MumpsSchur<SparseMatrix<double>> solver4(NoDeterminant | Verbose | WorkingHost);
    if (solver4.getProcessRank() == 0) {
        EXPECT_TRUE(solver4.mumpsIcntl()[0] == 6);
        EXPECT_TRUE(solver4.mumpsIcntl()[1] == 6);
        EXPECT_TRUE(solver4.mumpsIcntl()[2] == 6);
        EXPECT_TRUE(solver4.mumpsIcntl()[3] == 4);
    }
    EXPECT_TRUE(solver4.mumpsIcntl()[32] == 0);
    EXPECT_TRUE(solver4.mumpsRawStruct().par == 1);

    MumpsSchur<SparseMatrix<double>> solver5(NoDeterminant | Verbose);
    if (solver5.getProcessRank() == 0) {
        EXPECT_TRUE(solver5.mumpsIcntl()[0] == 6);
        EXPECT_TRUE(solver5.mumpsIcntl()[1] == 6);
        EXPECT_TRUE(solver5.mumpsIcntl()[2] == 6);
        EXPECT_TRUE(solver5.mumpsIcntl()[3] == 4);
    }
    EXPECT_TRUE(solver5.mumpsIcntl()[32] == 0);
    EXPECT_TRUE(solver5.mumpsRawStruct().par == 0);

    MumpsSchur<SparseMatrix<double>> solver6(NoDeterminant | WorkingHost);
    EXPECT_TRUE(solver6.mumpsIcntl()[0] == -1);
    EXPECT_TRUE(solver6.mumpsIcntl()[1] == -1);
    EXPECT_TRUE(solver6.mumpsIcntl()[2] == -1);
    EXPECT_TRUE(solver6.mumpsIcntl()[3] == 0);
    EXPECT_TRUE(solver6.mumpsIcntl()[32] == 0);
    EXPECT_TRUE(solver6.mumpsRawStruct().par == 1);

    MumpsSchur<SparseMatrix<double>> solver7(Verbose | WorkingHost);
    if (solver7.getProcessRank() == 0) {
        EXPECT_TRUE(solver7.mumpsIcntl()[0] == 6);
        EXPECT_TRUE(solver7.mumpsIcntl()[1] == 6);
        EXPECT_TRUE(solver7.mumpsIcntl()[2] == 6);
        EXPECT_TRUE(solver7.mumpsIcntl()[3] == 4);
    }
    EXPECT_TRUE(solver7.mumpsIcntl()[32] == 1);
    EXPECT_TRUE(solver7.mumpsRawStruct().par == 1);

    SparseMatrix<double> A;
    Eigen::loadMarket(A, "../data/matrix_fullrank.mtx");

    MumpsSchur solver8(A, schur_size, NoDeterminant);
    EXPECT_TRUE(solver8.mumpsIcntl()[0] == -1);
    EXPECT_TRUE(solver8.mumpsIcntl()[1] == -1);
    EXPECT_TRUE(solver8.mumpsIcntl()[2] == -1);
    EXPECT_TRUE(solver8.mumpsIcntl()[3] == 0);
    EXPECT_TRUE(solver8.mumpsIcntl()[32] == 0);
    EXPECT_TRUE(solver8.mumpsRawStruct().par == 0);

    MumpsSchur solver9(A, schur_size, Verbose);
    if (solver9.getProcessRank() == 0) {
        EXPECT_TRUE(solver9.mumpsIcntl()[0] == 6);
        EXPECT_TRUE(solver9.mumpsIcntl()[1] == 6);
        EXPECT_TRUE(solver9.mumpsIcntl()[2] == 6);
        EXPECT_TRUE(solver9.mumpsIcntl()[3] == 4);
    }
    EXPECT_TRUE(solver9.mumpsIcntl()[32] == 1);
    EXPECT_TRUE(solver9.mumpsRawStruct().par == 0);

    MumpsSchur solver10(A, schur_size, WorkingHost);
    EXPECT_TRUE(solver10.mumpsIcntl()[0] == -1);
    EXPECT_TRUE(solver10.mumpsIcntl()[1] == -1);
    EXPECT_TRUE(solver10.mumpsIcntl()[2] == -1);
    EXPECT_TRUE(solver10.mumpsIcntl()[3] == 0);
    EXPECT_TRUE(solver10.mumpsIcntl()[32] == 1);
    EXPECT_TRUE(solver10.mumpsRawStruct().par == 1);

    MumpsSchur solver11(A, schur_size, NoDeterminant | Verbose | WorkingHost);
    if (solver11.getProcessRank() == 0) {
        EXPECT_TRUE(solver11.mumpsIcntl()[0] == 6);
        EXPECT_TRUE(solver11.mumpsIcntl()[1] == 6);
        EXPECT_TRUE(solver11.mumpsIcntl()[2] == 6);
        EXPECT_TRUE(solver11.mumpsIcntl()[3] == 4);
    }
    EXPECT_TRUE(solver11.mumpsIcntl()[32] == 0);
    EXPECT_TRUE(solver11.mumpsRawStruct().par == 1);

    MumpsSchur solver12(A, schur_size, NoDeterminant | Verbose);
    if (solver12.getProcessRank() == 0) {
        EXPECT_TRUE(solver12.mumpsIcntl()[0] == 6);
        EXPECT_TRUE(solver12.mumpsIcntl()[1] == 6);
        EXPECT_TRUE(solver12.mumpsIcntl()[2] == 6);
        EXPECT_TRUE(solver12.mumpsIcntl()[3] == 4);
    }
    EXPECT_TRUE(solver12.mumpsIcntl()[32] == 0);
    EXPECT_TRUE(solver12.mumpsRawStruct().par == 0);

    MumpsSchur solver13(A, schur_size, NoDeterminant | WorkingHost);
    EXPECT_TRUE(solver13.mumpsIcntl()[0] == -1);
    EXPECT_TRUE(solver13.mumpsIcntl()[1] == -1);
    EXPECT_TRUE(solver13.mumpsIcntl()[2] == -1);
    EXPECT_TRUE(solver13.mumpsIcntl()[3] == 0);
    EXPECT_TRUE(solver13.mumpsIcntl()[32] == 0);
    EXPECT_TRUE(solver13.mumpsRawStruct().par == 1);

    MumpsSchur solver14(A, schur_size, Verbose | WorkingHost);
    if (solver14.getProcessRank() == 0) {
        EXPECT_TRUE(solver14.mumpsIcntl()[0] == 6);
        EXPECT_TRUE(solver14.mumpsIcntl()[1] == 6);
        EXPECT_TRUE(solver14.mumpsIcntl()[2] == 6);
        EXPECT_TRUE(solver14.mumpsIcntl()[3] == 4);
    }
    EXPECT_TRUE(solver14.mumpsIcntl()[32] == 1);
    EXPECT_TRUE(solver14.mumpsRawStruct().par == 1);
}

TEST(MumpsSchur_test, colmajor_vs_rowmajor) {
    SparseMatrix<double> A_colmajor;
    SparseMatrix<double, RowMajor> A_rowmajor;
    Eigen::loadMarket(A_colmajor, "../data/matrix_fullrank.mtx");
    Eigen::loadMarket(A_rowmajor, "../data/matrix_fullrank.mtx");

    MumpsSchur solver_colmajor(A_colmajor, schur_size);
    MumpsSchur solver_rowmajor(A_rowmajor, schur_size);

    EXPECT_TRUE(solver_colmajor.info() == Success);
    EXPECT_TRUE(solver_rowmajor.info() == Success);
    EXPECT_TRUE(solver_colmajor.rows() == A_colmajor.rows());
    EXPECT_TRUE(solver_rowmajor.rows() == A_rowmajor.rows());
    EXPECT_TRUE(solver_colmajor.cols() == A_colmajor.cols());
    EXPECT_TRUE(solver_rowmajor.cols() == A_rowmajor.cols());
    double det_colmajor = solver_colmajor.determinant();
    double det_rowmajor = solver_rowmajor.determinant();

    VectorXd x_colmajor, x_rowmajor, b;
    b = VectorXd::Ones(A_colmajor.rows());
    x_colmajor = solver_colmajor.solve(b);
    x_rowmajor = solver_rowmajor.solve(b);

    Matrix<double, Dynamic, Dynamic> X_colmajor_colcol, X_colmajor_rowrow, X_colmajor_colrow, X_colmajor_rowcol,
      B_colmajor;
    Matrix<double, Dynamic, Dynamic, RowMajor> X_rowmajor_colcol, X_rowmajor_rowrow, X_rowmajor_colrow,
      X_rowmajor_rowcol, B_rowmajor;
    B_colmajor = MatrixXd::Ones(A_colmajor.rows(), 3);
    B_rowmajor = MatrixXd::Ones(A_rowmajor.rows(), 3);
    X_colmajor_colcol = solver_colmajor.solve(B_colmajor);
    X_colmajor_rowrow = solver_rowmajor.solve(B_rowmajor);
    X_colmajor_colrow = solver_colmajor.solve(B_rowmajor);
    X_colmajor_rowcol = solver_rowmajor.solve(B_colmajor);
    X_rowmajor_colcol = solver_colmajor.solve(B_colmajor);
    X_rowmajor_rowrow = solver_rowmajor.solve(B_rowmajor);
    X_rowmajor_colrow = solver_colmajor.solve(B_rowmajor);
    X_rowmajor_rowcol = solver_rowmajor.solve(B_colmajor);

    EXPECT_TRUE(X_colmajor_colcol.isApprox(X_colmajor_colrow, DOUBLE_TOLERANCE));
    EXPECT_TRUE(X_colmajor_colrow.isApprox(X_colmajor_rowcol, DOUBLE_TOLERANCE));
    EXPECT_TRUE(X_colmajor_rowcol.isApprox(X_colmajor_rowrow, DOUBLE_TOLERANCE));

    EXPECT_TRUE(X_rowmajor_colcol.isApprox(X_rowmajor_colrow, DOUBLE_TOLERANCE));
    EXPECT_TRUE(X_rowmajor_colrow.isApprox(X_rowmajor_rowcol, DOUBLE_TOLERANCE));
    EXPECT_TRUE(X_rowmajor_rowcol.isApprox(X_rowmajor_rowrow, DOUBLE_TOLERANCE));

    EXPECT_TRUE(X_colmajor_colcol.isApprox(X_rowmajor_colcol, DOUBLE_TOLERANCE));

    Matrix<double, Dynamic, Dynamic> complement_colmajor_col = solver_colmajor.complement();
    Matrix<double, Dynamic, Dynamic> complement_colmajor_row = solver_rowmajor.complement();
    EXPECT_TRUE(complement_colmajor_col.rows() == schur_size);
    EXPECT_TRUE(complement_colmajor_col.cols() == schur_size);
    EXPECT_TRUE(complement_colmajor_row.rows() == schur_size);
    EXPECT_TRUE(complement_colmajor_row.cols() == schur_size);
    EXPECT_TRUE(complement_colmajor_col.isApprox(complement_colmajor_row, DOUBLE_TOLERANCE));

    Matrix<double, Dynamic, Dynamic, RowMajor> complement_rowmajor_col = solver_colmajor.complement();
    Matrix<double, Dynamic, Dynamic, RowMajor> complement_rowmajor_row = solver_rowmajor.complement();
    EXPECT_TRUE(complement_rowmajor_col.rows() == schur_size);
    EXPECT_TRUE(complement_rowmajor_col.cols() == schur_size);
    EXPECT_TRUE(complement_rowmajor_row.rows() == schur_size);
    EXPECT_TRUE(complement_rowmajor_row.cols() == schur_size);
    EXPECT_TRUE(complement_rowmajor_col.isApprox(complement_rowmajor_row, DOUBLE_TOLERANCE));
}

TEST(MumpsSchur_test, icntl_check) {
    MumpsSchur<SparseMatrix<double>> solver;
    EXPECT_TRUE(solver.mumpsIcntl()[18] == 3);
}

TEST(MumpsSchur_test, split_analyze_factorize_sparse) {
    SparseMatrix<double> A;
    Eigen::loadMarket(A, "../data/matrix_fullrank.mtx");

    MumpsSchur<SparseMatrix<double>> solver;
    EXPECT_TRUE(solver.rows() == 0);
    EXPECT_TRUE(solver.cols() == 0);

    solver.setSchurSize(schur_size);

    solver.analyzePattern(A);
    EXPECT_TRUE(solver.info() == Success);
    EXPECT_TRUE(solver.rows() == A.rows());
    EXPECT_TRUE(solver.cols() == A.cols());

    solver.factorize(A);
    EXPECT_TRUE(solver.info() == Success);
    double det = solver.determinant();

    SparseMatrix<double> x, b;
    b = VectorXd::Ones(A.rows()).sparseView();
    x = solver.solve(b);

    SparseMatrix<double> X, B;
    B = MatrixXd::Ones(A.rows(), 3).sparseView();
    X = solver.solve(B);
}

TEST(MumpsSchur_test, compute_sparse) {
    SparseMatrix<double> A;
    Eigen::loadMarket(A, "../data/matrix_fullrank.mtx");

    MumpsSchur<SparseMatrix<double>> solver;
    EXPECT_TRUE(solver.rows() == 0);
    EXPECT_TRUE(solver.cols() == 0);

    solver.setSchurSize(schur_size);

    solver.compute(A);
    EXPECT_TRUE(solver.info() == Success);
    EXPECT_TRUE(solver.rows() == A.rows());
    EXPECT_TRUE(solver.cols() == A.cols());
    double det = solver.determinant();

    SparseVector<double> x, b;
    b = VectorXd::Ones(A.rows()).sparseView();
    x = solver.solve(b);

    SparseMatrix<double> X, B;
    B = MatrixXd::Ones(A.rows(), 3).sparseView();
    X = solver.solve(B);
}

TEST(MumpsSchur_test, type_deduction_sparse) {
    SparseMatrix<double> A;
    Eigen::loadMarket(A, "../data/matrix_fullrank.mtx");

    MumpsSchur solver(A, schur_size);
    EXPECT_TRUE(solver.info() == Success);
    EXPECT_TRUE(solver.rows() == A.rows());
    EXPECT_TRUE(solver.cols() == A.cols());
    double det = solver.determinant();

    SparseVector<double> x, b;
    b = VectorXd::Ones(A.rows()).sparseView();
    x = solver.solve(b);

    SparseMatrix<double> X, B;
    B = MatrixXd::Ones(A.rows(), 3).sparseView();
    X = solver.solve(B);
}

TEST(MumpsSchur_test, colmajor_vs_rowmajor_sparse) {
    SparseMatrix<double> A_colmajor;
    SparseMatrix<double, RowMajor> A_rowmajor;
    Eigen::loadMarket(A_colmajor, "../data/matrix_fullrank.mtx");
    Eigen::loadMarket(A_rowmajor, "../data/matrix_fullrank.mtx");

    MumpsSchur solver_colmajor(A_colmajor, schur_size);
    MumpsSchur solver_rowmajor(A_rowmajor, schur_size);

    EXPECT_TRUE(solver_colmajor.info() == Success);
    EXPECT_TRUE(solver_rowmajor.info() == Success);
    EXPECT_TRUE(solver_colmajor.rows() == A_colmajor.rows());
    EXPECT_TRUE(solver_rowmajor.rows() == A_rowmajor.rows());
    EXPECT_TRUE(solver_colmajor.cols() == A_colmajor.cols());
    EXPECT_TRUE(solver_rowmajor.cols() == A_rowmajor.cols());
    double det_colmajor = solver_colmajor.determinant();
    double det_rowmajor = solver_rowmajor.determinant();
    EXPECT_TRUE(det_colmajor != 0);
    EXPECT_TRUE(det_rowmajor != 0);

    SparseVector<double> x_colmajor, x_rowmajor, b;
    b = VectorXd::Ones(A_colmajor.rows()).sparseView();
    x_colmajor = solver_colmajor.solve(b);
    x_rowmajor = solver_rowmajor.solve(b);
    EXPECT_TRUE(x_colmajor.isApprox(x_rowmajor, DOUBLE_TOLERANCE));

    SparseMatrix<double> X_colmajor_colcol, X_colmajor_rowrow, X_colmajor_colrow, X_colmajor_rowcol, B_colmajor;
    SparseMatrix<double, RowMajor> X_rowmajor_colcol, X_rowmajor_rowrow, X_rowmajor_colrow, X_rowmajor_rowcol,
      B_rowmajor;
    B_colmajor = MatrixXd::Ones(A_colmajor.rows(), 3).sparseView();
    B_rowmajor = MatrixXd::Ones(A_rowmajor.rows(), 3).sparseView();
    X_colmajor_colcol = solver_colmajor.solve(B_colmajor);
    X_colmajor_rowrow = solver_rowmajor.solve(B_rowmajor);
    X_colmajor_colrow = solver_colmajor.solve(B_rowmajor);
    X_colmajor_rowcol = solver_rowmajor.solve(B_colmajor);
    X_rowmajor_colcol = solver_colmajor.solve(B_colmajor);
    X_rowmajor_rowrow = solver_rowmajor.solve(B_rowmajor);
    X_rowmajor_colrow = solver_colmajor.solve(B_rowmajor);
    X_rowmajor_rowcol = solver_rowmajor.solve(B_colmajor);

    EXPECT_TRUE(X_colmajor_colcol.isApprox(X_colmajor_colrow, DOUBLE_TOLERANCE));
    EXPECT_TRUE(X_colmajor_colrow.isApprox(X_colmajor_rowcol, DOUBLE_TOLERANCE));
    EXPECT_TRUE(X_colmajor_rowcol.isApprox(X_colmajor_rowrow, DOUBLE_TOLERANCE));

    EXPECT_TRUE(X_rowmajor_colcol.isApprox(X_rowmajor_colrow, DOUBLE_TOLERANCE));
    EXPECT_TRUE(X_rowmajor_colrow.isApprox(X_rowmajor_rowcol, DOUBLE_TOLERANCE));
    EXPECT_TRUE(X_rowmajor_rowcol.isApprox(X_rowmajor_rowrow, DOUBLE_TOLERANCE));

    EXPECT_TRUE(X_colmajor_colcol.isApprox(X_rowmajor_colcol, DOUBLE_TOLERANCE));

    Matrix<double, Dynamic, Dynamic> complement_colmajor_col = solver_colmajor.complement();
    Matrix<double, Dynamic, Dynamic> complement_colmajor_row = solver_rowmajor.complement();
    EXPECT_TRUE(complement_colmajor_col.rows() == schur_size);
    EXPECT_TRUE(complement_colmajor_col.cols() == schur_size);
    EXPECT_TRUE(complement_colmajor_row.rows() == schur_size);
    EXPECT_TRUE(complement_colmajor_row.cols() == schur_size);
    EXPECT_TRUE(complement_colmajor_col.isApprox(complement_colmajor_row, DOUBLE_TOLERANCE));

    Matrix<double, Dynamic, Dynamic, RowMajor> complement_rowmajor_col = solver_colmajor.complement();
    Matrix<double, Dynamic, Dynamic, RowMajor> complement_rowmajor_row = solver_rowmajor.complement();
    EXPECT_TRUE(complement_rowmajor_col.rows() == schur_size);
    EXPECT_TRUE(complement_rowmajor_col.cols() == schur_size);
    EXPECT_TRUE(complement_rowmajor_row.rows() == schur_size);
    EXPECT_TRUE(complement_rowmajor_row.cols() == schur_size);
    EXPECT_TRUE(complement_rowmajor_col.isApprox(complement_rowmajor_row, DOUBLE_TOLERANCE));
}
