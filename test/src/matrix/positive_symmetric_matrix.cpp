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
#include <cmath>

using namespace fdapde;

TEST(matrix_test, SPDMatrixView) {

    using Scalar = double;

    // data
    constexpr int n = 2;
    Scalar data[ n * (n + 1) / 2 ] = {0};
    Scalar data_new[ n * (n + 1) / 2 ] = {0};

    // A 2x2 SPD matrix A = [[2, 1], [1, 2]]
    // Symmetric packed storage (upper triangle):
    // indices: (0,0)->0, (0,1)->1, (1,1)->2
    std::array<Scalar, 3> Aarr = {2.0, 1.0, 2.0};

    // Construct a view on raw memory and assign a valid SPD matrix
    SPDMatrixView<Scalar, 2> m_raw(data);
    m_raw = Aarr; // check() runs inside operator=
    {
        EigenDecomposition<decltype(m_raw)> evd(m_raw);
        auto evals = evd.eigenvalues();
        auto Q = evd.eigenvectors();
        EXPECT_TRUE(evals[0] > 0.0 && evals[1] > 0.0);
        EXPECT_TRUE(almost_equal(Q.transpose() * Q, DiagonalMatrix<Scalar, 2>::Identity()));
    }
    std::cout << "SPDMatrixView (A)" << std::endl;
    std::cout << m_raw << std::endl;
    std::cout << std::endl;

    // assignment from std::array
    SPDMatrixView<Scalar, 2> m_arr(data_new);
    m_arr = Aarr;
    EXPECT_TRUE(almost_equal(m_arr(0,0), 2.0));
    EXPECT_TRUE(almost_equal(m_arr(0,1), 1.0));
    EXPECT_TRUE(almost_equal(m_arr(1,0), 1.0));
    EXPECT_TRUE(almost_equal(m_arr(1,1), 2.0));
    {
        EigenDecomposition<decltype(m_arr)> evd(m_arr);
        auto ev = evd.eigenvalues();
        EXPECT_TRUE(ev[0] > 0.0 && ev[1] > 0.0);
    }
    std::cout << "Assignment from std::array (A)" << std::endl;
    std::cout << m_arr << std::endl;
    std::cout << std::endl;

    // assignment from C-array
    Scalar Ac[3] = {2.0, 1.0, 2.0};
    m_arr = Ac;
    {
        EigenDecomposition<decltype(m_arr)> evd(m_arr);
        auto ev = evd.eigenvalues();
        EXPECT_TRUE(ev[0] > 0.0 && ev[1] > 0.0);
    }
    std::cout << "Assignment from C-array (A)" << std::endl;
    std::cout << m_arr << std::endl;
    std::cout << std::endl;

    // assignment from std::vector
    std::vector<Scalar> Avec = {2.0, 1.0, 2.0};
    SPDMatrixView<Scalar, 2> m_vec(data);
    m_vec = Avec;
    {
        EigenDecomposition<decltype(m_vec)> evd(m_vec);
        auto ev = evd.eigenvalues();
        EXPECT_TRUE(ev[0] > 0.0 && ev[1] > 0.0);
    }
    std::cout << "Assignment from std::vector (A)" << std::endl;
    std::cout << m_vec << std::endl;
    std::cout << std::endl;

    // Callable constructor/assignment (scaled identity 2*I)
    auto callable = [](){
        return std::array<double, 3>{2.0, 0.0, 2.0};
    };
    SPDMatrixView<Scalar, 2> m_callable(data_new);
    m_callable = callable;
    {
        EigenDecomposition<decltype(m_callable)> evd(m_callable);
        auto ev = evd.eigenvalues();
        EXPECT_TRUE(almost_equal(ev[0], 2.0) && almost_equal(ev[1], 2.0));
    }
    std::cout << "Callable assignment (2*I)" << std::endl;
    std::cout << m_callable << std::endl;
    std::cout << std::endl;

    // Copy and assignment from another SPDMatrixView
    {
        SPDMatrixView<Scalar, 2> m_copy(data);
        m_copy = m_arr;
        EigenDecomposition<decltype(m_copy)> evd(m_copy);
        auto ev = evd.eigenvalues();
        EXPECT_TRUE(ev[0] > 0.0 && ev[1] > 0.0);
        std::cout << "Assign from another SPDMatrixView" << std::endl;
        std::cout << m_copy << std::endl;
        std::cout << std::endl;
    }

    // Copy and assignment from a MatrixBase/SquareMatrixBase expression
    {
        Matrix<Scalar, 2, 2> M({2.0, 1.0,
                                1.0, 2.0}); // row-major ctor for the dense Matrix
        SPDMatrixView<Scalar, 2> m_assign(data);
        m_assign = M; // check() runs inside view's operator=
        EigenDecomposition<decltype(m_assign)> evd(m_assign);
        auto ev = evd.eigenvalues();
        EXPECT_TRUE(ev[0] > 0.0 && ev[1] > 0.0);
        std::cout << "Assign from Matrix expression (A)" << std::endl;
        std::cout << m_assign << std::endl;
        std::cout << std::endl;
    }

    #ifdef __FDAPDE_HAS_EIGEN__
        // Eigen assignment
        Scalar raw_eigen[3] = {0};
        Eigen::Matrix<Scalar, 2, 2> emat;
        emat << 2.0, 1.0,
                1.0, 2.0;
        SPDMatrixView<Scalar, 2> m_eigen_assign(raw_eigen);
        m_eigen_assign = emat;
        {
            EigenDecomposition<decltype(m_eigen_assign)> evd(m_eigen_assign);
            auto ev = evd.eigenvalues();
            EXPECT_TRUE(ev[0] > 0.0 && ev[1] > 0.0);
        }
        std::cout << "Eigen assignment (A)" << std::endl;
        std::cout << m_eigen_assign << std::endl;
        std::cout << std::endl;
    #endif
}

TEST(matrix_test, SPDMatrix) {

    using Scalar = double;

    // Default constructor -> Identity
    SPDMatrix<Scalar, 3> m_default;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            EXPECT_TRUE(m_default(i,j) == (i == j ? Scalar(1) : Scalar(0)));
    {
        auto ev = m_default.eigenvalues();
        EXPECT_TRUE(almost_equal(ev[0], 1.0) && almost_equal(ev[1], 1.0) && almost_equal(ev[2], 1.0));
        auto Q = m_default.eigenvectors();
        EXPECT_TRUE(almost_equal(Q.transpose() * Q, DiagonalMatrix<Scalar, 3>::Identity()));
    }
    std::cout << "Default (Identity)" << std::endl;
    std::cout << m_default << std::endl;
    std::cout << std::endl;

    // Constructor from std::array (3x3 SPD tridiagonal)
    // A = [[3,1,0],[1,3,1],[0,1,3]]
    // packed (upper): (0,0)=3,(0,1)=1,(0,2)=0,(1,1)=3,(1,2)=1,(2,2)=3
    std::array<Scalar, 6> Aarr = {3.0, 1.0, 0.0, 3.0, 1.0, 3.0};
    SPDMatrix<Scalar, 3> m_arr(Aarr);
    {
        auto ev = m_arr.eigenvalues();
        EXPECT_TRUE(ev[0] > 0.0 && ev[1] > 0.0 && ev[2] > 0.0);
        auto Q = m_arr.eigenvectors();
        EXPECT_TRUE(almost_equal(Q.transpose() * Q, DiagonalMatrix<Scalar, 3>::Identity()));
    }
    std::cout << "Constructor from std::array (SPD tridiagonal)" << std::endl;
    std::cout << m_arr << std::endl;
    std::cout << std::endl;

    // Constructor from std::vector
    std::vector<Scalar> Avec = {3.0, 1.0, 0.0, 3.0, 1.0, 3.0};
    SPDMatrix<Scalar, 3> m_vec(Avec);
    {
        auto ev = m_vec.eigenvalues();
        EXPECT_TRUE(ev[0] > 0.0 && ev[1] > 0.0 && ev[2] > 0.0);
    }
    std::cout << "Constructor from std::vector (SPD tridiagonal)" << std::endl;
    std::cout << m_vec << std::endl;
    std::cout << std::endl;

    // Callable constructor (2*I)
    SPDMatrix<Scalar, 3> m_callable([]() {
        return std::array<Scalar, 6>{2.0, 0.0, 0.0, 2.0, 0.0, 2.0};
    });
    {
        auto ev = m_callable.eigenvalues();
        EXPECT_TRUE(almost_equal(ev[0], 2.0) && almost_equal(ev[1], 2.0) && almost_equal(ev[2], 2.0));
    }
    std::cout << "Callable constructor (2*I)" << std::endl;
    std::cout << m_callable << std::endl;
    std::cout << std::endl;

    // Copy and assignment from another SPDMatrix
    {
        SPDMatrix<Scalar, 3> m_copy(m_arr);
        auto ev = m_copy.eigenvalues();
        EXPECT_TRUE(ev[0] > 0.0 && ev[1] > 0.0 && ev[2] > 0.0);
        std::cout << "Copy from another SPDMatrix" << std::endl;
        std::cout << m_copy << std::endl;
        std::cout << std::endl;

        SPDMatrix<Scalar, 3> m_assign;
        m_assign = m_vec;
        auto ev2 = m_assign.eigenvalues();
        EXPECT_TRUE(ev2[0] > 0.0 && ev2[1] > 0.0 && ev2[2] > 0.0);
        std::cout << "Assign from another SPDMatrix" << std::endl;
        std::cout << m_assign << std::endl;
        std::cout << std::endl;
    }

    // Copy and assignment from a MatrixBase expression (must be SPD)
    {
        Matrix<double, 3, 3> A({3.0, 1.0, 0.0,
                                1.0, 3.0, 1.0,
                                0.0, 1.0, 3.0}); // row-major ctor
        SPDMatrix<Scalar, 3> m_copy(A);
        auto ev = m_copy.eigenvalues();
        EXPECT_TRUE(ev[0] > 0.0 && ev[1] > 0.0 && ev[2] > 0.0);
        std::cout << "Copy from MatrixBase expression (SPD tridiagonal)" << std::endl;
        std::cout << m_copy << std::endl;
        std::cout << std::endl;

        SPDMatrix<Scalar, 3> m_assign;
        m_assign = A;
        auto ev2 = m_assign.eigenvalues();
        EXPECT_TRUE(ev2[0] > 0.0 && ev2[1] > 0.0 && ev2[2] > 0.0);
        std::cout << "Assign from MatrixBase expression (SPD tridiagonal)" << std::endl;
        std::cout << m_assign << std::endl;
        std::cout << std::endl;
    }

    #ifdef __FDAPDE_HAS_EIGEN__
        // Eigen constructor and assignment using L^T L (SPD by construction)
        Eigen::Matrix<Scalar, 3, 3> L;
        L << 1.0, 0.0, 0.0,
             1.0, 1.0, 0.0,
             0.0, 1.0, 1.0;
        Eigen::Matrix<Scalar, 3, 3> emat = L.transpose() * L;

        SPDMatrix<Scalar, 3> m_eigen(emat);
        {
            auto ev = m_eigen.eigenvalues();
            EXPECT_TRUE(ev[0] > 0.0 && ev[1] > 0.0 && ev[2] > 0.0);
        }
        std::cout << "Eigen constructor (L^T L)" << std::endl;
        std::cout << m_eigen << std::endl;
        std::cout << std::endl;

        SPDMatrix<Scalar, 3> m_eigen_assign;
        m_eigen_assign = emat;
        {
            auto ev = m_eigen_assign.eigenvalues();
            EXPECT_TRUE(ev[0] > 0.0 && ev[1] > 0.0 && ev[2] > 0.0);
        }
        std::cout << "Eigen assignment (L^T L)" << std::endl;
        std::cout << m_eigen_assign << std::endl;
        std::cout << std::endl;
    #endif
}

TEST(matrix_test, SPSDMatrixView) {

    using Scalar = double;

    // data
    constexpr int n = 2;
    Scalar data[ n * (n + 1) / 2 ]     = {0};
    Scalar data_new[ n * (n + 1) / 2 ] = {0};

    // A 2x2 SPSD matrix B = [[1, 1], [1, 1]] with eigenvalues {2, 0}
    std::array<Scalar, 3> Barr = {1.0, 1.0, 1.0};

    // Construct a view on raw memory and assign SPSD
    SPSDMatrixView<Scalar, 2> m_raw(data);
    m_raw = Barr; // check() runs inside operator=
    {
        EigenDecomposition<decltype(m_raw)> evd(m_raw);
        auto ev = evd.eigenvalues();
        EXPECT_TRUE(ev[0] >= 0.0 && ev[1] >= 0.0);
        EXPECT_TRUE(almost_equal(std::min(ev[0], ev[1]), 0.0));
        auto Q = evd.eigenvectors();
        EXPECT_TRUE(almost_equal(Q.transpose() * Q, DiagonalMatrix<Scalar, 2>::Identity()));
    }
    std::cout << "SPSDMatrixView (B)" << std::endl;
    std::cout << m_raw << std::endl;
    std::cout << std::endl;

    // assignment from std::array
    SPSDMatrixView<Scalar, 2> m_arr(data_new);
    m_arr = Barr;
    EXPECT_TRUE(almost_equal(m_arr(0,0), 1.0));
    EXPECT_TRUE(almost_equal(m_arr(0,1), 1.0));
    EXPECT_TRUE(almost_equal(m_arr(1,0), 1.0));
    EXPECT_TRUE(almost_equal(m_arr(1,1), 1.0));
    {
        EigenDecomposition<decltype(m_arr)> evd(m_arr);
        auto ev = evd.eigenvalues();
        EXPECT_TRUE(ev[0] >= 0.0 && ev[1] >= 0.0);
    }
    std::cout << "Assignment from std::array (B)" << std::endl;
    std::cout << m_arr << std::endl;
    std::cout << std::endl;

    // assignment from C-array
    Scalar Bc[3] = {1.0, 1.0, 1.0};
    m_arr = Bc;
    {
        EigenDecomposition<decltype(m_arr)> evd(m_arr);
        auto ev = evd.eigenvalues();
        EXPECT_TRUE(ev[0] >= 0.0 && ev[1] >= 0.0);
        EXPECT_TRUE(almost_equal(std::min(ev[0], ev[1]), 0.0));
    }
    std::cout << "Assignment from C-array (B)" << std::endl;
    std::cout << m_arr << std::endl;
    std::cout << std::endl;

    // assignment from std::vector
    std::vector<Scalar> Bvec = {1.0, 1.0, 1.0};
    SPSDMatrixView<Scalar, 2> m_vec(data);
    m_vec = Bvec;
    {
        EigenDecomposition<decltype(m_vec)> evd(m_vec);
        auto ev = evd.eigenvalues();
        EXPECT_TRUE(ev[0] >= 0.0 && ev[1] >= 0.0);
        EXPECT_TRUE(almost_equal(std::min(ev[0], ev[1]), 0.0));
    }
    std::cout << "Assignment from std::vector (B)" << std::endl;
    std::cout << m_vec << std::endl;
    std::cout << std::endl;

    // Callable assignment (rank-1 projector: u u^T with u=[1,0]^T)
    auto callable = [](){
        return std::array<double, 3>{1.0, 0.0, 0.0}; // [[1,0],[0,0]]
    };
    SPSDMatrixView<Scalar, 2> m_callable(data_new);
    m_callable = callable;
    {
        EigenDecomposition<decltype(m_callable)> evd(m_callable);
        auto ev = evd.eigenvalues();
        EXPECT_TRUE(ev[0] >= 0.0 && ev[1] >= 0.0);
        EXPECT_TRUE(almost_equal(std::min(ev[0], ev[1]), 0.0));
    }
    std::cout << "Callable assignment (rank-1 projector)" << std::endl;
    std::cout << m_callable << std::endl;
    std::cout << std::endl;

    // Copy and assignment from another SPSDMatrixView
    {
        SPSDMatrixView<Scalar, 2> m_copy(data);
        m_copy = m_arr;
        EigenDecomposition<decltype(m_copy)> evd(m_copy);
        auto ev = evd.eigenvalues();
        EXPECT_TRUE(ev[0] >= 0.0 && ev[1] >= 0.0);
        std::cout << "Assign from another SPSDMatrixView" << std::endl;
        std::cout << m_copy << std::endl;
        std::cout << std::endl;
    }

    // Copy and assignment from a MatrixBase/SquareMatrixBase expression
    {
        Matrix<Scalar, 2, 2> B({1.0, 1.0,
                                1.0, 1.0}); // row-major ctor
        SPSDMatrixView<Scalar, 2> m_assign(data);
        m_assign = B; // check() runs inside view's operator=
        EigenDecomposition<decltype(m_assign)> evd(m_assign);
        auto ev = evd.eigenvalues();
        EXPECT_TRUE(ev[0] >= 0.0 && ev[1] >= 0.0);
        EXPECT_TRUE(almost_equal(std::min(ev[0], ev[1]), 0.0));
        std::cout << "Assign from Matrix expression (B)" << std::endl;
        std::cout << m_assign << std::endl;
        std::cout << std::endl;
    }

    #ifdef __FDAPDE_HAS_EIGEN__
        // Eigen assignment
        Scalar raw_eigen[3] = {0};
        Eigen::Matrix<Scalar, 2, 2> emat;
        emat << 1.0, 1.0,
                1.0, 1.0;
        SPSDMatrixView<Scalar, 2> m_eigen_assign(raw_eigen);
        m_eigen_assign = emat;
        {
            EigenDecomposition<decltype(m_eigen_assign)> evd(m_eigen_assign);
            auto ev = evd.eigenvalues();
            EXPECT_TRUE(ev[0] >= 0.0 && ev[1] >= 0.0);
            EXPECT_TRUE(almost_equal(std::min(ev[0], ev[1]), 0.0));
        }
        std::cout << "Eigen assignment (B)" << std::endl;
        std::cout << m_eigen_assign << std::endl;
        std::cout << std::endl;
    #endif
}

TEST(matrix_test, SPSDMatrix) {

    using Scalar = double;

    // Default constructor -> Identity (valid SPSD as well)
    SPSDMatrix<Scalar, 3> m_default;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            EXPECT_TRUE(m_default(i,j) == (i == j ? Scalar(1) : Scalar(0)));
    {
        auto ev = m_default.eigenvalues();
        EXPECT_TRUE(ev[0] >= 0.0 && ev[1] >= 0.0 && ev[2] >= 0.0);
        auto Q = m_default.eigenvectors();
        EXPECT_TRUE(almost_equal(Q.transpose() * Q, DiagonalMatrix<Scalar, 3>::Identity()));
    }
    std::cout << "Default (Identity)" << std::endl;
    std::cout << m_default << std::endl;
    std::cout << std::endl;

    // Constructor from std::array (rank-deficient SPSD)
    // C = [[1,1,0],[1,1,0],[0,0,0]]
    std::array<Scalar, 6> Carr = {1.0, 1.0, 0.0, 1.0, 0.0, 0.0};
    SPSDMatrix<Scalar, 3> m_arr(Carr);
    {
        auto ev = m_arr.eigenvalues();
        EXPECT_TRUE(ev[0] >= 0.0 && ev[1] >= 0.0 && ev[2] >= 0.0);
        EXPECT_TRUE(almost_equal(std::min({ev[0], ev[1], ev[2]}), 0.0));
    }
    std::cout << "Constructor from std::array (rank-deficient SPSD)" << std::endl;
    std::cout << m_arr << std::endl;
    std::cout << std::endl;

    // Constructor from std::vector
    std::vector<Scalar> Cvec = {1.0, 1.0, 0.0, 1.0, 0.0, 0.0};
    SPSDMatrix<Scalar, 3> m_vec(Cvec);
    {
        auto ev = m_vec.eigenvalues();
        EXPECT_TRUE(ev[0] >= 0.0 && ev[1] >= 0.0 && ev[2] >= 0.0);
        EXPECT_TRUE(almost_equal(std::min({ev[0], ev[1], ev[2]}), 0.0));
    }
    std::cout << "Constructor from std::vector (rank-deficient SPSD)" << std::endl;
    std::cout << m_vec << std::endl;
    std::cout << std::endl;

    // Callable constructor (diag{1,0,0})
    SPSDMatrix<Scalar, 3> m_callable([]() {
        return std::array<Scalar, 6>{1.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    });
    {
        auto ev = m_callable.eigenvalues();
        EXPECT_TRUE(ev[0] >= 0.0 && ev[1] >= 0.0 && ev[2] >= 0.0);
        EXPECT_TRUE(almost_equal(std::min({ev[0], ev[1], ev[2]}), 0.0));
    }
    std::cout << "Callable constructor (diag{1,0,0})" << std::endl;
    std::cout << m_callable << std::endl;
    std::cout << std::endl;

    // Copy and assignment from another SPSDMatrix
    {
        SPSDMatrix<Scalar, 3> m_copy(m_arr);
        auto ev = m_copy.eigenvalues();
        EXPECT_TRUE(ev[0] >= 0.0 && ev[1] >= 0.0 && ev[2] >= 0.0);
        std::cout << "Copy from another SPSDMatrix" << std::endl;
        std::cout << m_copy << std::endl;
        std::cout << std::endl;

        SPSDMatrix<Scalar, 3> m_assign;
        m_assign = m_vec;
        auto ev2 = m_assign.eigenvalues();
        EXPECT_TRUE(ev2[0] >= 0.0 && ev2[1] >= 0.0 && ev2[2] >= 0.0);
        std::cout << "Assign from another SPSDMatrix" << std::endl;
        std::cout << m_assign << std::endl;
        std::cout << std::endl;
    }

    // Copy and assignment from a MatrixBase expression (must be SPSD)
    {
        Matrix<double, 3, 3> C({1.0, 1.0, 0.0,
                                1.0, 1.0, 0.0,
                                0.0, 0.0, 0.0}); // row-major ctor
        SPSDMatrix<Scalar, 3> m_copy(C);
        auto ev = m_copy.eigenvalues();
        EXPECT_TRUE(ev[0] >= 0.0 && ev[1] >= 0.0 && ev[2] >= 0.0);
        EXPECT_TRUE(almost_equal(std::min({ev[0], ev[1], ev[2]}), 0.0));
        std::cout << "Copy from MatrixBase expression (rank-deficient SPSD)" << std::endl;
        std::cout << m_copy << std::endl;
        std::cout << std::endl;

        SPSDMatrix<Scalar, 3> m_assign;
        m_assign = C;
        auto ev2 = m_assign.eigenvalues();
        EXPECT_TRUE(ev2[0] >= 0.0 && ev2[1] >= 0.0 && ev2[2] >= 0.0);
        EXPECT_TRUE(almost_equal(std::min({ev2[0], ev2[1], ev2[2]}), 0.0));
        std::cout << "Assign from MatrixBase expression (rank-deficient SPSD)" << std::endl;
        std::cout << m_assign << std::endl;
        std::cout << std::endl;
    }

    #ifdef __FDAPDE_HAS_EIGEN__
        // Eigen constructor and assignment using L^T L with rank-deficient L (SPSD)
        Eigen::Matrix<Scalar, 3, 2> L; // rank 2 -> semidefinite in 3x3
        L << 1.0, 0.0,
             0.0, 1.0,
             0.0, 0.0;
        Eigen::Matrix<Scalar, 3, 3> emat = L * L.transpose(); // diag{1,1,0}

        SPSDMatrix<Scalar, 3> m_eigen(emat);
        {
            auto ev = m_eigen.eigenvalues();
            EXPECT_TRUE(ev[0] >= 0.0 && ev[1] >= 0.0 && ev[2] >= 0.0);
            EXPECT_TRUE(almost_equal(std::min({ev[0], ev[1], ev[2]}), 0.0));
        }
        std::cout << "Eigen constructor (L L^T, rank-deficient)" << std::endl;
        std::cout << m_eigen << std::endl;
        std::cout << std::endl;

        SPSDMatrix<Scalar, 3> m_eigen_assign;
        m_eigen_assign = emat;
        {
            auto ev = m_eigen_assign.eigenvalues();
            EXPECT_TRUE(ev[0] >= 0.0 && ev[1] >= 0.0 && ev[2] >= 0.0);
            EXPECT_TRUE(almost_equal(std::min({ev[0], ev[1], ev[2]}), 0.0));
        }
        std::cout << "Eigen assignment (L L^T, rank-deficient)" << std::endl;
        std::cout << m_eigen_assign << std::endl;
        std::cout << std::endl;
    #endif
}