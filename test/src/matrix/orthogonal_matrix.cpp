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

TEST(matrix_test, OrthogonalMatrixView) {

    using Scalar = double;

    // data
    constexpr int n = 2;
    Scalar data[n * n]     = {0};
    Scalar data_new[n * n] = {0};

    // A 2x2 rotation R(theta) = [[c, -s], [s, c]]
    // NOTE: storage of OrthogonalMatrix is column-major (index(i,j)=i+j*N),
    // so the flat array for R is {c, s, -s, c}.
    const Scalar theta = 0.37;
    const Scalar c = std::cos(theta);
    const Scalar s = std::sin(theta);

    std::array<Scalar, 4> Rarr = {c, s, -s, c};

    // Construct a view on raw memory and assign a valid orthogonal matrix
    OrthogonalMatrixView<Scalar, 2> m_raw(data);
    m_raw = Rarr; // check() runs inside operator=
    assert(almost_equal(m_raw.transpose() * m_raw, DiagonalMatrix<Scalar, 2>::Identity()));
    std::cout << "OrthogonalMatrixView (R)" << std::endl;
    std::cout << m_raw << std::endl;
    std::cout << std::endl;

    // assignment from std::array
    OrthogonalMatrixView<Scalar, 2> m_arr(data_new);
    m_arr = Rarr;
    assert(m_arr(0,0) == c);
    assert(m_arr(0,1) == -s);
    assert(m_arr(1,0) == s);
    assert(m_arr(1,1) == c);
    std::cout << "Assignment from std::array (R)" << std::endl;
    std::cout << m_arr << std::endl;
    std::cout << std::endl;

    // assignment from C-array
    Scalar Rc[4] = {c, s, -s, c};
    m_arr = Rc;
    assert(almost_equal(m_arr.transpose() * m_arr, DiagonalMatrix<Scalar, 2>::Identity()));
    std::cout << "Assignment from C-array (R)" << std::endl;
    std::cout << m_arr << std::endl;
    std::cout << std::endl;

    // assignment from std::vector
    std::vector<Scalar> Rvec = {c, s, -s, c};
    OrthogonalMatrixView<Scalar, 2> m_vec(data);
    m_vec = Rvec;
    assert(almost_equal(m_vec.transpose() * m_vec, DiagonalMatrix<Scalar, 2>::Identity()));
    std::cout << "Assignment from std::vector (R)" << std::endl;
    std::cout << m_vec << std::endl;
    std::cout << std::endl;

    // Callable constructor/assignment (rotation with a different angle)
    auto callable = [](){
        double t = 0.9;
        double cc = std::cos(t), ss = std::sin(t);
        return std::array<double, 4>{cc, ss, -ss, cc};
    };
    OrthogonalMatrixView<Scalar, 2> m_callable(data_new);
    m_callable = callable;
    assert(almost_equal(m_callable.transpose() * m_callable, DiagonalMatrix<Scalar, 2>::Identity()));
    std::cout << "Callable assignment (R(0.9))" << std::endl;
    std::cout << m_callable << std::endl;
    std::cout << std::endl;

    // Copy and assignment from another OrthogonalMatrixView
    {
        OrthogonalMatrixView<Scalar, 2> m_copy(data);
        m_copy = m_arr;
        assert(almost_equal(m_copy.transpose() * m_copy, DiagonalMatrix<Scalar, 2>::Identity()));
        std::cout << "Assign from another OrthogonalMatrixView" << std::endl;
        std::cout << m_copy << std::endl;
        std::cout << std::endl;
    }

    // Copy and assignment from a MatrixBase/SquareMatrixBase expression
    {
        Matrix<Scalar, 2, 2> M({c, -s, s, c}); // row-major ctor for the dense Matrix
        OrthogonalMatrix<Scalar, 2> m_assign(data);
        m_assign = M; // check() runs inside view's operator=
        assert(almost_equal(m_assign.transpose() * m_assign, DiagonalMatrix<Scalar, 2>::Identity()));
        std::cout << "Assign from Matrix expression (R)" << std::endl;
        std::cout << m_assign << std::endl;
        std::cout << std::endl;
    }

    #ifdef __FDAPDE_HAS_EIGEN__
        // Eigen assignment
        Scalar raw_eigen[4] = {0};
        Eigen::Matrix<Scalar, 2, 2> emat;
        emat << c, -s, s, c; // row-major stream creates the same rotation
        OrthogonalMatrixView<Scalar, 2> m_eigen_assign(raw_eigen);
        m_eigen_assign = emat;
        assert(almost_equal(m_eigen_assign.transpose() * m_eigen_assign, DiagonalMatrix<Scalar, 2>::Identity()));
        std::cout << "Eigen assignment (R)" << std::endl;
        std::cout << m_eigen_assign << std::endl;
        std::cout << std::endl;

        // Eigen map view (no copy)
        auto emap = m_eigen_assign.as_eigen_map();
        assert(std::abs(emap(0,1) + s) < 1e-12);
    #endif
}

TEST(matrix_test, OrthogonalMatrix) {

    using Scalar = double;

    // Default constructor -> Identity
    OrthogonalMatrix<Scalar, 3> m_default;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            assert(m_default(i,j) == (i == j ? Scalar(1) : Scalar(0)));
    std::cout << "Default (Identity)" << std::endl;
    std::cout << m_default << std::endl;
    std::cout << std::endl;

    // Constructor from std::array (a 3x3 permutation matrix is orthogonal)
    // Columns: e1, e3, e2  -> column-major array {1,0,0, 0,0,1, 0,1,0}
    std::array<Scalar, 9> Parr = {1,0,0, 0,0,1, 0,1,0};
    OrthogonalMatrix<Scalar, 3> m_arr(Parr);
    assert(almost_equal(m_arr.transpose() * m_arr, DiagonalMatrix<Scalar, 3>::Identity()));
    std::cout << "Constructor from std::array (Permutation)" << std::endl;
    std::cout << m_arr << std::endl;
    std::cout << std::endl;

    // Constructor from std::vector
    std::vector<Scalar> Pvec = {1,0,0, 0,0,1, 0,1,0};
    OrthogonalMatrix<Scalar, 3> m_vec(Pvec);
    assert(almost_equal(m_vec.transpose() * m_vec, DiagonalMatrix<Scalar, 3>::Identity()));
    std::cout << "Constructor from std::vector (Permutation)" << std::endl;
    std::cout << m_vec << std::endl;
    std::cout << std::endl;

    // Callable constructor
    OrthogonalMatrix<Scalar, 3> m_callable([]() {
        return std::array<Scalar, 9>{1,0,0, 0,1,0, 0,0,1}; // Identity
    });
    assert(almost_equal(m_callable.transpose() * m_callable, DiagonalMatrix<Scalar, 3>::Identity()));
    std::cout << "Callable constructor (Identity)" << std::endl;
    std::cout << m_callable << std::endl;
    std::cout << std::endl;

    // Copy and assignment from another OrthogonalMatrix
    {
        OrthogonalMatrix<Scalar, 3> m_copy(m_arr);
        assert(almost_equal(m_copy.transpose() * m_copy, DiagonalMatrix<Scalar, 3>::Identity()));
        std::cout << "Copy from another OrthogonalMatrix" << std::endl;
        std::cout << m_copy << std::endl;
        std::cout << std::endl;

        OrthogonalMatrix<Scalar, 3> m_assign;
        m_assign = m_vec;
        assert(almost_equal(m_assign.transpose() * m_assign, DiagonalMatrix<Scalar, 3>::Identity()));
        std::cout << "Assign from another OrthogonalMatrix" << std::endl;
        std::cout << m_assign << std::endl;
        std::cout << std::endl;
    }

    // Copy and assignment from a MatrixBase expression (must be orthogonal)
    {
        Matrix<double, 3, 3> P({1,0,0, 0,0,1, 0,1,0}); // same permutation, row-major ctor
        OrthogonalMatrix<Scalar, 3> m_copy(P);
        assert(almost_equal(m_copy.transpose() * m_copy, DiagonalMatrix<Scalar, 3>::Identity()));
        std::cout << "Copy from MatrixBase expression (Permutation)" << std::endl;
        std::cout << m_copy << std::endl;
        std::cout << std::endl;

        OrthogonalMatrix<Scalar, 3> m_assign;
        m_assign = P;
        assert(almost_equal(m_assign.transpose() * m_assign, DiagonalMatrix<Scalar, 3>::Identity()));
        std::cout << "Assign from MatrixBase expression (Permutation)" << std::endl;
        std::cout << m_assign << std::endl;
        std::cout << std::endl;
    }

    #ifdef __FDAPDE_HAS_EIGEN__
        // Eigen constructor and assignment (2D rotation embedded in 3D)
        Eigen::Matrix<Scalar, 3, 3> emat;
        Scalar t = 0.51, cc = std::cos(t), ss = std::sin(t);
        // rotation in the (x,y) plane
        emat << cc, -ss, 0,
                ss,  cc, 0,
                 0,   0, 1;
        OrthogonalMatrix<Scalar, 3> m_eigen(emat);
        assert(almost_equal(m_eigen.transpose() * m_eigen, DiagonalMatrix<Scalar, 3>::Identity()));
        std::cout << "Eigen constructor (planar rotation)" << std::endl;
        std::cout << m_eigen << std::endl;
        std::cout << std::endl;

        OrthogonalMatrix<Scalar, 3> m_eigen_assign;
        m_eigen_assign = emat;
        assert(almost_equal(m_eigen_assign.transpose() * m_eigen_assign, DiagonalMatrix<Scalar, 3>::Identity()));
        std::cout << "Eigen assignment (planar rotation)" << std::endl;
        std::cout << m_eigen_assign << std::endl;
        std::cout << std::endl;
    #endif
}