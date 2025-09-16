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

TEST(matrix_test, RotationOp) {

    using Scalar = double;

    // Identity in 2D
    RotationOp<Scalar, 2> R2_id(0.0);

    // Basic access
    EXPECT_TRUE(almost_equal(R2_id(0,0), 1.0) && almost_equal(R2_id(0,1), 0.0));
    EXPECT_TRUE(almost_equal(R2_id(1,0), 0.0) && almost_equal(R2_id(1,1), 1.0));

    std::cout << "RotationOp<Scalar, 2> Identity" << std::endl;
    std::cout << R2_id << std::endl;
    std::cout << std::endl;

    // Identity in 3D
    RotationOp<Scalar, 3> R3_id(0.0, 0.0, 0.0);

    // Basic access
    EXPECT_TRUE(almost_equal(R3_id(0,0),1.0) && almost_equal(R3_id(0,1),0.0) && almost_equal(R3_id(0,2),0.0));
    EXPECT_TRUE(almost_equal(R3_id(1,0),0.0) && almost_equal(R3_id(1,1),1.0) && almost_equal(R3_id(1,2),0.0));
    EXPECT_TRUE(almost_equal(R3_id(2,0),0.0) && almost_equal(R3_id(2,1),0.0) && almost_equal(R3_id(2,2),1.0));

    std::cout << "RotationOp<Scalar, 3> Identity" << std::endl;
    std::cout << R3_id << std::endl;
    std::cout << std::endl;

    std::cout << "as_matrix()" << std::endl;
    std::cout << R2_id.as_matrix() << std::endl;
    std::cout << R3_id.as_matrix() << std::endl;
    std::cout << std::endl;

    // Named constructors (3D) on basis vectors
    const double half_pi = 0.5 * std::numbers::pi;

    // Rz(+90°): e_x -> e_y
    auto Rz90 = RotationOp<Scalar, 3>::Rz(half_pi);
    Matrix<double,3,1> ex({1.0,0.0,0.0});
    auto Rz_ex = Rz90 * ex;
    EXPECT_TRUE(almost_equal(Rz_ex(0,0), 0.0) && almost_equal(Rz_ex(1,0), 1.0) && almost_equal(Rz_ex(2,0), 0.0));
    std::cout << "Rz(+90°)" << std::endl;
    std::cout << Rz90 << std::endl;
    std::cout << std::endl;

    // Rx(+90°): e_y -> e_z
    auto Rx90 = RotationOp<Scalar, 3>::Rx(half_pi);
    Matrix<double,3,1> ey({0.0,1.0,0.0});
    auto Rx_ey = Rx90 * ey;
    EXPECT_TRUE(almost_equal(Rx_ey(0,0), 0.0) && almost_equal(Rx_ey(1,0), 0.0) && almost_equal(Rx_ey(2,0), 1.0));
    std::cout << "Rx(+90°)" << std::endl;
    std::cout << Rx90 << std::endl;
    std::cout << std::endl;

    // Ry(+90°): e_z -> e_x
    auto Ry90 = RotationOp<Scalar, 3>::Ry(half_pi);
    Matrix<double,3,1> ez({0.0,0.0,1.0});
    auto Ry_ez = Ry90 * ez;
    EXPECT_TRUE(almost_equal(Ry_ez(0,0), 1.0) && almost_equal(Ry_ez(1,0), 0.0) && almost_equal(Ry_ez(2,0), 0.0));
    std::cout << "Ry(+90°)" << std::endl;
    std::cout << Ry90 << std::endl;
    std::cout << std::endl;

    std::cout << "RotationOp<Scalar, 3> named constructors (Rx, Ry, Rz) checked on basis vectors." << std::endl;
    std::cout << std::endl;
}

TEST(matrix_test, RotationOpAlgebra) {

    using Scalar = double;

    // Dense 3x3 with distinct entries
    Matrix<Scalar,3,3> M3({1,2,3, 4,5,6, 7,8,9});
    std::cout << "M3" << std::endl;
    std::cout << M3 << std::endl;
    std::cout << std::endl;

    // 3D: Left/Right multiply with Rz(+90°)
    const double half_pi = 0.5 * std::numbers::pi;
    auto Rz90 = RotationOp<Scalar, 3>::Rz(half_pi);

    // Left-multiply
    auto L = Rz90 * M3;
    Matrix<Scalar,3,3> L_expected;
    for (int j = 0; j < 3; ++j) {
        L_expected(0,j) = -M3(1,j);
        L_expected(1,j) =  M3(0,j);
        L_expected(2,j) =  M3(2,j);
    }
    EXPECT_TRUE(almost_equal(L, L_expected));
    std::cout << "Rz(90°) * M3" << std::endl;
    std::cout << L << std::endl;
    std::cout << std::endl;

    // Right-multiply
    auto R = M3 * Rz90;
    Matrix<Scalar,3,3> R_expected;
    for (int i = 0; i < 3; ++i) {
        R_expected(i,0) =  M3(i,1);
        R_expected(i,1) = -M3(i,0);
        R_expected(i,2) =  M3(i,2);
    }
    EXPECT_TRUE(almost_equal(R, R_expected));
    std::cout << "M3 * Rz(90°)" << std::endl;
    std::cout << R << std::endl;
    std::cout << std::endl;

    // Identity rotation leaves M unchanged (both sides)
    RotationOp<Scalar, 3> R3_id(0.0, 0.0, 0.0);
    auto L_id = R3_id * M3;
    auto R_id = M3 * R3_id;
    EXPECT_TRUE(L_id == M3);
    EXPECT_TRUE(R_id == M3);
    std::cout << "R3_id * M3 and M3 * R3_id (identity)" << std::endl;
    std::cout << L_id << std::endl;
    std::cout << std::endl;

    // 2D: Algebra & composition
    Matrix<Scalar,2,2> M2({1,2, 3,4});
    std::cout << "M2" << std::endl;
    std::cout << M2 << std::endl;
    std::cout << std::endl;

    RotationOp<Scalar, 2> R2_id(0.0);
    auto L2_id = R2_id * M2;
    auto R2id  = M2 * R2_id;
    EXPECT_TRUE(L2_id == M2);
    EXPECT_TRUE(R2id  == M2);
    std::cout << "R2_id * M2 and M2 * R2_id (identity)" << std::endl;
    std::cout << L2_id << std::endl;
    std::cout << std::endl;

    // Composition in 2D: R(θ2) * (R(θ1) * M) == R(θ1+θ2) * M
    const double th1 = std::numbers::pi / 6.0;   // 30 deg
    const double th2 = -std::numbers::pi / 3.0;  // -60 deg
    RotationOp<Scalar, 2> R2_1(th1);
    RotationOp<Scalar, 2> R2_2(th2);
    auto left_seq_2d = R2_2 * (R2_1 * M2);
    RotationOp<Scalar, 2> R2_12(th1 + th2);
    auto left_comp_2d = R2_12 * M2;
    EXPECT_TRUE(almost_equal(left_seq_2d, left_comp_2d));

    std::cout << "2D Composition: R(th2) * (R(th1) * M2) == R(th1+th2) * M2" << std::endl;
    std::cout << left_seq_2d << std::endl;
    std::cout << std::endl;

    // 3D: Composition via as_matrix()
    auto R1 = RotationOp<Scalar, 3>::Rz(std::numbers::pi/3.0);
    auto R2 = RotationOp<Scalar, 3>::Ry(std::numbers::pi/4.0);

    auto left_seq_3d = R2 * (R1 * M3);
    auto R_comp_full = R2.as_matrix() * R1.as_matrix();
    auto left_comp_3d = R_comp_full * M3;

    EXPECT_TRUE(almost_equal(left_seq_3d, left_comp_3d));

    std::cout << "3D Composition (left): R2*(R1*M3) == (R2*R1)*M3 using as_matrix()" << std::endl;
    std::cout << left_seq_3d << std::endl;
    std::cout << std::endl;

    // Right-side composition: (M3*R1)*R2 == M3*(R1*R2)
    auto right_seq_3d = (M3 * R1) * R2;
    auto right_comp_3d = M3 * (R1.as_matrix() * R2.as_matrix());
    EXPECT_TRUE(almost_equal(right_seq_3d, right_comp_3d));

    std::cout << "3D Composition (right): (M3*R1)*R2 == M3*(R1*R2) using as_matrix()" << std::endl;
    std::cout << right_seq_3d << std::endl;
    std::cout << std::endl;
}