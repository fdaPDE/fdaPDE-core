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

TEST(matrix_test, PermutationMatrix) {

    // Identity permutation (0,1,2)
    std::array<int,3> id = {0,1,2};
    PermutationMatrix<3> P_id(id);

    // Basic access
    assert(P_id(0,0) == 1 && P_id(0,1) == 0);
    assert(P_id(1,1) == 1 && P_id(1,2) == 0);
    assert(P_id(2,2) == 1 && P_id(2,0) == 0);

    std::cout << "PermutationMatrix Identity" << std::endl;
    std::cout << P_id << std::endl;
    std::cout << std::endl;

    // Swap rows/cols 0 and 1: permutation = (1,0,2)
    std::array<int,3> swap01 = {1,0,2};
    PermutationMatrix<3> P_swap(swap01);

    // Check structure: ones at (i, permutation[i])
    assert(P_swap(0,1) == 1 && P_swap(0,0) == 0);
    assert(P_swap(1,0) == 1 && P_swap(1,1) == 0);
    assert(P_swap(2,2) == 1);

    std::cout << "PermutationMatrix swap(0,1)" << std::endl;
    std::cout << P_swap << std::endl;
    std::cout << std::endl;

    std::cout << "as_matrix" << std::endl;
    std::cout << P_id.as_matrix() << std::endl;
    std::cout << P_swap.as_matrix() << std::endl;
    std::cout << std::endl;


}

TEST(matrix_test, PermutationMatrixAlgebra) {

    using Scalar = double;

    // Dense 3x3 with distinct entries
    Matrix<Scalar,3,3> M({1,2,3, 4,5,6, 7,8,9});
    std::cout << "M" << std::endl;
    std::cout << M << std::endl;
    std::cout << std::endl;

    // Swap rows/cols 0 and 1
    PermutationMatrix<3> P({1,0,2});

    // Left-multiply: permute rows (row i <- old row permutation[i])
    auto PM = P * M;
    assert(PM(0,0) == 4 && PM(0,1) == 5 && PM(0,2) == 6); // row 0 <- old row 1
    assert(PM(1,0) == 1 && PM(1,1) == 2 && PM(1,2) == 3); // row 1 <- old row 0
    assert(PM(2,0) == 7 && PM(2,1) == 8 && PM(2,2) == 9); // row 2 unchanged
    std::cout << "P * M (row permutation)" << std::endl;
    std::cout << PM << std::endl;
    std::cout << std::endl;

    // Right-multiply: permute columns (col i <- old col permutation[i])
    auto MP = M * P;
    assert(MP(0,0) == 2 && MP(1,0) == 5 && MP(2,0) == 8); // col 0 <- old col 1
    assert(MP(0,1) == 1 && MP(1,1) == 4 && MP(2,1) == 7); // col 1 <- old col 0
    assert(MP(0,2) == 3 && MP(1,2) == 6 && MP(2,2) == 9); // col 2 unchanged
    std::cout << "M * P (column permutation)" << std::endl;
    std::cout << MP << std::endl;
    std::cout << std::endl;

    // Identity permutation leaves M unchanged (both sides)
    PermutationMatrix<3> P_id({0,1,2});
    auto PM_id = P_id * M;
    auto MP_id = M * P_id;
    assert(PM_id == M);
    assert(MP_id == M);
    std::cout << "P_id * M and M * P_id (identity)" << std::endl;
    std::cout << PM_id << std::endl;
    std::cout << std::endl;

    // Composition check for left multiplication:
    // P2 * (P1 * M) == P12 * M where P12[i] = P1[P2[i]]
    PermutationMatrix<3> P1({1,0,2}); // swap 0<->1
    PermutationMatrix<3> P2({2,1,0}); // reverse 0<->2
    auto left_seq = P2 * (P1 * M);

    std::array<int,3> P12 = { P1.permutation()[ P2.permutation()[0] ],
                              P1.permutation()[ P2.permutation()[1] ],
                              P1.permutation()[ P2.permutation()[2] ] };
    PermutationMatrix<3> P_comp(P12);
    auto left_comp = P_comp * M;


    assert(left_seq == left_comp);

    std::cout << "Composition (left): P2*(P1*M) == (P1∘P2)*M" << std::endl;
    std::cout << left_seq << std::endl;
    std::cout << std::endl;
}