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

// compile from the repository root with c++ -std=c++20 -O2 -pthread -I.
// examples/linear_algebra/04_symmetric_and_skew_storage.cpp -o /tmp/04_symmetric_and_skew_storage
#include <fdaPDE/dense_linear_algebra.h>

#include <array>
#include <iostream>

using namespace fdapde;

// represent symmetry explicitly, then split a general operator into its symmetric and skew parts
int main() {
    // 1. construct a symmetric matrix from its six independent lower-triangle coordinates
    // packed coordinates follow rows: (s00,s10,s11,s20,s21,s22)
    const Vector<double, 6> coordinates({2., .5, 3., -.25, .75, 4.});
    SymmetricMatrix<double, 3> symmetric(coordinates);
    std::cout << "1. Six coordinates describe this 3 x 3 symmetric matrix:\n" << symmetric << '\n';

    // 2. edit through a borrowed view and scale the matrix while preserving symmetry
    // either triangle addresses the same independent entry
    auto symmetric_view = symmetric.view();
    symmetric_view(0, 2) = 1.;
    const SymmetricMatrix<double, Dynamic> dynamic_symmetric(symmetric);
    const SymmetricMatrix<double, 3> scaled(2. * dynamic_symmetric);
    std::cout << "2. After writing entry (0,2), entry (2,0) is " << symmetric(2, 0)
              << "\nScaling retains symmetric storage:\n"
              << scaled << '\n';

    // 3. construct and edit a skew-symmetric matrix through external packed storage
    // coordinates store the strict upper triangle: (k01,k02,k12); the diagonal is zero
    const SkewSymmetricMatrix<double, 3> skew({.2, -.1, .3});
    std::array<double, 3> packed {};
    SkewSymmetricMatrixView<double, 3> skew_view(packed.data());
    skew_view = skew;
    skew_view(2, 0) = .4;
    const SkewSymmetricMatrix<double, Dynamic> dynamic_skew(skew_view);
    std::cout << "3. Writing (2,0) also changes the sign at (0,2):\n" << dynamic_skew << '\n';

    // 4. interpret one selected triangle as symmetric or skew-symmetric data
    // these adaptors use only the selected half of the original matrix
    const Matrix<double, 3, 3> dense({2., 8., 9., .5, 3., 7., 1., .75, 4.});
    const SymmetricMatrix<double, 3> lower_selected(dense.as_symmetric<Lower>());
    const SkewSymmetricMatrix<double, 3> upper_selected(dense.as_skew_symmetric<Upper>());
    std::cout << "4. General matrix:\n"
              << dense << "\nSymmetry from its lower triangle:\n"
              << lower_selected << "\nSkew symmetry from its upper triangle:\n"
              << upper_selected << '\n';

    // 5. split a general matrix using both halves: S=(A+transpose(A))/2 and K=(A-transpose(A))/2
    const Matrix<double, 3, 3> symmetric_part(dense.symm_part());
    const Matrix<double, 3, 3> skew_part(dense.skew_part());
    std::cout << "5. Symmetric part S:\n" << symmetric_part << "\nSkew part K:\n" << skew_part << '\n';

    // 6. multiply structured operands into a dense owner, since the product can be general
    const Matrix<double, 3, 3> product(symmetric * dynamic_skew);
    std::cout << "6. Product of a symmetric and a skew matrix:\n" << product << '\n';
}
