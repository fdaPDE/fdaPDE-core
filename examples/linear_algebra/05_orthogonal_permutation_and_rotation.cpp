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
// examples/linear_algebra/05_orthogonal_permutation_and_rotation.cpp -o /tmp/05_orthogonal_permutation_and_rotation
#include <fdaPDE/dense_linear_algebra.h>

#include <iostream>
#include <vector>

using namespace fdapde;

// change an orthonormal basis, reorder observations and rotate points in the plane
int main() {
    // 1. build an orthonormal basis from independent columns and express a point in that basis
    // orthogonalize returns an independent orthogonal matrix; solve uses its transpose as the inverse
    const Matrix<double, 2, 2> directions({1., 1., 1., 0.});
    const auto basis = directions.orthogonalize();   // type: OrthogonalMatrix<double, 2>
    const Vector<double, 2> point {2., -1.};
    const auto coordinates = basis.solve(point);
    std::cout << "1. Input directions:\n"
              << directions << "\nOrthonormal basis:\n"
              << basis << "\nPoint in the original coordinates:\n"
              << point << "\nCoordinates in this basis, obtained with its transpose:\n"
              << coordinates << '\n';

    // 2. construct from already orthogonal data and reflect a point
    // orthogonal matrices can have determinant minus one, unlike proper rotations
    const OrthogonalMatrix<double, Dynamic> reflection(std::vector<double> {1., 0., 0., -1.});
    const Vector<double, 2> reflected(reflection * point);
    std::cout << "2. Reflection across the horizontal axis:\n"
              << reflected << "\nReflection determinant: " << reflection.determinant() << '\n';

    // 3. reorder rows by left multiplication and restore them with the inverse permutation
    const PermutationMatrix<3> permutation({2, 0, 1});
    const Matrix<double, 3, 2> observations({1., 2., 3., 4., 5., 6.});
    const Matrix<double, 3, 2> rows(permutation * observations);
    const Matrix<double, 3, 2> restored(permutation.inv() * rows);
    std::cout << "3. Observations:\n"
              << observations << "\nRows in order 2, 0, 1:\n"
              << rows << "\nOriginal order restored:\n"
              << restored << '\n';

    // 4. reorder columns by right multiplication using a runtime index vector
    const PermutationMatrix<Dynamic> dynamic_permutation(std::vector<int> {2, 0, 1});
    const Matrix<double, 2, 3> columns(observations.transpose() * dynamic_permutation);
    std::cout << "4. Right permutation of the transposed table:\n" << columns << '\n';

    // 5. rotate a point, compose rotations and apply the inverse
    // the RotationMatrix constructor validates a complete proper rotation
    using Rotation = RotationMatrix<double, 2>;
    const Matrix<double, 2, 2> quarter_turn({0., -1., 1., 0.});
    const Rotation turn(quarter_turn);
    const Vector<double, 2> rotated(turn * point);
    const auto half_turn = turn * turn;
    const auto inverse = turn.inv();
    const Vector<double, 2> returned(inverse * rotated);
    std::cout << "5. Point after a quarter turn:\n"
              << rotated << "\nTwo quarter turns composed:\n"
              << half_turn << "\nApply the inverse rotation:\n"
              << returned << '\n';

    // 6. replace a complete valid rotation through a borrowed view
    // individual coefficients are read-only to preserve the rotation constraint
    Rotation rotation = Rotation::Identity();
    auto rotation_view = rotation.view();
    rotation_view = turn;
    std::cout << "6. Rotation replaced through its view:\n" << rotation << '\n';
}
