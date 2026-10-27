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
// examples/linear_algebra/02_dense_expressions_and_products.cpp -o /tmp/02_dense_expressions_and_products
#include <fdaPDE/dense_linear_algebra.h>

#include <iostream>

using namespace fdapde;

// combine measurements, transform them and summarize each observation
int main() {
    Matrix<double, 2, 2> a({1., 2., 3., 4.});
    const Matrix<double, 2, 2> b({2., 0., 1., 2.});

    // 1. build a lazy expression borrowing a and b; constructing a Matrix stores the result
    // keep named operands alive until evaluation, and materialize when a snapshot is needed
    const auto average = (a + b) / 2.;
    const Matrix<double, 2, 2> snapshot(average);
    a(0, 0) = 3.;
    const Matrix<double, 2, 2> updated_average(average);
    std::cout << "1. Average saved before editing a:\n"
              << snapshot << "\nThe same expression evaluated after editing a:\n"
              << updated_average << '\n';

    // 2. distinguish matrix products from coefficient products with cwise
    // matrix constructors accept coefficientwise expressions directly
    const Matrix<double, 2, 2> transformed(a * b);
    const Matrix<double, 2, 2> weighted(a.cwise() * b.cwise());
    std::cout << "2. Matrix product a * b:\n" << transformed << "\nEntry-by-entry product:\n" << weighted << '\n';

    // mwise returns to matrix algebra before combining the weighted columns with a matrix-vector product
    // here the conversion is needed by operator*, before the result reaches the Matrix constructor
    const Vector<double, 2> column_weights {1., .5};
    const Vector<double, 2> scores((a.cwise() * b.cwise()).mwise() * column_weights);
    std::cout << "Weight entries, then combine columns into one score per row:\n" << scores << '\n';

    // 3. reduce rows and columns, then broadcast one mean per row to center each observation
    const Vector<double, 2> row_means(a.rowwise().mean());
    const Matrix<double, 1, 2> column_sums(a.colwise().sum());
    Matrix<double, 2, 2> centered(a);
    centered.rowwise() -= row_means;
    std::cout << "3. Row means:\n"
              << row_means << "\nCentered observations:\n"
              << centered << "\nColumn totals:\n"
              << column_sums << '\n';

    // 4. compare coefficients to build Boolean masks and compute coefficient powers
    const Matrix<bool, 2, 2> large(a.cwise() > 2.);
    const Matrix<double, 2, 2> squared(a.cwise().pow2());
    std::cout << "4. Entries above two: " << large.count() << "\nSquared entries:\n"
              << squared << "\nFrobenius norm: " << a.norm() << "\nTrace: " << a.trace() << '\n';

    // 5. use vector products and assemble repeated blocks with a Kronecker product
    const Vector<double, 3> x {1., 0., 0.}, y {0., 1., 0.};
    const Vector<double, 3> z(x.cross(y));
    const IdentityMatrix<double, 2, 2> identity;
    const Matrix<double, 4, 4> repeated(kron(identity, b));
    std::cout << "5. Dot product of x and y: " << x.dot(y) << "\nCross product:\n"
              << z << "\nTwo copies of b on the block diagonal:\n"
              << repeated << '\n';

    // 6. generate matrix entries on demand from their indices without allocating a coefficient buffer
    const auto coefficient = [](int i, int j) { return double(2 * i + j + 1); };
    const ProceduralMatrix<decltype(coefficient), Dynamic, Dynamic> generated(2, 2, coefficient);
    const Matrix<double, 2, 2> generated_values(generated);
    std::cout << "6. Matrix generated from row and column indices:\n" << generated_values << '\n';
}
