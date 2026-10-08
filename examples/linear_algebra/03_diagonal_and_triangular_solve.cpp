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
// examples/linear_algebra/03_diagonal_and_triangular_solve.cpp -o /tmp/03_diagonal_and_triangular_solve
#include <fdaPDE/dense_linear_algebra.h>

#include <array>
#include <iostream>

using namespace fdapde;

// solve systems whose structure allows storing and using only the independent coefficients
int main() {
    // 1. store and edit only the diagonal entries, then solve a diagonal system
    // operator[] addresses the stored diagonal coefficients
    DiagonalMatrix<double, 3> diagonal(2., 3., 4.);
    diagonal[1] = 5.;
    const Vector<double, 3> rhs {4., 10., 12.};
    const auto rescaled = diagonal.solve(rhs);
    std::cout << "1. Undo diagonal scaling in D*x = b:\nD =\n"
              << diagonal << "\nb =\n"
              << rhs << "\nx =\n"
              << rescaled << "\ndet(D) = " << diagonal.determinant() << '\n';

    // 2. borrow external diagonal storage, choose a runtime order and expand into a dense matrix
    // the diagonal buffer contains only three entries, while the dense matrix contains nine
    std::array<double, 3> diagonal_buffer {};
    DiagonalMatrixView<double, 3> diagonal_view(diagonal_buffer.data());
    diagonal_view = diagonal;
    const DiagonalMatrix<double, Dynamic> dynamic_diagonal(diagonal_view);
    // direct construction selects fixed dimensions; as_matrix keeps the source's runtime dimensions
    const Matrix<double, 3, 3> full_diagonal(dynamic_diagonal);
    const auto expanded_diagonal = dynamic_diagonal.as_matrix();   // type: Matrix<double, Dynamic, Dynamic>
    std::cout << "2. Direct construction of a fixed-size dense matrix:\n"
              << full_diagonal << "\nDynamic dense matrix returned by as_matrix():\n"
              << expanded_diagonal << '\n';

    // 3. solve lower and upper triangular systems by forward and backward substitution
    // lower triangular initialization follows packed rows: (l00), (l10,l11), (l20,l21,l22)
    const LowerTriangularMatrix<double, 3> lower({2., 1., 3., -1., 2., 4.});
    const UpperTriangularMatrix<double, 3> upper(lower.transpose());
    const auto forward = lower.solve(rhs);
    const auto backward = upper.solve(rhs);
    std::cout << "3. Forward substitution, L*x = b:\nL =\n"
              << lower << "\nx =\n"
              << forward << "\nBackward substitution, transpose(L)*x = b:\n"
              << backward << '\n';

    // 4. solve several systems together, placing one right-hand side in each column
    const LowerTriangularMatrix<double, Dynamic> dynamic_lower(lower);
    const Matrix<double, 3, 2> loads({4., 2., 10., 0., 12., 4.});
    const auto solutions = dynamic_lower.solve(loads);
    std::cout << "4. Two loads solved together:\n" << solutions << "\nInverse of L:\n" << lower.inv() << '\n';

    // 5. edit a triangular matrix through a view over six externally stored entries
    std::array<double, 6> packed {};
    LowerTriangularMatrixView<double, 3> lower_view(packed.data());
    lower_view = lower;
    lower_view(2, 0) = .5;
    std::cout << "5. Edit an independent entry through a packed view:\n" << lower_view << '\n';

    // 6. edit triangular and diagonal regions of a dense matrix through borrowed views
    // these selections retain the dense matrix's full storage
    Matrix<double, 3, 3> dense = Matrix<double, 3, 3>::Ones();
    auto lower_block = dense.triangular_block<Lower>();
    lower_block(2, 0) = .5;
    dense.diagonal() = diagonal.diagonal();
    std::cout << "6. Dense matrix after triangular and diagonal edits:\n" << dense << '\n';
}
