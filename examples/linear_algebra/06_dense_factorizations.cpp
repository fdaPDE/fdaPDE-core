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
// examples/linear_algebra/06_dense_factorizations.cpp -o /tmp/06_dense_factorizations
#include <fdaPDE/dense_linear_algebra.h>

using namespace fdapde;

// use algorithm classes, then inspect factors borrowed from each decomposition
// maybe_unused keeps unconsumed tutorial values free of compiler warnings
int main() {
    // 1. pivoted LU: solve A*X=B for two right-hand sides
    {
        const Matrix<double, 3, 3> A({0., 2., 1., 1., 1., 0., 2., 0., 1.});
        const Matrix<double, 3, 2> B({1., 2., 2., 0., 3., -1.});
        const PartialPivLU lu(A);                      // type: PartialPivLU<Matrix<double, 3, 3>>
        [[maybe_unused]] const auto X = lu.solve(B);   // type: Matrix<double, 3, 2>

        // factors satisfy P*A=L*U and remain valid until lu is recomputed or destroyed
        [[maybe_unused]] const auto& P = lu.P();   // type: const PermutationMatrix<3>&
        [[maybe_unused]] const auto& L = lu.L();   // type: const LowerTriangularMatrix<double, 3>&
        [[maybe_unused]] const auto& U = lu.U();   // type: const UpperTriangularMatrix<double, 3>&
    }

    // 2. Householder QR: minimize the norm of A*X-B for three observations and two unknowns
    {
        const Matrix<double, 3, 2> A({1., 2., 2., 0., -1., 3.});
        const Vector<double, 3> B {1., 2., 3.};
        const HouseholderQR qr(A);                     // type: HouseholderQR<double, 3, 2>
        [[maybe_unused]] const auto X = qr.solve(B);   // type: Vector<double, 2>

        // factors satisfy A=Q*R; full Q also contains the orthogonal complement
        [[maybe_unused]] const auto& Q = qr.Q();   // type: const OrthogonalMatrix<double, 3>&
        [[maybe_unused]] const auto& R = qr.R();   // type: const Matrix<double, 3, 2>&
    }

    // 3. symmetric EVD: solve A*X=B through an orthonormal eigenvector basis
    {
        const Matrix<double, 2, 2> A({2., 1., 1., 2.});
        const Vector<double, 2> B {1., 0.};
        const EVD evd(A);                               // type: EVD<Matrix<double, 2, 2>>
        [[maybe_unused]] const auto X = evd.solve(B);   // type: Vector<double, 2>

        // eigenvector columns correspond to the unsorted eigenvalues
        [[maybe_unused]] const auto& Q = evd.eigenvectors();       // type: const OrthogonalMatrix<double, 2>&
        [[maybe_unused]] const auto& values = evd.eigenvalues();   // type: const Vector<double, 2>&
    }

    // 4. Cholesky: solve A*X=B for a symmetric positive-definite matrix
    {
        const Matrix<double, 2, 2> A({2., 1., 1., 2.});
        const Vector<double, 2> B {1., 0.};
        const Cholesky chol(A);                          // type: Cholesky<Matrix<double, 2, 2>>
        [[maybe_unused]] const auto X = chol.solve(B);   // type: Vector<double, 2>

        // the positive-diagonal lower factor satisfies A=L*L.transpose()
        [[maybe_unused]] const auto& L = chol.L();   // type: const LowerTriangularMatrix<double, 2>&
    }
}
