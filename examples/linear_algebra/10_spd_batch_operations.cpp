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
// examples/linear_algebra/10_spd_batch_operations.cpp -o /tmp/10_spd_batch_operations
#include <fdaPDE/dense_linear_algebra.h>

using namespace fdapde;

// apply structured matrix operations to an SPD batch with a cache chosen for each operation
// maybe_unused keeps unconsumed tutorial values free of compiler warnings
int main() {
    parallel_set_num_threads(4);
    using SPD = SPDMatrix<double, 2>;

    // 1. SPD batches start as identities; assigning a complete element validates its positive definiteness
    MatrixBatch<SPD> input(2);
    input[0] = SPD(Vector<double, 3> {2., 1., 2.});
    input[1] = SPD(Vector<double, 3> {4., 0., 1.});

    // 2. retain logarithms in the input batch and spectral factors in the output for the subsequent exponential
    {
        const MatrixBatch<SPDMatrix<double, 2, Cache::Log>> A(input);
        const auto logs = A.log<Cache::Spectral>(execution_par);
        // type: MatrixBatch<SymmetricMatrix<double, 2, Cache::Spectral>>
        [[maybe_unused]] const auto restored = logs.exp(execution_par);   // type: MatrixBatch<SPDMatrix<double, 2>>
    }

    // 3. reuse the principal square roots retained by Cache::Sqrt
    {
        const MatrixBatch<SPDMatrix<double, 2, Cache::Sqrt>> A(input);
        [[maybe_unused]] const auto roots = A.sqrt(execution_par);   // type: MatrixBatch<SPDMatrix<double, 2>>
    }

    // 4. reuse the inverse principal square roots retained by Cache::InverseSqrt
    {
        const MatrixBatch<SPDMatrix<double, 2, Cache::InverseSqrt>> A(input);
        [[maybe_unused]] const auto inverse_roots = A.inv_sqrt(execution_par);
        // type: MatrixBatch<SPDMatrix<double, 2>>
    }

    // 5. compute inverses using the lower factors retained by Cache::Cholesky
    {
        const MatrixBatch<SPDMatrix<double, 2, Cache::Cholesky>> A(input);
        [[maybe_unused]] const auto inverses = A.inv(execution_par);   // type: MatrixBatch<SPDMatrix<double, 2>>
    }

    // 6. reuse Cache::Spectral to extract an eigenvalue vector and an orthogonal basis for each matrix
    {
        const MatrixBatch<SPDMatrix<double, 2, Cache::Spectral>> A(input);
        [[maybe_unused]] const auto eigenvalues =
          A.eigenvalues(execution_par);                            // type: MatrixBatch<Vector<double, 2>>
        const auto eigenvectors = A.eigenvectors(execution_par);   // type: MatrixBatch<OrthogonalMatrix<double, 2>>
        [[maybe_unused]] const auto inverse_bases = eigenvectors.inv(execution_par);
        // type: MatrixBatch<OrthogonalMatrix<double, 2>>
    }

    // 7. scalar operations produce one-by-one matrices; diagonal extraction produces column vectors
    {
        [[maybe_unused]] const auto traces = input.trace(execution_par);   // type: MatrixBatch<Matrix<double, 1, 1>>
        [[maybe_unused]] const auto determinants = input.determinant(execution_par);
        // type: MatrixBatch<Matrix<double, 1, 1>>
        [[maybe_unused]] const auto norms = input.norm(execution_par);   // type: MatrixBatch<Matrix<double, 1, 1>>
        [[maybe_unused]] const auto squared_norms = input.squared_norm(execution_par);
        // type: MatrixBatch<Matrix<double, 1, 1>>
        [[maybe_unused]] const auto diagonals = input.diagonal(execution_par);   // type: MatrixBatch<Vector<double, 2>>
    }
}
