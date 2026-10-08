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
// examples/linear_algebra/07_spd_spectral_and_caches.cpp -o /tmp/07_spd_spectral_and_caches
#include <fdaPDE/dense_linear_algebra.h>

using namespace fdapde;

// choose the input cache for each matrix operation
// maybe_unused keeps unconsumed tutorial values free of compiler warnings
int main() {
    // lower-triangle coefficients follow the order {A00, A10, A11}
    // SPD construction validates positive definiteness and prepares the selected cache
    const Vector<double, 3> coefficients {2., 1., 2.};

    // 1. retain the logarithm with Cache::Log and map it back to SPD space through the exponential
    {
        const SPDMatrix<double, 2, Cache::Log> A(coefficients);
        // the output's spectral cache supports the subsequent exponential
        const auto logarithm = A.log<Cache::Spectral>();          // type: SymmetricMatrix<double, 2, Cache::Spectral>
        [[maybe_unused]] const auto restored = logarithm.exp();   // type: SPDMatrix<double, 2>
    }

    // 2. reuse the principal square root retained by Cache::Sqrt
    {
        const SPDMatrix<double, 2, Cache::Sqrt> A(coefficients);
        [[maybe_unused]] const auto root = A.sqrt();   // type: SPDMatrix<double, 2>
    }

    // 3. reuse the inverse principal square root retained by Cache::InverseSqrt
    {
        const SPDMatrix<double, 2, Cache::InverseSqrt> A(coefficients);
        [[maybe_unused]] const auto inverse_root = A.inv_sqrt();   // type: SPDMatrix<double, 2>
    }

    // 4. compute the inverse using the lower factor retained by Cache::Cholesky
    {
        const SPDMatrix<double, 2, Cache::Cholesky> A(coefficients);
        [[maybe_unused]] const auto inverse = A.inv();   // type: SPDMatrix<double, 2>
    }

    // 5. obtain an owning Cholesky factor directly from Cache::Cholesky
    {
        const SPDMatrix<double, 2, Cache::Cholesky> A(coefficients);
        [[maybe_unused]] const auto L = A.cholesky_factor();   // type: LowerTriangularMatrix<double, 2>
    }

    // 6. request eigenvalues and eigenvectors directly, reusing Cache::Spectral
    {
        const SymmetricMatrix<double, 2, Cache::Spectral> A(coefficients);
        [[maybe_unused]] const auto eigenvalues = A.eigenvalues();     // type: Vector<double, 2>
        [[maybe_unused]] const auto eigenvectors = A.eigenvectors();   // type: OrthogonalMatrix<double, 2>
    }

    // 7. complete SPD replacement through a view refreshes Cache::Log
    // individual SPD coefficients are read-only to preserve positive definiteness
    {
        SPDMatrix<double, 2, Cache::Log> A(coefficients);
        const auto identity = SPDMatrix<double, 2, Cache::Log>::Identity();   // type: SPDMatrix<double, 2, Cache::Log>
        auto A_view = A.view();   // type: SPDMatrixView<double, 2, Cache::Log>
        A_view = identity;
        [[maybe_unused]] const auto replacement_log = A.log();   // type: SymmetricMatrix<double, 2>
    }
}
