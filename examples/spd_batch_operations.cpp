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

// compile from the repository root with c++ -std=c++20 -O2 -pthread -I. examples/spd_batch_operations.cpp
#include <fdaPDE/manifold_optimization.h>

using namespace fdapde;
using namespace fdapde::manifold;

// samples a cached SPD geodesic and evaluates owning batch operations in parallel
int main() {
    parallel_set_num_threads(4);

    using Policy = Cache::Union<Cache::Spectral, Cache::Log, Cache::Sqrt, Cache::InverseSqrt>;
    using SPD = SPDMatrix<double, 2, Policy>;

    const SPD A(Vector<double, 3> {2., 0.3, 1.});
    const SPD B(Vector<double, 3> {1., 0.2, 3.});
    const LogEuclideanGeometry<SPD> geometry;
    constexpr int count = 10;

    // prepared geodesic callable, evaluated at any parameter t
    const auto curve = geometry.geodesic(A, B);
    const SPD midpoint(curve(0.5));

    // MatrixBatch<SPD>, uniformly spaced samples including both endpoints
    const auto points = geometry.interpolate<Policy>(A, B, count, execution_par);

    // MatrixBatch<SymmetricMatrix<double, 2, Cache::Spectral>>
    const auto logs = points.log<Cache::Spectral>(execution_par);

    // MatrixBatch<Vector<double, 2>>
    const auto eigenvalues = logs.eigenvalues(execution_par);

    // MatrixBatch<OrthogonalMatrix<double, 2, 2>>
    const auto eigenvectors = logs.eigenvectors(execution_par);

    // MatrixBatch<Matrix<double, 1, 1>>
    const auto traces = logs.trace(execution_par);

    // MatrixBatch<Matrix<double, 1, 1>>
    const auto determinants = logs.determinant(execution_par);

    // MatrixBatch<Matrix<double, 1, 1>>
    const auto norms = logs.norm(execution_par);

    // MatrixBatch<Matrix<double, 1, 1>>
    const auto squared_norms = logs.squared_norm(execution_par);

    // MatrixBatch<Vector<double, 2>>
    const auto diagonals = logs.diagonal(execution_par);

    // MatrixBatch<SPD>
    const auto recovered = logs.exp<Policy>(execution_par);

    // MatrixBatch<SPDMatrix<double, 2, Cache::Spectral>>
    const auto roots = points.sqrt<Cache::Spectral>(execution_par);

    // MatrixBatch<SPDMatrix<double, 2, Cache::Spectral>>
    const auto inverse_roots = points.inv_sqrt<Cache::Spectral>(execution_par);

    // MatrixBatch<SPDMatrix<double, 2, Cache::Spectral>>
    const auto inverses = points.inv<Cache::Spectral>(execution_par);

    // MatrixBatch<OrthogonalMatrix<double, 2, 2>>
    const auto inverse_bases = eigenvectors.inv(execution_par);
}
