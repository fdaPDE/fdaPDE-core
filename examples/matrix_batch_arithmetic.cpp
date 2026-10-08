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

// compile from the core root with c++ -std=c++20 -O2 -pthread -I. examples/matrix_batch_arithmetic.cpp
#include <fdaPDE/manifold_geometry.h>

using namespace fdapde;
using namespace fdapde::manifold;

// combines compatible log-domain batches using lazy arithmetic and explicit owning evaluation
int main() {
    parallel_set_num_threads(4);

    using Policy = Cache::Union<Cache::Spectral, Cache::Log>;
    using SPD = SPDMatrix<double, 2, Policy>;
    using Sym = SymmetricMatrix<double, 2>;
    const SPD A(Vector<double, 3> {2., 0.3, 1.});
    const SPD B(Vector<double, 3> {1., 0.2, 3.});
    const LogEuclideanGeometry<SPD> geometry;
    constexpr int count = 10;

    // both curves have the same sample count and matrix shape, as required by batch-to-batch arithmetic
    // MatrixBatch<SPDMatrix<double, 2, Cache::Union<Cache::Spectral, Cache::Log>>>
    const auto points = geometry.interpolate<Policy>(A, B, count, execution_par);
    // MatrixBatch<SPDMatrix<double, 2, Cache::Union<Cache::Spectral, Cache::Log>>>
    const auto reverse_points = geometry.interpolate<Policy>(B, A, count, execution_par);
    // MatrixBatch<SymmetricMatrix<double, 2, Cache::Spectral>>
    const auto logs = points.log<Cache::Spectral>(execution_par);
    // MatrixBatch<SymmetricMatrix<double, 2, Cache::Spectral>>
    const auto reverse_logs = reverse_points.log<Cache::Spectral>(execution_par);

    // MatrixBatchExpr expression borrowing operands; keep logs and reverse_logs alive until evaluation
    const auto sum_expr = logs + reverse_logs;
    // MatrixBatchExpr expression borrowing operands; composition defers the division as well
    const auto mean_expr = sum_expr / 2.;

    // MatrixBatch<SymmetricMatrix<double, 2>>: arithmetic retains symmetry and starts without spectral caches
    const auto sum_seq = sum_expr.eval(execution_seq);
    // MatrixBatch<SymmetricMatrix<double, 2>>: the same expression can be evaluated with either policy
    const auto sum_par = sum_expr.eval(execution_par);
    // MatrixBatch<SymmetricMatrix<double, 2>>: opposing curves have the same log-domain mean at every sample
    const auto means = mean_expr.eval(execution_par);
    // MatrixBatch<SymmetricMatrix<double, 2>>: paired differences measure separation along the curves
    const auto differences = (logs - reverse_logs).eval(execution_par);
    // MatrixBatch<SymmetricMatrix<double, 2>>
    const auto left_scaled = (2. * logs).eval(execution_par);
    // MatrixBatch<SymmetricMatrix<double, 2>>
    const auto right_scaled = (logs * 2.).eval(execution_par);
    // MatrixBatch<SymmetricMatrix<double, 2>>
    const auto halved = (logs / 2.).eval(execution_seq);

    // a symmetric matrix is broadcast to every sample while preserving packed symmetric storage
    const Sym shift(Vector<double, 3> {0.2, 0.05, 0.1});
    // MatrixBatch<SymmetricMatrix<double, 2>>
    const auto shifted = (logs + shift).eval(execution_par);
    // MatrixBatch<SymmetricMatrix<double, 2>>
    const auto restored = (shifted - shift).eval(execution_seq);

    // a general matrix broadcast and matrix products produce ordinary dense matrix owners
    const Matrix<double, 2, 2> mixing({1., 0.25, -0.1, 1.});
    // MatrixBatch<Matrix<double, 2, 2>>
    const auto dense_shifted = (logs + mixing).eval(execution_par);
    // MatrixBatch<Matrix<double, 2, 2>>
    const auto dense_restored = (dense_shifted - mixing).eval(execution_seq);
    // MatrixBatch<Matrix<double, 2, 2>>: right multiplication mixes each matrix's columns
    const auto right_mixed = (logs * mixing).eval(execution_par);
    // MatrixBatch<Matrix<double, 2, 2>>: left multiplication mixes each matrix's rows
    const auto left_mixed = (mixing * logs).eval(execution_par);
    // MatrixBatch<Matrix<double, 2, 2>>: each sample is a matrix product, with no symmetry guarantee
    const auto products = (logs * reverse_logs).eval(execution_par);

    // reuse owning destinations when an algorithm replaces or updates a batch
    MatrixBatch<Sym> assigned;
    assigned = mean_expr;
    MatrixBatch<Sym> updated;
    updated.assign(mean_expr, execution_par);
    updated += shift;
}
