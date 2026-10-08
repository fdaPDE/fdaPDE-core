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
// examples/linear_algebra/09_matrix_batch_arithmetic.cpp -o /tmp/09_matrix_batch_arithmetic
#include <fdaPDE/dense_linear_algebra.h>

#include <array>

using namespace fdapde;

// apply matrix operations to uniformly shaped collections using deferred batch expressions
// maybe_unused keeps unconsumed tutorial values free of compiler warnings
int main() {
    parallel_set_num_threads(4);
    using Dense = Matrix<double, 2, 2>;
    using Batch = MatrixBatch<Dense>;

    // 1. create three zero-initialized matrices and assign each entry through its native view
    Batch A(3);
    A[0] = Dense({1., 2., 3., 4.});
    A[1] = Dense({2., 0., 0., 2.});
    A[2] = Dense({1., 0., 0., 3.});
    const Batch B(A);

    // 2. indexing borrows an element; the copied batch B owns independent coefficients
    auto first = A[0];   // type: MatrixView<double, 2, 2>
    first(0, 0) = 2.;

    // 3. batch arithmetic combines corresponding entries and remains lazy until evaluation
    // keep A and B alive until the expression has been evaluated
    const auto mean_expr = (A + B) / 2.;
    [[maybe_unused]] const auto mean_seq = mean_expr.eval(execution_seq);   // type: MatrixBatch<Matrix<double, 2, 2>>
    const auto mean_par = mean_expr.eval(execution_par);                    // type: MatrixBatch<Matrix<double, 2, 2>>
    [[maybe_unused]] const auto differences = (A - B).eval();               // type: MatrixBatch<Matrix<double, 2, 2>>

    // 4. a single matrix is broadcast to every entry of the batch
    const Dense shift({1., 0., 0., 1.});
    const auto shifted = (A + shift).eval();                           // type: MatrixBatch<Matrix<double, 2, 2>>
    [[maybe_unused]] const auto restored = (shifted - shift).eval();   // type: MatrixBatch<Matrix<double, 2, 2>>

    // 5. multiplication applies ordinary matrix products to each batch entry
    const Dense mixing({1., 1., 0., 1.});
    [[maybe_unused]] const auto right_products = (A * mixing).eval();   // type: MatrixBatch<Matrix<double, 2, 2>>
    [[maybe_unused]] const auto left_products = (mixing * A).eval();    // type: MatrixBatch<Matrix<double, 2, 2>>
    [[maybe_unused]] const auto paired_products = (A * B).eval();       // type: MatrixBatch<Matrix<double, 2, 2>>

    // 6. assign evaluates an expression into an owning destination; compound updates replace its values
    Batch assigned;
    assigned = mean_expr;
    Batch updated;
    updated.assign(mean_expr, execution_par);
    updated += shift;
    updated *= 2.;

    // 7. symmetric addition retains packed symmetric storage; a general matrix produces dense results
    using Sym = SymmetricMatrix<double, 2>;
    const std::array<Sym, 2> matrices {Sym(Vector<double, 3> {2., 1., 3.}), Sym(Vector<double, 3> {4., 0., 2.})};
    const MatrixBatch<Sym> symmetric(matrices);
    const Sym symmetric_shift(Vector<double, 3> {1., 0., 1.});
    [[maybe_unused]] const auto symmetric_sum =
      (symmetric + symmetric_shift).eval();   // type: MatrixBatch<SymmetricMatrix<double, 2>>
    [[maybe_unused]] const auto dense_sum = (symmetric + mixing).eval();   // type: MatrixBatch<Matrix<double, 2, 2>>

    // 8. select borrows entries in the requested order and supports direct squared norms
    const std::array<int, 2> indices {2, 0};
    const auto selected = mean_par.select(indices);
    [[maybe_unused]] const auto norms = selected.squared_norm();   // type: MatrixBatch<Matrix<double, 1, 1>>

    // 9. map extracts the first coefficient of each selected matrix
    const auto first_coefficients_expr = selected.map([](const auto& value) { return value(0, 0); });
    const auto first_coefficients = first_coefficients_expr.eval();   // type: MatrixBatch<Matrix<double, 1, 1>>
    // scalar map results become one-by-one matrices; redux accumulates their values sequentially
    [[maybe_unused]] const double total =
      first_coefficients.redux(0., [](double sum, const auto& value) { return sum + value(0, 0); });
}
