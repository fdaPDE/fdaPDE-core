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
// examples/linear_algebra/01_dense_storage_and_views.cpp -o /tmp/01_dense_storage_and_views
#include <fdaPDE/dense_linear_algebra.h>

#include <array>
#include <iostream>

using namespace fdapde;

// start with a small table, then edit its rows and reuse memory supplied by another component
int main() {
    // 1. create a matrix with fixed dimensions: rows are observations, columns are measured quantities
    // dimensions are template arguments; the initializer lists coefficients row by row
    const Matrix<double, 2, 3> observations({1., 2., 3., 4., 5., 6.});
    std::cout << "1. Two observations, three quantities:\n" << observations << '\n';

    // 2. choose dimensions at runtime with Dynamic; assignment sizes the owner to fit the source
    Matrix<double, Dynamic, Dynamic> table;
    table = observations;
    Matrix<double, Dynamic, 3> three_columns(observations);
    std::cout << "2. Runtime shape: " << table.rows() << " x " << table.cols()
              << "\nFixed column count, runtime rows:\n"
              << three_columns << '\n';

    // 3. edit rows, columns and blocks through views that borrow the parent storage
    table.row(1) = table.row(0);
    table.block<1, 2>(0, 1) = observations.block<1, 2>(1, 1);
    table.col(0) *= 2.;
    std::cout << "3. Copy a row, replace a block and scale the first column:\n" << table << '\n';

    // 4. use an external buffer as a matrix with MatrixView, which borrows its memory
    // keep the buffer alive while using either the writable or read-only view
    std::array<double, 6> buffer {};
    MatrixView<double, 2, 3> view(buffer.data());
    view = table;
    view(0, 0) = 10.;
    const MatrixView<const double, Dynamic, Dynamic> read_only(buffer.data(), 2, 3);
    std::cout << "4. External buffer viewed as a matrix:\n"
              << read_only << "\nThe first raw buffer entry is now " << buffer[0] << '\n';

    // 5. choose the memory layout independently of logical coordinates; the default is RowMajor
    const Matrix<double, 2, 3, ColMajor> columns(observations);
    std::cout << "5. Column-major buffer, in memory order:";
    for (int i = 0; i < 6; ++i) std::cout << ' ' << columns.data()[i];
    std::cout << '\n';

    // 6. reshape storage and apply the same view operations to vectors
    // reshape borrows the coefficients; transpose assignment can reuse its input
    Matrix<double, 2, 2> square({1., 2., 3., 4.});
    square = square.transpose();
    auto flattened = square.reshape<4>();
    flattened[1] = 9.;
    std::cout << "6. Transpose, then edit through a flattened view:\n" << square << '\n';

    // vectors have the same owner/view distinction and can generate a regular sampling grid
    Vector<double, Dynamic> axis = Vector<double, Dynamic>::LinSpaced(3, 0., 1.);
    VectorView<double, Dynamic> axis_view(axis.data(), axis.size());
    axis_view[1] = .25;
    std::cout << "Sampling locations after moving the middle point:\n" << axis << '\n';
}
