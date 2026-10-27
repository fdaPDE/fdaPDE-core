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
// -isystem /path/to/eigen3
// examples/linear_algebra/11_eigen_sparse_blocks_and_products.cpp -o /tmp/11_eigen_sparse_blocks_and_products
#include <fdaPDE/linear_algebra.h>

#include <iostream>
#include <vector>

// assemble block operators and use Eigen sparse solvers through the Eigen-dependent public header
int main() {
    // 1. an explicit ColMajor view shares Eigen's default dense layout; a native owner copies it
    Eigen::Matrix2d dense;
    dense << 2., 1., 1., 3.;
    const fdapde::MatrixView<const double, 2, 2, fdapde::ColMajor> native(dense.data());
    const fdapde::Matrix<double, 2, 2> native_copy(native);
    std::cout << "1. Eigen coefficients copied into a native matrix:\n" << native_copy << '\n';

    // 2. triplets describe nonzero entries as (row, column, value), then populate Eigen sparse storage
    using Sparse = Eigen::SparseMatrix<double>;
    const std::vector<fdapde::Triplet<double>> entries {
      {0, 0, 2.},
      {1, 0, 1.},
      {0, 1, 1.},
      {1, 1, 3.}
    };
    Sparse sparse(2, 2);
    sparse.setFromTriplets(entries.begin(), entries.end());
    sparse.makeCompressed();

    // 3. scalar zeros inherit the block row and column dimensions, giving a 4 x 4 block diagonal operator
    fdapde::SparseBlockMatrix<double, 2, 2> blocks(sparse, 0., 0., sparse);
    const Sparse block_diagonal(blocks);
    std::cout << "3. Two uncoupled blocks:\n" << Eigen::MatrixXd(block_diagonal) << '\n';

    // 4. add a coupling block, then address one coefficient with global matrix coordinates
    blocks.block(0, 1) = sparse;
    blocks.coeffRef(0, 3) = .5;
    const Sparse assembled(blocks);
    std::cout << "4. After adding and editing the upper-right coupling:\n" << Eigen::MatrixXd(assembled) << '\n';

    // 5. kronecker repeats an operator over independent components; both dense and sparse inputs work
    const Eigen::Matrix2d identity = Eigen::Matrix2d::Identity();
    const Eigen::Matrix4d tensor = fdapde::kronecker(identity, dense);
    const Sparse sparse_identity = identity.sparseView();
    const Sparse sparse_tensor = fdapde::kronecker(sparse_identity, sparse);
    std::cout << "5. Dense Kronecker product I x A:\n"
              << tensor << "\nSparse Kronecker product:\n"
              << Eigen::MatrixXd(sparse_tensor) << '\n';

    // 6. lump replaces each row by its sum on the diagonal, preserving that row's total weight
    const auto lumped = fdapde::lump(sparse);   // type: Eigen::SparseMatrix<double>
    const auto dense_lumped = fdapde::lump(dense);
    // type: Eigen::DiagonalMatrix<double, Eigen::Dynamic, Eigen::Dynamic>
    std::cout << "6. Row-sum lumping of the sparse operator:\n"
              << Eigen::MatrixXd(lumped) << "\nRow-sum lumping of the dense operator:\n"
              << dense_lumped.toDenseMatrix() << '\n';

    // 7. pass the sparse SPD operator to Eigen's LDLT solver and solve sparse*X=B for one load vector
    Eigen::SimplicialLDLT<Sparse> solver;
    solver.compute(sparse);
    const Eigen::Vector2d B(1., -1.);
    const Eigen::Vector2d X = solver.solve(B);
    std::cout << "7. Solve sparse*X=B:\nB =\n" << B << "\nX =\n" << X << '\n';
}
