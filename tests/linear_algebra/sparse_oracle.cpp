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

#include <fdaPDE/sparse_linear_algebra.h>
#include <gtest/gtest.h>

#include <Eigen/SparseCore>
#include <vector>

namespace {

using native_triplet = fdapde::Triplet<double>;
using eigen_sparse = Eigen::SparseMatrix<double, Eigen::RowMajor, int>;

// emits overlapping triangle contributions with exactly representable duplicate values
std::vector<native_triplet> make_grid_triplets(int subdivisions) {
    const int nodes_per_side = subdivisions + 1;
    std::vector<native_triplet> triplets;
    triplets.reserve(static_cast<std::size_t>(18 * subdivisions * subdivisions));
    const auto append_triangle = [&triplets](int first, int second, int third) {
        const int nodes[3] {first, second, third};
        for (int row = 0; row < 3; ++row) {
            for (int col = 0; col < 3; ++col) {
                triplets.emplace_back(nodes[row], nodes[col], row == col ? 2.0 : -0.5);
            }
        }
    };
    for (int row = 0; row < subdivisions; ++row) {
        for (int col = 0; col < subdivisions; ++col) {
            const int lower_left = row * nodes_per_side + col;
            const int lower_right = lower_left + 1;
            const int upper_left = lower_left + nodes_per_side;
            const int upper_right = upper_left + 1;
            append_triangle(lower_left, lower_right, upper_right);
            append_triangle(lower_left, upper_right, upper_left);
        }
    }
    return triplets;
}

// compares canonical CSR rows and values against Eigen triplet compression
void expect_same_compression(int rows, int cols, const std::vector<native_triplet>& triplets) {
    const fdapde::SparseMatrix<double> native(rows, cols, triplets);

    std::vector<Eigen::Triplet<double, int>> eigen_triplets;
    eigen_triplets.reserve(triplets.size());
    for (const auto& triplet : triplets) { eigen_triplets.emplace_back(triplet.row(), triplet.col(), triplet.value()); }
    eigen_sparse oracle(rows, cols);
    oracle.setFromTriplets(eigen_triplets.begin(), eigen_triplets.end());
    oracle.prune(0.0);
    oracle.makeCompressed();

    // native and Eigen compression retain the same row count
    ASSERT_EQ(native.rows(), oracle.rows());
    // native and Eigen compression retain the same column count
    ASSERT_EQ(native.cols(), oracle.cols());
    // both implementations store the same number of entries after zero pruning
    ASSERT_EQ(native.non_zeros(), oracle.nonZeros());
    for (int row = 0; row < rows; ++row) {
        auto native_it = native.row(row).begin();
        const auto native_end = native.row(row).end();
        for (eigen_sparse::InnerIterator oracle_it(oracle, row); oracle_it; ++oracle_it) {
            // each Eigen row entry has a corresponding native entry
            ASSERT_NE(native_it, native_end);
            // native row iteration matches the Eigen column order
            EXPECT_EQ((*native_it).column(), oracle_it.col());
            // native duplicate sums match the independent Eigen coefficients
            EXPECT_DOUBLE_EQ((*native_it).value(), oracle_it.value());
            ++native_it;
        }
        // native iteration has no extra entries after the Eigen row ends
        EXPECT_EQ(native_it, native_end);
    }
}

// checks rectangular and assembly-shaped duplicate compression against Eigen
TEST(NativeSparseOracle, MatchesEigenTripletCompression) {
    // compare a small rectangular matrix with duplicates and explicit zeros
    expect_same_compression(
      3, 4,
      std::vector<native_triplet> {
        {2, 3, 4.0 },
        {0, 1, 2.0 },
        {2, 1, -1.0},
        {0, 1, 3.0 },
        {2, 1, 1.0 },
        {0, 3, 0.0 }
    });

    const int subdivisions = 32;
    const int nodes = (subdivisions + 1) * (subdivisions + 1);
    // compare all shared-node contributions from a structured triangle grid
    expect_same_compression(nodes, nodes, make_grid_triplets(subdivisions));
}

// exercises empty, rectangular, sparse and duplicate-heavy layouts in both compression paths
TEST(NativeSparseOracle, MatchesEigenAcrossShapesAndDensities) {
    for (int rows : {0, 1, 4, 11}) {
        for (int cols : {0, 1, 7, 101}) {
            for (int count : {0, 3, 150}) {
                std::vector<native_triplet> triplets;
                if (rows > 0 && cols > 0) {
                    for (int k = 0; k < count; ++k) {
                        triplets.emplace_back((7 * k + 3) % rows, (13 * k + 1) % cols, (k % 5) - 2.0);
                    }
                }
                // compare every row, sorted coordinate and combined coefficient with Eigen
                expect_same_compression(rows, cols, triplets);
            }
        }
    }
}

// compares compressed dimensions, sorted row patterns and coefficient values against Eigen
void expect_same_sparse(const fdapde::SparseMatrix<double>& native, const eigen_sparse& oracle) {
    // the native sparse row count must match Eigen before row traversal
    ASSERT_EQ(native.rows(), oracle.rows());
    // the native sparse column count must match Eigen before coefficient comparison
    ASSERT_EQ(native.cols(), oracle.cols());
    // the native stored count must match the compressed Eigen pattern
    ASSERT_EQ(native.non_zeros(), oracle.nonZeros());
    for (int row = 0; row < native.rows(); ++row) {
        auto native_it = native.row(row).begin();
        const auto native_end = native.row(row).end();
        for (eigen_sparse::InnerIterator oracle_it(oracle, row); oracle_it; ++oracle_it) {
            // each Eigen entry must have a corresponding native row entry
            ASSERT_NE(native_it, native_end);
            // native columns must follow the same sorted order as Eigen
            EXPECT_EQ((*native_it).column(), oracle_it.col());
            // native stored values must equal Eigen at the corresponding position
            EXPECT_DOUBLE_EQ((*native_it).value(), oracle_it.value());
            ++native_it;
        }
        // native rows must contain no entries beyond the Eigen pattern
        EXPECT_EQ(native_it, native_end);
    }
}

// constructs and prunes an independent Eigen oracle from the shared triplets
eigen_sparse make_eigen_sparse(int rows, int cols, const std::vector<native_triplet>& triplets) {
    std::vector<Eigen::Triplet<double, int>> eigen_triplets;
    eigen_triplets.reserve(triplets.size());
    for (const auto& triplet : triplets) { eigen_triplets.emplace_back(triplet.row(), triplet.col(), triplet.value()); }
    eigen_sparse result(rows, cols);
    result.setFromTriplets(eigen_triplets.begin(), eigen_triplets.end());
    result.prune(0.0);
    result.makeCompressed();
    return result;
}

// compares transforms, mixed dense products, reductions and quadratic forms with Eigen
TEST(NativeSparseOracle, MatchesEigenRequiredOperations) {
    const std::vector<native_triplet> triplets {
      {0, 0, 2.0 },
      {0, 2, -1.0},
      {1, 1, 3.0 },
      {1, 3, 4.0 },
      {2, 0, 5.0 }
    };
    const fdapde::SparseMatrix<double> native(3, 4, triplets);
    const eigen_sparse oracle = make_eigen_sparse(3, 4, triplets);

    eigen_sparse eigen_transpose = oracle.transpose();
    eigen_transpose.makeCompressed();
    // the owning transpose matches the complete compressed Eigen transpose
    expect_same_sparse(native.transpose(), eigen_transpose);

    const std::vector<native_triplet> lower_triplets {
      {0, 0, 2.0 },
      {1, 0, -1.0},
      {1, 1, 3.0 },
      {2, 0, 4.0 },
      {2, 2, 5.0 }
    };
    const fdapde::SparseMatrix<double> native_lower(3, 3, lower_triplets);
    const eigen_sparse eigen_lower = make_eigen_sparse(3, 3, lower_triplets);
    eigen_sparse eigen_symmetric = eigen_lower.selfadjointView<Eigen::Lower>();
    eigen_symmetric.makeCompressed();
    const auto native_symmetric = native_lower.symmetric_expanded(fdapde::Lower);
    // lower-triangle expansion matches the complete Eigen self-adjoint pattern
    expect_same_sparse(native_symmetric, eigen_symmetric);

    const std::vector<native_triplet> upper_triplets {
      {0, 0, 2.0 },
      {0, 1, -1.0},
      {0, 2, 4.0 },
      {1, 1, 3.0 },
      {2, 2, 5.0 }
    };
    const fdapde::SparseMatrix<double> native_upper(3, 3, upper_triplets);
    const eigen_sparse eigen_upper = make_eigen_sparse(3, 3, upper_triplets);
    eigen_sparse eigen_upper_symmetric = eigen_upper.selfadjointView<Eigen::Upper>();
    eigen_upper_symmetric.makeCompressed();
    // upper-triangle expansion matches the complete Eigen self-adjoint pattern
    expect_same_sparse(native_upper.symmetric_expanded(fdapde::Upper), eigen_upper_symmetric);

    const fdapde::Vector<double, 4> native_vector({1.0, 2.0, 3.0, 4.0});
    const Eigen::Vector4d eigen_vector(1.0, 2.0, 3.0, 4.0);
    const auto native_vector_product = native * native_vector;
    const Eigen::VectorXd eigen_vector_product = oracle * eigen_vector;
    // the sparse-vector result must have the same length as Eigen
    ASSERT_EQ(native_vector_product.size(), eigen_vector_product.size());
    for (int i = 0; i < native_vector_product.size(); ++i) {
        // each sparse-vector coefficient matches the independently evaluated Eigen product
        EXPECT_DOUBLE_EQ(native_vector_product[i], eigen_vector_product[i]);
    }

    const double dense_values[8] {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    const fdapde::Matrix<double, 4, 2, fdapde::ColMajor> native_dense(dense_values);
    Eigen::Matrix<double, 4, 2> eigen_dense;
    eigen_dense << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0;
    const auto native_dense_product = native * native_dense;
    const Eigen::MatrixXd eigen_dense_product = oracle * eigen_dense;
    // the sparse-dense result must have the same row count as Eigen
    ASSERT_EQ(native_dense_product.rows(), eigen_dense_product.rows());
    // the sparse-dense result must have the same column count as Eigen
    ASSERT_EQ(native_dense_product.cols(), eigen_dense_product.cols());
    for (int i = 0; i < native_dense_product.rows(); ++i) {
        for (int j = 0; j < native_dense_product.cols(); ++j) {
            // column-major right-hand-side products match Eigen coefficient by coefficient
            EXPECT_DOUBLE_EQ(native_dense_product(i, j), eigen_dense_product(i, j));
        }
    }

    const fdapde::Matrix<double, 4, 2> native_row_major_dense(dense_values);
    Eigen::Matrix<double, 4, 2, Eigen::RowMajor> eigen_row_major_dense;
    eigen_row_major_dense << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0;
    const auto native_row_major_product = native * native_row_major_dense;
    const Eigen::Matrix<double, 3, 2, Eigen::RowMajor> eigen_row_major_product = oracle * eigen_row_major_dense;
    for (int i = 0; i < native_row_major_product.rows(); ++i) {
        for (int j = 0; j < native_row_major_product.cols(); ++j) {
            // row-major right-hand-side products match Eigen coefficient by coefficient
            EXPECT_DOUBLE_EQ(native_row_major_product(i, j), eigen_row_major_product(i, j));
        }
    }

    const auto native_sums = native.row_sums();
    const Eigen::VectorXd eigen_sums = oracle * Eigen::VectorXd::Ones(oracle.cols());
    for (int i = 0; i < native_sums.size(); ++i) {
        // each row sum matches multiplication by an Eigen vector of ones
        EXPECT_DOUBLE_EQ(native_sums[i], eigen_sums[i]);
    }

    const auto native_diagonal = native_symmetric.diagonal();
    const Eigen::VectorXd eigen_diagonal = eigen_symmetric.diagonal();
    for (int i = 0; i < native_diagonal.size(); ++i) {
        // each extracted diagonal coefficient matches Eigen extraction
        EXPECT_DOUBLE_EQ(native_diagonal[i], eigen_diagonal[i]);
    }
    // rebuilding the extracted diagonal matches the independent Eigen diagonal matrix
    expect_same_sparse(
      fdapde::SparseMatrix<double>::from_diagonal(native_diagonal),
      eigen_diagonal.asDiagonal().toDenseMatrix().sparseView());

    const fdapde::Vector<double, 3> native_quadratic_vector({1.0, 2.0, 3.0});
    const Eigen::Vector3d eigen_quadratic_vector(1.0, 2.0, 3.0);
    // the direct quadratic form matches Eigen matrix multiplication followed by a dot product
    EXPECT_DOUBLE_EQ(
      native_symmetric.quadratic_form(native_quadratic_vector),
      eigen_quadratic_vector.dot(eigen_symmetric * eigen_quadratic_vector));
}

// compares sparse products and transposes with Eigen across empty, tall, wide and square inputs
TEST(NativeSparseOracle, MatchesEigenOperationsAcrossShapes) {
    for (const int rows : {0, 1, 3, 9}) {
        for (const int cols : {0, 1, 4, 9}) {
            std::vector<native_triplet> triplets;
            for (int row = 0; row < rows; ++row) {
                for (int col = 0; col < cols; ++col) {
                    if ((row + 3 * col) % 4 != 0) triplets.emplace_back(row, col, (row - col + 2) * 0.125);
                }
            }
            const fdapde::SparseMatrix<double> native(rows, cols, triplets);
            const eigen_sparse oracle = make_eigen_sparse(rows, cols, triplets);
            eigen_sparse eigen_transpose = oracle.transpose();
            eigen_transpose.makeCompressed();
            // every rectangular transpose matches Eigen dimensions, sorted pattern and coefficients
            expect_same_sparse(native.transpose(), eigen_transpose);
            fdapde::Vector<double, fdapde::Dynamic> vector(cols);
            Eigen::VectorXd eigen_vector(cols);
            fdapde::Matrix<double, fdapde::Dynamic, fdapde::Dynamic, fdapde::ColMajor> dense(cols, 3);
            Eigen::MatrixXd eigen_dense(cols, 3);
            for (int row = 0; row < cols; ++row) {
                vector[row] = eigen_vector[row] = (row - 3) * 0.25;
                for (int col = 0; col < 3; ++col) dense(row, col) = eigen_dense(row, col) = (row + col - 2) * 0.5;
            }
            const auto vector_product = native * vector;
            const Eigen::VectorXd eigen_vector_product = oracle * eigen_vector;
            const auto dense_product = native * dense;
            const Eigen::MatrixXd eigen_dense_product = oracle * eigen_dense;
            const auto sums = native.row_sums();
            const Eigen::VectorXd eigen_sums = oracle * Eigen::VectorXd::Ones(cols);
            for (int row = 0; row < rows; ++row) {
                // each rectangular sparse-vector dot product matches the independent Eigen result
                EXPECT_DOUBLE_EQ(vector_product[row], eigen_vector_product[row]);
                // row summation matches Eigen multiplication by a vector of ones
                EXPECT_DOUBLE_EQ(sums[row], eigen_sums[row]);
                for (int col = 0; col < 3; ++col) {
                    // dynamic column-major right-hand sides match Eigen for all three output columns
                    EXPECT_DOUBLE_EQ(dense_product(row, col), eigen_dense_product(row, col));
                }
            }
            if (rows == cols) {
                // the bilinear form also matches Eigen on nonsymmetric square fixtures
                EXPECT_DOUBLE_EQ(native.quadratic_form(vector), eigen_vector.dot(oracle * eigen_vector));
            }
        }
    }
}

// applies row and column elimination through Eigen before pruning the resulting zeros
void rebuild_eigen_constraints(eigen_sparse& matrix, const std::vector<int>& dofs) {
    for (const int dof : dofs) {
        matrix.row(dof) *= 0.0;
        matrix.col(dof) *= 0.0;
        matrix.coeffRef(dof, dof) = 1.0;
    }
    matrix.prune(0.0);
    matrix.makeCompressed();
}

// compares constraint rebuilding with Eigen on nonsymmetric and symmetric systems
TEST(NativeSparseOracle, MatchesEigenConstraintRebuilding) {
    const std::vector<native_triplet> nonsymmetric_triplets {
      {0, 0, 2.0 },
      {0, 1, 3.0 },
      {0, 3, 4.0 },
      {1, 0, 5.0 },
      {1, 2, 6.0 },
      {2, 1, 7.0 },
      {2, 2, 8.0 },
      {2, 3, 9.0 },
      {3, 0, 10.0},
      {3, 2, 11.0},
      {3, 3, 12.0}
    };
    fdapde::SparseMatrix<double> native_nonsymmetric(4, 4, nonsymmetric_triplets);
    eigen_sparse eigen_nonsymmetric = make_eigen_sparse(4, 4, nonsymmetric_triplets);
    const std::vector<int> nonsymmetric_dofs {3, 1, 3};
    native_nonsymmetric.rebuild_with_constraints(nonsymmetric_dofs);
    rebuild_eigen_constraints(eigen_nonsymmetric, nonsymmetric_dofs);
    // repeated constraints match the full compressed Eigen result on a nonsymmetric matrix
    expect_same_sparse(native_nonsymmetric, eigen_nonsymmetric);

    const std::vector<native_triplet> lower_triplets {
      {0, 0, 4.0 },
      {1, 0, 1.0 },
      {1, 1, 5.0 },
      {2, 0, 2.0 },
      {2, 1, 3.0 },
      {2, 2, 6.0 },
      {3, 0, 7.0 },
      {3, 1, 8.0 },
      {3, 2, 9.0 },
      {3, 3, 10.0}
    };
    fdapde::SparseMatrix<double> native_symmetric =
      fdapde::SparseMatrix<double>(4, 4, lower_triplets).symmetric_expanded(fdapde::Lower);
    eigen_sparse eigen_lower = make_eigen_sparse(4, 4, lower_triplets);
    eigen_sparse eigen_symmetric = eigen_lower.selfadjointView<Eigen::Lower>();
    const std::vector<int> symmetric_dofs {1, 3};
    native_symmetric.rebuild_with_constraints(symmetric_dofs);
    rebuild_eigen_constraints(eigen_symmetric, symmetric_dofs);
    // symmetric constraint elimination matches Eigen dimensions, pattern and coefficients
    expect_same_sparse(native_symmetric, eigen_symmetric);
}

// compares lumping with Eigen row sums across dense layouts and sparse patterns, including zero sums
TEST(NativeSparseOracle, MatchesEigenLumping) {
    for (const int size : {0, 1, 7, 31}) {
        std::vector<native_triplet> triplets;
        fdapde::Matrix<double, fdapde::Dynamic, fdapde::Dynamic> dense(size, size);
        for (int row = 0; row < size; ++row) {
            for (int col = 0; col < size; ++col) {
                if ((row + col) % 3 != 0) {
                    const double value = (row - col) * 0.125;
                    triplets.emplace_back(row, col, value);
                    dense(row, col) = value;
                }
            }
        }
        const fdapde::SparseMatrix<double> native(size, size, triplets);
        const eigen_sparse oracle = make_eigen_sparse(size, size, triplets);
        const Eigen::VectorXd sums = oracle * Eigen::VectorXd::Ones(size);
        std::vector<Eigen::Triplet<double>> diagonal_triplets;
        for (int row = 0; row < size; ++row) diagonal_triplets.emplace_back(row, row, sums[row]);
        eigen_sparse expected(size, size);
        expected.setFromTriplets(diagonal_triplets.begin(), diagonal_triplets.end());
        // the sparse result matches Eigen row sums with an explicitly stored entry at every diagonal position
        expect_same_sparse(fdapde::lump(native), expected);
        const auto row_major_lumped = fdapde::lump(dense);
        const fdapde::Matrix<double, fdapde::Dynamic, fdapde::Dynamic, fdapde::ColMajor> column_major(dense);
        const auto column_major_lumped = fdapde::lump(column_major);
        const Eigen::MatrixXd eigen_dense(oracle);
        const Eigen::VectorXd dense_sums = eigen_dense.rowwise().sum();
        // the dense result preserves Eigen's row count even for the empty fixture
        ASSERT_EQ(row_major_lumped.rows(), dense_sums.size());
        for (int row = 0; row < size; ++row) {
            // row-major dense accumulation matches the independent Eigen row reduction
            EXPECT_DOUBLE_EQ(row_major_lumped[row], dense_sums[row]);
            // column-major storage yields the same logical row sum as Eigen
            EXPECT_DOUBLE_EQ(column_major_lumped[row], dense_sums[row]);
        }
    }
}

}   // namespace
