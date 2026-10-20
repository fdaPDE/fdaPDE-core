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

#include <fdaPDE/linear_algebra.h>
#include <gtest/gtest.h>

namespace {
// screened Neumann graph Laplacian with a known constant eigenvector
Eigen::SparseMatrix<double> grid(int width, double shift) {
    std::vector<Eigen::Triplet<double>> entries;
    for (int i = 0; i < width * width; ++i) {
        double diagonal = shift;
        for (int j : {i - width, i + width, i % width ? i - 1 : -1, i % width + 1 < width ? i + 1 : -1}) {
            if (j < 0 || j >= width * width) continue;
            entries.emplace_back(i, j, -1.);
            diagonal += 1.;
        }
        entries.emplace_back(i, i, diagonal);
    }
    Eigen::SparseMatrix<double> a(width * width, width * width);
    a.setFromTriplets(entries.begin(), entries.end());
    return a;
}
}   // namespace

// compare multilevel inversion of slow and oscillatory fields to their known preimages
TEST(SparseMultigrid, ResolvesScreenedLowModes) {
    const auto a = grid(32, .001);
    const fdapde::SparseMultigrid solver(a, 4, 32);
    Eigen::VectorXd exact = Eigen::VectorXd::Ones(a.rows());
    // more than one level ensures this exercises coarsening rather than only exact Cholesky
    EXPECT_GT(solver.level_sizes().size(), 1u);
    // the screened constant mode must be recovered to one percent relative error
    EXPECT_LT((solver.solve(a * exact) - exact).norm() / exact.norm(), .01);
    for (int i = 0; i < exact.size(); ++i) exact[i] = std::sin(double(i));
    // the same hierarchy must also invert an oscillatory manufactured field
    EXPECT_LT((solver.solve(a * exact) - exact).norm() / exact.norm(), .01);
}

// check the fixed linear SPD map used by optimizer coordinate transformations
TEST(SparseMultigrid, LinearSymmetricPositiveAndScaleInvariant) {
    const auto a = grid(12, .01);
    const fdapde::SparseMultigrid solver(a, 3, 16), rescaled(7. * a, 3, 16);
    Eigen::VectorXd u(a.rows()), v(a.rows());
    for (int i = 0; i < u.size(); ++i) {
        u[i] = std::sin(double(i));
        v[i] = std::cos(double(i));
    }
    const Eigen::VectorXd bu = solver.solve(u), bv = solver.solve(v);
    // superposition distinguishes a fixed cycle from a tolerance-stopped iterative solve
    EXPECT_LT((solver.solve(2. * u - v) - (2. * bu - bv)).norm() / bu.norm(), 1e-12);
    // paired forward/backward smoothing must give the same bilinear form in either order
    EXPECT_NEAR(u.dot(bv), v.dot(bu), 1e-10);
    // positive energy is required for the optimizer's invertible chart
    EXPECT_GT(u.dot(bu), 0.);
    // uniform matrix scaling must produce inverse scaling of its action
    EXPECT_LT((7. * rescaled.solve(u) - bu).norm() / bu.norm(), 1e-12);
    // repeated calls must not reuse a previous right hand side as an initial guess
    EXPECT_EQ((solver.solve(u) - bu).norm(), 0.);
}

// compare all columns of a multilevel approximate inverse to a dense spectral oracle
TEST(SparseMultigrid, FullSmallInverseIsSymmetricPositive) {
    const auto a = grid(6, .1);
    const fdapde::SparseMultigrid solver(a, 2, 4);
    Eigen::MatrixXd inverse(a.rows(), a.rows());
    for (int i = 0; i < a.rows(); ++i) inverse.col(i) = solver.solve(Eigen::VectorXd::Unit(a.rows(), i));
    // checking the entire matrix catches asymmetry missed by a single probe
    EXPECT_LT((inverse - inverse.transpose()).norm() / inverse.norm(), 1e-12);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> spectrum(inverse);
    // every eigenvalue must be strictly positive for a valid linear chart
    EXPECT_GT(spectrum.eigenvalues().minCoeff(), 0.);
}

// recover independent constant modes and isolated screened degrees of freedom without merging components
TEST(SparseMultigrid, DisconnectedComponentsAndSingletons) {
    const auto block = grid(12, .001);
    const int n = block.rows();
    Eigen::SparseMatrix<double> a(2 * n + 3, 2 * n + 3);
    for (int j = 0; j < block.outerSize(); ++j)
        for (Eigen::SparseMatrix<double>::InnerIterator it(block, j); it; ++it) {
            a.insert(it.row(), it.col()) = it.value();
            a.insert(n + it.row(), n + it.col()) = it.value();
        }
    for (int i = 0; i < 3; ++i) a.insert(2 * n + i, 2 * n + i) = i + 1.;
    Eigen::VectorXd exact = Eigen::VectorXd::Ones(a.rows());
    exact.segment(n, n).setConstant(-2.);
    const fdapde::SparseMultigrid solver(a, 4, 16);
    // known fields on disconnected blocks test coarse constants and singleton handling together
    EXPECT_LT((solver.solve(a * exact) - exact).norm() / exact.norm(), .01);
}

// reject malformed public inputs and indefinite coarse systems before using the map
TEST(SparseMultigrid, RejectsInvalidInputs) {
    Eigen::SparseMatrix<double> a(2, 2);
    a.insert(0, 0) = 1.;
    a.insert(1, 1) = -1.;
    // exact coarse factorization detects this indefinite diagonal matrix
    EXPECT_THROW(fdapde::SparseMultigrid bad(a), std::domain_error);
    a.coeffRef(1, 1) = 1.;
    a.insert(0, 1) = .5;
    // asymmetric input violates the SPD map contract
    EXPECT_THROW(fdapde::SparseMultigrid bad(a), std::invalid_argument);
    a.coeffRef(0, 1) = 0.;
    // zero cycles would create a singular map
    EXPECT_THROW(fdapde::SparseMultigrid bad(a, 0), std::invalid_argument);
    const fdapde::SparseMultigrid solver(a);
    // a mismatched right hand side must not reach Eigen's internal assertions
    EXPECT_THROW(solver.solve(Eigen::VectorXd::Zero(3)), std::invalid_argument);
}
