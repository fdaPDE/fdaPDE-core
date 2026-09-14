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

#include <fdaPDE/core.h>
#include <gtest/gtest.h>

namespace {
// checks cardinality and partition of unity through the current FEM Vandermonde solve
template <int Dim, int Order> void check_basis() {
    constexpr typename fdapde::FeP<Order, 1>::template cell_dof_descriptor<Dim> dofs;
    constexpr fdapde::LagrangeBasis<Dim, Order> basis(dofs.dofs_phys_coords());
    for (int i = 0; i < basis.size(); ++i) {
        for (int j = 0; j < basis.size(); ++j) {
            // evaluates every basis function at every interpolation node against the Kronecker delta
            EXPECT_NEAR(basis[i](dofs.dofs_phys_coords().row(j).transpose()), i == j ? 1.0 : 0.0, 1e-12);
        }
    }
    fdapde::Vector<double, Dim> point;
    for (int d = 0; d < Dim; ++d) point[d] = 0.1 * (d + 1);
    double sum = 0;
    for (int i = 0; i < basis.size(); ++i) sum += basis[i](point);
    // the nodal basis must sum to one away from the interpolation nodes too
    EXPECT_NEAR(sum, 1.0, 1e-12);
}
}   // namespace

// exercises constexpr P1 and P2 basis construction in each supported simplex dimension
TEST(DenseCallers, LagrangeBasesPreserveCardinality) {
    check_basis<1, 1>();
    check_basis<1, 2>();
    check_basis<2, 1>();
    check_basis<2, 2>();
    check_basis<3, 1>();
    check_basis<3, 2>();
}

// compares current P1 assembly with exact element matrices on a non-axis-aligned triangle
TEST(DenseCallers, TriangleMassAndStiffnessPreserveCoefficients) {
    Eigen::Matrix<double, 3, 2> nodes;
    nodes << 0, 0, 2, 1, 1, 3;
    Eigen::Matrix<int, 1, 3> cells;
    cells << 0, 1, 2;
    Eigen::Matrix<int, 3, 1> boundary = Eigen::Matrix<int, 3, 1>::Ones();
    fdapde::Triangulation<2, 2> domain(nodes, cells, boundary);
    fdapde::FeSpace space(domain, fdapde::P1<1>);
    fdapde::TrialFunction u(space);
    fdapde::TestFunction v(space);
    auto mass = fdapde::integral(domain)(u * v).assemble();
    auto stiffness = fdapde::integral(domain)(fdapde::dot(fdapde::grad(u), fdapde::grad(v))).assemble();
    const Eigen::Vector2d point(1.0, 4.0 / 3.0);
    for (int i = 0; i < 3; ++i) {
        // the physical centroid has equal barycentric coordinates
        EXPECT_NEAR(space.eval_cell_value(i, point), 1.0 / 3.0, 1e-12);
    }
    const auto gradient = space.eval_cell_grad(0, point);
    // maps the first reference gradient using the inverse physical Jacobian
    EXPECT_NEAR(gradient[0], -0.4, 1e-12);
    // checks the second physical component after crossing the Eigen boundary
    EXPECT_NEAR(gradient[1], -0.2, 1e-12);
    constexpr double area = 2.5;
    const double gradients[3][2] = {
      {-0.4, -0.2},
      {0.6,  -0.2},
      {-0.2, 0.4 }
    };
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            // integrates lambda_i lambda_j exactly over the physical triangle
            EXPECT_NEAR(mass.coeff(i, j), area * (i == j ? 2 : 1) / 12.0, 1e-12);
            // compares each stiffness entry with the dot product of analytic physical gradients
            EXPECT_NEAR(
              stiffness.coeff(i, j), area * (gradients[i][0] * gradients[j][0] + gradients[i][1] * gradients[j][1]),
              1e-12);
        }
    }
}

// checks the current P2 assembly against polynomial moments and constant-field nullspace
TEST(DenseCallers, QuadraticTrianglePreservesMassAndStiffness) {
    Eigen::Matrix<double, 3, 2> nodes;
    nodes << 0, 0, 2, 1, 1, 3;
    Eigen::Matrix<int, 1, 3> cells;
    cells << 0, 1, 2;
    Eigen::Matrix<int, 3, 1> boundary = Eigen::Matrix<int, 3, 1>::Ones();
    fdapde::Triangulation<2, 2> domain(nodes, cells, boundary);
    fdapde::FeSpace space(domain, fdapde::P2<1>);
    fdapde::TrialFunction u(space);
    fdapde::TestFunction v(space);
    auto mass = fdapde::integral(domain)(u * v).assemble();
    auto stiffness = fdapde::integral(domain)(fdapde::dot(fdapde::grad(u), fdapde::grad(v))).assemble();
    // three vertex and three edge degrees of freedom are retained
    ASSERT_EQ(mass.rows(), 6);
    // partition of unity integrates to the physical triangle area
    EXPECT_NEAR(mass.sum(), 2.5, 1e-12);
    const double gradient_norms[3] = {0.2, 0.4, 0.2};
    for (int i = 0; i < 6; ++i) {
        // exact quartic moments distinguish the vertex and edge basis functions
        EXPECT_NEAR(mass.coeff(i, i), i < 3 ? 2.5 / 30.0 : 2.5 * 8.0 / 45.0, 1e-12);
        double row_sum = 0;
        for (int j = 0; j < 6; ++j) row_sum += stiffness.coeff(i, j);
        // a constant finite-element function has zero gradient
        EXPECT_NEAR(row_sum, 0.0, 1e-11);
        if (i < 3) {
            // the integrated squared vertex gradient is area times the linear gradient norm
            EXPECT_NEAR(stiffness.coeff(i, i), 2.5 * gradient_norms[i], 1e-12);
        }
    }
}

// verifies mixed signed and unsigned extents keep the same row and column indexing
TEST(DenseCallers, MixedExtentTypesPreserveLayout) {
    fdapde::MdArray<double, fdapde::MdExtents<fdapde::Dynamic, fdapde::Dynamic>, fdapde::ColMajor> values(
      std::size_t(2), 3);
    values(1, 2) = 7;
    // column-major position five is row one of column two
    EXPECT_DOUBLE_EQ(values.data()[5], 7);
}

// checks the existing column storage and null-mask consumer against the new MdArray views
TEST(DenseCallers, DataColumnsPreserveValuesAndMissingMask) {
    fdapde::internals::scalar_data_layer data;
    data.append_vec("value", std::vector<double> {1, std::numeric_limits<double>::quiet_NaN(), 3});
    auto column = data.col<double>("value");
    // the first value survives construction through a strided multidimensional slice
    EXPECT_DOUBLE_EQ(column(0, 0), 1);
    // the final value retains its original row index
    EXPECT_DOUBLE_EQ(column(2, 0), 3);
    column = std::vector<double> {2, std::numeric_limits<double>::quiet_NaN(), 4};
    // assignment writes through the current column view
    EXPECT_DOUBLE_EQ(column(2, 0), 4);
    data.append_vec("second", std::vector<double> {5, 6, 7});
    auto refreshed = data.col<double>("value");
    // growing the shared storage preserves the previous column values
    EXPECT_DOUBLE_EQ(refreshed(0, 0), 2);
    auto selected = data.select({0, 2}).col<double>("value").data();
    // the selected final row follows the first row in copied multidimensional storage
    EXPECT_DOUBLE_EQ(selected(1, 0), 4);
    auto mask = data.nan(std::string("value"));
    // only the middle row carries the missing-value marker
    EXPECT_TRUE(mask(1, 0));
    // an ordinary numeric value must remain unmarked
    EXPECT_FALSE(mask(0, 0));
}

// verifies that the grid view preserves Eigen storage order and flat-vector row order
TEST(DenseCallers, GridSearchPreservesGridCoordinates) {
    Eigen::Matrix<double, 3, 2, Eigen::ColMajor> columns;
    columns << 8, 9, 1, 2, 4, 5;
    const Eigen::Matrix<double, 3, 2, Eigen::RowMajor> rows = columns;
    const std::vector<double> flat {8, 9, 1, 2, 4, 5};
    auto objective = [](const Eigen::Vector2d& point) { return point.squaredNorm(); };
    auto check = [&](const auto& grid) {
        fdapde::GridSearch<2> search;
        const auto optimum = search.optimize(objective, grid);
        // the second grid row minimizes the objective for every physical storage order
        EXPECT_DOUBLE_EQ(optimum[0], 1);
        // both coordinates must come from the same selected row
        EXPECT_DOUBLE_EQ(optimum[1], 2);
        // every candidate row must be evaluated once
        EXPECT_EQ(search.values().size(), 3u);
    };
    check(columns);
    check(rows);
    check(flat);
}
