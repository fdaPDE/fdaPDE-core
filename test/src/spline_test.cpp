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

#include <fdaPDE/splines.h>
#include <fdaPDE/utils.h>
#include <gtest/gtest.h>   // testing framework

#include <unsupported/Eigen/SparseExtra>
using fdapde::core::Assembler;
using fdapde::core::bilaplacian;
using fdapde::core::GaussLegendre;
using fdapde::core::IntegratorTable;
using fdapde::core::reaction;
using fdapde::core::SPLINE;
using fdapde::core::SplineBasis;
using fdapde::core::spline_order;

#include "utils/utils.h"
using fdapde::testing::almost_equal;

// test definition of spline basis
TEST(spline_test, cox_de_boor_definition) {
    // define vector of equidistant knots on unit interval [0,1]
    DVector<double> knots;
    knots.resize(11);
    std::size_t i = 0;
    for (double x = 0; x <= 1; x += 0.1, ++i) knots[i] = x;

    // define cubic B-spline basis over [0,1]
    SplineBasis<3> basis(knots);

    SpMatrix<double> expected;
    Eigen::loadMarket(expected, "../data/mtx/spline_test_1.mtx");

    for (std::size_t i = 0; i < 1; ++i) {
        // evaluate i-th spline over [0,1]
        std::vector<double> result;
        result.reserve(101);
        for (double x = 0; x <= 1.01; x += 0.01) { result.push_back(basis[i](SVector<1>(x))); }
        // check results within double tolerance
        for (std::size_t j = 0; j < result.size(); ++j) { EXPECT_TRUE(almost_equal(result[j], expected.coeff(j, i))); }
    }
}

// test definition of spline basis derivative
TEST(spline_test, cubic_spline_second_derivative) {
    // define vector of equidistant knots on unit interval [0,1]
    DVector<double> knots;
    knots.resize(11);
    std::size_t i = 0;
    for (double x = 0; x <= 1; x += 0.1, ++i) knots[i] = x;

    // define cubic B-spline basis over [0,1]
    SplineBasis<3> basis(knots);

    SpMatrix<double> expected;
    Eigen::loadMarket(expected, "../data/mtx/spline_test_2.mtx");

    for (std::size_t i = 0; i < basis.size(); ++i) {
        // evaluate i-th spline over [0,1]
        std::vector<double> result;
        result.reserve(101);
        for (double x = 0; x <= 1.01; x += 0.01) { result.push_back(basis[i].derive<2>()(SVector<1>(x))); }
        // check results within double tolerance
        for (std::size_t j = 0; j < result.size(); ++j) { EXPECT_TRUE(almost_equal(result[j], expected.coeff(j, i))); }
    }
}

TEST(spline_test, cubic_spline_reaction_operator) {
    Mesh<1, 1> unit_interval(0, 2, 10);
    // define PDE
    auto L = reaction<SPLINE>(1.0);
    PDE<Mesh<1, 1>, decltype(L), DMatrix<double>, SPLINE, spline_order<3>> pde_(unit_interval, L);
    pde_.init();
    // load expected data from file
    SpMatrix<double> expected;
    Eigen::loadMarket(expected, "../data/mtx/spline_test_3.mtx");
    EXPECT_TRUE(almost_equal(expected, pde_.stiff()));
}

TEST(spline_test, cubic_spline_bilaplacian_operator) {
    Mesh<1, 1> unit_interval(0, 2, 10);
    // define PDE
    auto L = -bilaplacian<SPLINE>();
    PDE<Mesh<1, 1>, decltype(L), DMatrix<double>, SPLINE, spline_order<3>> pde_(unit_interval, L);
    pde_.init();
    // load expected data from file
    SpMatrix<double> expected;
    Eigen::loadMarket(expected, "../data/mtx/spline_test_4.mtx");
    EXPECT_TRUE(almost_equal(expected, pde_.stiff()));
}


