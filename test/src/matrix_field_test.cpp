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

#include <gtest/gtest.h>   // testing framework
#include <array>
#include <functional>

#include <fdaPDE/utils.h>
#include <fdaPDE/fields.h>
using fdapde::core::MatrixField;

// test if a matrix field can be built from a vector of lambdas (static/dynamic version)
TEST(matrix_field_test, define_from_lambda) {
    // define vector field using list initialization
    std::function<double(SVector<2>)> xx_comp = [](SVector<2> x) -> double { return 4 * x[0]; };
    std::function<double(SVector<2>)> xy_comp = [](SVector<2> x) -> double { return x[1]; };
    std::function<double(SVector<2>)> yx_comp = [](SVector<2> x) -> double { return x[0] + x[1]; };
    std::function<double(SVector<2>)> yy_comp = [](SVector<2> x) -> double { return 3; };
    MatrixField<2> field1({xx_comp, xy_comp, yx_comp, yy_comp});
    // define evaluation point
    SVector<2> p(1, 1);
    // field evaluates in p: [1,1,2,1]
    SMatrix<2> expected;
    expected << 4, 1, 2, 3;
    SMatrix<2> computed1 = field1(p);
    for (std::size_t i = 0; i < 2; ++i) {
        for (std::size_t j = 0; j < 2; ++j) EXPECT_DOUBLE_EQ(computed1(i, j), expected(i, j));
    }

    // test dynamic constructor
    MatrixField<fdapde::Dynamic> field2(2, 2, 2);
    field2(0, 0) = xx_comp;
    field2(0, 1) = xy_comp;
    field2(1, 0) = yx_comp;
    field2(1, 1) = yy_comp;
    DMatrix<double> computed2 = field2(p);
    for (std::size_t i = 0; i < 2; ++i) {
        for (std::size_t j = 0; j < 2; ++j) EXPECT_DOUBLE_EQ(computed2(i, j), expected(i, j));
    }
}

// check if a matrix field component can be assigned to any ScalarExpr on the rhs
TEST(matrix_field_test, assign_scalar_expr_rhs) {
    // define an empty matrix field
    MatrixField<2> m_field;
    // define a lambda expression valid to be wrapped by a ScalarField
    std::function<double(SVector<2>)> scalar_field = [](SVector<2> x) -> double {
        return std::log(x[0] + x[1]) - std::pow(x[0], 2) * x[1] + 5;
    };
    ScalarField<2> s_field(scalar_field);
    // set the field coordinates
    m_field(0, 0) = s_field;
    m_field(0, 1) = 2 * s_field + 1;   // built directly from a field expression
    m_field(1, 0) = m_field(0, 1);
    m_field(1, 1) = m_field(0, 0) + m_field(0, 1);
    // evaluation point
    SVector<2> p(1, 1);
    // evaluation of field at p: (ln(2) + 4, 2ln(2) + 9, 2ln(2) + 9, 3ln(2) + 13)
    SMatrix<2> eval;
    eval << std::log(2) + 4, 2 * std::log(2) + 9, 2 * std::log(2) + 9, 3 * std::log(2) + 13;
    for (std::size_t i = 0; i < 2; ++i) {
      for (std::size_t j = 0; j < 2; ++j) EXPECT_DOUBLE_EQ(m_field(i, j)(p), eval(i, j));
    }
}

// checks if the product between a matrix field and a costant vector is correct
TEST(matrix_field_test, product_with_svector) {
    // define matrix field
    std::function<double(SVector<2>)> comp = [](SVector<2> x) -> double { return 4 * x[0]; };
    MatrixField<2> field({comp, comp, comp, comp});
    // define constant vector
    SVector<2> v(2, 2);
    // build a functor representing the product between field and v
    auto f = field * v;
    // evaluation point
    SVector<2> p(1, 1);
    SVector<2> expected(16, 16); // evulation of field f at p: (16, 16)
    for (std::size_t i = 0; i < 2; ++i) { EXPECT_DOUBLE_EQ(f(p)[i], expected[i]); }
}

