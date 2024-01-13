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

// check if the product between a matrix field and a vector field is correct
TEST(matrix_field_test, product_with_vector_field) {
    // define matrix field
    std::function<double(SVector<2>)> comp = [](SVector<2> x) -> double { return 4 * x[0]; };
    MatrixField<2> m_field({comp, comp, comp, comp});
    // define vector field
    VectorField<2> v_field;
    v_field[0] = [](SVector<2> x) -> double { return std::pow(x[0], 2) + 1; };
    v_field[1] = [](SVector<2> x) -> double { return std::exp(x[0] * x[1]); };
    // build a functor representing the product between m_field and v_field
    auto f = m_field * v_field;
    // evaluation point
    SVector<2> p(1, 1);
    SVector<2> expected(8 + 4*std::exp(1), 8 + 4*std::exp(1)); // evulation of field f at p: (8+4*e, 8+4*e)
    for (std::size_t i = 0; i < 2; ++i) { EXPECT_DOUBLE_EQ(f(p)[i], expected[i]); }
}

// check matrix-matrix product
// TEST(matrix_field_test, matrix_matrix_product) {
//     // define matrix field
//     std::function<double(SVector<2>)> comp1 = [](SVector<2> x) -> double { return 4 * x[0]; };
//     MatrixField<2> field1({comp1, comp1, comp1, comp1});
//     // define constant matrix
//     SMatrix<2> M;
//     M << 1, 2, 3, 4;
//     // define evaluation point
//     SVector<2> p(1, 1);

//     // define different product expressions and check for correctness
//     auto f1 = field1 * M;
//     SMatrix<2> expected1;
//     expected1 << 16, 24, 16, 24;
//     for (std::size_t i = 0; i < 2; ++i) {
//       for (std::size_t j = 0; j < 2; ++j) { EXPECT_DOUBLE_EQ(f1(p)(i, j), expected1(i, j)); }
//     }
//     auto f2 = M * field1;
//     SMatrix<2> expected2;
//     expected2 << 12, 12, 28, 28;
//     for (std::size_t i = 0; i < 2; ++i) {
//       for (std::size_t j = 0; j < 2; ++j) { EXPECT_DOUBLE_EQ(f2(p)(i, j), expected2(i, j)); }
//     }

//     // MatrixField - MatrixField product
//     std::function<double(SVector<2>)> comp2 = [](SVector<2> x) -> double { return 5 * x[0]; };
//     std::function<double(SVector<2>)> comp3 = [](SVector<2> x) -> double { return 7 * x[0]; };
//     MatrixField<2> field2({comp3, comp3, comp2, comp2});
//     field1(0, 0) = comp2;

//     auto f3 = field1 * field2;
//     SMatrix<2> expected3;
//     expected3 << 55, 55, 48, 48;
//     for (std::size_t i = 0; i < 2; ++i) {
//       for (std::size_t j = 0; j < 2; ++j) { EXPECT_DOUBLE_EQ(f3(p)(i, j), expected3(i, j)); }
//     }
//     auto f4 = field2 * field1;
//     SMatrix<2> expected4;
//     expected4 << 63, 56, 45, 45;
//     for (std::size_t i = 0; i < 2; ++i) {
//       for (std::size_t j = 0; j < 2; ++j) { EXPECT_DOUBLE_EQ(f4(p)(i, j), expected4(i, j)); }
//     }
// }
