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
using fdapde::core::DotProduct;
using fdapde::core::ScalarField;
using fdapde::core::VectorField;
using fdapde::core::DiscretizedVectorField;

// test different constructors of VectorField
TEST(vector_field_test, define_from_lambda) {
    // define vector field using list initialization
    std::function<double(SVector<2>)> x_comp = [](SVector<2> x) -> double { return std::exp(x[0] * x[1]) + 2; };
    std::function<double(SVector<2>)> y_comp = [](SVector<2> x) -> double { return (std::pow(x[0], 2) + x[1]) / x[1]; };
    VectorField<2> field1({x_comp, y_comp});
    // define evaluation point
    SVector<2> p(1, 1);
    // field evaluates in p: (e+2, 2)
    SVector<2> trueEvaluation(std::exp(1) + 2, 2);
    for (std::size_t i = 0; i < 2; ++i) EXPECT_DOUBLE_EQ(field1(p)[i], trueEvaluation[i]);

    // define vector field explicitly declaring an array of lambdas
    std::vector<std::function<double(SVector<2>)>> comp_vect({x_comp, y_comp});
    VectorField<2> field2(comp_vect);
    for (std::size_t i = 0; i < 2; ++i) EXPECT_DOUBLE_EQ(field2(p)[i], trueEvaluation[i]);

    // if all asserts are verified by transitivity all the definitions give origin to the same object
}

// check if field obtained by subscript is the same as the one supplied in input
TEST(vector_field_test, const_subscript_operator) {
    // define vector field
    typedef std::function<double(SVector<2>)> field_component;
    field_component x_comp = [](SVector<2> x) -> double { return x[0] + 2; };
    field_component y_comp = [](SVector<2> x) -> double { return x[0] * x[1] / 4; };
    // wrap it in a VectorField
    VectorField<2> field1({x_comp, y_comp});

    // extract first dimension using const subscript
    ScalarField<2> x_extracted = field1[0];
    // evaluation point
    SVector<2> p(1, 1);
    EXPECT_DOUBLE_EQ(x_comp(p), x_extracted(p));
}

// check if a vector field component can be assigned to any ScalarExpr on the rhs
TEST(vector_field_test, assign_scalar_expr_rhs) {
    // define an empty vector field
    VectorField<2> v_field;
    // define a lambda expression valid to be wrapped by a ScalarField
    std::function<double(SVector<2>)> scalarField = [](SVector<2> x) -> double {
        return std::log(x[0] + x[1]) - std::pow(x[0], 2) * x[1] + 5;
    };
    ScalarField<2> s_field(scalarField);
    // set the field coordinates
    v_field[0] = scalarField;       // converting constructor ScalarField(std::function<double(SVector<2>)>) called
    v_field[1] = 2 * s_field + 1;   // built directly from a field expression

    // evaluation point
    SVector<2> p(1, 1);
    // evaluation of field at p: (ln(2) + 4, 2ln(2) + 9)
    SVector<2> eval(std::log(2) + 4, 2 * std::log(2) + 9);
    for (std::size_t i = 0; i < 2; ++i) { EXPECT_DOUBLE_EQ(v_field(p)[i], eval[i]); }
}

// checks if the inner product between two fields is correct
TEST(vector_field_test, dot_product_with_vector_field) {
    // define two vector fields
    VectorField<2> v_field1;
    v_field1[0] = [](SVector<2> x) -> double { return std::pow(x[0], 2) + 1; };
    v_field1[1] = [](SVector<2> x) -> double { return std::exp(x[0] * x[1]); };

    VectorField<2> v_field2;
    v_field2[0] = v_field1[0] + 2;
    v_field2[1] = 2 * v_field1[0] * v_field1[1];

    // build a vector field as the dot product of the previous two fields
    DotProduct dotProduct = v_field1.dot(v_field2);
    // evaluation point
    SVector<2> p(1, 1);
    // dot product functor evaluates same as dot product of evaluated fields
    EXPECT_DOUBLE_EQ(dotProduct(p), v_field1(p).dot(v_field2(p)));

    // dotProduct encodes the scalar field of equation: (x^2 + 1)*(x*2 + 3) + 2*e^{xy}*(x^2 + 1)*e^{xy}
    // dotProduct evaluates at p: (2)*(4) + 2*e^{2}*2 = 8 + 4*e^{2}
    EXPECT_DOUBLE_EQ(dotProduct(p), 8 + 4 * std::exp(2));
}

// checks if the inner product between a vector field and an SVector is correct
TEST(vector_field_test, dot_product_with_svector) {
    // vector field definition
    VectorField<2> field;
    field[0] = [](SVector<2> x) -> double { return std::pow(x[0], 2) + 1; };
    field[1] = [](SVector<2> x) -> double { return std::exp(x[0] * x[1]); };
    // define an SVector
    SVector<2> coeff(5, 2);
    // perform dot product
    auto dotProduct = field.dot(coeff);   // 5*(x^2 + 1) + 2*e^{xy}
    // evaluation point
    SVector<2> p(1, 1);
    EXPECT_DOUBLE_EQ(10 + 2 * std::exp(1), dotProduct(p));
}

// checks if the matrix * VectorField product returns a valid VectorField
TEST(vector_field_test, matrix_vector_field_product) {
    // vector field definition from single componentes
    std::function<double(SVector<2>)> x_comp = [](SVector<2> x) -> double {
        return std::exp(x[0] * x[1]) + 2;   // e^{xy} + 2
    };
    std::function<double(SVector<2>)> y_comp = [](SVector<2> x) -> double {
        return (std::pow(x[0], 2) + x[1]) / x[1];   // (x^2 + y)/y
    };

    VectorField<2> field({x_comp, y_comp});

    // define a coefficient matrix M
    SMatrix<2> M;
    M << 1, 2, 3, 4;
    // obtain product field M*field
    auto productField = M * field;   // (e^{xy} + 2 + 2*(x^2 + y)/y; 3*e^{xy} + 6 + 4*(x^2 + y)/y)
    // evaluation point
    SVector<2> p(1, 1);
    SVector<2> eval(std::exp(1) + 6, 3 * std::exp(1) + 14);
    for (std::size_t i = 0; i < 2; ++i) { EXPECT_DOUBLE_EQ(productField(p)[i], eval[i]); }
}

// check expression template mechanism for VectorField
TEST(vector_field_test, static_expressions) {
    // create vector field from assignment operator
    VectorField<2> vf1;
    vf1[0] = [](SVector<2> x) -> double { return std::pow(x[0], 2) + 1; };   // x^2 + 1
    vf1[1] = [](SVector<2> x) -> double { return std::exp(x[0] * x[1]); };   // e^{xy}

    // define vector field definying single components
    std::function<double(SVector<2>)> x_comp = [](SVector<2> x) -> double {
        return std::exp(x[0] * x[1]) + 2;   // e^{xy} + 2
    };
    std::function<double(SVector<2>)> y_comp = [](SVector<2> x) -> double {
        return (std::pow(x[0], 2) + x[1]) / x[1];   // (x^2 + y)/y
    };
    VectorField<2> vf2({x_comp, y_comp});

    // evaluation point
    SVector<2> p(1, 1);
    SVector<2> vf1_eval(2, std::exp(1));       // vf1 -> (2, e)
    SVector<2> vf2_eval(std::exp(1) + 2, 2);   // vf2 -> (e + 2, 2)

    // build various expressions and test for equality
    auto vf3 = vf1 + vf2;
    for (std::size_t i = 0; i < 2; ++i) EXPECT_DOUBLE_EQ(vf3(p)[i], (vf1_eval + vf2_eval)[i]);
    auto vf4 = vf1 - vf2;
    for (std::size_t i = 0; i < 2; ++i) EXPECT_DOUBLE_EQ(vf4(p)[i], (vf1_eval - vf2_eval)[i]);

    SMatrix<2> M;
    M << 1, 2, 3, 4;
    auto vf5 = M * vf1 - vf2 + vf3;
    for (std::size_t i = 0; i < 2; ++i) EXPECT_DOUBLE_EQ(vf5(p)[i], (M * vf1_eval + vf1_eval)[i]);

    // the following are ScalarField expressions coming from VectorField operations
    auto vf6 = vf1.dot(vf2) + vf3[0];
    EXPECT_DOUBLE_EQ(vf6(p), 5 * std::exp(1) + 8);
}

// check a vectorial expression can be wrapped in a VectorField
TEST(vector_field_test, define_from_expression) {
    // define some field expression
    VectorField<2> vf1;
    vf1[0] = [](SVector<2> x) -> double { return std::pow(x[0], 2) + 1; };   // x^2 + 1
    vf1[1] = [](SVector<2> x) -> double { return std::exp(x[0] * x[1]); };   // e^{xy}

    // define vector field definying single components
    std::function<double(SVector<2>)> x_comp = [](SVector<2> x) -> double {
        return std::exp(x[0] * x[1]) + 2;   // e^{xy} + 2
    };
    std::function<double(SVector<2>)> y_comp = [](SVector<2> x) -> double {
        return (std::pow(x[0], 2) + x[1]) / x[1];   // (x^2 + y)/y
    };
    VectorField<2> vf2({x_comp, y_comp});

    auto vf3 = vf1 + vf2;
    // convert the expression in a valid vector field
    VectorField<2> vf4(vf3);
    // evaluation point
    SVector<2> p(1, 1);
    SVector<2> vf1_eval(2, std::exp(1));       // vf1 -> (2, e)
    SVector<2> vf2_eval(std::exp(1) + 2, 2);   // vf2 -> (e + 2, 2)

    for (std::size_t i = 0; i < 2; ++i) EXPECT_DOUBLE_EQ(vf4(p)[i], (vf1_eval + vf2_eval)[i]);
}

// check definition of field with unequal sizes for domain and codomain space dimensions
TEST(vector_field_test, input_output_different_dimensions) {
    // define a field from a 2D space to a 3D space
    VectorField<2, 3> vf1;
    vf1[0] = [](SVector<2> x) -> double { return std::pow(x[0], 2) + 1; };   // x^2 + 1
    vf1[1] = [](SVector<2> x) -> double { return std::exp(x[0] * x[1]); };   // e^{xy}
    vf1[2] = [](SVector<2> x) -> double { return x[0] + x[1]; };             // x+y

    // evaluation point
    SVector<2> p(1, 1);
    // access to a single element of the field returns a 2D scalar field
    EXPECT_DOUBLE_EQ(vf1[0](p), 2);
    // check the entire field is evaluated correctly0
    SVector<3> vf1_eval(2, std::exp(1), 2);   // vf1 -> (2, e, 2)
    for (std::size_t i = 0; i < 3; ++i) EXPECT_DOUBLE_EQ(vf1[i](p), vf1_eval[i]);

    // define another field for better testing
    VectorField<2, 3> vf2;
    vf2[0] = [](SVector<2> x) -> double { return std::pow(x[0], 3) + std::pow(x[1], 2); };   // x^3 + y^2
    vf2[1] = [](SVector<2> x) -> double { return std::exp(x[0] * x[1]); };                   // e^{xy}
    vf2[2] = [](SVector<2> x) -> double { return x[0] + 1; };                                // x+1
    SVector<3> vf2_eval(2, std::exp(1), 2);   // evaluation of field vf2 in (1,1)

    // check expression template mechanism
    auto vf3 = vf1 + vf2;
    SVector<3> vf3_eval(4, 2 * std::exp(1), 4);
    for (std::size_t i = 0; i < 3; ++i) EXPECT_DOUBLE_EQ(vf3[i](p), vf3_eval[i]);

    auto vf5 = 2 * vf1;
    SVector<3> vf5_eval = 2 * vf1_eval;
    for (std::size_t i = 0; i < 3; ++i) EXPECT_DOUBLE_EQ(vf5[i](p), vf5_eval[i]);

    auto vf6 = vf1 + 2 * vf2;
    SVector<3> vf6_eval = vf1_eval + 2 * vf2_eval;
    for (std::size_t i = 0; i < 3; ++i) EXPECT_DOUBLE_EQ(vf6[i](p), vf6_eval[i]);

    // matrix-vectorfield product
    Eigen::Matrix<double, 2, 3> M;
    M << 1, 1, 2, 2, 3, 0;

    // expected field with equation
    //     1*(x^2 + 1) + 1*e^{xy} + 2*(x+y)
    //     2*(x^2 + 1) + 3*e^{xy}
    VectorField<2, 2> vf4 = M * vf1;
    SVector<2> vf4_eval(6 + std::exp(1), 4 + 3 * std::exp(1));
    for (std::size_t i = 0; i < 2; ++i) EXPECT_DOUBLE_EQ(vf4[i](p), vf4_eval[i]);

    VectorField<2> vf7;
    vf7[0] = [](SVector<2> x) -> double { return std::pow(x[0], 3) + std::pow(x[1], 2); };   // x^3 + y^2
    vf7[1] = [](SVector<2> x) -> double { return std::exp(x[0] * x[1]); };                   // e^{xy}
    Eigen::Matrix<double, 3, 2> K;
    K << 1, 1, 2, 2, 3, 0;

    // expected 2x3 vector field with equation
    //     x^3 + y^2 + e^{xy}
    //     2*(x^3 + y^2) + 2*e^{xy}
    //     3*(x^3 + y^2)
    auto vf8 = K * vf7;
    SVector<3> vf8_eval(2 + std::exp(1), 4 + 2 * std::exp(1), +6);
    for (std::size_t i = 0; i < 3; ++i) EXPECT_DOUBLE_EQ(vf8[i](p), vf8_eval[i]);

    // check dot product
    auto dotProduct = vf1.dot(vf2);
    // expected scalar field of expression: (x^2+1)*(x^3+y^2) + e^{2*xy} + (x+y)*(x+1)
    EXPECT_DOUBLE_EQ(dotProduct(p), 2 * 2 + std::exp(2) + 2 * 2);
}

TEST(vector_field_test, unary_negation) {
    // define some field expression
    VectorField<2> vf_1;
    vf_1[0] = [](SVector<2> x) -> double { return std::pow(x[0], 2) + 1; };   // x^2 + 1
    vf_1[1] = [](SVector<2> x) -> double { return 2 * x[0] + x[1]; };         // 2*x + y
    // define evaluation point and expected result
    SVector<2> p({1, 1});
    SVector<2> vf_p({2, 3});

    SVector<2> res_1 = (-vf_1)(p);
    for (std::size_t i = 0; i < 2; ++i) EXPECT_DOUBLE_EQ(res_1[i], -vf_p[i]);

    // define negation of expression
    auto vf_2 = 2 * vf_1;

    SVector<2> res_2 = (-vf_2)(p);
    for (std::size_t i = 0; i < 2; ++i) EXPECT_DOUBLE_EQ(res_2[i], -(2 * vf_p[i]));
}

TEST(vector_field_test, discretized_vector_field) {
    // define some field expression
    VectorField<2> vf;
    vf[0] = [](SVector<2> x) -> double { return std::pow(x[0], 2) + 1; };   // x^2 + 1
    vf[1] = [](SVector<2> x) -> double { return 2 * x[0] + x[1]; };         // 2*x + y

    // define vector of data
    DMatrix<double, Eigen::RowMajor> data;
    data.resize(10, 2);
    for (std::size_t i = 0; i < 10; i++) {
        data(i, 0) = i;
        data(i, 1) = i;
    }
    // wrap data into a field
    DiscretizedVectorField<2,2> k(data);

    auto vf_1 = vf + k;
    vf_1.forward(4);   // k = 4
    // define evaluation point
    SVector<2> p(1, 1);
    SVector<2> eval = vf_1(p);

    EXPECT_DOUBLE_EQ(eval[0], 6.0);
    EXPECT_DOUBLE_EQ(eval[1], 7.0);
}

TEST(vector_field_test, dynamic_inner_outer_size) {
    // define dynamic vector field from \mathbb{R}^3 to \mathbb{R}^3
    VectorField<Dynamic> vf(3, 3);
    vf[0] = [](DVector<double> x) -> double { return std::pow(x[0], 2) + 1; };   // x^2 + 1
    vf[1] = [](DVector<double> x) -> double { return 2 * x[0] + x[1]; };         // 2*x + y
    vf[2] = [](DVector<double> x) -> double { return std::exp(x[0] * x[2]); };   // e^{xz}

    // define evaluation point
    DVector<double> p(3);
    p << 1, 1, 1;
    SVector<3> vf_eval(2, 3, std::exp(1));
    for (std::size_t i = 0; i < 3; ++i) { EXPECT_DOUBLE_EQ(vf(p)[i], vf_eval[i]); }
}

TEST(vector_field_test, dynamic_expressions) {
    // define two dynamic vector fields from \mathbb{R}^2 to \mathbb{R}^3
    VectorField<Dynamic> vf1(2, 3);
    vf1[0] = [](DVector<double> x) -> double { return std::pow(x[0], 2) + 1; };   // x^2 + 1
    vf1[1] = [](DVector<double> x) -> double { return 2 * x[0] + x[1]; };         // 2*x + y
    vf1[2] = [](DVector<double> x) -> double { return x[0]; };                    // x

    VectorField<Dynamic> vf2(2, 3);
    vf2[0] = [](DVector<double> x) -> double { return 3 * x[0]; };                   // 3*x
    vf2[1] = [](DVector<double> x) -> double { return std::pow(x[1], 3) + x[1]; };   // y^3 + y
    vf2[2] = [](DVector<double> x) -> double { return x[0] * x[1]; };                // x*y

    // define evaluation point
    DVector<double> p(2);
    p << 1, 1;
    // build various expressions and test for equality
    auto vf3 = vf1 + vf2;
    SVector<3> vf3_eval(5, 5, 2);
    for (std::size_t i = 0; i < 3; ++i) { EXPECT_DOUBLE_EQ(vf3(p)[i], vf3_eval[i]); }
    auto vf4 = vf1 + 4*vf2;
    SVector<3> vf4_eval(14, 11, 5);
    for (std::size_t i = 0; i < 3; ++i) { EXPECT_DOUBLE_EQ(vf4(p)[i], vf4_eval[i]); }
    auto dot_product = vf1.dot(vf2);
    EXPECT_DOUBLE_EQ(dot_product(p), 13);
    ScalarField<Dynamic> f(2);
    f = [](DVector<double> x) -> double { return x[0] + x[1]; };
    VectorField<Dynamic> vf5(2,2);
    vf5[0] = vf1[0];
    vf5[1] = vf1[1];
    auto vf6 = vf5 + f.derive();
    SVector<2> vf6_eval(3, 4);
    for (std::size_t i = 0; i < 2; ++i) { EXPECT_DOUBLE_EQ(vf6(p)[i], vf6_eval[i]); }
}
