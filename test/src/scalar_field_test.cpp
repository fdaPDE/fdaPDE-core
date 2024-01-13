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
#include <functional>
#include <vector>

#include <fdaPDE/utils.h>
#include <fdaPDE/fields.h>
using fdapde::core::DifferentiableScalarField;
using fdapde::core::ScalarField;
using fdapde::core::TwiceDifferentiableScalarField;
using fdapde::core::VectorField;
using fdapde::core::MatrixField;
using fdapde::core::DiscretizedScalarField;
using fdapde::Dynamic;

#include "utils/constants.h"
using fdapde::testing::DOUBLE_TOLERANCE;

// check if ScalarField class wraps lambda correctly
TEST(scalar_field_test, define_from_lambda) {
    // define field expression
    auto fieldExpr = [](SVector<2> x) -> double {   // e^x + x^2*y*log(y)
        return std::exp(x[0]) + std::pow(x[0], 2) * x[1] * std::log(x[1]);
    };

    // build the ScalarField object
    ScalarField<2> field(fieldExpr);
    // test if the ScalarField wraps correctly the lambda
    SVector<2> p(1, 1);
    double result = std::exp(1);
    // expect equality
    EXPECT_EQ(field(p), result);

    // initialize directly with a lambda
    ScalarField<2> lambda_field([](SVector<2> x) -> double { return std::pow(x[0], 2) + x[1]; });
    result = 2;
    EXPECT_EQ(lambda_field(p), result);
}

// checks if ScalarField approximates correctly its analytical gradient
TEST(scalar_field_test, gradient_approximation) {
    auto fieldExpr = [](SVector<2> x) -> double {   // [e^(2x+y)]/x
        return std::exp(2 * x[0] + x[1]) / x[0];
    };
    // build the ScalarField object
    ScalarField<2> field(fieldExpr);
    field.set_step(0.01);
    // get the gradient function
    auto grad = field.derive();
    // define evaluation point and true gradient vector
    SVector<2> p(1, 1);
    SVector<2> gradient = SVector<2>(std::exp(3), std::exp(3));
    // test if the obtained gradient is approximated correctly (truncation error is of order O(h^2), h is the step size)
    EXPECT_TRUE((grad(p) - gradient).squaredNorm() < std::pow(0.01, 2));

    // access to gradient approximation directly, without passing to a VectorField object
    SVector<2> approxGradient = field.derive()(p);
    EXPECT_TRUE((approxGradient - gradient).squaredNorm() < std::pow(0.01, 2));
}

// checks if ScalarField approximates correctly its analytical hessian
TEST(scalar_field_test, hessian_approximation) {
    auto fieldExpr = [](SVector<2> x) -> double {   // e^x + x^2*y*log(y)
        return std::exp(x[0]) + std::pow(x[0], 2) * x[1] * std::log(x[1]);
    };
    // build the ScalarField object
    ScalarField<2> field(fieldExpr);
    field.set_step(0.01);
    // get the hessian function
    auto hess = field.derive_twice();
    // define evaluation point and true hessian matrix
    SVector<2> p(1, 1);
    SMatrix<2> hessian;
    hessian << std::exp(1), 2, 2, 1;
    // test if the obtained gradient is approximated correctly (truncation error is of order O(h^2), h is the step size)
    EXPECT_TRUE((hess(p) - hessian).squaredNorm() < std::pow(0.01, 2));
}

// check if ScalarField handles a discontinuous function correctly
TEST(scalar_field_test, discontinuous_field) {
    // define field expression
    auto fieldExpr = [](SVector<2> x) -> double {   // I_{x>0, y>0}
        if (x[0] > 0 && x[1] > 0) return 1;
        return 0;
    };

    // build the ScalarField object
    ScalarField<2> field(fieldExpr);
    field.set_step(0.01);
    // test if the ScalarField wraps correctly the lambda
    SVector<2> p(1, 1);
    double result = 1;

    // expect equality
    EXPECT_EQ(field(p), result);

    auto grad = field.derive();
    // define true gradient vector
    SVector<2> gradient = SVector<2>(0, 0);
    // test if the obtained gradient is approximated correctly (truncation error is of order O(h^2), h is the step size)
    EXPECT_TRUE((grad(p) - gradient).squaredNorm() < std::pow(0.01, 2));

    // get the hessian function
    auto hess = field.derive_twice();
    SMatrix<2> hessian;
    hessian << 0, 0, 0, 0;
    // test if the obtained gradient is approximated correctly (truncation error is of order O(h^2), h is the step size)
    EXPECT_TRUE((hess(p) - hessian).squaredNorm() < std::pow(0.01, 2));
}

// check expression template mechanism for ScalarField
TEST(scalar_field_test, static_expressions) {
    // define two scalar fields sf1 and sf2
    auto fieldExpr1 = [](SVector<2> x) -> double {   // x^3 + y
        return std::pow(x[0], 3) + x[1];
    };
    ScalarField<2> sf1(fieldExpr1);
    auto fieldExpr2 = [](SVector<2> x) -> double {   // [e^(2x+y)]/x
        return std::exp(2 * x[0] + x[1]) / x[0];
    };
    ScalarField<2> sf2(fieldExpr2);
    // define evaluation point
    SVector<2> p(1, 1);

    // evaluate fields at point p
    double x1 = sf1(p);
    double x2 = sf2(p);

    // build various expressions and test for equality
    auto sf3 = sf1 + sf2;
    ASSERT_DOUBLE_EQ(sf3(p), (x1 + x2));
    auto sf4 = sf1 - sf2;
    ASSERT_DOUBLE_EQ(sf4(p), (x1 - x2));
    auto sf5 = sf1 * sf2;
    ASSERT_DOUBLE_EQ(sf5(p), (x1 * x2));
    auto sf6 = sf1 / sf2;
    ASSERT_DOUBLE_EQ(sf6(p), (x1 / x2));
    // combine many previous expressions and scalar values
    auto sf7 = sf1 + sf2 / sf3 - sf2 * 2 + 5;
    ASSERT_DOUBLE_EQ(sf7(p), (x1 + x2 / (x1 + x2) - x2 * 2 + 5));

    // define a discontinuous field and combine with a polynomial field via a summation
    std::function<double(SVector<2>)> poly = [](SVector<2> x) -> double { return x[0] + x[1]; };
    ScalarField<2> polyField(poly);
    // this evaluates 2 in (1,1)
    std::function<double(SVector<2>)> step = [](SVector<2> x) -> double {
        if (x[0] > 0 && x[1] > 0) return 1;
        return std::pow(x[0], 2);
    };
    ScalarField<2> stepField(step);
    // this evaluates 1 in (1,1)

    auto sf8 = polyField + stepField;
    ASSERT_DOUBLE_EQ(sf8(p), 3);
}

// use case for wrapping the result of a field expression in a valid ScalarField
TEST(scalar_field_test, define_from_expression) {
    // define a field expression
    auto fieldExpr1 = [](SVector<2> x) -> double {   // x^3 + y
        return std::pow(x[0], 3) + x[1];
    };
    ScalarField<2> sf1(fieldExpr1);
    auto fieldExpr2 = [](SVector<2> x) -> double {   // [e^(2x+y)]/x
        return std::exp(2 * x[0] + x[1]) / x[0];
    };
    ScalarField<2> sf2(fieldExpr2);

    auto sf3 = sf1 + sf2 / sf1;   // x^3 + y + (e^(2x+y))/(x^4 + xy)

    // wrap sf3 in a scalar field
    auto fieldExpr3 = [=](SVector<2> x) -> double { return sf3(x); };
    ScalarField<2> sf4(fieldExpr3);
    sf4.set_step(1e-2);
    
    // check expression is wrapped correctly
    SVector<2> p(1, 1);
    EXPECT_EQ(sf4(p), sf1(p) + sf2(p) / sf1(p));

    VectorField<2> grad = sf4.derive();
    // exact gradient evaluated at p
    SVector<2> gradient = SVector<2>(3 - 0.25 * std::exp(3), 1 + 0.25 * std::exp(3));
    EXPECT_TRUE((grad(p) - gradient).squaredNorm() < std::pow(0.01, 2));
}

// checks if DEF_FIELD_UNARY_OPERATOR and DEF_FIELD_UNARY_FUNCTOR defines valid operators
TEST(scalar_field_test, unary_operators) {
    // define a scalar field
    auto fieldExpr1 = [](SVector<2> x) -> double {   // x^3 + y
        return std::pow(x[0], 3) + x[1];
    };
    ScalarField<2> sf1(fieldExpr1);
    // define a new scalar field as the application of sin to sf1
    ScalarField<2> sf2 = sin(sf1);
    // define evaluation point
    SVector<2> p(1, 1);
    // field sf1 evaluated in (1,1) gives 2, sf2 expects equal to sin(2) ~ 0,03489949670250097165
    EXPECT_DOUBLE_EQ(std::sin(2), sf2(p));

    // apply sin function to a field expression
    auto sf3 = sin(2 * sf1 + sf2) + sf1;
    // evaluate it at p. expected value: sin(2*2 + sin(2)) + 2
    EXPECT_DOUBLE_EQ(std::sin(2 * 2 + std::sin(2)) + 2, sf3(p));
}

TEST(scalar_field_test, unary_negation) {
    // define field
    ScalarField<2> f;
    f = [](const SVector<2> x) -> double { return x[0] + x[1]; };
    // define evaluation point
    SVector<2> p({1, 1});
    EXPECT_EQ((-f)(p), -2);

    // define negation of expression
    auto g = 2 * f + 1;
    EXPECT_EQ((-g)(p), -5);

    // use negation in an expression
    auto h = -f + 2 * f + 4;
    EXPECT_EQ((-h)(p), -6);
}

// checks DifferentiableScalarField calls analytical gradient on .derive() call
TEST(scalar_field_test, differentiable_field) {
    // define analytical gradient expression
    std::function<double(SVector<2>)> baseField = [](SVector<2> x) -> double {   // x^3 + y
        return std::pow(x[0], 3) + x[1];
    };
    // wrap baseField into a scalar field
    ScalarField<2> sf(baseField);
    // build explicitly the gradient field using list initialization
    std::function<double(SVector<2>)> dx = [](SVector<2> x) -> double { return 3 * std::pow(x[0], 2); };
    std::function<double(SVector<2>)> dy = [](SVector<2> x) -> double { return 1; };
    DifferentiableScalarField<2> df_(baseField, std::vector<decltype(dx)> {dx, dy});

    // define evaluation point
    SVector<2> p(1, 1);
    SVector<2> gradient = SVector<2>(3, 1);
    // check derive() calls exact gradient
    ASSERT_FALSE((sf.derive()(p) - df_.derive()(p)).norm() < DOUBLE_TOLERANCE);
    // check exact gradient is actually called (should be equal to analytical gradient)
    ASSERT_TRUE((df_.derive()(p) - gradient).norm() < DOUBLE_TOLERANCE);
}

// checks TwiceDifferentiableScalarField calls analytical hessian on .deriveTwice() call
TEST(scalar_field_test, twice_differentiable_field) {
    // define analytical gradient and hessian expression
    std::function<double(SVector<2>)> baseField = [](SVector<2> x) -> double {   // x^3 + y
        return std::pow(x[0], 3) + x[1];
    };
    std::function<double(SVector<2>)> dx = [](SVector<2> x) -> double { return 3 * std::pow(x[0], 2); };
    std::function<double(SVector<2>)> dy = [](SVector<2> x) -> double { return 1; };
    MatrixField<2> hess;
    hess(0,0) = [](SVector<2> x) -> double { return 6 * x[0]; };
    hess(1,0) = [](SVector<2> x) -> double { return 0; };
    hess(0,1) = [](SVector<2> x) -> double { return 0; };
    hess(1,1) = [](SVector<2> x) -> double { return 0; };
    
    // wrap all in a twice differentiable scalar field
    ScalarField<2> sf(baseField);
    TwiceDifferentiableScalarField<2> tdf(baseField, std::vector<decltype(dx)> {dx, dy}, hess);
    // define evaluation point
    SVector<2> p(1, 1);
    SMatrix<2> hessian;
    hessian << 6, 0, 0, 0;
    // check exact hessian is actually called
    ASSERT_TRUE((tdf.derive_twice()(p) - hessian).norm() < DOUBLE_TOLERANCE);
}

TEST(scalar_field_test, discretized_scalar_field) {
    // define a scalar field
    ScalarField<2> f;
    f = [](SVector<2> x) -> double { return x[0] + x[1]; };

    // define vector of data
    DMatrix<double, Eigen::RowMajor> data;
    data.resize(10, 1);
    for (std::size_t i = 0; i < 10; i++) { data(i, 0) = i; }
    // wrap data into a field
    DiscretizedScalarField<2> k(data);

    auto sf = f + k;
    sf.forward(4);   // k = 4
    // define evaluation point
    SVector<2> p(1, 1);
    double eval = sf(p);

    EXPECT_EQ(eval, 6.0);
}

TEST(scalar_field_test, dynamic_expressions) {
    // define dynamic scalar fields
    ScalarField<Dynamic> f(4);
    f = [](DVector<double> x) -> double { return x[0] + x[1]; };
    ScalarField<Dynamic> g(4);
    g = [](DVector<double> x) -> double {   // implements \sum_i x_i
        double value = 0;
        for (int i = 0; i < x.rows(); ++i) { value += x[i]; }
        return value;
    };
    // evaluation point
    DVector<double> p(4);
    p << 1, 1, 2, 2;
    // build various expressions and test for equality
    auto h1 = f + 2 * g + 5;
    EXPECT_DOUBLE_EQ(h1(p), 19.0);
    auto h2 = sin(h1) / 2;
    EXPECT_DOUBLE_EQ(h2(p), std::sin(19) / 2);
    auto h3 = -h2;
    EXPECT_DOUBLE_EQ(h3(p), -std::sin(19) / 2);
    auto h4 = h1 - h2 + h3;
    EXPECT_DOUBLE_EQ(h4(p), 19.0 - std::sin(19));
}

TEST(scalar_field_test, expression_differentials_approximation) {
    // define scalar expression and extract its gradient and hessian
    ScalarField<2> f;
    f = [](SVector<2> x) -> double { return x[0] + x[1]; };
    auto expr = sin(f) + 2 * f;
    expr.set_step(1e-4);
    auto grad = expr.derive();
    auto hess = expr.derive_twice();
    // evaluation point
    SVector<2> p(1,1);

    SVector<2> gradient;
    gradient << std::cos(2) + 2, std::cos(2) + 2;
    ASSERT_TRUE((grad(p) - gradient).norm() < DOUBLE_TOLERANCE);

    SMatrix<2> hessian;
    hessian << -std::sin(2), -std::sin(2), -std::sin(2), -std::sin(2);
    ASSERT_TRUE((hess(p) - hessian).norm() < DOUBLE_TOLERANCE);
}

double f_impl(SVector<2> x) { return x[0]; }

struct g_container { int i = 0; };
struct g_struct {
private:
    g_container& c_;
public:
    g_struct(g_container& c) : c_(c) {};
    double g_impl(SVector<2> x) {
        c_.i++;
        return 1 + x[1] + c_.i;
    }
    double g_impl_const(SVector<2> x) const { return 1 + x[1]; }
};

TEST(scalar_field_test, free_and_member_function_pointers) {
    // construct from a free function pointer
    ScalarField<2, double (*)(SVector<2>)> f(&f_impl);
    // evaluation point
    SVector<2> p(1, 1);
    EXPECT_DOUBLE_EQ(f(p), 1);
    auto g = 2 * f + f;
    EXPECT_DOUBLE_EQ(g(p), 3);

    // construct from a non-const member function pointer
    g_container c;
    g_struct h_obj(c);
    ScalarField<2, double (g_struct::*)(SVector<2>)> h(&h_obj, &g_struct::g_impl);
    EXPECT_DOUBLE_EQ(h(p), 3);
    // state of h_obj changes between function calls (pointer to non-const member function)
    EXPECT_DOUBLE_EQ(h(p), 4);
    // construct from a const member function pointer
    ScalarField<2, double (g_struct::*)(SVector<2>) const> h_const(&h_obj, &g_struct::g_impl_const);
    EXPECT_DOUBLE_EQ(h_const(p), 2);

    // check approximated derivatives
    auto dh = h_const.derive();
    EXPECT_DOUBLE_EQ(dh(p)[0], 0);
    EXPECT_DOUBLE_EQ(dh(p)[1], 1);
}
