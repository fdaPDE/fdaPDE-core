#include <fdaPDE/finite_elements.h>
#include <gtest/gtest.h>

#include <cmath>

namespace fdapde {
namespace {

template <typename Quadrature> double integrate_monomial(int degree) {
    double value = 0;
    for (int i = 0; i < Quadrature::order; ++i) {
        value += Quadrature::weights[i] * std::pow(Quadrature::nodes[i], degree);
    }
    return value;
}

template <typename Quadrature> void expect_exact_moments() {
    for (int degree = 0; degree <= Quadrature::degree; ++degree) {
        EXPECT_NEAR(integrate_monomial<Quadrature>(degree), 1.0 / (degree + 1), 2e-14);
    }
}

}   // namespace

TEST(fe_quadrature, high_order_gauss_legendre_rules_integrate_all_exact_moments) {
    expect_exact_moments<internals::fe_quadrature_simplex<1, 5>>();
    expect_exact_moments<internals::fe_quadrature_simplex<1, 6>>();
}

TEST(dg_quadrature, high_order_traces_select_sufficient_face_degree) {
    using DG4FaceQuadrature = FeDG<4, 1>::face_quadrature_t<2>;
    using DG5FaceQuadrature = FeDG<5, 1>::face_quadrature_t<2>;

    static_assert(DG4FaceQuadrature::order == 5);
    static_assert(DG5FaceQuadrature::order == 6);
    static_assert(DG4FaceQuadrature::degree >= 2 * FeDG<4, 1>::order);
    static_assert(DG5FaceQuadrature::degree >= 2 * FeDG<5, 1>::order);

    EXPECT_NEAR(integrate_monomial<DG4FaceQuadrature>(8), 1.0 / 9.0, 2e-14);
    EXPECT_NEAR(integrate_monomial<DG5FaceQuadrature>(10), 1.0 / 11.0, 2e-14);
}

TEST(fe_quadrature, high_order_one_dimensional_aliases_are_exposed) {
    static_assert(decltype(QS1DP9)::degree == 9);
    static_assert(decltype(QS1DP11)::degree == 11);
    EXPECT_EQ(QS1DP9.order, 5);
    EXPECT_EQ(QS1DP11.order, 6);
}

}   // namespace fdapde
