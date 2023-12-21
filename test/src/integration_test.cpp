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
#include <cstddef>

#include <fdaPDE/utils.h>
#include <fdaPDE/mesh.h>
#include <fdaPDE/linear_algebra.h>
#include <fdaPDE/finite_elements.h>
using fdapde::core::Element;
using fdapde::core::Integrator;
using fdapde::core::IntegratorTable;
using fdapde::core::LagrangianBasis;
using fdapde::core::FEM;

#include "utils/mesh_loader.h"
using fdapde::testing::MESH_TYPE_LIST;
using fdapde::testing::MeshLoader;
#include "utils/utils.h"
using fdapde::testing::almost_equal;

// test fixture
template <typename E> struct integration_test : public ::testing::Test {
    MeshLoader<E> meshLoader {};   // use default mesh
    static constexpr unsigned int M = MeshLoader<E>::M;
    static constexpr unsigned int N = MeshLoader<E>::N;
};
TYPED_TEST_SUITE(integration_test, MESH_TYPE_LIST);

// tests if the integration of the constant field 1 over an element equals its measure
TYPED_TEST(integration_test, constant_unitary_field) {
    // generate random element from mesh
    auto e = this->meshLoader.generate_random_element();
    Integrator<FEM, TestFixture::M, 1> integrator;   // define integrator
    // the integral of the constant field 1 over the mesh element equals its measure
    std::function<double(SVector<TestFixture::N>)> f = [](SVector<TestFixture::N> x) -> double { return 1; };
    EXPECT_TRUE(almost_equal(e.measure(), integrator.integrate(e, f)));
}

// test if linear fields can be integrated over mesh elements. In particular a closed formula for
// the volume of a truncated prism defined over an element e having height h1, h2, ..., hm at the m vertices is known as
//     e.measure()*(h1 + h2 + ... hm)/m
TYPED_TEST(integration_test, linear_field) {
    // generate random element from mesh
    auto e = this->meshLoader.generate_random_element();
    Integrator<FEM, TestFixture::M, 1> integrator;   // define integrator
    // a linear function over an element e defines a truncated prism over e
    std::function<double(SVector<TestFixture::N>)> f = [](SVector<TestFixture::N> x) -> double { return x[0] + x[1]; };
    // compute volume of truncated rectangular prism: 1/(M+1)*V*(h1 + h2 + ... + hM), where V is the element's measure
    double h = 0;
    for (auto p : e) h += f(p);
    double measure = e.measure() * h / (TestFixture::M + 1);
    // test for equality
    EXPECT_TRUE(almost_equal(measure, integrator.integrate(e, f)));
}

// test if is possible to integrate a field over the entire mesh
TEST(integration_test, integrate_over_triangulation) {
    // load sample mesh
    MeshLoader<Mesh2D> CShaped("unit_square");
    Integrator<FEM, 2, 1> integrator {};
    // define field to integrate
    std::function<double(SVector<2>)> f = [](SVector<2> x) -> double { return 1; };
    EXPECT_TRUE(almost_equal(1.0, integrator.integrate(CShaped.mesh, f)));
}

// test correctness of integrator tables
template <typename E> struct quadrature_rules_test : public ::testing::Test {
    static constexpr unsigned int M = E::value;
};
using DIMENSIONS_TYPE_LIST = ::testing::Types<
  std::integral_constant<unsigned int, 1>, std::integral_constant<unsigned int, 2>,
  std::integral_constant<unsigned int, 3>>;
TYPED_TEST_SUITE(quadrature_rules_test, DIMENSIONS_TYPE_LIST);

// test if all integrator tables produce the same result (this proves weights and quadrature nodes are correct)
using INTEGRATOR_TABLES_TYPE_LIST = std::tuple<
  std::tuple<IntegratorTable<1, 2>, IntegratorTable<1, 3>>,   // 1D integrators
  std::tuple<
    IntegratorTable<2, 3>, IntegratorTable<2, 6>, IntegratorTable<2, 7>, IntegratorTable<2, 12>>,   // 2D integrators
  std::tuple<IntegratorTable<3, 4>, IntegratorTable<3, 5>, IntegratorTable<3, 11>>                  // 3D integrators
  >;

template <unsigned int M, typename I, typename F>
void compute_quadrature(const F& f, const I& integratorTable, std::vector<double>& result) {
    // perform integration on reference element
    double value = 0;
    for (std::size_t iq = 0; iq < integratorTable.num_nodes; ++iq) {
        SVector<M> p = SVector<M>(integratorTable.nodes[iq].data());
        value += f(p) * integratorTable.weights[iq];
    }
    result.push_back(value / M);
    return;
}

// test if integration works on linear fields, expect all results equal
TYPED_TEST(quadrature_rules_test, check_correctness) {
    // define lagrangian basis on reference element
    auto b = LagrangianBasis<Mesh<TestFixture::M, TestFixture::M>, 1>::ref_basis();
    // space where results will be stored
    std::vector<double> results;
    // perform integration
    std::apply([&](auto... integrator) {
      ((compute_quadrature<TestFixture::M>(b[0], integrator, results)), ...);
    }, typename std::tuple_element<TestFixture::M - 1, INTEGRATOR_TABLES_TYPE_LIST>::type());

    // check all computed integrals are within double tolerance
    for (std::size_t i = 0; i < results.size(); ++i) {
        for (std::size_t j = i + 1; j < results.size(); ++j) { EXPECT_TRUE(almost_equal(results[i], results[j])); }
    }
}
