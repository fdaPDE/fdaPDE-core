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
#include <fdaPDE/fields.h>
#include <fdaPDE/mesh.h>
#include <fdaPDE/finite_elements.h>
using fdapde::core::Element;
using fdapde::core::Integrator;
using fdapde::core::LagrangianBasis;
using fdapde::core::FEM;
using fdapde::core::laplacian;
using fdapde::core::MatrixConst;
using fdapde::core::MatrixPtr;
using fdapde::core::ScalarPtr;
using fdapde::core::VectorPtr;

#include "utils/mesh_loader.h"
using fdapde::testing::MESH_TYPE_LIST;
using fdapde::testing::MeshLoader;
#include "utils/utils.h"
using fdapde::testing::almost_equal;

// test integration of Laplacian weak form for a LagrangianElement of order 2
TEST(fem_operators_test, laplacian_order_2) {
    // load sample mesh, request an order 2 basis support
    MeshLoader<Mesh2D> CShaped("c_shaped");
    auto e = CShaped.mesh.element(175);   // reference element for this test
    Integrator<FEM, 2, 2> integrator {};

    // define differential operator
    auto L = -laplacian<FEM>();
    // define functional space
    auto basis = LagrangianBasis<Mesh2D, 2>::ref_basis();

    using BasisType = typename LagrangianBasis<Mesh2D, 2>::ReferenceBasis::ElementType;
    using NablaType = decltype(std::declval<BasisType>().derive());
    BasisType buff_psi_i, buff_psi_j;               // basis functions \psi_i, \psi_j
    NablaType buff_nabla_psi_i, buff_nabla_psi_j;   // gradient of basis functions \nabla \psi_i, \nabla \psi_j
    MatrixConst<2, 2, 2> buff_invJ;   // (J^{-1})^T, being J the inverse of the barycentric matrix relative to element e
    DVector<double> f(ct_nnodes(Mesh2D::local_dimension, 2));   // active solution coefficients on current element e
    // prepare buffer to be sent to bilinear form
    auto mem_buffer = std::make_tuple(
      ScalarPtr(&buff_psi_i), ScalarPtr(&buff_psi_j), VectorPtr(&buff_nabla_psi_i), VectorPtr(&buff_nabla_psi_j),
      MatrixPtr(&buff_invJ), &f);

    // develop bilinear form expression in an integrable field
    auto weak_form = L.integrate(mem_buffer);
    buff_invJ = e.inv_barycentric_matrix().transpose();

    std::vector<double> integrals;

    for (size_t i = 0; i < basis.size(); ++i) {
        buff_psi_i = basis[i];
        buff_nabla_psi_i = buff_psi_i.derive();
        for (size_t j = 0; j < basis.size(); ++j) {
            buff_psi_j = basis[j];
            buff_nabla_psi_j = buff_psi_j.derive();
            double value = integrator.template integrate<decltype(L)>(e, weak_form);
            integrals.push_back(value);
        }
    }

    // define vector of expected results, each row defines the result of the integration \int_e [\nabla \psi_i * \nabla
    // \psi_j] in the order basis functions are traversed on the reference element. Recall that the enumeration of basis
    // functions is 1 -> (0,0), 2 -> (1,0), 3 -> (0,1), 4 -> (0.5, 0), 5-> (0, 0.5), 6 -> (0.5, 0.5)
    std::vector<double> expected({
       0.7043890316492852,  0.1653830261033185,  0.0694133177797771,
      -0.6615321044132733, -0.2776532711191089,  0.0000000000000013,   // \psi_1 \psi_j
       0.1653830261033185,  0.7043890316492852,  0.0694133177797769,
      -0.6615321044132735,  0.0000000000000003, -0.2776532711191076,   // \psi_2 \psi_j
       0.0694133177797771,  0.0694133177797769,  0.4164799066786617,
       0.0000000000000002, -0.2776532711191083, -0.2776532711191075,   // \psi_3 \psi_j
      -0.6615321044132733, -0.6615321044132735,  0.0000000000000002,
       2.4336772933029756, -0.5553065422382126, -0.5553065422382162,   // \psi_5 \psi_j
      -0.2776532711191089,  0.0000000000000003, -0.2776532711191083,
      -0.5553065422382126,  2.4336772933029738, -1.3230642088265447,   // \psi_4 \psi_j
       0.0000000000000013, -0.2776532711191075, -0.2776532711191076,
      -0.5553065422382162, -1.3230642088265447,  2.4336772933029751    // \psi_6 \psi_j
    });

    // check for double equality of all computed integrals
    for (std::size_t i = 0; i < expected.size(); ++i) { EXPECT_TRUE(almost_equal(integrals[i], expected[i])); }
}
