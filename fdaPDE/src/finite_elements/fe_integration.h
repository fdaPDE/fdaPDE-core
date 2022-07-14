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

#ifndef __FDAPDE_FE_INTEGRATION_H__
#define __FDAPDE_FE_INTEGRATION_H__

#include "header_check.h"

namespace fdapde {
namespace internals {

struct fe_quadrature_simplex_base { };
template <typename T>
concept is_fe_quadrature_simplex = std::is_base_of_v<fe_quadrature_simplex_base, T>;
template <typename T> [[maybe_unused]] static constexpr bool is_fe_quadrature_simplex_v = is_fe_quadrature_simplex<T>;

template <typename LhsQuadrature, typename RhsQuadrature>
    requires(
      LhsQuadrature::local_dim == RhsQuadrature::local_dim && is_fe_quadrature_simplex_v<LhsQuadrature> &&
      is_fe_quadrature_simplex_v<RhsQuadrature>)
struct higher_degree_fe_quadrature :
    std::type_identity<
      std::conditional_t<(LhsQuadrature::order > RhsQuadrature::order), LhsQuadrature, RhsQuadrature>> { };
template <typename LhsQuadrature, typename RhsQuadrature>
using higher_degree_fe_quadrature_t = higher_degree_fe_quadrature<LhsQuadrature, RhsQuadrature>::type;

template <int LocalDim, int Size> struct fe_quadrature_simplex;

// degree: highest polynomial degree correctly integrated by the formula
// order : number of quadrature nodes

// https://pomax.github.io/bezierinfo/legendre-gauss.html (obtained mapping the [-1, 1] gauss quadratures on [0, 1])
// 1D 1 point formula, degree 1
template <> struct fe_quadrature_simplex<1, 1> : public fe_quadrature_simplex_base {
    static constexpr int local_dim = 1;
    static constexpr int order  = 1;
    static constexpr int degree = 1;

    static constexpr Vector<double, order> nodes {
      std::array<double, order> {
	0.500000000000000}
    };
    static constexpr Vector<double, order> weights {
      std::array<double, order> {
	1.000000000000000}
    };
};

// 1D 2 point formula, degree 3
template <> struct fe_quadrature_simplex<1, 2> : public fe_quadrature_simplex_base {
    static constexpr int local_dim = 1;
    static constexpr int order  = 2;
    static constexpr int degree = 3;

    static constexpr Vector<double, order> nodes {
      std::array<double, order> {
	0.211324865405187, 0.788675134594812}
    };
    static constexpr Vector<double, order> weights {
      std::array<double, order> {
	0.500000000000000, 0.500000000000000}
    };
};

// 1D 3 point formula, degree 5
template <> struct fe_quadrature_simplex<1, 3> : public fe_quadrature_simplex_base {
    static constexpr int local_dim = 1;
    static constexpr int order  = 3;
    static constexpr int degree = 5;

    static constexpr Vector<double, order> nodes {
      std::array<double, order> {
	0.112701665379258, 0.500000000000000, 0.887298334620741}
    };
    static constexpr Vector<double, order> weights {
      std::array<double, order> {
	0.277777777777778, 0.444444444444444, 0.277777777777778}
    };
};

// 1D 4 point formula, degree 7
template <> struct fe_quadrature_simplex<1, 4> : public fe_quadrature_simplex_base {
    static constexpr int local_dim = 1;
    static constexpr int order  = 4;
    static constexpr int degree = 7;

    static constexpr Vector<double, order> nodes {
      std::array<double, order> {
	0.069431844202973, 0.330009478207571, 0.669990521792428, 0.930568155797026}
    };
    static constexpr Vector<double, order> weights {
      std::array<double, order> {
	0.173927422568726, 0.326072577431273, 0.326072577431273, 0.173927422568726}
    };
};
  
// https://people.sc.fsu.edu/~jburkardt/datasets/quadrature_rules_tri/quadrature_rules_tri.html
// 2D 1 point formula, degree 1
template <> struct fe_quadrature_simplex<2, 1> : public fe_quadrature_simplex_base {
    static constexpr int local_dim = 2;
    static constexpr int order  = 1;
    static constexpr int degree = 1;

    static constexpr Matrix<double, order, local_dim> nodes {
      std::array<double, order * local_dim> {
	0.333333333333333, 0.333333333333333}
    };
    static constexpr Vector<double, order> weights {
      std::array<double, order> {
	1.000000000000000}
    };
};

// 2D 3 point formula, degree 2
template <> struct fe_quadrature_simplex<2, 3> : public fe_quadrature_simplex_base {
    static constexpr int local_dim = 2;
    static constexpr int order  = 3;
    static constexpr int degree = 2;

    static constexpr Matrix<double, order, local_dim> nodes {
      std::array<double, order * local_dim> {
	0.166666666666667, 0.166666666666667, 0.666666666666667, 0.166666666666667, 0.166666666666667,
	0.666666666666667}
    };
    static constexpr Vector<double, order> weights {
      std::array<double, order> {
	0.333333333333333, 0.333333333333333, 0.333333333333333}
    };
};

// 2D 4 point formula, degree 3
template <> struct fe_quadrature_simplex<2, 4> : public fe_quadrature_simplex_base {
    static constexpr int local_dim = 2;
    static constexpr int order  = 4;
    static constexpr int degree = 3;

    static constexpr Matrix<double, order, local_dim> nodes {
      std::array<double, order * local_dim> {
	0.333333333333333, 0.333333333333333, 0.600000000000000, 0.200000000000000, 0.200000000000000,
	0.600000000000000, 0.200000000000000, 0.200000000000000}
    };
    static constexpr Vector<double, order> weights {
      std::array<double, order> {
       -0.562500000000000, 0.520833333333333, 0.520833333333333, 0.520833333333333}
    };
};

// 2D 6 point formula, degree 4
template <> struct fe_quadrature_simplex<2, 6> : public fe_quadrature_simplex_base {
    static constexpr int local_dim = 2;
    static constexpr int order  = 6;
    static constexpr int degree = 4;

    static constexpr Matrix<double, order, local_dim> nodes {
      std::array<double, order * local_dim> {
	0.445948490915965, 0.445948490915965, 0.445948490915965, 0.108103018168070, 0.108103018168070,
	0.445948490915965, 0.091576213509771, 0.091576213509771, 0.091576213509771, 0.816847572980459,
	0.816847572980459, 0.091576213509771}
    };
    static constexpr Vector<double, order> weights {
      std::array<double, order> {
	0.223381589678011, 0.223381589678011, 0.223381589678011, 0.109951743655322, 0.109951743655322,
	0.109951743655322}
    };
};

// 2D 7 point formula, degree 5
template <> struct fe_quadrature_simplex<2, 7> : public fe_quadrature_simplex_base {
    static constexpr int local_dim = 2;
    static constexpr int order  = 7;
    static constexpr int degree = 5;

    static constexpr Matrix<double, order, local_dim> nodes {
      std::array<double, order * local_dim> {
	0.333333333333333, 0.333333333333333, 0.101286507323456, 0.101286507323456, 0.101286507323456,
	0.797426985353087, 0.797426985353087, 0.101286507323456, 0.470142064105115, 0.470142064105115,
	0.470142064105115, 0.059715871789770, 0.059715871789770, 0.470142064105115}
    };
    static constexpr Vector<double, order> weights {
      std::array<double, order> {
	0.225000000000000, 0.125939180544827, 0.125939180544827, 0.125939180544827, 0.132394152788506,
	0.132394152788506, 0.132394152788506}
    };
};

// 2D 12 point formula, degree 6
template <> struct fe_quadrature_simplex<2, 12> : public fe_quadrature_simplex_base {
    static constexpr int local_dim = 2;
    static constexpr int order  = 12;
    static constexpr int degree = 6;

    static constexpr Matrix<double, order, local_dim> nodes {
      std::array<double, order * local_dim> {
	0.873821971016996, 0.063089014491502, 0.063089014491502, 0.873821971016996, 0.063089014491502,
	0.063089014491502, 0.501426509658179, 0.249286745170910, 0.249286745170910, 0.501426509658179,
	0.249286745170910, 0.249286745170910, 0.636502499121399, 0.310352451033785, 0.636502499121399,
	0.053145049844816, 0.310352451033785, 0.636502499121399, 0.310352451033785, 0.053145049844816,
	0.053145049844816, 0.636502499121399, 0.053145049844816, 0.310352451033785}
    };
    static constexpr Vector<double, order> weights {
      std::array<double, order> {
	0.050844906370207, 0.050844906370207, 0.050844906370207, 0.116786275726379, 0.116786275726379,
	0.116786275726379, 0.082851075618374, 0.082851075618374, 0.082851075618374, 0.082851075618374,
	0.082851075618374, 0.082851075618374}
    };
};

// https://people.sc.fsu.edu/~jburkardt/datasets/quadrature_rules_tet/quadrature_rules_tet.html
// 3D 1 point formula, degree 1
template <> struct fe_quadrature_simplex<3, 1> : public fe_quadrature_simplex_base {
    static constexpr int local_dim = 3;
    static constexpr int order = 1;
    static constexpr int degree = 1;

    static constexpr Matrix<double, order, local_dim> nodes {
      std::array<double, order * local_dim> {
	0.250000000000000, 0.250000000000000, 0.250000000000000}
    };
    static constexpr Vector<double, order> weights {
      std::array<double, order> {
	1.000000000000000}
    };
};

// 3D 4 point formula, degree 2
template <> struct fe_quadrature_simplex<3, 4> : public fe_quadrature_simplex_base {
    static constexpr int local_dim = 3;
    static constexpr int order = 4;
    static constexpr int degree = 2;

    static constexpr Matrix<double, order, local_dim> nodes {
      std::array<double, order * local_dim> {
	0.585410196624969, 0.138196601125011, 0.138196601125011, 0.138196601125011, 0.138196601125011,
	0.138196601125011, 0.138196601125011, 0.138196601125011, 0.585410196624969, 0.138196601125011,
	0.585410196624969, 0.138196601125011}
    };
    static constexpr Vector<double, order> weights {
      std::array<double, order> {
	0.250000000000000, 0.250000000000000, 0.250000000000000, 0.250000000000000}
    };
};

// 3D 5 point formula, degree 3
template <> struct fe_quadrature_simplex<3, 5> : public fe_quadrature_simplex_base {
    static constexpr int local_dim = 3;
    static constexpr int order = 5;
    static constexpr int degree = 3;

    static constexpr Matrix<double, order, local_dim> nodes {
      std::array<double, order * local_dim> {
	0.250000000000000, 0.250000000000000, 0.250000000000000, 0.500000000000000, 0.166666666666667,
	0.166666666666667, 0.166666666666667, 0.500000000000000, 0.166666666666667, 0.166666666666667,
	0.166666666666667, 0.500000000000000, 0.166666666666667, 0.166666666666667, 0.166666666666667}
    };
    static constexpr Vector<double, order> weights {
      std::array<double, order> {
        -0.80000000000000, 0.450000000000000, 0.450000000000000, 0.450000000000000, 0.450000000000000}
    };
};

// 3D 11 point formula, degree 4
template <> struct fe_quadrature_simplex<3, 11> : public fe_quadrature_simplex_base {
    static constexpr int local_dim = 3;
    static constexpr int order = 11;
    static constexpr int degree = 4;

    static constexpr Matrix<double, order, local_dim> nodes {
      std::array<double, order * local_dim> {
	0.250000000000000, 0.250000000000000, 0.250000000000000, 0.785714285714285, 0.071428571428571,
	0.071428571428571, 0.071428571428571, 0.071428571428571, 0.071428571428571, 0.071428571428571,
	0.071428571428571, 0.785714285714285, 0.071428571428571, 0.785714285714285, 0.071428571428571,
	0.100596423833200, 0.399403576166799, 0.399403576166799, 0.399403576166799, 0.100596423833200,
	0.399403576166799, 0.399403576166799, 0.399403576166799, 0.100596423833200, 0.399403576166799,
	0.100596423833200, 0.100596423833200, 0.100596423833200, 0.399403576166799, 0.100596423833200,
	0.100596423833200, 0.100596423833200, 0.399403576166799}
    };
    static constexpr Vector<double, order> weights {
      std::array<double, order> {
       -0.078933333333333, 0.045733333333333, 0.045733333333333, 0.045733333333333, 0.045733333333333,
	0.149333333333333, 0.149333333333333, 0.149333333333333, 0.149333333333333, 0.149333333333333,
	0.149333333333333}
    };
};

// 3D 15 point formula, degree 5
template <> struct fe_quadrature_simplex<3, 15> : public fe_quadrature_simplex_base {
    static constexpr int local_dim = 3;
    static constexpr int order = 15;
    static constexpr int degree = 5;

    static constexpr Matrix<double, order, local_dim> nodes {
      std::array<double, order * local_dim> {
	0.250000000000000, 0.250000000000000, 0.250000000000000, 0.000000000000000, 0.333333333333333,
	0.333333333333333, 0.333333333333333, 0.333333333333333, 0.333333333333333, 0.333333333333333,
	0.333333333333333, 0.000000000000000, 0.333333333333333, 0.000000000000000, 0.333333333333333,
	0.727272727272727, 0.090909090909090, 0.090909090909090, 0.090909090909090, 0.090909090909090,
	0.090909090909090, 0.090909090909090, 0.090909090909090, 0.727272727272727, 0.090909090909090,
	0.727272727272727, 0.090909090909090, 0.433449846426335, 0.066550153573664, 0.066550153573664,
	0.066550153573664, 0.433449846426335, 0.066550153573664, 0.066550153573664, 0.066550153573664,
	0.433449846426335, 0.066550153573664, 0.433449846426335, 0.433449846426335, 0.433449846426335,
	0.066550153573664, 0.433449846426335, 0.433449846426335, 0.433449846426335, 0.066550153573664}
    };
    static constexpr Vector<double, order> weights {
      std::array<double, order> {
	0.181702068582535, 0.036160714285714, 0.036160714285714, 0.036160714285714, 0.036160714285714,
	0.069871494516173, 0.069871494516173, 0.069871494516173, 0.069871494516173, 0.065694849368318,
	0.065694849368318, 0.065694849368318, 0.065694849368318, 0.065694849368318, 0.065694849368318}
    };
};

// 3D 24 point formula, degree 6
template <> struct fe_quadrature_simplex<3, 24> : public fe_quadrature_simplex_base {
    static constexpr int local_dim = 3;
    static constexpr int order = 24;
    static constexpr int degree = 6;

    static constexpr Matrix<double, order, local_dim> nodes {
      std::array<double, order * local_dim> {
	0.356191386222544, 0.214602871259151, 0.214602871259151, 0.214602871259151, 0.214602871259151,
	0.214602871259151, 0.214602871259151, 0.214602871259151, 0.356191386222544, 0.214602871259151,
	0.356191386222544, 0.214602871259151, 0.877978124396166, 0.040673958534611, 0.040673958534611,
	0.040673958534611, 0.040673958534611, 0.040673958534611, 0.040673958534611, 0.040673958534611,
	0.877978124396166, 0.040673958534611, 0.877978124396166, 0.040673958534611, 0.032986329573173,
	0.322337890142275, 0.322337890142275, 0.322337890142275, 0.322337890142275, 0.322337890142275,
	0.322337890142275, 0.322337890142275, 0.032986329573173, 0.322337890142275, 0.032986329573173,
	0.322337890142275, 0.269672331458315, 0.063661001875017, 0.063661001875017, 0.063661001875017,
	0.269672331458315, 0.063661001875017, 0.063661001875017, 0.063661001875017, 0.269672331458315,
	0.603005664791649, 0.063661001875017, 0.063661001875017, 0.063661001875017, 0.603005664791649,
	0.063661001875017, 0.063661001875017, 0.063661001875017, 0.603005664791649, 0.063661001875017,
	0.269672331458315, 0.603005664791649, 0.269672331458315, 0.603005664791649, 0.063661001875017,
	0.603005664791649, 0.063661001875017, 0.269672331458315, 0.063661001875017, 0.603005664791649,
	0.269672331458315, 0.269672331458315, 0.063661001875017, 0.603005664791649, 0.603005664791649,
	0.269672331458315, 0.063661001875017}
    };
    static constexpr Vector<double, order> weights {
      std::array<double, order> {
	0.039922750258167, 0.039922750258167, 0.039922750258167, 0.039922750258167, 0.010077211055320,
	0.010077211055320, 0.010077211055320, 0.010077211055320, 0.055357181543654, 0.055357181543654,
	0.055357181543654, 0.055357181543654, 0.048214285714285, 0.048214285714285, 0.048214285714285,
	0.048214285714285, 0.048214285714285, 0.048214285714285, 0.048214285714285, 0.048214285714285,
	0.048214285714285, 0.048214285714285, 0.048214285714285, 0.048214285714285}
    };
};
  
}   // namespace internals

// 1D formulas
[[maybe_unused]] static struct QS1DP1_ : internals::fe_quadrature_simplex<1, 1>  { } QS1DP1;
[[maybe_unused]] static struct QS1DP3_ : internals::fe_quadrature_simplex<1, 2>  { } QS1DP3;
[[maybe_unused]] static struct QS1DP5_ : internals::fe_quadrature_simplex<1, 3>  { } QS1DP5;
[[maybe_unused]] static struct QS1DP7_ : internals::fe_quadrature_simplex<1, 4>  { } QS1DP7;
// 2D formulas
[[maybe_unused]] static struct QS2DP1_ : internals::fe_quadrature_simplex<2, 1>  { } QS2DP1;
[[maybe_unused]] static struct QS2DP2_ : internals::fe_quadrature_simplex<2, 3>  { } QS2DP2;
[[maybe_unused]] static struct QS2DP3_ : internals::fe_quadrature_simplex<2, 4>  { } QS2DP3;
[[maybe_unused]] static struct QS2DP4_ : internals::fe_quadrature_simplex<2, 6>  { } QS2DP4;
[[maybe_unused]] static struct QS2DP5_ : internals::fe_quadrature_simplex<2, 7>  { } QS2DP5;
[[maybe_unused]] static struct QS2DP6_ : internals::fe_quadrature_simplex<2, 12> { } QS2DP6;
// 3D formulas
[[maybe_unused]] static struct QS3DP1_ : internals::fe_quadrature_simplex<3, 1>  { } QS3DP1;
[[maybe_unused]] static struct QS3DP2_ : internals::fe_quadrature_simplex<3, 4>  { } QS3DP2;
[[maybe_unused]] static struct QS3DP3_ : internals::fe_quadrature_simplex<3, 5>  { } QS3DP3;
[[maybe_unused]] static struct QS3DP4_ : internals::fe_quadrature_simplex<3, 11> { } QS3DP4;
[[maybe_unused]] static struct QS3DP5_ : internals::fe_quadrature_simplex<3, 15> { } QS3DP5;
[[maybe_unused]] static struct QS3DP6_ : internals::fe_quadrature_simplex<3, 24> { } QS3DP6;

// compute quadrature nodes on physical domain
template <int LocalDim, int EmbedDim, typename Quadrature>
Eigen::Matrix<double, Dynamic, Dynamic>
simplex_quadrature_nodes(const Triangulation<LocalDim, EmbedDim>& triangulation, const Quadrature& quadrature) {
    fdapde_static_assert(
      internals::is_fe_quadrature_simplex_v<Quadrature>, THIS_METHOD_IS_FOR_FE_QUADRATURE_SIMPLEX_ONLY);
    constexpr int n_quadrature_nodes = Quadrature::order;
  
    Eigen::Matrix<double, Dynamic, Dynamic> quad_nodes;
    Eigen::Map<const Eigen::Matrix<
      double, n_quadrature_nodes, Quadrature::local_dim,
      Quadrature::local_dim != 1 ? Eigen::RowMajor : Eigen::ColMajor>>
      ref_quad_nodes(quadrature.nodes.data());
    quad_nodes.resize(n_quadrature_nodes * triangulation.n_cells(), EmbedDim);
    int local_cell_id = 0;
    for (auto it = triangulation.cells_begin(); it != triangulation.cells_end(); ++it) {
        for (int q_k = 0; q_k < n_quadrature_nodes; ++q_k) {
            quad_nodes.row(local_cell_id * n_quadrature_nodes + q_k) =
              it->J() * ref_quad_nodes.row(q_k).transpose() + it->node(0);
        }
        local_cell_id++;
    }
    return quad_nodes;
}

}   // namespace fdapde

#endif   // __FDAPDE_FE_INTEGRATION_H__
