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

#ifndef __FE_INTEGRATION_H__
#define __FE_INTEGRATION_H__

#include "../linear_algebra/constexpr_matrix.h"

namespace fdapde {
namespace internals {

template <int LocalDim, int Size> struct fe_quadrature_simplex;

// 1D 2 point formula
template <> struct fe_quadrature_simplex<1, 2> {
    static constexpr int local_dim = 1;
    static constexpr int n_nodes = 2;

    static constexpr cexpr::Vector<double, n_nodes> nodes {
      std::array<double, n_nodes> {0.211324865405187, 0.788675134594812}
    };
    static constexpr cexpr::Vector<double, n_nodes> weights {
      std::array<double, n_nodes> {0.500000000000000, 0.500000000000000}
    };
};

// 1D 3 point formula
template <> struct fe_quadrature_simplex<1, 3> {
    static constexpr int local_dim = 1;
    static constexpr int n_nodes = 3;

    static constexpr cexpr::Vector<double, n_nodes> nodes {
      std::array<double, n_nodes> {0.112701665379258, 0.500000000000000, 0.887298334620741}
    };
    static constexpr cexpr::Vector<double, n_nodes> weights {
      std::array<double, n_nodes> {0.277777777777778, 0.444444444444444, 0.277777777777778}
    };
};

// 2D 1 point formula
template <> struct fe_quadrature_simplex<2, 1> {
    static constexpr int local_dim = 2;
    static constexpr int n_nodes = 1;

    static constexpr cexpr::Matrix<double, n_nodes, local_dim> nodes {
      std::array<double, n_nodes * local_dim> {0.333333333333333, 0.333333333333333}
    };
    static constexpr cexpr::Vector<double, n_nodes> weights {
      std::array<double, n_nodes> {1.000000000000000}
    };
};

// 2D 3 point formula
template <> struct fe_quadrature_simplex<2, 3> {
    static constexpr int local_dim = 2;
    static constexpr int n_nodes = 3;

    static constexpr cexpr::Matrix<double, n_nodes, local_dim> nodes {
      std::array<double, n_nodes * local_dim> {
        0.166666666666667, 0.166666666666667,
        0.666666666666667, 0.166666666666667,
        0.166666666666667, 0.666666666666667}
    };
    static constexpr cexpr::Vector<double, n_nodes> weights {
      std::array<double, n_nodes> {0.333333333333333, 0.333333333333333, 0.333333333333333}
    };
};

// 2D 6 point formula
template <> struct fe_quadrature_simplex<2, 6> {
    static constexpr int local_dim = 2;
    static constexpr int n_nodes = 6;

    static constexpr cexpr::Matrix<double, n_nodes, local_dim> nodes {
      std::array<double, n_nodes * local_dim> {
	0.445948490915965, 0.445948490915965,
	0.445948490915965, 0.108103018168070,
	0.108103018168070, 0.445948490915965,
	0.091576213509771, 0.091576213509771,
	0.091576213509771, 0.816847572980459,
	0.816847572980459, 0.091576213509771}
    };
    static constexpr cexpr::Vector<double, n_nodes> weights {
      std::array<double, n_nodes> {
	0.223381589678011, 0.223381589678011, 0.223381589678011,
	0.109951743655322, 0.109951743655322, 0.109951743655322}
    };
};

// 2D 6 point formula
template <> struct fe_quadrature_simplex<2, 7> {
    static constexpr int local_dim = 2;
    static constexpr int n_nodes = 7;

    static constexpr cexpr::Matrix<double, n_nodes, local_dim> nodes {
      std::array<double, n_nodes * local_dim> {
	0.333333333333333, 0.333333333333333,
	0.101286507323456, 0.101286507323456,
	0.101286507323456, 0.797426985353087,
	0.797426985353087, 0.101286507323456,
	0.470142064105115, 0.470142064105115,
	0.470142064105115, 0.059715871789770,
	0.059715871789770, 0.470142064105115}
    };
    static constexpr cexpr::Vector<double, n_nodes> weights {
      std::array<double, n_nodes> {
	0.225000000000000, 0.125939180544827, 0.125939180544827,
	0.125939180544827, 0.132394152788506, 0.132394152788506,
	0.132394152788506}
    };
};
  
}   // namespace internals
}   // namespace fdapde

#endif   // __FE_INTEGRATION_H__
