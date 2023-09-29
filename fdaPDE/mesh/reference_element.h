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

#ifndef __REFERENCE_ELEMENT_H__
#define __REFERENCE_ELEMENT_H__

#include <array>

#include "../utils/symbols.h"

namespace fdapde {
namespace core {

// Definition of the M-dimensional unit reference simplices of order R.
template <int M, int R> struct ReferenceElement;

template <>   // 1D first order basis
struct ReferenceElement<1, 1> {
    static constexpr std::array<std::array<double, 1>, 2> nodes = {
      {{0}, {1}}
    };
    const std::array<SVector<2>, 2> bary_coords = {
      SVector<2>(1, 0), SVector<2>(0, 1)
    };
};
template <>   // 1D second order basis
struct ReferenceElement<1, 2> {
    static constexpr std::array<std::array<double, 1>, 3> nodes = {
      {{0}, {1}, {0.5}}
    };
    const std::array<SVector<2>, 3> bary_coords = {
      SVector<2>(1, 0), SVector<2>(0, 1), SVector<2>(0.5, 0.5)
    };
};

template <>   // 2D first order basis
struct ReferenceElement<2, 1> {
    static constexpr std::array<std::array<double, 2>, 3> nodes = {
      {{0, 0}, {1, 0}, {0, 1}}
    };
    const std::array<SVector<3>, 3> bary_coords = {
      SVector<3>(1, 0, 0), SVector<3>(0, 1, 0), SVector<3>(0, 0, 1)
    };
};
template <>   // 2D second order basis
struct ReferenceElement<2, 2> {
    static constexpr std::array<std::array<double, 2>, 6> nodes = {
      {{0, 0}, {1, 0}, {0, 1}, {0.5, 0}, {0, 0.5}, {0.5, 0.5}}
    };
    const std::array<SVector<3>, 6> bary_coords = {
      SVector<3>(1, 0, 0),     SVector<3>(0, 1, 0),     SVector<3>(0, 0, 1),
      SVector<3>(0.5, 0.5, 0), SVector<3>(0.5, 0, 0.5), SVector<3>(0, 0.5, 0.5)
    };
};
template <>   // 2D third order basis
struct ReferenceElement<2, 3> {
    static constexpr std::array<std::array<double, 2>, 10> nodes = {
      {{0, 0}, {1, 0}, {0, 1}, {1. / 3, 0}, {2. / 3, 0},
       {0, 1. / 3}, {0, 2. / 3}, {2. / 3, 1. / 3}, {1. / 3, 2. / 3}, {1. / 3, 1. / 3}}
    };
    const std::array<SVector<3>, 10> bary_coords = {
      SVector<3>(1, 0, 0),           SVector<3>(0, 1, 0),           SVector<3>(0, 0, 1),
      SVector<3>(2. / 3, 1. / 3, 0), SVector<3>(1. / 3, 2. / 3, 0), SVector<3>(2. / 3, 0, 1. / 3),
      SVector<3>(1. / 3, 0, 2. / 3), SVector<3>(0, 2. / 3, 1. / 3), SVector<3>(0, 1. / 3, 2. / 3),
      SVector<3>(1. / 3, 1. / 3, 1. / 3)
    };
};

template <>   // 3D first order basis
struct ReferenceElement<3, 1> {
    static constexpr std::array<std::array<double, 3>, 4> nodes = {
      {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}}
    };
    const std::array<SVector<4>, 4> bary_coords = {
      SVector<4>(1, 0, 0, 0), SVector<4>(0, 1, 0, 0), SVector<4>(0, 0, 1, 0), SVector<4>(0, 0, 0, 1)
    };
};
template <>   // 3D second order basis
struct ReferenceElement<3, 2> {
    static constexpr std::array<std::array<double, 3>, 10> nodes = {
      {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {0.5, 0.5, 0},
       {0, 0.5, 0}, {0.5, 0, 0}, {0.5, 0, 0.5}, {0, 0.5, 0.5}, {0, 0, 0.5}}
    };
};

}   // namespace core
}   // namespace fdapde

#endif   // __REFERENCE_ELEMENT_H__
