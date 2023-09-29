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

#ifndef __FEM_SYMBOLS_H__
#define __FEM_SYMBOLS_H__

namespace fdapde {
namespace core {

// finite element strategy tag for PDE discretization
struct FEM { };

// utility macro to import symbols from memory buffer recived from assembly loop to fe operators
#define IMPORT_FEM_MEM_BUFFER_SYMBOLS(mem_buff)                                                                        \
    /* pair of basis functions \psi_i, \psi_j*/                                                                        \
    auto psi_i = std::get<0>(mem_buff);                                                                                \
    auto psi_j = std::get<1>(mem_buff);                                                                                \
    /* gradient of \psi_i, \psi_j */                                                                                   \
    auto nabla_psi_i = std::get<2>(mem_buff);                                                                          \
    auto nabla_psi_j = std::get<3>(mem_buff);                                                                          \
    /* affine map to reference element */                                                                              \
    auto invJ = std::get<4>(mem_buff);                                                                                 \
    /* for non-linear operators, the current approximated solution */                                                  \
    auto f = *std::get<5>(mem_buff);

// finite element order type (just a type wrapper around an int)
template <int R> struct fem_order {
    static constexpr int value = R;
};

}   // namespace core
}   // namespace fdapde

#endif   // __FEM_SYMBOLS_H__
