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

#ifndef __SPLINE_SYMBOLS_H__
#define __SPLINE_SYMBOLS_H__

namespace fdapde {
namespace core {

// spline-based discretization strategy tag for PDE discretization
struct SPLINE { };

// utility macro to import symbols from memory buffer recived from assembly loop to spline operators
#define IMPORT_SPLINE_MEM_BUFFER_SYMBOLS(mem_buff)                                                                     \
    /* pair of basis functions \psi_i, \psi_j */                                                                       \
    auto psi_i = std::get<0>(mem_buff);                                                                                \
    auto psi_j = std::get<1>(mem_buff);

}   // namespace core
}   // namespace fdapde

#endif   // __SPLINE_SYMBOLS_H__
