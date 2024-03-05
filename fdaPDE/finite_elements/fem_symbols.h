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

// finite element order type (just a type wrapper around an int)
template <int R> struct fem_order {
    static constexpr int value = R;
};

}   // namespace core
}   // namespace fdapde

#endif   // __FEM_SYMBOLS_H__
