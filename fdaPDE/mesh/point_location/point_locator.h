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

#ifndef __POINT_LOCATOR_H__
#define __POINT_LOCATOR_H__

#include "../../utils/symbols.h"
#include "../element.h"

namespace fdapde {
namespace core {

// interface for point locations algorithms
template <unsigned int M, unsigned int N, unsigned int R> struct PointLocator {
    // solves the point location problem. returns nullptr if p is not found
    virtual const Element<M, N, R>* locate(const SVector<N>& p) const = 0;
};

}   // namespace core
}   // namespace fdapde

#endif   // __POINT_LOCATOR_H__
