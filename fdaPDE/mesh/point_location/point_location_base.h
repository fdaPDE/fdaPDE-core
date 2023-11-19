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

#include <memory>

#include "../../utils/symbols.h"
#include "../../utils/assert.h"
#include "../element.h"

namespace fdapde {
namespace core {

// forward declaration
template <int M, int N> class Mesh;
  
// interface for point locations algorithms
template <int M, int N> struct PointLocationBase {
    // solves the point location problem. returns nullptr if p is not found
    virtual const Element<M, N>* locate(const SVector<N>& p) const = 0;

    // solves the point location problem for a set of points
    DVector<int> locate(const DMatrix<double>& points) const {
        fdapde_assert(points.cols() == N);

        DVector<int> elements;
        elements.resize(points.rows());
        // solve point location for each given point
        for (std::size_t i = 0; i < points.rows(); ++i) { elements[i] = this->locate(SVector<N>(points.row(i)))->ID(); }
        return elements;
    }
};

// supported point location strategies
enum PointLocationStrategy { naive_search, barycentric_walk, tree_search };
  
}   // namespace core
}   // namespace fdapde

#endif   // __POINT_LOCATOR_H__
