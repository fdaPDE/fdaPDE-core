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

// interface for point locations algorithms
template <unsigned int M, unsigned int N, unsigned int R> struct PointLocator {
    // solves the point location problem. returns nullptr if p is not found
    virtual const Element<M, N, R>* locate(const SVector<N>& p) const = 0;

    // solves the point location problem for a set of points
    std::vector<const Element<M, N, R>*> locate(const DMatrix<double>& points) const {
        fdapde_assert(points.cols() == N);

        std::vector<const Element<M, N, R>*> elements;
        elements.reserve(points.rows());
        // solve point location for each given point
        for (std::size_t i = 0; i < points.rows(); ++i) { elements.emplace_back(this->locate(points.row(i))); }
        return elements;
    }
};

// supported point location strategies
enum PointLocationStrategy { naive_search, barycentric_walk, tree_search };
  
}   // namespace core
}   // namespace fdapde

#endif   // __POINT_LOCATOR_H__
