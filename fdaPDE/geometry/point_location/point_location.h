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

#ifndef __POINT_LOCATION_H__
#define __POINT_LOCATION_H__

#include "../../utils/symbols.h"
#include "../../utils/assert.h"
#include "../../utils/type_erasure.h"
#include "../element.h"

namespace fdapde {
namespace core {

// supported point location strategies
enum PointLocationStrategy {
    naive_search,
    barycentric_walk,
    tree_search
};

template <int M, int N> struct PointLocation__ {
    template <typename T> using fn_ptrs = fdapde::mem_fn_ptrs<&T::locate>;
    // solves the point location problem for a set of points, outputs -1 if element not found. For a fine-grained API,
    // interact directly with the location algorithm
    DVector<int> locate(const DMatrix<double>& points) const {
        fdapde_assert(points.cols() == N);
        DVector<int> result(points.rows());
        for (int i = 0; i < points.rows(); ++i) {
            auto e = fdapde::invoke<const Element<M, N>*, 0>(*this, SVector<N>(points.row(i)));
            result[i] = e ? e->ID() : -1;
        }
        return result;
    }
};
template <int M, int N> using PointLocation = fdapde::erase<fdapde::heap_storage, PointLocation__<M, N>>;

}   // namespace core
}   // namespace fdapde

#endif   // __POINT_LOCATION_H__
