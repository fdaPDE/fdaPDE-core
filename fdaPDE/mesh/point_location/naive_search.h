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

#ifndef __NAIVE_SEARCH_H__
#define __NAIVE_SEARCH_H__

#include <memory>

#include "../element.h"
#include "../mesh.h"
#include "point_location_base.h"

namespace fdapde {
namespace core {

// naive search strategy for point location problem
template <int M, int N> class NaiveSearch : public PointLocationBase<M, N> {
   private:
    const Mesh<M, N>& mesh_;
   public:
    NaiveSearch(const Mesh<M, N>& mesh) : mesh_(mesh) {};
    // solves the point location problem
    virtual const Element<M, N>* locate(const SVector<N>& p) const {
        // loop over all mesh elements
        for (const auto& element : mesh_) {
            if (element.contains(p)) { return &element; }
        }
        // no element in mesh found
        return nullptr;
    }
};

}   // namespace core
}   // namespace fdapde

#endif   // __NAIVE_SEARCH_H__
