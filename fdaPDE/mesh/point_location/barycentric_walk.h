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

#ifndef __BARYCENTRIC_WALK_H__
#define __BARYCENTRIC_WALK_H__

#include <random>
#include <set>

#include "point_location_base.h"

namespace fdapde {
namespace core {

// barycentric walk strategy for point location problem, works only for 2D and 3D *convex* triangualtions
template <int M, int N> class BarycentricWalk : public PointLocationBase<M, N> {
    static_assert(M == N, "barycentric walk cannot be applied to manifold domains");
   private:
    const Mesh<M, N>& mesh_;
   public:
    BarycentricWalk(const Mesh<M, N>& mesh) : mesh_(mesh) {};
    // solves the point location problem
    virtual const Element<M, N>* locate(const SVector<N>& p) const {
        // define uniform distribution over the ID space
        typedef std::uniform_int_distribution<std::size_t> Distribution;
        std::random_device rng {};
        Distribution uniform_int = Distribution(0, mesh_.n_elements() - 1);
        // start from an element at random
        std::size_t next_id = uniform_int(rng);

        std::set<std::size_t> visited_;
        while (!mesh_.element(next_id).contains(p) || visited_.find(next_id) != visited_.end()) {
            visited_.insert(next_id);
            // compute barycantric coordinates
            SVector<N + 1> bary_coord = mesh_.element(next_id).to_barycentric_coords(p);

            // pick the vertices corresponding to the n highest coordinates, and move into the adjacent element that
            // shares those vertices (equivalent to find the minimum baricentric coordinate and move to
            // the element adjacent to the face opposite to it)
            std::size_t min_bary_coord_index;
            bary_coord.minCoeff(&min_bary_coord_index);

            // the i-th value in neighbors refers to the element adjacent to the face oppsite the i-th vertex
            next_id = mesh_.element(next_id).neighbors()[min_bary_coord_index];
        }
        if (mesh_.element(next_id).contains(p)) {
            return &mesh_.element(next_id);
        } else {
            return nullptr;
        }
    }
};

}   // namespace core
}   // namespace fdapde

#endif   // __BARYCENTRIC_WALK_H__
