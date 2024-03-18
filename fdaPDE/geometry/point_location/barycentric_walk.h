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

namespace fdapde {
namespace core {

// barycentric walk strategy for point location problem, works only for 2D and 3D *convex* triangualtions
template <typename MeshType> class BarycentricWalk {
    const MeshType* mesh_;
   public:
    BarycentricWalk() = default;
    BarycentricWalk(const MeshType* mesh) : mesh_(mesh) {
        static_assert(MeshType::local_dim == MeshType::embed_dim);
    };
    // finds element containing p, returns nullptr if element not found
    const typename MeshType::ElementType* locate(const SVector<MeshType::embed_dim>& p) const {
        // start search from random element
        std::random_device rng {};
        std::uniform_int_distribution<std::size_t> uniform_int(0, mesh_->n_elements() - 1);
        std::size_t next_id = uniform_int(rng);

        std::set<std::size_t> visited_;
        while (!mesh_->element(next_id).contains(p) || visited_.find(next_id) != visited_.end()) {
            visited_.insert(next_id);
            // compute barycantric coordinates
            SVector<MeshType::embed_dim + 1> bary_coord = mesh_->element(next_id).to_barycentric_coords(p);
            // find minimum baricentric coordinate and move to element insisting of opposite face
            std::size_t min_bary_coord_index;
            bary_coord.minCoeff(&min_bary_coord_index);
            // the i-th value in neighbors refers to the element adjacent to the face oppsite the i-th vertex
            next_id = mesh_->element(next_id).neighbors()[min_bary_coord_index];
        }
        return mesh_->element(next_id).contains(p) ? &mesh_->element(next_id) : nullptr;
    }
};

}   // namespace core
}   // namespace fdapde

#endif   // __BARYCENTRIC_WALK_H__
