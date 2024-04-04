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

#ifndef __TREE_SEARCH_H__
#define __TREE_SEARCH_H__

#include "../utils/symbols.h"
#include "kd_tree.h"

namespace fdapde {
namespace core {

// tree-based point location over triangulation. Based on "J. Bonet, J. Peraire (1991), An alternating digital tree
// (ADT) algorithm for 3D geometric searching and intersection problems"
template <typename MeshType> class TreeSearch {
   private:
    static constexpr int embed_dim = MeshType::embed_dim;
    static constexpr int local_dim = MeshType::local_dim;
    KDTree<2 * embed_dim> tree_;
    const MeshType* mesh_;
    SVector<embed_dim> c_;   // normalization constants
   public:
    TreeSearch() = default;
    TreeSearch(const MeshType* mesh) : mesh_(mesh) {
        // the i-th row of data contains the bounding box of the i-th element, stored as the vector [lower-left,
        // upper-right] corner. This moves each element to a point in R^{2N}
        DMatrix<double> data;
        data.resize(mesh_->n_elements(), 2 * embed_dim);
        for (int dim = 0; dim < embed_dim; ++dim) { c_[dim] = 1.0 / (mesh_->range()(1, dim) - mesh_->range()(0, dim)); }
        int i = 0;
        for (const auto& e : *mesh_) {
            std::pair<SVector<embed_dim>, SVector<embed_dim>> bbox = e.bounding_box();
            // unit hypercube point scaling
            data.row(i).leftCols(embed_dim)  = (bbox.first  - mesh_->range().row(0).transpose()).array() * c_.array();
            data.row(i).rightCols(embed_dim) = (bbox.second - mesh_->range().row(0).transpose()).array() * c_.array();
            ++i;
        }
        tree_ = KDTree<2 * embed_dim>(std::move(data));   // organize elements in a KD-tree structure
    }
    // finds element containing p, returns nullptr if element not found
    const typename MeshType::ElementType* locate(const SVector<embed_dim>& p) const {
        // build search query
        SVector<embed_dim> scaled_p = (p - mesh_->range().row(0).transpose()).array() * c_.array();
        SVector<2 * embed_dim> ll, ur;
        ll << SVector<embed_dim>::Zero(), scaled_p;
        ur << scaled_p, SVector<embed_dim>::Ones();
        auto found = tree_.range_search({ll, ur});
        // exhaustively scan the query results to get the searched mesh element
        for (int id : found) {
            auto element = mesh_->element(id);
            if (element.contains(p)) { return &mesh_->element(id); }
        }
        return nullptr;   // no element found
    }
};

}   // namespace core
}   // namespace fdapde

#endif   // __TREE_SEARCH_H__
