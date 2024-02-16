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

#include "../../utils/symbols.h"
#include "../kd_tree.h"

namespace fdapde {
namespace core {

// TreeSearch is used as default point location strategy in Mesh, need to forward declare Mesh here
template <int M, int N> class Mesh;

// tree-based point location over triangulation. Based on "J. Bonet, J. Peraire (1991), An alternating digital tree
// (ADT) algorithm for 3D geometric searching and intersection problems"
template <int M, int N> class TreeSearch {
   private:
    KDTree<2 * N> tree_;
    const Mesh<M, N>& mesh_;
    SVector<N> c_;   // normalization constants
   public:
    TreeSearch(const Mesh<M, N>& mesh) : mesh_(mesh) {
        // the i-th row of data contains the bounding box of the i-th element, stored as the vector [lower-left,
        // upper-right] corner. This moves each element to a point in R^{2N}
        DMatrix<double> data;
        data.resize(mesh.n_elements(), 2 * N);
        for (int dim = 0; dim < N; ++dim) { c_[dim] = 1.0 / (mesh_.range()(1, dim) - mesh_.range()(0, dim)); }
        int i = 0;
        for (const auto& e : mesh_) {
            std::pair<SVector<N>, SVector<N>> bounding_box = e.bounding_box();
            // unit hypercube point scaling
            data.row(i).leftCols(N)  = (bounding_box.first  - mesh_.range().row(0).transpose()).array() * c_.array();
            data.row(i).rightCols(N) = (bounding_box.second - mesh_.range().row(0).transpose()).array() * c_.array();
            ++i;
        }
        tree_ = KDTree<2 * N>(std::move(data));   // organize elements in a KD-tree structure
    }
    // finds element containing p, returns nullptr if element not found
    const Element<M, N>* locate(const SVector<N>& p) const {
        // build search query
        SVector<N> scaled_p = (p - mesh_.range().row(0).transpose()).array() * c_.array();
        SVector<2 * N> ll, ur;
        ll << SVector<N>::Zero(), scaled_p;
        ur << scaled_p, SVector<N>::Ones();
        auto found = tree_.range_search({ll, ur});
        // exhaustively scan the query results to get the searched mesh element
        for (int id : found) {
            auto element = mesh_.element(id);
            if (element.contains(p)) { return &mesh_.element(id); }
        }
        return nullptr;   // no element found
    }
};

}   // namespace core
}   // namespace fdapde

#endif   // __TREE_SEARCH_H__
