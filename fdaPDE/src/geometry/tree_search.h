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

#ifndef __FDAPDE_TREE_SEARCH_H__
#define __FDAPDE_TREE_SEARCH_H__

#include "header_check.h"

namespace fdapde {

// tree-based point location over triangulation. Based on "J. Bonet, J. Peraire (1991), An alternating digital tree
// (ADT) algorithm for 3D geometric searching and intersection problems"
template <typename MeshType> class TreeSearch {
   private:
    static constexpr int embed_dim = MeshType::embed_dim;
    static constexpr int local_dim = MeshType::local_dim;
    KDTree<2 * embed_dim> tree_;
    const MeshType* mesh_;
    Eigen::Matrix<double, embed_dim, 1> c_;   // normalization constants
    // build search query for point p
    KDTree<2 * embed_dim>::RangeType query(const Eigen::Matrix<double, embed_dim, 1>& p) const {
        Eigen::Matrix<double, embed_dim, 1> scaled_p = (p - mesh_->bbox().row(0).transpose()).array() * c_.array();
	Eigen::Matrix<double, 2 * embed_dim, 1> ll, ur;
        ll << Eigen::Matrix<double, embed_dim, 1>::Zero(), scaled_p;
        ur << scaled_p, Eigen::Matrix<double, embed_dim, 1>::Ones();
        return {ll, ur};
    }
   public:
    TreeSearch() = default;
    TreeSearch(const MeshType* mesh) : mesh_(mesh) {
        // the i-th row of data contains the bounding box of the i-th element, stored as the vector [lower-left,
        // upper-right] corner. This moves each element to a point in R^{2N}
        Eigen::Matrix<double, Dynamic, Dynamic> data;
        data.resize(mesh_->n_cells(), 2 * embed_dim);
        for (int dim = 0; dim < embed_dim; ++dim) { c_[dim] = 1.0 / (mesh_->bbox()(1, dim) - mesh_->bbox()(0, dim)); }
        int i = 0;
        for (typename MeshType::cell_iterator it = mesh_->cells_begin(); it != mesh_->cells_end(); ++it) {
            std::pair<Eigen::Matrix<double, embed_dim, 1>, Eigen::Matrix<double, embed_dim, 1>> bbox =
              it->bounding_box();
            // unit hypercube point scaling
            data.row(i).leftCols(embed_dim)  = (bbox.first  - mesh_->bbox().row(0).transpose()).array() * c_.array();
            data.row(i).rightCols(embed_dim) = (bbox.second - mesh_->bbox().row(0).transpose()).array() * c_.array();
            ++i;
        }
        tree_ = KDTree<2 * embed_dim>(std::move(data));   // organize elements in a KD-tree structure
    }
    // finds all the elements containing p
    std::vector<int> all_locate(const Eigen::Matrix<double, embed_dim, 1>& p) const {
        std::vector<int> result;
        for (int id : tree_.range_search(query(p))) {
            typename MeshType::CellType c = mesh_->cell(id);
            if (c.contains(p)) { result.push_back(c.id()); }
        }
        return result;
    }
    // finds element containing p, returns -1 if element not found
    int locate(const Eigen::Matrix<double, embed_dim, 1>& p) const {
        // exhaustively scan the query results to get the searched mesh element
        for (int id : tree_.range_search(query(p))) {
            typename MeshType::CellType c = mesh_->cell(id);
            if (c.contains(p)) { return c.id(); }
        }
        return -1;   // no element found
    }
    template <typename CoordsMatrix>
        requires(internals::is_eigen_dense_xpr_v<CoordsMatrix>)
    Eigen::Matrix<int, Dynamic, 1> locate(const CoordsMatrix& locs) const {
        fdapde_assert(locs.cols() == embed_dim);
        Eigen::Matrix<int, Dynamic, 1> ids(locs.rows());
        for (int i = 0; i < locs.rows(); ++i) { ids[i] = locate(Eigen::Matrix<double, embed_dim, 1>(locs.row(i))); }
        return ids;
    }
};

}   // namespace fdapde

#endif   // __FDAPDE_TREE_SEARCH_H__
