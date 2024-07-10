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

#ifndef __PROJECT_H__
#define __PROJECT_H__

#include "kd_tree.h"
#include "../utils/symbols.h"

namespace fdapde {
namespace core {

template <typename TriangulationType> class Projection {
   private:
    const TriangulationType* mesh_;
    mutable std::optional<KDTree<TriangulationType::embed_dim>> tree_;
   public:
    Projection() = default;
    explicit Projection(const TriangulationType& mesh) : mesh_(&mesh) { }

    DMatrix<double> operator()(const DMatrix<double>& points, tag_exact) const {
        DVector<double> best = DVector<double>::Constant(points.rows(), std::numeric_limits<double>::max());
        DMatrix<double> proj(points.rows(), TriangulationType::embed_dim);
        for (typename TriangulationType::cell_iterator it = mesh_->cells_begin(); it != mesh_->cells_end(); ++it) {
            for (int i = 0; i < points.rows(); ++i) {
                SVector<TriangulationType::embed_dim> proj_point = it->nearest(points.row(i));
                double dist = (proj_point - points.row(i).transpose()).norm();
                if (dist < best[i]) {
                    best[i] = dist;
                    proj.row(i) = proj_point;
                }
            }
        }
        return proj;
    }

    DMatrix<double> operator()(const DMatrix<double>& points, tag_not_exact) const {
        DMatrix<double> proj(points.rows(), TriangulationType::embed_dim);
        // build kdtree of mesh nodes for fast nearest neighborhood searches
        if (!tree_.has_value()) tree_ = KDTree<TriangulationType::embed_dim>(mesh_->nodes());
        for (int i = 0; i < points.rows(); ++i) {
            // find nearest mesh node (in euclidean sense, approximation)
            typename KDTree<TriangulationType::embed_dim>::iterator it = tree_->nn_search(points.row(i));
            // search nearest element in the node patch
            double best = std::numeric_limits<double>::max();
            for (int j : mesh_->node_patch(*it)) {
                SVector<TriangulationType::embed_dim> proj_point = mesh_->cell(j).nearest(points.row(i));
                double dist = (proj_point - points.row(i).transpose()).norm();
                if (dist < best) {
                    best = dist;
                    proj.row(i) = proj_point;
                }
            }
        }
        return proj;
    }
    DMatrix<double> operator()(const DMatrix<double>& points) const { return operator()(points, fdapde::NotExact); }
};

}   // namespace core
}   // namespace fdapde

#endif // __PROJECT_H__
