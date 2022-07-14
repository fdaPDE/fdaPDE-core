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

#ifndef __FDAPDE_PROJECTION_H__
#define __FDAPDE_PROJECTION_H__

#include "header_check.h"

namespace fdapde {

template <typename TriangulationType> class Projection {
   private:
    const TriangulationType* mesh_;
    mutable std::optional<KDTree<TriangulationType::embed_dim>> tree_;
   public:
    Projection() = default;
    explicit Projection(const TriangulationType& mesh) : mesh_(&mesh) { }

    Eigen::Matrix<double, Dynamic, Dynamic>
    operator()(const Eigen::Matrix<double, Dynamic, Dynamic>& points, tag_exact) const {
        Eigen::Matrix<double, Dynamic, 1> best =
          Eigen::Matrix<double, Dynamic, 1>::Constant(points.rows(), std::numeric_limits<double>::max());
        Eigen::Matrix<double, Dynamic, Dynamic> proj(points.rows(), TriangulationType::embed_dim);
        for (typename TriangulationType::cell_iterator it = mesh_->cells_begin(); it != mesh_->cells_end(); ++it) {
            for (int i = 0; i < points.rows(); ++i) {
                Eigen::Matrix<double, TriangulationType::embed_dim, 1> proj_point = it->nearest(points.row(i));
                double dist = (proj_point - points.row(i).transpose()).norm();
                if (dist < best[i]) {
                    best[i] = dist;
                    proj.row(i) = proj_point;
                }
            }
        }
        return proj;
    }

    Eigen::Matrix<double, Dynamic, Dynamic>
    operator()(const Eigen::Matrix<double, Dynamic, Dynamic>& points, tag_not_exact) const {
        Eigen::Matrix<double, Dynamic, Dynamic> proj(points.rows(), TriangulationType::embed_dim);
        // build kdtree of mesh nodes for fast nearest neighborhood searches
        if (!tree_.has_value()) tree_ = KDTree<TriangulationType::embed_dim>(mesh_->nodes());
        for (int i = 0; i < points.rows(); ++i) {
            // find nearest mesh node (in euclidean sense, approximation)
            typename KDTree<TriangulationType::embed_dim>::iterator it = tree_->nn_search(points.row(i));
            // search nearest element in the node patch
            double best = std::numeric_limits<double>::max();
            for (int j : mesh_->node_patch(*it)) {
                Eigen::Matrix<double, TriangulationType::embed_dim, 1> proj_point =
                  mesh_->cell(j).nearest(points.row(i));
                double dist = (proj_point - points.row(i).transpose()).norm();
                if (dist < best) {
                    best = dist;
                    proj.row(i) = proj_point;
                }
            }
        }
        return proj;
    }
    Eigen::Matrix<double, Dynamic, Dynamic> operator()(const Eigen::Matrix<double, Dynamic, Dynamic>& points) const {
        return operator()(points, NotExact);
    }
};

}   // namespace fdapde

#endif   // __FDAPDE_PROJECTION_H__
