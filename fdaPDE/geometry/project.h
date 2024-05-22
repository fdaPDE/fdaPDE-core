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

template <typename Policy> struct Project;

template <> struct Project<Exact> {
    template <typename MeshType> static DMatrix<double> compute(const DMatrix<double>& points, const MeshType& mesh) {
        DVector<double> best = DVector<double>::Constant(points.rows(), std::numeric_limits<double>::max());
        DMatrix<double> proj(points.rows(), MeshType::embed_dim);
        for (typename MeshType::cell_iterator it = mesh.cells_begin(); it != mesh.cells_end(); ++it) {
            for (int i = 0; i < points.rows(); ++i) {
                SVector<MeshType::embed_dim> proj_point = it->nearest(points.row(i));
                double dist = (proj_point - points.row(i).transpose()).norm();
                if (dist < best[i]) {
                    best[i] = dist;
                    proj.row(i) = proj_point;
                }
            }
        }
        return proj;
    }
};

template <> struct Project<NotExact> {
    template <typename MeshType> static DMatrix<double> compute(const DMatrix<double>& points, const MeshType& mesh) {
        DMatrix<double> proj(points.rows(), MeshType::embed_dim);
        // build kdtree of mesh nodes for fast nearest neighborhood searches
        KDTree<MeshType::embed_dim> tree_(mesh.nodes());
        for (int i = 0; i < points.rows(); ++i) {
            // find nearest mesh node (in euclidean sense, approximation)
            typename KDTree<MeshType::embed_dim>::iterator it = tree_.nn_search(points.row(i));
            // search nearest element in the node patch
            double best = std::numeric_limits<double>::max();
            for (int j : mesh.node_patch(*it)) {
                SVector<MeshType::embed_dim> proj_point = mesh.cell(j).nearest(points.row(i));
                double dist = (proj_point - points.row(i).transpose()).norm();
                if (dist < best) {
                    best = dist;
                    proj.row(i) = proj_point;
                }
            }
        }
        return proj;
    }
};

}   // namespace core
}   // namespace fdapde

#endif // __PROJECT_H__
