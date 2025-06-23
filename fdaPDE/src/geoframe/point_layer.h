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

#ifndef __FDAPDE_POINT_LAYER_H__
#define __FDAPDE_POINT_LAYER_H__

#include "header_check.h"

namespace fdapde {
namespace internals {

template <typename Triangulation_> struct point_layer {
    using Triangulation = typename std::remove_pointer_t<std::decay_t<Triangulation_>>;
    using layer_category = point_layer_tag;
    static constexpr int local_dim = Triangulation::local_dim;
    static constexpr int embed_dim = Triangulation::embed_dim;
    using geometry_type = Eigen::Matrix<double, embed_dim, 1>;

    point_layer() : triangulation_(nullptr), coords_(), points_at_dofs_(false) { }
    template <typename GeoDescriptor>
        requires(is_eigen_dense_xpr_v<GeoDescriptor> || is_vector_like_v<GeoDescriptor>)
    point_layer(Triangulation_* triangulation, const GeoDescriptor& coords) noexcept :
        triangulation_(triangulation), points_at_dofs_(false) {
        if constexpr (is_eigen_dense_xpr_v<GeoDescriptor>) {
            coords_.reserve(coords.size());
            for (int i = 0; i < coords.rows(); ++i) {
                for (int j = 0; j < coords.cols(); ++j) { coords_.push_back(coords(i, j)); }
            }
            n_rows_ = coords.rows();
        } else {
            fdapde_assert(coords.size() % embed_dim == 0);
            coords_.reserve(coords.size());
            for (int i = 0, n = coords.size(); i < n; ++i) { coords_.push_back(coords[i]); }
            n_rows_ = coords.size() / embed_dim;
	}
    }
    // geometrical coordinates coincidint with triangulation nodes
    point_layer(Triangulation_* triangulation) noexcept : triangulation_(triangulation), points_at_dofs_(true) {
        coords_.reserve(triangulation_->n_nodes() * embed_dim);
        const auto& coords = triangulation_->nodes();
        for (int i = 0; i < coords.rows(); ++i) {
            for (int j = 0; j < coords.cols(); ++j) { coords_.push_back(coords(i, j)); }
        }
	n_rows_ = triangulation_->n_nodes();
    }
    // observers
    geometry_type coord(int i) const { return coordinates().row(i); }
    geometry_type operator[](int i) const { return coord(i); }
    int rows() const { return n_rows_; }
    bool points_at_dofs() const { return points_at_dofs_; }
    // modifiers
    template <typename... CoordsT>
        requires(sizeof...(CoordsT) == embed_dim) && (std::is_convertible_v<CoordsT, double> && ...)
    void push_back(CoordsT... coords) {
        internals::for_each_index_and_args<embed_dim>(
          [&, this]<int Ns, typename T>(T t) { coords_.push_back(t); }, coords...);
	n_rows_++;
	return;
    }
    template <typename CoordsT>
        requires(is_eigen_dense_xpr_v<CoordsT> || is_vector_like_v<CoordsT>)
    void push_back(const CoordsT& coords) {
        if constexpr (is_eigen_dense_xpr_v<CoordsT>) {
            for (int i = 0; i < coords.rows(); ++i) {
                for (int j = 0; j < coords.cols(); ++j) { coords_.push_back(coords(i, j)); }
            }
            n_rows_ += coords.rows();
        } else {
            fdapde_assert(coords.size() % embed_dim == 0);
	    coords_.insert(coords_.end(), coords.begin(), coords.end());
	    n_rows_ += coords.size() / embed_dim;
        }
	return;
    }
    // geometry
    Triangulation& triangulation() { return *triangulation_; }
    const Triangulation& triangulation() const { return *triangulation_; }
    Eigen::Map<const Eigen::Matrix<double, Dynamic, Dynamic, Eigen::RowMajor>> coordinates() const {
        return Eigen::Map<const Eigen::Matrix<double, Dynamic, Dynamic, Eigen::RowMajor>>(
          coords_.data(), n_rows_, embed_dim);
    }
    // O(n) unique coordinates extraction
    Eigen::Matrix<double, Dynamic, Dynamic> unique_coordinates() const {
        if (points_at_dofs_) { return triangulation().nodes(); }
        // find unique coordinates
        std::unordered_set<geometry_type, eigen_matrix_hash> coords_set;
        std::vector<int> coords_idx;
        auto coords_view = coordinates();   // map coords_ vector into a RowMajor format matrix
        for (int i = 0; i < n_rows_; ++i) {
            Eigen::Matrix<double, embed_dim, 1> p(coords_view.row(i));
            if (!coords_set.contains(p)) {
                coords_set.insert(p);
                coords_idx.push_back(i);
            }
        }
        Eigen::Matrix<double, Dynamic, Dynamic> coords_unique(coords_set.size(), embed_dim);
        int i = 0;
        for (int idx : coords_idx) { coords_unique.row(i++) = coords_view.row(idx); }
        return coords_unique;
    }
    Eigen::Matrix<int, Dynamic, 1> locate() const {
	Eigen::Matrix<double, Dynamic, Dynamic> coords = coordinates();
	return triangulation_->locate(coords);
    }
   private:
    bool points_at_dofs_ = false;
    Triangulation* triangulation_;
    std::vector<double> coords_;
    int n_rows_ = 0;
};

}   // namespace internals
}   // namespace fdapde

#endif // __FDAPDE_POINT_LAYER_H__
