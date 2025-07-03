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

#ifndef __FDAPDE_AREAL_LAYER_H__
#define __FDAPDE_AREAL_LAYER_H__

#include "header_check.h"

namespace fdapde {
namespace internals {

  // TODO: count number of points in a region
  
template <typename Triangulation_> struct areal_layer {
    using Triangulation = typename std::remove_pointer_t<std::decay_t<Triangulation_>>;
    using layer_category = areal_layer_tag;
    using binary_t = BinaryMatrix<Dynamic, Dynamic>;
    static constexpr int local_dim = Triangulation::local_dim;
    static constexpr int embed_dim = Triangulation::embed_dim;

    areal_layer() : triangulation_(nullptr), incidence_matrix_(), n_regions_(0) { }
    template <typename GeoDescriptor>
        requires(is_vector_like_v<GeoDescriptor> &&
                 std::is_convertible_v<subscript_result_of_t<GeoDescriptor, int>, int>)
    areal_layer(Triangulation_* triangulation, const GeoDescriptor& regions) noexcept :
        triangulation_(triangulation), incidence_matrix_(), n_regions_(0) {
        fdapde_assert(std::cmp_equal(regions.size() FDAPDE_COMMA triangulation_->n_cells()));
        // count number of regions
        std::unordered_set<int> unique_region_ids;
        for (int i = 0, n = regions.size(); i < n; ++i) {
            fdapde_assert(regions[i] >= 0);
            unique_region_ids.insert(regions[i]);
        }
        n_regions_ = unique_region_ids.size();
        // compute incidence matrix
        incidence_matrix_.resize(n_regions_, triangulation_->n_cells());
        for (int i = 0, n = triangulation_->n_cells(); i < n; ++i) { incidence_matrix_.set(regions[i], i); }
    }
    areal_layer(Triangulation* triangulation, const binary_t& incidence_matrix) noexcept :
        triangulation_(triangulation), incidence_matrix_(incidence_matrix), n_regions_(incidence_matrix.rows()) {
        fdapde_assert(incidence_matrix.cols() == triangulation_->n_cells());
    }
    template <typename GeoDescriptor>
        requires(is_vector_like_v<GeoDescriptor> &&
                 std::is_same_v<
                   std::decay_t<subscript_result_of_t<GeoDescriptor, int>>, MultiPolygon<local_dim, embed_dim>>)
    areal_layer(Triangulation_* triangulation, const GeoDescriptor& regions) noexcept :
        triangulation_(triangulation), incidence_matrix_(), n_regions_(0) {
        fdapde_assert(regions.size() == triangulation_->n_cells());
        // count number of regions
        n_regions_ = regions.size();
        incidence_matrix_.resize(n_regions_, triangulation_->n_cells());
        for (auto it = triangulation_->cells_begin(); it != triangulation_->cells_end(); ++it) {
            for (int i = 0; i < n_regions_; ++i) {
                if (regions[i].contains(it->barycenter())) { incidence_matrix_[it->id()] = i; }
            }
	}
    }
    areal_layer(Triangulation_* triangulation, const std::vector<BinaryMatrix<Dynamic, 1>>& regions) noexcept :
        triangulation_(triangulation), incidence_matrix_(), n_regions_(regions.size()) {
        incidence_matrix_.resize(n_regions_, triangulation_->n_cells());
        for (int i = 0; i < n_regions_; ++i) {
            fdapde_assert(regions[i].rows() == triangulation_->n_cells());
            for (int j = 0, n = triangulation_->n_cells(); j < n; ++j) {
                if (regions[i][j]) incidence_matrix_.set(i, j);
            }
        }
    }

    // observers
    int rows() const { return n_regions_; }
    // geometry
    BinaryVector<Dynamic> operator[](int i) const {
        fdapde_assert(i < n_regions_);
        return incidence_matrix_.row(i);
    }
    Triangulation& triangulation() { return *triangulation_; }
    const Triangulation& triangulation() const { return *triangulation_; }
    // computes measures of subdomains
    std::vector<double> measures() const {
        std::vector<double> m_(n_regions_, 0);
        for (auto it = triangulation().cells_begin(); it != triangulation().cells_end(); ++it) {
            for (int i = 0; i < n_regions_; ++i) {
                if (incidence_matrix_(i, it->id())) { m_[i] += it->measure(); }
            }
        }
        return m_;
    }
    const BinaryMatrix<Dynamic, Dynamic>& incidence_matrix() const { return incidence_matrix_; }
   private:
    Triangulation* triangulation_;
    binary_t incidence_matrix_;   // [M]_{ij} : [M]_{ij} == 1 \iff cell j is inside region i, 0 otherwise
    int n_regions_;
};

}   // namespace internals
}   // namespace fdapde

#endif // __FDAPDE_AREAL_LAYER_H__
