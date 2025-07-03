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

#ifndef __FDAPDE_GEO_LAYER_H__
#define __FDAPDE_GEO_LAYER_H__

#include "header_check.h"

namespace fdapde {

namespace internals {

struct point_layer_descriptor {
    using layer_tag = point_layer_tag;
    template <int LocalDim, int EmbedDim> using value_type = double;
};
struct polygon_layer_descriptor {
    using layer_tag = areal_layer_tag;
    template <int LocalDim, int EmbedDim> using value_type = BinaryMatrix<Dynamic, 1>;
};

template <typename LayerTag, typename Triangulation>
using layer_type_from_layer_tag = std::conditional_t<
  std::is_same_v<LayerTag, point_layer_descriptor>, point_layer<Triangulation>, areal_layer<Triangulation>>;

// a geo-column is a named single-typed block of **contiguous** data from a plain_data_layer, paired with geometry
template <typename Scalar_, typename GeoLayer>
struct geo_col_view :
    public plain_col_view<
      Scalar_,
      std::conditional_t<
        std::is_const_v<GeoLayer>, std::add_const_t<typename GeoLayer::storage_t>, typename GeoLayer::storage_t>> {
    using Base = plain_col_view<
      Scalar_,
      std::conditional_t<
        std::is_const_v<GeoLayer>, std::add_const_t<typename GeoLayer::storage_t>, typename GeoLayer::storage_t>>;
    using GeoInfo = typename GeoLayer::GeoInfo;
    using size_t = typename Base::size_t;
    using index_t = typename Base::index_t;
    static constexpr int Order = GeoLayer::Order;
    using Base::rows_;

    geo_col_view() noexcept = default;
    template <typename FieldDescriptor>
    geo_col_view(GeoLayer* geo_data, index_t row_begin, index_t row_end, const FieldDescriptor& desc) noexcept :
        Base(geo_data->data(), row_begin, row_end, desc), row_begin_(row_begin), geo_data_(geo_data) { }
    template <typename FieldDescriptor>   // column row constructor
    geo_col_view(GeoLayer* geo_data, index_t row, const FieldDescriptor& desc) noexcept :
        Base(geo_data->data(), row, desc), row_begin_(row), geo_data_(geo_data) { }
    template <typename FieldDescriptor>   // full column constructor
    geo_col_view(GeoLayer* geo_data, const FieldDescriptor& desc) noexcept :
        Base(geo_data->data(), desc), row_begin_(0), geo_data_(geo_data) { }
    // observers
    auto geometry(int i) const {
        fdapde_assert(i < rows_);
        return geo_data_->geometry(row_begin_ + i);
    }
    const GeoLayer& geo_data() const { return *geo_data_; }
   private:
    size_t row_begin_;
    GeoLayer* geo_data_;
};

template <typename Scalar_, typename GeoLayer>
struct random_access_geo_col_view :
    public random_access_col_view<
      Scalar_,
      std::conditional_t<
        std::is_const_v<GeoLayer>, std::add_const_t<typename GeoLayer::storage_t>, typename GeoLayer::storage_t>> {
    using Base = random_access_col_view<
      Scalar_,
      std::conditional_t<
        std::is_const_v<GeoLayer>, std::add_const_t<typename GeoLayer::storage_t>, typename GeoLayer::storage_t>>;
    using GeoInfo = typename GeoLayer::GeoInfo;
    using size_t = typename Base::size_t;
    using index_t = typename Base::index_t;
    static constexpr int Order = GeoLayer::Order;

    random_access_geo_col_view() noexcept = default;
    template <typename FieldDescriptor>
    random_access_geo_col_view(
      GeoLayer* geo_data, const std::vector<index_t>& idxs, const FieldDescriptor& desc) noexcept :
        Base(geo_data->data(), idxs, desc), geo_data_(geo_data) { }
    // observers
    auto geometry(int i) const { return geo_data_->geometry(this->idxs_[i]); }
    const GeoLayer& geo_data() const { return *geo_data_; }
   private:
    GeoLayer* geo_data_;
};

template <typename GeoLayer>
struct geo_row_view :
    public plain_row_view<std::conditional_t<
      std::is_const_v<GeoLayer>, std::add_const_t<typename GeoLayer::storage_t>, typename GeoLayer::storage_t>> {
    using Base = plain_row_view<std::conditional_t<
      std::is_const_v<GeoLayer>, std::add_const_t<typename GeoLayer::storage_t>, typename GeoLayer::storage_t>>;
    using This = geo_row_view<GeoLayer>;
    using GeoInfo = typename GeoLayer::GeoInfo;
    static constexpr int Order = GeoLayer::Order;
    using Base::row_;

    geo_row_view() noexcept = default;
    geo_row_view(GeoLayer* geo_data, int row) : Base(std::addressof(geo_data->data()), row), geo_data_(geo_data) {
        fdapde_assert(row < geo_data_->rows());
    }
    // observers
    auto geometry() const { return geo_data_->geometry(this->id()); }
    const GeoLayer& geo_data() const { return *geo_data_; }
    // accessors
    template <typename T> geo_col_view<T, This> col(const std::string& colname) {
        return geo_col_view<T, This>(geo_data_, row_, Base::field_descriptor(colname));
    }
    template <typename T> geo_col_view<T, const This> col(const std::string& colname) const {
        return geo_col_view<T, const This>(geo_data_, row_, Base::field_descriptor(colname));
    }
    template <typename T> geo_col_view<T, This> col(size_t col) {
        return geo_col_view<T, This>(geo_data_, row_, Base::field_descriptor(col));
    }
    template <typename T> geo_col_view<T, const This> col(size_t col) const {
        return geo_col_view<T, const This>(geo_data_, row_, Base::field_descriptor(col));
    }
   private:
    GeoLayer* geo_data_;
};

// an indexed set of rows coupled with geometric information
template <typename GeoLayer>
struct random_access_geo_row_view :
    public random_access_row_view<std::conditional_t<
      std::is_const_v<GeoLayer>, std::add_const_t<typename GeoLayer::storage_t>, typename GeoLayer::storage_t>> {
    using Base = random_access_row_view<std::conditional_t<
      std::is_const_v<GeoLayer>, std::add_const_t<typename GeoLayer::storage_t>, typename GeoLayer::storage_t>>;
    using This = random_access_geo_row_view<GeoLayer>;
    using GeoInfo = typename GeoLayer::GeoInfo;
    using size_t = typename GeoLayer::size_t;
    using index_t = typename GeoLayer::index_t;
    static constexpr int Order = GeoLayer::Order;
    using Base::idxs_;

    random_access_geo_row_view() noexcept = default;
    // index-based filtering mode (inherited from plain_data_layer)
    template <typename Iterator>
    random_access_geo_row_view(GeoLayer* geo_data, Iterator begin, Iterator end) :
        Base(std::addressof(geo_data->data()), begin, end), geo_data_(geo_data) {
        fdapde_assert(std::all_of(begin FDAPDE_COMMA end FDAPDE_COMMA [this](index_t i) {
            return std::cmp_less(i FDAPDE_COMMA geo_data_->rows());
        }));
    }
    template <typename Filter>
        requires(requires(Filter f, index_t i) {
                    { f(i) } -> std::same_as<bool>;
                })
    random_access_geo_row_view(GeoLayer* geo_data, Filter&& f) :
        Base(std::addressof(geo_data->data()), f), geo_data_(geo_data) { }
    template <typename LogicalVec>
        requires(requires(LogicalVec vec, int i) {
                    { vec.size() } -> std::convertible_to<size_t>;
                    { vec[i] } -> std::convertible_to<bool>;
                })
    random_access_geo_row_view(GeoLayer* geo_data, const LogicalVec& vec) :
        Base(std::addressof(geo_data->data()), vec), geo_data_(geo_data) {
        fdapde_assert(vec.size() < geo_data_->rows());
    }
    // observers
    auto geometry(int i) const { return geo_data_->geometry(this->idxs_[i]); }
    const GeoLayer& geo_data() const { return *geo_data_; }
    // accessors
    template <typename T> random_access_geo_col_view<T, GeoLayer> col(const std::string& colname) {
        return random_access_geo_col_view<T, GeoLayer>(geo_data_, idxs_, Base::field_descriptor(colname));
    }
    template <typename T> random_access_geo_col_view<T, const GeoLayer> col(const std::string& colname) const {
        return random_access_geo_col_view<T, const GeoLayer>(geo_data_, idxs_, Base::field_descriptor(colname));
    }
    template <typename T> random_access_geo_col_view<T, GeoLayer> col(size_t col) {
        return random_access_geo_col_view<T, GeoLayer>(geo_data_, idxs_, Base::field_descriptor(col));
    }
    template <typename T> random_access_geo_col_view<T, const GeoLayer> col(size_t col) const {
        return random_access_geo_col_view<T, const GeoLayer>(geo_data_, idxs_, Base::field_descriptor(col));
    }
   private:
    GeoLayer* geo_data_;
};

}   // namespace internals

// available geometric features
using POLYGON = internals::polygon_layer_descriptor;
using POINT   = internals::point_layer_descriptor;

// geometrical layer, a plain_data_layer coupled with geometrical indexing
template <typename Triangulation_, typename GeoInfo_>
    requires(
      internals::is_tuple_v<GeoInfo_> && internals::is_tuple_v<Triangulation_> &&
      std::tuple_size_v<Triangulation_> == std::tuple_size_v<GeoInfo_>)
struct GeoLayer {
   private:
    template <typename U, typename V> struct geo_storage_t_impl;
    template <typename... Us, typename... Vs> struct geo_storage_t_impl<std::tuple<Us...>, std::tuple<Vs...>> {
        using type = std::tuple<internals::layer_type_from_layer_tag<Us, Vs>...>;
    };
    template <typename U> struct triangulation_storage;
    template <typename... Us> struct triangulation_storage<std::tuple<Us...>> {
        using type = std::tuple<std::add_pointer_t<Us>...>;
    };
    using triangulation_t = typename triangulation_storage<Triangulation_>::type;
    using This = GeoLayer<Triangulation_, GeoInfo_>;
    template <int N, typename GeoTag>
    static constexpr bool is_geo_v = std::is_same_v<std::tuple_element_t<N, GeoInfo_>, GeoTag>;
   public:
    using storage_t = internals::scalar_data_layer;
    using geo_storage_t = typename geo_storage_t_impl<GeoInfo_, Triangulation_>::type;
    using index_t = typename storage_t::index_t;
    using size_t = typename storage_t::size_t;
    using logical_t = typename storage_t::logical_t;
    using Triangulation = Triangulation_;
    using GeoInfo = GeoInfo_;
    template <typename T> using reference = typename storage_t::template reference<T>;
    template <typename T> using const_reference = typename storage_t::template const_reference<T>;
    static constexpr int Order = std::tuple_size_v<Triangulation>;
    static constexpr std::array<int, Order> local_dim = internals::apply_index_pack<Order>(
      []<int... Ns>() { return std::array<int, Order> {std::tuple_element_t<Ns, Triangulation>::local_dim...}; });
    static constexpr std::array<int, Order> embed_dim = internals::apply_index_pack<Order>(
      []<int... Ns>() { return std::array<int, Order> {std::tuple_element_t<Ns, Triangulation>::embed_dim...}; });
    static constexpr bool is_full_geo_point = []() {
        bool value_ = true;
        internals::for_each_index_in_pack<Order>([&]<int Ns>() {
            value_ &= std::is_same_v<std::tuple_element_t<Ns, GeoInfo>, internals::point_layer_descriptor>;
        });
        return value_;
    }();

   public:
    // constructor
    GeoLayer(triangulation_t triangulation) :
        data_(), geo_data_(), triangulation_(triangulation), n_rows_(0), strides_(), extents_(), structured_(false) {
        std::fill(strides_.begin(), strides_.end(), 1);   // default to unstructured layer
    }
    template <typename GeoData_>
    GeoLayer(triangulation_t triangulation, const GeoData_& geo_data) :
        data_(), geo_data_(), triangulation_(triangulation), n_rows_(0), strides_(), extents_(), structured_(false) {
        std::fill(strides_.begin(), strides_.end(), 1);   // default to unstructured layer

        if constexpr (Order == 1 || (!internals::is_tuple_v<GeoData_> && is_full_geo_point)) {
            load_geometry_(geo_data);
        } else {
            static_assert(std::tuple_size_v<GeoData_> == Order);
            // load geomtric indexes
            internals::for_each_index_in_pack<std::tuple_size_v<GeoData_>>(
              [&]<int Ns_>() { load_geometry_index_<Ns_>(std::get<Ns_>(geo_data)); });
            // move to structured layer
            if constexpr (is_full_geo_point) {
                make_structured();
            } else {
                n_rows_ = 1;
                internals::for_each_index_in_pack<Order>([&]<int Ns_>() {
                    extents_[Ns_] = std::get<Ns_>(geo_data_).rows();
                    n_rows_ *= extents_[Ns_];
                });
                structured_ = true;
                // set strides
                strides_[0] = 1;
                for (int i = 1; i < Order; ++i) {
                    for (int j = 0; j < i; ++j) { strides_[i] *= extents_[j]; }
                }
            }
        }
    }
    template <typename LayerType>
        requires(LayerType::Order == Order && std::is_same_v<typename LayerType::GeoInfo, GeoInfo>)
    GeoLayer(const internals::random_access_geo_row_view<LayerType>& row_filter, const std::vector<std::string>& cols) :
        data_(row_filter, cols),
        triangulation_(row_filter.geo_data().triangulation()),
        n_rows_(row_filter.rows()),
        strides_(),
        extents_(),
        structured_(false) {
        std::fill(strides_.begin(), strides_.end(), 1);   // unstructured layer by construction
	// collect geometry from filtering predicate
        using mem_t = decltype(internals::apply_index_pack<Order>([]<int... Ns>() {
            return std::tuple<std::vector<
              typename std::tuple_element_t<Ns, GeoInfo>::template value_type<local_dim[Ns], embed_dim[Ns]>>...> {};
        }));
        mem_t geo_data;
        for (int i = 0, n = row_filter.rows(); i < n; ++i) {
            auto geometry = row_filter.geometry(i);
            internals::for_each_index_in_pack<Order>([&]<int Ns_>() {
                if constexpr (is_geo_v<Ns_, POINT>) {
                    for (int i = 0; i < embed_dim[Ns_]; ++i) {
                        std::get<Ns_>(geo_data).push_back(std::get<Ns_>(geometry)[i]);
                    };
                }
                if constexpr (is_geo_v<Ns_, POLYGON>) { std::get<Ns_>(geo_data).push_back(std::get<Ns_>(geometry)); }
            });
        }
	// initialize geo indexes
        internals::for_each_index_in_pack<Order>([&, this]<int Ns_>() {
            std::get<Ns_>(geo_data_) =
              std::tuple_element_t<Ns_, geo_storage_t>(std::get<Ns_>(triangulation_), std::get<Ns_>(geo_data));
        });
    }
    template <typename LayerType>
        requires(LayerType::Order == Order && std::is_same_v<typename LayerType::GeoInfo, GeoInfo>)
    GeoLayer(const internals::random_access_geo_row_view<LayerType>& row_filter) :
        GeoLayer(row_filter, row_filter.colnames()) { }
    // observers
    size_t rows() const { return n_rows_; }
    size_t cols() const { return data_.cols(); }
    size_t size() const { return rows() * cols(); }
    std::string crs() const { return crs_; }
    const auto& field_descriptor(const std::string& colname) const { return data_.field_descriptor(colname); }
    const auto& header() const { return data_.header(); }
    bool contains(const std::string& column) const { return data_.contains(column); }
    const storage_t& data() const { return data_; }
    logical_t nan(const std::string& colname) const { return data_.nan(colname); }
    logical_t nan(const std::vector<std::string>& colnames) const { return data_.nan(colnames); }
    bool is_structured() const { return structured_; }
    // modifiers
    storage_t& data() { return data_; }
    void make_structured() {
        if (Order == 1 || structured_) { return; }
        internals::for_each_index_in_pack<Order>(
          [&]<int Ns_>() { fdapde_assert(std::get<Ns_>(geo_data_).rows() != 0); });

        constexpr int full_embed_dim = std::accumulate(embed_dim.begin(), embed_dim.end(), 0);	
        using point_t = std::array<double, full_embed_dim>;
        // compute full point set for each layer direction
        geo_storage_t grid_geo_data_;
        internals::for_each_index_in_pack<Order>([&, this]<int Ns_>() mutable {
            auto& layer = std::get<Ns_>(geo_data_);
            std::get<Ns_>(grid_geo_data_) =
              std::tuple_element_t<Ns_, geo_storage_t>(std::get<Ns_>(triangulation_), layer.unique_coordinates());
            extents_[Ns_] = std::get<Ns_>(grid_geo_data_).rows();
        });
        // build temporary point index for amortized O(1) search
        std::unordered_set<point_t, internals::std_array_hash<double, full_embed_dim>> grid_set_;
        point_t p;
        if (data_.rows() != 0) {
            for (int i = 0; i < n_rows_; ++i) {
                int offset = 0;
                internals::for_each_index_in_pack<Order>([&]<int Ns_>() {
                    const auto& coords = std::get<Ns_>(geo_data_).coordinates();
                    for (int j = 0; j < coords.cols(); ++j) { p[offset + j] = coords(i, j); }
                    offset += embed_dim[Ns_];
                });
                grid_set_.insert(p);
            }
        }
        // overall number of points in structured grid
        n_rows_ = std::accumulate(extents_.begin(), extents_.end(), 1, std::multiplies<int>());
        if (data_.rows() != 0) {
            storage_t grid_data(data_.header());
            // reserve memory
            internals::foreach_dtype([&]<typename T>() {
                if (data_.template data<T>().extent(0) > 0) {
                    internals::apply_index_pack<Order>([&]<int... Ns_>() {
                        grid_data.template conservative_resize<T>(
                          (void(Ns_), Ns_ == 0 ? n_rows_ : data_.template data<T>().extent(Ns_))...);
                    });
                }
            });
            // start loop
            std::array<int, Order> multi_index_ {};
            std::fill_n(multi_index_.begin(), Order, 0);
            int src_row = 0;
            for (int i = 0; i < n_rows_; ++i) {
                // check if coordinate was in the initial unstructured grid
                int offset = 0;
                internals::for_each_index_in_pack<Order>([&]<int Ns_>() {   // build tensor product coordinate
                    const auto& coords = std::get<Ns_>(grid_geo_data_).coord(multi_index_[Ns_]);
                    for (int j = 0; j < embed_dim[Ns_]; ++j) { p[offset + j] = coords[j]; }
                    offset += embed_dim[Ns_];
                });
                bool found = grid_set_.contains(p);
                {   // increase multi index
                    multi_index_[0]++;
                    int k = 0;
                    while (multi_index_[k] >= extents_[k]) {
                        multi_index_[k] = 0;
                        multi_index_[++k]++;
                    }
                }
		// directly copy in memory storage, or place NA if coordinate was not in data
                for (const auto& field : data_.header()) {
                    internals::dispatch_to_dtype(
                      field.type_id(),
                      [&]<typename T>(storage_t& dst) {
                          auto slice = dst.template data<T>().template slice<0>(i);
                          if (found) {
                              slice.assign_inplace_from(data_.template data<T>().template slice<0>(src_row));
                          } else {
                              // push NA
                              for (auto it = slice.begin(); it != slice.end(); ++it) {
                                  if constexpr (std::is_same_v<T, std::string>) {
                                      *it = std::string("NA");
                                  } else {
                                      *it = std::numeric_limits<T>::quiet_NaN();
                                  }
                              }
                          }
                      },
                      grid_data);
                }
                if (found) { src_row++; }
            }
            data_ = grid_data;
        }
        geo_data_ = grid_geo_data_;
	// set strides
	strides_[0] = 1;
        for (int i = 1; i < Order; ++i) {
            for (int j = 0; j < i; ++j) { strides_[i] *= extents_[j]; }
        }
        structured_ = true;
        return;
    }

    // data import
    template <typename T> void load_csv(const std::vector<std::string>& colnames, const std::string& filename) {
        auto csv = read_csv<T>(filename);
	fdapde_assert(csv.cols() == colnames.size());
	load_table_<T>(colnames, csv);
        return;
    }
    template <typename T> void load_csv(const std::string& filename) {
        auto csv = read_csv<T>(filename);
        load_table_<T>(csv.colnames(), csv);
        return;
    }
    template <typename T> void load_txt(const std::vector<std::string>& colnames, const std::string& filename) {
        auto txt = read_txt<T>(filename);
        fdapde_assert(txt.cols() == colnames.size());
        load_table_<T>(colnames, txt);
        return;
    }
    template <typename T> void load_txt(const std::string& filename) {
        auto txt = read_txt<T>(filename);
        load_table_<T>(txt.colnames(), txt);
        return;
    }
    void load_shp(const std::string& filename) {
        fdapde_static_assert(
          Order == 1 && std::is_same_v<std::tuple_element_t<0 FDAPDE_COMMA GeoInfo> FDAPDE_COMMA
                                         internals::polygon_layer_descriptor>,
          THIS_METHOD_IS_FOR_POLYGONAL_ORDER_ONE_LAYERS_ONLY);
        std::string filename_ = std::filesystem::current_path().string() + "/" + filename;
        if (!std::filesystem::exists(filename_)) { throw std::runtime_error("file " + filename_ + " not found."); }
        SHPFile shp(filename_);
        n_rows_ = shp.n_records();
        // dispatch to processing logic
        switch (shp.shape_type()) {
        case shp_reader::Polygon: {
            // load polygon as areal layer
            std::vector<MultiPolygon<local_dim[0], embed_dim[0]>> mpolys;
            mpolys.reserve(shp.n_records());
            for (int i = 0, n = shp.n_records(); i < n; ++i) { mpolys.emplace_back(shp.polygon(i).nodes()); }
            // compute region partitioning
            using Triangulation = std::tuple_element_t<0, Triangulation_>;
            const Triangulation& D = *std::get<0>(triangulation_);
            std::vector<int> regions(D.n_cells(), 0);
            for (auto it = D.cells_begin(); it != D.cells_end(); ++it) {
                for (int i = 0, n = mpolys.size(); i < n; ++i) {
                    if (mpolys[i].contains(it->barycenter())) {
                        regions[it->id()] = i;
                        break;
                    }
                }
            }
            std::get<0>(geo_data_) = std::tuple_element_t<0, geo_storage_t>(std::get<0>(triangulation_), regions);
            break;
        }
        }
        for (const auto& [name, field_type] : shp.field_descriptors()) {
            if (field_type == 'N' || field_type == 'F') { load_vec(name, shp.get<double>(name)); }
            if (field_type == 'C') { load_vec(name, shp.get<std::string>(name)); }
        }
	crs_ = shp.gcs();
	return;
    }
    template <typename... Ts>
        requires(internals::is_vector_like_v<Ts> && ...)
    void load_vec(const std::vector<std::string>& colnames, const Ts&... data) {
        fdapde_assert(colnames.size() == sizeof...(data));
        internals::for_each_index_and_args<sizeof...(data)>(
          [&, this]<int Ns_, typename Ts_>(const Ts_& ts) {
              fdapde_assert(std::cmp_equal(ts.size() FDAPDE_COMMA n_rows_));
              data_.append_vec(colnames[Ns_], ts);
          },
          data...);
        return;
    }
    template <typename T>
        requires(internals::is_vector_like_v<T>)
    void load_vec(const std::string& colname, const T& data) {
        load_vec(std::vector<std::string> {colname}, data);
        return;
    }
  
    template <typename T> void load_blk(const std::string& colname, const T& data) {
        fdapde_assert(std::cmp_equal(data.rows() FDAPDE_COMMA n_rows_));
        data_.append_blk(colname, data);
        return;
    }
    // erase column
    void erase(const std::string& colname) { data_.erase(colname); }
    // geometry access
    const triangulation_t& triangulation() const { return triangulation_; }
    template <int N> auto& geometry() {
        fdapde_static_assert(N < Order, OUT_OF_BOUND_ACCESS);
        return std::get<N>(geo_data_);
    }
    template <int N> const auto& geometry() const {
        fdapde_static_assert(N < Order, OUT_OF_BOUND_ACCESS);
        return std::get<N>(geo_data_);
    }
    auto geometry(int i) const {
        fdapde_assert(i < n_rows_);
	using internals::apply_index_pack;
        if constexpr (Order == 1) {
            return apply_index_pack<Order>([&]<int... Ns>() { return std::make_tuple(geometry<Ns>()[i]...); });
        } else {
            if (structured_) {
                // move index i to tensorized index (i_1, i_2, ..., i_Order)
                std::array<int, Order> index {};
                for (int j = 0; j < Order - 1; ++j) {
                    while ((i - strides_[Order - 1 - j]) >= 0) {
                        i = i - strides_[Order - 1 - j];
                        index[Order - 1 - j]++;
                    }
                }
                index[0] = i;
                return apply_index_pack<Order>(
                  [&]<int... Ns>() { return std::make_tuple(geometry<Ns>()[index[Ns]]...); });
            } else {
                return apply_index_pack<Order>([&]<int... Ns>() { return std::make_tuple(geometry<Ns>()[i]...); });
            }
        }
    }

    // column access
    template <typename T> internals::geo_col_view<T, GeoLayer> col(size_t col) {
        fdapde_assert(col < cols());
        return internals::geo_col_view<T, GeoLayer>(this, data_.header()[col]);
    }
    template <typename T> internals::geo_col_view<T, const GeoLayer> col(size_t col) const {
        fdapde_assert(col < cols());
        return internals::geo_col_view<T, const GeoLayer>(this, data_.header()[col]);
    }
    template <typename T> internals::geo_col_view<T, GeoLayer> col(const std::string& colname) {
        fdapde_assert(data_.contains(colname));
	int i = 0;
        for (; i < cols() && data_.header()[i].colname() != colname; ++i);
        return col<T>(i);
    }
    template <typename T> internals::geo_col_view<T, const GeoLayer> col(const std::string& colname) const {
        fdapde_assert(data_.contains(colname));
	int i = 0;
        for (; i < cols() && data_.header()[i].colname() != colname; ++i);
        return col<T>(i);
    }
    // row access
    internals::geo_row_view<This> row(size_t row) { return internals::geo_row_view<This>(this, row); }
    internals::geo_row_view<const This> row(size_t row) const { return internals::geo_row_view<const This>(this, row); }
  
    // row filtering operations (const access)
    template <typename Iterator>
        requires(internals::is_integer_v<typename Iterator::value_type>)
    internals::random_access_geo_row_view<const This> select(Iterator begin, Iterator end) const {
        return internals::random_access_geo_row_view<const This>(this, begin, end);
    }
    template <typename T>
        requires(std::is_convertible_v<T, index_t>)
    internals::random_access_geo_row_view<const This> select(const std::initializer_list<T>& idxs) const {
        return internals::random_access_geo_row_view<const This>(this, idxs.begin(), idxs.end());
    }
    template <typename LogicalPred>
        requires(
	  requires(LogicalPred pred, index_t i) { { pred(i) } -> std::convertible_to<bool>; } ||
          requires(LogicalPred pred, index_t i) { { pred[i] } -> std::convertible_to<bool>; })
    internals::random_access_geo_row_view<const This> select(const LogicalPred& pred) const {
        return internals::random_access_geo_row_view<const This>(this, pred);
    }
    // row filtering operations (non-const access)
    template <typename Iterator>
        requires(internals::is_integer_v<typename Iterator::value_type>)
    internals::random_access_geo_row_view<This> select(Iterator begin, Iterator end) {
        return internals::random_access_geo_row_view<This>(this, begin, end);
    }
    template <typename T>
        requires(std::is_convertible_v<T, index_t>)
    internals::random_access_geo_row_view<This> select(const std::initializer_list<T>& idxs) {
        return internals::random_access_geo_row_view<This>(this, idxs.begin(), idxs.end());
    }
    template <typename LogicalPred>
        requires(
	  requires(LogicalPred pred, index_t i) { { pred(i) } -> std::convertible_to<bool>; } ||
          requires(LogicalPred pred, index_t i) { { pred[i] } -> std::convertible_to<bool>; })
    internals::random_access_geo_row_view<This> select(const LogicalPred& pred) {
        return internals::random_access_geo_row_view<This>(this, pred);
    }
  
    // output stream
    friend std::ostream& operator<<(std::ostream& os, const GeoLayer& data) {
      	int n_rows = std::min(size_t(8), data.rows());
        int n_cols = Order + data.cols();
        std::vector<std::vector<std::string>> out;
        out.resize(n_cols);
        std::vector<std::size_t> max_size(n_cols, 0);
        auto stringify = []<typename T>(const T& t, bool is_na) {
            if (is_na) { return std::string("NA"); }
            if constexpr (!std::is_same_v<T, std::string>) {
                return std::to_string(t);
            } else {
                return t;
            }
        };
        using field = typename storage_t::field_t;
        auto print = [&]<typename T>(std::vector<std::string>& out_, const std::string& typestring, const field& desc) {
            out_.push_back(desc.colname());
            std::string info_ = "<" + std::to_string(desc.size()) + ",1:" + typestring + ">";
            out_.push_back(info_);
            // print actual data
            auto col = internals::plain_col_view<T, const storage_t>(data.data(), 0, n_rows, desc);
            auto nan = col.nan();   // extract column nan pattern
		
            for (int i = 0; i < n_rows; ++i) {
                std::string datastr;
                datastr += stringify(col(i, 0), nan(i, 0));
                if (desc.size() == 2) { datastr += " " + stringify(col(i, 1), nan(i, 1)); }
                if (desc.size() >= 3) {
                    datastr += " ... " + stringify(col(i, desc.size() - 1), nan(i, desc.size() - 1));
                }
                out_.push_back(datastr);
            }
        };
        // push geometrical description
        internals::for_each_index_in_pack<Order>([&]<int Ns>() {
            out[Ns].push_back("");
            using geo_t = std::tuple_element_t<Ns, GeoInfo>;
            if constexpr (std::is_same_v<geo_t, internals::point_layer_descriptor>) {
                // print point coordinate
                out[Ns].push_back("<POINT>");
                for (int i = 0; i < n_rows; ++i) {
                    auto coord = std::get<Ns>(data.geometry(i));
                    std::string coord_str = "";
                    for (int j = 0; j < coord.size() - 1; ++j) { coord_str += std::to_string(coord[j]) + ", "; }
                    coord_str += std::to_string(coord[coord.size() - 1]) + ")";
                    out[Ns].push_back(coord_str);
                }
            }
            if constexpr (std::is_same_v<geo_t, internals::polygon_layer_descriptor>) {
                // print polygon incidence matrix
                out[Ns].push_back("<POLYGON>");
                int n_bits = 6;
                for (int i = 0; i < n_rows; ++i) {
                    auto region = std::get<Ns>(data.geometry(i));
                    std::string coord_str = "";
                    if (region.rows() < n_bits) {
                        for (int j = 0; j < n_bits - 1; j++) { coord_str += region[j] ? "1 " : "0 "; }
                        coord_str += region[n_bits - 1] ? "1" : "0";
                    } else {
                        for (int j = 0; j < n_bits / 2; j++) { coord_str += region[j] ? "1 " : "0 "; }
                        coord_str += ".... ";
                        for (int j = n_bits / 2; j < n_bits - 1; j++) {
                            coord_str += region[region.rows() - j - 1] ? "1 " : "0 ";
                        }
                        coord_str += region[region.rows() - n_bits - 2] ? "1" : "0";
                    }
                    coord_str += ")";
                    out[Ns].push_back(coord_str);
                }
            }
        });
	
        using internals::dtype;
        for (int i = Order, n = n_cols; i < n; ++i) {
            const auto& desc = data.header()[i - Order];
            dtype coltype = desc.type_id();
            if (coltype == dtype::flt64) { print.template operator()<double      >(out[i], "flt64", desc); }
            if (coltype == dtype::flt32) { print.template operator()<float       >(out[i], "flt32", desc); }
            if (coltype == dtype::int64) { print.template operator()<std::int64_t>(out[i], "int64", desc); }
            if (coltype == dtype::int32) { print.template operator()<std::int32_t>(out[i], "int32", desc); }
            if (coltype == dtype::bin)   { print.template operator()<bool        >(out[i], "bin"  , desc); }
            if (coltype == dtype::str)   { print.template operator()<std::string >(out[i], "str"  , desc); }
        }
        // pretty format
        for (int i = 0, n = n_cols; i < n; ++i) {
            for (int j = 0, m = n_rows + 2; j < m; ++j) { max_size[i] = std::max(max_size[i], out[i][j].size()); }
        }
	// pad with spaces
        for (int i = 0, n = n_cols; i < n; ++i) {
            for (int j = 0, m = out[i].size(); j < m; ++j) {
                out[i][j].insert(0, max_size[i] - out[i][j].size() + (i == 0 ? 0 : 1), ' ');   // first pad with spaces
		if (i < Order && j < 2) { out[i][j].insert(0, 1, ' '); }
                if (i < Order && j > 1) { out[i][j].insert(0 + (i > 0 ? 1 : 0), 1, '('); }
            }
	    if (i >= Order && data.header()[i - Order].size() > 1) {   // block columns formatting
                std::vector<std::size_t> posmin;
                std::size_t posmax = std::numeric_limits<std::size_t>::min();
                for (int j = 2, m = out[i].size(); j < m; ++j) {
                    posmax = std::max(posmax, out[i][j].size() - out[i][j].find(" ... ") - 5);
                    posmin.push_back(out[i][j].size() - out[i][j].find(" ... ") - 5);
                }
		// align " ... " pattern
                for (int j = 2, m = out[i].size(); j < m; ++j) {
                    out[i][j].insert(out[i][j].size() - posmin[j - 2], std::string(posmax - posmin[j - 2], ' '));
                    out[i][j].erase(0, posmax - posmin[j - 2]);   // remove eventual extra chars
                }
            }
        }
	// send to output stream
        for (int j = 0, m = out[0].size(); j < m - 1; ++j) {
            for (int i = 0, n = n_cols; i < n; ++i) { os << out[i][j]; }
            os << std::endl;
        }
        for (int i = 0, n = n_cols; i < n; ++i) { os << out[i][out[0].size() - 1]; }
        return os;
    }
   private:
    // internal reading utility
    template <typename T, typename ParsedFile>
    void load_table_(const std::vector<std::string>& colnames, const ParsedFile& parsed_file) {
        fdapde_assert(colnames.size() == parsed_file.cols());
	int i = 0, n_col = parsed_file.cols();
        std::vector<std::vector<T>> data(parsed_file.cols());
        for (const T& s : parsed_file.data()) {
            data[i].push_back(s);
            i = (i + 1) % n_col;
        }
        for (int i = 0; i < n_col; ++i) { data_.append_vec(colnames[i], data[i]); }
	fdapde_assert(n_rows_ == data_.rows());
	return;
    }
    template <typename Scalar>
    Eigen::Matrix<Scalar, Dynamic, Dynamic>
    parse_file_(const std::string& filename, bool header, bool index_col) const {
        std::filesystem::path filepath(filename);
        if (!std::filesystem::exists(filepath)) { throw std::runtime_error("file " + filename + " not found."); }
	// check if able to parse
        std::set<std::string> supported_extensions({".csv", ".txt"});
        if (!supported_extensions.contains(filepath.extension().string())) {
            throw std::runtime_error("not supported format.");
        }
        // read file
        Eigen::Matrix<Scalar, Dynamic, Dynamic> data;
        if (filepath.extension().string() == ".csv") data = read_csv<Scalar>(filename, header, index_col).as_matrix();
        if (filepath.extension().string() == ".txt") data = read_txt<Scalar>(filename, header, index_col).as_matrix();
        return data;
    }
    // load geometry
    void load_geometry_(const Eigen::Matrix<double, Dynamic, Dynamic>& coords) {
        fdapde_static_assert(
          Order == 1 || is_full_geo_point, THIS_METHOD_IS_FOR_ORDER_ONE_OR_FULL_POINT_GEOFRAMES_ONLY);
        constexpr int full_embed_dim = std::accumulate(embed_dim.begin(), embed_dim.end(), 0);
        fdapde_assert((n_rows_ == 0 || n_rows_ == coords.rows()) && coords.cols() == full_embed_dim);
        int offset = 0;
        internals::for_each_index_in_pack<Order>([&]<int Ns_>() {
            Eigen::Matrix<double, Dynamic, Dynamic> coords_ = coords.middleCols(offset, embed_dim[Ns_]);
            std::get<Ns_>(geo_data_) = std::tuple_element_t<Ns_, geo_storage_t>(std::get<Ns_>(triangulation_), coords_);
            offset += embed_dim[Ns_];
        });
        if (n_rows_ == 0) { n_rows_ = coords.rows(); }
        return;
    }
    void load_geometry_(const std::string& filename, bool header = true, bool index_col = true) {
        fdapde_static_assert(
          Order == 1 || is_full_geo_point, THIS_METHOD_IS_FOR_ORDER_ONE_OR_FULL_POINT_GEOFRAMES_ONLY);
        if constexpr (is_full_geo_point) {
            load_geometry_(parse_file_<double>(filename, header, index_col));
        } else {
            if constexpr (is_geo_v<0, POLYGON>) {
                using binary_t = BinaryMatrix<Dynamic, Dynamic>;
                load_geometry_(binary_t(parse_file_<int>(filename, header, index_col)));
            }
            if constexpr (is_geo_v<0, POINT>) { load_geometry_(parse_file_<double>(filename, header, index_col)); }
        }
        return;
    }
    void load_geometry_(int flag) {
        internals::for_each_index_in_pack<Order>([&]<int Ns>() { load_geometry_index_<Ns>(flag); });
    }
    template <typename GeoDescriptor>
        requires(
          internals::is_vector_like_v<GeoDescriptor> &&
          std::is_convertible_v<internals::subscript_result_of_t<GeoDescriptor, int>, int>)
    void load_geometry_(const GeoDescriptor& regions) {
        load_geometry_index_<0>(regions);
    }
    // single geometric indexes reading utilities
    template <int N> void load_geometry_index_(const Eigen::Matrix<double, Dynamic, Dynamic>& coords) {
        fdapde_assert(coords.cols() == embed_dim[N]);
        std::get<N>(geo_data_) = std::tuple_element_t<N, geo_storage_t>(std::get<N>(triangulation_), coords);
        if (n_rows_ == 0) { n_rows_ = coords.rows(); }
        return;
    }
    template <int N> void load_geometry_index_(int flag) {
        fdapde_static_assert(
          std::is_same_v<POINT FDAPDE_COMMA std::tuple_element_t<N FDAPDE_COMMA GeoInfo>>,
          THIS_METHOD_IS_FOR_POINT_INDEXES_ONLY);
        fdapde_assert(flag == MESH_NODES);
        std::get<N>(geo_data_) = std::tuple_element_t<N, geo_storage_t>(std::get<N>(triangulation_));
        if (n_rows_ == 0) { n_rows_ = std::get<N>(triangulation_)->nodes().rows(); }
        return;
    }
    template <int N>
    void load_geometry_index_(const std::string& filename, bool header = true, bool index_col = true) {
        if constexpr (is_geo_v<N, POINT>) {
            load_geometry_index_<N>(parse_file_<double>(filename, header, index_col));
        } else {
            using binary_t = BinaryMatrix<Dynamic, Dynamic>;
            load_geometry_index_<N>(binary_t(parse_file_<double>(filename, header, index_col)));
        }
        return;
    }
    template <int N, typename GeoDescriptor>
        requires(
          std::is_same_v<GeoDescriptor, BinaryMatrix<Dynamic, Dynamic>> ||
          (internals::is_vector_like_v<GeoDescriptor> &&
           std::is_convertible_v<internals::subscript_result_of_t<GeoDescriptor, int>, int>))
    void load_geometry_index_(const GeoDescriptor& regions) {
        fdapde_static_assert(
          std::is_same_v<POLYGON FDAPDE_COMMA std::tuple_element_t<N FDAPDE_COMMA GeoInfo>>,
          THIS_METHOD_IS_FOR_POLYGON_INDEXES_ONLY);
        std::get<N>(geo_data_) = std::tuple_element_t<N, geo_storage_t>(std::get<N>(triangulation_), regions);
        if (n_rows_ == 0) { n_rows_ = std::get<N>(geo_data_).rows(); }
    }

    storage_t data_;
    geo_storage_t geo_data_;
    triangulation_t triangulation_;
    int n_rows_;
    std::array<int, Order> strides_;   // for unstructured layers: array of 1s
    std::array<int, Order> extents_;
    bool structured_;
    std::string crs_ = "UNDEFINED";   // geodedics CRS
};

// geo_filter outstream
template <typename GeoLayer>
std::ostream& operator<<(std::ostream& os, const internals::random_access_geo_row_view<GeoLayer>& data) {
    return operator<<(os, GeoLayer(data));
}
template <typename GeoLayer>
std::ostream& operator<<(std::ostream& os, const internals::geo_row_view<GeoLayer>& data) {
    return operator<<(os, data.geo_data().select({data.id()}));
}

}   // namespace fdapde

#endif // __FDAPDE_GEO_LAYER_H__
