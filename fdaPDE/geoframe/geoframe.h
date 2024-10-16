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

#ifndef __GEOFRAME_H__
#define __GEOFRAME_H__

#include "../linear_algebra/mdarray.h"
#include "../utils/symbols.h"
#include "../utils/traits.h"

#include <memory>
#include <filesystem>

namespace fdapde {

// generates a sequence of strings as {"base_0", "base_1", \ldots, "base_{n-1}"}
std::vector<std::string> seq(const std::string& base, int n) {
    std::vector<std::string> vec;
    vec.reserve(n);
    for(int i = 0; i < n; ++i) { vec.emplace_back(base + std::to_string(i)); }
    return vec;
}

// generates a sequence of arithmetic values as {base + 0, base + 1, \ldots, base + n-1}
template <typename T>
    requires(std::is_arithmetic_v<T>)
  std::vector<int> seq(T begin, int n, int by = 1) {
    std::vector<T> vec;
    vec.reserve(n);
    for (int i = 0; i < n; i += by) { vec.emplace_back(begin + i); }
    return vec;
}

template <typename T> struct is_shared_ptr : std::false_type { };
template <typename T> struct is_shared_ptr<std::shared_ptr<T>> : std::true_type { };
template <typename T> constexpr bool is_shared_ptr_v = is_shared_ptr<T>::value;

template <typename T>
concept is_random_access_iterable = requires(T t) {
    typename T::iterator;
    { t.begin() } -> std::same_as<typename T::iterator>;
    {  t.end()  } -> std::same_as<typename T::iterator>;
} &&  std::random_access_iterator<typename T::iterator>;
  
namespace internals {

template <template <typename...> typename T, typename U> struct strip_tuple_into;
template <template <typename...> typename T, typename... Us>
struct strip_tuple_into<T, std::tuple<Us...>> : std::type_identity<T<Us...>> { };
template <template <typename...> typename T, typename... Us>
using strip_tuple_into_t = typename strip_tuple_into<T, std::tuple<Us...>>::type;

void throw_geoframe_error(const std::string& msg) { throw std::runtime_error("GeoFrame: " + msg); }

#define geoframe_assert(condition, msg)                                                                                \
    if (!(condition)) { internals::throw_geoframe_error(msg); }

// contiguous block selection

  
template <typename DataLayer> struct geoframe_row {
    using Scalar = typename DataLayer::Scalar;
    using index_t = typename DataLayer::index_t;
    using size_t = typename DataLayer::size_t;
    using storage_t = std::conditional_t<
      std::is_const_v<DataLayer>, std::add_const_t<typename DataLayer::storage_t>, typename DataLayer::storage_t>;
    static constexpr int Order = DataLayer::Order;

    geoframe_row() noexcept = default;
    geoframe_row(DataLayer* layer, index_t i) noexcept :
        layer_(layer), i_(i), slice_(internals::apply_index_pack<Order - 1>([&]<int... Ns_> {
            return layer->data().template slice<Ns_...>(((void)Ns_, i % layer->data().mapping().stride(Ns_))...);
        })) {
        fdapde_assert(i < layer_->rows());
    }
    // observers
    size_t rows() const { return 1; }
    size_t cols() const { return layer_->cols(); }
    size_t size() const { return layer_->cols(); }
    index_t id() const { return i_; }
    // accessors
    Scalar& operator()(index_t i) { return slice_(i); }
    const Scalar& operator()(index_t i) const { return slice_(i); }
    Scalar& operator()(const std::string& colname) { return slice_(layer_->get_col_idx_(colname)); }
    const Scalar& operator()(const std::string& colname) const { return slice_(layer_->get_col_idx_(colname)); }
   private:
    std::decay_t<decltype(internals::apply_index_pack<Order - 1>(
      []<int... Ns_> { return MdArraySlice<storage_t, ((void)Ns_, Ns_)...> {}; }))>
      slice_;
    DataLayer* layer_;
    index_t i_;
};

// an indexed set of rows
template <typename DataLayer> struct geoframe_row_random_access {
    using Scalar = typename DataLayer::Scalar;
    using index_t = typename DataLayer::index_t;
    using size_t = typename DataLayer::size_t;
    using storage_t = std::conditional_t<
      std::is_const_v<DataLayer>, std::add_const<typename DataLayer::storage_t>, typename DataLayer::storage_t>;
    static constexpr int Order = DataLayer::Order;

    geoframe_row_random_access() noexcept = default;
    template <typename Iterator>
        requires(std::is_convertible_v<typename Iterator::value_type, index_t>)
    geoframe_row_random_access(DataLayer* layer, Iterator begin, Iterator end) : layer_(layer), idxs_(begin, end) { }
    template <typename Filter>
        requires(requires(Filter f, index_t i) {
            { f(i) } -> std::same_as<bool>;
        })
    geoframe_row_random_access(DataLayer* layer, Filter&& f) : layer_(layer) {
        for (size_t i = 0, n = layer_->rows(); i < n; ++i) {
            if (f(i)) idxs_.push_back(i);
        }
    }
    // observers
    size_t rows() const { return idxs_.size(); }
    size_t cols() const { return layer_->cols(); }
    size_t size() const { return idxs_.size() * layer_->cols(); }
    // accessors
    geoframe_row<DataLayer> operator()(index_t i) {
        fdapde_assert(i < idxs_.size());
        return layer_->row(idxs_[i]);
    }
    geoframe_row<const DataLayer> operator()(index_t i) const {
        fdapde_assert(i < idxs_.size());
        return layer_->row(idxs_[i]);
    }
    // iterator support
    class iterator {
       public:
        using value_type = geoframe_row<DataLayer>;
        using pointer_t = std::add_pointer_t<value_type>;
        using reference = std::add_lvalue_reference_t<value_type>;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;
        using iterator_category = std::forward_iterator_tag;

        iterator() = default;
        iterator(int index, geoframe_row_random_access* accessor) : index_(index), accessor_(accessor), val_() {
            if (index_ != accessor_->rows()) { val_ = accessor_->operator()(index); }
            index_++;
        }
        reference operator*() { return val_; }
        const reference operator*() const { return val_; }
        pointer_t operator->() { return std::addressof(val_); }
        const pointer_t operator->() const { return std::addressof(val_); }
        iterator& operator++() {
            if (index_ != accessor_->rows()) { [[likely]] val_ = accessor_->operator()(index_); }
            index_++;
            return *this;
        }
        friend bool operator==(const iterator& lhs, const iterator& rhs) { return lhs.index_ == rhs.index_; }
        friend bool operator!=(const iterator& lhs, const iterator& rhs) { return lhs.index_ != rhs.index_; }
       private:
        index_t index_;
        geoframe_row_random_access* accessor_;
        value_type val_;
    };
    iterator begin() { return iterator(0, this); }
    iterator end() { return iterator(idxs_.size(), this); }
   private:
    DataLayer* layer_;
    std::vector<index_t> idxs_;
};

template <typename GeoFrame_> class data_layer {  
   public:
    using Scalar = typename GeoFrame_::Scalar;
    using storage_t = typename GeoFrame_::storage_t;
    using index_t = typename storage_t::index_t;
    using size_t = typename storage_t::size_t;
    static constexpr int Order = storage_t::Order;

    data_layer() noexcept : data_(std::make_shared<data_t>()) { }
    template <typename... Extents_>
        requires(std::is_convertible_v<Extents_, index_t> && ...) && (sizeof...(Extents_) == Order)
    data_layer(Extents_... exts) noexcept : data_(std::make_shared<data_t>()) {
        std::array<index_t, Order> exts_ {static_cast<index_t>(exts)...};
	std::size_t rows_ = 1;
        for (size_t i = 0; i < Order - 1; ++i) { rows_ *= exts_[i]; }
	data_->rows_ = rows_;
        data_->cols_ = exts_[Order - 1];
        data_->storage_.resize(static_cast<index_t>(exts)...);
    }
    data_layer(const data_layer&) noexcept = default;
    data_layer(data_layer&&) noexcept = default;
  
    // observers
    std::vector<std::string> colnames() const {
        std::vector<std::string> keys;
	std::size_t s = data_->colnames_.size();
        if (s != 0) {
            keys.reserve(s);
            for (const auto& [k, v] : data_->colnames_) { keys.emplace_back(k); }
        }
        return keys;
    }
    size_t rows() const { return data_->rows_; }
    size_t cols() const { return data_->cols_; }
    // modifiers
    void set_colnames(const std::vector<std::string>& colnames) {
        data_->colnames_.clear();
        std::vector<std::string> tmp = colnames;
        std::sort(tmp.begin(), tmp.end());
        if (
          colnames.size() != data_->cols_ || std::unique(tmp.begin(), tmp.end()) != tmp.end() ||
          !std::accumulate(
            colnames.begin(), colnames.end(), true, [](bool v, const auto& name) { return (v & !name.empty()); })) {
            internals::throw_geoframe_error("not unique or empty column names.");
        }
        for (size_t i = 0; i < colnames.size(); ++i) { data_->colnames_[colnames[i]] = i; }
        return;
     }
     // accessors
     template <typename... Idxs>
         requires(std::is_convertible_v<Idxs, index_t> && ...) && (sizeof...(Idxs) == Order)
     Scalar& operator()(Idxs... idxs) {
         return data().operator()(static_cast<index_t>(idxs)...);
     }
     template <typename... Idxs>
         requires(std::is_convertible_v<Idxs, index_t> && ...) && (sizeof...(Idxs) == Order)
     const Scalar& operator()(Idxs... idxs) const {
         return data().operator()(static_cast<index_t>(idxs)...);
     }
     geoframe_row<data_layer<GeoFrame_>> row(size_t row) { return geoframe_row<data_layer<GeoFrame_>>(this, row); }
     geoframe_row<const data_layer<GeoFrame_>> row(size_t row) const {
         return geoframe_row<const data_layer<GeoFrame_>>(this, row);
     }
     auto col(size_t col) const { return data().template slice<Order - 1>(col); }
     auto col(const std::string& colname) const { return col(get_col_idx_(colname)); }
     auto col(size_t col) { return data().template slice<Order - 1>(col); }
     auto col(const std::string& colname) { return col(get_col_idx_(colname)); }
     // row access by subscript
     geoframe_row<data_layer<GeoFrame_>> operator()(size_t i) { return row(i); }
     geoframe_row<const data_layer<GeoFrame_>> operator()(size_t i) const { return row(i); }
     template <typename IndexList>
         requires(requires(IndexList l) {
             typename IndexList::iterator;
             { l.begin() } -> std::same_as<typename IndexList::iterator>;
             { l.end() } -> std::same_as<typename IndexList::iterator>;
         })
     geoframe_row_random_access<data_layer<GeoFrame_>> operator()(IndexList&& idxs) {
         return geoframe_row_random_access<data_layer<GeoFrame_>>(this, idxs.begin(), idxs.end());
     }
     template <typename T>
         requires(std::is_convertible_v<T, index_t>)
     geoframe_row_random_access<data_layer<GeoFrame_>> operator()(const std::initializer_list<T>& idxs) {
         return geoframe_row_random_access<data_layer<GeoFrame_>>(this, idxs.begin(), idxs.end());
     }
     template <typename Filter>
         requires(requires(Filter f, index_t i) {
             { f(i) } -> std::same_as<bool>;
         })
     geoframe_row_random_access<data_layer<GeoFrame_>> operator()(Filter&& f) {
         return geoframe_row_random_access<data_layer<GeoFrame_>>(this, std::forward<Filter>(f));
     }

     storage_t& data() { return data_->storage_; }
     const storage_t& data() const { return data_->storage_; }
    private:
     index_t get_col_idx_(const std::string& colname) const {
         geoframe_assert(
           !colname.empty() && data_->colnames_.contains(colname), std::string("column " + colname + " not found."));
         return data_->colnames_.at(colname);
     }
     // guarantee shallow-copy semantic
     struct data_t {
         std::unordered_map<std::string, index_t> colnames_;
         size_t rows_, cols_;
         storage_t storage_;
         data_t() : colnames_(), rows_(0), cols_(0), storage_() {};
     };
     std::shared_ptr<data_t> data_;
};

// geometric specific layers
template <typename GeoFrame_> struct multipoint_layer : public data_layer<GeoFrame_> {
     using Base = data_layer<GeoFrame_>;
     using Scalar  = typename Base::Scalar;
     using index_t = typename Base::index_t;
     using size_t  = typename Base::size_t;
     static constexpr int Order = Base::Order;

     multipoint_layer() noexcept : Base() { }
     template <typename... Extents_>
         requires(std::is_convertible_v<Extents_, index_t> && ...) && (sizeof...(Extents_) == Order)
     multipoint_layer(GeoFrame_* geoframe, Extents_... exts) noexcept : Base(exts...), geoframe_(geoframe), coords_() { }
     template <typename... Extents_, typename CoordsType>
         requires(std::is_convertible_v<Extents_, index_t> && ...) &&
                   (sizeof...(Extents_) == Order && std::is_convertible_v<CoordsType, DMatrix<Scalar>>)
     multipoint_layer(GeoFrame_* geoframe, const std::shared_ptr<CoordsType>& coords, Extents_... exts) noexcept :
         Base(exts...), geoframe_(geoframe), coords_(coords) { }

     // observers
     const DMatrix<Scalar>& coordinates() const { return coords_ ? *coords_ : triangulation().nodes(); }
    private:
     auto& triangulation() { return geoframe_->triangulation(); }
     const auto& triangulation() const { return geoframe_->triangulation(); }
  
     GeoFrame_* geoframe_;
     std::shared_ptr<DMatrix<Scalar>> coords_;
};

template <typename GeoFrame_> struct areal_layer : public data_layer<GeoFrame_> {
     using Base = data_layer<GeoFrame_>;
     using Scalar  = typename Base::Scalar;
     using index_t = typename Base::index_t;
     using size_t  = typename Base::size_t;
     static constexpr int Order = Base::Order;

    areal_layer() : Base() { }
    template <typename... Extents_>
        requires(std::is_convertible_v<Extents_, index_t> && ...) && (sizeof...(Extents_) == Order)
    areal_layer(Extents_... exts) noexcept : Base(exts...) { }
    private:
     DVector<int> regions_;
};

}   // namespace internals

namespace layer_t {
  
struct multipoint_t { } multipoint;
struct areal_t { } areal;

}   // namespace layer_t

template <typename Triangulation_, typename Scalar_ = double, int Order_ = 2> struct GeoFrame {
    fdapde_static_assert(Order_ > 1, GEOFRAME_MUST_HAVE_ORDER_TWO_OR_HIGHER);
   private:
    using This = GeoFrame<Triangulation_, Scalar_, Order_>;
    using layers_t = std::tuple<internals::multipoint_layer<This>, internals::areal_layer<This>>;
    template <typename... Ts> using LayerMap_ = std::tuple<std::unordered_map<std::string, Ts>...>;
    template <typename T, typename U> auto& fetch_(U& u) { return std::get<index_of_type<T, layers_t>::index>(u); }
    template <typename T, typename U> const auto& fetch_(const U& u) const {
        return std::get<index_of_type<T, layers_t>::index>(u);
    }
   public:
    static constexpr int local_dim = Triangulation_::LocalDim;
    static constexpr int embed_dim = Triangulation_::EmbedDim;
    static constexpr int Order = Order_;
    using Scalar = Scalar_;
    using LayerMap  = typename internals::strip_tuple_into<LayerMap_, layers_t>::type;
    using LookupMap = std::unordered_map<std::string, internals::data_layer<This>*>;
    using storage_t = MdArray<Scalar, full_dynamic_extent_t<Order>>;
    using index_t = typename storage_t::index_t;
    using size_t  = typename storage_t::size_t;

    // constructors
    GeoFrame() noexcept : triangulation_(nullptr), plain_lookup_(), layers_() { }
    explicit GeoFrame(Triangulation_& triangulation) noexcept :
        triangulation_(std::addressof(triangulation)), plain_lookup_(), layers_() { }
    GeoFrame(const GeoFrame&) noexcept = default;
    GeoFrame(GeoFrame&&) noexcept = default;

    // modifiers
    // multipoint layer with geometrical locations at mesh nodes
    void push(const std::string& name, layer_t::multipoint_t, size_t n) {
        geoframe_assert(!name.empty() && !plain_lookup_.contains(name), "empty or duplicated name.");
        using layer_t = internals::multipoint_layer<This>;
        fetch_<layer_t>(layers_).insert({name, layer_t(this, triangulation_->n_nodes(), n)});
        plain_lookup_.insert(
          {name, reinterpret_cast<internals::data_layer<This>*>(std::addressof(fetch_<layer_t>(layers_).at(name)))});
        return;
    }
    // multipoint layer with specified locations
    template <typename ColnamesContainer>
        requires(
          is_random_access_iterable<ColnamesContainer> &&
          std::is_convertible_v<typename ColnamesContainer::iterator::value_type, std::string>)
    void push(const ColnamesContainer& colnames, layer_t::multipoint_t, size_t n) {
        for (const auto& name : colnames) { push(name, layer_t::multipoint, n); }
    }
    template <typename T>
        requires(std::is_convertible_v<T, std::string>)
    void push(const std::initializer_list<T>& colnames, layer_t::multipoint_t, size_t n) {
        for (auto it = colnames.begin(); it != colnames.end(); ++it) { push(*it, layer_t::multipoint, n); }
    }
    template <typename CoordsType>
        requires(
          (fdapde::is_eigen_dense_v<CoordsType> && (CoordsType::Cols == Dynamic || CoordsType::Cols == embed_dim) &&
           std::is_floating_point_v<typename CoordsType::Scalar>) ||
          (requires(CoordsType c) {
              typename CoordsType::value_type;
              { c.size() } -> std::same_as<std::size_t>;
              { c.data() } -> std::same_as<typename CoordsType::value_type*>;
          } && std::contiguous_iterator<typename CoordsType::iterator> &&
           std::is_floating_point_v<typename CoordsType::value_type>) || fdapde::is_shared_ptr_v<CoordsType>)
    void push(const std::string& name, layer_t::multipoint_t, const CoordsType& coords, size_t n) {
        geoframe_assert(!name.empty() && !plain_lookup_.contains(name), "empty or duplicated name.");
	using layer_t = internals::multipoint_layer<This>;
        if constexpr (fdapde::is_eigen_dense_v<CoordsType>) {
            geoframe_assert(
              coords.cols() == embed_dim && coords.rows() > 0, "empty or wrongly sized coordinate matrix.");
            using Scalar__ = typename CoordsType::Scalar;
            std::shared_ptr<DMatrix<Scalar__>> coords_ptr = std::make_shared<DMatrix<Scalar__>>(coords);
            fetch_<layer_t>(layers_).insert({name, layer_t(this, coords_ptr, coords.rows(), n)});
        } else if constexpr (fdapde::is_shared_ptr_v<CoordsType>) {
	  // allows multiple layers to share the same locations
	  fetch_<layer_t>(layers_).insert({name, layer_t(this, coords, coords->rows(), n)});
        } else {
            geoframe_assert(
              coords.size() > 0 && coords.size() % embed_dim == 0, "empty or wrongly sized coordinate matrix.");
            using Scalar__ = typename CoordsType::value_type;
            int n_rows = coords.size() / embed_dim;
            int n_cols = embed_dim;
            std::shared_ptr<DMatrix<Scalar__>> coords_ptr =
              std::make_shared<DMatrix<Scalar__>>(Eigen::Map<DMatrix<Scalar__>>(coords.data(), n_rows, n_cols));
	    fetch_<layer_t>(layers_).insert({name, layer_t(this, coords_ptr, n_rows, n)});
        }
        plain_lookup_.insert(
          {name, reinterpret_cast<internals::data_layer<This>*>(std::addressof(fetch_<layer_t>(layers_).at(name)))});
        return;
    }
    // packed multipoint layers sharing the same locations
    template <typename T, typename CoordsType>
        requires(
          (fdapde::is_eigen_dense_v<CoordsType> ||
           requires(CoordsType c) {
               typename CoordsType::value_type;
               { c.size() } -> std::same_as<std::size_t>;
               { c.data() } -> std::same_as<typename CoordsType::value_type*>;
           }) &&
          std::is_convertible_v<T, std::string>)
    void push(const std::initializer_list<T>& colnames, layer_t::multipoint_t, const CoordsType& coords, size_t n) {
        // layers share the same locations, allocate here once
        using Scalar__ = decltype([]() {
            if constexpr (fdapde::is_eigen_dense_v<CoordsType>) return typename CoordsType::Scalar();
            else return typename CoordsType::value_type();
        }());
        std::shared_ptr<DMatrix<Scalar__>> coords_ptr;
        if constexpr (fdapde::is_eigen_dense_v<CoordsType>) {
            coords_ptr = std::make_shared<DMatrix<Scalar__>>(coords);
        } else {
            int n_rows = coords.size() / embed_dim;
            int n_cols = embed_dim;
            coords_ptr =
              std::make_shared<DMatrix<Scalar__>>(Eigen::Map<DMatrix<Scalar__>>(coords.data(), n_rows, n_cols));
        }
        for (auto it = colnames.begin(); it != colnames.end(); ++it) { push(*it, layer_t::multipoint, coords_ptr, n); }
    }
    void erase(const std::string& layer_name) {
        // search for layer_name (if no layer found does nothing)
        if (!plain_lookup_.contains(layer_name)) return;
        plain_lookup_.erase(layer_name);
        std::apply(
          [&](auto&&... layer) {
              (std::erase_if(
                 layer,
                 [&](const auto& item) {
                     auto const& [k, v] = item;
                     return k == layer_name;
                 }),
               ...);
          },
          layers_);
        return;
    }
    // file import
    void load_csv(const std::string& name, const std::string& file_name, layer_t::multipoint_t) {
        geoframe_assert(std::filesystem::exists(file_name), "file " + file_name + " not found.");
        auto csv = fdapde::read_csv<Scalar>(file_name);
        geoframe_assert(csv.rows() == triangulation_->n_nodes(), "wrong csv size.");
        push(name, layer_t::multipoint, csv.cols());
        // move data in memory buffer
        get_as(layer_t::multipoint, name).data().assign_inplace_from(csv.data());
	get_as(layer_t::multipoint, name).set_colnames(csv.colnames());
        return;
    }
    template <typename CoordsType>
        requires(
          (fdapde::is_eigen_dense_v<CoordsType> ||
           requires(CoordsType c) {
               typename CoordsType::value_type;
               { c.size() } -> std::same_as<std::size_t>;
               { c.data() } -> std::same_as<typename CoordsType::value_type*>;
           }))
    void
    load_csv(const std::string& name, const std::string& file_name, layer_t::multipoint_t, const CoordsType& coords) {
        geoframe_assert(std::filesystem::exists(file_name), "file " + file_name + " not found.");
        auto csv = fdapde::read_csv<Scalar>(file_name);
        if constexpr (fdapde::is_eigen_dense_v<CoordsType>) {
            geoframe_assert(csv.rows() == coords.rows(), "wrong csv size.");
        } else {
            geoframe_assert(csv.rows() == coords.size() / embed_dim, "wrong csv size.");
        }
        push(name, layer_t::multipoint, coords, csv.cols());
        // move data in memory buffer
        get_as(layer_t::multipoint, name).data().assign_inplace_from(csv.data());
	get_as(layer_t::multipoint, name).set_colnames(csv.colnames());
        return;
    }
    // iterators
    auto begin(layer_t::multipoint_t) { return fetch_<layer_type_from_tag<layer_t::multipoint_t>>.begin(); }
    auto end  (layer_t::multipoint_t) { return fetch_<layer_type_from_tag<layer_t::multipoint_t>>.end(); }
    // layer access
    internals::data_layer<This>& operator[](const std::string& name) {
        geoframe_assert(plain_lookup_.contains(name), std::string("key " + name + " not found."));
        return *plain_lookup_.at(name);
    }
    const internals::data_layer<This>& operator[](const std::string& name) const {
        geoframe_assert(plain_lookup_.contains(name), std::string("key " + name + " not found."));
        return *plain_lookup_.at(name);
    }
    template <typename Tag> auto& get_as(Tag, const std::string& name) {
        return get_as_<layer_type_from_tag<Tag>>(name);
    }
    template <typename Tag> const auto& get_as(Tag, const std::string& name) const {
        return get_as_<layer_type_from_tag<Tag>>(name);
    }
    bool has_layer(const std::string& name) const { return plain_lookup_.contains(name); }
    // geometry access
    Triangulation_& triangulation() { return *triangulation_; }
    const Triangulation_& triangulation() const { return *triangulation_; }
   private:
    // internal utilities
    template <typename LayerType> decltype(auto) get_as_(const std::string& name) {
        geoframe_assert(fetch_<LayerType>(layers_).contains(name), std::string("key " + name + " not found."));
        return fetch_<LayerType>(layers_).at(name);
    }
    template <typename Tag> class layer_type_from_tag_impl {
        static auto layer_type_from_tag_(Tag t) {
            if constexpr (std::is_same_v<Tag, layer_t::multipoint_t>) return internals::multipoint_layer<This> {};
        }
       public:
        using type = decltype(layer_type_from_tag_(std::declval<Tag>()));
    };
    template <typename Tag> using layer_type_from_tag = layer_type_from_tag_impl<Tag>::type;
    // data members
    Triangulation_* triangulation_ = nullptr;
    LayerMap layers_ {};
    LookupMap plain_lookup_ {};
};

}   // namespace fdapde

#endif // __GEOFRAME_H__
