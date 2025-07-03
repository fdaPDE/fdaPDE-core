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

#ifndef __FDAPDE_DATA_LAYER_H__
#define __FDAPDE_DATA_LAYER_H__

#include "header_check.h"

namespace fdapde {
namespace internals {

// computes boolean nan pattern from a contiguous block of data
template <typename MdArray_> auto compute_na_mask(MdArray_&& data) {
    using MdArrayClean = std::decay_t<MdArray_>;
    using logical_t = MdArray<bool, typename MdArrayClean::extents_t, typename MdArrayClean::layout_t>;
    using Scalar = typename MdArrayClean::Scalar;

    logical_t na_mask(data.extents());
    if constexpr (std::is_same_v<Scalar, double> || std::is_same_v<Scalar, float>) {
        for (auto it = data.begin(); it != data.end(); ++it) {
            if (std::isnan(*it)) { na_mask(it.index()).set(); }
        }
    } else if constexpr (std::is_same_v<Scalar, std::string>) {
        for (auto it = data.begin(); it != data.end(); ++it) {
            const std::string& str = *it;
            if (str == "NaN" || str == "nan" || str == "NA") { na_mask(it.index()).set(); }
        }
    } else {
        // for other integer types, there is no nan encoding, do nothing.
    }
    return na_mask;
}

// a column is a named single-typed block of **contiguous** data from a plain_data_layer
template <typename Scalar_, typename DataObj> struct plain_col_view {
    using Scalar = Scalar_;
    using index_t = typename DataObj::index_t;
    using size_t = typename DataObj::size_t;
    static constexpr int Order = DataObj::Order;
   private:
    using data_table = MdArray<Scalar_, full_dynamic_extent_t<Order>, internals::layout_left>;
    using extents_t = typename data_table::extents_t;
   public:
    using storage_t = std::conditional_t<
      std::is_const_v<DataObj>, MdArrayBlock<std::add_const_t<data_table>, extents_t>,
      MdArrayBlock<data_table, extents_t>>;
    using reference = typename DataObj::template reference<Scalar>;
    using const_reference = typename DataObj::template const_reference<Scalar>;
    using logical_t = MdArray<bool, extents_t, internals::layout_left>;

    plain_col_view() noexcept = default;
    template <typename FieldDescriptor>
    plain_col_view(DataObj& data, index_t row_begin, index_t row_end, const FieldDescriptor& desc) noexcept :
        data_(std::addressof(data)),
        block_(internals::apply_index_pack<Order>([&]<int... Ns_>() {
            return data.template data<Scalar>().block(
              ((void)Ns_, Ns_ == 0 ?
                            std::pair {row_begin, row_end - 1} :
                            (Ns_ == 1 ? std::pair {desc.offset(), desc.offset() + desc.size() - 1} :
                                        std::pair {0, index_t(data.template data<Scalar>().extent(Ns_)) - 1}))...);
        })),
        // metadata
        rows_(row_end - row_begin),
        blk_sz_(desc.size()),
        row_begin_(row_begin),
        row_end_(row_end),
        colname_(desc.colname()),
        type_id_(desc.type_id()) { }
  
    template <typename FieldDescriptor>   // column row constructor
    plain_col_view(DataObj& data, index_t row, const FieldDescriptor& desc) noexcept :
        plain_col_view(data, row, row + 1, desc) { }
    // full column constructor
    template <typename FieldDescriptor>   // column row constructor
    plain_col_view(DataObj& data, const FieldDescriptor& desc) noexcept :
        plain_col_view(data, 0, data.rows(), desc) { }

    // observers
    size_t rows() const { return rows_; }
    size_t cols() const { return 1; }
    size_t size() const { return rows_ * blk_sz_; }
    size_t blk_sz() const { return blk_sz_; }
    const storage_t& data() const { return block_; }
    storage_t& data() { return block_; }
    const std::string& colname() const { return colname_; }
    dtype type_id() const { return type_id_; }
    logical_t nan() const { return compute_na_mask(block_); }
    bool has_nan() const {
        for (auto it = block_.begin(); it != block_.end(); ++it) {
            if constexpr (std::is_same_v<Scalar, double> || std::is_same_v<Scalar, float>) {
                if (std::isnan(*it)) { return true; }
            }
            if constexpr (std::is_same_v<Scalar, std::string>) {
                const std::string& str = *it;
                if (str == "NaN" || str == "nan" || str == "NA") { return true; }
            }
        }
        return false;
    }
    template <typename Dst>   // direct copy column content in suscriptable destination
        requires(is_subscriptable<Dst, int>)
    void assign_to(Dst&& dst) const {
        block_.assign_to(dst);
    }
    // accessors
    template <typename... Idxs>
        requires(std::is_convertible_v<Idxs, index_t> && ...) && (sizeof...(Idxs) == Order && !std::is_const_v<DataObj>)
    constexpr reference operator()(Idxs&&... idxs) {
        return block_(static_cast<index_t>(idxs)...);
    }
    template <typename IndexPack>   // access via index-pack object
        requires(internals::is_subscriptable<IndexPack, index_t>)
    constexpr reference operator()(IndexPack&& index_pack) {
        return block_(index_pack);
    }
    template <typename... Idxs>
        requires(std::is_convertible_v<Idxs, index_t> && ...) && (sizeof...(Idxs) == Order)
    constexpr const_reference operator()(Idxs&&... idxs) const {
        return block_(static_cast<index_t>(idxs)...);
    }
    template <typename IndexPack>   // access via index-pack object
        requires(internals::is_subscriptable<IndexPack, index_t>)
    constexpr const_reference operator()(IndexPack&& index_pack) const {
        return block_(index_pack);
    }
    // iterators
    auto begin() const { return block_.begin(); }
    auto end() const { return block_.end(); }
    // logical comparison
    logical_t operator==(const Scalar& rhs) const { return logical_apply_(rhs, std::equal_to<Scalar>      {}); }
    logical_t operator!=(const Scalar& rhs) const { return logical_apply_(rhs, std::not_equal_to<Scalar>  {}); }
    logical_t operator< (const Scalar& rhs) const { return logical_apply_(rhs, std::less<Scalar>          {}); }
    logical_t operator> (const Scalar& rhs) const { return logical_apply_(rhs, std::greater<Scalar>       {}); }
    logical_t operator<=(const Scalar& rhs) const { return logical_apply_(rhs, std::less_equal<Scalar>    {}); }
    logical_t operator>=(const Scalar& rhs) const { return logical_apply_(rhs, std::greater_equal<Scalar> {}); }
#ifdef __FDAPDE_HAS_EIGEN__
    Eigen::Map<Eigen::Matrix<Scalar, Dynamic, Dynamic, Eigen::ColMajor>> as_matrix()
        requires(!std::is_const_v<DataObj>) {
        fdapde_static_assert(Order == 2, THIS_METHOD_IS_FOR_ORDER_TWO_MDARRAYS_ONLY);
        return Eigen::Map<Eigen::Matrix<Scalar, Dynamic, Dynamic, Eigen::ColMajor>>(block_.data(), rows(), blk_sz_);
    }
    Eigen::Map<const Eigen::Matrix<Scalar, Dynamic, Dynamic, Eigen::ColMajor>> as_matrix() const {
        fdapde_static_assert(Order == 2, THIS_METHOD_IS_FOR_ORDER_TWO_MDARRAYS_ONLY);
        return Eigen::Map<const Eigen::Matrix<Scalar, Dynamic, Dynamic, Eigen::ColMajor>>(
          block_.data(), rows(), blk_sz_);
    }
#endif
    // assignment
    template <typename Src>
        requires(is_vector_like_v<Src> || is_indexable_v<Src, Order, index_t>)
    plain_col_view& operator=(Src&& src) {
        // inplace assignment
        if constexpr (is_vector_like_v<Src> && !is_indexable_v<Src, Order, index_t>) {
            fdapde_assert(src.size() == rows_);
            if (blk_sz_ == 1) {   // vector - vector assign
                block_.assign_inplace_from(src);
                return *this;
            }
        } else {
            fdapde_assert(src.rows() == rows_);
            if (blk_sz_ == src.cols()) {   // block - block asssign
                block_.assign_inplace_from(src);
                return *this;
            }
        }
        // reallocation required
	index_t col_id = data_->col_id(colname_);
	data_->erase (colname_);
        data_->insert(colname_, col_id, src);
        // update block-view
        const auto& desc = data_->field_descriptor(colname_);
        blk_sz_ = desc.size();
        block_ = internals::apply_index_pack<Order>([&]<int... Ns_>() {
            return data_->template data<Scalar>().block(
              ((void)Ns_, Ns_ == 0 ?
                            std::pair {row_begin_, row_end_ - 1} :
                            (Ns_ == 1 ? std::pair {desc.offset(), desc.offset() + desc.size() - 1} :
                                        std::pair {0, index_t(data_->template data<Scalar>().extent(Ns_)) - 1}))...);
        });
        return *this;
    }
   protected:
    template <typename Functor> std::vector<bool> logical_apply_(const Scalar& rhs, Functor&& f) const {
        logical_t mask(block_.extents());
        for (auto it = block_.begin(); it != block_.end(); ++it) {
            if (f(*it, rhs)) { mask(it.index()).set(); }
        }
        return mask;
    }

    DataObj* data_;
    storage_t block_;
    // metadata
    size_t rows_, blk_sz_;
    index_t row_begin_, row_end_;
    std::string colname_;
    dtype type_id_;
};

// random access column view
template <typename Scalar_, typename DataObj> struct random_access_col_view {
    using Scalar = Scalar_;
    using index_t = typename DataObj::index_t;
    using size_t = typename DataObj::size_t;
    static constexpr int Order = DataObj::Order;
   private:
    using data_table = MdArray<Scalar_, full_dynamic_extent_t<Order>, internals::layout_left>;
    using extents_t = typename data_table::extents_t;
   public:
    using storage_t = std::conditional_t<
      std::is_const_v<DataObj>, MdArrayBlock<std::add_const_t<data_table>, extents_t>,
      MdArrayBlock<data_table, extents_t>>;
    using reference = typename DataObj::template reference<Scalar>;
    using const_reference = typename DataObj::template const_reference<Scalar>;
    using logical_t = MdArray<bool, extents_t, internals::layout_left>;

    random_access_col_view() noexcept = default;
    template <typename FieldDescriptor>
    random_access_col_view(DataObj& data, const std::vector<index_t>& idxs, const FieldDescriptor& desc) noexcept :
        data_(internals::apply_index_pack<Order>([&]<int... Ns_>() {
            return data.template data<Scalar>().block(
              ((void)Ns_, Ns_ == 1 ? std::pair {desc.offset(), desc.offset() + desc.size() - 1} :
                                     std::pair {0, index_t(data.template data<Scalar>().extent(Ns_)) - 1})...);
        })),
        idxs_(idxs),
        rows_(idxs.size()),
        blk_sz_(desc.size()),
        colname_(desc.colname()),
        type_id_(desc.type_id()) {
        // set up extents
        if constexpr (Order == 2) { extents_.resize(rows_, blk_sz_); }
        if constexpr (Order == 3) { extents_.resize(rows_, blk_sz_, data_.extent(2)); }
    }

    // observers
    size_t rows() const { return rows_; }
    size_t cols() const { return 1; }
    size_t size() const { return rows_ * blk_sz_; }
    // access to data requires copying in contiguous memory
    data_table data() const {
        data_table data_blk(extents_);
        for (auto it = data_blk.begin(); it != data_blk.end(); ++it) {
            *it = internals::apply_index_pack<Order>(
              [&]<int... Ns_>() { return data_((Ns_ == 0 ? idxs_[it.index()[0]] : it.index()[Ns_])...); });
        }
        return data_blk;
    }
    const std::string& colname() const { return colname_; }
    dtype type_id() const { return type_id_; }
    logical_t nan() const { return compute_na_mask(data()); }
    // accessors
    template <typename... Idxs>
        requires(std::is_convertible_v<Idxs, index_t> && ...) && (sizeof...(Idxs) == Order && !std::is_const_v<DataObj>)
    constexpr reference operator()(Idxs&&... idxs) {
        return internals::apply_index_pack<sizeof...(Idxs)>([&]<int... Ns_>() -> decltype(auto) {
            return data_((Ns_ == 0 ? static_cast<index_t>(idxs_[idxs]) : idxs)...);
        });
    }
    template <typename IndexPack>   // access via index-pack object
        requires(internals::is_subscriptable<IndexPack, index_t>)
    constexpr reference operator()(IndexPack&& index_pack) {
        return block_(index_pack);
    }
    template <typename... Idxs>
        requires(std::is_convertible_v<Idxs, index_t> && ...) && (sizeof...(Idxs) == Order)
    constexpr const_reference operator()(Idxs&&... idxs) const {
        return internals::apply_index_pack<sizeof...(Idxs)>([&]<int... Ns_>() -> decltype(auto) {
            return data_((Ns_ == 0 ? static_cast<index_t>(idxs_[idxs]) : idxs)...);
        });
    }
    template <typename IndexPack>   // access via index-pack object
        requires(internals::is_subscriptable<IndexPack, index_t>)
    constexpr const_reference operator()(IndexPack&& index_pack) const {
        return block_(index_pack);
    }
    // logical comparison
    logical_t operator==(const Scalar& rhs) const { return logical_apply_(rhs, std::equal_to<Scalar>      {}); }
    logical_t operator!=(const Scalar& rhs) const { return logical_apply_(rhs, std::not_equal_to<Scalar>  {}); }
    logical_t operator< (const Scalar& rhs) const { return logical_apply_(rhs, std::less<Scalar>          {}); }
    logical_t operator> (const Scalar& rhs) const { return logical_apply_(rhs, std::greater<Scalar>       {}); }
    logical_t operator<=(const Scalar& rhs) const { return logical_apply_(rhs, std::less_equal<Scalar>    {}); }
    logical_t operator>=(const Scalar& rhs) const { return logical_apply_(rhs, std::greater_equal<Scalar> {}); }
#ifdef __FDAPDE_HAS_EIGEN__
    // requesting a matrix requires to copy data, no view is possible for random access
    Eigen::Matrix<Scalar, Dynamic, Dynamic, Eigen::ColMajor> as_matrix() const {
        fdapde_static_assert(Order == 2, THIS_METHOD_IS_FOR_ORDER_TWO_MDARRAYS_ONLY);
        Eigen::Matrix<Scalar, Dynamic, Dynamic, Eigen::ColMajor> mat(rows_, blk_sz_);
        for (int i = 0; i < rows_; ++i) {
            for (int j = 0; j < blk_sz_; ++j) { mat(i, j) = operator()(i, j); }
        }
        return mat;
    }
#endif
   protected:
    template <typename Functor> std::vector<bool> logical_apply_(const Scalar& rhs, Functor&& f) const {
        logical_t mask(extents_);
        for (auto it = mask.begin(); it != mask.end(); ++it) {
            if (internals::apply_index_pack([&]<int... Ns_>() -> bool {
                    return f(data_((Ns_ == 0 ? idxs_[it.index()[0]] : it.index()[Ns_])...), rhs);
                })) {
                it.set();
            }
        }
        return mask;
    }
    storage_t data_;
    std::vector<index_t> idxs_;
    extents_t extents_;   // contiguous-like memory block extention
    // metadata
    size_t rows_, blk_sz_, offset_;
    std::string colname_;
    dtype type_id_;
};

// a row is an heterogeneous view of data from a plain_data_layer
template <typename DataLayer> struct plain_row_view {
    using This = plain_row_view<DataLayer>;
    using index_t = typename DataLayer::index_t;
    using size_t = typename DataLayer::size_t;
    using storage_t = std::conditional_t<
      std::is_const_v<DataLayer>, std::add_const_t<typename DataLayer::storage_t>, typename DataLayer::storage_t>;
    static constexpr int Order = DataLayer::Order;
    using logical_t = MdArray<bool, full_dynamic_extent_t<Order>, internals::layout_left>;
  
    plain_row_view() noexcept = default;
    plain_row_view(DataLayer* data, index_t row) noexcept : data_(data), row_(row) { }
    // observers
    size_t rows() const { return 1; }
    size_t cols() const { return data_->cols(); }
    index_t id() const { return row_; }
    const auto& field_descriptors() const { return data_->field_descriptors(); }
    const auto& field_descriptor(const std::string& colname) const { return data_->field_descriptor(colname); }
    const DataLayer& data() const { return *data_; }
    DataLayer& data() { return *data_; }
    // accessors
    template <typename T> plain_col_view<T, This> col(const std::string& colname) {
        return plain_col_view<T, This>(data_, row_, data_->field_descriptor(colname));
    }
    template <typename T> plain_col_view<T, const This> col(const std::string& colname) const {
        return plain_col_view<T, const This>(data_, row_, data_->field_descriptor(colname));
    }
    template <typename T> plain_col_view<T, This> col(size_t col) {
        return plain_col_view<T, This>(data_, row_, data_->field_descriptor(col));
    }
    template <typename T> plain_col_view<T, const This> col(size_t col) const {
        return plain_col_view<T, const This>(data_, row_, data_->field_descriptor(col));
    }
    // modifiers
    plain_row_view& operator=(const plain_row_view& src) {
        for (const auto& f : field_descriptors()) {
            internals::dispatch_to_dtype(
              f.type_id(),
              [&]<typename T>(plain_row_view& dst) mutable {
                  dst.col<T>(f.colname).data() = src.col<T>(f.colname).data();
              },
              *this);
        }
        return *this;
    }
   protected:
    DataLayer* data_;
    index_t row_;
};

// an indexed set of rows
template <typename DataLayer> struct random_access_row_view {
    using index_t = typename DataLayer::index_t;
    using size_t = typename DataLayer::size_t;
    using row_view = typename DataLayer::row_view;
    using const_row_view = typename DataLayer::const_row_view;
    using storage_t = std::conditional_t<
      std::is_const_v<DataLayer>, std::add_const<typename DataLayer::storage_t>, typename DataLayer::storage_t>;
    static constexpr int Order = DataLayer::Order;

    random_access_row_view() noexcept = default;
    template <typename Iterator>
    random_access_row_view(DataLayer* data, Iterator begin, Iterator end) : data_(data), idxs_(begin, end) {
        fdapde_assert(
          *begin >= 0 && std::cmp_less(*begin FDAPDE_COMMA data->rows()) && *(end - 1) >= *begin &&
          std::cmp_less(*(end - 1) FDAPDE_COMMA data->rows()));
    }
    template <typename Filter>
        requires(requires(Filter f, index_t i) {
            { f(i) } -> std::same_as<bool>;
        })
    random_access_row_view(DataLayer* data, Filter&& f) : data_(data) {
        for (size_t i = 0, n = data_->rows(); i < n; ++i) {
            if (f(i)) idxs_.push_back(i);
        }
    }
    template <typename LogicalVec>
        requires(requires(LogicalVec vec, int i) {
            { vec.size() } -> std::convertible_to<size_t>;
            { vec[i] } -> std::convertible_to<bool>;
        })
    random_access_row_view(DataLayer* data, const LogicalVec& vec) : data_(data) {
        fdapde_assert(vec.size() == data_->rows());
        for (size_t i = 0, n = vec.size(); i < n; ++i) {
            if (vec[i]) { idxs_.push_back(i); }
        }
    }
    // observers
    size_t rows() const { return idxs_.size(); }
    size_t cols() const { return data_->cols(); }
    size_t size() const { return rows() * cols(); }
    const auto& field_descriptors() const { return data_->field_descriptors(); }
    const auto& field_descriptor(const std::string& colname) const { return data_->field_descriptor(colname); }
    std::vector<std::string> colnames() const { return data_->colnames(); }
    // accessors
    plain_row_view<DataLayer> operator[](index_t i) {
        fdapde_assert(i < idxs_.size());
        return data_->row(idxs_[i]);
    }
    plain_row_view<const DataLayer> operator[](index_t i) const {
        fdapde_assert(i < idxs_.size());
        return data_->row(idxs_[i]);
    }
    template <typename T> random_access_col_view<T, DataLayer> col(const std::string& colname) {
        return random_access_col_view<T, DataLayer>(*data_, idxs_, data_->field_descriptor(colname));
    }
    template <typename T> random_access_col_view<T, const DataLayer> col(const std::string& colname) const {
        return random_access_col_view<T, const DataLayer>(*data_, idxs_, data_->field_descriptor(colname));
    }
    template <typename T> random_access_col_view<T, DataLayer> col(size_t col) {
        return random_access_col_view<T, DataLayer>(*data_, idxs_, data_->field_descriptor(col));
    }
    template <typename T> random_access_col_view<T, const DataLayer> col(size_t col) const {
        return random_access_col_view<T, const DataLayer>(*data_, idxs_, data_->field_descriptor(col));
    }
    // iterator support
    class iterator {
       public:
        using value_type = plain_row_view<DataLayer>;
        using pointer_t = std::add_pointer_t<value_type>;
        using reference = std::add_lvalue_reference_t<value_type>;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;
        using iterator_category = std::forward_iterator_tag;

        iterator() = default;
        iterator(int index, random_access_row_view* filter) : index_(index), filter_(filter), val_() {
            if (index_ != filter_->rows()) { val_ = filter_->operator[](index); }
            index_++;
        }
        reference operator*() { return val_; }
        const reference operator*() const { return val_; }
        pointer_t operator->() { return std::addressof(val_); }
        const pointer_t operator->() const { return std::addressof(val_); }
        iterator& operator++() {
            if (index_ != filter_->rows()) { [[likely]] val_ = filter_->operator[](index_); }
            index_++;
            return *this;
        }
        friend bool operator==(const iterator& lhs, const iterator& rhs) { return lhs.index_ == rhs.index_; }
        friend bool operator!=(const iterator& lhs, const iterator& rhs) { return lhs.index_ != rhs.index_; }
       private:
        index_t index_;
        random_access_row_view* filter_;
        value_type val_;
    };
    iterator begin() { return iterator(0, this); }
    iterator end() { return iterator(idxs_.size(), this); }
   protected:
    DataLayer* data_;
    std::vector<index_t> idxs_;
};
  
// heterogeneous container
class scalar_data_layer {
    using types = std::tuple<
      double,         // data_t::flt64
      float,          // data_t::flt32
      std::int64_t,   // data_t::int64
      std::int32_t,   // data_t::int32
      bool,           // data_t::bin
      std::string     // data_t::str
      >;
    template <typename T> struct mapped_type {
        using type = std::decay_t<decltype([](T) {
            if constexpr (internals::is_integer_v<T> && !std::is_same_v<T, std::int64_t>) {
                return std::int32_t();   // map integral, not 64 bit and not boolean types, to 32 bit int
            } else {
                return T();   // do not map other types
            }
        }(T()))>;
    };
    template <typename T> using mapped_type_t = typename mapped_type<T>::type;
    // factory method for a map mapping dynamic types to T
    template <typename T> std::unordered_map<dtype, T> make_dtyped_map() const {
        return std::unordered_map<dtype, T> {
          {dtype::flt64, T()},
          {dtype::flt32, T()},
          {dtype::int64, T()},
          {dtype::int32, T()},
          {dtype::bin,   T()},
          {dtype::str,   T()}
        };
    }
   public:
    static constexpr int Order = 2;   // MdArray order
    template <typename T> static constexpr bool is_type_supported_v  = has_type<T, types>::value;
    template <typename T> static constexpr bool is_dtype_supported_v = is_type_supported_v<typename T::type>;
   private:
    template <typename T> using data_table = MdArray<T, full_dynamic_extent_t<Order>, internals::layout_left>;
    template <typename... Ts> using data_map_ = std::tuple<data_table<Ts>...>;
    using This = scalar_data_layer;
    template <typename T, typename U>
        requires(is_type_supported_v<T>)
    auto& fetch_(U& u) {
        return std::get<index_of<T, types>::value>(u);
    }
    template <typename T, typename U>
        requires(is_type_supported_v<T>)
    const auto& fetch_(const U& u) const {
        return std::get<index_of<T, types>::value>(u);
    }
    bool has_column_(const std::string& colname) const { return col_idx_.contains(colname); }
    template <typename T> bool has_column_of_type_(const std::string& colname) const {
        dtype col_dtype = internals::dtype_from_static_type<T>().type_id;
        for (const auto& field : header_) {
            if (field.type_id() == col_dtype && field.colname() == colname) { return true; }
        }
        return false;
    }
    struct field {
        std::string colname_;
        int size_, offset_;   // block size and first indexed MdArray column
        dtype type_id_;
      
       public:
        field() noexcept = default;
        field(const std::string& colname, int offset, int size, dtype type_id) noexcept :
            colname_(colname), size_(size), offset_(offset), type_id_(type_id) { }
        field(const std::string& colname, int offset, dtype type_id) noexcept : field(colname, offset, 1, type_id) { }
        field(const std::string& colname, dtype type_id) noexcept : field(colname, 0, 0, type_id) { }
        // observers
        const std::string& colname() const { return colname_; }
        int size() const { return size_; }
        int offset() const { return offset_; }
        const dtype& type_id() const { return type_id_; }
        // modifiers
        void set_colname(const std::string& colname) { colname_ = colname; }
    };
    template <typename T>
    struct is_valid_pair {
      using colname_t = std::tuple_element_t<0, std::decay_t<T>>;
      using value_t   = mapped_type_t<std::tuple_element_t<1, std::decay_t<T>>>;
      static constexpr bool value =
	// first  element: column name
        (std::is_same_v<colname_t, std::string> || std::is_same_v<colname_t, const char*>) &&
	// second element: something subscriptable returning an accepted type
        (internals::is_subscriptable<value_t, int> &&
         is_type_supported_v<mapped_type_t<std::decay_t<decltype(std::declval<value_t>()[int()])>>>);
    };
    template <typename T> static constexpr bool is_valid_pair_v = is_valid_pair<T>::value;  
    // moves std::tuple<Ts...> to T<Ts...>
    template <template <typename...> typename T, typename U> struct strip_tuple_into;
    template <template <typename...> typename T, typename... Us>
    struct strip_tuple_into<T, std::tuple<Us...>> : std::type_identity<T<Us...>> { };
   public:
    template <typename T> using reference = typename data_table<T>::reference;
    template <typename T> using const_reference = typename data_table<T>::const_reference;
    using storage_t = typename strip_tuple_into<data_map_, types>::type;
    using logical_t = MdArray<bool, full_dynamic_extent_t<Order>, internals::layout_left>;
    using index_t = int;
    using size_t = std::size_t;
    using field_t = field;
    template <typename T> class is_indexable {
        using T_ = std::decay_t<T>;
       public:
        static constexpr bool value = []() {
            return internals::apply_index_pack<Order>([]<int... Ns_>() {
                if constexpr (requires(T_ t) { t(((void)Ns_, index_t())...); }) {
                    return is_type_supported_v<std::decay_t<decltype(std::declval<T_>()(((void)Ns_, index_t())...))>>;
                } else {
                    return false;
                }
            });
        }();
    };
    template <typename T> static constexpr bool is_indexable_v = is_indexable<T>::value;
    template <typename T> class is_vector_like {
        using T_ = std::decay_t<T>;
       public:
        static constexpr bool value = []() {
            if constexpr (std::is_pointer_v<T_>) {
                return is_type_supported_v<std::remove_pointer_t<T_>>;
            } else {
                if constexpr (is_subscriptable<T_, index_t> && requires(T t) {
                                  { t.size() } -> std::convertible_to<size_t>;
                              }) {
                    return is_type_supported_v<std::decay_t<decltype(std::declval<T_>()[index_t()])>>;
                } else {
                    return false;
                }
            }
        }();
    };
    template <typename T> static constexpr bool is_vector_like_v = is_vector_like<T>::value;

    using row_view = plain_row_view<scalar_data_layer>;
    using const_row_view = plain_row_view<const scalar_data_layer>;

    scalar_data_layer() noexcept : rows_(0), cols_(0) { }
    // build from header
    scalar_data_layer(const std::vector<field>& header) :
        header_(header), freemem_(make_dtyped_map<std::vector<bool>>()), rows_(0), cols_(header.size()) {
        for (size_t i = 0; i < cols_; ++i) { col_idx_[header_[i].colname()] = i; }
    }
    template <typename... DataT>
        requires(
          ([]() {
              if constexpr (internals::is_pair_v<std::decay_t<DataT>>) {
                  return is_valid_pair_v<std::decay_t<DataT>>;
              } else if constexpr (requires(DataT data, index_t i) {
                                       { data[i] };
                                   }) {
                  using subscript_t = std::decay_t<decltype(std::declval<DataT>()[std::declval<index_t>()])>;
                  if constexpr (internals::is_pair_v<subscript_t>) {   // check if subscripting DataT returns a tuple
                      return is_valid_pair_v<subscript_t>;
                  } else {
                      return false;
                  }
              } else {
                  return false;
              }
          }()) &&
          ...)
    scalar_data_layer(DataT&&... data) : freemem_(make_dtyped_map<std::vector<bool>>()) {
        // for each type id, the number of columns of that type
        std::unordered_map<dtype, int> type_id_map = make_dtyped_map<int>();
        std::unordered_map<dtype, int> type_id_col = make_dtyped_map<int>();
        auto push_column_descriptor = [&, this]<typename T>(const std::string& colname, const T& t) mutable {
            fdapde_assert(!has_column_(colname) && colname.size() != 0);
            if (rows_ == 0) {
                rows_ = t.size();
            } else {
                fdapde_assert(rows_ != 0 && rows_ == t.size());
            }
            using MappedT = mapped_type_t<std::decay_t<decltype(std::declval<T>()[std::declval<index_t>()])>>;
            // add field descriptor
            auto dtype_ = internals::dtype_from_static_type<MappedT>();
            header_.emplace_back(colname, dtype_.type_id);
            type_id_map[dtype_.type_id]++;
            col_idx_[colname] = header_.size() - 1;
        };
        // push column descriptors
        internals::for_each_index_and_args<sizeof...(DataT)>(
          [&]<int Ns_, typename T>(T t) {
              if constexpr (internals::is_pair_v<T>) {
                  push_column_descriptor(std::get<0>(t), std::get<1>(t));
		  cols_++;
              } else {   // vector of pairs
                  for (const auto& pair : t) {
                      push_column_descriptor(std::get<0>(pair), std::get<1>(pair));
                      cols_++;
                  }
              }
          },
          data...);
        // reserve space for all types in pack, copy data in internal storage
        internals::for_each_index_and_args<sizeof...(DataT)>(
          [&]<int Ns_, typename T>(T t) {
              using MappedT = mapped_type_t<std::decay_t<decltype([t]() {
                  if constexpr (internals::is_pair_v<T>) {
                      return std::get<1>(t);
                  } else {
                      return std::get<1>(t[0]);
                  }
              }().operator[](std::declval<index_t>()))>>;
              dtype type_id = internals::dtype_from_static_type<MappedT>().type_id;
              fetch_<MappedT>(data_).resize(rows_, type_id_map[type_id]);
              if constexpr (internals::is_pair_v<T>) {
                  fetch_<MappedT>(data_).template slice<1>(type_id_col[type_id]).assign_inplace_from(std::get<1>(t));
                  type_id_col[type_id]++;
		  freemem_[type_id].push_back(false);
              } else {   // map-like object
                  for (const auto& [colname, data] : t) {
                      fetch_<MappedT>(data_).template slice<1>(type_id_col[type_id]).assign_inplace_from(data);
                      type_id_col[type_id]++; // ------------------------------------ change in type_id_col_cnt
		      freemem_[type_id].push_back(false);
                  }
              }
          },
          data...);
    }
    template <typename... DataT>
        requires(
          (std::contiguous_iterator<typename DataT::iterator> && is_type_supported_v<typename DataT::value_type>) &&
          ...)
    scalar_data_layer(const std::vector<std::string>& colnames, const DataT&... data) :
        freemem_(make_dtyped_map<std::vector<bool>>()) {
        fdapde_assert(colnames.size() == sizeof...(data));
        // for each type id, the number of columns of that type
        std::unordered_map<dtype, int> type_id_map = make_dtyped_map<int>();
        std::unordered_map<dtype, int> type_id_col = make_dtyped_map<int>();
        // push column descriptors
        internals::for_each_index_and_args<sizeof...(DataT)>(
          [&]<int Ns_, typename T>(T t) {
              fdapde_assert(!has_column_(colnames[Ns_]) && colnames[Ns_].size() != 0);
              if (rows_ == 0) {
                  rows_ = t.size();
              } else {
                  fdapde_assert(rows_ != 0 && rows_ == t.size());
              }
              using MappedT = mapped_type_t<std::decay_t<decltype(std::declval<T>()[std::declval<index_t>()])>>;
              // add field descriptor
              dtype type_id = internals::dtype_from_static_type<MappedT>().type_id;
              header_.emplace_back(colnames[Ns_], type_id);
              type_id_map[type_id]++;
              col_idx_[colnames[Ns_]] = header_.size() - 1;
              cols_++;
          },
          data...);
        // reserve space for all types in pack, copy data in internal storage
        internals::for_each_index_and_args<sizeof...(DataT)>(
          [&]<int Ns_, typename T>(const T& t) {
              using MappedT = mapped_type_t<std::decay_t<decltype(std::declval<T>()[std::declval<index_t>()])>>;
              dtype type_id = internals::dtype_from_static_type<MappedT>().type_id;
              fetch_<MappedT>(data_).resize(rows_, type_id_map[type_id]);
              fetch_<MappedT>(data_).template slice<1>(type_id_col[type_id]).assign_inplace_from(t);
              type_id_col[type_id]++;
	      freemem_[type_id].push_back(false);
          },
          data...);
    }
    template <typename LayerType>
    scalar_data_layer(const random_access_row_view<LayerType>& row_filter, const std::vector<std::string>& cols) :
        freemem_(make_dtyped_map<std::vector<bool>>()) {
        if (cols.size() == 0) return;
        // for each type id, the requested memory block size
        std::unordered_map<dtype, int> type_id_map = make_dtyped_map<int>();
        std::unordered_map<dtype, int> offset = make_dtyped_map<int>();
        // push column descriptors
        rows_ = row_filter.rows();
	cols_ = cols.size();
        for (size_t i = 0; i < cols_; ++i) {
            auto field = row_filter.field_descriptor(cols[i]);
            dtype type_id = field.type_id();
            header_.emplace_back(cols[i], offset[type_id], field.size(), type_id);
            offset[type_id] += field.size();
            type_id_map[type_id]++;
            col_idx_[field.colname()] = header_.size() - 1;
        }
        // reserve space, copy data in internal storage
	std::unordered_map<dtype, int> tmp = make_dtyped_map<int>();
        std::apply(
          [&](const auto&... ts) {
              (
                [&]() {
                    using T = std::decay_t<decltype(ts)>;
		    dtype type_id = internals::dtype_from_static_type<T>().type_id;
                    if (type_id_map[type_id] != 0) {
                        fetch_<T>(data_).resize(rows_, offset[type_id]);
			// take typed data from filter
                        for (const auto& colname : cols) {
                            auto desc = row_filter.field_descriptor(colname);
                            if (desc.type_id() == type_id) {
                                fetch_<T>(data_)
                                  .block(full_extent, std::pair {tmp[type_id], tmp[type_id] + desc.size() - 1})
                                  .assign_inplace_from(row_filter.template col<T>(colname).data());
                                for (int i = 0; i < desc.size(); ++i) { freemem_[type_id].push_back(false); }
				tmp[type_id] += desc.size();
                            }
                        }
                    }
                }(),
                ...);
          },
          types {});
    }
    template <typename LayerType>
    scalar_data_layer(const random_access_row_view<LayerType>& row_filter) :
        scalar_data_layer(row_filter, row_filter.colnames()) { }

    // observers
    const field& field_descriptor(const std::string& colname) const { return header_[col_idx_.at(colname)]; }
    index_t col_id(const std::string& colname) const { return col_idx_.at(colname); }
    const std::vector<field>& header() const { return header_; }
    std::vector<std::string> colnames() const {
        std::vector<std::string> colnames_;
        for (int i = 0, n = header_.size(); i < n; ++i) { colnames_.push_back(header_[i].colname()); }
        return colnames_;
    }
    bool contains(const std::string& column) const {   // true if this layer contains column
        return has_column_(column);
    }
    logical_t nan(const std::vector<std::string>& colnames) const {
        fdapde_assert(colnames.size() > 0);
        size_t n_rows = rows_, n_cols = 0;
	std::vector<std::size_t> offset;
	offset.push_back(0);
        for (const std::string& col : colnames) {
            fdapde_assert(has_column_(col));
            offset.push_back(offset.back() + header_.at(col_idx_.at(col)).size());
        }
        logical_t nan(n_rows, n_cols);
        for (size_t i = 0; i < colnames.size(); ++i) { extract_nan_pattern_(colnames[i], nan, offset[i]); }
        return nan;
    }
    logical_t nan(const std::string& colname) const {
        fdapde_assert(has_column_(colname));
        field f = header_.at(col_idx_.at(colname));
        logical_t nan(rows_, f.size());
        extract_nan_pattern_(colname, nan, 0);
        return nan;
    }
    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }
    size_t size() const { return rows_* cols_; }
    // maximum number of elements of type T which can currently be holded
    template <typename T> size_t capacity() const { return fetch_<mapped_type_t<T>>(data_).size(); }

    // accessors
    // column access
    template <typename T> plain_col_view<T, scalar_data_layer> col(size_t col) {
        fdapde_assert(col < cols_);
        return plain_col_view<T, scalar_data_layer>(*this, header_[col]);
    }
    template <typename T> plain_col_view<T, const scalar_data_layer> col(size_t col) const {
        fdapde_assert(col < cols_);
        return plain_col_view<T, const scalar_data_layer>(*this, header_[col]);
    }
    template <typename T> plain_col_view<T, scalar_data_layer> col(const std::string& colname) {
        fdapde_assert(has_column_(colname));
        return col<T>(col_idx_.at(colname));
    }
    template <typename T> plain_col_view<T, const scalar_data_layer> col(const std::string& colname) const {
        fdapde_assert(has_column_(colname));
        return col<T>(col_idx_.at(colname));
    }
    // row access
    plain_row_view<scalar_data_layer> row(size_t row) {
        fdapde_assert(row < rows_);
        return plain_row_view<scalar_data_layer>(this, row);
    }
    plain_row_view<const scalar_data_layer> row(size_t row) const {
        fdapde_assert(row < rows_);
        return plain_row_view<const scalar_data_layer>(this, row);
    }
    // row filtering operations
    template <typename Iterator>
        requires(internals::is_integer_v<typename Iterator::value_type>)
    random_access_row_view<const scalar_data_layer> select(Iterator begin, Iterator end) const {
        return random_access_row_view<const scalar_data_layer>(this, begin, end);
    }
    template <typename T>
        requires(std::is_convertible_v<T, index_t>)
    random_access_row_view<const scalar_data_layer> select(const std::initializer_list<T>& idxs) const {
        return random_access_row_view<const scalar_data_layer>(this, idxs.begin(), idxs.end());
    }
    template <typename LogicalPred>
        requires(
	  requires(LogicalPred pred, index_t i) { { pred(i) } -> std::convertible_to<bool>; } ||
          requires(LogicalPred pred, index_t i) { { pred[i] } -> std::convertible_to<bool>; })
    random_access_row_view<const scalar_data_layer> select(LogicalPred&& pred) const {
        return random_access_row_view<const scalar_data_layer>(this, std::forward<LogicalPred>(pred));
    }
  
    template <typename T> const data_table<T>& data() const { return fetch_<mapped_type_t<T>>(data_); }
    template <typename T> data_table<T>& data() { return fetch_<mapped_type_t<T>>(data_); }
    // modifiers
    void set_colnames(const std::vector<std::string>& colnames) {
        fdapde_assert(colnames.size() == header_.size());
        for (size_t i = 0; i < colnames.size(); ++i) { header_[i].set_colname(colnames[i]); }
        return;
    }
    template <typename Scalar, typename... Extents_>   // reserve memory for Extents_... Scalar
        requires(std::is_convertible_v<Extents_, index_t> && ...) &&
                (sizeof...(Extents_) == Order && is_type_supported_v<Scalar>)
    void resize(Extents_... exts) {
        auto& data = fetch_<Scalar>(data_);
        if (internals::apply_index_pack<Order>(
              [&]<int... Ns_>() { return ((std::cmp_equal(exts, data.extent(Ns_))) && ...); })) {
            return;   // exts coincide with current size, skip resizing
        }
        data.resize(static_cast<index_t>(exts)...);   // resize storage discarding old values
	fdapde_assert(rows_ == 0 || rows_ == data.extent(0));
        rows_ = data.extent(0);
        dtype type_id = internals::dtype_from_static_type<Scalar>().type_id;
        freemem_[type_id].resize(data.extent(1));
	// invalidate memory
        for (typename std::vector<bool>::reference b : freemem_[type_id]) { b = true; }
        for (auto it = header_.begin(); it != header_.end();) {
            if (it->type_id() == type_id) {
                col_idx_.erase(it->colname());
                it = header_.erase(it);
		cols_--;
            } else {
                ++it;
            }
        }
        return;
    }
    // resize mdarray storage, preserving old values
    template <typename Scalar, typename... Extents_>
        requires(std::is_convertible_v<Extents_, index_t> && ...) &&
                (sizeof...(Extents_) == Order && is_type_supported_v<Scalar>)
    void conservative_resize(Extents_... exts) {
        using mem_t = MdArray<Scalar, full_dynamic_extent_t<Order>, internals::layout_left>;
        auto& data = fetch_<Scalar>(data_);
        if (internals::apply_index_pack<Order>([&]<int... Ns_>() { return ((exts == data.extent(Ns_)) && ...); })) {
            return;   // exts coincide with current size, skip resizing
        }
        dtype type_id = internals::dtype_from_static_type<Scalar>().type_id;
        if (data.size() != 0) {
            std::array<index_t, Order> exts_;
            internals::for_each_index_and_args<Order>(
              [&]<int Ns, typename Ts>(const Ts& ts) {
                  exts_[Ns] = std::min(ts, fetch_<Scalar>(data_).extent(Ns)) - 1;
              },
              exts...);
            mem_t tmp = internals::apply_index_pack<Order>(   // copy old values
              [&, this]<int... Ns>() { return fetch_<Scalar>(data_).block(std::make_pair(0, exts_[Ns])...); });
            int old_size = tmp.extent(1);
            // allocate memory
            data.resize(static_cast<index_t>(exts)...);
	    int new_size = data.extent(1);
            fdapde_assert(rows_ == 0 || rows_ == data.extent(0));
            rows_ = data.extent(0);
	    // flag new memory as free
            for (int i = 0; i < new_size - old_size; ++i) { freemem_[type_id].push_back(true); }
            fetch_<Scalar>(data_)
              .block(std::make_pair(0, exts_[0]), std::make_pair(0, exts_[1]))
              .assign_inplace_from(tmp);	    
        } else {   // nothing to copy
            data.resize(static_cast<index_t>(exts)...);   // resize storage discarding old values
            fdapde_assert(rows_ == 0 || rows_ == data.extent(0));
            rows_ = data.extent(0);
            freemem_[type_id].resize(data.extent(1));
            for (typename std::vector<bool>::reference b : freemem_[type_id]) { b = true; }
        }
        return;
    }

    template <typename Src>
        requires(is_vector_like_v<Src>)
    void append_vec(const std::string& colname, const Src& src) {
        using SrcType = std::conditional_t<
          std::is_pointer_v<Src>, std::remove_pointer_t<Src>, std::decay_t<decltype(std::declval<Src>()[index_t()])>>;
        using SrcType_ = mapped_type_t<SrcType>;
        if constexpr (!std::is_pointer_v<Src>) {
            fdapde_assert(fetch_<SrcType_>(data_).extent(0) == 0 || src.size() == fetch_<SrcType_>(data_).extent(0));
        }
	fdapde_assert(!has_column_(colname));
        dtype type_id = internals::dtype_from_static_type<SrcType_>().type_id;
        // check if there is already allocated free memory to hold src
        index_t offset = find_free_blk_idx_<SrcType_>(1);
        if (offset == -1) {   // memory allocation requested
            offset = 0;
            for (const field& f : header_) {
                if (f.type_id() == type_id) { offset += f.size(); }
            }
            if (offset == 0) {
                resize<SrcType_>(src.size(), 1);
            } else if (std::cmp_equal(offset, fetch_<SrcType>(data_).extent(1))) {
                // double number of columns, guarantees amortized constant time insertion
                internals::apply_index_pack<Order>([&]<int... Ns_>() {
                    conservative_resize<SrcType_>(
                      (Ns_ == 1 ? (2 * fetch_<SrcType_>(data_).extent(Ns_)) : fetch_<SrcType_>(data_).extent(Ns_))...);
                });
            }
        }
        // update header
        header_.emplace_back(colname, offset, 1, type_id);
        col_idx_[colname] = header_.size() - 1;
        freemem_[type_id][offset] = false;
        // copy src into data_
        fetch_<SrcType_>(data_).template slice<1>(offset).assign_inplace_from(src);
        cols_++;
        if (rows_ == 0) { rows_ = src.size(); }
        return;
    }
    template <typename Src>
        requires(is_indexable_v<Src>)
    void append_blk(const std::string& colname, const Src& src) {
        using ValueType_ =
          decltype(internals::apply_index_pack<Order>([&]<int... Ns_>() { return src(((void)Ns_, index_t())...); }));
        using SrcType_ = mapped_type_t<std::decay_t<ValueType_>>;
        fdapde_assert(fetch_<SrcType_>(data_).extent(0) == 0 || src.rows() == fetch_<SrcType_>(data_).extent(0));
        fdapde_assert(!has_column_(colname));
        dtype type_id = internals::dtype_from_static_type<SrcType_>().type_id;
        // check if there is already allocated free memory to hold src
        index_t offset = find_free_blk_idx_<SrcType_>(src.cols());
        if (offset == -1) {   // memory allocation requested
            offset = 0;
            for (const field& f : header_) {
                if (f.type_id() == type_id) { offset += f.size(); }
            }
            if (offset == 0) {
                resize<SrcType_>(src.rows(), src.cols());
            } else if (offset + src.cols() > fetch_<SrcType_>(data_).extent(1)) {
                internals::apply_index_pack<Order>([&]<int... Ns_>() {
                    conservative_resize<SrcType_>(
                      (Ns_ == 1 ? (2 * src.cols() + fetch_<SrcType_>(data_).extent(Ns_)) :
                                  fetch_<SrcType_>(data_).extent(Ns_))...);
                });
            }
        }
        // update header
        header_.emplace_back(colname, offset, src.cols(), type_id);
        col_idx_[colname] = header_.size() - 1;
        for (int i = 0; i < src.cols(); ++i) { freemem_[type_id][offset + i] = false; }
        // copy src into data_
        fetch_<SrcType_>(data_).block(full_extent, std::pair{offset, offset + src.cols() - 1}).assign_inplace_from(src);
        cols_++;
        if (rows_ == 0) { rows_ = src.rows(); }
        return;
    }
    // insert src at index position
    template <typename Src>
        requires(is_vector_like_v<Src> || is_indexable_v<Src>)
    void insert(const std::string& colname, index_t index, const Src& src) {
        if constexpr (is_vector_like_v<Src> && !is_indexable_v<Src>) {
            append_vec(colname, src);
        } else {
            append_blk(colname, src);
        }
        // adjust header to put colname at index
        field tmp = header_[index];
        header_[index] = header_[col_idx_[colname]];
	col_idx_[colname] = index;
        for (size_t i = index + 1; i < cols_; ++i) {
            field cur = header_[i];
            header_[i] = tmp;
	    col_idx_[tmp.colname()] = i;
            tmp = cur;
	}
    }
    // do not perform any memory reallocation, sets the corresponding freemem_ bits to 1 and update header
    void erase(const std::string& colname) {
        fdapde_assert(has_column_(colname));
        auto it = std::find_if(header_.begin(), header_.end(), [&](const field& f) { return f.colname() == colname; });
        for (int i = it->offset(); i < it->offset() + it->size(); ++i) { freemem_[it->type_id()][i] = true; }
        col_idx_.erase(colname);
        header_.erase(it);
        cols_--;
    }
    void shrink_to_fit() { }

    // merge all columns of type T in a single block
    template <typename T> void merge(const std::string& colname) {
        using MappedT = mapped_type_t<T>;
        dtype type_id = internals::dtype_from_static_type<MappedT>().type_id;
	int size = 0, index = -1;;
        // erase all columns of type T from header
        for (auto it = header_.begin(); it != header_.end();) {
            if (it->type_id() == type_id) {
                if (index == -1) { index = col_idx_.at(it->colname()); }
                col_idx_.erase(it->colname());
                it = header_.erase(it);
		cols_--;
		size++;
            } else {
                ++it;
            }
        }
        if (index == -1) { return; }   // nothing to merge
        // update header
        header_.emplace_back(colname, 0, size, type_id);
	cols_++;
	col_idx_[colname] = index;
	return;
    }

    // output stream
    friend std::ostream& operator<<(std::ostream& os, const scalar_data_layer& data) {
        std::vector<std::vector<std::string>> out;
        std::vector<std::size_t> max_size(data.header().size(), 0);
	int n_rows = std::min(size_t(8), data.rows());
        int n_cols = data.cols();
        if (n_cols == 0) { return os; }   // empty frame, nothing to print
        out.resize(n_cols);
        auto stringify = []<typename T>(const T& t, bool is_na) {
            if (is_na) { return std::string("NA"); }
            if constexpr (!std::is_same_v<T, std::string>) {
                return std::to_string(t);
            } else {
                return t;
            }
        };
        auto print = [&]<typename T>(std::vector<std::string>& out_, const std::string& typestring, const field& desc) {
            out_.push_back(desc.colname());
            std::string info_ = "<" + std::to_string(desc.size()) + ",1:" + typestring + ">";
            out_.push_back(info_);
            // print actual data
            auto col = plain_col_view<T, const scalar_data_layer>(data, 0, n_rows, desc);
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
        for (int i = 0, n = n_cols; i < n; ++i) {
            const auto& desc = data.header()[i];
            dtype coltype = desc.type_id();
            if (coltype == dtype::flt64) { print.template operator()<double      >(out[i], "flt64", desc); }
            if (coltype == dtype::flt32) { print.template operator()<float       >(out[i], "flt32", desc); }
            if (coltype == dtype::int64) { print.template operator()<std::int64_t>(out[i], "int64", desc); }
            if (coltype == dtype::int32) { print.template operator()<std::int32_t>(out[i], "int32", desc); }
            if (coltype == dtype::bin)   { print.template operator()<bool        >(out[i], "bin"  , desc); }
            if (coltype == dtype::str)   { print.template operator()<std::string >(out[i], "str"  , desc); }
        }
        // pretty format (compute maximum number of chars to print)
        for (int i = 0, n = n_cols; i < n; ++i) {
            for (int j = 0, m = n_rows + 2; j < m; ++j) { max_size[i] = std::max(max_size[i], out[i][j].size()); }
        }
	// pad with spaces
        for (int i = 0, n = n_cols; i < n; ++i) {
            for (int j = 0, m = out[i].size(); j < m; ++j) {
                out[i][j].insert(0, max_size[i] - out[i][j].size() + (i == 0 ? 0 : 1), ' ');
            }
            if (data.header()[i].size() > 1) {   // block columns formatting
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
    // extract nan_pattern from column
    void extract_nan_pattern_(const std::string& colname, logical_t& nan_pattern, size_t offset) const {
        field f = header_.at(col_idx_.at(colname));
        internals::dispatch_to_dtype(
          f.type_id(),
          [&]<typename T>(logical_t& dst) mutable {
              dst.block(full_extent, std::pair {offset, offset + f.size() - 1})
                .assign_inplace_from(col<T>(f.colname()).nan());
          },
          nan_pattern);
        return;
    }
    // returns the index of the first typed memory column which can hold a **contiguous** block of blk_sz columns, or -1
    // if there is no such available (already allocated) memory
    template <typename Scalar> index_t find_free_blk_idx_(size_t blk_sz) {
        fdapde_assert(blk_sz > 0);
        dtype type_id = dtype_from_static_type<Scalar>().type_id;
        const std::vector<bool>& freemem = freemem_[type_id];
        int j = 0;
        for (size_t i = 0; i < freemem.size(); ++i) {
            if (freemem[i]) {   // possible candidate point found
                j = i;
                size_t cnt = 0;
                while (i < freemem.size() && freemem[i] && cnt < blk_sz) {
                    cnt++;
                    i++;
                }
                if (cnt == blk_sz) { return j; }
            }
        }
        return -1;
    }
    bool unique_colnames_() const {   // check colnames are unique
        std::unordered_set<std::string> colnames_;
        for (const field& f : header_) { colnames_.insert(f.colname()); }
        return colnames_.size() == header_.size();
    }

    storage_t data_;
    std::vector<field> header_;
    std::unordered_map<std::string, index_t> col_idx_;
    std::unordered_map<dtype, std::vector<bool>> freemem_;
    size_t rows_ = 0, cols_ = 0;
};

// filter outstream
template <typename DataLayer>
std::ostream& operator<<(std::ostream& os, const internals::random_access_row_view<DataLayer>& filter) {
    return operator<<(os, internals::scalar_data_layer(filter));
}
template <typename DataLayer>
std::ostream& operator<<(std::ostream& os, const internals::plain_row_view<DataLayer>& row) {
    return operator<<(os, row.data().select({row.id()}));
}
  
}   // namespace internals
}   // namespace fdapde

#endif // __FDAPDE_DATA_LAYER_H__
