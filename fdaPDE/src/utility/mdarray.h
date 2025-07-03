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

#ifndef __FDAPDE_MDARRAY_H__
#define __FDAPDE_MDARRAY_H__

#include "header_check.h"

namespace fdapde {

static constexpr int full_extent = -2;   // indicates to take the whole extent in subsetting

// forward declarations
template <typename Scalar_, typename Extents_, typename LayoutPolicy_> class MdArray;
template <typename Scalar_, typename Extents_, typename LayoutPolicy_> class MdMap;

namespace internals {

template <typename MdArray> struct md_traits;
template <typename Scalar_, typename Extents_, typename LayoutPolicy_>
struct md_traits<MdArray<Scalar_, Extents_, LayoutPolicy_>> {
    using extents_t = Extents_;
    using index_t = typename extents_t::index_t;
    using order_t = typename extents_t::order_t;
    using size_t  = typename extents_t::size_t;
    using layout_t = LayoutPolicy_;
    using mapping_t = typename layout_t::template mapping<extents_t>;
    using Scalar = Scalar_;
    using storage_t = std::conditional_t<
      extents_t::DynamicOrder != 0, std::vector<Scalar>, std::array<Scalar, std::size_t(extents_t::StaticSize)>>;
    using reference = typename storage_t::reference;
    using const_reference = typename storage_t::const_reference;
};
template <typename Scalar_, typename Extents_, typename LayoutPolicy_>
struct md_traits<MdMap<Scalar_, Extents_, LayoutPolicy_>> {
    using extents_t = Extents_;
    using index_t = typename extents_t::index_t;
    using order_t = typename extents_t::order_t;
    using size_t  = typename extents_t::size_t;
    using layout_t = LayoutPolicy_;
    using mapping_t = typename layout_t::template mapping<extents_t>;
    using Scalar = Scalar_;
    using storage_t = std::add_pointer_t<Scalar_>;
    using reference = std::add_lvalue_reference_t<Scalar>;
    using const_reference = std::add_const_t<reference>;
};

template <typename Extent, typename Idx>
constexpr bool is_index_in_extent(Extent ext, Idx idx)
    requires(std::is_convertible_v<Extent, int> && std::is_convertible_v<Idx, int>) {
    if constexpr (std::is_signed_v<Idx>) {
        if (idx < 0) return false;
    }
    return std::cmp_less(idx, ext);
}
template <typename Extent_, typename... Idxs>
constexpr bool is_multidimensional_index_in_extent(const Extent_& ext, Idxs... idxs) {
    return internals::apply_index_pack<sizeof...(Idxs)>(
      [&]<int... Ns_>() { return (is_index_in_extent(ext.extent(Ns_), idxs) && ...); });
}

template <int N, typename... Ts> constexpr auto ith_element_from_pack(Ts&&... ts) {
    return std::get<N>(std::make_tuple(std::forward<Ts>(ts)...));
}

// finds the Idx-th component of the lowest multidimensional index spanned by slicers...
template <int Idx_, typename... Slicers_> constexpr int smallest_index_in_mdarray_blk(Slicers_&&... slicers) {
    using Slicer = pack_element_t<Idx_, std::decay_t<Slicers_>...>;
    const auto& slicer = ith_element_from_pack<Idx_>(std::forward<Slicers_>(slicers)...);
    if constexpr (internals::is_integer_v<Slicer>) {
        return (slicer == full_extent) ? 0 : slicer;
    } else if constexpr (
      is_pair_v<Slicer> && std::convertible_to<std::tuple_element_t<0, Slicer>, int> &&
      std::convertible_to<std::tuple_element_t<1, Slicer>, int>) {
        fdapde_assert(
          std::cmp_not_equal(std::get<0>(slicer) FDAPDE_COMMA full_extent) &&
          std::cmp_not_equal(std::get<1>(slicer) FDAPDE_COMMA full_extent));
        return static_cast<int>(std::get<0>(slicer));
    }
}
  
// tests for the presence of duplicated entries in index pack
template <typename... IndexType>
    requires(std::is_convertible_v<IndexType, int> && ...) && (sizeof...(IndexType) > 0)
constexpr bool no_duplicates_in_pack(IndexType... idxs) {
    std::array<int, sizeof...(IndexType)> tmp {idxs...};
    for (std::size_t i = 0; i < sizeof...(IndexType) - 1; ++i) {
        for (std::size_t j = i + 1; j < sizeof...(IndexType); ++j) {
            if (tmp[i] == tmp[j]) return false;
        }
    }
    return true;
}

}   // namespace internals

template <int... Extents> class MdExtents {
    fdapde_static_assert(
      sizeof...(Extents) > 0 && internals::apply_index_pack<sizeof...(Extents)>([]<int... Ns_>() {
          return ((static_cast<int>(Extents) > 0 || static_cast<int>(Extents) == Dynamic) && ...);
      }),
      EXTENTS_OF_ZERO_ORDER_OR_WITH_NEGATIVE_VALUES_ARE_ILL_FORMED);
   public:
    // typedefs for integral types used
    using index_t = int;
    using order_t = std::size_t;
    using size_t  = std::size_t;
  
    static constexpr order_t Order = sizeof...(Extents);
    static constexpr order_t DynamicOrder = ((Extents == Dynamic) + ... + 0);
    static constexpr order_t StaticOrder  = Order - DynamicOrder;
    static constexpr size_t  StaticSize   = DynamicOrder == 0 ? (Extents * ... * 1) : Dynamic;
    static constexpr std::array<index_t, Order> static_extents {Extents...};

    constexpr MdExtents() = default;
    template <typename... Exts>
        requires(std::is_convertible_v<Exts, index_t> && ...) && (DynamicOrder > 0 && sizeof...(Exts) == DynamicOrder)
    constexpr MdExtents(Exts... exts) {
        std::array<index_t, DynamicOrder> exts_ {static_cast<index_t>(exts)...};
        for (order_t i = 0; i < DynamicOrder; ++i) { extents_[dynamic_extent_map_[i]] = exts_[i]; }
    }
    template <typename... Exts>
        requires(std::is_convertible_v<Exts, index_t> && ...) &&
                (sizeof...(Exts) != DynamicOrder && sizeof...(Exts) == Order)
    constexpr MdExtents(Exts... exts) {
        std::array<index_t, Order> exts_ {static_cast<index_t>(exts)...};
        for (order_t i = 0, j = 0; i < Order; ++i) {
            if (static_extents[i] == Dynamic) {
                extents_[dynamic_extent_map_[j++]] = exts_[i];
            } else {
                // check supplied extents match static, non-dynamic, ones
                fdapde_constexpr_assert(exts_[i] == static_cast<index_t>(static_extents[i]));
            }
        }
    }
    template <std::size_t Size, typename Exts>
        requires(Size == Order && std::is_convertible_v<Exts, index_t> && Order == DynamicOrder)
    constexpr MdExtents(const std::array<Exts, Size>& extents) : extents_(extents) { }

    // accessors
    constexpr order_t order() const noexcept { return Order; }
    constexpr order_t order_dynamic() const noexcept { return DynamicOrder; }
    constexpr index_t extent(order_t i) const noexcept { return extents_[i]; }
    constexpr index_t& extent(order_t i) noexcept { return extents_[i]; }
    constexpr size_t size() const { return l_prod(Order); }
    template <typename... Exts>
        requires(std::is_convertible_v<Exts, index_t> && ...) && (sizeof...(Exts) == DynamicOrder)
    constexpr void resize(Exts... exts) {
        std::array<index_t, sizeof...(Exts)> exts_ {static_cast<index_t>(exts)...};
        for (size_t i = 0; i < sizeof...(Exts); ++i) { extents_[dynamic_extent_map_[i]] = exts_[i]; }
        return;
    }
    template <typename... Exts>
        requires(std::is_convertible_v<Exts, index_t> && ...) && (sizeof...(Exts) == Order && Order != DynamicOrder)
    constexpr void resize(Exts... exts) {
        std::array<index_t, sizeof...(Exts)> exts_ {static_cast<index_t>(exts)...};
        for (size_t i = 0; i < sizeof...(Exts); ++i) {
            if (static_extents[i] == Dynamic) {
                extents_[dynamic_extent_map_[i]] = exts_[i];
            } else {
                fdapde_constexpr_assert(exts_[i] == static_extents[i]);
            }
        }
        return;
    }
    // given extents (i, j, k, ..., h), computes the product of extents from left to right up to Order r
    constexpr index_t l_prod(order_t r) const requires(Order > 0) {
        index_t l_prod_ = 1;
        for (order_t i = 0; i < r; ++i) { l_prod_ *= extents_[i]; }
        return l_prod_;
    }
    // given extents (i, j, k, ..., h), computes the product of extents from right to left up to Order r
    constexpr index_t r_prod(order_t r) const requires(Order > 0) {
        index_t r_prod_ = 1;
        for (order_t i = r; i < Order - 1; ++i) { r_prod_ *= extents_[i]; }
        return r_prod_;
    }
   private:
    // internal utilities
    static constexpr std::array<index_t, DynamicOrder> dynamic_extent_map_ {[]() {
        std::array<index_t, DynamicOrder> result_ {};
        for (size_t i = 0, j = 0; i < Order; ++i) {
            if (static_extents[i] == Dynamic) { result_[j++] = i; }
        }
        return result_;
    }()};
    std::array<index_t, Order> extents_ {(Extents == Dynamic ? 0 : Extents)...};
};

template <int N>
using full_dynamic_extent_t = std::decay_t<decltype(internals::apply_index_pack<N>(
  []<int... Ns_> { return MdExtents<((void)Ns_, Dynamic)...> {Ns_...}; }))>;

namespace internals {

// memory layout strategies for MdArray (insipred from C++23 std::mdspan<>)

struct layout_left {   // corresponds to a ColMajor storage for order 2 mdarrays

    template <typename Extents> struct mapping {
        using extents_t = Extents;
        using index_t = typename extents_t::index_t;
        using order_t = typename extents_t::order_t;
        using size_t  = typename extents_t::size_t;
        using layout_type = layout_left;    
        static constexpr order_t Order = extents_t::Order;

        constexpr mapping() noexcept = default;
        constexpr mapping(const Extents& extents) noexcept
            requires(Order > 0)
            : extents_(extents), strides_() {
            init_strides_(extents_);
        }
        constexpr mapping(const mapping& other) : extents_(other.extents_), strides_() { init_strides_(extents_); }
        constexpr mapping& operator=(const mapping& other) noexcept {
            extents_ = other.extents_;
            init_strides_(extents_);
	    return *this;
        }

        template <typename... Idxs>   // index pack to mdarray memory index
            requires(std::is_convertible_v<Idxs, index_t> && ...) && (sizeof...(Idxs) == Order)
        constexpr index_t operator()(Idxs... idx) const {
            fdapde_constexpr_assert(
              internals::is_multidimensional_index_in_extent(extents_, static_cast<index_t>(idx)...));
            return internals::apply_index_pack<Order>(
              [&]<int... Ns_>() { return ((static_cast<index_t>(idx) * strides_[Ns_]) + ... + 0); });
        }
        template <typename IndexPack>   // index-pack to memory index
            requires(internals::is_subscriptable<IndexPack, index_t>)
        constexpr index_t operator()(IndexPack&& arr) const {
            index_t idx = 0;
            for (order_t i = 0; i < Order; ++i) { idx += static_cast<index_t>(arr[i]) * strides_[i]; }
            return idx;
        }
        constexpr index_t stride(order_t r) const requires(Order > 0) {
            fdapde_constexpr_assert(r < Order);
	    return strides_[r];
        }
        template <typename OtherExtents>
            requires(Order == OtherExtents::Order)
        friend constexpr bool operator==(const mapping& lhs, const mapping<OtherExtents>& rhs) {
            return lhs.extents_ == rhs.extents_;
        }
       private:
        constexpr void init_strides_(const extents_t& extents) {
            for (order_t i = 0; i < Order + 1; ++i) {
                strides_[i] = 1;
                for (order_t j = 0; j < i; ++j) { strides_[i] *= extents.extent(j); }
            }
        }
        extents_t extents_ {};
        std::array<index_t, Order + 1> strides_ {};
    };
};

struct layout_right {   // corresponds to a RowMajor storage for order 2 mdarrays

    template <typename Extents> struct mapping {
        using extents_t = Extents;
        using index_t = typename extents_t::index_t;
        using order_t = typename extents_t::order_t;
        using size_t  = typename extents_t::size_t;
        using layout_type = layout_right;    
        static constexpr order_t Order = extents_t::Order;

        constexpr mapping() noexcept = default;
        constexpr mapping(const mapping&) noexcept = default;
        constexpr mapping& operator=(const mapping&) noexcept = default;
        constexpr mapping(const Extents& extents) noexcept
            requires(Order > 0)
            : extents_(extents), strides_() {
            for (order_t i = 0; i < Order + 1; ++i) {
                strides_[i] = 1;
                for (order_t j = Order - 1; j > i; --j) { strides_[i] *= extents_.extent(j); }
            }
        }

        template <typename... Idxs>   // index pack to mdarray memory index
            requires(std::is_convertible_v<Idxs, index_t> && ...) && (sizeof...(Idxs) == Order)
        constexpr index_t operator()(Idxs... idx) const {
            fdapde_constexpr_assert(
              internals::is_multidimensional_index_in_extent(extents_, static_cast<index_t>(idx)...));
            return internals::apply_index_pack<Order>(
              [&]<int... Ns_>() { return ((static_cast<index_t>(idx) * strides_[Ns_]) + ... + 0); });
        }
        template <typename IndexPack>   // index-pack to memory index
            requires(internals::is_subscriptable<IndexPack, index_t>)
        constexpr index_t operator()(IndexPack&& arr) const {
            index_t idx = 0;
            for (order_t i = 0; i < Order; ++i) { idx += static_cast<index_t>(arr[i]) * strides_[i]; }
            return idx;
        }
        constexpr index_t stride(order_t r) const requires(Order > 0) {
            fdapde_constexpr_assert(r < Order);
	    return strides_[r];
        }
        template <typename OtherExtents>
            requires(Order == OtherExtents::Order)
        friend constexpr bool operator==(const mapping& lhs, const mapping<OtherExtents>& rhs) {
            return lhs.extents_ == rhs.extents_;
        }
       private:
        Extents extents_ {};
        std::array<index_t, Order + 1> strides_ {};
    };
};

}   // namespace internals
  
// a multidimensional view of a multidimensional MdArray
template <typename MdArray_, typename BlkExtents_> class MdArrayBlock { 
   public:
    using extents_t = BlkExtents_;
    using layout_t  = typename MdArray_::layout_t;
    using mapping_t = typename MdArray_::mapping_t;
    using index_t = typename extents_t::index_t;
    using order_t = typename extents_t::order_t;
    using size_t  = typename extents_t::size_t;
    using reference = typename MdArray_::reference;
    using const_reference = typename MdArray_::const_reference;
    using Scalar  = typename MdArray_::Scalar;
    static constexpr order_t Order = MdArray_::Order;
    static constexpr order_t StaticOrder = extents_t::StaticOrder;
    static constexpr order_t DynamicOrder = extents_t::DynamicOrder;
    static constexpr std::array<index_t, Order> static_extents = extents_t::static_extents;

    constexpr MdArrayBlock() noexcept = default;
    template <typename... Slicers>
        requires(sizeof...(Slicers) == Order && BlkExtents_::Order == MdArray_::Order) &&
                  ((internals::is_integer_v<Slicers> || internals::is_pair_v<Slicers>) && ...)
    constexpr MdArrayBlock(MdArray_* mdarray, BlkExtents_ blk_extents, Slicers&&... slicers) noexcept :
        extents_(blk_extents), mdarray_(mdarray) {
        internals::for_each_index_in_pack<Order>([&]<int Ns_>() mutable {
            offset_[Ns_] = internals::smallest_index_in_mdarray_blk<Ns_>(slicers...);
            if (extents_.extent(Ns_) == full_extent) extents_.extent(Ns_) = mdarray_->extent(Ns_);
        });
    }

    // observers
    constexpr size_t size() const { return extents_.size(); }
    constexpr size_t extent(order_t r) const {
        fdapde_constexpr_assert(r < Order);
        return extents_.extent(r);
    }
    constexpr const extents_t& extents() const { return extents_; }
    constexpr const mapping_t& mapping() const { return mdarray_->mapping(); }
    constexpr order_t order() const { return Order; }
    // pointer to first memory address mapped by this block (NB: blocks are not necessarily contiguous in memory)
    constexpr const Scalar* data() const {
        int off_ = internals::apply_index_pack<Order>(
          [&]<int... Ns_>() { return ((offset_[Ns_] * mdarray_->mapping().stride(Ns_)) + ... + 0); });
        return mdarray_->data() + off_;
    }
    constexpr Scalar* data() requires(!std::is_const_v<MdArray_>) {
        int off_ = internals::apply_index_pack<Order>(
          [&]<int... Ns_>() { return ((offset_[Ns_] * mdarray_->mapping().stride(Ns_)) + ... + 0); });
        return mdarray_->data() + off_;
    }
    // iterator
    template <typename MdArrayBlock_> struct iterator {
        constexpr iterator() noexcept = default;
        template <typename... Idxs>
            requires(std::is_convertible_v<Idxs, index_t> && ...) && (sizeof...(Idxs) == Order)
        constexpr iterator(MdArrayBlock_* mdarray, Idxs... idxs) noexcept :
            mdarray_(mdarray), index_({static_cast<index_t>(idxs)...}) { }
        constexpr iterator(MdArrayBlock_* mdarray, const std::array<index_t, Order>& index) noexcept :
            mdarray_(mdarray), index_(index) { }

        constexpr iterator& operator++() noexcept {
            index_[Order - 1]++;
            order_t i = Order - 1;
            while (i > 0 && std::cmp_greater_equal(index_[i], mdarray_->extent(i))) {
                index_[i] = 0;
                index_[--i]++;
            }
            return *this;
        }
        constexpr const std::array<index_t, Order>& index() const { return index_; }
        constexpr index_t mapped_index() const { return mdarray_->mapping()(index_); }
        // const access
        constexpr const_reference operator*() const { return fetch_at_(index_); }
        constexpr const Scalar* operator->() const { return std::addressof(fetch_at(index_)); }
        // non-const access
        constexpr reference operator*() requires(!std::is_const_v<MdArrayBlock_>) { return fetch_at_(index_); }
        constexpr Scalar* operator->() requires(!std::is_const_v<MdArrayBlock_>) {
            return std::addressof(fetch_at(index_));
        }
        // comparison
        constexpr friend bool operator==(const iterator& lhs, const iterator& rhs) { return lhs.index_ == rhs.index_; }
        constexpr friend bool operator!=(const iterator& lhs, const iterator& rhs) { return lhs.index_ != rhs.index_; }
       private:
        template <typename IndexType> constexpr decltype(auto) fetch_at_(IndexType&& index) {
            return internals::apply_index_pack<Order>(
              [&]<int... Ns_>() -> decltype(auto) { return mdarray_->operator()(((void)Ns_, index[Ns_])...); });
        }
        template <typename IndexType> constexpr decltype(auto) fetch_at_(IndexType&& index) const {
            return internals::apply_index_pack<Order>(
              [&]<int... Ns_>() -> decltype(auto) { return mdarray_->operator()(((void)Ns_, index[Ns_])...); });
        }
        MdArrayBlock_* mdarray_;
        std::array<index_t, Order> index_;
    };
    // const iterators
    constexpr iterator<const MdArrayBlock<MdArray_, BlkExtents_>> begin() const noexcept {
        return internals::apply_index_pack<Order>(
          [&]<int... Ns_> { return iterator<const MdArrayBlock<MdArray_, BlkExtents_>> {this, ((void)Ns_, 0)...}; });
    }
    constexpr iterator<const MdArrayBlock<MdArray_, BlkExtents_>> end() const noexcept {
        return internals::apply_index_pack<Order - 1>([&]<int... Ns_> {
            return iterator<const MdArrayBlock<MdArray_, BlkExtents_>> {this, extents_.extent(0), ((void)Ns_, 0)...};
        });
    }
    // non-const iterators
    constexpr iterator<MdArrayBlock<MdArray_, BlkExtents_>> begin() noexcept {
        return internals::apply_index_pack<Order>(
          [&]<int... Ns_> { return iterator<MdArrayBlock<MdArray_, BlkExtents_>> {this, ((void)Ns_, 0)...}; });
    }
    constexpr iterator<MdArrayBlock<MdArray_, BlkExtents_>> end() noexcept {
        return internals::apply_index_pack<Order - 1>([&]<int... Ns_> {
            return iterator<MdArrayBlock<MdArray_, BlkExtents_>> {this, extents_.extent(0), ((void)Ns_, 0)...};
        });
    }
    // constant access
    template <typename... Idxs>
        requires(std::is_convertible_v<Idxs, index_t> && ...) && (sizeof...(Idxs) == BlkExtents_::Order)
    constexpr const_reference operator()(Idxs... idxs) const {
        fdapde_constexpr_assert(
          internals::is_multidimensional_index_in_extent(extents_, static_cast<index_t>(idxs)...));
        return mdarray_->operator[](internals::apply_index_pack<Order>([&]<int... Ns_>() {
            return (((static_cast<index_t>(idxs) + offset_[Ns_]) * mdarray_->mapping().stride(Ns_)) + ... + 0);
        }));
    }
    template <typename IndexPack>   // access via index-pack object
        requires(internals::is_subscriptable<IndexPack, index_t>)
    constexpr const_reference operator()(IndexPack&& index_pack) const {
        return internals::apply_index_pack<Order>(
          [&]<int... Ns_>() -> decltype(auto) { return operator()(index_pack[Ns_]...); });
    }
    // non-constant access
    template <typename... Idxs>
        requires(std::is_convertible_v<Idxs, index_t> && ...) &&
                (sizeof...(Idxs) == BlkExtents_::Order && !std::is_const_v<MdArray_>)
    constexpr reference operator()(Idxs... idxs) {
        fdapde_constexpr_assert(
          internals::is_multidimensional_index_in_extent(extents_, static_cast<index_t>(idxs)...));
        return mdarray_->operator[](internals::apply_index_pack<Order>([&]<int... Ns_>() {
            return (((static_cast<index_t>(idxs) + offset_[Ns_]) * mdarray_->mapping().stride(Ns_)) + ... + 0);
        }));
    }
    template <typename IndexPack>   // access via index-pack object
        requires(internals::is_subscriptable<IndexPack, index_t>)
    constexpr reference operator()(IndexPack&& index_pack) {
        return internals::apply_index_pack<Order>(
          [&]<int... Ns_>() -> decltype(auto) { return operator()(index_pack[Ns_]...); });
    }
    template <typename Src>
        requires(
#ifdef __FDAPDE_HAS_EIGEN__
          !internals::is_eigen_dense_xpr_v<Src> &&
#endif
          (std::is_pointer_v<Src> || internals::is_subscriptable<Src, int>) &&
          !internals::is_indexable_v<Src, Order, index_t>)
    constexpr MdArrayBlock& assign_inplace_from(Src&& src) {
        if constexpr (!std::is_pointer_v<Src>) { fdapde_assert(src.size() == size()); }
        int i = 0;
        for (reference v : *this) {
            if constexpr (std::is_same_v<Scalar, bool>) {
                if (src[i++]) { v.set(); }
            } else {
                v = src[i++];
            }
        }
        return *this;
    }
    template <typename Src>
        requires(Order == 2 && internals::is_indexable_v<Src, Order, index_t>)
    constexpr MdArrayBlock& assign_inplace_from(Src&& src) {
        fdapde_static_assert(Order == 2, THIS_METHOD_IS_FOR_ORDER_TWO_MDARRAYS_ONLY);
        fdapde_assert(src.rows() == extent(0) && src.cols() == extent(1));
        for (size_t i = 0; i < extent(0); ++i) {
            for (size_t j = 0; j < extent(1); ++j) { operator()(i, j) = src(i, j); }
        }
        return *this;
    }

    template <typename Scalar_, typename Extents_, typename LayoutPolicy_>
        requires(std::is_same_v<Scalar_, Scalar> && Extents_::Order == Order)
    constexpr MdArrayBlock& assign_inplace_from(const MdArray<Scalar_, Extents_, LayoutPolicy_>& src) {
        for (size_t i = 0; i < Order; ++i) { fdapde_assert(extent(i) == src.extent(i)); }
        iterator jt = begin();
        for (auto it = src.begin(); it != src.end(); ++it, ++jt) { *jt = *it; }
        return *this;
    }
    template <typename Dst>
        requires(internals::is_subscriptable<Dst, index_t> || internals::is_indexable_v<Dst, Order, index_t>)
    void assign_to(Dst&& dst) const {
        if constexpr (internals::is_subscriptable<Dst, index_t> && !internals::is_indexable_v<Dst, Order, index_t>) {
            index_t i = 0;
            for (auto& v : *this) { dst[i++] = v; }
        } else {
            for (auto it = begin(); it != end(); ++it) {
                internals::apply_index_pack<Order>([&]<int... Ns_>() { dst(it.index()[Ns_]...) = *it; });
            }
        }
    }
   private:
    std::array<index_t, Order> offset_ {};
    extents_t extents_;
    MdArray_* mdarray_;
};

template <typename MdArray, typename... Slicers>
    requires(sizeof...(Slicers) == std::decay_t<MdArray>::Order && std::is_reference_v<MdArray>) &&
            ((internals::is_integer_v<Slicers> || internals::is_pair_v<Slicers>) && ...)
constexpr auto submdarray(MdArray&& mdarray, Slicers... slicers) {
    using MdArray_ = std::remove_reference_t<MdArray>;
    using index_t  = typename MdArray_::index_t;
    constexpr int Order = std::decay_t<MdArray_>::Order;
    // check block is within MdArray
    internals::for_each_index_and_args<Order>(
      [&]<int Ns_, typename Slicer_>(Slicer_ s) {
          if constexpr (internals::is_integer_v<Slicer_>) {
              fdapde_constexpr_assert(
                std::cmp_equal(s FDAPDE_COMMA full_extent) ||
                (s >= 0 && std::cmp_less(s FDAPDE_COMMA mdarray.extent(Ns_))));
          } else if constexpr (internals::is_pair_v<Slicer_>) {
              fdapde_constexpr_assert(
                std::cmp_not_equal(std::get<0>(s) FDAPDE_COMMA full_extent) &&
                std::cmp_not_equal(std::get<1>(s) FDAPDE_COMMA full_extent) && std::get<1>(s) >= std::get<0>(s) &&
                std::cmp_less(std::get<1>(s) FDAPDE_COMMA mdarray.extent(Ns_)));
          }
      },
      slicers...);
    // prepare block extents
    std::array<index_t, Order> blk_extents {};
    internals::for_each_index_and_args<Order>(
      [&]<int Ns_, typename Slicer_>(Slicer_ s) mutable {
          if constexpr (internals::is_integer_v<Slicer_>) {
              blk_extents[Ns_] = (std::cmp_equal(s, full_extent) ? mdarray.extent(Ns_) : 1);
          } else if constexpr (internals::is_pair_v<Slicer_>) {
              blk_extents[Ns_] = 1 + (std::get<1>(s) - std::get<0>(s));
          }
      },
      slicers...);
    return MdArrayBlock<MdArray_, full_dynamic_extent_t<Order>>(
      &mdarray, full_dynamic_extent_t<Order>(blk_extents), slicers...);
}

namespace internals {

// check if slicing the mdarray with slicers Slicers... brings to a strided or contiguous memory access
template <typename mapping, int... Slicers>
    requires(sizeof...(Slicers) > 0 && sizeof...(Slicers) < mapping::Order)
consteval bool slices_to_contiguous_memory() {
    std::array<int, sizeof...(Slicers)> slicers_ {Slicers...};
    std::sort(slicers_.begin(), slicers_.end());
    int i = std::is_same_v<typename mapping::layout_type, layout_right> ? 0 : mapping::Order - 1 - sizeof...(Slicers);
    for (size_t j = 0; j < sizeof...(Slicers); ++j) {
        if (slicers_[j] != (std::is_same_v<typename mapping::layout_type, layout_right> ? i++ : ++i)) return false;
    }
    return true;
}

}   // namespace internals

// a slice is a lower-order view of an MdArray in which one or more dimensions have been fixed
template <typename MdArray, int... Slicers> class MdArraySlice {
    static_assert(sizeof...(Slicers) < MdArray::Order && ((Slicers >= 0 && Slicers < MdArray::Order) && ...));
   public:
    using extents_t = typename MdArray::extents_t;
    using mapping_t = typename MdArray::mapping_t;
    using index_t = typename extents_t::index_t;
    using order_t = typename extents_t::order_t;
    using size_t  = typename extents_t::size_t;
    using reference = typename MdArray::reference;
    using const_reference = typename MdArray::const_reference;
    static constexpr order_t Order = MdArray::Order - sizeof...(Slicers);
    static constexpr order_t DynamicOrder =
      MdArray::DynamicOrder - ((MdArray::static_extents[Slicers] == Dynamic) + ... + 0);
    static constexpr bool contiguous_access =
      internals::slices_to_contiguous_memory<typename MdArray::mapping_t, Slicers...>();

    static constexpr std::array<index_t, sizeof...(Slicers)> static_slicers {Slicers...};
    fdapde_static_assert(internals::no_duplicates_in_pack(Slicers...), SLICING_DIRECTIONS_MUST_BE_UNIQUE);
    // free_extents_idxs_ = {0, \ldots, MdArray_::Order - 1} \setminus static_slicers
    static constexpr std::array<index_t, Order> free_extents_idxs_ {[]() {
        std::array<index_t, Order> map {};
        for (order_t i = 0, j = 0; i < MdArray::Order; ++i) {
            if (std::find(static_slicers.begin(), static_slicers.end(), i) == static_slicers.end()) { map[j++] = i; }
        }
        return map;
    }()};
    static constexpr std::array<index_t, DynamicOrder> free_dynamic_extents_idxs_ {[]() {
        if constexpr (DynamicOrder == 0) {
            return std::array<index_t, DynamicOrder> {};
        } else {
            std::array<index_t, DynamicOrder> map {};
            for (order_t i = 0, j = 0; i < MdArray::Order; ++i) {
                if (
                  std::find(static_slicers.begin(), static_slicers.end(), i) == static_slicers.end() &&
                  MdArray::static_extents[i] == Dynamic) {
                    map[j++] = i;
                }
            }
            return map;
        }
    }()};
    using Scalar = typename MdArray::Scalar;

    constexpr MdArraySlice() noexcept = default;
    constexpr explicit MdArraySlice(MdArray* mdarray) noexcept : internal_stride_(), offset_(0), mdarray_(mdarray) { }
    template <typename... Slicers_>
        requires(sizeof...(Slicers_) == sizeof...(Slicers)) && (std::is_convertible_v<Slicers_, index_t> && ...)
    constexpr MdArraySlice(MdArray* mdarray, Slicers_... slicers) : internal_stride_(), offset_(0), mdarray_(mdarray) {
        internals::for_each_index_and_args<sizeof...(Slicers_)>(
          [&]<int Ns_, typename Slicer__>(Slicer__ s) {
              fdapde_constexpr_assert(std::cmp_less(s, mdarray_->extent(static_slicers[Ns_])));
          },
          slicers...);
        // compute offset in linearized memory due to slicing
        internals::for_each_index_and_args<sizeof...(Slicers_)>(
          [&]<int Ns_, typename Slicer__>(Slicer__ s) {
              offset_ += static_cast<index_t>(s) * mdarray_->mapping().stride(static_slicers[Ns_]);
          },
          slicers...);
        // compute internal strides
        for (order_t i = 0; i < Order; ++i) { internal_stride_[i] = mdarray_->mapping().stride(free_extents_idxs_[i]); }
    }

    // observers
    constexpr size_t size() const {
        size_t size_ = 1;
        for (order_t i = 0; i < Order; ++i) { size_ *= extent(i); }
        return size_;
    }
    constexpr size_t extent(order_t r) const {
        fdapde_constexpr_assert(r < Order);
        return mdarray_->extent(free_extents_idxs_[r]);
    }
    // iterator
    template <typename MdArraySlice_> struct iterator {
        constexpr iterator() noexcept = default;
        template <typename... Idxs>
            requires(std::is_convertible_v<Idxs, index_t> && ...) && (sizeof...(Idxs) == Order)
        constexpr iterator(MdArraySlice_* mdarray, Idxs... idxs) noexcept :
            mdarray_(mdarray), index_({static_cast<index_t>(idxs)...}) { }

        constexpr iterator& operator++() noexcept {
            index_[Order - 1]++;
            order_t i = Order - 1;
            while (i > 0 && index_[i] >= mdarray_->extent(i)) {
                index_[i] = 0;
                index_[--i]++;
            }
            return *this;
        }
        constexpr const std::array<index_t, Order>& index() const { return index_; }
        constexpr index_t mapped_index() const { return mdarray_->mapping()(index_); }
        // const access
        constexpr const_reference operator*() const { return fetch_at_(index_); }
        constexpr const Scalar* operator->() const { return std::addressof(fetch_at(index_)); }
        // non-const access
        constexpr reference operator*() requires(!std::is_const_v<MdArraySlice_>) { return fetch_at_(index_); }
        constexpr Scalar* operator->()  requires(!std::is_const_v<MdArraySlice_>) {
            return std::addressof(fetch_at(index_));
        }
        // comparison
        constexpr friend bool operator==(const iterator& lhs, const iterator& rhs) { return lhs.index_ == rhs.index_; }
        constexpr friend bool operator!=(const iterator& lhs, const iterator& rhs) { return lhs.index_ != rhs.index_; }
       private:
        template <typename IndexType> constexpr decltype(auto) fetch_at_(IndexType&& index) {
            return internals::apply_index_pack<Order>(
              [&]<int... Ns_>() -> decltype(auto) { return mdarray_->operator()(((void)Ns_, index[Ns_])...); });
        }
        template <typename IndexType> constexpr decltype(auto) fetch_at_(IndexType&& index) const {
            return internals::apply_index_pack<Order>(
              [&]<int... Ns_>() -> decltype(auto) { return mdarray_->operator()(((void)Ns_, index[Ns_])...); });
        }
        MdArraySlice_* mdarray_;
        std::array<index_t, Order> index_;
    };
    // const iterators
    constexpr iterator<const MdArraySlice<MdArray, Slicers...>> begin() const noexcept {
        return internals::apply_index_pack<Order>([&]<int... Ns_> {
            return iterator<const MdArraySlice<MdArray, Slicers...>> {this, ((void)Ns_, 0)...};
        });
    }
    constexpr iterator<const MdArraySlice<MdArray, Slicers...>> end() const noexcept {
        return internals::apply_index_pack<Order - 1>([&]<int... Ns_> {
            return iterator<const MdArraySlice<MdArray, Slicers...>> {this, extent(0), ((void)Ns_, 0)...};
        });
    }
    // non-const iterators
    constexpr iterator<MdArraySlice<MdArray, Slicers...>> begin() noexcept {
        return internals::apply_index_pack<Order>(
          [&]<int... Ns_> { return iterator<MdArraySlice<MdArray, Slicers...>> {this, ((void)Ns_, 0)...}; });
    }
    constexpr iterator<MdArraySlice<MdArray, Slicers...>> end() noexcept {
        return internals::apply_index_pack<Order - 1>(
          [&]<int... Ns_> { return iterator<MdArraySlice<MdArray, Slicers...>> {this, extent(0), ((void)Ns_, 0)...}; });
    }
    // modifiers
    template <typename... Slicers_>
        requires(sizeof...(Slicers_) == sizeof...(Slicers)) && (std::is_convertible_v<Slicers_, index_t> && ...)
    void move(Slicers_... slicers) {
        // update offset in linearized memory due to slicing
        internals::for_each_index_and_args<sizeof...(Slicers_)>(
          [&]<int Ns_, typename Slicer__>(Slicer__ s) {
              offset_ += static_cast<index_t>(s) * mdarray_->mapping().stride(static_slicers[Ns_]);
          },
          slicers...);
        return;
    }
    // constant access
    template <typename... Idxs>
        requires(std::is_convertible_v<Idxs, index_t> && ...) && (sizeof...(Idxs) == Order)
    constexpr const_reference operator()(Idxs... idxs) const {
        internals::for_each_index_and_args<Order>(
          [&]<int Ns_, typename Slicer__>(Slicer__ s) {
              fdapde_constexpr_assert(s < mdarray_->extent(free_extents_idxs_[Ns_]));
          },
          idxs...);
        return mdarray_->operator[](internals::apply_index_pack<Order>(
          [&]<int... Ns_>() { return ((static_cast<index_t>(idxs) * internal_stride_[Ns_]) + ... + offset_); }));
    }
    template <typename IndexPack>   // access via index-pack object
        requires(internals::is_subscriptable<IndexPack, index_t>)
    constexpr const_reference operator()(IndexPack&& index_pack) const {
        return internals::apply_index_pack<Order>(
          [&]<int... Ns_>() -> decltype(auto) { return operator()(index_pack[Ns_]...); });
    }
    constexpr const_reference operator[](int index) const { return mdarray_->operator[](offset_ + index); }
    // non-constant access
    template <typename... Idxs>
        requires(std::is_convertible_v<Idxs, index_t> && ...) && (sizeof...(Idxs) == Order && !std::is_const_v<MdArray>)
    constexpr reference operator()(Idxs... idxs) {
        internals::for_each_index_and_args<Order>(
          [&]<int Ns_, typename Slicer__>(Slicer__ s) {
              fdapde_constexpr_assert(s < static_cast<index_t>(mdarray_->extent(free_extents_idxs_[Ns_])));
          },
          idxs...);
        return mdarray_->operator[](internals::apply_index_pack<Order>(
          [&]<int... Ns_>() { return ((static_cast<index_t>(idxs) * internal_stride_[Ns_]) + ... + offset_); }));
    }
    template <typename IndexPack>   // access via index-pack object
        requires(internals::is_subscriptable<IndexPack, index_t>)
    constexpr reference operator()(IndexPack&& index_pack) {
        return internals::apply_index_pack<Order>(
          [&]<int... Ns_>() -> decltype(auto) { return operator()(index_pack[Ns_]...); });
    }
    constexpr reference operator[](int index) { return mdarray_->operator[](offset_ + index); }
    constexpr auto as_matrix() const {
        static_assert((Order == 2 || Order == 1) && ((Slicers != Dynamic) && ...));
        constexpr int rows = MdArray::static_extents[free_extents_idxs_[0]];
        if constexpr (Order == 2) {
            constexpr int cols = MdArray::static_extents[free_extents_idxs_[1]];
            fdapde_static_assert(rows != Dynamic && cols != Dynamic, THIS_METHOD_IS_FOR_STATIC_SIZED_MDARRAYS_ONLY);
            Matrix<Scalar, rows, cols> mtx {};
            for (size_t i = 0, n = rows; i < n; ++i) {
                for (size_t j = 0, n = cols; j < n; ++j) { mtx(i, j) = operator()(i, j); }
            }
            return mtx;
        } else {
            fdapde_static_assert(rows != Dynamic, THIS_METHOD_IS_FOR_STATIC_SIZED_MDARRAYS_ONLY);
            Vector<Scalar, rows> vec {};
            for (size_t i = 0, n = rows; i < n; ++i) { vec[i] = operator()(i); }
            return vec;
        }
    }
#ifdef __FDAPDE_HAS_EIGEN__
   private:
    template <typename Ptr_>
        requires(std::is_pointer_v<std::decay_t<Ptr_>>)
    constexpr auto as_eigen_map_(Ptr_ ptr) const {
        static_assert(contiguous_access && (Order == 2 || Order == 1) && ((Slicers != Dynamic) && ...));
        using Scalar_ = std::remove_pointer_t<Ptr_>;
        constexpr int rows = MdArray::static_extents[free_extents_idxs_[0]];
        if constexpr (Order == 2) {
            constexpr int cols = MdArray::static_extents[free_extents_idxs_[1]];
            constexpr int storage_layout =
              std::is_same_v<typename mapping_t::layout_type, internals::layout_right> ? ColMajor : RowMajor;
            using matrix_t = Eigen::Matrix<std::decay_t<Scalar_>, rows, cols, storage_layout>;
            return Eigen::Map<std::conditional_t<std::is_const_v<Scalar_>, const matrix_t, matrix_t>>(
              ptr, extent(0), extent(1));
        } else {
	  using vector_t = Eigen::Matrix<std::decay_t<Scalar_>, rows, 1>;
            return Eigen::Map<std::conditional_t<std::is_const_v<Scalar_>, const vector_t, vector_t>>(
              data(), extent(0), 1);
        }
    }
   public:
    constexpr auto as_eigen_map() const { return as_eigen_map_(data()); }
    constexpr auto as_eigen_map() { return as_eigen_map_(data()); }
#endif
    constexpr const Scalar* data() const requires(contiguous_access) { return mdarray_->data() + offset_; }
    constexpr Scalar* data() requires(!std::is_const_v<MdArray> && contiguous_access) {
        return mdarray_->data() + offset_;
    }
    template <typename Src>
        requires(std::is_pointer_v<Src> || internals::is_vector_like_v<Src>)
    constexpr MdArraySlice& assign_inplace_from(Src&& src) {
        if constexpr (!std::is_pointer_v<Src>) fdapde_assert(src.size() == size());
        if constexpr (contiguous_access) {
            // for pointer types, this could lead to ub. is caller responsibility to guarantee bounded access
            for (int i = 0, n = size(); i < n; ++i) { operator[](i) = src[i]; }
        } else {
            int i = 0;
            if constexpr (std::is_same_v<Scalar, bool>) {
                for (auto v : *this) {
                    if (src[i++]) { v.set(); }
                }
            } else {
                for (auto& v : *this) { v = src[i++]; }
            }
        }
        return *this;
    }
    template <typename Src>
        requires(
          Order == 2 && !std::is_pointer_v<Src> && !internals::is_vector_like_v<Src> &&
          internals::is_indexable_v<Src, Order, index_t>)
    constexpr MdArraySlice& assign_inplace_from(Src&& other) {
        fdapde_static_assert(Order == 2, THIS_METHOD_IS_FOR_ORDER_TWO_MDARRAYS_ONLY);
        fdapde_assert(other.rows() == extent(0) && other.cols() == extent(1));
        for (size_t i = 0; i < extent(0); ++i) {
            for (size_t j = 0; j < extent(1); ++j) { operator()(i, j) = other(i, j); }
        }
        return *this;
    }
    template <typename MdArray_, int... Slicers_>
        requires(std::is_same_v<typename MdArray::Scalar, typename MdArray_::Scalar>)
    constexpr MdArraySlice& assign_inplace_from(const MdArraySlice<MdArray_, Slicers_...>& src) {
        fdapde_assert(size() == src.size());
        for (auto it = src.begin(); it != src.end(); ++it) {
            if constexpr (std::is_same_v<Scalar, bool>) {
                if (*it) { operator()(it.index()).set(); }
            } else {
                operator()(it.index()) = *it;
            }
        }
        return *this;
    }
    template <typename Dst>
        requires(internals::is_subscriptable<Dst, index_t> || internals::is_indexable_v<Dst, Order, index_t>)
    void assign_to(Dst&& dst) const {
        if constexpr (contiguous_access) {
            for (int i = 0, n = size(); i < n; ++i) { dst[i] = operator[](i); }
        } else {
            if constexpr (
              internals::is_subscriptable<Dst, index_t> && !internals::is_indexable_v<Dst, Order, index_t>) {
                index_t i = 0;
                for (auto& v : *this) { dst[i++] = v; }
            } else {
                for (auto it = begin(); it != end(); ++it) {
                    internals::apply_index_pack<Order>([&]<int... Ns_>() { dst(it.index()[Ns_]...) = *it; });
                }
            }
        }
    }
   private:
    std::array<index_t, Order> internal_stride_;
    index_t offset_ = 0;
    MdArray* mdarray_;
};

namespace internals {

template <typename Derived> class md_handler_base {
   public:
    using extents_t = typename md_traits<Derived>::extents_t;
    using layout_t  = typename md_traits<Derived>::layout_t;
    using storage_t = typename md_traits<Derived>::storage_t;
    using reference = typename md_traits<Derived>::reference;
    using const_reference = typename md_traits<Derived>::const_reference;
    using Scalar  = typename md_traits<Derived>::Scalar;
    using index_t = typename extents_t::index_t;
    using order_t = typename extents_t::order_t;
    using size_t  = typename extents_t::size_t;
    using mapping_t = typename md_traits<Derived>::mapping_t;
    static constexpr order_t Order = extents_t::Order;
    static constexpr order_t DynamicOrder = extents_t::DynamicOrder;
    static constexpr std::array<index_t, Order> static_extents = extents_t::static_extents;

    constexpr md_handler_base()
        requires(std::is_default_constructible_v<extents_t> && std::is_default_constructible_v<storage_t>)
        : extents_(), mapping_(extents_) { }
    template <typename... Exts_>
        requires(extents_t::DynamicOrder != 0 && extents_t::DynamicOrder == sizeof...(Exts_)) &&
                  (std::is_convertible_v<Exts_, index_t> && ...)
    constexpr md_handler_base(Exts_... exts) : extents_(static_cast<index_t>(exts)...), mapping_(extents_) { }
    template <typename Extents_, typename Mapping_>
        requires(std::is_constructible_v<extents_t, Extents_> && std::is_constructible_v<mapping_t, Mapping_>)
    constexpr md_handler_base(const Extents_& extents, const Mapping_& mapping) : extents_(extents), mapping_(mapping) {
        if constexpr (extents_t::StaticOrder > 0) {
            for (int i = 0; i < extents_t::Order; ++i) {
                fdapde_constexpr_assert(
                  extents_t::static_extents[i] == Dynamic ||
                  extents_t::static_extents[i] == Extents_::static_extents[i]);
            }
        }
    }
    template <typename Extents_>
        requires(
          !std::is_convertible_v<Extents_, index_t> && std::is_constructible_v<extents_t, Extents_> &&
          std::is_constructible_v<mapping_t, Extents_>)
    constexpr md_handler_base(const Extents_& extents) : md_handler_base(extents, mapping_t(extents)) { }

    constexpr const Derived& derived() const { return static_cast<const Derived&>(*this); }
    constexpr Derived& derived() { return static_cast<Derived&>(*this); }
    // assignment
    template <typename Src>
        requires(std::is_pointer_v<Src> || internals::is_subscriptable<Src, int>)
    constexpr Derived& assign_inplace_from(const Src& other) {
        for (int i = 0, n = size(); i < n; ++i) { derived().data()[i] = other[i]; }
        return derived();
    }

    // observers
    constexpr size_t size() const { return extents_.size(); }
    constexpr size_t extent(order_t r) const { return extents_.extent(r); }
    constexpr const extents_t& extents() const { return extents_; }
    constexpr size_t rows() const {
        fdapde_static_assert(Order == 1 || Order == 2, THIS_METHOD_IS_FOR_MATRIX_LIKE_MDARRAYS_ONLY);
        return extent(0);
    }
    constexpr size_t cols() const {
        fdapde_static_assert(Order == 1 || Order == 2, THIS_METHOD_IS_FOR_MATRIX_LIKE_MDARRAYS_ONLY);
        return Order == 1 ? 1 : extent(1);
    }  
    constexpr const mapping_t& mapping() const { return mapping_; }
    // iterator
    template <typename Derived_> struct iterator {
        constexpr iterator() noexcept = default;
        template <typename... Idxs>
            requires(std::is_convertible_v<Idxs, index_t> && ...) && (sizeof...(Idxs) == Order)
        constexpr iterator(Derived_* mdarray, Idxs... idxs) noexcept :
            mdarray_(mdarray), index_({static_cast<index_t>(idxs)...}) { }
        constexpr iterator(Derived_* mdarray, const std::array<index_t, Order>& index) noexcept :
            mdarray_(mdarray), index_(index) { }

        constexpr iterator& operator++() noexcept {
            index_[Order - 1]++;
            order_t i = Order - 1;
            while (i > 0 && std::cmp_greater_equal(index_[i], mdarray_->extent(i))) {
                index_[i] = 0;
                index_[--i]++;
            }
            return *this;
        }
        const std::array<index_t, Order>& index() const { return index_; }
        index_t mapped_index() const { return mdarray_->mapping()(index_); }
        // const access
        constexpr const_reference operator*() const { return fetch_at_(index_); }
        constexpr const Scalar* operator->() const { return std::addressof(fetch_at_(index_)); }
        // non-const access
        constexpr reference operator*() requires(!std::is_const_v<Derived_>) { return fetch_at_(index_); }
        constexpr Scalar* operator->()  requires(!std::is_const_v<Derived_>) {
            return std::addressof(fetch_at_(index_));
        }
        // comparison
        constexpr friend bool operator==(const iterator& lhs, const iterator& rhs) { return lhs.index_ == rhs.index_; }
        constexpr friend bool operator!=(const iterator& lhs, const iterator& rhs) { return lhs.index_ != rhs.index_; }
       private:
        template <typename IndexType> constexpr decltype(auto) fetch_at_(IndexType&& index) {
            return internals::apply_index_pack<Order>(
              [&]<int... Ns_>() -> decltype(auto) { return mdarray_->operator()(((void)Ns_, index[Ns_])...); });
        }
        template <typename IndexType> constexpr decltype(auto) fetch_at_(IndexType&& index) const {
            return internals::apply_index_pack<Order>(
              [&]<int... Ns_>() -> decltype(auto) { return mdarray_->operator()(((void)Ns_, index[Ns_])...); });
        }
        Derived_* mdarray_;
        std::array<index_t, Order> index_;
    };
    // const iterators
    constexpr iterator<const Derived> begin() const noexcept {
        return internals::apply_index_pack<Order>(
          [&]<int... Ns_> { return iterator<const Derived> {std::addressof(derived()), ((void)Ns_, 0)...}; });
    }
    constexpr iterator<const Derived> end() const noexcept {
        return internals::apply_index_pack<Order - 1>([&]<int... Ns_> {
            return iterator<const Derived> {std::addressof(derived()), extents_.extent(0), ((void)Ns_, 0)...};
        });
    }
    // non-const iterators
    constexpr iterator<Derived> begin() noexcept {
        return internals::apply_index_pack<Order>(
          [&]<int... Ns_> { return iterator<Derived> {std::addressof(derived()), ((void)Ns_, 0)...}; });
    }
    constexpr iterator<Derived> end() noexcept {
        return internals::apply_index_pack<Order - 1>([&]<int... Ns_> {
            return iterator<Derived> {std::addressof(derived()), extents_.extent(0), ((void)Ns_, 0)...};
        });
    }
    // constant access
    template <typename... Idxs>
        requires(std::is_convertible_v<Idxs, index_t> && ...) && (sizeof...(Idxs) == extents_t::Order)
    constexpr const_reference operator()(Idxs... idxs) const {
        return derived().data()[mapping_(static_cast<index_t>(idxs)...)];
    }
    template <typename IndexPack>   // access via index-pack object
        requires(internals::is_subscriptable<IndexPack, index_t>)
    constexpr const_reference operator()(IndexPack&& index_pack) const {
        return internals::apply_index_pack<Order>(
          [&]<int... Ns_>() -> decltype(auto) { return operator()(index_pack[Ns_]...); });
    }
    constexpr const_reference operator[](index_t i) const { return derived().data()[i]; }
    // non-constant access
    template <typename... Idxs>
        requires(std::is_convertible_v<Idxs, index_t> && ...) && (sizeof...(Idxs) == extents_t::Order)
    constexpr reference operator()(Idxs... idxs) {
        return derived().data()[mapping_(static_cast<index_t>(idxs)...)];
    }
    template <typename IndexPack>   // access via index-pack object
        requires(internals::is_subscriptable<IndexPack, index_t>)
    constexpr reference operator()(IndexPack&& index_pack) {
        return internals::apply_index_pack<Order>(
          [&]<int... Ns_>() -> decltype(auto) { return operator()(index_pack[Ns_]...); });
    }
    constexpr reference operator[](index_t i) { return derived().data()[i]; }
    // block-access operations
    template <typename... Slicers_>   // dynamic-sized
        requires(sizeof...(Slicers_) == Order) &&
                ((internals::is_integer_v<Slicers_> || internals::is_pair_v<Slicers_>) && ...)
    constexpr auto block(Slicers_... slicers) {
        return submdarray(derived(), slicers...);
    }
    template <typename IndexPack>   // access via index-pack object
        requires(internals::is_subscriptable<IndexPack, index_t>)
    constexpr auto block(IndexPack&& lower_index, IndexPack&& upper_index) {
        return internals::apply_index_pack<Order>(
          [&]<int... Ns_> { return submdarray(derived(), std::make_pair(lower_index[Ns_], upper_index[Ns_])...); });
    }
    template <typename... Slicers_>   // dynamic-sized
        requires(sizeof...(Slicers_) == Order) &&
                ((internals::is_integer_v<Slicers_> || internals::is_pair_v<Slicers_>) && ...)
    constexpr auto block(Slicers_... slicers) const {
        return submdarray(derived(), slicers...);
    }
    template <typename IndexPack>   // access via index-pack object
        requires(internals::is_subscriptable<IndexPack, index_t>)
    constexpr auto block(IndexPack&& lower_index, IndexPack&& upper_index) const {
        return internals::apply_index_pack<Order>(
          [&]<int... Ns_> { return submdarray(derived(), std::make_pair(lower_index[Ns_], upper_index[Ns_])...); });
    }
    template <int... Exts_, typename... Slicers_>   // static-sized (const access)
        requires(sizeof...(Exts_) == Order && sizeof...(Exts_) == sizeof...(Slicers_)) &&
                (internals::is_integer_v<Slicers_> && ...)
    constexpr MdArrayBlock<const Derived, MdExtents<Exts_...>> block(Slicers_... slicers) const {
        // check block is within MdArray
        std::array<index_t, Order> static_block_extents {Exts_...};
        internals::for_each_index_and_args<Order>(
          [&]<int Ns_, typename Slicer__>(Slicer__ s) {
              fdapde_constexpr_assert(static_block_extents[Ns_] + s < 1 + extent(Ns_));
          },
          slicers...);
        return MdArrayBlock<const Derived, MdExtents<Exts_...>>(
          std::addressof(derived()), MdExtents<Exts_...>(), slicers...);
    }
    template <int... Exts_, typename... Slicers_>   // static-sized (non-const access)
        requires(sizeof...(Exts_) == Order && sizeof...(Exts_) == sizeof...(Slicers_)) &&
                (internals::is_integer_v<Slicers_> && ...)
    constexpr MdArrayBlock<Derived, MdExtents<Exts_...>> block(Slicers_... slicers) {
        // check block is within MdArray
        std::array<index_t, Order> static_block_extents {Exts_...};
        internals::for_each_index_and_args<Order>(
          [&]<int Ns_, typename Slicer__>(Slicer__ s) {
              fdapde_constexpr_assert(static_block_extents[Ns_] + s < 1 + extent(Ns_));
          },
          slicers...);
        return MdArrayBlock<Derived, MdExtents<Exts_...>>(std::addressof(derived()), MdExtents<Exts_...>(), slicers...);
    }
    // special matrix-like accessors
    constexpr auto row(index_t i) {
        fdapde_static_assert(Order == 1 || Order == 2, THIS_METHOD_IS_FOR_MATRIX_LIKE_MDARRAYS_ONLY);
	fdapde_assert(i >= 0 && std::cmp_less(i FDAPDE_COMMA extent(0)));
        return block(i, full_extent);
    }
    constexpr auto row(index_t i) const {
        fdapde_static_assert(Order == 1 || Order == 2, THIS_METHOD_IS_FOR_MATRIX_LIKE_MDARRAYS_ONLY);
	fdapde_assert(i >= 0 && std::cmp_less(i FDAPDE_COMMA extent(0)));
        return block(i, full_extent);
    }
    constexpr auto col(index_t i) {
        fdapde_static_assert(Order == 1 || Order == 2, THIS_METHOD_IS_FOR_MATRIX_LIKE_MDARRAYS_ONLY);
	fdapde_assert(i >= 0 && std::cmp_less(i FDAPDE_COMMA extent(1)));
        return block(full_extent, i);
    }
    constexpr auto col(index_t i) const {
        fdapde_static_assert(Order == 1 || Order == 2, THIS_METHOD_IS_FOR_MATRIX_LIKE_MDARRAYS_ONLY);
	fdapde_assert(i >= 0 && std::cmp_less(i FDAPDE_COMMA extent(1)));
        return block(full_extent, i);
    }
    // slicing operations
    template <int... Slicers, typename... Slicers__>
        requires(std::is_convertible_v<Slicers__, int> && ...)
    constexpr auto slice(Slicers__... slicers) {
        return MdArraySlice<Derived, Slicers...>(std::addressof(derived()), slicers...);
    }
    template <int... Slicers, typename IndexPack>
        requires(internals::is_subscriptable<IndexPack, index_t>)
    constexpr auto slice(IndexPack&& index_pack) {
        return internals::apply_index_pack<sizeof...(Slicers)>(
          [&]<int... Ns_> { return slice<Slicers...>(index_pack[Ns_]...); });
    }
    template <int... Slicers, typename... Slicers__>
        requires(std::is_convertible_v<Slicers__, int> && ...)
    constexpr auto slice(Slicers__... slicers) const {
        return MdArraySlice<const Derived, Slicers...>(std::addressof(derived()), slicers...);
    }
    template <int... Slicers, typename IndexPack>
        requires(internals::is_subscriptable<IndexPack, index_t>)
    constexpr auto slice(IndexPack&& index_pack) const {
        return internals::apply_index_pack<sizeof...(Slicers)>(
          [&]<int... Ns_> { return slice<Slicers...>(index_pack[Ns_]...); });
    }

    constexpr auto as_matrix() const {
        fdapde_static_assert(Order == 2 || Order == 1, THIS_METHOD_IS_FOR_MDARRAYS_OF_ORDER_ONE_OR_TWO_ONLY);
        if constexpr (Order == 2) {
            fdapde_static_assert(
              static_extents[0] != Dynamic && static_extents[1] != Dynamic, THIS_METHOD_IS_FOR_STATIC_EXTENTS_ONLY);
            constexpr int storage_layout =
              std::is_same_v<typename mapping_t::layout_type, internals::layout_right> ? ColMajor : RowMajor;
            Map<const Scalar, static_extents[0], static_extents[1], storage_layout> map(
              derived().data(), extent(0), extent(1));
            return map;
        } else {
            fdapde_static_assert(static_extents[0] != Dynamic, THIS_METHOD_IS_FOR_STATIC_EXTENTS_ONLY);
            Map<const Scalar, static_extents[0], 1, RowMajor> map(derived().data(), extent(0), 1);
            return map;
        }
    }
#ifdef __FDAPDE_HAS_EIGEN__
    constexpr auto as_eigen_map() const {
        fdapde_static_assert(Order == 2 || Order == 1, THIS_METHOD_IS_FOR_MDARRAYS_OF_ORDER_ONE_OR_TWO_ONLY);
        if constexpr (Order == 2) {
            constexpr int storage_layout =
              std::is_same_v<typename mapping_t::layout_type, internals::layout_right> ? ColMajor : RowMajor;
            Eigen::Map<const Eigen::Matrix<Scalar, static_extents[0], static_extents[1], storage_layout>> map(
              derived().data(), extent(0), extent(1));
            return map;
        } else {
            Eigen::Map<const Eigen::Matrix<Scalar, static_extents[0], 1>> map(derived().data(), extent(0), 1);
            return map;
        }
    }
#endif
   protected:  
    extents_t extents_ {};
    mapping_t mapping_ {};
};

}   // namespace internals

// MdArray: an owning C++20 multidimensional array.
// If Extents_ is such that Extents_::DynamicOrder == 0, MdArray can be used in a constexpr context.
template <typename Scalar_, typename Extents_, typename LayoutPolicy_ = internals::layout_right>
class MdArray : public internals::md_handler_base<MdArray<Scalar_, Extents_, LayoutPolicy_>> {
    using Base   = internals::md_handler_base<MdArray<Scalar_, Extents_, LayoutPolicy_>>;
    using traits = internals::md_traits<MdArray<Scalar_, Extents_, LayoutPolicy_>>;
   public:
    using Scalar = Scalar_;
    using layout_t = LayoutPolicy_;
    using index_t = typename traits::index_t;
    using order_t = typename traits::order_t;
    using size_t  = typename traits::size_t;
    using extents_t = Extents_;
    using mapping_t = typename traits::mapping_t;
    using storage_t = typename traits::storage_t;
    using reference = typename traits::reference;
    using const_reference = typename traits::const_reference;
    static constexpr int Order = Base::Order;
    static constexpr int DynamicOrder = Base::DynamicOrder;
    static constexpr std::array<index_t, Order> static_extents = Base::static_extents;
    using Base::extents_;
    using Base::mapping_;

    constexpr MdArray()
        requires(std::is_default_constructible_v<storage_t>)
        : Base(), data_() {
        for (size_t i = 0; i < extents_.size(); ++i) { data_[i] = Scalar(); }
    }
    template <typename... Exts_>
        requires(extents_t::DynamicOrder != 0 && extents_t::DynamicOrder == sizeof...(Exts_)) &&
                  (std::is_convertible_v<Exts_, index_t> && ...) && std::is_default_constructible_v<storage_t>
    constexpr MdArray(Exts_... exts) : Base(std::forward<Exts_>(exts)...), data_() {
        data_.resize(extents_.size(), Scalar());
    }
    template <typename OtherExtents, typename OtherMapping>
        requires(!std::is_convertible_v<OtherExtents, index_t> && std::is_constructible_v<extents_t, OtherExtents> &&
                 !std::is_convertible_v<OtherMapping, index_t> && std::is_constructible_v<mapping_t, OtherMapping> &&
                 std::is_default_constructible_v<storage_t>)
    constexpr MdArray(const OtherExtents& extents, const OtherMapping& mapping) : Base(extents, mapping), data_() {
        data_.resize(extents_.size(), Scalar());
    }
    template <typename OtherExtents>
        requires(!std::is_convertible_v<OtherExtents, index_t> && std::is_constructible_v<extents_t, OtherExtents> &&
                 std::is_default_constructible_v<storage_t>)
    constexpr MdArray(const OtherExtents& extents) : Base(extents), data_() {
        data_.resize(extents_.size(), Scalar());
    }
    // construct from callable
    template <typename Callable>
        requires(extents_t::DynamicOrder == 0 && std::is_invocable_v<Callable>)
    constexpr explicit MdArray(Callable callable) : Base(), data_() {
        fdapde_static_assert(
          std::is_convertible_v<typename decltype(std::function {callable})::result_type FDAPDE_COMMA storage_t>,
          CALLABLE_RETURN_TYPE_NOT_CONVERTIBLE_TO_MDARRAY_STORAGE_TYPE);
	mapping_ = mapping_t(extents_);
        data_ = callable();
    }
    template <typename Callable, typename... Exts_>
        requires(extents_t::DynamicOrder != 0 && extents_t::DynamicOrder == sizeof...(Exts_)) &&
                  std::is_invocable_v<Callable> && (std::is_convertible_v<Exts_, index_t> && ...)
    constexpr MdArray(Callable callable, Exts_... exts) : Base(), data_() {
        fdapde_static_assert(
          std::is_convertible_v<typename decltype(std::function {callable})::result_type FDAPDE_COMMA storage_t>,
          CALLABLE_RETURN_TYPE_NOT_CONVERTIBLE_TO_MDARRAY_STORAGE_TYPE);
	extents_ = extents_t(static_cast<index_t>(exts)...);
	mapping_ = mapping_t(extents_);
        data_ = callable();
        fdapde_constexpr_assert(extents_.size() == data_.size());
    }
    // construct from other MdArray
    template <typename OtherScalar, typename OtherExtents, typename OtherLayoutPolicy>
        requires(std::is_convertible_v<OtherScalar, Scalar> &&
                 std::is_constructible_v<
                   mapping_t, const typename OtherLayoutPolicy::template mapping<OtherExtents>&> &&
                 std::is_constructible_v<extents_t, const OtherExtents&>)
    constexpr MdArray(const MdArray<OtherScalar, OtherExtents, OtherLayoutPolicy>& other) :
        Base(), data_(*other.data()) {
        if constexpr (extents_t::StaticOrder > 0) {
            for (order_t i = 0; i < extents_t::Order; ++i) {
                fdapde_constexpr_assert(
                  extents_t::static_extents[i] == Dynamic ||
                  extents_t::static_extents[i] == OtherExtents::static_extents[i]);
            }
        }
	mapping_ = other.mapping();
	extents_ = other.extents();
    }
    // construct from MdArraySlice
    template <typename OtherMdArray, int... OtherSlicers>
        requires(Order == OtherMdArray::Order - sizeof...(OtherSlicers) &&
                 std::is_same_v<layout_t, typename OtherMdArray::layout_t> &&
                 std::is_default_constructible_v<storage_t> && std::is_default_constructible_v<mapping_t> &&
                 std::is_default_constructible_v<extents_t>)
    constexpr MdArray(const MdArraySlice<OtherMdArray, OtherSlicers...>& other) : Base(), data_() {
        assign_from_slice_(other);
    }
    // construct from MdArrayBlock
    template <typename OtherMdArray, typename OtherBlkExtents>
        requires(Order == OtherMdArray::Order && std::is_same_v<layout_t, typename OtherMdArray::layout_t> &&
                 std::is_default_constructible_v<storage_t> && std::is_default_constructible_v<mapping_t> &&
                 std::is_default_constructible_v<extents_t>)
    constexpr MdArray(const MdArrayBlock<OtherMdArray, OtherBlkExtents>& other) : Base(), data_() {
        assign_from_block_(other);
    }
    // assignment
    template <typename OtherDerived, int... OtherSlicers>
        requires(Order == OtherDerived::Order - sizeof...(OtherSlicers))
    constexpr MdArray& operator=(const MdArraySlice<OtherDerived, OtherSlicers...>& other) {
        assign_from_slice_(other);
        return *this;
    }
    template <typename OtherDerived, typename OtherBlkExtents>
        requires(Order == OtherDerived::Order)
    constexpr MdArray& operator=(const MdArrayBlock<OtherDerived, OtherBlkExtents>& other) {
        assign_from_block_(other);
        return *this;
    }    

    constexpr const Scalar* data() const { return data_.data(); }
    constexpr Scalar* data() { return data_.data(); }
    // resize dynamic MdArray (only dynamic extents). allocated memory is left uninitialized
    template <typename... Exts_>
        requires(extents_t::DynamicOrder != 0 && (std::is_convertible_v<Exts_, index_t> && ...))
    constexpr void resize(Exts_... exts) {
        fdapde_static_assert(
          sizeof...(Exts_) == extents_t::DynamicOrder,
          YOU_SUPPLIED_A_WRONG_NUMBER_OF_ARGUMENTS_TO_RESIZE__NUMBER_OF_ARGUMENTS_MUST_MATCH_NUMBER_OF_DYNAMIC_EXTENTS);
        extents_.resize(static_cast<index_t>(exts)...);
        mapping_ = mapping_t(extents_);
        data_.resize(Base::size());   // re-allocate space
    }
    template <typename IndexPack>
        requires(internals::is_subscriptable<IndexPack, index_t>)
    constexpr void resize(IndexPack&& index_pack) {
        internals::apply_index_pack<Order>([&]<int... Ns_> { resize(index_pack[Ns_]...); });
    }
   private:
    template <typename OtherDerived, int... OtherSlicers>
        requires(Order == OtherDerived::Order - sizeof...(OtherSlicers))
    constexpr void assign_from_slice_(const MdArraySlice<OtherDerived, OtherSlicers...>& other) {
        using slice_t = MdArraySlice<OtherDerived, OtherSlicers...>;
        if constexpr (extents_t::StaticOrder > 0) {
            for (order_t i = 0; i < slice_t::Order; ++i) {
                order_t extent_ = slice_t::free_extents_idxs_[i];
                fdapde_constexpr_assert(
                  extents_t::static_extents[i] == Dynamic ||
                  extents_t::static_extents[i] == OtherDerived::static_extents[slice_t::free_extents_idxs_[i]]);
            }
        }
        if constexpr (extents_t::DynamicOrder > 0) {
            if (Base::size() != other.size()) { data_.resize(other.size()); }
        }
        // to avoid aliasing, first copy data, then update mapping
        if constexpr (internals::slices_to_contiguous_memory<typename OtherDerived::mapping_t, OtherSlicers...>()) {
            for (int i = 0, n = other.size(); i < n; ++i) { data_[i] = other[i]; }   // copy from contiguous memory
        } else {
            for (auto it = other.begin(); it != other.end(); ++it) { data_[it.mapped_index()] = *it; }
        }
        if constexpr (extents_t::DynamicOrder > 0) {
            if (Base::size() != other.size()) {
                internals::apply_index_pack<Order>(
                  [&]<int... Ns_> { extents_.resize(static_cast<index_t>(other.extent(Ns_))...); });
            }
            mapping_ = mapping_t(extents_);
        }
    }
    template <typename OtherDerived, typename OtherBlkExtents>
        requires(Order == OtherDerived::Order)
    constexpr void assign_from_block_(const MdArrayBlock<OtherDerived, OtherBlkExtents>& other) {
        using block_t = MdArrayBlock<OtherDerived, OtherBlkExtents>;
	
        if constexpr (extents_t::StaticOrder > 0) {
            for (order_t i = 0; i < extents_t::Order; ++i) {
                fdapde_constexpr_assert(
                  extents_t::static_extents[i] == Dynamic ||
                  extents_t::static_extents[i] == block_t::static_extents[i]);
            }
        }
        if constexpr (extents_t::DynamicOrder > 0) {
            // to avoid aliasing, first copy data, then update mapping
            if (Base::size() != other.size()) { data_.resize(other.size()); }
            for (auto it = other.begin(); it != other.end(); ++it) { data_[it.mapped_index()] = *it; }

            if (Base::size() != other.size()) {
                internals::apply_index_pack<Order>(
                  [&]<int... Ns_> { extents_.resize(static_cast<index_t>(other.extent(Ns_))...); });
            }
            mapping_ = mapping_t(extents_);	    
        } else {
            for (auto it = other.begin(); it != other.end(); ++it) { data_[it.mapped_index()] = *it; }
            extents_ = other.extents();
            mapping_ = mapping_t(extents_);
        }
    }
    storage_t data_ {};
};

namespace internals {

template <typename Extents_, typename LayoutPolicy_> struct md_traits<MdArray<bool, Extents_, LayoutPolicy_>> {
    using extents_t = Extents_;
    using index_t = typename extents_t::index_t;
    using order_t = typename extents_t::order_t;
    using size_t = typename extents_t::size_t;
    using bitpack_t = std::uintmax_t;
    static constexpr int PackSize = sizeof(bitpack_t) * 8;
    using data_t = std::conditional_t<
      extents_t::DynamicOrder != 0, std::vector<bitpack_t>,
      std::array<bitpack_t, std::size_t(fdapde::ceil(extents_t::StaticSize / PackSize))>>;
    using layout_t = LayoutPolicy_;
    using mapping_t = typename layout_t::template mapping<extents_t>;
    // struct to proxy the behaviour of reference to a single bit of the MdArray
    template <typename BitPackT>
        requires(std::is_same_v<std::decay_t<BitPackT>, bitpack_t>)
    struct bit_proxy {
       private:
        template <typename... Idxs>
            requires(std::is_convertible_v<Idxs, index_t> && ...) && (sizeof...(Idxs) == extents_t::Order)
        constexpr std::pair<int, int> pack_of_(Idxs... idxs) const {
            int map = mapping_(static_cast<index_t>(idxs)...);
            return std::make_pair(map / PackSize, map % PackSize);   // pack id and bit position in bitpack
        }
       public:
        friend bit_proxy<bitpack_t>;
        friend bit_proxy<const bitpack_t>;
      
        constexpr bit_proxy() noexcept : data_(nullptr), pack_id_(0), bitmask_(0) { }
        template <typename BitPackT_>
        constexpr bit_proxy(const bit_proxy<BitPackT_>& other) :
            data_(const_cast<BitPackT*>(other.data_)), pack_id_(other.pack_id_), bitmask_(other.bitmask_) { }
        template <typename BitPackT_> constexpr bit_proxy& operator=(const bit_proxy<BitPackT_>& other) {
            data_ = const_cast<BitPackT*>(other.data_);
            pack_id_ = other.pack_id_;
            bitmask_ = other.bitmask_;
            return *this;
        }

        template <typename... Idxs>
            requires(std::is_convertible_v<Idxs, index_t> && ...) && (sizeof...(Idxs) == extents_t::Order)
        explicit constexpr bit_proxy(BitPackT* data, Idxs... idxs) : data_(data), pack_id_(), bitmask_() {
            auto [pack_id, bit_off] = pack_of_(idxs...);
            pack_id_ = pack_id;
            bitmask_ = bitpack_t(1) << bit_off;
        }
        explicit constexpr bit_proxy(BitPackT* data, index_t i) :
            data_(data), pack_id_(i / PackSize), bitmask_(bitpack_t(1) << i % PackSize) { }
        // modifiers
        constexpr void set()   { data_[pack_id_] |=  bitmask_; }
        constexpr void clear() { data_[pack_id_] &= ~bitmask_; }
        template <typename T>
            requires(std::is_convertible_v<T, bool>)
        constexpr bit_proxy& operator=(T b) {
            b ? set() : clear();
	    return *this;
        }
        // observers
        constexpr operator bool() const { return (data_[pack_id_] & bitmask_) != 0; }
        constexpr operator bool() { return (data_[pack_id_] & bitmask_) != 0; }
       private:
        BitPackT* data_;
        index_t pack_id_;
        bitpack_t bitmask_;
    };
    using Scalar = bool;
    using reference = bit_proxy<bitpack_t>;
    using const_reference = bit_proxy<const bitpack_t>;
   private:
    // struct to proxy the behaviour of a bool*
    template <typename BitPackT>
        requires(std::is_same_v<std::decay_t<BitPackT>, bitpack_t>)
    class storage_t_impl {
        BitPackT* data_;
       public:
        storage_t_impl() noexcept : data_(nullptr) { }
        storage_t_impl(BitPackT* data) : data_(data) { }
        reference operator[](int i) requires(!std::is_const_v<BitPackT>) { return reference(data_, i); }
        const_reference operator[](int i) const requires(std::is_const_v<BitPackT>) {
	    return const_reference(data_, i);
	}
        // raw data access
        bitpack_t& operator*() { return *data_; }
        const bitpack_t& operator*() const { return *data_; }
        // pointer access
        BitPackT* data() { return data_; }
        const BitPackT* data() const { return data_; }
    };
   public:
    using storage_t = storage_t_impl<bitpack_t>;
    using const_storage_t = storage_t_impl<const bitpack_t>;
};

}   // namespace internals

// MdArray specialization for bool Scalar type, compactly indexing bit values
template <typename Extents_, typename LayoutPolicy_>
class MdArray<bool, Extents_, LayoutPolicy_> :
    public internals::md_handler_base<MdArray<bool, Extents_, LayoutPolicy_>> {
    using Base = internals::md_handler_base<MdArray<bool, Extents_, LayoutPolicy_>>;
    using traits = internals::md_traits<MdArray<bool, Extents_, LayoutPolicy_>>;
   private:
    using data_t = typename traits::data_t;   // physical data structure
    using Base::extents_;
    using Base::mapping_;
   public:
    using layout_t = LayoutPolicy_;
    using index_t = typename traits::index_t;
    using order_t = typename traits::order_t;
    using size_t  = typename traits::size_t;
    using extents_t = Extents_;
    using mapping_t = typename traits::mapping_t;
    using storage_t = typename traits::storage_t;
    using const_storage_t = typename traits::const_storage_t;
    using reference = typename traits::reference;
    using Scalar = typename traits::Scalar;
    using const_reference = typename traits::const_reference;
    static constexpr int PackSize = traits::PackSize;
    static constexpr int Order = Base::Order;
    static constexpr int DynamicOrder = Base::DynamicOrder;
    static constexpr std::array<index_t, Order> static_extents = Base::static_extents;

    constexpr MdArray()
        requires(std::is_default_constructible_v<data_t>)
        : Base(), data_() {
        if constexpr (DynamicOrder == 0) {
            std::fill_n(data_.begin(), std::size_t(std::ceil(extents_t::StaticSize / PackSize)), 0);
        }
    }
    template <typename... Exts_>
        requires(extents_t::DynamicOrder != 0 && extents_t::DynamicOrder == sizeof...(Exts_)) &&
                  (std::is_convertible_v<Exts_, index_t> && ...) && std::is_default_constructible_v<data_t>
    constexpr MdArray(Exts_... exts) : Base(std::forward<Exts_>(exts)...), data_() {
        data_.resize(int_ceil(extents_.size(), PackSize), 0);
    }
    template <typename OtherExtents, typename OtherMapping>
        requires(!std::is_convertible_v<OtherExtents, index_t> && std::is_constructible_v<extents_t, OtherExtents> &&
                 !std::is_convertible_v<OtherMapping, index_t> && std::is_constructible_v<mapping_t, OtherMapping> &&
                 std::is_default_constructible_v<data_t>)
    constexpr MdArray(const OtherExtents& extents, const OtherMapping& mapping) : Base(extents, mapping), data_() {
        data_.resize(int_ceil(extents_.size(), PackSize), 0);
    }
    template <typename OtherExtents>
        requires(!std::is_convertible_v<OtherExtents, index_t> && std::is_constructible_v<extents_t, OtherExtents> &&
                 std::is_default_constructible_v<data_t>)
    constexpr MdArray(const OtherExtents& extents) : Base(extents), data_() {
        data_.resize(int_ceil(extents_.size(), PackSize), 0);
    }
    // construct from MdArraySlice
    template <typename OtherMdArray, int... OtherSlicers>
        requires(Order == OtherMdArray::Order - sizeof...(OtherSlicers) &&
                 std::is_same_v<layout_t, typename OtherMdArray::layout_t> &&
                 std::is_default_constructible_v<storage_t> && std::is_default_constructible_v<mapping_t> &&
                 std::is_default_constructible_v<extents_t>)
    constexpr MdArray(const MdArraySlice<OtherMdArray, OtherSlicers...>& other) : Base(), data_() {
        assign_from_slice_(other);
    }
    // construct from MdArrayBlock
    template <typename OtherMdArray, typename OtherBlkExtents>
        requires(Order == OtherMdArray::Order && std::is_same_v<layout_t, typename OtherMdArray::layout_t> &&
                 std::is_default_constructible_v<storage_t> && std::is_default_constructible_v<mapping_t> &&
                 std::is_default_constructible_v<extents_t>)
    constexpr MdArray(const MdArrayBlock<OtherMdArray, OtherBlkExtents>& other) : Base(), data_() {
        assign_from_block_(other);
    }  
    // assignment
    template <typename OtherDerived, int... OtherSlicers>
        requires(
          Order == OtherDerived::Order - sizeof...(OtherSlicers) &&
          std::is_convertible_v<typename OtherDerived::Scalar, bool>)
    constexpr MdArray& operator=(const MdArraySlice<OtherDerived, OtherSlicers...>& other) {
        assign_from_slice_(other);
        return *this;
    }
    template <typename OtherDerived, typename OtherBlkExtents>
        requires(Order == OtherDerived::Order && std::is_convertible_v<typename OtherDerived::Scalar, bool>)
    constexpr MdArray& operator=(const MdArrayBlock<OtherDerived, OtherBlkExtents>& other) {
        assign_from_block_(other);
        return *this;
    }
    // modifiers
    // resize dynamic MdArray (only dynamic extents). allocated memory is left uninitialized
    template <typename... Exts_>
        requires(extents_t::DynamicOrder != 0 && (std::is_convertible_v<Exts_, index_t> && ...))
    constexpr void resize(Exts_... exts) {
        fdapde_static_assert(
          sizeof...(Exts_) == extents_t::DynamicOrder,
          YOU_SUPPLIED_A_WRONG_NUMBER_OF_ARGUMENTS_TO_RESIZE__NUMBER_OF_ARGUMENTS_MUST_MATCH_NUMBER_OF_DYNAMIC_EXTENTS);
        extents_.resize(static_cast<index_t>(exts)...);
        mapping_ = mapping_t(extents_);
        data_.resize(int_ceil(extents_.size(), PackSize), 0);
    }
    template <typename IndexPack>
        requires(internals::is_subscriptable<IndexPack, index_t>)
    constexpr void resize(IndexPack&& index_pack) {
        internals::apply_index_pack<Order>([&]<int... Ns_> { resize(index_pack[Ns_]...); });
    }
    constexpr void set() {
        std::for_each(data_.begin(), data_.end(), [](auto& b) { b = -1; });
    }
    constexpr void clear() {
        std::for_each(data_.begin(), data_.end(), [](auto& b) { b =  0; });
    }
    // observers
    constexpr const_storage_t data() const { return const_storage_t(data_.data()); }
    constexpr storage_t data() { return storage_t(data_.data()); }
    constexpr size_t bitpacks() const { return data_.size(); }
    constexpr auto as_matrix() const {
        fdapde_static_assert(Order == 2 || Order == 1, THIS_METHOD_IS_FOR_MDARRAYS_OF_ORDER_ONE_OR_TWO_ONLY);
        if constexpr (Order == 2) {
            BinaryMap<static_extents[0], static_extents[1], const typename traits::bitpack_t> map(
              data().data(), Base::extent(0), Base::extent(1));
            return map;
        } else {
            BinaryMap<static_extents[0], 1, const typename traits::bitpack_t> map(data().data(), Base::extent(0), 1);
            return map;
        }
    }
   private:
    template <typename OtherDerived, int... OtherSlicers>
        requires(Order == OtherDerived::Order - sizeof...(OtherSlicers))
    constexpr void assign_from_slice_(const MdArraySlice<OtherDerived, OtherSlicers...>& other) {
        using slice_t = MdArraySlice<OtherDerived, OtherSlicers...>;
        if constexpr (extents_t::StaticOrder > 0) {
            for (order_t i = 0; i < slice_t::Order; ++i) {
                order_t extent_ = slice_t::free_extents_idxs_[i];
                fdapde_constexpr_assert(
                  extents_t::static_extents[i] == Dynamic ||
                  extents_t::static_extents[i] == OtherDerived::static_extents[slice_t::free_extents_idxs_[i]]);
            }
        }
        // to avoid aliasing, first copy data, then update mapping
        data_t tmp(int_ceil(other.size(), PackSize));
        int i = 0;
        for (const auto& v : other) {
            typename traits::bit_proxy bit(tmp.data(), i++);
            if (v) { bit.set(); }
        }
        data_ = tmp;
        if constexpr (extents_t::DynamicOrder > 0) {
            if (Base::size() != other.size()) {
                internals::apply_index_pack<Order>(
                  [&]<int... Ns_> { extents_.resize(static_cast<index_t>(other.extent(Ns_))...); });
            }
            mapping_ = mapping_t(extents_);
        } else {
            extents_ = other.extents();
            mapping_ = other.mapping();
        }
    }
    template <typename OtherDerived, typename OtherBlkExtents>
        requires(Order == OtherDerived::Order)
    constexpr void assign_from_block_(const MdArrayBlock<OtherDerived, OtherBlkExtents>& other) {
        using block_t = MdArrayBlock<OtherDerived, OtherBlkExtents>;
        if constexpr (extents_t::StaticOrder > 0) {
            for (order_t i = 0; i < extents_t::Order; ++i) {
                fdapde_constexpr_assert(
                  extents_t::static_extents[i] == Dynamic ||
                  extents_t::static_extents[i] == block_t::static_extents[i]);
            }
        }
        // to avoid aliasing, first copy data, then update mapping
        data_t tmp(int_ceil(other.size(), PackSize));
        int i = 0;
        for (const auto& v : other) {
            typename traits::bit_proxy bit(tmp.data(), i++);
            if (v) { bit.set(); }
        }
        data_ = tmp;
        if constexpr (extents_t::DynamicOrder > 0) {
            if (Base::size() != other.size()) {
                internals::apply_index_pack<Order>(
                  [&]<int... Ns_> { extents_.resize(static_cast<index_t>(other.extent(Ns_))...); });
            }
            mapping_ = mapping_t(extents_);
        } else {
            extents_ = other.extents();
            mapping_ = other.mapping();
        } 
    }
    data_t data_ {};
};

// a not-owning multidimensional view of a flat memory region
template <typename Scalar_, typename Extents_, typename LayoutPolicy_ = internals::layout_right>
class MdMap : public internals::md_handler_base<MdMap<Scalar_, Extents_, LayoutPolicy_>> {
    using Base   = internals::md_handler_base<MdMap<Scalar_, Extents_, LayoutPolicy_>>;
    using traits = internals::md_traits<MdMap<Scalar_, Extents_, LayoutPolicy_>>;
   public:
    using Scalar = Scalar_;
    using layout_t = LayoutPolicy_;
    using index_t = typename traits::index_t;
    using order_t = typename traits::order_t;
    using size_t  = typename traits::size_t;
    using extents_t = Extents_;
    using mapping_t = typename traits::mapping_t;
    using storage_t = typename traits::storage_t;
    using reference = typename traits::reference;
    using const_reference = typename traits::const_reference;
    static constexpr int Order = Base::Order;
    static constexpr int DynamicOrder = Base::DynamicOrder;
    static constexpr std::array<index_t, Order> static_extents = Base::static_extents;
    using Base::extents_;
    using Base::mapping_;

    constexpr MdMap()
        requires(std::is_default_constructible_v<extents_t> && std::is_default_constructible_v<storage_t>)
        : Base(), data_(nullptr) { }
    template <typename... Exts_>
        requires(extents_t::DynamicOrder != 0 && extents_t::DynamicOrder == sizeof...(Exts_)) &&
                  (std::is_convertible_v<Exts_, index_t> && ...) && std::is_pointer_v<storage_t>
    constexpr MdMap(Scalar_* data, Exts_... exts) : Base(std::forward<Exts_>(exts)...), data_(data) { }
    // pointer to raw mapped data
    constexpr const Scalar* data() const { return data_; }
    constexpr Scalar* data() { return data_; }
   private:
    storage_t data_ = nullptr;
};

}   // namespace fdapde

#endif   // __FDAPDE_MDARRAY_H__
