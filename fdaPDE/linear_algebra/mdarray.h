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

#ifndef __MDARRAY_H__
#define __MDARRAY_H__

#include <array>
#include <numeric>

#include "../utils/assert.h"
#include "../utils/symbols.h"
#include "../utils/traits.h"
#include "constexpr_matrix.h"

namespace fdapde {

static constexpr int full_extent = -2;   // indicates to take the whole extent in subsetting
  
namespace internals {
  
// apply lambda F_ to each pack index {0, ..., N_ - 1}
template <int N_, typename F_> constexpr decltype(auto) apply_index_pack(F_&& f) {
    return [&]<int... Ns_>(std::integer_sequence<int, Ns_...>) -> decltype(auto) {
        return f.template operator()<Ns_...>();
    }(std::make_integer_sequence<int, N_> {});
}
template <int N_, typename F_> constexpr void for_each_index_in_pack(F_&& f) {
    [&]<int... Ns_>(std::integer_sequence<int, Ns_...>) {
        (f.template operator()<Ns_>(), ...);
    }(std::make_integer_sequence<int, N_> {});
}

// apply lambda F_ to each pack index and args pair {(0, args[0]), ..., (N_ - 1, args[N_ - 1])}
template <int N_, typename F_, typename... Args_>
    requires(sizeof...(Args_) == N_)
constexpr void for_each_index_and_args(F_&& f, Args_&&... args) {
    [&]<int... Ns_>(std::integer_sequence<int, Ns_...>) {
        (f.template operator()<Ns_, Args_>(std::forward<Args_>(args)), ...);
    }(std::make_integer_sequence<int, N_> {});
}

template <typename Extent, typename Idx>
constexpr bool is_index_in_extent(Extent ext, Idx idx)
    requires(std::is_convertible_v<Extent, int> && std::is_convertible_v<Idx, int>) {
    if constexpr (std::is_signed_v<Idx>) {
        if (idx < 0) return false;
    }
    return static_cast<int>(idx) < static_cast<int>(ext);
}
template <typename Extent_, typename... Idxs>
constexpr bool is_multidimensional_index_in_extent(const Extent_& ext, Idxs... idxs) {
    return internals::apply_index_pack<sizeof...(Idxs)>(
      [&]<int... Ns_>() { return (is_index_in_extent(ext.extent(Ns_), idxs) && ...); });
}

template <typename T> class is_integer {
    using T_ = std::decay_t<T>;
   public:
    static constexpr bool value =
      std::is_arithmetic_v<T_> && !std::is_same_v<T_, bool> && !std::is_floating_point_v<T_>;
};
template <typename T> static constexpr bool is_integer_v = is_integer<T>::value;

template <typename T> struct is_tuple_impl : std::false_type { };
template <typename... Ts> struct is_tuple_impl<std::tuple<Ts...>> : std::true_type { };
template <typename T1, typename T2> struct is_tuple_impl<std::pair<T1, T2>> : std::true_type { };

template <typename T> static constexpr bool is_tuple_v = is_tuple_impl<std::decay_t<T>>::value;
template <typename T>
static constexpr bool is_pair_v = []() {
    if constexpr (is_tuple_v<std::decay_t<T>>) {
        if constexpr (std::tuple_size_v<std::decay_t<T>> == 2) { return true; }
	else { return false; }
    }   else { return false; }
}();

// get i-th element from pack
template <int N, typename... Ts>
    requires(sizeof...(Ts) >= N)
struct Nth_type_from_pack : std::type_identity<std::tuple_element_t<N, std::tuple<Ts...>>> { };

template <int N, typename... Ts> using Nth_type_from_pack_t = Nth_type_from_pack<N, Ts...>::type;

template <int N, typename... Ts> constexpr auto Nth_element_from_pack(Ts&&... ts) {
    return std::get<N>(std::make_tuple(std::forward<Ts>(ts)...));
}

// finds the Idx-th component of the lowest multidimensional index spanned by slicers...
template <int Idx_, typename... Slicers_> constexpr int smallest_index_in_mdarray_blk(Slicers_&&... slicers) {
    using Slicer = Nth_type_from_pack_t<Idx_, std::decay_t<Slicers_>...>;
    const auto& slicer = Nth_element_from_pack<Idx_>(std::forward<Slicers_>(slicers)...);
    if constexpr (internals::is_integer_v<Slicer>) {
        return (slicer == full_extent) ? 0 : slicer;
    } else if constexpr (
      is_pair_v<Slicer> && std::convertible_to<std::tuple_element_t<0, Slicer>, int> &&
      std::convertible_to<std::tuple_element_t<1, Slicer>, int>) {
        fdapde_assert(std::get<0>(slicer) != full_extent && std::get<1>(slicer) != full_extent);
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
        constexpr mapping& operator=(const mapping&) noexcept = default;
        constexpr mapping(const Extents& extents) noexcept
            requires(Order > 0)
            : extents_(extents), strides_() {
            for (order_t i = 0; i < Order + 1; ++i) {
                strides_[i] = 1;
                for (order_t j = 0; j < i; ++j) { strides_[i] *= extents_.extent(j); }
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
        template <int IndexSize, typename IndexType>   // array index to mdarray memory index
            requires(std::is_convertible_v<IndexType, index_t> && IndexSize == Order)
        constexpr index_t operator()(const std::array<IndexType, IndexSize>& arr) {
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

struct layout_right {   // corresponds to a RowMajor storage for order 2 mdarrays

    template <typename Extents> struct mapping {
        using extents_t = Extents;
        using index_t = typename extents_t::index_t;
        using order_t = typename extents_t::order_t;
        using size_t  = typename extents_t::size_t;
        using layout_type = layout_right;    
        static constexpr order_t Order = extents_t::Order;

        constexpr mapping() noexcept = default;
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
        template <int IndexSize, typename IndexType>   // array index to mdarray memory index
            requires(std::is_convertible_v<IndexType, index_t> && IndexSize == Order)
        constexpr index_t operator()(const std::array<IndexType, IndexSize>& arr) {
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
template <typename MdArray, typename BlkExtents> class MdArrayBlock { 
   public:
    using extents_t = BlkExtents;
    using index_t = typename extents_t::index_t;
    using order_t = typename extents_t::order_t;
    using size_t  = typename extents_t::size_t;
    using Scalar  = typename MdArray::Scalar;
    static constexpr order_t Order = MdArray::Order;

    constexpr MdArrayBlock() noexcept = default;
    template <typename... Slicers>
        requires(sizeof...(Slicers) == Order && BlkExtents::Order == MdArray::Order) &&
                  ((internals::is_integer_v<Slicers> || internals::is_pair_v<Slicers>) && ...)
    constexpr MdArrayBlock(MdArray* mdarray, BlkExtents blk_extents, Slicers&&... slicers) noexcept :
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
    constexpr order_t order() const { return Order; }
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
            while (i > 0 && index_[i] >= static_cast<index_t>(mdarray_->extent(i))) {
                index_[i] = 0;
                index_[--i]++;
            }
            return *this;
        }
        // const access
        constexpr const Scalar& operator*()  const { return fetch_at_(index_); }
        constexpr const Scalar* operator->() const { return std::addressof(fetch_at(index_)); }
        // non-const access
        constexpr Scalar& operator*()  requires(!std::is_const_v<MdArrayBlock_>) { return fetch_at_(index_); }
        constexpr Scalar* operator->() requires(!std::is_const_v<MdArrayBlock_>) {
            return std::addressof(fetch_at(index_));
        }
        // comparison
        constexpr friend bool operator==(const iterator& lhs, const iterator& rhs) { return lhs.index_ == rhs.index_; }
        constexpr friend bool operator!=(const iterator& lhs, const iterator& rhs) { return lhs.index_ != rhs.index_; }
       private:
        template <typename IndexType> constexpr decltype(auto) fetch_at_(IndexType&& index) const {
            return internals::apply_index_pack<Order>(
              [&]<int... Ns_>() -> decltype(auto) { return mdarray_->operator()(((void)Ns_, index[Ns_])...); });
        }
        MdArrayBlock_* mdarray_;
        std::array<index_t, Order> index_;
    };
    // const iterators
    constexpr iterator<const MdArrayBlock<MdArray, BlkExtents>> begin() const noexcept {
        return internals::apply_index_pack<Order>(
          [&]<int... Ns_> { return iterator<const MdArrayBlock<MdArray, BlkExtents>> {this, ((void)Ns_, 0)...}; });
    }
    constexpr iterator<const MdArrayBlock<MdArray, BlkExtents>> end() const noexcept {
        return internals::apply_index_pack<Order - 1>([&]<int... Ns_> {
            return iterator<const MdArrayBlock<MdArray, BlkExtents>> {this, extents_.extent(0), ((void)Ns_, 0)...};
        });
    }
    // non-const iterators
    constexpr iterator<MdArrayBlock<MdArray, BlkExtents>> begin() noexcept {
        return internals::apply_index_pack<Order>(
          [&]<int... Ns_> { return iterator<MdArrayBlock<MdArray, BlkExtents>> {this, ((void)Ns_, 0)...}; });
    }
    constexpr iterator<MdArrayBlock<MdArray, BlkExtents>> end() noexcept {
        return internals::apply_index_pack<Order - 1>([&]<int... Ns_> {
            return iterator<MdArrayBlock<MdArray, BlkExtents>> {this, extents_.extent(0), ((void)Ns_, 0)...};
        });
    }
    // modifiers
    constexpr void set_constant(Scalar c) {
        for (Scalar& value : *this) { value = Scalar(c); }
    }
    constexpr void set_zero() { set_constant(Scalar(0)); }
    constexpr void set_ones() { set_constant(Scalar(1)); }
    // constant access
    template <typename... Idxs>
        requires(std::is_convertible_v<Idxs, index_t> && ...) && (sizeof...(Idxs) == BlkExtents::Order)
    constexpr Scalar operator()(Idxs... idxs) const {
        fdapde_constexpr_assert(
          internals::is_multidimensional_index_in_extent(extents_, static_cast<index_t>(idxs)...));
        return mdarray_->operator[](internals::apply_index_pack<Order>([&]<int... Ns_>() {
            return (((static_cast<index_t>(idxs) + offset_[Ns_]) * mdarray_->mapping().stride(Ns_)) + ... + 0);
        }));
    }
    // non-constant access
    template <typename... Idxs>
        requires(std::is_convertible_v<Idxs, index_t> && ...) &&
                (sizeof...(Idxs) == BlkExtents::Order && !std::is_const_v<MdArray>)
    constexpr Scalar& operator()(Idxs... idxs) {
        fdapde_constexpr_assert(
          internals::is_multidimensional_index_in_extent(extents_, static_cast<index_t>(idxs)...));
        return mdarray_->operator[](internals::apply_index_pack<Order>([&]<int... Ns_>() {
            return (((static_cast<index_t>(idxs) + offset_[Ns_]) * mdarray_->mapping().stride(Ns_)) + ... + 0);
        }));
    }
   private:
    std::array<index_t, Order> offset_ {};
    extents_t extents_;
    MdArray* mdarray_;
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
              fdapde_constexpr_assert(s < static_cast<index_t>(mdarray.extent(Ns_)));
          } else if constexpr (internals::is_pair_v<Slicer_>) {
              fdapde_constexpr_assert(
                std::get<0>(s) != full_extent && std::get<1>(s) != full_extent && std::get<1>(s) > std::get<0>(s) &&
                std::get<1>(s) < static_cast<index_t>(mdarray.extent(Ns_)));
          }
      },
      slicers...);
    // prepare block extents
    std::array<index_t, Order> blk_extents {};
    internals::for_each_index_and_args<Order>(
      [&]<int Ns_, typename Slicer_>(Slicer_ s) mutable {
          if constexpr (internals::is_integer_v<Slicer_>) {
              blk_extents[Ns_] = (s == full_extent) ? mdarray.extent(Ns_) : 1;
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
    for (int j = 0; j < sizeof...(Slicers); ++j) {
        if (slicers_[j] != i++) return false;
    }
    return true;
}

}   // namespace internals

// a slice is a lower-order view of an MdArray in which one or more dimensions have been fixed
template <typename MdArray, int... Slicers> class MdArraySlice {
    static_assert(sizeof...(Slicers) < MdArray::Order && ((Slicers >= 0 && Slicers < MdArray::Order) && ...));
   public:
    using extents_t = typename MdArray::extents_t;
    using index_t = typename extents_t::index_t;
    using order_t = typename extents_t::order_t;
    using size_t  = typename extents_t::size_t;  
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
        if constexpr (DynamicOrder == 0)
            return std::array<index_t, DynamicOrder> {};
        else {
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
              fdapde_constexpr_assert(s < mdarray_->extent(static_slicers[Ns_]));
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
        // const access
        constexpr const Scalar& operator*()  const { return fetch_at_(index_); }
        constexpr const Scalar* operator->() const { return std::addressof(fetch_at(index_)); }
        // non-const access
        constexpr Scalar& operator*()  requires(!std::is_const_v<MdArraySlice_>) { return fetch_at_(index_); }
        constexpr Scalar* operator->() requires(!std::is_const_v<MdArraySlice_>) {
            return std::addressof(fetch_at(index_));
        }
        // comparison
        constexpr friend bool operator==(const iterator& lhs, const iterator& rhs) { return lhs.index_ == rhs.index_; }
        constexpr friend bool operator!=(const iterator& lhs, const iterator& rhs) { return lhs.index_ != rhs.index_; }
       private:
        template <typename IndexType> constexpr decltype(auto) fetch_at_(IndexType&& index) const {
            return internals::apply_index_pack<Order>(
              [&]<int... Ns_>() -> decltype(auto) { return mdarray_->operator()(((void)Ns_, index[Ns_])...); });
        }
        MdArraySlice_* mdarray_;
        std::array<index_t, Order> index_;
    };
    // const iterators
    constexpr iterator<const MdArraySlice<MdArray, Slicers...>> begin() const noexcept {
        return internals::apply_index_pack<Order>(
          [&]<int... Ns_> { return iterator<const MdArraySlice<MdArray, Slicers...>> {this, ((void)Ns_, 0)...}; });
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
    constexpr void set_constant(Scalar c) {
        for (Scalar& value : *this) { value = Scalar(c); }
    }
    constexpr void set_zero() { set_constant(Scalar(0)); }
    constexpr void set_ones() { set_constant(Scalar(1)); }
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
    constexpr const Scalar& operator()(Idxs... idxs) const {
        internals::for_each_index_and_args<Order>(
          [&]<int Ns_, typename Slicer__>(Slicer__ s) {
              fdapde_constexpr_assert(s < mdarray_->extent(free_extents_idxs_[Ns_]));
          },
          idxs...);
        return mdarray_->operator[](internals::apply_index_pack<Order>(
          [&]<int... Ns_>() { return ((static_cast<index_t>(idxs) * internal_stride_[Ns_]) + ... + offset_); }));
    }
    constexpr const Scalar& operator[](int index) const { return mdarray_->operator[](offset_ + index); }
    // non-constant access
    template <typename... Idxs>
        requires(std::is_convertible_v<Idxs, index_t> && ...) && (sizeof...(Idxs) == Order && !std::is_const_v<MdArray>)
    constexpr Scalar& operator()(Idxs... idxs) {
        internals::for_each_index_and_args<Order>(
          [&]<int Ns_, typename Slicer__>(Slicer__ s) {
              fdapde_constexpr_assert(s < static_cast<index_t>(mdarray_->extent(free_extents_idxs_[Ns_])));
          },
          idxs...);
        return mdarray_->operator[](internals::apply_index_pack<Order>(
          [&]<int... Ns_>() { return ((static_cast<index_t>(idxs) * internal_stride_[Ns_]) + ... + offset_); }));
    }
    constexpr Scalar& operator[](int index) { return mdarray_->operator[](offset_ + index); }
    constexpr auto matrix() const {
        static_assert((Order == 2 || Order == 1) && ((Slicers != Dynamic) && ...));
        if constexpr (Order == 2) {
            constexpr int rows = MdArray::static_extents[free_extents_idxs_[0]];
            constexpr int cols = MdArray::static_extents[free_extents_idxs_[1]];
            cexpr::Matrix<Scalar, rows, cols> mtx {};
            for (size_t i = 0, n = rows; i < n; ++i) {
                for (size_t j = 0, n = cols; j < n; ++j) { mtx(i, j) = operator()(i, j); }
            }
            return mtx;
        } else {
            constexpr int rows = MdArray::static_extents[free_extents_idxs_[0]];
            cexpr::Vector<Scalar, rows> vec {};
            for (size_t i = 0, n = rows; i < n; ++i) { vec[i] = operator()(i); }
            return vec;
        }
    }
    constexpr const Scalar* data() const requires(contiguous_access) { return mdarray_->data() + offset_; }
    constexpr Scalar* data()
        requires(!std::is_const_v<MdArray> && contiguous_access) {
        return mdarray_->data() + offset_;
    }
    template <typename Src>
        requires(
          std::is_pointer_v<Src> || (fdapde::is_subscriptable<Src, int> &&
	  requires(Src src) {
	    { src.size() } -> std::convertible_to<size_t>;
	  }))
    constexpr MdArraySlice& assign_inplace_from(Src&& src) {
        if constexpr (!std::is_pointer_v<Src>) fdapde_assert(src.size() == size());
        if constexpr (contiguous_access) {
            // for pointer types, this could lead to ub. is caller responsibility to guarantee bounded access
            for (int i = 0; i < size(); ++i) { operator[](i) = src[i]; }
        } else {
            int i = 0;
            for (auto& v : *this) v = src[i++];
        }
        return *this;
    }
   private:
    std::array<index_t, Order> internal_stride_;
    index_t offset_ = 0;
    MdArray* mdarray_;
};

// MdArray: an owning C++20 multidimensional array.
// If Extents_ is such that Extents_::DynamicOrder == 0, MdArray can be used in a constexpr context.
template <typename Scalar_, typename Extents_, typename LayoutPolicy_ = internals::layout_right> class MdArray {
   public:
    using extents_t = Extents_;
    using index_t = typename extents_t::index_t;
    using order_t = typename extents_t::order_t;
    using size_t  = typename extents_t::size_t;  
    using layout_t = LayoutPolicy_;
    using mapping_t = typename layout_t::mapping<extents_t>;
    using Scalar = Scalar_;
    using storage_t = std::conditional_t<
      extents_t::DynamicOrder != 0, std::vector<Scalar>, std::array<Scalar, std::size_t(extents_t::StaticSize)>>;
    static constexpr order_t Order = extents_t::Order;
    static constexpr order_t DynamicOrder = extents_t::DynamicOrder;
    static constexpr std::array<index_t, Order> static_extents = extents_t::static_extents;

    constexpr MdArray()
        requires(std::is_default_constructible_v<extents_t> && std::is_default_constructible_v<storage_t>)
        : extents_(), mapping_(extents_), data_() {
        for (size_t i = 0; i < extents_.size(); ++i) { data_[i] = Scalar(0); }
    }
    constexpr MdArray(const MdArray&) = default;
    constexpr MdArray(MdArray&&) = default;

    template <typename... Exts_>
        requires(extents_t::DynamicOrder != 0 && extents_t::DynamicOrder == sizeof...(Exts_)) &&
                  (std::is_convertible_v<Exts_, index_t> && ...) && std::is_default_constructible_v<storage_t>
    constexpr MdArray(Exts_... exts) : extents_(static_cast<index_t>(exts)...), mapping_(extents_), data_() {
        data_.resize(extents_.size(), Scalar(0));
    }
    template <typename... Exts_>
        requires(extents_t::DynamicOrder != sizeof...(Exts_) && extents_t::Order == sizeof...(Exts_)) &&
                  (std::is_convertible_v<Exts_, index_t> && ...) && std::is_default_constructible_v<storage_t>
    constexpr MdArray(Exts_... exts) : extents_(static_cast<index_t>(exts)...), mapping_(extents_), data_() {
        data_.resize(extents_.size(), Scalar(0));
    }
    template <typename OtherExtents, typename OtherMapping>
        requires(std::is_constructible_v<extents_t, OtherExtents> && std::is_constructible_v<mapping_t, OtherMapping> &&
                 std::is_default_constructible_v<storage_t>)
    constexpr MdArray(const OtherExtents& extents, const OtherMapping& mapping) :
        extents_(extents), mapping_(mapping), data_() {
        if constexpr (extents_t::StaticOrder > 0) {
            for (int i = 0; i < extents_t::Order; ++i) {
                fdapde_constexpr_assert(
                  extents_t::static_extents[i] == Dynamic ||
                  extents_t::static_extents[i] == OtherExtents::static_extents[i]);
            }
        }
    }
    // construct from callable
    template <typename Callable>
        requires(extents_t::DynamicOrder == 0 && std::is_invocable_v<Callable>)
    constexpr explicit MdArray(Callable callable) : extents_(), mapping_(extents_), data_() {
        fdapde_static_assert(
          std::is_convertible_v<typename decltype(std::function {callable})::result_type FDAPDE_COMMA storage_t>,
          CALLABLE_RETURN_TYPE_NOT_CONVERTIBLE_TO_MDARRAY_STORAGE_TYPE);
        data_ = callable();
    }
    template <typename Callable, typename... Exts_>
        requires(extents_t::DynamicOrder != 0 && extents_t::DynamicOrder == sizeof...(Exts_)) &&
                  std::is_invocable_v<Callable> && (std::is_convertible_v<Exts_, index_t> && ...)
    constexpr MdArray(Callable callable, Exts_... exts) :
        extents_(static_cast<index_t>(exts)...), mapping_(extents_), data_() {
        fdapde_static_assert(
          std::is_convertible_v<typename decltype(std::function {callable})::result_type FDAPDE_COMMA storage_t>,
          CALLABLE_RETURN_TYPE_NOT_CONVERTIBLE_TO_MDARRAY_STORAGE_TYPE);
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
        data_(*other.data()), mapping_(other.mapping()), extents_(other.extents()) {
        if constexpr (extents_t::StaticOrder > 0) {
            for (order_t i = 0; i < extents_t::Order; ++i) {
                fdapde_constexpr_assert(
                  extents_t::static_extents[i] == Dynamic ||
                  extents_t::static_extents[i] == OtherExtents::static_extents[i]);
            }
        }
    }
    // construct from MdArraySlice
    template <typename OtherMdArray, int... OtherSlicers>
        requires(Order == OtherMdArray::Order - sizeof...(OtherSlicers) &&
                 std::is_same_v<layout_t, typename OtherMdArray::layout_t> &&
                 std::is_default_constructible_v<storage_t> && std::is_default_constructible_v<mapping_t> &&
                 std::is_default_constructible_v<extents_t>)
    constexpr MdArray(const MdArraySlice<OtherMdArray, OtherSlicers...>& other) :
        extents_(), mapping_(), data_() {
        assign_from_slice_(other);
    }
    // assignemnt
    template <typename OtherMdArray, int... OtherSlicers>
        requires(Order == OtherMdArray::Order - sizeof...(OtherSlicers))
    constexpr MdArray& operator=(const MdArraySlice<OtherMdArray, OtherSlicers...>& other) {
        assign_from_slice_(other);
        return *this;
    }
    template <typename Src>
        requires(
          fdapde::is_subscriptable<Src, int> &&
          requires(Src src) {
              { src.size() } -> std::convertible_to<size_t>;
          })
    constexpr MdArray& assign_inplace_from(const Src& other) {
        for (int i = 0, n = size(); i < n; ++i) { data_[i] = other[i]; }
	return *this;
    }
    // observers
    constexpr size_t size() const { return extents_.size(); }
    constexpr size_t extent(order_t r) const { return extents_.extent(r); }
    constexpr const Scalar* data() const { return data_.data(); }
    constexpr const mapping_t& mapping() const { return mapping_; }
    // modifiers
    constexpr Scalar* data() { return data_.data(); }
    constexpr void set_constant(Scalar c) {
        for (size_t i = 0, n = extents_.size(); i < n; ++i) { data_[i] = Scalar(c); }
    }
    constexpr void set_zero() { set_constant(Scalar(0)); }
    constexpr void set_ones() { set_constant(Scalar(1)); }
    // iterator
    template <typename MdArray_> struct iterator {
        constexpr iterator() noexcept = default;
        template <typename... Idxs>
            requires(std::is_convertible_v<Idxs, index_t> && ...) && (sizeof...(Idxs) == Order)
        constexpr iterator(MdArray_* mdarray, Idxs... idxs) noexcept :
            mdarray_(mdarray), index_({static_cast<index_t>(idxs)...}) { }
        constexpr iterator(MdArray_* mdarray, const std::array<index_t, Order>& index) noexcept :
            mdarray_(mdarray), index_(index) { }

        constexpr iterator& operator++() noexcept {
            index_[Order - 1]++;
            order_t i = Order - 1;
            while (i > 0 && index_[i] >= mdarray_->extent(i)) {
                index_[i] = 0;
                index_[--i]++;
            }
            return *this;
        }
        // const access
        constexpr const Scalar& operator*()  const { return fetch_at_(index_); }
        constexpr const Scalar* operator->() const { return std::addressof(fetch_at_(index_)); }
        // non-const access
        constexpr Scalar& operator*()  requires(!std::is_const_v<MdArray>) { return fetch_at_(index_); }
        constexpr Scalar* operator->() requires(!std::is_const_v<MdArray>) { return std::addressof(fetch_at_(index_)); }
        // comparison
        constexpr friend bool operator==(const iterator& lhs, const iterator& rhs) { return lhs.index_ == rhs.index_; }
        constexpr friend bool operator!=(const iterator& lhs, const iterator& rhs) { return lhs.index_ != rhs.index_; }
       private:
        template <typename IndexType> constexpr decltype(auto) fetch_at_(IndexType&& index) {
            return internals::apply_index_pack<Order>(
              [&]<int... Ns_>() -> decltype(auto) { return mdarray_->operator()(((void)Ns_, index_[Ns_])...); });
        }
        MdArray_* mdarray_;
        std::array<index_t, Order> index_;
    };
    // const iterators
    constexpr iterator<const MdArray<Scalar, extents_t, layout_t>> begin() const noexcept {
        return internals::apply_index_pack<Order>(
          [&]<int... Ns_> { return iterator<const MdArray<Scalar, extents_t, layout_t>> {this, ((void)Ns_, 0)...}; });
    }
    constexpr iterator<const MdArray<Scalar, extents_t, layout_t>> end() const noexcept {
        return internals::apply_index_pack<Order - 1>([&]<int... Ns_> {
            return iterator<const MdArray<Scalar, extents_t, layout_t>> {this, extents_.extent(0), ((void)Ns_, 0)...};
        });
    }
    // non-const iterators
    constexpr iterator<MdArray<Scalar, extents_t, layout_t>> begin() noexcept {
        return internals::apply_index_pack<Order>(
          [&]<int... Ns_> { return iterator<MdArray<Scalar, extents_t, layout_t>> {this, ((void)Ns_, 0)...}; });
    }
    constexpr iterator<MdArray<Scalar, extents_t, layout_t>> end() noexcept {
        return internals::apply_index_pack<Order - 1>([&]<int... Ns_> {
            return iterator<MdArray<Scalar, extents_t, layout_t>> {this, extents_.extent(0), ((void)Ns_, 0)...};
        });
    }
    // resize dynamic MdArray (only dynamic extents). allocated memory is left uninitialized
    template <typename... Exts_>
        requires(extents_t::DynamicOrder != 0 && (std::is_convertible_v<Exts_, index_t> && ...))
    constexpr void resize(Exts_... exts) {
        fdapde_static_assert(
          sizeof...(Exts_) == extents_t::DynamicOrder,
          YOU_SUPPLIED_A_WRONG_NUMBER_OF_ARGUMENTS_TO_RESIZE__NUMBER_OF_ARGUMENTS_MUST_MATCH_NUMBER_OF_DYNAMIC_EXTENTS);
        extents_.resize(static_cast<index_t>(exts)...);
        mapping_ = mapping_t(extents_);
        data_.resize(size());   // re-allocate space
    }
    // constant access
    template <typename... Idxs>
        requires(std::is_convertible_v<Idxs, index_t> && ...) && (sizeof...(Idxs) == extents_t::Order)
    constexpr const Scalar& operator()(Idxs... idxs) const {
        return data_[mapping_(static_cast<index_t>(idxs)...)];
    }
    constexpr const Scalar& operator[](index_t i) const { return data_[i]; }
    // non-constant access
    template <typename... Idxs>
        requires(std::is_convertible_v<Idxs, index_t> && ...) && (sizeof...(Idxs) == extents_t::Order)
    constexpr Scalar& operator()(Idxs... idxs) {
        return data_[mapping_(static_cast<index_t>(idxs)...)];
    }
    constexpr Scalar& operator[](index_t i) { return data_[i]; }
    // block-access operations
    template <typename... Slicers_>   // dynamic-sized
        requires(sizeof...(Slicers_) == Order) &&
                ((internals::is_integer_v<Slicers_> || internals::is_pair_v<Slicers_>) && ...)
    constexpr auto block(Slicers_... slicers) {
        return fdapde::submdarray(*this, slicers...);
    }
    template <typename... Slicers_>   // dynamic-sized
        requires(sizeof...(Slicers_) == Order) &&
                ((internals::is_integer_v<Slicers_> || internals::is_pair_v<Slicers_>) && ...)
    constexpr auto block(Slicers_... slicers) const {
        return fdapde::submdarray(*this, slicers...);
    }
    template <int... Exts_, typename... Slicers_>   // static-sized (const access)
        requires(sizeof...(Exts_) == Order && sizeof...(Exts_) == sizeof...(Slicers_)) &&
                (internals::is_integer_v<Slicers_> && ...)
    constexpr MdArrayBlock<const MdArray<Scalar, extents_t, layout_t>, MdExtents<Exts_...>> block(
      Slicers_... slicers) const {
        // check block is within MdArray
        std::array<index_t, Order> static_block_extents {Exts_...};
        internals::for_each_index_and_args<Order>(
          [&]<int Ns_, typename Slicer__>(Slicer__ s) {
              fdapde_constexpr_assert(static_block_extents[Ns_] + s < 1 + extent(Ns_));
          },
          slicers...);
        return MdArrayBlock<MdArray<Scalar, extents_t, layout_t>, MdExtents<Exts_...>>(
          this, MdExtents<Exts_...>(), slicers...);
    }
    template <int... Exts_, typename... Slicers_>   // static-sized (non-const access)
        requires(sizeof...(Exts_) == Order && sizeof...(Exts_) == sizeof...(Slicers_)) &&
                (internals::is_integer_v<Slicers_> && ...)
    constexpr MdArrayBlock<MdArray<Scalar, extents_t, layout_t>, MdExtents<Exts_...>> block(Slicers_... slicers) {
        // check block is within MdArray
        std::array<index_t, Order> static_block_extents {Exts_...};
        internals::for_each_index_and_args<Order>(
          [&]<int Ns_, typename Slicer__>(Slicer__ s) {
              fdapde_constexpr_assert(static_block_extents[Ns_] + s < 1 + extent(Ns_));
          },
          slicers...);
        return MdArrayBlock<MdArray<Scalar, extents_t, layout_t>, MdExtents<Exts_...>>(
          this, MdExtents<Exts_...>(), slicers...);
    }
    // slicing operations
    template <int... Slicers, typename... Slicers__>
        requires(std::is_convertible_v<Slicers__, int> && ...)
    constexpr auto slice(Slicers__... slicers) {
        return MdArraySlice<MdArray<Scalar, extents_t, layout_t>, Slicers...>(this, slicers...);
    }
    template <int... Slicers, typename... Slicers__>
        requires(std::is_convertible_v<Slicers__, int> && ...)
    constexpr auto slice(Slicers__... slicers) const {
        return MdArraySlice<const MdArray<Scalar, extents_t, layout_t>, Slicers...>(this, slicers...);
    }
    constexpr auto matrix() const {
        static_assert(Order == 2);
        if constexpr (Order == 2) { static_assert(static_extents[0] != Dynamic && static_extents[1] != Dynamic); }
        return cexpr::Map<const Scalar, static_extents[0], static_extents[1], RowMajor>(data());
    }
   private:
    template <typename OtherMdArray, int... OtherSlicers>
        requires(Order == OtherMdArray::Order - sizeof...(OtherSlicers))
    constexpr void assign_from_slice_(const MdArraySlice<OtherMdArray, OtherSlicers...>& other) {
        using slice_t = MdArraySlice<OtherMdArray, OtherSlicers...>;
        if constexpr (extents_t::StaticOrder > 0) {
            for (order_t i = 0; i < slice_t::Order; ++i) {
                order_t extent_ = slice_t::free_extents_idxs_[i];
                fdapde_constexpr_assert(
                  extents_t::static_extents[i] == Dynamic ||
                  extents_t::static_extents[i] == OtherMdArray::static_extents[slice_t::free_extents_idxs_[i]]);
            }
        }
        if constexpr (extents_t::DynamicOrder > 0) {
            if (size() != other.size()) {
                internals::apply_index_pack<Order>(
                  [&]<int... Ns_> { extents_.resize(static_cast<index_t>(other.extent(Ns_))...); });
                data_.resize(size());   // re-allocate space
            }
            mapping_ = mapping_t(extents_);
        }
        if constexpr (internals::slices_to_contiguous_memory<typename OtherMdArray::mapping_t, OtherSlicers...>()) {
            for (int i = 0, n = size(); i < n; ++i) { data_[i] = other[i]; }   // copy from contiguous memory
        } else {
            int i = 0;
            for (auto value : other) { data_[i++] = value; }
        }
    }

    extents_t extents_ {};
    mapping_t mapping_ {};
    storage_t data_ {};
};

}   // namespace fdapde

#endif   // __MDARRAY_H__
