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
#include "../utils/traits.h"
#include "../utils/assert.h"
#include "../utils/symbols.h"

namespace fdapde {

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
  
// internals support for MdArray blocks

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
        return slicer;
    } else if constexpr (
      is_pair_v<Slicer> && std::convertible_to<std::tuple_element_t<0, Slicer>, int> &&
      std::convertible_to<std::tuple_element_t<1, Slicer>, int>) {
        return static_cast<int>(std::get<0>(slicer));
    }
}

// checks for the presence of duplicated entries in index pack
template <typename... IndexType>
    requires(std::is_convertible_v<IndexType, int> && ...) && (sizeof...(IndexType) > 0)
constexpr bool no_duplicates_in_pack(IndexType... idxs) {
    std::array<int, sizeof...(IndexType)> tmp {idxs...};
    for (int i = 0; i < sizeof...(IndexType) - 1; ++i) {
        for (int j = i + 1; j < sizeof...(IndexType); ++j) {
            if (tmp[i] == tmp[j]) return false;
        }
    }
    return true;
}

}   // namespace internals

template <int... Extents_> class extents {
    fdapde_static_assert(
      sizeof...(Extents_) > 0 && internals::apply_index_pack<sizeof...(Extents_)>([]<int... Ns_>() {
          return ((static_cast<int>(Extents_) > 0 || static_cast<int>(Extents_) == Dynamic) && ...);
      }),
      EXTENTS_OF_ZERO_ORDER_OR_WITH_NEGATIVE_VALUES_ARE_ILL_FORMED);
   public:
    static constexpr int Order = sizeof...(Extents_);
    static constexpr int DynamicOrder = ((Extents_ == Dynamic) + ... + 0);
    static constexpr int StaticOrder  = Order - DynamicOrder;
    static constexpr int StaticSize   = DynamicOrder == 0 ? (Extents_ * ... * 1) : Dynamic;
    static constexpr std::array<int, Order> static_extents {Extents_...};
  
    constexpr extents() requires(DynamicOrder == 0) = default;
    template <typename... Exts_>
        requires(std::is_convertible_v<Exts_, int> && ...) && (DynamicOrder > 0 && sizeof...(Exts_) == DynamicOrder)
    constexpr extents(Exts_... exts) {
        std::array<int, DynamicOrder> exts_ {static_cast<int>(exts)...};
        for (int i = 0; i < DynamicOrder; ++i) { extents_[dynamic_extent_map_[i]] = exts_[i]; }
    }
    template <typename... Exts_>
        requires(std::is_convertible_v<Exts_, int> && ...) &&
                (sizeof...(Exts_) != DynamicOrder && sizeof...(Exts_) == Order)
    constexpr extents(Exts_... exts) {
        std::array<int, Order> exts_ {static_cast<int>(exts)...};
        for (int i = 0, j = 0; i < Order; ++i) {
            if (static_extents[i] == Dynamic) {
                extents_[dynamic_extent_map_[j++]] = exts_[i];
            } else {
                // check supplied extents match static, non-dynamic, ones
                fdapde_constexpr_assert(exts_[i] == static_cast<int>(static_extents[i]));
            }
        }
    }
    template <std::size_t Size, typename Exts>
        requires(Size == Order && std::is_convertible_v<Exts, int> && Order == DynamicOrder)
    constexpr extents(const std::array<Exts, Size>& extents) : extents_(extents) { }

    constexpr int extent(int i) const noexcept { return extents_[i]; }
    constexpr int size() const { return l_prod(Order); }
    template <typename... Exts_>
        requires(std::is_convertible_v<Exts_, int> && ...) && (sizeof...(Exts_) == DynamicOrder)
    constexpr void resize(Exts_... exts) {
        std::array<int, sizeof...(Exts_)> exts_ {static_cast<int>(exts)...};
        for (int i = 0; i < sizeof...(Exts_); ++i) { extents_[dynamic_extent_map_[i]] = exts_[i]; }
	return;
    }
    // given extents (i, j, k, ..., h), computes the product of extents from left to right up to Order r
    constexpr int l_prod(int r) const requires(Order > 0) {
        int prod = 1;
        for (int i = 0; i < r; ++i) { prod *= extents_[i]; }
        return prod;
    }
    // given extents (i, j, k, ..., h), computes the product of extents from right to left up to Order r
    constexpr int r_prod(int r) const requires(Order > 0) {
        int prod = 1;
        for (int i = r; i < Order - 1; ++i) { prod *= extents_[i]; }
        return prod;
    }
   private:
    // internal utilities
    static constexpr std::array<int, DynamicOrder> dynamic_extent_map_ {[]() {
        std::array<int, DynamicOrder> result_ {};
        for (int i = 0, j = 0; i < Order; ++i) {
            if (static_extents[i] == Dynamic) { result_[j++] = i; }
        }
        return result_;
    }()};
    std::array<int, Order> extents_ {Extents_...};
};

template <int N>
using full_dynamic_extent_t = std::decay_t<decltype(internals::apply_index_pack<N>(
  []<int... Ns_> { return extents<(Ns_, Dynamic)...> {Ns_...}; }))>;

namespace internals {

// memory layout strategies for MdArray (insipred from C++23 std::mdspan<>)

struct layout_left {
    template <typename Extents> struct mapping {
        static constexpr int Order = Extents::Order;
        using extents_type = Extents;
        using layout_type = layout_left;

        constexpr mapping() noexcept = default;
        constexpr mapping& operator=(const mapping&) noexcept = default;
        constexpr mapping(const Extents& extents) noexcept
            requires(Order > 0)
            : extents_(extents), strides_() {
            for (int i = 0; i < Order + 1; ++i) {
                strides_[i] = 1;
                for (int j = 0; j < i; ++j) { strides_[i] *= extents_.extent(j); }
            }
        }

        template <typename... Idxs>   // index pack to mdarray memory index
            requires(std::is_convertible_v<Idxs, int> && ...) && (sizeof...(Idxs) == Order)
        constexpr int operator()(Idxs... idx) const {
            fdapde_constexpr_assert(internals::is_multidimensional_index_in_extent(extents_, static_cast<int>(idx)...));
            return internals::apply_index_pack<Order>(
              [&]<int... Ns_>() { return ((static_cast<int>(idx) * strides_[Ns_]) + ... + 0); });
        }
        template <int IndexSize, typename IndexType>   // array index to mdarray memory index
            requires(std::is_convertible_v<IndexType, int> && IndexSize == Order)
        constexpr int operator()(const std::array<IndexType, IndexSize>& arr) {
            int idx = 0;
            for (int i = 0; i < Order; ++i) { idx += static_cast<int>(arr[i]) * strides_[i]; }
            return idx;
        }
        constexpr int stride(int r) const requires(Order > 0) {
            fdapde_constexpr_assert(r < Order);
	    return strides_[r];
        }
        template <typename OtherExtents>
            requires(Order == OtherExtents::Order)
        friend constexpr bool operator==(const mapping& lhs, const mapping<OtherExtents>& rhs) {
            return lhs.extents_ == rhs.extents_;
        }
       private:
        std::array<int, Order + 1> strides_ {};
        Extents extents_ {};
    };
};

struct layout_right {
    template <typename Extents> struct mapping {
        static constexpr int Order = Extents::Order;
        using extents_type = Extents;
        using layout_type = layout_right;

        constexpr mapping() noexcept = default;
        constexpr mapping& operator=(const mapping&) noexcept = default;
        constexpr mapping(const Extents& extents) noexcept
            requires(Order > 0)
            : extents_(extents), strides_() {
            for (int i = 0; i < Order + 1; ++i) {
                strides_[i] = 1;
                for (int j = i; j < Order - 1; ++j) { strides_[i] *= extents_.extent(j); }
            }
        }

        template <typename... Idxs>   // index pack to mdarray memory index
            requires(std::is_convertible_v<Idxs, int> && ...) && (sizeof...(Idxs) == Order)
        constexpr int operator()(Idxs... idx) const {
            fdapde_constexpr_assert(internals::is_multidimensional_index_in_extent(extents_, static_cast<int>(idx)...));
            return internals::apply_index_pack<Order>(
              [&]<int... Ns_>() { return ((static_cast<int>(idx) * strides_[Ns_]) + ... + 0); });
        }
        template <int IndexSize, typename IndexType>   // array index to mdarray memory index
            requires(std::is_convertible_v<IndexType, int> && IndexSize == Order)
        constexpr int operator()(const std::array<IndexType, IndexSize>& arr) {
            int idx = 0;
            for (int i = 0; i < Order; ++i) { idx += static_cast<int>(arr[i]) * strides_[i]; }
            return idx;
        }
        constexpr int stride(int r) const requires(Order > 0) {
            fdapde_constexpr_assert(r < Order);
	    return strides_[r];
        }
        template <typename OtherExtents>
            requires(Order == OtherExtents::Order)
        friend constexpr bool operator==(const mapping& lhs, const mapping<OtherExtents>& rhs) {
            return lhs.extents_ == rhs.extents_;
        }
       private:
        std::array<int, Order + 1> strides_ {};
        Extents extents_ {};
    };
};

}   // namespace internals

// a multidimensional view of a multidimensional MdArray
template <typename MdArray_, typename BlkExtents_> class MdArrayBlock { 
   public:
    static constexpr int Order = MdArray_::Order;
    using Scalar  = typename MdArray_::Scalar;
    using BlkExtents = BlkExtents_;

    constexpr MdArrayBlock() noexcept = default;
    template <typename... Slicers_>
        requires(sizeof...(Slicers_) == Order && BlkExtents_::Order == MdArray_::Order) &&
                  ((internals::is_integer_v<Slicers_> || internals::is_pair_v<Slicers_>) && ...)
    constexpr MdArrayBlock(MdArray_* mdarray, BlkExtents_ blk_extents, Slicers_&&... slicers) noexcept :
        extents_(blk_extents), mdarray_(mdarray) {
        internals::for_each_index_in_pack<Order>(
          [&]<int Ns_>() mutable { offset_[Ns_] = internals::smallest_index_in_mdarray_blk<Ns_>(slicers...); });
    }

    // observers
    constexpr int size() const { return extents_.size(); }
    constexpr int extent(int r) const {
        fdapde_constexpr_assert(r < Order);
        return extents_.extent(r);
    }
    constexpr int order() const { return Order; }
    // iterator
    template <typename MdArrayBlock_> struct iterator {
        constexpr iterator() noexcept = default;
        template <typename... Idxs>
            requires(std::is_convertible_v<Idxs, int> && ...) && (sizeof...(Idxs) == Order)
        constexpr iterator(MdArrayBlock_* mdarray, Idxs... idxs) noexcept : mdarray_(mdarray), index_({idxs...}) { }
        constexpr iterator(MdArrayBlock_* mdarray, const std::array<int, Order>& index) noexcept :
            mdarray_(mdarray), index_(index) { }

        constexpr iterator& operator++() noexcept {
            index_[Order - 1]++;
            int i = Order - 1;
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
        constexpr Scalar& operator*()  requires(!std::is_const_v<MdArray_>) { return fetch_at_(index_); }
        constexpr Scalar* operator->() requires(!std::is_const_v<MdArray_>) { return std::addressof(fetch_at(index_)); }
        // comparison
        constexpr friend bool operator==(const iterator& lhs, const iterator& rhs) { return lhs.index_ == rhs.index_; }
        constexpr friend bool operator!=(const iterator& lhs, const iterator& rhs) { return lhs.index_ != rhs.index_; }
       private:
        template <typename IndexType> constexpr decltype(auto) fetch_at_(IndexType&& index) {
            return internals::apply_index_pack<Order>(
              [&]<int... Ns_>() -> decltype(auto) { return mdarray_->operator()((Ns_, index[Ns_])...); });
        }
        MdArrayBlock_* mdarray_;
        std::array<int, Order> index_;
    };
    // const iterators
    constexpr iterator<const MdArrayBlock<MdArray_, BlkExtents_>> begin() const noexcept {
        return internals::apply_index_pack<Order>(
          [&]<int... Ns_> { return iterator<const MdArrayBlock<MdArray_, BlkExtents_>> {this, (Ns_, 0)...}; });
    }
    constexpr iterator<const MdArrayBlock<MdArray_, BlkExtents_>> end() const noexcept {
        return internals::apply_index_pack<Order - 1>([&]<int... Ns_> {
            return iterator<const MdArrayBlock<MdArray_, BlkExtents_>> {this, extents_.extent(0), (Ns_, 0)...};
        });
    }
    // non-const iterators
    constexpr iterator<MdArrayBlock<MdArray_, BlkExtents_>> begin() noexcept {
        return internals::apply_index_pack<Order>(
          [&]<int... Ns_> { return iterator<MdArrayBlock<MdArray_, BlkExtents_>> {this, (Ns_, 0)...}; });
    }
    constexpr iterator<MdArrayBlock<MdArray_, BlkExtents_>> end() noexcept {
        return internals::apply_index_pack<Order - 1>([&]<int... Ns_> {
            return iterator<MdArrayBlock<MdArray_, BlkExtents_>> {this, extents_.extent(0), (Ns_, 0)...};
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
        requires(std::is_convertible_v<Idxs, int> && ...) && (sizeof...(Idxs) == BlkExtents_::Order)
    constexpr Scalar operator()(Idxs... idxs) const {
        fdapde_constexpr_assert(internals::is_multidimensional_index_in_extent(extents_, static_cast<int>(idxs)...));
        return mdarray_->operator[](internals::apply_index_pack<Order>([&]<int... Ns_>() {
            return (((static_cast<int>(idxs) + offset_[Ns_]) * mdarray_->mapping().stride(Ns_)) + ... + 0);
        }));
    }
    // non-constant access
    template <typename... Idxs>
        requires(std::is_convertible_v<Idxs, int> && ...) &&
                (sizeof...(Idxs) == BlkExtents_::Order && !std::is_const_v<MdArray_>)
    constexpr Scalar& operator()(Idxs... idxs) {
        fdapde_constexpr_assert(internals::is_multidimensional_index_in_extent(extents_, static_cast<int>(idxs)...));
        return mdarray_->operator[](internals::apply_index_pack<Order>([&]<int... Ns_>() {
            return (((static_cast<int>(idxs) + offset_[Ns_]) * mdarray_->mapping().stride(Ns_)) + ... + 0);
        }));
    }
   private:
    std::array<int, Order> offset_ {};
    BlkExtents extents_;
    MdArray_* mdarray_;
};

template <typename MdArray_, typename... Slicers>
    requires(sizeof...(Slicers) == std::decay_t<MdArray_>::Order && std::is_reference_v<MdArray_>) &&
            ((internals::is_integer_v<Slicers> || internals::is_pair_v<Slicers>) && ...)
constexpr auto submdarray(MdArray_&& mdarray, Slicers... slicers) {
    using MdArray = std::remove_reference_t<MdArray_>;
    constexpr int Order = std::decay_t<MdArray>::Order;
    // check block is within MdArray
    internals::for_each_index_and_args<Order>(
      [&]<int Ns_, typename Slicer_>(Slicer_ s) {
          if constexpr (internals::is_integer_v<Slicer_>) {
              fdapde_constexpr_assert(s < mdarray.extent(Ns_));
          } else if constexpr (internals::is_pair_v<Slicer_>) {
              fdapde_constexpr_assert(std::get<1>(s) > std::get<0>(s) && std::get<1>(s) < mdarray.extent(Ns_));
          }
      },
      slicers...);
    // prepare block extents
    std::array<int, Order> blk_extents {};
    internals::for_each_index_and_args<Order>(
      [&]<int Ns_, typename Slicer_>(Slicer_ s) mutable {
          if constexpr (internals::is_integer_v<Slicer_>) {
              blk_extents[Ns_] = 1;
          } else if constexpr (internals::is_pair_v<Slicer_>) {
              blk_extents[Ns_] = (std::get<1>(s) - std::get<0>(s)) + 1;
          }
      },
      slicers...);
    return MdArrayBlock<MdArray, full_dynamic_extent_t<Order>>(
      &mdarray, full_dynamic_extent_t<Order>(blk_extents), slicers...);
}

// a slice is a lower-order view of an MdArray in which one dimension has been fixed
template <typename MdArray_, int... Slicers_> class MdArraySlice {
    static_assert(sizeof...(Slicers_) < MdArray_::Order && ((Slicers_ >= 0 && Slicers_ < MdArray_::Order) && ...));
   public:
    static constexpr int Order = MdArray_::Order - sizeof...(Slicers_);
    static constexpr std::array<int, sizeof...(Slicers_)> static_slicers {Slicers_...};
    fdapde_static_assert(internals::no_duplicates_in_pack(Slicers_...), SLICING_DIRECTIONS_MUST_BE_UNIQUE);
    // inv_map_ = {0, \ldots, MdArray_::Order - 1} \setminus static_slicers
    static constexpr std::array<int, Order> free_extents_ {[]() {
        std::array<int, Order> map {};
        for (int i = 0, j = 0; i < MdArray_::Order; ++i) {
            if (std::find(static_slicers.begin(), static_slicers.end(), i) == static_slicers.end()) { map[j++] = i; }
        }
        return map;
    }()};
    using Scalar = MdArray_::Scalar;

    template <typename... Slicers>
        requires(sizeof...(Slicers) == sizeof...(Slicers_)) && (internals::is_integer_v<Slicers> && ...)
    constexpr MdArraySlice(MdArray_* mdarray, Slicers... slicers) noexcept :
        slicers_({slicers...}), mdarray_(mdarray), offset_(1) {
        internals::for_each_index_and_args<sizeof...(Slicers)>(
          [&]<int Ns_, typename Slicer__>(Slicer__ s) {
              fdapde_constexpr_assert(s < mdarray_->extent(extents_mapping_[Ns_]));
          },
          slicers...);
	// compute offset in linearized memory due to slicing
        for (int i = 0; i < sizeof...(Slicers); ++i) {
            offset_ *= slicers_[i] * mdarray_->mapping().stride(static_slicers[i]);
        }
    }
    // observers
    constexpr int size() const {
        int size_ = 1;
        for (int i = 0; i < Order; ++i) { size_ *= mdarray_->extent(static_slicers[i]); }
        return size_;
    }
    constexpr int extent(int r) const {
        fdapde_constexpr_assert(r < Order);
        return mdarray_->extent(static_slicers[r]);
    }
    // iterator
    template <typename MdArraySlice_> struct iterator {
        constexpr iterator() noexcept = default;
        template <typename... Idxs>
            requires(std::is_convertible_v<Idxs, int> && ...) && (sizeof...(Idxs) == Order)
        constexpr iterator(MdArraySlice_* mdarray, Idxs... idxs) noexcept : mdarray_(mdarray), index_({idxs...}) { }
        constexpr iterator(MdArraySlice_* mdarray, const std::array<int, Order>& index) noexcept :
            mdarray_(mdarray), index_(index) { }

        constexpr iterator& operator++() noexcept {
            index_[Order - 1]++;
            int i = Order - 1;
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
        constexpr Scalar& operator*()  requires(!std::is_const_v<MdArray_>) { return fetch_at_(index_); }
        constexpr Scalar* operator->() requires(!std::is_const_v<MdArray_>) { return std::addressof(fetch_at(index_)); }
        // comparison
        constexpr friend bool operator==(const iterator& lhs, const iterator& rhs) { return lhs.index_ == rhs.index_; }
        constexpr friend bool operator!=(const iterator& lhs, const iterator& rhs) { return lhs.index_ != rhs.index_; }
       private:
        template <typename IndexType> constexpr decltype(auto) fetch_at_(IndexType&& index) {
            return internals::apply_index_pack<Order>(
              [&]<int... Ns_>() -> decltype(auto) { return mdarray_->operator()((Ns_, index[Ns_])...); });
        }
        MdArraySlice_* mdarray_;
        std::array<int, Order> index_;
    };
    // const iterators
    constexpr iterator<const MdArraySlice<MdArray_, Slicers_...>> begin() const noexcept {
        return internals::apply_index_pack<Order>(
          [&]<int... Ns_> { return iterator<const MdArraySlice<MdArray_, Slicers_...>> {this, (Ns_, 0)...}; });
    }
    constexpr iterator<const MdArraySlice<MdArray_, Slicers_...>> end() const noexcept {
        return internals::apply_index_pack<Order - 1>([&]<int... Ns_> {
            return iterator<const MdArraySlice<MdArray_, Slicers_...>> {this, extent(0), (Ns_, 0)...};
        });
    }
    // non-const iterators
    constexpr iterator<MdArraySlice<MdArray_, Slicers_...>> begin() noexcept {
        return internals::apply_index_pack<Order>(
          [&]<int... Ns_> { return iterator<MdArraySlice<MdArray_, Slicers_...>> {this, (Ns_, 0)...}; });
    }
    constexpr iterator<MdArraySlice<MdArray_, Slicers_...>> end() noexcept {
        return internals::apply_index_pack<Order - 1>(
          [&]<int... Ns_> { return iterator<MdArraySlice<MdArray_, Slicers_...>> {this, extent(0), (Ns_, 0)...}; });
    }
    // modifiers
    constexpr void set_constant(Scalar c) {
        for (Scalar& value : *this) { value = Scalar(c); }
    }
    constexpr void set_zero() { set_constant(Scalar(0)); }
    constexpr void set_ones() { set_constant(Scalar(1)); }
    // constant access
    template <typename... Idxs>
        requires(std::is_convertible_v<Idxs, int> && ...) && (sizeof...(Idxs) == Order)
    constexpr Scalar operator()(Idxs... idxs) const {
        internals::for_each_index_and_args<Order>(
          [&]<int Ns_, typename Slicer__>(Slicer__ s) { fdapde_constexpr_assert(idxs < extents(Ns_)); }, idxs...);
        return mdarray_->operator[](internals::apply_index_pack<Order>([&]<int... Ns_>() {
            return ((static_cast<int>(idxs) * mdarray_->mapping().stride(free_extents_[Ns_])) + ... + offset_);
        }));
    }
    // non-constant access
    template <typename... Idxs>
        requires(std::is_convertible_v<Idxs, int> && ...) && (sizeof...(Idxs) == Order && !std::is_const_v<MdArray_>)
    constexpr Scalar& operator()(Idxs... idxs) {
        internals::for_each_index_and_args<Order>(
          [&]<int Ns_, typename Slicer__>(Slicer__ s) { fdapde_constexpr_assert(idxs < extents(Ns_)); }, idxs...);
        return mdarray_->operator[](internals::apply_index_pack<Order>([&]<int... Ns_>() {
            return ((static_cast<int>(idxs) * mdarray_->mapping().stride(free_extents_[Ns_])) + ... + offset_);
        }));
    }
   private:
    int offset_ = 0;
    std::array<int, sizeof...(Slicers_)> slicers_;
    MdArray_* mdarray_;
};

// MdArray: an owning C++20 multidimensional array.
// If Extents_ is such that Extents_::DynamicOrder == 0, MdArray can be used in a constexpr context.
template <typename Scalar_, typename Extents_, typename LayoutPolicy_ = internals::layout_right> class MdArray {
   public:
    static constexpr int Order = Extents_::Order;
    using Scalar = Scalar_;
    using Extents = Extents_;
    using Storage = std::conditional_t<
      Extents_::DynamicOrder != 0, std::vector<Scalar>, std::array<Scalar, std::size_t(Extents_::StaticSize)>>;
    using LayoutPolicy = LayoutPolicy_;
    using Mapping = typename LayoutPolicy::mapping<Extents>;

    constexpr MdArray()
        requires(Extents_::DynamicOrder == 0) && std::is_default_constructible_v<Extents_> &&
                  std::is_default_constructible_v<Storage>
        : extents_(), mapping_(extents_), data_() {
        for (int i = 0; i < extents_.size(); ++i) { data_[i] = Scalar(0); }
    }
    constexpr MdArray(const MdArray&) = default;
    constexpr MdArray(MdArray&&) = default;

    template <typename... Exts_>
        requires(Extents_::DynamicOrder != 0 && Extents_::DynamicOrder == sizeof...(Exts_)) &&
                  (std::is_convertible_v<Exts_, int> && ...) && std::is_default_constructible_v<Storage>
    constexpr MdArray(Exts_... exts) : extents_(static_cast<int>(exts)...), mapping_(extents_), data_() {
        data_.resize(extents_.size(), Scalar(0));
    }
    template <typename... Exts_>
        requires(Extents_::DynamicOrder != sizeof...(Exts_) && Extents_::Order == sizeof...(Exts_)) &&
                  (std::is_convertible_v<Exts_, int> && ...) && std::is_default_constructible_v<Storage>
    constexpr MdArray(Exts_... exts) : extents_(static_cast<int>(exts)...), mapping_(extents_), data_() {
        data_.resize(extents_.size(), Scalar(0));
    }
    template <typename OtherExtents, typename OtherMapping>
        requires(std::is_constructible_v<Extents, OtherExtents> && std::is_constructible_v<Mapping, OtherMapping> &&
                 std::is_default_constructible_v<Storage>)
    constexpr MdArray(const OtherExtents& extents, const OtherMapping& mapping) :
        extents_(extents), mapping_(mapping), data_() {
        if constexpr (Extents_::StaticOrder > 0) {
            for (int i = 0; i < Extents_::Order; ++i) {
                fdapde_constexpr_assert(
                  Extents_::static_extents[i] == Dynamic ||
                  Extents_::static_extents[i] == OtherExtents::static_extents[i]);
            }
        }
    }
    // construct from callable
    template <typename Callable>
        requires(Extents_::DynamicOrder == 0 && std::is_invocable_v<Callable>)
    constexpr explicit MdArray(Callable callable) : extents_(), mapping_(extents_), data_() {
        fdapde_static_assert(
          std::is_convertible_v<typename decltype(std::function {callable})::result_type FDAPDE_COMMA Storage>,
          CALLABLE_RETURN_TYPE_NOT_CONVERTIBLE_TO_MDARRAY_STORAGE_TYPE);
        data_ = callable();
    }
    template <typename Callable, typename... Exts_>
        requires(Extents_::DynamicOrder != 0 && Extents_::DynamicOrder == sizeof...(Exts_)) &&
                  std::is_invocable_v<Callable> && (std::is_convertible_v<Exts_, int> && ...)
    constexpr MdArray(Callable callable, Exts_... exts) :
        extents_(static_cast<int>(exts)...), mapping_(extents_), data_() {
        fdapde_static_assert(
          std::is_convertible_v<typename decltype(std::function {callable})::result_type FDAPDE_COMMA Storage>,
          CALLABLE_RETURN_TYPE_NOT_CONVERTIBLE_TO_MDARRAY_STORAGE_TYPE);
        data_ = callable();
        fdapde_constexpr_assert(extents_.size() == data_.size());
    }
    // construct from other MdArray
    template <typename OtherScalar, typename OtherExtents, typename OtherLayoutPolicy>
        requires(std::is_convertible_v<OtherScalar, Scalar> &&
                 std::is_constructible_v<Mapping, const typename OtherLayoutPolicy::template mapping<OtherExtents>&> &&
                 std::is_constructible_v<Extents, const OtherExtents&>)
    constexpr MdArray(const MdArray<OtherScalar, OtherExtents, OtherLayoutPolicy>& other) :
        data_(*other.data()), mapping_(other.mapping()), extents_(other.extents()) {
        if constexpr (Extents_::StaticOrder > 0) {
            for (int i = 0; i < Extents_::Order; ++i) {
                fdapde_constexpr_assert(
                  Extents_::static_extents[i] == Dynamic ||
                  Extents_::static_extents[i] == OtherExtents::static_extents[i]);
            }
        }
    }
    // observers
    constexpr int size() const { return extents_.size(); }
    constexpr int extent(int r) const { return extents_.extent(r); }
    constexpr const Storage* data() const { return &data_; }
    constexpr const Mapping& mapping() const { return mapping_; }
    // modifiers
    constexpr Storage* data() { return &data_; }
    constexpr void set_constant(Scalar c) {
        for (int i = 0, n = extents_.size(); i < n; ++i) { data_[i] = Scalar(c); }
    }
    constexpr void set_zero() { set_constant(Scalar(0)); }
    constexpr void set_ones() { set_constant(Scalar(1)); }
    // iterator
    template <typename MdArray_> struct iterator {
        constexpr iterator() noexcept = default;
        template <typename... Idxs>
            requires(std::is_convertible_v<Idxs, int> && ...) && (sizeof...(Idxs) == Order)
        constexpr iterator(MdArray_* mdarray, Idxs... idxs) noexcept : mdarray_(mdarray), index_({idxs...}) { }
        constexpr iterator(MdArray_* mdarray, const std::array<int, Order>& index) noexcept :
            mdarray_(mdarray), index_(index) { }

        constexpr iterator& operator++() noexcept {
            index_[Order - 1]++;
            int i = Order - 1;
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
        constexpr Scalar& operator*()  requires(!std::is_const_v<MdArray_>) { return fetch_at_(index_); }
        constexpr Scalar* operator->() requires(!std::is_const_v<MdArray_>) {
	    return std::addressof(fetch_at_(index_));
	}      
        // comparison
        constexpr friend bool operator==(const iterator& lhs, const iterator& rhs) { return lhs.index_ == rhs.index_; }
        constexpr friend bool operator!=(const iterator& lhs, const iterator& rhs) { return lhs.index_ != rhs.index_; }
       private:
        template <typename IndexType> constexpr decltype(auto) fetch_at_(IndexType&& index) {
            return internals::apply_index_pack<Order>(
              [&]<int... Ns_>() -> decltype(auto) { return mdarray_->operator()((Ns_, index_[Ns_])...); });
        }
        MdArray_* mdarray_;
        std::array<int, Order> index_;
    };
    // const iterators
    constexpr iterator<const MdArray<Scalar_, Extents_, LayoutPolicy_>> begin() const noexcept {
        return internals::apply_index_pack<Order>(
          [&]<int... Ns_> { return iterator<const MdArray<Scalar_, Extents_, LayoutPolicy_>> {this, (Ns_, 0)...}; });
    }
    constexpr iterator<const MdArray<Scalar_, Extents_, LayoutPolicy_>> end() const noexcept {
        return internals::apply_index_pack<Order - 1>([&]<int... Ns_> {
            return iterator<const MdArray<Scalar_, Extents_, LayoutPolicy_>> {this, extents_.extent(0), (Ns_, 0)...};
        });
    }
    // non-const iterators
    constexpr iterator<MdArray<Scalar_, Extents_, LayoutPolicy_>> begin() noexcept {
        return internals::apply_index_pack<Order>(
          [&]<int... Ns_> { return iterator<MdArray<Scalar_, Extents_, LayoutPolicy_>> {this, (Ns_, 0)...}; });
    }
    constexpr iterator<MdArray<Scalar_, Extents_, LayoutPolicy_>> end() noexcept {
        return internals::apply_index_pack<Order - 1>([&]<int... Ns_> {
            return iterator<MdArray<Scalar_, Extents_, LayoutPolicy_>> {this, extents_.extent(0), (Ns_, 0)...};
        });
    }
    // resize dynamic MdArray (only dynamic extents). allocated memory is left uninitialized
    template <typename... Exts_>
        requires(std::is_convertible_v<Exts_, int> && ...)
    constexpr void resize(Exts_... exts) {
        fdapde_static_assert(
          sizeof...(Exts_) == Extents_::DynamicOrder,
          YOU_SUPPLIED_A_WRONG_NUMBER_OF_ARGUMENTS_TO_RESIZE__NUMBER_OF_ARGUMENTS_MUST_MATCH_NUMBER_OF_DYNAMIC_EXTENTS);
        extents_.resize(static_cast<int>(exts)...);
        mapping_ = Mapping(extents_);
        data_.resize(size());   // re-allocate space
    }
    // constant access
    template <typename... Idxs>
        requires(std::is_convertible_v<Idxs, int> && ...) && (sizeof...(Idxs) == Extents_::Order)
    constexpr const Scalar& operator()(Idxs... idxs) const {
        return data_[mapping_(static_cast<int>(idxs)...)];
    }
    constexpr const Scalar& operator[](int i) const { return data_[i]; }
    // non-constant access
    template <typename... Idxs>
        requires(std::is_convertible_v<Idxs, int> && ...) && (sizeof...(Idxs) == Extents_::Order)
    constexpr Scalar& operator()(Idxs... idxs) {
        return data_[mapping_(static_cast<int>(idxs)...)];
    }
    constexpr Scalar& operator[](int i) { return data_[i]; }
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
    constexpr MdArrayBlock<const MdArray<Scalar_, Extents_, LayoutPolicy_>, extents<Exts_...>> block(
      Slicers_... slicers) const {
        // check block is within MdArray
        std::array<int, Order> static_block_extents {Exts_...};
        internals::for_each_index_and_args<Order>(
          [&]<int Ns_, typename Slicer__>(Slicer__ s) {
              fdapde_constexpr_assert(static_block_extents[Ns_] + s < 1 + mdarray.extent(Ns_));
          },
          slicers...);
        return MdArrayBlock<MdArray<Scalar_, Extents_, LayoutPolicy_>, extents<Exts_...>>(
          this, extents<Exts_...>(), slicers...);
    }
    template <int... Exts_, typename... Slicers_>   // static-sized (non-const access)
        requires(sizeof...(Exts_) == Order && sizeof...(Exts_) == sizeof...(Slicers_)) &&
                (internals::is_integer_v<Slicers_> && ...)
    constexpr MdArrayBlock<MdArray<Scalar_, Extents_, LayoutPolicy_>, extents<Exts_...>> block(Slicers_... slicers) {
        // check block is within MdArray
        std::array<int, Order> static_block_extents {Exts_...};
        internals::for_each_index_and_args<Order>(
          [&]<int Ns_, typename Slicer__>(Slicer__ s) {
              fdapde_constexpr_assert(static_block_extents[Ns_] + s < 1 + mdarray.extent(Ns_));
          },
          slicers...);
        return MdArrayBlock<MdArray<Scalar_, Extents_, LayoutPolicy_>, extents<Exts_...>>(
          this, extents<Exts_...>(), slicers...);
    }

    template <int... Slicers, typename... Slicers__>
        requires(std::is_convertible_v<Slicers__, int> && ...)
    constexpr auto slice(Slicers__... slicers) {
        return MdArraySlice<MdArray<Scalar_, Extents_, LayoutPolicy_>, Slicers...>(this, slicers...);
    }
    template <int... Slicers, typename... Slicers__>
        requires(std::is_convertible_v<Slicers__, int> && ...)
    constexpr auto slice(Slicers__... slicers) const {
        return MdArraySlice<const MdArray<Scalar_, Extents_, LayoutPolicy_>, Slicers...>(this, slicers...);
    }
   private:
    Extents extents_ {};
    Mapping mapping_ {};
    Storage data_ {};
};

}   // namespace fdapde

#endif   // __MDARRAY_H__
