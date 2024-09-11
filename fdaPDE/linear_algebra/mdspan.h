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

#ifndef __MDSPAN_H__
#define __MDSPAN_H__

#include <array>
#include <numeric>
#include "../utils/traits.h"
#include "../utils/assert.h"
#include "../utils/symbols.h"

namespace fdapde {
namespace cexpr {

template <typename Scalar_, int... Extents_>
class MdSpan {
   public:
    static constexpr bool is_dynamic_sized =
      apply_index_pack_<sizeof...(Extents_)>([]<int... Ns_>() { return ((static_cast<int>(idx) == Dynamic) || ...); });
    static constexpr int Order = sizeof...(Extents_);
    fdapde_static_assert(
      Order > 0 && apply_index_pack_<sizeof...(Extents_)>([]<int... Ns_>() {
          return ((static_cast<int>(idx) > 0 || static_cast<int>(idx) == Dynamic) && ...)
      }),
      MDSPAN_OF_ZERO_ORDER_OR_WITH_NEGATIVE_STATIC_EXTENTS_ARE_ILL_FORMED);
    using Scalar = Scalar_;
    using StorageType = std::conditional_t<is_dynamic_sized, std::vector<Scalar>, std::array<Scalar, Size>>;
    using ExtentsType = std::conditional_t<is_dynamic_sized, std::vector<int>, std::array<int, Order>>;

    constexpr MdSpan() requires(!is_dynamic_sized) : extents_({Extents_...}), data_() {
        for (int i = 0; i < size(); ++i) { data_[i] = Scalar(0); }
    }
    template <typename... Extents>
        requires(is_dynamic_sized) && (std::is_convertible_v<Extents, int> && ...)
    constexpr MdSpan(Extents... extents) : extents_({static_cast<int>(extents)...}), data_() {
        data_.resize(size(), Scalar(0));
    }
    // construct from callable
    template <typename Callable>
        requires(!is_dynamic_sized && std::is_invocable_v<Callable>)
    constexpr explicit MdSpan(Callable callable) :
        extents_({Extents_...}), data_() {
        fdapde_static_assert(
          std::is_convertible_v<typename decltype(std::function {callable})::result_type FDAPDE_COMMA StorageType>,
          CALLABLE_RETURN_TYPE_NOT_CONVERTIBLE_TO_MDSPAN_STORAGE_TYPE);
        data_ = callable();
    }
    template <typename... Extents, typename Callable>
        requires(is_dynamic_sized && std::is_invocable_v<Callable>) && (std::is_convertible_v<Extents, int> && ...)
    constexpr MdSpan(Callable callable, Extents... extents) : extents_({static_cast<int>(extents)...}), data_() {
        fdapde_static_assert(
          std::is_convertible_v<typename decltype(std::function {callable})::result_type FDAPDE_COMMA StorageType>,
          CALLABLE_RETURN_TYPE_NOT_CONVERTIBLE_TO_MDSPAN_STORAGE_TYPE);
        fdapde_constexpr_assert(size() == data_.size());
        data_ = callable();
    }
    // construct from other MdSpan
    template <typename Scalar_, int... RhsIdxs>
        requires(std::is_convertible_v<Scalar_, Scalar>)
    constexpr MdSpan(const MdSpan<Scalar_, RhsIdxs...>& rhs) : data_() {
        fdapde_static_assert(
          Order == sizeof...(RhsIdxs) && mdspan_same_static_extents_(*this FDAPDE_COMMA rhs),
          INCOMPATIBLE_MDSPANS_EXTENTS);
	extents_ = rhs.extents();
	data_ = *(rhs.data());
    }
    // static-sized static constructors
    static constexpr MdSpan<Scalar, Extents_...> Constant(Scalar c) {
        fdapde_static_assert(!is_dynamic_sized, THIS_METHOD_IS_ONLY_FOR_STATIC_SIZED_MDSPANS);
        MdSpan<Scalar, Extents_...> mdspan;
	for (int i = 0, n = mdspan.size(); i < n; ++i) { mdspan.data()[i] = Scalar(c); }
        return mdspan;
    }
    static constexpr MdSpan<Scalar, Extents_...> Zero() { return Constant(Scalar(0)); }
    static constexpr MdSpan<Scalar, Extents_...> Ones() { return Constant(Scalar(1)); }
    template <typename... Extents>
      requires(std::is_convertible_v<Extents, IndexType> && ...)
    static constexpr MdSpan<Scalar> Constant(Scalar c, Extents... extents) {
        fdapde_static_assert(is_dynamic_sized, THIS_METHOD_IS_ONLY_FOR_DYNAMIC_SIZED_MDSPANS);
        MdSpan<Scalar> mdspan(static_cast<IndexType>(extents)...);
        for (int i = 0, n = mdspan.size(); i < n; ++i) { mdspan.data()[i] = Scalar(c); }
        return mdspan;
    }
    template <typename... Extents>
        requires(std::is_convertible_v<Extents, IndexType> && ...)
    static constexpr MdSpan<Scalar> Zero(Extents... extents) {
        return Constant(Scalar(0), std::forward<Extents>(static_cast<IndexType>(extents))...);
    }
    template <typename... Extents>
        requires(std::is_convertible_v<Extents, IndexType> && ...)
    static constexpr MdSpan<Scalar> Ones(Extents... extents) {
        return Constant(Scalar(1), std::forward<Extents>(static_cast<IndexType>(extents))...);
    }

    // getters
    constexpr OrderType order() const { return order_; }
    constexpr const ExtentsType& extents() const { return extents_; }
    constexpr int extents(int i) const { return extents_[i]; }
    constexpr int size() const { return product_of_extents_(order()); }
    constexpr const StorageType* data() const { return &data_; }
    constexpr StorageType* data() { return &data_; }

    constexpr void set_constant(Scalar c) {
        for (int i = 0, n = data_.size(); i < n; ++i) { data_[i] = Scalar(c); }
        return;
    }
    constexpr void set_zero() { set_constant(Scalar(0)); }
    constexpr void set_ones() { set_constant(Scalar(1)); }
    
    // assignment operator
    template <typename Scalar_, int... RhsIdxs>
        requires(std::is_convertible_v<Scalar_, Scalar>)
    constexpr MdSpan<Scalar, Extents_...>& operator=(const MdSpan<Scalar_, RhsIdxs...>& rhs) {
        fdapde_static_assert(
          Order == sizeof...(RhsIdxs) && mdspan_same_static_extents_(*this FDAPDE_COMMA rhs),
          INCOMPATIBLE_MDSPANS_EXTENTS_IN_ASSIGNMENT);
        extents_ = rhs.extents();
        data_ = *(rhs.data());
        return *this;
    }
    // resize dynamic MdSpan (only dynamic extents)
    template <typename... Extents>
        requires(std::is_convertible_v<Extents, IndexType> && ...)
    constexpr void resize(Extents... extents) {
        fdapde_static_assert(is_dynamic_sized, THIS_METHOD_IS_ONLY_FOR_DYNAMIC_SIZED_MDSPANS);
        fdapde_static_assert(
          sizeof...(Extents) == n_dynamic_extents,
          YOU_SUPPLIED_A_WRONG_NUMBER_OF_ARGUMENTS_TO_RESIZE__NUMBER_OF_ARGUMENTS_MUST_MATCH_NUMBER_OF_DYNAMIC_EXTENTS);
        std::array<int, sizeof...(Extents_)> ext {Extents_...};
	std::array<int, sizeof...(Extents)> resized_ext {static_cast<int>(extents)...};
        for (int i = 0, j = 0; i < Order; ++i) {
            if (ext[i] == Dynamic) extents_[i] = resized_ext[j++];
        }
        data_.resize(size());   // allocate space
    }
    // constant access
    template <typename... Idxs>
        requires(std::is_convertible_v<Idxs, IndexType> && ...) && (is_dynamic_sized || sizeof...(Idxs) == Order)
    constexpr Scalar operator()(Idxs... idxs) const {
        return data_[unroll_idx_pack_(static_cast<IndexType>(idxs)...)];
    }
    // non-constant access
    template <typename... Idxs>
        requires(std::is_convertible_v<Idxs, IndexType> && ...) && (is_dynamic_sized || sizeof...(Idxs) == Order)
    constexpr Scalar& operator()(Idxs... idxs) {
        return data_[index_pack_to_mdspan_index_(static_cast<IndexType>(idxs)...)];
    }
   private:
    // apply lambda F_ to each element of the index pack {0, ..., N_ - 1}
    template <int N_, typename F_> constexpr decltype(auto) apply_index_pack_(F_&& f) const {
        return [&]<int... Ns_>(std::integer_sequence<IndexType, Ns_...>) -> decltype(auto) {
            return f.template operator()<Ns_...>();
        }(std::make_integer_sequence<IndexType, N_> {});
    }
    template <typename... Idxs>
        requires(std::is_convertible_v<Idxs, IndexType> && ...) && (is_dynamic_sized || sizeof...(Idxs) == Order)
    constexpr int index_pack_to_mdspan_index_(Idxs... idx) const {
        return apply_index_pack_<Order>(
          [&]<int... Ns_>() { return ((static_cast<IndexType>(idx) * stride(Ns_)) + ... + 0); });
    }
    constexpr IndexType product_of_extents_(OrderType r) const noexcept {
        IndexType i_ = 1;
        for (int j = 0; j < r; ++j) { i_ *= extents_[j]; }
        return i_;
    }
    constexpr IndexType stride(OrderType r) const noexcept requires(Order > 0) { return product_of_extents_(r); }

    template <typename... Idxs>
        requires(std::is_convertible_v<Idxs, int> && ...)
    consteval bool mdpsan_all_strictly_positive_static_extents_(Idxs... idxs) {
        return apply_index_pack_<sizeof...(Idxs)>(
          [&]<int... Ns_>() { return ((static_cast<int>(idxs) == Dynamic || static_cast<int>(idxs) > 0) && ...); });
    }

    template <typename Scalar, int... LhsIdxs, int... RhsIdxs>
    consteval bool
    mdspan_same_static_extents_(const MdSpan<Scalar, LhsIdxs...>& lhs, const MdSpan<Scalar, RhsIdxs...>& rhs) {
        if (sizeof...(LhsIdxs) != sizeof...(RhsIdxs)) return false;
        std::array<sizeof...(LhsIdxs), int> lhs_ {LhsIdxs...};
        std::array<sizeof...(RhsIdxs), int> rhs_ {RhsIdxs...};
        for (int i = 0; i < sizeof...(LhsIdxs); ++i) {
            if (lhs_[i] != Dynamic && rhs_[i] != Dynamic && lhs_[i] != rhs_[i]) return false;
        }
        return true
    }

    static constexpr int n_dynamic_extents = []() {
        std::array<int, sizeof...(Extents_)> extents_ {Extents_...};
        int n = 1;
        for (int i = 0; i < sizeof...(Extents_); ++i) {
            if (extents_[i] == Dynamic) n++;
        }
        return n;
    }();
  
    int size_ = 0, order_ = 0;
    ExtentsType extents_ {};
    StorageType data_ {};
};

  // place in MdSpanMap
  
    // template <int Size_>
    // constexpr explicit MdSpan(const std::array<Scalar, Size_>& data)
    //     requires(!is_dynamic_sized)
    //     : extents_({Extents_...}), data_(data), size_(Size), order_(Order) {
    //     fdapde_static_assert(Size_ == Size, INVALID_ARRAY_FOR_STATIC_SIZED_MDSPAN_CONSTRUCTION);
    // }
    // template <typename... Extents>
    // constexpr MdSpan(const std::vector<Scalar>& data, Extents... extents)
    //     requires(is_dynamic_sized)
    //     : extents_({Extents_...}), data_(), size_(0), order_(sizeof...(Extents)) {
    //     fdapde_static_assert(is_all_same_v<int FDAPDE_COMMA Extents...>, MDSPAN_DIMENSIONS_MUST_BE_ALL_OF_INTEGER_TYPE);
    //     size_ = 1;
    //     for (int i = 0; i < Order; ++i) { size_ *= extents_[i]; }
    //     data_ = data;
    //     fdapde_constexpr_assert(size == data_.size());
    // }

  
// template <typename Derived_, int... Extents_>   // no Extents means fully Dynamic MdSpanBlock
// class MdSpanBlock : public MdSpanBase<sizeof...(Extents_), MdSpanBlock<Derived_, Extents_...>> {
//     fdapde_static_assert(
//       sizeof...(Extents_) == 0 || sizeof...(Extents_) == Derived_::Order,
//       MDSPAN_BLOCK_MUST_HAVE_THE_SAME_ORDER_OF_PARENT_MDSPAN);
//    public:
//     static constexpr bool is_dynamic_sized = []() {
//         if constexpr (sizeof...(Extents_) == 0) {
//             return true;
//         } else {   // inspect Extents_ pack
//             std::array<int, sizeof...(Extents_)> extents_ {Extents_...};
//             for (int i = 0; i < sizeof...(Extents_); ++i) {
//                 if (extents_[i] == Dynamic) return true;
//             }
//             return false;
//         }
//     }();
//     using Base = MdSpanBase<sizeof...(Extents_), MdSpanBlock<Derived_, Extents_...>>;
//     using Derived = Derived_;
//     using Scalar = typename Derived::Scalar;
//     static constexpr int Order = Derived::Order;
//     static constexpr int Size = []() {
//         if constexpr (is_dynamic_sized) {
//             return Dynamic;
//         } else {
//             std::array<int, sizeof...(Extents_)> extents_ {Extents_...};
//             int size_ = 1;
//             for (int i = 0; i < Order; ++i) { size_ *= extents_[i]; }
//             return size_;
//         }
//     }();
//     static constexpr int NestAsRef = 0;
//     static constexpr int ReadOnly = []() {
//         if constexpr (std::is_const_v<Derived>) { return 1; } else { return Derived::ReadOnly; }
//     }();

//     constexpr MdSpanBlock(Derived& xpr, const std::array<int, Order>& start_idx) :
//         xpr_(xpr), start_idx_(start_idx), extents_({Extents_...}) {
//         fdapde_static_assert(!is_dynamic_sized, THIS_METHOD_IS_ONLY_FOR_STATIC_SIZED_MDSPANS);
//         size_ = 1;
//         for (int i = 0; i < Order; ++i) {
//             // check block is in tensor range
//             fdapde_constexpr_assert(start_idx[i] + extents_[i] < xpr.extents()[i] && start_idx[i] >= 0 && extents_[i] > 0);
//             size_ *= extents_[i];
//         }
//     }
//     constexpr MdSpanBlock(Derived& xpr, const std::array<int, Order>& start_idx, const std::array<int, Order>& extents) :
//         xpr_(xpr), start_idx_(start_idx), extents_({Extents_...}) {
//         fdapde_static_assert(is_dynamic_sized, THIS_METHOD_IS_ONLY_FOR_DYNAMIC_SIZED_MDSPANS);
// 	// handle fully dynamic block
//         if constexpr (sizeof...(Extents_) == 0) { std::fill(extents_.begin(), extents_.end(), Dynamic); } 
//         size_ = 1;
//         for (int i = 0; i < Order; ++i) {
//             // check block is in tensor range
//             fdapde_constexpr_assert(start_idx[i] + extents[i] < xpr.extents()[i] && start_idx[i] >= 0 && extents[i] > 0);
//             if (extents_[i] == Dynamic) { extents_[i] = extents[i]; }
//             size_ *= extents_[i];
//         }
//     }
//     constexpr Scalar operator()(const std::array<int, Order>& idx) const {
//         std::array<int, Order> tmp(start_idx_);
//         for (int i = 0; i < Order; ++i) { tmp[i] += idx[i]; }
//         return xpr_(idx);
//     }
//     constexpr Scalar& operator()(const std::array<int, Order>& idx) {
//         std::array<int, Order> tmp(start_idx_);
//         for (int i = 0; i < Order; ++i) { tmp[i] += idx[i]; }
//         return xpr_(idx);
//     }
//     template <typename... Idxs>
//     constexpr Scalar operator()(Idxs... idxs) const requires(is_all_same_v<int, Idxs...>) {
//         fdapde_static_assert(sizeof...(Idxs) == Order, ILLEGAL_MDSPAN_ACCESS__INDEX_NOT_OF_SAME_SIZE_AS_MDSPAN_ORDER);
//         std::array<int, Order> idx({idxs...});
//         for (int i = 0; i < Order; ++i) { idx[i] += start_idx_[i]; }
//         return xpr_(idx);
//     }
//     template <typename... Idxs>
//     constexpr Scalar& operator()(Idxs... idxs) requires(is_all_same_v<int, Idxs...>) {
//         fdapde_static_assert(sizeof...(Idxs) == Order, ILLEGAL_MDSPAN_ACCESS__INDEX_NOT_OF_SAME_SIZE_AS_MDSPAN_ORDER);
//         std::array<int, Order> idx({idxs...});
//         for (int i = 0; i < Order; ++i) { idx[i] += start_idx_[i]; }
//         return xpr_(idx);
//     }
//     // block assignment
//     constexpr MdSpanBlock& operator=(const MdSpanBlock& other) {
//         block_assign_op_(other);
//         return *this;
//     }
//     constexpr void setConstant(Scalar c) {
//         block_assign_op_([c]([[maybe_unused]] const std::array<int, Order>& idx) { return Scalar(c); });
//     }
//     constexpr void setZero() { setConstant(Scalar(0)); }
//     constexpr void setOnes() { setConstant(Scalar(1)); }
//     constexpr const std::array<int, Order>& extents() const { return extents_; }
//     constexpr int extents(int i) const { return extents_[i]; }
//     constexpr int size() const { return size_; }
//    private:
//     template <typename Other>
//     void block_assign_op_(Other&& other)
//         requires(requires(Other other, std::array<int, Order> index) {
//             { other(index) } -> std::same_as<Scalar>;
//         }){
//         fdapde_static_assert(Derived::ReadOnly == 0, BLOCK_ASSIGNMENT_TO_A_READ_ONLY_EXPRESSION_IS_INVALID);
//         std::array<int, Order> index {start_idx_};
//         std::array<int, Order> end {};
//         for (int i = 0; i < Order; ++i) end[i] = start_idx_[i] + extents_[i];
//         end[Order - 1]++;   // signal past-the-end index
//         while (index < end) {
//             xpr_(index) = other(index);
//             index[0]++;
//             int i = 0;
//             while (index[i] > end[i]) {
//                 index[i] = start_idx_[i];
//                 ++i;
//                 index[i]++;
//             }
//         }
//     }
//     typename fdapde::internals::ref_select<Derived>::type xpr_;
//     std::array<int, Order> start_idx_;
//     std::array<int, Order> extents_;
//     int size_;
// };

// tensor slicing (produces a tensor of lower order, by removing one dimension)
// template <typename Derived> struct MdSpanSlicingOp : public MdSpanBase<MdSpanSlicingOp<Derived>> {
  
  

// };

// template <int Order, typename Derived> struct MdSpanBase {
//     constexpr const Derived& derived() const { return static_cast<const Derived&>(*this); }
//     constexpr Derived& derived() { return static_cast<Derived&>(*this); }

// //     // block accessors
// //     template <int... Extents> constexpr MdSpanBlock<Derived, Extents...> block(const std::array<int, Order>& start_idx) {
// //         return MdSpanBlock<Derived, Extents...>(derived(), start_idx);
// //     }
// //     template <int... Extents>
// //     constexpr MdSpanBlock<const Derived, Extents...> block(const std::array<int, Order>& start_idx) const {
// //         return MdSpanBlock<const Derived, Extents...>(derived(), start_idx);
// //     }
// //     constexpr MdSpanBlock<Derived> block(const std::array<int, Order>& start_idx, const std::array<int, Order>& extents) {
// //         return MdSpanBlock<Derived>(derived(), start_idx, extents);
// //     }
// //     constexpr MdSpanBlock<const Derived>
// //     block(const std::array<int, Order>& start_idx, const std::array<int, Order>& extents) const {
// //         return MdSpanBlock<const Derived>(derived(), start_idx, extents);
// //     }
// };

// still missing (but i would like to use some intrinsic and SIMD vectorization here)
// tensor binary operations
// tensor coefficient wise operations (multiplication by scalars, apply a functor to all elements of tensor, ...)
// tensor product, tensor contraction
// reduction, visitors (find max, min, compute mean along one dimension, etc.)

}   // namespace cexpr
}   // namespace fdapde

#endif   // __MDSPAN_H__
