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
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.

#ifndef __FDAPDE_LINALG_NATIVE_MDARRAY_H__
#define __FDAPDE_LINALG_NATIVE_MDARRAY_H__

#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <iterator>
#include <limits>
#include <memory>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "header_check.h"

namespace fdapde::linalg {

inline constexpr int full_extent = -2;

namespace internals {

template <typename T>
concept md_index = std::integral<std::remove_cvref_t<T>> && (!std::same_as<std::remove_cvref_t<T>, bool>);

template <md_index T> constexpr bool checked_index_cast(T value, int& result) {
    if (!std::in_range<int>(value)) return false;
    result = static_cast<int>(value);
    return true;
}

template <std::size_t Size> constexpr bool checked_shape_size(const std::array<int, Size>& extents, int& size) {
    bool empty = false;
    for (int extent : extents) {
        if (extent < 0) return false;
        empty = empty || extent == 0;
    }
    if (empty) {
        size = 0;
        return true;
    }

    std::size_t product = 1;
    for (int extent : extents) {
        if (product > static_cast<std::size_t>(std::numeric_limits<int>::max()) / static_cast<std::size_t>(extent)) {
            return false;
        }
        product *= static_cast<std::size_t>(extent);
    }
    size = static_cast<int>(product);
    return true;
}

template <int... Extents> consteval int static_md_size() {
    if constexpr (((Extents == Dynamic) || ...)) {
        return Dynamic;
    } else {
        constexpr std::array<int, sizeof...(Extents)> extents {Extents...};
        std::size_t product = 1;
        for (int extent : extents) {
            if (
              extent <= 0 ||
              product > static_cast<std::size_t>(std::numeric_limits<int>::max()) / static_cast<std::size_t>(extent)) {
                return Dynamic;
            }
            product *= static_cast<std::size_t>(extent);
        }
        return static_cast<int>(product);
    }
}

template <int... Values> consteval bool unique_axes() {
    constexpr std::array<int, sizeof...(Values)> values {Values...};
    for (std::size_t i = 0; i < values.size(); ++i) {
        for (std::size_t j = i + 1; j < values.size(); ++j) {
            if (values[i] == values[j]) return false;
        }
    }
    return true;
}

template <typename T> inline constexpr bool is_pair_v = fdapde::internals::is_pair_v<std::remove_cvref_t<T>>;

template <typename T>
concept md_index_pack = requires(const T& value) {
    { value.size() } -> std::convertible_to<std::size_t>;
    requires md_index<decltype(value[0])>;
};

}   // namespace internals

template <int... StaticExtents> class MdExtents {
    static_assert(sizeof...(StaticExtents) > 0, "MdExtents requires at least one axis");
    static_assert(
      ((StaticExtents > 0 || StaticExtents == Dynamic) && ...), "MdExtents values must be positive or Dynamic");
   public:
    using index_t = int;
    using order_t = std::size_t;
    using size_t = std::size_t;

    static constexpr order_t Order = sizeof...(StaticExtents);
    static constexpr order_t DynamicOrder = ((StaticExtents == Dynamic) + ... + 0);
    static constexpr order_t StaticOrder = Order - DynamicOrder;
    static constexpr int StaticSize = internals::static_md_size<StaticExtents...>();
    static constexpr std::array<int, Order> static_extents {StaticExtents...};

    static_assert(DynamicOrder != 0 || StaticSize != Dynamic, "Static MdExtents exceeds the supported int index range");

    constexpr MdExtents() : extents_ {((StaticExtents == Dynamic) ? 0 : StaticExtents)...}, valid_(true) { }

    template <internals::md_index... Dims>
        requires(DynamicOrder > 0 && sizeof...(Dims) == DynamicOrder)
    constexpr explicit MdExtents(Dims... dims) : MdExtents() {
        if (!assign_dynamic_(std::array {dims...})) {
            valid_ = false;
            fdapde_assert(false);
        }
    }

    template <internals::md_index... Dims>
        requires(DynamicOrder > 0 && sizeof...(Dims) == Order && sizeof...(Dims) != DynamicOrder)
    constexpr explicit MdExtents(Dims... dims) : MdExtents() {
        if (!assign_full_(std::array {dims...})) {
            valid_ = false;
            fdapde_assert(false);
        }
    }

    template <internals::md_index T> constexpr explicit MdExtents(const std::array<T, Order>& dims) : MdExtents() {
        if (!assign_full_(dims)) {
            valid_ = false;
            fdapde_assert(false);
        }
    }

    constexpr order_t order() const noexcept { return Order; }
    constexpr order_t order_dynamic() const noexcept { return DynamicOrder; }
    constexpr int extent(order_t axis) const noexcept {
        const bool valid = axis < Order;
        fdapde_assert(valid);
        return valid_ && valid ? extents_[axis] : 0;
    }
    constexpr int size() const noexcept {
        if (!valid_) return 0;
        int result = 0;
        const bool valid = internals::checked_shape_size(extents_, result);
        fdapde_assert(valid);
        return valid ? result : 0;
    }
    constexpr bool valid() const noexcept { return valid_; }

    template <internals::md_index... Dims>
        requires(DynamicOrder > 0 && sizeof...(Dims) == DynamicOrder)
    constexpr bool resize(Dims... dims) {
        MdExtents candidate(*this);
        if (!candidate.assign_dynamic_(std::array {dims...})) {
            fdapde_assert(false);
            return false;
        }
        *this = candidate;
        valid_ = true;
        return true;
    }

    template <internals::md_index... Dims>
        requires(DynamicOrder > 0 && sizeof...(Dims) == Order && sizeof...(Dims) != DynamicOrder)
    constexpr bool resize(Dims... dims) {
        MdExtents candidate(*this);
        if (!candidate.assign_full_(std::array {dims...})) {
            fdapde_assert(false);
            return false;
        }
        *this = candidate;
        valid_ = true;
        return true;
    }

    template <internals::md_index T> constexpr bool resize(const std::array<T, Order>& dims) {
        MdExtents candidate(*this);
        if (!candidate.assign_full_(dims)) {
            fdapde_assert(false);
            return false;
        }
        *this = candidate;
        valid_ = true;
        return true;
    }

    friend constexpr bool operator==(const MdExtents&, const MdExtents&) = default;
   private:
    template <typename Dims, std::size_t... Is>
    constexpr bool assign_dynamic_impl_(const Dims& dims, std::index_sequence<Is...>) {
        std::array<int, DynamicOrder> values {};
        bool valid = (internals::checked_index_cast(dims[Is], values[Is]) && ...);
        if (!valid) return false;
        std::size_t dynamic_index = 0;
        for (std::size_t axis = 0; axis < Order; ++axis) {
            if (static_extents[axis] == Dynamic) extents_[axis] = values[dynamic_index++];
        }
        int ignored = 0;
        return internals::checked_shape_size(extents_, ignored);
    }

    template <typename Dims> constexpr bool assign_dynamic_(const Dims& dims) {
        const auto original = extents_;
        if (!assign_dynamic_impl_(dims, std::make_index_sequence<DynamicOrder> {})) {
            extents_ = original;
            return false;
        }
        return true;
    }

    template <typename Dims, std::size_t... Is>
    constexpr bool assign_full_impl_(const Dims& dims, std::index_sequence<Is...>) {
        std::array<int, Order> values {};
        bool valid = (internals::checked_index_cast(dims[Is], values[Is]) && ...);
        if (!valid) return false;
        for (std::size_t axis = 0; axis < Order; ++axis) {
            if (static_extents[axis] != Dynamic && values[axis] != static_extents[axis]) return false;
        }
        int ignored = 0;
        if (!internals::checked_shape_size(values, ignored)) return false;
        extents_ = values;
        return true;
    }

    template <typename Dims> constexpr bool assign_full_(const Dims& dims) {
        return assign_full_impl_(dims, std::make_index_sequence<Order> {});
    }

    std::array<int, Order> extents_;
    bool valid_;
};

namespace internals {

template <int N, std::size_t... Is>
auto make_full_dynamic_extents(std::index_sequence<Is...>) -> MdExtents<((void)Is, Dynamic)...>;

template <typename Extents, int StorageOrder> class MdMapping {
    static_assert(StorageOrder == RowMajor || StorageOrder == ColMajor, "Unsupported MdArray storage order");
   public:
    static constexpr int Order = static_cast<int>(Extents::Order);
    using extents_t = Extents;

    constexpr MdMapping() : extents_(), strides_() { initialize_(); }
    constexpr explicit MdMapping(const Extents& extents) : extents_(extents), strides_() { initialize_(); }

    constexpr int stride(int axis) const noexcept {
        const bool valid = axis >= 0 && axis < Order;
        fdapde_assert(valid);
        return valid ? strides_[axis] : 0;
    }
    constexpr const Extents& extents() const noexcept { return extents_; }

    template <internals::md_index... Indices>
        requires(sizeof...(Indices) == Extents::Order)
    constexpr int operator()(Indices... indices) const {
        std::array<int, Order> values {};
        std::size_t i = 0;
        bool valid = (internals::checked_index_cast(indices, values[i++]) && ...);
        valid = valid && indices_valid_(values);
        fdapde_assert(valid);
        return valid ? map_unchecked_(values) : 0;
    }

    constexpr int operator()(const std::array<int, Order>& indices) const {
        const bool valid = indices_valid_(indices);
        fdapde_assert(valid);
        return valid ? map_unchecked_(indices) : 0;
    }

    friend constexpr bool operator==(const MdMapping&, const MdMapping&) = default;
   private:
    constexpr void initialize_() {
        constexpr std::size_t MaxIndex = static_cast<std::size_t>(std::numeric_limits<int>::max());
        std::size_t stride = 1;
        auto consume = [&](int axis) {
            strides_[axis] = stride <= MaxIndex ? static_cast<int>(stride) : 0;
            const std::size_t extent = static_cast<std::size_t>(extents_.extent(static_cast<std::size_t>(axis)));
            if (extent == 0) {
                stride = 0;
            } else if (stride > MaxIndex / extent) {
                stride = MaxIndex + 1;
            } else {
                stride *= extent;
            }
        };
        if constexpr (StorageOrder == RowMajor) {
            for (int axis = Order - 1; axis >= 0; --axis) consume(axis);
        } else {
            for (int axis = 0; axis < Order; ++axis) consume(axis);
        }
    }

    constexpr bool indices_valid_(const std::array<int, Order>& indices) const {
        for (int axis = 0; axis < Order; ++axis) {
            if (indices[axis] < 0 || indices[axis] >= extents_.extent(static_cast<std::size_t>(axis))) return false;
        }
        return true;
    }

    constexpr int map_unchecked_(const std::array<int, Order>& indices) const {
        int result = 0;
        for (int axis = 0; axis < Order; ++axis) result += indices[axis] * strides_[axis];
        return result;
    }

    Extents extents_;
    std::array<int, Order> strides_;
};

template <typename T>
concept md_readable = requires(const T& value) {
    { T::Order } -> std::convertible_to<int>;
    { value.size() } -> std::convertible_to<int>;
    { value.extent(0) } -> std::convertible_to<int>;
    { value.valid() } -> std::convertible_to<bool>;
    value.begin();
    value.end();
};

}   // namespace internals

template <int N>
using full_dynamic_extent_t = decltype(internals::make_full_dynamic_extents<N>(std::make_index_sequence<N> {}));

template <typename Parent, int Order> class MdView;

namespace internals {

template <typename Derived, typename Scalar_, int Order_> class MdAccessBase {
   public:
    using Scalar = Scalar_;
    static constexpr int Order = Order_;

    constexpr int size() const noexcept { return derived_().md_size(); }
    constexpr int extent(int axis) const noexcept { return derived_().md_extent(axis); }
    constexpr bool valid() const noexcept { return derived_().md_valid(); }

    template <internals::md_index... Indices>
        requires(sizeof...(Indices) == Order)
    constexpr decltype(auto) operator()(Indices... indices) & {
        return coefficient_(make_indices_(indices...));
    }
    template <internals::md_index... Indices>
        requires(sizeof...(Indices) == Order)
    constexpr decltype(auto) operator()(Indices... indices) const& {
        return coefficient_(make_indices_(indices...));
    }
    template <internals::md_index... Indices> constexpr void operator()(Indices...) && = delete;
    template <internals::md_index... Indices> constexpr void operator()(Indices...) const&& = delete;

    template <internals::md_index_pack IndexPack> constexpr decltype(auto) operator()(const IndexPack& indices) & {
        return coefficient_(indices_from_pack_(indices));
    }
    template <internals::md_index_pack IndexPack> constexpr decltype(auto) operator()(const IndexPack& indices) const& {
        return coefficient_(indices_from_pack_(indices));
    }

    template <typename Owner> class iterator {
       public:
        using difference_type = std::ptrdiff_t;
        using value_type = std::remove_cv_t<Scalar>;
        using reference = decltype(std::declval<Owner&>().md_coefficient_at_position(0));
        using iterator_category = std::forward_iterator_tag;

        constexpr iterator() noexcept : owner_(nullptr), position_(0) { }
        constexpr iterator(Owner* owner, int position) noexcept : owner_(owner), position_(position) { }
        constexpr decltype(auto) operator*() const { return owner_->md_coefficient_at_position(position_); }
        constexpr iterator& operator++() noexcept {
            ++position_;
            return *this;
        }
        constexpr iterator operator++(int) noexcept {
            iterator copy(*this);
            ++(*this);
            return copy;
        }
        constexpr int mapped_index() const { return owner_->md_mapped_index_at_position(position_); }
        friend constexpr bool operator==(const iterator&, const iterator&) = default;
       private:
        Owner* owner_;
        int position_;
    };

    constexpr auto begin() & noexcept { return iterator<Derived>(std::addressof(derived_()), 0); }
    constexpr auto end() & noexcept { return iterator<Derived>(std::addressof(derived_()), size()); }
    constexpr auto begin() const& noexcept { return iterator<const Derived>(std::addressof(derived_()), 0); }
    constexpr auto end() const& noexcept { return iterator<const Derived>(std::addressof(derived_()), size()); }
    constexpr void begin() && = delete;
    constexpr void end() && = delete;
    constexpr void begin() const&& = delete;
    constexpr void end() const&& = delete;

    constexpr decltype(auto) md_coefficient_at_position(int position) & {
        const bool in_range = position >= 0 && position < size();
        fdapde_assert(in_range);
        return coefficient_(indices_at_position_(in_range ? position : 0));
    }
    constexpr decltype(auto) md_coefficient_at_position(int position) const& {
        const bool in_range = position >= 0 && position < size();
        fdapde_assert(in_range);
        return coefficient_(indices_at_position_(in_range ? position : 0));
    }
    constexpr int md_mapped_index_at_position(int position) const {
        const bool in_range = position >= 0 && position < size();
        fdapde_assert(in_range);
        if (!in_range) return 0;
        const auto indices = indices_at_position_(position);
        int result = derived_().md_offset();
        for (int axis = 0; axis < Order; ++axis) result += indices[axis] * derived_().md_stride(axis);
        return result;
    }
   protected:
    constexpr Derived& derived_() { return static_cast<Derived&>(*this); }
    constexpr const Derived& derived_() const { return static_cast<const Derived&>(*this); }
   private:
    template <internals::md_index... Indices> constexpr std::array<int, Order> make_indices_(Indices... indices) const {
        std::array<int, Order> result {};
        std::size_t i = 0;
        if (!(internals::checked_index_cast(indices, result[i++]) && ...)) fdapde_assert(false);
        return result;
    }

    template <typename IndexPack> constexpr std::array<int, Order> indices_from_pack_(const IndexPack& indices) const {
        std::array<int, Order> result {};
        bool valid = std::cmp_equal(indices.size(), Order);
        if (valid) {
            for (int axis = 0; axis < Order; ++axis) {
                valid = internals::checked_index_cast(indices[axis], result[axis]) && valid;
            }
        }
        fdapde_assert(valid);
        return result;
    }

    constexpr std::array<int, Order> indices_at_position_(int position) const {
        std::array<int, Order> result {};
        for (int axis = Order - 1; axis >= 0; --axis) {
            const int axis_extent = extent(axis);
            if (axis_extent == 0) return {};
            result[axis] = position % axis_extent;
            position /= axis_extent;
        }
        return result;
    }

    constexpr bool indices_valid_(const std::array<int, Order>& indices) const {
        if (!valid()) return false;
        for (int axis = 0; axis < Order; ++axis) {
            if (indices[axis] < 0 || indices[axis] >= extent(axis)) return false;
        }
        return true;
    }

    constexpr decltype(auto) coefficient_(const std::array<int, Order>& indices) & {
        const bool valid_indices = indices_valid_(indices);
        fdapde_assert(valid_indices);
        int mapped = derived_().md_offset();
        for (int axis = 0; axis < Order; ++axis) mapped += indices[axis] * derived_().md_stride(axis);
        return derived_().md_linear_at(valid_indices ? mapped : 0);
    }
    constexpr decltype(auto) coefficient_(const std::array<int, Order>& indices) const& {
        const bool valid_indices = indices_valid_(indices);
        fdapde_assert(valid_indices);
        int mapped = derived_().md_offset();
        for (int axis = 0; axis < Order; ++axis) mapped += indices[axis] * derived_().md_stride(axis);
        return derived_().md_linear_at(valid_indices ? mapped : 0);
    }
};

template <typename Derived, typename Scalar, int Order>
class MdViewableBase : public MdAccessBase<Derived, Scalar, Order> {
    using Base = MdAccessBase<Derived, Scalar, Order>;
   public:
    using Base::operator();

    template <typename... Slicers>
        requires(
          sizeof...(Slicers) == Order && ((internals::md_index<Slicers> || internals::is_pair_v<Slicers>) && ...))
    constexpr auto block(Slicers... slicers) & {
        return make_block_(this->derived_(), slicers...);
    }
    template <typename... Slicers>
        requires(
          sizeof...(Slicers) == Order && ((internals::md_index<Slicers> || internals::is_pair_v<Slicers>) && ...))
    constexpr auto block(Slicers... slicers) const& {
        return make_block_(std::as_const(this->derived_()), slicers...);
    }
    template <typename... Slicers> constexpr void block(Slicers...) && = delete;
    template <typename... Slicers> constexpr void block(Slicers...) const&& = delete;

    template <int... Axes, internals::md_index... Indices>
        requires(sizeof...(Axes) == sizeof...(Indices) && sizeof...(Axes) > 0 && sizeof...(Axes) < Order)
    constexpr auto slice(Indices... indices) & {
        return make_slice_<Axes...>(this->derived_(), indices...);
    }
    template <int... Axes, internals::md_index... Indices>
        requires(sizeof...(Axes) == sizeof...(Indices) && sizeof...(Axes) > 0 && sizeof...(Axes) < Order)
    constexpr auto slice(Indices... indices) const& {
        return make_slice_<Axes...>(std::as_const(this->derived_()), indices...);
    }
    template <int... Axes, typename... Indices> constexpr void slice(Indices...) && = delete;
    template <int... Axes, typename... Indices> constexpr void slice(Indices...) const&& = delete;

    constexpr auto row(int index) &
        requires(Order == 2)
    {
        return block(index, full_extent);
    }
    constexpr auto row(int index) const&
        requires(Order == 2)
    {
        return block(index, full_extent);
    }
    constexpr auto col(int index) &
        requires(Order == 2)
    {
        return block(full_extent, index);
    }
    constexpr auto col(int index) const&
        requires(Order == 2)
    {
        return block(full_extent, index);
    }
    constexpr void row(int) && = delete;
    constexpr void row(int) const&& = delete;
    constexpr void col(int) && = delete;
    constexpr void col(int) const&& = delete;
   private:
    template <typename Parent, typename Tuple, std::size_t... Is>
    static constexpr auto make_block_impl_(Parent& parent, const Tuple& slicers, std::index_sequence<Is...>) {
        std::array<int, Order> extents {};
        std::array<int, Order> strides {};
        int offset = parent.md_offset();
        bool valid = parent.valid();

        auto consume = [&]<std::size_t Axis>(const auto& slicer) {
            strides[Axis] = parent.md_stride(static_cast<int>(Axis));
            using Slicer = std::remove_cvref_t<decltype(slicer)>;
            if constexpr (internals::md_index<Slicer>) {
                int index = 0;
                valid = internals::checked_index_cast(slicer, index) && valid;
                if (index == full_extent) {
                    extents[Axis] = parent.extent(static_cast<int>(Axis));
                } else {
                    valid = index >= 0 && index < parent.extent(static_cast<int>(Axis)) && valid;
                    extents[Axis] = 1;
                    if (valid && parent.size() > 0) offset += index * strides[Axis];
                }
            } else {
                int first = 0;
                int last = 0;
                valid = internals::checked_index_cast(slicer.first, first) &&
                        internals::checked_index_cast(slicer.second, last) && valid;
                valid = first >= 0 && last >= first && last < parent.extent(static_cast<int>(Axis)) && valid;
                if (valid) {
                    extents[Axis] = last - first + 1;
                    if (parent.size() > 0) offset += first * strides[Axis];
                }
            }
        };
        (consume.template operator()<Is>(std::get<Is>(slicers)), ...);
        if (!valid) {
            fdapde_assert(valid);
            extents.fill(0);
            offset = 0;
        }
        return MdView<Parent, Order>(std::addressof(parent), extents, strides, offset, valid);
    }

    template <typename Parent, typename... Slicers>
    static constexpr auto make_block_(Parent& parent, Slicers... slicers) {
        return make_block_impl_(parent, std::tuple<Slicers...>(slicers...), std::index_sequence_for<Slicers...> {});
    }

    template <int... Axes, typename Parent, typename... Indices>
    static constexpr auto make_slice_(Parent& parent, Indices... indices) {
        static_assert(((Axes >= 0 && Axes < Order) && ...), "Slice axis is out of range");
        static_assert(internals::unique_axes<Axes...>(), "Slice axes must be unique");
        constexpr int ViewOrder = Order - sizeof...(Axes);
        constexpr std::array<int, sizeof...(Axes)> axes {Axes...};
        std::array<int, sizeof...(Axes)> fixed {};
        std::size_t fixed_index = 0;
        bool valid = parent.valid() && (internals::checked_index_cast(indices, fixed[fixed_index++]) && ...);
        int offset = parent.md_offset();
        std::array<bool, Order> removed {};
        for (std::size_t i = 0; i < axes.size(); ++i) {
            const int axis = axes[i];
            removed[axis] = true;
            valid = fixed[i] >= 0 && fixed[i] < parent.extent(axis) && valid;
            if (valid && parent.size() > 0) offset += fixed[i] * parent.md_stride(axis);
        }
        std::array<int, ViewOrder> extents {};
        std::array<int, ViewOrder> strides {};
        int target = 0;
        for (int axis = 0; axis < Order; ++axis) {
            if (!removed[axis]) {
                extents[target] = parent.extent(axis);
                strides[target] = parent.md_stride(axis);
                ++target;
            }
        }
        if (!valid) {
            fdapde_assert(valid);
            extents.fill(0);
            offset = 0;
        }
        return MdView<Parent, ViewOrder>(std::addressof(parent), extents, strides, offset, valid);
    }
};

}   // namespace internals

template <typename Parent, int Order_>
class MdView :
    public internals::MdAccessBase<MdView<Parent, Order_>, typename std::remove_const_t<Parent>::Scalar, Order_> {
    using Base = internals::MdAccessBase<MdView<Parent, Order_>, typename std::remove_const_t<Parent>::Scalar, Order_>;
    using raw_scalar = std::remove_const_t<typename std::remove_const_t<Parent>::Scalar>;
   public:
    using Scalar = typename Base::Scalar;
    static constexpr int Order = Order_;

    MdView() = delete;
    constexpr MdView(const MdView&) = default;
    constexpr MdView(
      Parent* parent, const std::array<int, Order>& extents, const std::array<int, Order>& strides, int offset,
      bool valid) :
        parent_(parent), extents_(extents), strides_(strides), offset_(offset), valid_(valid) { }

    constexpr MdView& operator=(const MdView& other) &
        requires(!std::is_const_v<Parent>)
    {
        return assign_inplace_from(other);
    }
    constexpr MdView& operator=(const MdView&) &
        requires(std::is_const_v<Parent>)
    = delete;
    template <internals::md_readable Other>
        requires(Other::Order == Order && !std::is_const_v<Parent> && !std::same_as<std::remove_cvref_t<Other>, MdView>)
    constexpr MdView& operator=(const Other& other) & {
        return assign_inplace_from(other);
    }

    template <internals::md_readable Other>
        requires(Other::Order == Order && !std::is_const_v<Parent>)
    constexpr MdView& assign_inplace_from(const Other& other) {
        bool compatible = other.valid() && other.size() == this->size();
        for (int axis = 0; axis < Order; ++axis) compatible = compatible && other.extent(axis) == this->extent(axis);
        fdapde_assert(compatible);
        if (!compatible) return *this;
        std::vector<raw_scalar> temporary;
        temporary.reserve(static_cast<std::size_t>(other.size()));
        for (const auto& value : other) temporary.push_back(static_cast<raw_scalar>(value));
        int i = 0;
        for (auto&& value : *this) value = temporary[static_cast<std::size_t>(i++)];
        return *this;
    }

    constexpr decltype(auto) operator[](int position) & { return this->md_coefficient_at_position(position); }
    constexpr decltype(auto) operator[](int position) const& { return this->md_coefficient_at_position(position); }
    constexpr void operator[](int) && = delete;
    constexpr void operator[](int) const&& = delete;

    constexpr int md_size() const noexcept {
        int size = 0;
        return valid_ && internals::checked_shape_size(extents_, size) ? size : 0;
    }
    constexpr int md_extent(int axis) const noexcept {
        const bool in_range = axis >= 0 && axis < Order;
        fdapde_assert(in_range);
        return valid_ && in_range ? extents_[axis] : 0;
    }
    constexpr bool md_valid() const noexcept { return valid_ && parent_ != nullptr && parent_->valid(); }
    constexpr int md_stride(int axis) const noexcept {
        const bool in_range = axis >= 0 && axis < Order;
        fdapde_assert(in_range);
        return valid_ && in_range ? strides_[axis] : 0;
    }
    constexpr int md_offset() const noexcept { return offset_; }
    constexpr decltype(auto) md_linear_at(int index) { return parent_->md_linear_at(index); }
    constexpr decltype(auto) md_linear_at(int index) const { return std::as_const(*parent_).md_linear_at(index); }
   private:
    Parent* parent_;
    std::array<int, Order> extents_;
    std::array<int, Order> strides_;
    int offset_;
    bool valid_;
};

template <typename Parent, int Order> using MdArrayBlock = MdView<Parent, Order>;
template <typename Parent, int Order> using MdArraySlice = MdView<Parent, Order>;

template <typename Scalar_, typename Extents_, int StorageOrder_ = RowMajor>
class MdArray : public internals::MdViewableBase<MdArray<Scalar_, Extents_, StorageOrder_>, Scalar_, Extents_::Order> {
    static_assert(!std::is_const_v<Scalar_>, "Owning MdArray scalars cannot be const");
    static_assert(StorageOrder_ == RowMajor || StorageOrder_ == ColMajor, "Unsupported MdArray storage order");
    using Base = internals::MdViewableBase<MdArray<Scalar_, Extents_, StorageOrder_>, Scalar_, Extents_::Order>;
    static constexpr int StorageSize = Extents_::DynamicOrder == 0 ? Extents_::StaticSize : Dynamic;
    using storage_t = Vector<Scalar_, StorageSize>;
   public:
    using Scalar = Scalar_;
    using extents_t = Extents_;
    using mapping_t = internals::MdMapping<extents_t, StorageOrder_>;
    static constexpr int Order = static_cast<int>(extents_t::Order);
    static constexpr int DynamicOrder = static_cast<int>(extents_t::DynamicOrder);
    static constexpr int StorageOrder = StorageOrder_;
    static constexpr auto static_extents = extents_t::static_extents;

    constexpr MdArray() : extents_(), mapping_(extents_), storage_() { }

    template <internals::md_index... Dims>
        requires(DynamicOrder > 0 && sizeof...(Dims) == extents_t::DynamicOrder)
    constexpr explicit MdArray(Dims... dims) : extents_(dims...), mapping_(extents_), storage_(extents_.size()) { }

    template <internals::md_index... Dims>
        requires(DynamicOrder > 0 && sizeof...(Dims) == extents_t::Order && sizeof...(Dims) != extents_t::DynamicOrder)
    constexpr explicit MdArray(Dims... dims) : extents_(dims...), mapping_(extents_), storage_(extents_.size()) { }

    template <internals::md_readable Other>
        requires(Other::Order == Order)
    constexpr explicit MdArray(const Other& other) : MdArray() {
        assign_from_(other);
    }

    constexpr MdArray& operator=(const MdArray& other) & {
        if (this == std::addressof(other)) return *this;
        const bool compatible = other.valid();
        fdapde_assert(compatible);
        if (!compatible) return *this;
        extents_ = other.extents_;
        mapping_ = other.mapping_;
        storage_ = other.storage_;
        return *this;
    }

    template <internals::md_readable Other>
        requires(Other::Order == Order && !std::same_as<std::remove_cvref_t<Other>, MdArray>)
    constexpr MdArray& operator=(const Other& other) & {
        assign_from_(other);
        return *this;
    }

    template <internals::md_index... Dims>
        requires(DynamicOrder > 0 && sizeof...(Dims) == extents_t::DynamicOrder)
    constexpr bool resize(Dims... dims) {
        extents_t candidate(extents_);
        if (!candidate.resize(dims...)) return false;
        return resize_from_extents_(candidate);
    }

    template <internals::md_index... Dims>
        requires(DynamicOrder > 0 && sizeof...(Dims) == extents_t::Order && sizeof...(Dims) != extents_t::DynamicOrder)
    constexpr bool resize(Dims... dims) {
        extents_t candidate(extents_);
        if (!candidate.resize(dims...)) return false;
        return resize_from_extents_(candidate);
    }

    constexpr decltype(auto) operator[](int index) & {
        const bool valid = index >= 0 && index < md_size();
        fdapde_assert(valid);
        return storage_[valid ? index : 0];
    }
    constexpr decltype(auto) operator[](int index) const& {
        const bool valid = index >= 0 && index < md_size();
        fdapde_assert(valid);
        return storage_[valid ? index : 0];
    }
    constexpr void operator[](int) && = delete;
    constexpr void operator[](int) const&& = delete;

    constexpr decltype(auto) data() { return storage_.data(); }
    constexpr decltype(auto) data() const { return storage_.data(); }
    constexpr const mapping_t& mapping() const noexcept { return mapping_; }
    constexpr int bitpacks() const
        requires(std::same_as<Scalar, bool>)
    {
        return storage_.bitpacks();
    }
    constexpr void set()
        requires(std::same_as<Scalar, bool>)
    {
        storage_.set();
    }
    constexpr void clear()
        requires(std::same_as<Scalar, bool>)
    {
        storage_.clear();
    }

    constexpr int md_size() const noexcept { return extents_.size(); }
    constexpr int md_extent(int axis) const noexcept { return extents_.extent(static_cast<std::size_t>(axis)); }
    constexpr bool md_valid() const noexcept { return extents_.valid(); }
    constexpr int md_stride(int axis) const noexcept { return mapping_.stride(axis); }
    constexpr int md_offset() const noexcept { return 0; }
    constexpr decltype(auto) md_linear_at(int index) { return storage_[index]; }
    constexpr decltype(auto) md_linear_at(int index) const { return storage_[index]; }
   private:
    template <internals::md_readable Other> constexpr void assign_from_(const Other& other) {
        std::array<int, Order> dimensions {};
        for (int axis = 0; axis < Order; ++axis) dimensions[axis] = other.extent(axis);
        extents_t candidate(extents_);
        bool compatible = other.valid();
        if constexpr (DynamicOrder > 0) {
            compatible = compatible && candidate.resize(dimensions);
        } else {
            for (int axis = 0; axis < Order; ++axis)
                compatible = compatible && dimensions[axis] == static_extents[axis];
        }
        compatible = compatible && other.size() == candidate.size();
        fdapde_assert(compatible);
        if (!compatible) return;

        std::vector<Scalar> temporary;
        temporary.reserve(static_cast<std::size_t>(other.size()));
        for (const auto& value : other) temporary.push_back(static_cast<Scalar>(value));
        if constexpr (DynamicOrder > 0) resize_from_extents_(candidate);
        int i = 0;
        for (auto&& value : *this) value = temporary[static_cast<std::size_t>(i++)];
    }

    constexpr bool resize_from_extents_(const extents_t& extents) {
        if (extents == extents_) return true;
        storage_.resize(extents.size());
        extents_ = extents;
        mapping_ = mapping_t(extents_);
        return true;
    }

    extents_t extents_;
    mapping_t mapping_;
    storage_t storage_;
};

template <typename Scalar_, typename Extents_, int StorageOrder_ = RowMajor>
class MdMap : public internals::MdViewableBase<MdMap<Scalar_, Extents_, StorageOrder_>, Scalar_, Extents_::Order> {
    static_assert(StorageOrder_ == RowMajor || StorageOrder_ == ColMajor, "Unsupported MdMap storage order");
    using Base = internals::MdViewableBase<MdMap<Scalar_, Extents_, StorageOrder_>, Scalar_, Extents_::Order>;
   public:
    using Scalar = Scalar_;
    using extents_t = Extents_;
    using mapping_t = internals::MdMapping<extents_t, StorageOrder_>;
    using pointer = Scalar_*;
    static constexpr int Order = static_cast<int>(extents_t::Order);
    static constexpr int DynamicOrder = static_cast<int>(extents_t::DynamicOrder);
    static constexpr int StorageOrder = StorageOrder_;
    static constexpr auto static_extents = extents_t::static_extents;

    constexpr MdMap()
        requires(DynamicOrder > 0)
        : extents_(), mapping_(extents_), data_(nullptr), valid_(true) { }
    constexpr MdMap()
        requires(DynamicOrder == 0)
    = delete;

    constexpr explicit MdMap(pointer data)
        requires(DynamicOrder == 0)
        :
        extents_(),
        mapping_(extents_),
        data_(data),
        valid_(extents_.valid() && (data != nullptr || extents_.size() == 0)) {
        fdapde_assert(valid_);
    }

    template <internals::md_index... Dims>
        requires(DynamicOrder > 0 && sizeof...(Dims) == extents_t::DynamicOrder)
    constexpr MdMap(pointer data, Dims... dims) :
        extents_(dims...),
        mapping_(extents_),
        data_(data),
        valid_(extents_.valid() && (data != nullptr || extents_.size() == 0)) {
        fdapde_assert(valid_);
    }

    template <internals::md_index... Dims>
        requires(DynamicOrder > 0 && sizeof...(Dims) == extents_t::Order && sizeof...(Dims) != extents_t::DynamicOrder)
    constexpr MdMap(pointer data, Dims... dims) :
        extents_(dims...),
        mapping_(extents_),
        data_(data),
        valid_(extents_.valid() && (data != nullptr || extents_.size() == 0)) {
        fdapde_assert(valid_);
    }

    constexpr decltype(auto) operator[](int index) & {
        const bool valid = valid_ && index >= 0 && index < md_size();
        fdapde_assert(valid);
        return data_[valid ? index : 0];
    }
    constexpr decltype(auto) operator[](int index) const& {
        const bool valid = valid_ && index >= 0 && index < md_size();
        fdapde_assert(valid);
        using raw_scalar = std::remove_const_t<Scalar>;
        return static_cast<const raw_scalar&>(data_[valid ? index : 0]);
    }
    constexpr void operator[](int) && = delete;
    constexpr void operator[](int) const&& = delete;

    constexpr pointer data() noexcept { return data_; }
    constexpr const std::remove_const_t<Scalar>* data() const noexcept { return data_; }
    constexpr const mapping_t& mapping() const noexcept { return mapping_; }

    constexpr int md_size() const noexcept { return valid_ ? extents_.size() : 0; }
    constexpr int md_extent(int axis) const noexcept {
        return valid_ ? extents_.extent(static_cast<std::size_t>(axis)) : 0;
    }
    constexpr bool md_valid() const noexcept { return valid_; }
    constexpr int md_stride(int axis) const noexcept { return mapping_.stride(axis); }
    constexpr int md_offset() const noexcept { return 0; }
    constexpr decltype(auto) md_linear_at(int index) { return data_[index]; }
    constexpr decltype(auto) md_linear_at(int index) const {
        using raw_scalar = std::remove_const_t<Scalar>;
        return static_cast<const raw_scalar&>(data_[index]);
    }
   private:
    extents_t extents_;
    mapping_t mapping_;
    pointer data_;
    bool valid_;
};

template <typename Parent, typename... Slicers> constexpr auto submdarray(Parent& parent, Slicers... slicers) {
    return parent.block(slicers...);
}
template <typename Parent, typename... Slicers> void submdarray(Parent&&, Slicers...) = delete;

}   // namespace fdapde::linalg

#endif   // __FDAPDE_LINALG_NATIVE_MDARRAY_H__
