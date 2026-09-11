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

#ifndef __FDAPDE_LINALG_MDARRAY_H__
#define __FDAPDE_LINALG_MDARRAY_H__

#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <iterator>
#include <limits>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "header_check.h"

namespace fdapde {

inline constexpr int full_extent = -2;

namespace internals {

template <typename T>
concept md_index = std::integral<std::remove_cvref_t<T>> && (!std::same_as<std::remove_cvref_t<T>, bool>);

/// @brief converts an index after checking its supported range
template <md_index T> constexpr bool checked_index_cast(T value, int& result) {
    if (!std::in_range<int>(value)) return false;
    result = static_cast<int>(value);
    return true;
}

/// @brief validates and converts an axis length
template <md_index T> constexpr int checked_md_dimension(T value) {
    fdapde_assert(!std::cmp_less(value, 0), std::invalid_argument, "MdArray extents must be nonnegative");
    fdapde_strong_assert(
      std::in_range<int>(value), std::length_error, "MdArray extent exceeds the supported int range");
    const int result = static_cast<int>(value);
    fdapde_assert(!(result < 0), std::invalid_argument, "MdArray extents must be nonnegative");
    return result;
}

/// @brief validates and converts a multidimensional coordinate
template <md_index T> constexpr int checked_md_coordinate(T value) {
    fdapde_assert(!(!std::in_range<int>(value)), std::out_of_range, "MdArray index is outside the supported range");
    return static_cast<int>(value);
}

/// @brief computes the shape size with overflow checks
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

/// @brief computes the multidimensional coefficient count with overflow checks
template <std::size_t Size> constexpr int checked_md_shape_size(const std::array<int, Size>& extents) {
    bool empty = false;
    for (int extent : extents) {
        fdapde_assert(!(extent < 0), std::invalid_argument, "MdArray extents must be nonnegative");
        empty = empty || extent == 0;
    }
    if (empty) return 0;

    std::size_t product = 1;
    for (int extent : extents) {
        fdapde_strong_assert(
          !(product > static_cast<std::size_t>(std::numeric_limits<int>::max()) / static_cast<std::size_t>(extent)),
          std::length_error, "MdArray size exceeds the supported int range");
        product *= static_cast<std::size_t>(extent);
    }
    return static_cast<int>(product);
}

/// @brief computes the compile-time multidimensional coefficient count
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

/// @brief checks that slice axes are distinct
template <int... Values> consteval bool unique_axes() {
    constexpr std::array<int, sizeof...(Values)> values {Values...};
    for (std::size_t i = 0; i < values.size(); ++i) {
        for (std::size_t j = i + 1; j < values.size(); ++j) {
            if (values[i] == values[j]) return false;
        }
    }
    return true;
}

template <typename T>
concept md_slicer = md_index<T> || fdapde::internals::is_pair_v<std::remove_cvref_t<T>>;

template <typename T>
concept md_index_pack = requires(const T& value) {
    { value.size() } -> std::convertible_to<std::size_t>;
    requires md_index<decltype(value[0])>;
};

}   // namespace internals

/// @brief stores and validates multidimensional extents
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

    fdapde_static_assert(DynamicOrder != 0 || StaticSize != Dynamic, MDARRAY_STATIC_SIZE_EXCEEDS_SUPPORTED_RANGE);

    /// @brief initializes static axis lengths and sets each dynamic axis length to zero
    constexpr MdExtents() : extents_ {((StaticExtents == Dynamic) ? 0 : StaticExtents)...}, valid_(true) { }

    /// @brief sets only the dynamic axis lengths in declaration order
    template <internals::md_index... Dims>
        requires(DynamicOrder > 0 && sizeof...(Dims) == DynamicOrder)
    constexpr explicit MdExtents(Dims... dims) : MdExtents() {
        assign_dynamic_(std::array<int, sizeof...(Dims)> {internals::checked_md_dimension(dims)...});
    }

    /// @brief sets all axis lengths and checks agreement with compile-time extents
    template <internals::md_index... Dims>
        requires(DynamicOrder > 0 && sizeof...(Dims) == Order && sizeof...(Dims) != DynamicOrder)
    constexpr explicit MdExtents(Dims... dims) : MdExtents() {
        assign_full_(std::array<int, sizeof...(Dims)> {internals::checked_md_dimension(dims)...});
    }

    /// @brief validates every axis length supplied in the extent array
    template <internals::md_index T> constexpr explicit MdExtents(const std::array<T, Order>& dims) : MdExtents() {
        assign_full_(dims);
    }

    /// @brief returns the number of axes
    constexpr order_t order() const noexcept { return Order; }
    /// @brief returns the number of runtime-sized axes
    constexpr order_t order_dynamic() const noexcept { return DynamicOrder; }
    /// @brief returns the requested axis length
    constexpr int extent(order_t axis) const {
        fdapde_assert(!(axis >= Order), std::out_of_range, "MdArray extent axis is out of range");
        return valid_ ? extents_[axis] : 0;
    }
    /// @brief returns the coefficient count
    constexpr int size() const {
        if (!valid_) return 0;
        int result = 0;
        const bool valid = internals::checked_shape_size(extents_, result);
        fdapde_assert(valid, std::logic_error, "invalid mdarray precondition");
        return valid ? result : 0;
    }
    /// @brief reports whether the shape and storage are valid
    constexpr bool valid() const noexcept { return valid_; }

    /// @brief resizes the owned storage to the requested dimensions
    template <internals::md_index... Dims>
        requires(DynamicOrder > 0 && sizeof...(Dims) == DynamicOrder)
    constexpr bool resize(Dims... dims) {
        MdExtents candidate(*this);
        candidate.assign_dynamic_(std::array<int, sizeof...(Dims)> {internals::checked_md_dimension(dims)...});
        *this = candidate;
        valid_ = true;
        return true;
    }

    /// @brief resizes the owned storage to the requested dimensions
    template <internals::md_index... Dims>
        requires(DynamicOrder > 0 && sizeof...(Dims) == Order && sizeof...(Dims) != DynamicOrder)
    constexpr bool resize(Dims... dims) {
        MdExtents candidate(*this);
        candidate.assign_full_(std::array<int, sizeof...(Dims)> {internals::checked_md_dimension(dims)...});
        *this = candidate;
        valid_ = true;
        return true;
    }

    /// @brief resizes the owned storage to the requested dimensions
    template <internals::md_index T> constexpr bool resize(const std::array<T, Order>& dims) {
        MdExtents candidate(*this);
        candidate.assign_full_(dims);
        *this = candidate;
        valid_ = true;
        return true;
    }

    /// @brief compares iterator positions
    friend constexpr bool operator==(const MdExtents&, const MdExtents&) = default;
   private:
    /// @brief assigns runtime extents to the dynamic axes
    template <typename Dims, std::size_t... Is>
    constexpr bool assign_dynamic_impl_(const Dims& dims, std::index_sequence<Is...>) {
        const std::array<int, DynamicOrder> values {internals::checked_md_dimension(dims[Is])...};
        std::size_t dynamic_index = 0;
        for (std::size_t axis = 0; axis < Order; ++axis) {
            if (static_extents[axis] == Dynamic) extents_[axis] = values[dynamic_index++];
        }
        static_cast<void>(internals::checked_md_shape_size(extents_));
        return true;
    }

    /// @brief validates and assigns the dynamic axis lengths
    template <typename Dims> constexpr bool assign_dynamic_(const Dims& dims) {
        return assign_dynamic_impl_(dims, std::make_index_sequence<DynamicOrder> {});
    }

    /// @brief assigns the full axis-length sequence
    template <typename Dims, std::size_t... Is>
    constexpr bool assign_full_impl_(const Dims& dims, std::index_sequence<Is...>) {
        const std::array<int, Order> values {internals::checked_md_dimension(dims[Is])...};
        for (std::size_t axis = 0; axis < Order; ++axis) {
            fdapde_assert(
              !(static_extents[axis] != Dynamic && values[axis] != static_extents[axis]), std::invalid_argument,
              "MdArray extents do not match the static shape");
        }
        static_cast<void>(internals::checked_md_shape_size(values));
        extents_ = values;
        return true;
    }

    /// @brief validates and assigns all axis lengths
    template <typename Dims> constexpr bool assign_full_(const Dims& dims) {
        return assign_full_impl_(dims, std::make_index_sequence<Order> {});
    }

    std::array<int, Order> extents_;
    bool valid_;
};

namespace internals {

/// @brief forms an extent type with every axis sized at runtime
template <int N, std::size_t... Is>
auto make_full_dynamic_extents(std::index_sequence<Is...>) -> MdExtents<((void)Is, Dynamic)...>;

/// @brief maps multidimensional coordinates to contiguous row-major or column-major storage
template <typename Extents, int StorageOrder> class MdMapping {
    static_assert(StorageOrder == RowMajor || StorageOrder == ColMajor, "Unsupported MdArray storage order");
   public:
    static constexpr int Order = static_cast<int>(Extents::Order);
    using extents_t = Extents;

    /// @brief derives contiguous strides from default extents and the selected storage order
    constexpr MdMapping() : extents_(), strides_() { initialize_(); }
    /// @brief derives contiguous strides from the supplied extents and storage order
    constexpr explicit MdMapping(const Extents& extents) : extents_(extents), strides_() { initialize_(); }

    /// @brief returns the requested axis stride
    constexpr int stride(int axis) const {
        fdapde_assert(!(axis < 0 || axis >= Order), std::out_of_range, "MdArray stride axis is out of range");
        return strides_[axis];
    }
    /// @brief returns the stored multidimensional extents
    constexpr const Extents& extents() const noexcept { return extents_; }

    /// @brief validates separate axis indices and returns their physical storage offset
    template <internals::md_index... Indices>
        requires(sizeof...(Indices) == Extents::Order)
    constexpr int operator()(Indices... indices) const {
        const std::array<int, Order> values {internals::checked_md_coordinate(indices)...};
        fdapde_assert(!(!indices_valid_(values)), std::out_of_range, "MdArray index is out of range");
        return map_unchecked_(values);
    }

    /// @brief validates an index array and returns its physical storage offset
    constexpr int operator()(const std::array<int, Order>& indices) const {
        fdapde_assert(!(!indices_valid_(indices)), std::out_of_range, "MdArray index is out of range");
        return map_unchecked_(indices);
    }

    /// @brief compares iterator positions
    friend constexpr bool operator==(const MdMapping&, const MdMapping&) = default;
   private:
    /// @brief initializes view extents and parent-axis mapping
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

    /// @brief checks that every coordinate lies within its axis
    constexpr bool indices_valid_(const std::array<int, Order>& indices) const {
        for (int axis = 0; axis < Order; ++axis) {
            if (indices[axis] < 0 || indices[axis] >= extents_.extent(static_cast<std::size_t>(axis))) return false;
        }
        return true;
    }

    /// @brief maps coordinates to storage without bounds checks
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

/// @brief views a strided region of a multidimensional parent
template <typename Parent, int Order> class MdView;

namespace internals {

/// @brief provides checked coordinate access and logical row-order iteration
template <typename Derived, typename Scalar_, int Order_> class MdAccessBase {
   public:
    using Scalar = Scalar_;
    static constexpr int Order = Order_;

    /// @brief returns the coefficient count
    constexpr int size() const { return derived_().md_size(); }
    /// @brief returns the requested axis length
    constexpr int extent(int axis) const { return derived_().md_extent(axis); }
    /// @brief reports whether the shape and storage are valid
    constexpr bool valid() const noexcept { return derived_().md_valid(); }

    /// @brief accesses the coefficient at the supplied axis indices
    template <internals::md_index... Indices>
        requires(sizeof...(Indices) == Order)
    constexpr decltype(auto) operator()(Indices... indices) & {
        return coefficient_(make_indices_(indices...));
    }
    /// @brief reads the coefficient at the supplied axis indices through const access
    template <internals::md_index... Indices>
        requires(sizeof...(Indices) == Order)
    constexpr decltype(auto) operator()(Indices... indices) const& {
        return coefficient_(make_indices_(indices...));
    }
    /// @brief rejects coefficient access through a temporary parent
    template <internals::md_index... Indices> constexpr void operator()(Indices...) && = delete;
    /// @brief rejects coefficient access through a const temporary parent
    template <internals::md_index... Indices> constexpr void operator()(Indices...) const&& = delete;

    /// @brief accesses a coefficient using one container of axis indices
    template <internals::md_index_pack IndexPack> constexpr decltype(auto) operator()(const IndexPack& indices) & {
        return coefficient_(indices_from_pack_(indices));
    }
    /// @brief reads a coefficient using one container of axis indices through const access
    template <internals::md_index_pack IndexPack> constexpr decltype(auto) operator()(const IndexPack& indices) const& {
        return coefficient_(indices_from_pack_(indices));
    }

    /// @brief iterates over the logical coefficients of a multidimensional expression
    template <typename Owner> class iterator {
       public:
        using difference_type = std::ptrdiff_t;
        using value_type = std::remove_cv_t<Scalar>;
        using reference = decltype(std::declval<Owner&>().md_coefficient_at_position(0));
        using iterator_category = std::forward_iterator_tag;

        /// @brief creates a singular iterator without an array or logical position
        constexpr iterator() noexcept : owner_(nullptr), position_(0) { }
        /// @brief binds a parent array and a position in logical row-order traversal
        constexpr iterator(Owner* owner, int position) noexcept : owner_(owner), position_(position) { }
        /// @brief dereferences the current iterator position
        constexpr decltype(auto) operator*() const { return owner_->md_coefficient_at_position(position_); }
        /// @brief advances the iterator
        constexpr iterator& operator++() noexcept {
            ++position_;
            return *this;
        }
        /// @brief advances the iterator
        constexpr iterator operator++(int) noexcept {
            iterator copy(*this);
            ++(*this);
            return copy;
        }
        /// @brief returns the physical storage index of the current iterator position
        constexpr int mapped_index() const { return owner_->md_mapped_index_at_position(position_); }
        /// @brief compares iterator positions
        friend constexpr bool operator==(const iterator&, const iterator&) = default;
       private:
        Owner* owner_;
        int position_;
    };

    /// @brief returns an iterator to the first coefficient
    constexpr auto begin() & noexcept { return iterator<Derived>(std::addressof(derived_()), 0); }
    /// @brief returns the past-the-end iterator
    constexpr auto end() & noexcept { return iterator<Derived>(std::addressof(derived_()), size()); }
    /// @brief returns an iterator to the first coefficient
    constexpr auto begin() const& noexcept { return iterator<const Derived>(std::addressof(derived_()), 0); }
    /// @brief returns the past-the-end iterator
    constexpr auto end() const& noexcept { return iterator<const Derived>(std::addressof(derived_()), size()); }
    /// @brief rejects iterators whose parent wrapper would be destroyed at the end of the expression
    constexpr void begin() && = delete;
    /// @brief rejects iterators whose parent wrapper would be destroyed at the end of the expression
    constexpr void end() && = delete;
    /// @brief rejects iterators whose parent wrapper would be destroyed at the end of the expression
    constexpr void begin() const&& = delete;
    /// @brief rejects iterators whose parent wrapper would be destroyed at the end of the expression
    constexpr void end() const&& = delete;

    /// @brief accesses a coefficient in logical iteration order
    constexpr decltype(auto) md_coefficient_at_position(int position) & {
        fdapde_assert(!(position < 0 || position >= size()), std::out_of_range, "MdArray position is out of range");
        return coefficient_(indices_at_position_(position));
    }
    /// @brief accesses a coefficient in logical iteration order
    constexpr decltype(auto) md_coefficient_at_position(int position) const& {
        fdapde_assert(!(position < 0 || position >= size()), std::out_of_range, "MdArray position is out of range");
        return coefficient_(indices_at_position_(position));
    }
    /// @brief maps a logical iterator position to parent storage
    constexpr int md_mapped_index_at_position(int position) const {
        fdapde_assert(!(position < 0 || position >= size()), std::out_of_range, "MdArray position is out of range");
        const auto indices = indices_at_position_(position);
        int result = derived_().md_offset();
        for (int axis = 0; axis < Order; ++axis) result += indices[axis] * derived_().md_stride(axis);
        return result;
    }
   protected:
    /// @brief returns the concrete multidimensional expression
    constexpr Derived& derived_() { return static_cast<Derived&>(*this); }
    /// @brief returns the concrete multidimensional expression
    constexpr const Derived& derived_() const { return static_cast<const Derived&>(*this); }
   private:
    /// @brief converts a coordinate argument pack to an index array
    template <internals::md_index... Indices> constexpr std::array<int, Order> make_indices_(Indices... indices) const {
        return {internals::checked_md_coordinate(indices)...};
    }

    /// @brief validates and converts a coordinate container
    template <typename IndexPack> constexpr std::array<int, Order> indices_from_pack_(const IndexPack& indices) const {
        std::array<int, Order> result {};
        fdapde_assert(
          !(!std::cmp_equal(indices.size(), Order)), std::invalid_argument, "MdArray index pack has the wrong rank");
        for (int axis = 0; axis < Order; ++axis) { result[axis] = internals::checked_md_coordinate(indices[axis]); }
        return result;
    }

    /// @brief decodes a logical iteration position into coordinates
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

    /// @brief checks that every coordinate lies within its axis
    constexpr bool indices_valid_(const std::array<int, Order>& indices) const {
        if (!valid()) return false;
        for (int axis = 0; axis < Order; ++axis) {
            if (indices[axis] < 0 || indices[axis] >= extent(axis)) return false;
        }
        return true;
    }

    /// @brief accesses storage at the mapped multidimensional coordinate
    constexpr decltype(auto) coefficient_(const std::array<int, Order>& indices) & {
        fdapde_assert(!(!indices_valid_(indices)), std::out_of_range, "MdArray index is out of range");
        int mapped = derived_().md_offset();
        for (int axis = 0; axis < Order; ++axis) mapped += indices[axis] * derived_().md_stride(axis);
        return derived_().md_linear_at(mapped);
    }
    /// @brief accesses storage at the mapped multidimensional coordinate
    constexpr decltype(auto) coefficient_(const std::array<int, Order>& indices) const& {
        fdapde_assert(!(!indices_valid_(indices)), std::out_of_range, "MdArray index is out of range");
        int mapped = derived_().md_offset();
        for (int axis = 0; axis < Order; ++axis) mapped += indices[axis] * derived_().md_stride(axis);
        return derived_().md_linear_at(mapped);
    }
};

/// @brief constructs borrowed blocks and slices from persistent multidimensional parents
template <typename Derived, typename Scalar, int Order>
class MdViewableBase : public MdAccessBase<Derived, Scalar, Order> {
    using Base = MdAccessBase<Derived, Scalar, Order>;
   public:
    using Base::operator();

    /// @brief returns a view of the requested rectangular region
    template <typename... Slicers>
        requires(sizeof...(Slicers) == Order && (internals::md_slicer<Slicers> && ...))
    constexpr auto block(Slicers... slicers) & {
        return make_block_(this->derived_(), slicers...);
    }
    /// @brief returns a view of the requested rectangular region
    template <typename... Slicers>
        requires(sizeof...(Slicers) == Order && (internals::md_slicer<Slicers> && ...))
    constexpr auto block(Slicers... slicers) const& {
        return make_block_(std::as_const(this->derived_()), slicers...);
    }
    /// @brief rejects borrowing a block from a temporary parent
    template <typename... Slicers> constexpr void block(Slicers...) && = delete;
    /// @brief rejects borrowing a block from a temporary parent
    template <typename... Slicers> constexpr void block(Slicers...) const&& = delete;

    /// @brief returns a view with the selected axes fixed
    template <int... Axes, internals::md_index... Indices>
        requires(sizeof...(Axes) == sizeof...(Indices) && sizeof...(Axes) > 0 && sizeof...(Axes) < Order)
    constexpr auto slice(Indices... indices) & {
        return make_slice_<Axes...>(this->derived_(), indices...);
    }
    /// @brief returns a view with the selected axes fixed
    template <int... Axes, internals::md_index... Indices>
        requires(sizeof...(Axes) == sizeof...(Indices) && sizeof...(Axes) > 0 && sizeof...(Axes) < Order)
    constexpr auto slice(Indices... indices) const& {
        return make_slice_<Axes...>(std::as_const(this->derived_()), indices...);
    }
    /// @brief rejects borrowing a slice from a temporary parent
    template <int... Axes, typename... Indices> constexpr void slice(Indices...) && = delete;
    /// @brief rejects borrowing a slice from a temporary parent
    template <int... Axes, typename... Indices> constexpr void slice(Indices...) const&& = delete;

    /// @brief returns a view of the requested row
    constexpr auto row(int index) &
        requires(Order == 2)
    {
        return block(index, full_extent);
    }
    /// @brief returns a view of the requested row
    constexpr auto row(int index) const&
        requires(Order == 2)
    {
        return block(index, full_extent);
    }
    /// @brief returns a view of the requested column
    constexpr auto col(int index) &
        requires(Order == 2)
    {
        return block(full_extent, index);
    }
    /// @brief returns a view of the requested column
    constexpr auto col(int index) const&
        requires(Order == 2)
    {
        return block(full_extent, index);
    }
    /// @brief rejects borrowing a row view from a temporary parent
    constexpr void row(int) && = delete;
    /// @brief rejects borrowing a row view from a temporary parent
    constexpr void row(int) const&& = delete;
    /// @brief rejects borrowing a column view from a temporary parent
    constexpr void col(int) && = delete;
    /// @brief rejects borrowing a column view from a temporary parent
    constexpr void col(int) const&& = delete;
   private:
    /// @brief constructs a view from the selected axis ranges
    template <typename Parent, typename Tuple, std::size_t... Is>
    static constexpr auto make_block_impl_(Parent& parent, const Tuple& slicers, std::index_sequence<Is...>) {
        std::array<int, Order> extents {};
        std::array<int, Order> strides {};
        fdapde_assert(!(!parent.valid()), std::invalid_argument, "cannot create a view from an invalid MdArray");
        int offset = parent.md_offset();
        bool valid = true;

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
        fdapde_assert(!(!valid), std::out_of_range, "MdArray block is out of range");
        return MdView<Parent, Order>(std::addressof(parent), extents, strides, offset, true);
    }

    /// @brief validates axis ranges and constructs a block view
    template <typename Parent, typename... Slicers>
    static constexpr auto make_block_(Parent& parent, Slicers... slicers) {
        return make_block_impl_(parent, std::tuple<Slicers...>(slicers...), std::index_sequence_for<Slicers...> {});
    }

    /// @brief constructs a view with selected axes fixed
    template <int... Axes, typename Parent, typename... Indices>
    static constexpr auto make_slice_(Parent& parent, Indices... indices) {
        static_assert(((Axes >= 0 && Axes < Order) && ...), "Slice axis is out of range");
        static_assert(internals::unique_axes<Axes...>(), "Slice axes must be unique");
        constexpr int ViewOrder = Order - sizeof...(Axes);
        constexpr std::array<int, sizeof...(Axes)> axes {Axes...};
        std::array<int, sizeof...(Axes)> fixed {};
        std::size_t fixed_index = 0;
        fdapde_assert(!(!parent.valid()), std::invalid_argument, "cannot create a view from an invalid MdArray");
        bool valid = (internals::checked_index_cast(indices, fixed[fixed_index++]) && ...);
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
        fdapde_assert(!(!valid), std::out_of_range, "MdArray slice is out of range");
        return MdView<Parent, ViewOrder>(std::addressof(parent), extents, strides, offset, true);
    }
};

}   // namespace internals

/// @brief views a strided region of a multidimensional parent
template <typename Parent, int Order_>
// non-owning view. Operations that replace or release the parent's storage invalidate the view; direct construction
// also leaves offset and stride validity to the caller. Prefer the checked block, slice, row, and column factories.
class MdView :
    public internals::MdAccessBase<MdView<Parent, Order_>, typename std::remove_const_t<Parent>::Scalar, Order_> {
    using Base = internals::MdAccessBase<MdView<Parent, Order_>, typename std::remove_const_t<Parent>::Scalar, Order_>;
    using raw_scalar = std::remove_const_t<typename std::remove_const_t<Parent>::Scalar>;
   public:
    using Scalar = typename Base::Scalar;
    static constexpr int Order = Order_;
    static constexpr int ReadOnly =
      std::is_const_v<Parent> || std::is_const_v<typename std::remove_const_t<Parent>::Scalar>;

    /// @brief rejects a view without a parent and an explicit coordinate mapping
    MdView() = delete;
    /// @brief copies view metadata while sharing the same parent array
    constexpr MdView(const MdView&) = default;
    /// @brief binds a parent with precomputed extents, strides, offset and validity
    constexpr MdView(
      Parent* parent, const std::array<int, Order>& extents, const std::array<int, Order>& strides, int offset,
      bool valid) :
        parent_(parent), extents_(extents), strides_(strides), offset_(offset), valid_(valid) { }

    /// @brief copies a source snapshot into the view without changing its mapping
    constexpr MdView& operator=(const MdView& other) &
        requires(ReadOnly == 0)
    {
        return assign_inplace_from(other);
    }
    /// @brief rejects assignment through a read-only multidimensional view
    constexpr MdView& operator=(const MdView&) &
        requires(ReadOnly != 0)
    = delete;
    /// @brief copies a source snapshot with the same number of dimensions into the view without changing its mapping
    template <internals::md_readable Other>
        requires(Other::Order == Order && ReadOnly == 0 && !std::same_as<std::remove_cvref_t<Other>, MdView>)
    constexpr MdView& operator=(const Other& other) & {
        return assign_inplace_from(other);
    }

    /// @brief copies coefficients into this view after checking its extents
    template <internals::md_readable Other>
        requires(Other::Order == Order && ReadOnly == 0)
    constexpr MdView& assign_inplace_from(const Other& other) {
        bool compatible = other.valid() && other.size() == this->size();
        for (int axis = 0; axis < Order; ++axis) compatible = compatible && other.extent(axis) == this->extent(axis);
        fdapde_assert(!(!compatible), std::invalid_argument, "MdArray view dimensions do not match");
        std::vector<raw_scalar> temporary;
        temporary.reserve(static_cast<std::size_t>(other.size()));
        for (const auto& value : other) temporary.push_back(static_cast<raw_scalar>(value));
        int i = 0;
        for (auto&& value : *this) value = temporary[static_cast<std::size_t>(i++)];
        return *this;
    }

    /// @brief accesses the coefficient at a logical row-order position within the view
    constexpr decltype(auto) operator[](int position) & { return this->md_coefficient_at_position(position); }
    /// @brief reads the coefficient at a logical row-order position within the const view
    constexpr decltype(auto) operator[](int position) const& { return this->md_coefficient_at_position(position); }
    /// @brief rejects indexed access through a temporary parent to prevent dangling references or proxies
    constexpr void operator[](int) && = delete;
    /// @brief rejects indexed access through a temporary parent to prevent dangling references or proxies
    constexpr void operator[](int) const&& = delete;

    /// @brief returns the view's logical coefficient count, or zero for an invalid shape
    constexpr int md_size() const noexcept {
        int size = 0;
        return valid_ && internals::checked_shape_size(extents_, size) ? size : 0;
    }
    /// @brief returns the selected view axis length after validating the axis
    constexpr int md_extent(int axis) const {
        fdapde_assert(!(axis < 0 || axis >= Order), std::out_of_range, "MdArray extent axis is out of range");
        return valid_ ? extents_[axis] : 0;
    }
    /// @brief requires a valid view mapping and a live valid parent
    constexpr bool md_valid() const noexcept { return valid_ && parent_ != nullptr && parent_->valid(); }
    /// @brief returns the view's physical stride along the selected axis
    constexpr int md_stride(int axis) const {
        fdapde_assert(!(axis < 0 || axis >= Order), std::out_of_range, "MdArray stride axis is out of range");
        return valid_ ? strides_[axis] : 0;
    }
    /// @brief returns the view origin's physical offset in parent storage
    constexpr int md_offset() const noexcept { return offset_; }
    /// @brief accesses the parent storage at the supplied physical position
    constexpr decltype(auto) md_linear_at(int index) { return parent_->md_linear_at(index); }
    /// @brief accesses the parent storage at the supplied physical position
    constexpr decltype(auto) md_linear_at(int index) const { return std::as_const(*parent_).md_linear_at(index); }
   private:
    Parent* parent_;
    std::array<int, Order> extents_;
    std::array<int, Order> strides_;
    int offset_;
    bool valid_;
};

/// @brief owns a multidimensional array with the selected storage order
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

    /// @brief value-initializes fixed storage and leaves dynamic axes empty
    constexpr MdArray() : extents_(), mapping_(extents_), storage_() { }

    /// @brief allocates value-initialized storage using only the dynamic axis lengths
    template <internals::md_index... Dims>
        requires(DynamicOrder > 0 && sizeof...(Dims) == extents_t::DynamicOrder)
    constexpr explicit MdArray(Dims... dims) : extents_(dims...), mapping_(extents_), storage_(extents_.size()) { }

    /// @brief validates all axis lengths and allocates value-initialized storage
    template <internals::md_index... Dims>
        requires(DynamicOrder > 0 && sizeof...(Dims) == extents_t::Order && sizeof...(Dims) != extents_t::DynamicOrder)
    constexpr explicit MdArray(Dims... dims) : extents_(dims...), mapping_(extents_), storage_(extents_.size()) { }

    /// @brief copies extents, mapping and coefficients into independent storage
    constexpr MdArray(const MdArray&) = default;

    /// @brief copies a readable array or view by logical coordinates into independent storage
    template <internals::md_readable Other>
        requires(Other::Order == Order)
    constexpr explicit MdArray(const Other& other) : MdArray() {
        assign_from_(other);
    }

    /// @brief copies the source shape and coefficients into independent storage
    constexpr MdArray& operator=(const MdArray& other) & {
        if (this == std::addressof(other)) return *this;
        assign_from_(other);
        return *this;
    }

    /// @brief evaluates a readable source with the same number of dimensions into independent storage, adopting
    /// compatible dynamic extents
    template <internals::md_readable Other>
        requires(Other::Order == Order && !std::same_as<std::remove_cvref_t<Other>, MdArray>)
    constexpr MdArray& operator=(const Other& other) & {
        assign_from_(other);
        return *this;
    }

    /// @brief resizes the owned storage to the requested dimensions
    template <internals::md_index... Dims>
        requires(DynamicOrder > 0 && sizeof...(Dims) == extents_t::DynamicOrder)
    constexpr bool resize(Dims... dims) {
        extents_t candidate(extents_);
        candidate.resize(dims...);
        return resize_from_extents_(candidate);
    }

    /// @brief resizes the owned storage to the requested dimensions
    template <internals::md_index... Dims>
        requires(DynamicOrder > 0 && sizeof...(Dims) == extents_t::Order && sizeof...(Dims) != extents_t::DynamicOrder)
    constexpr bool resize(Dims... dims) {
        extents_t candidate(extents_);
        candidate.resize(dims...);
        return resize_from_extents_(candidate);
    }

    /// @brief accesses the coefficient at a physical storage position
    constexpr decltype(auto) operator[](int index) & {
        fdapde_assert(!(index < 0 || index >= md_size()), std::out_of_range, "MdArray position is out of range");
        return storage_[index];
    }
    /// @brief reads the coefficient at a physical storage position through const access
    constexpr decltype(auto) operator[](int index) const& {
        fdapde_assert(!(index < 0 || index >= md_size()), std::out_of_range, "MdArray position is out of range");
        return storage_[index];
    }
    /// @brief rejects indexed access through a temporary parent to prevent dangling references or proxies
    constexpr void operator[](int) && = delete;
    /// @brief rejects indexed access through a temporary parent to prevent dangling references or proxies
    constexpr void operator[](int) const&& = delete;

    /// @brief returns the underlying storage pointer
    constexpr decltype(auto) data() { return storage_.data(); }
    /// @brief returns the underlying storage pointer
    constexpr decltype(auto) data() const { return storage_.data(); }
    /// @brief returns the layout mapping
    constexpr const mapping_t& mapping() const noexcept { return mapping_; }
    /// @brief returns the number of occupied storage words
    constexpr int bitpacks() const
        requires(std::same_as<Scalar, bool>)
    {
        return storage_.bitpacks();
    }
    /// @brief sets the selected coefficients to true
    constexpr void set()
        requires(std::same_as<Scalar, bool>)
    {
        storage_.set();
    }
    /// @brief sets the selected coefficients to false
    constexpr void clear()
        requires(std::same_as<Scalar, bool>)
    {
        storage_.clear();
    }

    /// @brief returns the owned shape's logical coefficient count
    constexpr int md_size() const noexcept { return extents_.size(); }
    /// @brief returns the selected owner axis length after validating the axis
    constexpr int md_extent(int axis) const {
        fdapde_assert(!(axis < 0), std::out_of_range, "MdArray extent axis is out of range");
        return extents_.extent(static_cast<std::size_t>(axis));
    }
    /// @brief reports whether the owner's storage mapping is valid
    constexpr bool md_valid() const noexcept { return extents_.valid(); }
    /// @brief returns the owner mapping's physical stride along the selected axis
    constexpr int md_stride(int axis) const { return mapping_.stride(axis); }
    /// @brief returns the zero origin offset of owned contiguous storage
    constexpr int md_offset() const noexcept { return 0; }
    /// @brief accesses the parent storage at the supplied physical position
    constexpr decltype(auto) md_linear_at(int index) { return storage_[index]; }
    /// @brief accesses the parent storage at the supplied physical position
    constexpr decltype(auto) md_linear_at(int index) const { return storage_[index]; }
   private:
    /// @brief copies logical coefficients after checking the source extents
    template <internals::md_readable Other> constexpr void assign_from_(const Other& other) {
        fdapde_assert(!(!other.valid()), std::invalid_argument, "cannot assign from an invalid MdArray expression");
        std::array<int, Order> dimensions {};
        for (int axis = 0; axis < Order; ++axis) dimensions[axis] = other.extent(axis);
        extents_t candidate(extents_);
        if constexpr (DynamicOrder > 0) {
            candidate.resize(dimensions);
        } else {
            for (int axis = 0; axis < Order; ++axis) {
                fdapde_assert(
                  !(dimensions[axis] != static_extents[axis]), std::invalid_argument,
                  "MdArray assignment dimensions do not match the static shape");
            }
        }
        fdapde_assert(
          !(other.size() != candidate.size()), std::invalid_argument, "MdArray assignment dimensions are inconsistent");

        std::vector<Scalar> temporary;
        temporary.reserve(static_cast<std::size_t>(other.size()));
        for (const auto& value : other) temporary.push_back(static_cast<Scalar>(value));
        if constexpr (DynamicOrder > 0) resize_from_extents_(candidate);
        int i = 0;
        for (auto&& value : *this) value = temporary[static_cast<std::size_t>(i++)];
    }

    /// @brief resizes storage to match the supplied extents
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

// non-owning raw map. The caller owns the backing storage and must keep it alive and stationary for the lifetime of
// the map and every view derived from it; no generation or lifetime tracking is performed.
/// @brief maps external contiguous multidimensional storage
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

    /// @brief creates an empty map with zero-length dynamic axes and no storage pointer
    constexpr MdMap()
        requires(DynamicOrder > 0)
        : extents_(), mapping_(extents_), data_(nullptr), valid_(true) { }
    /// @brief rejects a fixed-size map without an explicit storage pointer
    constexpr MdMap()
        requires(DynamicOrder == 0)
    = delete;

    /// @brief binds external storage to the fixed extents and rejects a null nonempty mapping
    constexpr explicit MdMap(pointer data)
        requires(DynamicOrder == 0)
        : extents_(), mapping_(extents_), data_(data), valid_(true) {
        fdapde_strong_assert(
          !(data == nullptr && extents_.size() != 0), std::invalid_argument,
          "a nonempty MdMap requires nonnull storage");
    }

    /// @brief binds external storage using the supplied dynamic axis lengths
    template <internals::md_index... Dims>
        requires(DynamicOrder > 0 && sizeof...(Dims) == extents_t::DynamicOrder)
    constexpr MdMap(pointer data, Dims... dims) : extents_(dims...), mapping_(extents_), data_(data), valid_(true) {
        fdapde_strong_assert(
          !(data == nullptr && extents_.size() != 0), std::invalid_argument,
          "a nonempty MdMap requires nonnull storage");
    }

    /// @brief binds external storage after validating every supplied axis length
    template <internals::md_index... Dims>
        requires(DynamicOrder > 0 && sizeof...(Dims) == extents_t::Order && sizeof...(Dims) != extents_t::DynamicOrder)
    constexpr MdMap(pointer data, Dims... dims) : extents_(dims...), mapping_(extents_), data_(data), valid_(true) {
        fdapde_strong_assert(
          !(data == nullptr && extents_.size() != 0), std::invalid_argument,
          "a nonempty MdMap requires nonnull storage");
    }

    /// @brief accesses the coefficient at a physical position in mapped external storage
    constexpr decltype(auto) operator[](int index) & {
        fdapde_assert(
          !(!valid_ || index < 0 || index >= md_size()), std::out_of_range, "MdMap position is out of range");
        return data_[index];
    }
    /// @brief reads the coefficient at a physical position in mapped external storage
    constexpr decltype(auto) operator[](int index) const& {
        fdapde_assert(
          !(!valid_ || index < 0 || index >= md_size()), std::out_of_range, "MdMap position is out of range");
        using raw_scalar = std::remove_const_t<Scalar>;
        return static_cast<const raw_scalar&>(data_[index]);
    }
    /// @brief rejects indexed access through a temporary parent to prevent dangling references or proxies
    constexpr void operator[](int) && = delete;
    /// @brief rejects indexed access through a temporary parent to prevent dangling references or proxies
    constexpr void operator[](int) const&& = delete;

    /// @brief returns the underlying storage pointer
    constexpr pointer data() noexcept { return data_; }
    /// @brief returns the underlying storage pointer
    constexpr const std::remove_const_t<Scalar>* data() const noexcept { return data_; }
    /// @brief returns the layout mapping
    constexpr const mapping_t& mapping() const noexcept { return mapping_; }

    /// @brief returns the mapped shape's logical coefficient count
    constexpr int md_size() const noexcept { return valid_ ? extents_.size() : 0; }
    /// @brief returns the selected mapped axis length after validating the axis
    constexpr int md_extent(int axis) const {
        fdapde_assert(!(axis < 0), std::out_of_range, "MdArray extent axis is out of range");
        return valid_ ? extents_.extent(static_cast<std::size_t>(axis)) : 0;
    }
    /// @brief reports whether the external binding and storage mapping are valid
    constexpr bool md_valid() const noexcept { return valid_; }
    /// @brief returns the external mapping's physical stride along the selected axis
    constexpr int md_stride(int axis) const { return mapping_.stride(axis); }
    /// @brief returns the zero origin offset of mapped contiguous storage
    constexpr int md_offset() const noexcept { return 0; }
    /// @brief accesses the parent storage at the supplied physical position
    constexpr decltype(auto) md_linear_at(int index) { return data_[index]; }
    /// @brief accesses the parent storage at the supplied physical position
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

/// @brief returns a view of selected multidimensional ranges
template <typename Parent, typename... Slicers> constexpr auto submdarray(Parent& parent, Slicers... slicers) {
    return parent.block(slicers...);
}
/// @brief rejects borrowing a multidimensional view from a temporary parent
template <typename Parent, typename... Slicers> void submdarray(Parent&&, Slicers...) = delete;

}   // namespace fdapde

#endif   // __FDAPDE_LINALG_MDARRAY_H__
