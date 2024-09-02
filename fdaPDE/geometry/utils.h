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

#ifndef __MESH_UTILS_H__
#define __MESH_UTILS_H__

#include "../utils/constexpr.h"
#include "../utils/symbols.h"

namespace fdapde {

// flgas
[[maybe_unused]] static constexpr int cache_cells = 0x0001;

// special marker values
[[maybe_unused]] static constexpr int BoundaryAll = -1;
[[maybe_unused]] static constexpr int TriangulationAll = -2;
[[maybe_unused]] static constexpr int Unmarked = -3;

namespace internals {

// sorts a range of points in clockwise order around their geometrical center
template <typename T> struct clockwise_order {
   private:
    T c_ {};
   public:
    clockwise_order(const T& c) : c_(c) { }
    bool operator()(const T& a, const T& b) {
        if (a[0] - c_[0] >= 0 && b[0] - c_[0] < 0) return true;
        if (b[0] - c_[0] >= 0 && a[0] - c_[0] < 0) return false;
        if (a[0] - c_[0] == 0 && b[0] - c_[0] == 0) {
            return (a[1] - c_[1] >= 0 || b[1] - c_[1] >= 0) ? a[1] > b[1] : b[1] > a[1];
        }
        // check sign of the cross product of vectors CA and CB
        double aXb_sign = (a[0] - c_[0]) * (b[1] - c_[1]) - (b[0] - c_[0]) * (a[1] - c_[1]);
        if (aXb_sign < 0) return true;
        if (aXb_sign > 0) return false;
        // points a and b are on the same line from the center, sort wrt distance from the center
        return (a - c_).squaredNorm() > (b - c_).squaredNorm();
    }
};

template <typename IteratorType, typename ValueType> class index_iterator {
   public:
    using value_type = ValueType;
    using pointer = std::conditional_t<
      std::is_pointer_v<ValueType>, std::decay_t<ValueType>, std::add_pointer_t<std::decay_t<ValueType>>>;
    using reference = std::conditional_t<
      std::is_pointer_v<ValueType>, std::add_lvalue_reference_t<std::remove_pointer_t<std::decay_t<ValueType>>>,
      std::add_lvalue_reference_t<std::decay_t<ValueType>>>;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using iterator_category = std::bidirectional_iterator_tag;

    index_iterator() = default;
    index_iterator(int index, int begin, int end) : index_(index), begin_(begin), end_(end) { }
    reference operator*() { if constexpr(std::is_pointer_v<ValueType>) { return *val_; } else { return val_; } }
    const reference operator*() const {
        if constexpr (std::is_pointer_v<ValueType>) {
            return *val_;
        } else {
            return val_;
        }
    }
    pointer operator->() { if constexpr(std::is_pointer_v<ValueType>) { return val_; } else { return &val_; } }
    const pointer operator->() const {
        if constexpr (std::is_pointer_v<ValueType>) {
            return val_;
        } else {
            return &val_;
        }
    }
    IteratorType operator++(int) {
        IteratorType tmp(index_, static_cast<IteratorType*>(this));
        ++(derived());
        return tmp;
    }
    IteratorType operator--(int) {
        IteratorType tmp(index_, static_cast<IteratorType*>(this));
        --(derived());
        return tmp;
    }
    IteratorType& operator++() {
        index_++;
        if (index_ < end_) derived().operator()(index_);
        return derived();
    }
    IteratorType& operator--() {
        --index_;
        if (index_ >= begin_) derived().operator()(index_);
        return derived();
    }
    friend bool
    operator!=(const index_iterator<IteratorType, ValueType>& lhs, const index_iterator<IteratorType, ValueType>& rhs) {
        return lhs.index_ != rhs.index_;
    }
    friend bool
    operator==(const index_iterator<IteratorType, ValueType>& lhs, const index_iterator<IteratorType, ValueType>& rhs) {
        return lhs.index_ == rhs.index_;
    }
    int index() const { return index_; }
    IteratorType& derived() { return static_cast<IteratorType&>(*this); }
   protected:
    int index_;
    int begin_, end_;
    value_type val_;
};

template <typename IteratorType, typename ValueType>
class filtering_iterator : public index_iterator<IteratorType, ValueType> {
    using Base = index_iterator<IteratorType, ValueType>;
   protected:
    using Base::index_;
    BinaryVector<fdapde::Dynamic> filter_;
   public:
    filtering_iterator(int index, int begin, int end) : Base(index, begin, end) { }
    filtering_iterator(int index, int begin, int end, const BinaryVector<fdapde::Dynamic>& filter) :
        Base(index, begin, end), filter_(filter) { /* initialization is responsibility of IteratorType */ }
    IteratorType& operator++() {
        index_++;
        for (; index_ < Base::end_ && !filter_[index_]; ++index_);
        if (index_ == Base::end_) return Base::derived();
        return Base::derived().operator()(index_);
    }
    IteratorType& operator--() {
        index_--;
        for (; index_ >= Base::begin_ && !filter_[index_]; --index_);
        if (index_ == -1) return Base::derived();
        return Base::derived().operator()(index_);
    }
};

}   // namespace internals
}   // namespace fdapde

#endif   // __MESH_UTILS_H__
