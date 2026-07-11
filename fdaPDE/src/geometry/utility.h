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

#ifndef __FDAPDE_GEOMETRY_UTILITY_H__
#define __FDAPDE_GEOMETRY_UTILITY_H__

#include "header_check.h"

namespace fdapde {

// flags
[[maybe_unused]] static constexpr int cache_cells = 0x0001;

// special marker values
[[maybe_unused]] static constexpr int BoundaryAll = -1;
[[maybe_unused]] static constexpr int TriangulationAll = -2;
[[maybe_unused]] static constexpr int Unmarked = -3;

/** @brief One simple outer polygonal ring and zero or more disjoint simple hole rings. */
struct PlanarDomain {
    Matrix<double, Dynamic, Dynamic> outer;
    std::vector<Matrix<double, Dynamic, Dynamic>> holes;
};

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

template <typename IteratorType, typename StoredType> class index_iterator {
    using element_type =
      std::conditional_t<std::is_pointer_v<StoredType>, std::remove_pointer_t<StoredType>, StoredType>;
   public:
    using value_type = std::remove_cv_t<element_type>;
    using pointer = const value_type*;
    using reference = const value_type&;
    using difference_type = std::ptrdiff_t;
    using iterator_category = std::bidirectional_iterator_tag;
    using iterator_concept = std::bidirectional_iterator_tag;

    index_iterator() = default;
    index_iterator(int index, int begin, int end) : index_(index), begin_(begin), end_(end) { }
    reference operator*() const {
        if constexpr (std::is_pointer_v<StoredType>)
            return *val_;
        else
            return val_;
    }
    pointer operator->() const {
        if constexpr (std::is_pointer_v<StoredType>)
            return val_;
        else
            return std::addressof(val_);
    }
    IteratorType& operator++() {
        ++index_;
        if (index_ < end_) derived().operator()(index_);
        return derived();
    }
    IteratorType operator++(int) {
        IteratorType copy = derived();
        ++derived();
        return copy;
    }
    IteratorType& operator--() {
        --index_;
        if (index_ >= begin_) derived().operator()(index_);
        return derived();
    }
    IteratorType operator--(int) {
        IteratorType copy = derived();
        --derived();
        return copy;
    }
    friend bool operator==(const index_iterator& lhs, const index_iterator& rhs) { return lhs.index_ == rhs.index_; }
    friend bool operator!=(const index_iterator& lhs, const index_iterator& rhs) { return !(lhs == rhs); }
    int index() const { return index_; }
   protected:
    IteratorType& derived() { return static_cast<IteratorType&>(*this); }
    const IteratorType& derived() const { return static_cast<const IteratorType&>(*this); }

    int index_ = 0;
    int begin_ = 0;
    int end_ = 0;
    StoredType val_ {};
};

template <typename IteratorType, typename ValueType>
class filtering_iterator : public index_iterator<IteratorType, ValueType> {
    using Base = index_iterator<IteratorType, ValueType>;
   protected:
    using Base::index_;
    Vector<bool, Dynamic> filter_;
   public:
    using Base::operator++;
    using Base::operator--;

    filtering_iterator() = default;
    filtering_iterator(int index, int begin, int end) : Base(index, begin, end) { }
    filtering_iterator(int index, int begin, int end, const Vector<bool, Dynamic>& filter) :
        Base(index, begin, end), filter_(filter) { /* initialization is responsibility of IteratorType */ }
    IteratorType& operator++() {
        ++index_;
        for (; index_ < Base::end_ && filter_.size() != 0 && !filter_[index_]; ++index_);
        if (index_ == Base::end_) return Base::derived();
        return Base::derived().operator()(index_);
    }
    IteratorType& operator--() {
        --index_;
        for (; index_ >= Base::begin_ && filter_.size() != 0 && !filter_[index_]; --index_);
        if (index_ < Base::begin_) return Base::derived();
        return Base::derived().operator()(index_);
    }
};

}   // namespace internals
}   // namespace fdapde

#endif   // __FDAPDE_GEOMETRY_UTILITY_H__
