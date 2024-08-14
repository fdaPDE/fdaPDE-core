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

// trait to detect if a mesh is a manifold
template <int M, int N> struct is_manifold {
    static constexpr bool value = (M != N);
};

// macro for the definition of mesh type detection
#define DEFINE_MESH_TYPE_DETECTION_TRAIT(M_, N_, name)                                                                 \
    template <int M, int N> struct is_##name {                                                                         \
        static constexpr bool value =                                                                                  \
          std::conditional<(M == M_ && N == N_), std::true_type, std::false_type>::type::value;                        \
    };

DEFINE_MESH_TYPE_DETECTION_TRAIT(1, 2, network);   // is_network<M, N>
DEFINE_MESH_TYPE_DETECTION_TRAIT(2, 2, 2d);        // is_2d<M, N>
DEFINE_MESH_TYPE_DETECTION_TRAIT(2, 3, surface);   // is_surface<M, N>
DEFINE_MESH_TYPE_DETECTION_TRAIT(3, 3, 3d);        // is_3d<M, N>

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

template <typename Iterator, typename ValueType> class index_based_iterator {
   protected:
    using This = index_based_iterator<Iterator, ValueType>;
    int index_;
    int begin_, end_;
    ValueType val_;
   public:
    using value_type        = ValueType;
    using pointer           = const ValueType*;
    using reference         = const ValueType&;
    using size_type         = std::size_t;
    using difference_type   = std::ptrdiff_t;
    using iterator_category = std::bidirectional_iterator_tag;

    index_based_iterator() = default;
    index_based_iterator(int index, int begin, int end) : index_(index), begin_(begin), end_(end) { }
    reference operator*() const { return val_; }
    pointer operator->() const { return &val_; }

    Iterator operator++(int) {
        Iterator tmp(index_, static_cast<Iterator*>(this));
        ++(static_cast<Iterator&>(*this));
        return tmp;
    }
    Iterator operator--(int) {
        Iterator tmp(index_, static_cast<Iterator*>(this));
        --(static_cast<Iterator&>(*this));
        return tmp;
    }
    Iterator& operator++() {
        index_++;
        if (index_ < end_) static_cast<Iterator&>(*this)(index_);
        return static_cast<Iterator&>(*this);
    }
    Iterator& operator--() {
        --index_;
        if (index_ >= begin_) static_cast<Iterator&>(*this)(index_);
        return static_cast<Iterator&>(*this);
    }
    Iterator& operator+(int i) {
        index_ = index_ + i;
        return static_cast<Iterator&>(*this)(index_);
    }
    Iterator& operator-(int i) {
        index_ = index_ - i;
        return static_cast<Iterator&>(*this)(index_);
    }
    friend bool operator!=(const This& lhs, const This& rhs) { return lhs.index_ != rhs.index_; }
    friend bool operator==(const This& lhs, const This& rhs) { return lhs.index_ == rhs.index_; }
    int index() const { return index_; }
};

}   // namespace fdapde

#endif   // __MESH_UTILS_H__
