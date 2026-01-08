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

// generator view of a mesh
template <typename GeneratorType>
    requires(std::is_invocable_v<GeneratorType, std::ptrdiff_t>)
struct mesh_view : std::ranges::view_base {
    mesh_view(const GeneratorType& generator, int begin, int end) : generator_(generator), begin_(begin), end_(end) { }

    struct iterator {
        using iterator_category = std::random_access_iterator_tag;
        using difference_type = std::ptrdiff_t;
        using value_type = decltype(std::declval<GeneratorType>().operator()(difference_type()));

        iterator(const GeneratorType* generator, int index) : generator_(generator), index_(index) { }

        value_type operator*() { return (*generator_)(index_); }
        value_type operator[](difference_type n) const { return (*generator_)(index_ + n); }

        iterator& operator++() { ++index_; return *this; }
        iterator& operator--() { --index_; return *this; }
        iterator& operator+=(difference_type n) { index_ += n; return *this; }
        iterator& operator-=(difference_type n) { index_ -= n; return *this; }

        friend iterator operator+(iterator it, difference_type n) { return it += n; }
        friend iterator operator-(iterator it, difference_type n) { return it -= n; }

        friend difference_type operator-(iterator lhs, iterator rhs) { return lhs.index_ - rhs.index_; }

        friend bool operator==(iterator lhs, iterator rhs) { return lhs.index_ == rhs.index_; }
        friend auto operator<=>(iterator lhs, iterator rhs) { return lhs.index_ <=> rhs.index_; }
       private:
        const GeneratorType* generator_;
        int index_;
    };

    iterator begin() const { return iterator(&generator_, begin_); }
    iterator end() const { return iterator(&generator_, end_); }
   private:
    GeneratorType generator_;
    int begin_, end_;
};

// filtered generator view of a mesh
template <typename GeneratorType>
    requires(std::is_invocable_v<GeneratorType, std::ptrdiff_t>)
struct filtered_mesh_view : std::ranges::view_base {
    filtered_mesh_view(const GeneratorType& generator, int begin, int end, const Vector<bool, Dynamic>& filter) :
        generator_(generator), begin_(begin), end_(end) {
        fdapde_assert(filter.size() == (end - begin));
	// pre-compute indices to allow O(1) random access
	filter_.reserve(begin - end);
        for (int i = 0; i < (end - begin); ++i) {
            if (filter[i]) { filter_.push_back(begin + i); }
        }
    }

    struct iterator {
        using iterator_category = std::random_access_iterator_tag;
        using difference_type = std::ptrdiff_t;
        using value_type = decltype(std::declval<GeneratorType>().operator()(difference_type()));

        iterator(const GeneratorType* generator, int index, std::vector<int>* filter) :
            generator_(generator), index_(index), filter_(filter) {
        }

        value_type operator*() { return (*generator_)((*filter_)[index_]); }
        value_type operator[](difference_type n) const { return (*generator_)((*filter_)[index_ + n]); }

        iterator& operator++() { ++index_; return *this; }
        iterator& operator--() { --index_; return *this; }
        iterator& operator+=(difference_type n) { index_ += n; return *this; }
        iterator& operator-=(difference_type n) { index_ -= n; return *this; }

        friend iterator operator+(iterator it, difference_type n) { return it += n; }
        friend iterator operator-(iterator it, difference_type n) { return it -= n; }

        friend difference_type operator-(iterator lhs, iterator rhs) { return lhs.index_ - rhs.index_; }

        friend bool operator==(iterator lhs, iterator rhs) { return lhs.index_ == rhs.index_; }
        friend auto operator<=>(iterator lhs, iterator rhs) { return lhs.index_ <=> rhs.index_; }
       private:
        std::vector<int>* filter_;
        const GeneratorType* generator_;
        int index_;
    };

    iterator begin() const { return iterator(&generator_, 0, &filter_); }
    iterator end() const { return iterator(&generator_, filter_.size(), &filter_); }
   private:
    std::vector<int> filter_;
    GeneratorType generator_;
    int begin_, end_;
};

}   // namespace internals
}   // namespace fdapde

#endif   // __FDAPDE_GEOMETRY_UTILITY_H__
