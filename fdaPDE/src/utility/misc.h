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

#ifndef __FDAPDE_MISC_H__
#define __FDAPDE_MISC_H__

#include "header_check.h"

namespace fdapde {
namespace internals {

// hash functions
struct pair_hash {
    template <typename T1, typename T2> std::size_t operator()(const std::pair<T1, T2>& pair) const {
        std::size_t hash = 0;
        hash ^= std::hash<T1>()(pair.first ) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        hash ^= std::hash<T2>()(pair.second) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        return hash;
    }
};
template <typename Iterator> struct stl_iterator_hash {
    std::size_t operator()(const Iterator& begin, const Iterator& end) const {
        std::size_t hash = 0;
        for (auto it = begin; it != end; ++it) {
            hash ^= std::hash<typename std::iterator_traits<Iterator>::value_type>()(*it) + 0x9e3779b9 + (hash << 6) +
                    (hash >> 2);
        }
        return hash;
    }
};
template <typename Scalar, int Size> struct std_array_hash {
    std::size_t operator()(const std::array<Scalar, Size>& array) const {
      return stl_iterator_hash<typename std::array<Scalar, Size>::const_iterator>()(array.begin(), array.end());
    };
};

// reverse a std::unordered_map (if the map represents a bijection)
template <typename V, typename W> constexpr std::unordered_map<V, W> map_reverse(const std::unordered_map<W, V>& in) {
    std::unordered_map<V, W> out;
    for (const auto& [key, value] : in) {
        fdapde_constexpr_assert(
          out.find(value) == out.end());   // if this is not passed, in is not a bijection, cannot be inverted
        out[value] = key;
    }
    return out;
}

}   // namespace internals

// sequence generator for {"base_0", "base_1", \ldots, "base_{n-1}"}
constexpr std::vector<std::string> seq(const std::string& base, int n) {
    std::vector<std::string> vec;
    vec.reserve(n);
    for (int i = 0; i < n; ++i) { vec.emplace_back(base + std::to_string(i)); }
    return vec;
}
// sequence generator for {base + 0, base + 1, \ldots, base + n-1}
template <typename T>
    requires(std::is_arithmetic_v<T>)
constexpr std::vector<int> seq(T begin, int n, int by = 1) {
    std::vector<T> vec;
    vec.reserve(n);
    for (int i = 0; i < n; i += by) { vec.emplace_back(begin + i); }
    return vec;
}
  
}   // namespace fdapde

#endif   // __FDAPDE_MISC_H__
