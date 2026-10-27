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

#ifndef __FDAPDE_LINALG_CACHE_POLICY_H__
#define __FDAPDE_LINALG_CACHE_POLICY_H__

#include "header_check.h"

namespace fdapde {
namespace Cache {

/// @brief selects the algebraic quantities retained by a structured matrix at compile time
template <unsigned Flags_> struct Policy {
    fdapde_static_assert((Flags_ & ~127u) == 0, SPD_CACHE_POLICY_CONTAINS_UNKNOWN_FLAGS);
    static constexpr unsigned Flags = Flags_;
};
using None = Policy<0>;
using Spectral = Policy<1>;
using Log = Policy<2>;
using Sqrt = Policy<4>;
using InverseSqrt = Policy<8>;
using LogDividedDifferences = Policy<16>;
using Cholesky = Policy<32>;
using LogCholesky = Policy<64>;
template <typename... Policies> using Union = Policy<(Policies::Flags | ... | 0u)>;

}   // namespace Cache
namespace internals {
/// @brief supplies storage-free state for a disabled cache
template <int Tag = 0> struct empty_spd_cache { };
}   // namespace internals
}   // namespace fdapde
#endif
