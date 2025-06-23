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

#ifndef __FDAPDE_UTILITY_MODULE_H__
#define __FDAPDE_UTILITY_MODULE_H__

// clang-format off

// STL includes
#include <utility>
#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <numeric>
#include <limits>
#include <memory>   // for std::shared_ptr
#include <optional>
#include <random>
#include <type_traits>
#include <typeindex>
#include <string>
#include <numbers>
// common STL containers
#include <array>
#include <queue>
#include <stack>
#include <set>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <tuple>

// utils include
#include "src/utility/assert.h"
#include "src/utility/meta.h"
#include "src/utility/misc.h"

// define basic symbols
namespace fdapde {

[[maybe_unused]] constexpr int Dynamic     = -1;
[[maybe_unused]] constexpr int random_seed = -1;

// algorithm computation policies
[[maybe_unused]] static struct tag_exact     { } Exact;
[[maybe_unused]] static struct tag_not_exact { } NotExact;

}   // namespace fdapde

#include "src/utility/numeric.h"
#include "src/utility/matrix.h"
#include "src/utility/binary.h"
#include "src/utility/mdarray.h"
#include "src/utility/binary_tree.h"
#include "src/utility/type_erasure.h"

// clang-format on

#endif   // __FDAPDE_UTILITY_MODULE_H__
