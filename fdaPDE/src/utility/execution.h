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

#ifndef __FDAPDE_EXECUTION_H__
#define __FDAPDE_EXECUTION_H__

#include "header_check.h"

namespace fdapde {

// execution policy tags
struct sequential_execution_tag { } seq;   // sequential algorithms tag
struct parallel_execution_tag   { } par;   // parallel algorithms tag

template <typename T, typename U = std::decay_t<T>>
concept is_execution_policy =
  std::is_same_v<U, sequential_execution_tag> || std::is_same_v<U, parallel_execution_tag>;

}   // namespace fdapde

#endif   // __FDAPDE_EXECUTION_H__
