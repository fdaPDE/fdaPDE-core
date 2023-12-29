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

#ifndef __FDAPDE_ASSERT_H__
#define __FDAPDE_ASSERT_H__

namespace fdapde {
  
// throw an exception if condition is not met
#define fdapde_assert(condition)                                                                                       \
    if (!(condition)) throw std::runtime_error("Condition " #condition " failed");

// static assert with message
#define fdapde_static_assert(condition, message) static_assert(condition, #message);

}   // namespace fdapde

#endif   // __FDAPDE_ASSERT_H__
