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

#include <stdexcept>

#define FDAPDE_COMMA ,

/// @brief throws the requested exception on failure, including when debug checks are disabled
#define fdapde_strong_assert(condition, exception_type, message)                                                       \
    do {                                                                                                               \
        if (!(condition)) { throw exception_type(message); }                                                           \
    } while (false)

/// @brief checks a precondition in debug mode without evaluating disabled arguments
#ifdef FDAPDE_NO_DEBUG
#    define fdapde_assert(condition, exception_type, message) ((void)0)
#else
#    define fdapde_assert(condition, exception_type, message)                                                          \
        fdapde_strong_assert((condition), exception_type, (message))
#endif

#define fdapde_static_assert(condition, message) static_assert(condition, #message)

/// @brief preserves the legacy constexpr check with the same debug-only runtime behavior
#define fdapde_constexpr_assert(condition)       fdapde_assert((condition), std::logic_error, #condition)

#endif   // __FDAPDE_ASSERT_H__
