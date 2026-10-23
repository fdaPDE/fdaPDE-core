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
#include <string>
#include <string_view>

#define FDAPDE_COMMA ,

namespace fdapde::internals {

// formats a failed condition and appends the optional diagnostic
inline std::string assertion_message(const char* condition, std::string_view message) {
    std::string result = std::string("assertion (") + condition + ") failed";
    if (!message.empty()) result += std::string(": (") + std::string(message) + ")";
    return result;
}

// throws outside the macro body so legacy noexcept callers retain their existing behavior
template <typename Exception> [[noreturn]] inline void assertion_failed(const char* condition, std::string_view message) {
    throw Exception(assertion_message(condition, message));
}

}   // namespace fdapde::internals

#define FDAPDE_ASSERT_IMPL_(condition, condition_text, exception_type, message)                                        \
    do {                                                                                                             \
        if (!(condition)) {                                                                                          \
            ::fdapde::internals::assertion_failed<exception_type>(condition_text, (message));                  \
        }                                                                                                            \
    } while (false)

// throws the requested exception on failure, including when debug checks are disabled
#define fdapde_strong_assert(condition, exception_type, message)                                                      \
    FDAPDE_ASSERT_IMPL_((condition), #condition, exception_type, (message))

// checks a precondition in debug mode without evaluating disabled arguments
#ifdef FDAPDE_NO_DEBUG
#    define FDAPDE_ASSERT_DEBUG_(condition, condition_text, exception_type, message) ((void)0)
#else
#    define FDAPDE_ASSERT_DEBUG_(condition, condition_text, exception_type, message)                                   \
        FDAPDE_ASSERT_IMPL_((condition), condition_text, exception_type, (message))
#endif

#define fdapde_static_assert(condition, message) static_assert(condition, #message)

// accepts the existing one-argument calls while exposing the typed assertion api
#define FDAPDE_ASSERT_SELECT_(_1, _2, _3, NAME, ...) NAME
#define FDAPDE_ASSERT_ONE_(condition)                                                                                 \
    FDAPDE_ASSERT_DEBUG_((condition), #condition, std::runtime_error, "")
#define FDAPDE_ASSERT_THREE_(condition, exception_type, message)                                                      \
    FDAPDE_ASSERT_DEBUG_((condition), #condition, exception_type, (message))
#define fdapde_assert(...)                                                                                           \
    FDAPDE_ASSERT_SELECT_(__VA_ARGS__, FDAPDE_ASSERT_THREE_, FDAPDE_ASSERT_INVALID_, FDAPDE_ASSERT_ONE_)(__VA_ARGS__)

#endif   // __FDAPDE_ASSERT_H__
