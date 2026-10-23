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

#include <cstdio>
#include <exception>
#include <source_location>
#include <string_view>

#define FDAPDE_COMMA ,

namespace fdapde::internals {

// prints the failed condition, function and location before terminating
[[noreturn]] inline void assertion_failed(
  const char* condition, std::source_location location, std::string_view message = {}) noexcept {
    std::fprintf(
      stderr, "[%s:%u] assertion (%s) failed in %s\n", location.file_name(),
      static_cast<unsigned int>(location.line()), condition, location.function_name());
    if (!message.empty()) {
        std::fwrite(message.data(), 1, message.size(), stderr);
        std::fputc('\n', stderr);
    }
    std::fflush(stderr);
    std::terminate();
}

}   // namespace fdapde::internals

#define FDAPDE_ASSERT_IMPL_(condition, condition_text, ...)                                                            \
    do {                                                                                                               \
        if (!(condition)) {                                                                                            \
            ::fdapde::internals::assertion_failed(                                                                     \
              condition_text, std::source_location::current() __VA_OPT__(, __VA_ARGS__));                              \
        }                                                                                                              \
    } while (false)

// checks a precondition even when debug assertions are disabled
#define fdapde_strong_assert(condition, ...)                                                                          \
    FDAPDE_ASSERT_IMPL_((condition), #condition __VA_OPT__(, __VA_ARGS__))

// checks a precondition in debug mode without evaluating disabled arguments
#ifdef FDAPDE_NO_DEBUG
#    define fdapde_assert(condition, ...) ((void)0)
#else
#    define fdapde_assert(condition, ...)                                                                              \
        FDAPDE_ASSERT_IMPL_((condition), #condition __VA_OPT__(, __VA_ARGS__))
#endif

#define fdapde_static_assert(condition, message) static_assert(condition, #message)

#endif   // __FDAPDE_ASSERT_H__
