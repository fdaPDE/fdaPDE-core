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

#include <fdaPDE/src/utility/assert.h>
#include <gtest/gtest.h>

#include <string>

// verifies exception type, exact message and single evaluation on failed debug checks
TEST(Assertions, DebugFailurePreservesExceptionAndMessage) {
    int conditions = 0;
    int messages = 0;
    try {
        fdapde_assert(++conditions == 0, std::invalid_argument, (++messages, "invalid input"));
        // reaching this line means the false condition did not throw
        FAIL() << "debug assertion did not throw";
    } catch (const std::invalid_argument& error) {
        // compares the exception payload with the supplied message
        EXPECT_STREQ(error.what(), "invalid input");
    }
    // counts one evaluation of the condition
    EXPECT_EQ(conditions, 1);
    // counts one construction of the failure message
    EXPECT_EQ(messages, 1);
}

// verifies lazy messages on success and safe use within an unbraced if/else
TEST(Assertions, SuccessfulChecksAreSingleStatementsWithLazyMessages) {
    int conditions = 0;
    int messages = 0;
    if (true)
        fdapde_assert(++conditions == 1, std::logic_error, (++messages, "unused"));
    else
        ++messages;
    if (true)
        fdapde_strong_assert(++conditions == 2, std::logic_error, (++messages, "unused"));
    else
        ++messages;
    // checks that each successful primitive evaluated its condition once
    EXPECT_EQ(conditions, 2);
    // detects eager message evaluation or incorrect else binding
    EXPECT_EQ(messages, 0);
}

// verifies permanent checks preserve the user-selected exception payload
TEST(Assertions, StrongFailurePreservesExceptionAndMessage) {
    try {
        fdapde_strong_assert(false, std::out_of_range, std::string("outside domain"));
        // a permanent false check must not reach the next statement
        FAIL() << "strong assertion did not throw";
    } catch (const std::out_of_range& error) {
        // compares the exact message carried by the requested exception type
        EXPECT_STREQ(error.what(), "outside domain");
    }
}

namespace {
/// @brief exercises the retained constexpr helper in constant and runtime evaluation
constexpr int checked_value(int value) {
    fdapde_constexpr_assert(value > 0);
    return value;
}
}   // namespace

// verifies the legacy helper remains usable by stable constexpr callers
TEST(Assertions, ConstexprHelperChecksRuntimeFailures) {
    // constant evaluation must accept a satisfied precondition
    static_assert(checked_value(3) == 3);
    // runtime evaluation must throw the helper's documented logic_error
    EXPECT_THROW(checked_value(0), std::logic_error);
}
