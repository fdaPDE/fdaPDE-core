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

// this single focused exception to debug-only testing proves argument erasure
#include <fdaPDE/src/utility/assert.h>
#include <gtest/gtest.h>

// verifies disabled arguments are neither evaluated nor required to be valid C++ expressions
TEST(AssertionsDisabled, ErasesDebugArgumentsAndKeepsStrongChecks) {
    int evaluations = 0;
    fdapde_assert(++evaluations == 0, std::logic_error, (++evaluations, "unused"));
    fdapde_assert(undefined_condition(), UndefinedException, undefined_message());
    fdapde_constexpr_assert(undefined_constexpr_condition());
    // an unchanged counter proves that both disabled arguments were skipped
    EXPECT_EQ(evaluations, 0);
    try {
        fdapde_strong_assert(++evaluations == 0, std::invalid_argument, "always active");
        // a disabled debug mode must not suppress the permanent assertion
        FAIL() << "strong assertion did not throw";
    } catch (const std::invalid_argument& error) {
        // checks the exact permanent failure message with debug disabled
        EXPECT_STREQ(error.what(), "always active");
    }
    // the permanent condition still runs exactly once
    EXPECT_EQ(evaluations, 1);
}
