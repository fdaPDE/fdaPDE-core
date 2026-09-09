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

#include <gtest/gtest.h>

void configure_execution();
int execution_configuration();
const void* core_executor_address();
const void* execution_executor_address();
int execute_answer();

// verifies inline configuration, singleton identity and runtime calls across two include orders
TEST(PublicHeaders, CoreExecutionODR) {
    configure_execution();
    // observes a setting written by the other translation unit
    EXPECT_EQ(execution_configuration(), 2);
    // compares singleton addresses obtained through both public include orders
    EXPECT_EQ(core_executor_address(), execution_executor_address());
    // checks the shared runtime executes and joins a task returning a known result
    EXPECT_EQ(execute_answer(), 42);
}
