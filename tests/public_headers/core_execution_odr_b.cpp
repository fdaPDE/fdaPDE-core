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

// clang-format off
#include <fdaPDE/execution.h>
#include <fdaPDE/core.h>
// clang-format on

/// @brief observes configuration set from another translation unit
int execution_configuration() { return fdapde::parallel_get_num_threads(); }
/// @brief exposes the singleton address through execution-first inclusion
const void* execution_executor_address() { return &fdapde::internals::threaded_executor::instance(); }
/// @brief runs and joins a task through execution-first inclusion
int execute_answer() {
    auto answer = fdapde::parallel_async([] { return 42; });
    int result = answer.get();
    fdapde::parallel_join();
    return result;
}
