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

#ifndef __FDAPDE_EXECUTION_MODULE_H__
#define __FDAPDE_EXECUTION_MODULE_H__

// clang-format off

namespace fdapde {

/// @brief selects parallel execution
struct execution_par_t { };
/// @brief selects sequential execution
struct execution_seq_t { };
static constexpr execution_par_t execution_par;
static constexpr execution_seq_t execution_seq;

}   // namespace fdapde

#include <algorithm>
#include <atomic>
#include <charconv>
#include <cmath>
#include <concepts>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <functional>
#include <future>
#include <iterator>
#include <latch>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <numeric>
#include <optional>
#include <queue>
#include <random>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "src/utility/assert.h"
#include "src/execution/concurrency.h"

// basic threaded runtime
#include "src/execution/task_handle.h"
#include "src/execution/worker.h"
#include "src/execution/threaded_executor.h"

// high-level API
#include "src/execution/parallel_for.h"
#include "src/execution/parallel_for_each.h"
#include "src/execution/parallel_reduce.h"
#include "src/execution/task_graph.h"

// clang-format on

#endif   // __FDAPDE_EXECUTION_MODULE_H__
