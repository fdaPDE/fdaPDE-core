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

struct execution_par_t { };
struct execution_seq_t { };
static constexpr execution_par_t execution_par;
static constexpr execution_seq_t execution_seq;

}   // namespace fdapde

#include <algorithm>
#include <unordered_map>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cmath>
#include <compare>
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
#include <ranges>
#include <stdexcept>
#include <string_view>
#include <system_error>
#include <thread>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "src/execution/concurrency.h"

// basic threaded runtime
#include "src/execution/pool_allocator.h"
#include "src/execution/chase_lev_queue.h"
#include "src/execution/mpsc_queue.h"
#include "src/execution/task_handle.h"
#include "src/execution/execution_context.h"
#include "src/execution/threaded_executor.h"

// high-level API
#include "src/execution/task_batch.h"
#include "src/execution/parallel_for.h"
#include "src/execution/parallel_for_each.h"
#include "src/execution/parallel_reduce.h"
#include "src/execution/task_graph.h"

// clang-format on

#endif   // __FDAPDE_EXECUTION_MODULE_H__
