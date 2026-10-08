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

#ifndef __FDAPDE_GEOMETRIC_POINT_EXECUTION_H__
#define __FDAPDE_GEOMETRIC_POINT_EXECUTION_H__
#include "header_check.h"
namespace fdapde {
namespace gfe::internals {
/// @brief limits geometric point loops to the policies supplied by the execution module
template <typename Policy>
concept PointExecutionPolicy = std::same_as<Policy, execution_seq_t> || std::same_as<Policy, execution_par_t>;
/// @brief runs independent point work and rethrows the lowest-index failure after parallel workers complete
template <PointExecutionPolicy Policy, typename Body> void for_each_point(std::size_t count, Policy, Body body) {
    fdapde_assert(
      count <= static_cast<std::size_t>(std::numeric_limits<int>::max()), std::length_error,
      "geometric evaluation point count exceeds the execution index range");
    if constexpr (std::same_as<Policy, execution_seq_t>) {
        for (std::size_t i = 0; i < count; ++i) body(i);
    } else {
        if (count == 0) return;
        std::exception_ptr failure;
        std::size_t first_failure = count;
        std::mutex mutex;
        parallel_for(0, static_cast<int>(count), [&](int i) {
            try {
                body(static_cast<std::size_t>(i));
            } catch (...) {
                const std::lock_guard lock(mutex);
                if (static_cast<std::size_t>(i) < first_failure) {
                    first_failure = i;
                    failure = std::current_exception();
                }
            }
        });
        if (failure) std::rethrow_exception(failure);
    }
}
}   // namespace gfe::internals
}   // namespace fdapde
#endif
