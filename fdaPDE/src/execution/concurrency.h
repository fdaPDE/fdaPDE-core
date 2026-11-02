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

#ifndef __FDAPDE_EXECUTION_CONCURRENCY_H__
#define __FDAPDE_EXECUTION_CONCURRENCY_H__

#include "header_check.h"

// returns the maximum number of logical CPUs on which the current process is allowed to execute concurrently
#ifdef __linux__

#    include <sched.h>

#endif

namespace fdapde {
namespace internals {

/// @brief parses a complete positive worker count within the execution index range
inline std::optional<std::size_t> get_env_concurrency_count(const char* envvar) {
    const char* count = std::getenv(envvar);
    if (count == nullptr) return std::nullopt;

    const std::string_view text(count);
    std::size_t result = 0;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), result);
    if (
      error != std::errc {} || end != text.data() + text.size() || result == 0 ||
      result > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        return std::nullopt;
    }
    return result;
}

}   // namespace internals

/// @brief bounds executor capacity by process affinity and scheduler reservations independently of OpenMP
inline std::size_t available_concurrency() noexcept {
    const auto reservation = [] {
        if (auto count = internals::get_env_concurrency_count("SLURM_CPUS_PER_TASK")) return count;
        if (auto count = internals::get_env_concurrency_count("NCPUS")) return count;
        return internals::get_env_concurrency_count("SLURM_CPUS_ON_NODE");
    }();
#ifdef __linux__
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof(set), &set) == 0) {
        const int ncpus = CPU_COUNT(&set);
        if (ncpus > 0) {
            const auto capacity = static_cast<std::size_t>(ncpus);
            return reservation ? std::min(capacity, *reservation) : capacity;
        }
    }
#endif

    if (reservation) return *reservation;

    // fall back to the reported logical CPU count, or one if it is unknown
    const unsigned int ncpus = std::thread::hardware_concurrency();
    return ncpus > 0 ? ncpus : 1;
}

}   // namespace fdapde

#endif   // __FDAPDE_EXECUTION_CONCURRENCY_H__
