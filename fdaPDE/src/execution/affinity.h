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

#ifndef __FDAPDE_EXECUTION_AFFINITY_H__
#define __FDAPDE_EXECUTION_AFFINITY_H__

#include "../utility/assert.h"
#include "header_check.h"

#ifdef __linux__

#    include <sched.h>

namespace fdapde {
namespace internals {

/// @brief owns a dynamically sized Linux CPU mask
struct affinity_mask {
    /// @brief allocates and clears a mask covering the requested CPU range
    explicit affinity_mask(int capacity) : bytes(CPU_ALLOC_SIZE(capacity)), data(CPU_ALLOC(capacity)) {
        if (!data) throw std::bad_alloc();
        CPU_ZERO_S(bytes, data);
    }
    /// @brief releases the allocated Linux CPU mask
    ~affinity_mask() { CPU_FREE(data); }
    /// @brief prevents copying CPU mask ownership
    affinity_mask(const affinity_mask&) = delete;
    /// @brief prevents assignment of CPU mask ownership
    affinity_mask& operator=(const affinity_mask&) = delete;

    const std::size_t bytes;
    cpu_set_t* const data;
};

// returns the calling thread's logical CPU IDs
inline std::vector<int> allowed_cpus() {
    // grow the mask until it covers the kernel's CPU range
    for (int capacity = 128; capacity <= (1 << 20); capacity *= 2) {
        affinity_mask current(capacity);
        if (sched_getaffinity(0, current.bytes, current.data) == 0) {
            std::vector<int> cpus;
            for (int cpu = 0; cpu < capacity; ++cpu) {
                if (CPU_ISSET_S(cpu, current.bytes, current.data)) { cpus.push_back(cpu); }
            }
            return cpus;
        }
        const int error = errno;
        // only an undersized mask warrants another attempt
        if (error != EINVAL) { throw std::runtime_error("affinity error"); }
    }
    throw std::runtime_error("affinity error");
}

// binds only the calling thread; pass its previous allowed_cpus() result to restore the mask
inline void bind(const std::vector<int>& cpus) {
    fdapde_strong_assert(!cpus.empty(), std::invalid_argument, "invalid CPU specification");
    fdapde_strong_assert(
      std::all_of(cpus.begin(), cpus.end(), [](int cpu) { return cpu >= 0 && cpu < std::numeric_limits<int>::max(); }),
      std::invalid_argument, "invalid CPU specification");
    // cPU IDs can be sparse, so size the mask by the highest ID
    affinity_mask target(*std::max_element(cpus.begin(), cpus.end()) + 1);
    for (int cpu : cpus) { CPU_SET_S(cpu, target.bytes, target.data); }
    if (sched_setaffinity(0, target.bytes, target.data) != 0) { throw std::runtime_error("affinity error"); }
}

}   // namespace internals
}   // namespace fdapde

#elif defined(__FreeBSD__)

#    include <pthread.h>
#    include <pthread_np.h>
#    include <sys/cpuset.h>
#    include <sys/types.h>

namespace fdapde {
namespace internals {

// returns the calling thread's logical CPU IDs
inline std::vector<int> allowed_cpus() {
    cpuset_t current;
    CPU_ZERO(&current);
    const int error = pthread_getaffinity_np(pthread_self(), sizeof(current), &current);
    if (error != 0) { throw std::runtime_error("affinity error"); }
    std::vector<int> cpus;
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (CPU_ISSET(cpu, &current)) { cpus.push_back(cpu); }
    }
    return cpus;
}

// binds only the calling thread; pass its previous allowed_cpus() result to restore the mask
inline void bind(const std::vector<int>& cpus) {
    fdapde_strong_assert(!cpus.empty(), std::invalid_argument, "invalid CPUs specification");
    fdapde_strong_assert(
      std::all_of(cpus.begin(), cpus.end(), [](int cpu) { return cpu >= 0 && cpu < std::numeric_limits<int>::max(); }),
      std::invalid_argument, "invalid CPUs specification");
    // the fixed mask cannot represent IDs beyond CPU_SETSIZE
    fdapde_strong_assert(
      *std::max_element(cpus.begin(), cpus.end()) < CPU_SETSIZE, std::invalid_argument, "invalid CPUs specification");
    cpuset_t target;
    CPU_ZERO(&target);
    for (int cpu : cpus) { CPU_SET(cpu, &target); }
    const int error = pthread_setaffinity_np(pthread_self(), sizeof(target), &target);
    if (error != 0) { throw std::runtime_error("affinity error"); }
}

}   // namespace internals
}   // namespace fdapde

#elif defined(_WIN32)

// requires Windows 7 or later (_WIN32_WINNT >= 0x0601)
#    ifndef NOMINMAX
#        define NOMINMAX
#        define __FDAPDE_AFFINITY_UNDEF_NOMINMAX
#    endif
#    include <windows.h>
#    ifdef __FDAPDE_AFFINITY_UNDEF_NOMINMAX
#        undef NOMINMAX
#        undef __FDAPDE_AFFINITY_UNDEF_NOMINMAX
#    endif

namespace fdapde {
namespace internals {

// returns the calling thread's logical CPU IDs (group * 64 + index)
inline std::vector<int> allowed_cpus() {
    GROUP_AFFINITY current {};
    if (!GetThreadGroupAffinity(GetCurrentThread(), &current)) { throw std::runtime_error("affinity error"); }
    std::vector<int> cpus;
    for (int cpu = 0; cpu < std::numeric_limits<KAFFINITY>::digits; ++cpu) {
        // include the group so IDs remain distinct across processor groups
        if ((current.Mask & (KAFFINITY {1} << cpu)) != 0) cpus.push_back(static_cast<int>(current.Group) * 64 + cpu);
    }
    return cpus;
}

// binds only the calling thread; pass its previous allowed_cpus() result to restore the mask
inline void bind(const std::vector<int>& cpus) {
    fdapde_strong_assert(!cpus.empty(), std::invalid_argument, "invalid CPUs specification");
    fdapde_strong_assert(
      std::all_of(cpus.begin(), cpus.end(), [](int cpu) { return cpu >= 0 && cpu < std::numeric_limits<int>::max(); }),
      std::invalid_argument, "invalid CPUs specification");
    // a thread affinity mask can address only one processor group
    const int group = cpus.front() / 64;
    fdapde_strong_assert(
      group <= (std::numeric_limits<WORD>::max)(), std::invalid_argument, "invalid CPUs specification");
    GROUP_AFFINITY target {};
    target.Group = static_cast<WORD>(group);
    for (int cpu : cpus) {
        fdapde_strong_assert(cpu / 64 == group, std::invalid_argument, "affinity error");
        // convert the encoded ID back to a bit within the group
        const int index = cpu % 64;
        fdapde_strong_assert(index < std::numeric_limits<KAFFINITY>::digits, std::invalid_argument, "affinity error");
        target.Mask |= KAFFINITY {1} << index;
    }
    if (!SetThreadGroupAffinity(GetCurrentThread(), &target, nullptr)) { throw std::runtime_error("affinity error"); }
}

}   // namespace internals
}   // namespace fdapde

#else

// fallbacks have disabled affinity control

namespace fdapde {
namespace internals {

// returns the calling thread's logical CPU IDs
/// @brief returns logical CPU identifiers allowed for the calling thread
inline std::vector<int> allowed_cpus() { throw std::runtime_error("affinity not supported by the OS"); }
// binds only the calling thread; pass its previous allowed_cpus() result to restore the mask
/// @brief binds the calling thread to the supplied logical CPU identifiers
inline void bind([[maybe_unused]] const std::vector<int>& cpus) {
    throw std::runtime_error("affinity not supported by the OS");
}

}   // namespace internals
}   // namespace fdapde

#endif

namespace fdapde {
namespace internals {

inline int cpu_affinity_enabled = 0;   // affinity disabled by default

}   // namespace internals

// enable or disable CPU affinity. affinity must be set before executor initalization
/// @brief sets best-effort background affinity before runtime initialization
inline void parallel_set_affinity(bool cpu_affinity);

}   // namespace fdapde

#endif   // __FDAPDE_EXECUTION_AFFINITY_H__
