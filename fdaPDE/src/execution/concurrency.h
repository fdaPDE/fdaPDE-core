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

// returns the maximum number of logical CPUs on which the current process is allowed to execute concurrently.
#ifdef __linux__

#    include <sched.h>
#    include <unistd.h>

#endif

namespace fdapde {
namespace internals {

inline std::optional<std::size_t> get_env_concurrency_count(const char* envvar) {
    const char* count = std::getenv(envvar);
    if (count) {
        try {
            return static_cast<std::size_t>(std::stoul(count));
        } catch (...) { return std::nullopt; }
    }
    return std::nullopt;
}

}   // namespace internals

inline std::size_t available_concurrency() noexcept {
    // first, check if the process is scheduler-managed

    // detect the number of reserved cpus for processes running under:
    // open Portable Batch System (openPBS)
    if (auto PBS_NCPUS   = internals::get_env_concurrency_count("NCPUS")) { return *PBS_NCPUS; }
    // SLURM
    if (auto SLURM_NCPUS = internals::get_env_concurrency_count("SLURM_CPUS_ON_NODE")) { return *SLURM_NCPUS; }
    // check if OpenMP has been configured
    if (auto OMP_NCPUS   = internals::get_env_concurrency_count("OMP_NUM_THREADS")) {
        if (*OMP_NCPUS > 1) return *OMP_NCPUS;
    }

    // second, check OS-specific settings
#ifdef __linux__
    // on Linux, job schedulers limit the number of CPUs per process by setting the process CPU affinity mask
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof(set), &set) == 0) {
        int ncpus = CPU_COUNT(&set);
        if (ncpus > 0 && ncpus < (int)std::thread::hardware_concurrency()) { return static_cast<std::size_t>(ncpus); }
    }
#endif
    // if all of the previous failed, fallbacks to number of physical CPU cores
    int ncpus = std::thread::hardware_concurrency();
    return ncpus > 0 ? ncpus : 1;
}

}   // namespace fdapde

#endif   // __FDAPDE_EXECUTION_CONCURRENCY_H__
