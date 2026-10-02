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
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <http://www.gnu.org/licenses/>.

#include "../benchmarks/preallocated_replay.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace {

/// @brief provides a diagonal NNQP whose positive-part optimum is known analytically
fdapde_bench::ReplayInput identity_replay() {
    const double norm = std::sqrt(13.0);
    return {
      3, {{0, 0, 1.0}, {1, 1, 1.0}, {2, 2, 1.0}},
       {2.0,         -1.0,        3.0        },
       {1.0,         1.0,         1.0        },
       {2.0 / norm,  0.0,         3.0 / norm }
    };
}

}   // namespace

// certify the diagonal oracle and verify that profiling starts from the same physical warm direction
TEST(PreallocatedReplay, FreshCertifiedRuns) {
    const auto input = identity_replay();
    auto work = fdapde_bench::prepare_replay(input);
    const auto multiply = [&](const double* x, double* output) {
        std::fill_n(output, work.size, 0.0);
        for (const auto& entry : work.scaled_triplets) output[entry.row] += entry.value * x[entry.col];
    };
    const auto first = fdapde_bench::run_fista(work, multiply);
    // the explicit positive-part diagonal solution must pass the independent KKT and normalization checks
    EXPECT_TRUE(first.converged);
    // the projected iteration must preserve finite nonnegative coordinates
    EXPECT_TRUE(first.nonnegative);
    // stationarity and complementarity use the production scaled tolerance against the analytic optimum
    EXPECT_LE(first.kkt_relative, 1e-8);
    // the physical identity-metric norm must equal one after normalization
    EXPECT_LT(first.norm_error, 1e-12);
    // each normalized coefficient must equal the known positive-part direction
    for (int i = 0; i < work.size; ++i) EXPECT_NEAR(work.weight[i], input.weight[i], 1e-12);
    // only the two positive signal coordinates belong to the final support
    EXPECT_EQ(first.support_count, 2);
    // the ordered support ids zero and two have this independent fixed FNV-1a oracle
    EXPECT_EQ(first.support_hash, 16770648835361190631ULL);
    // a gradient restart can occur at most once during each projected step
    EXPECT_LE(first.restarts, first.iterations);
    // an unprofiled run must not accumulate nested timer measurements
    EXPECT_DOUBLE_EQ(first.spmv_ns + first.raw_spmv_ns + first.timer_overhead_ns, 0.0);

    const auto profiled = fdapde_bench::run_fista(work, multiply, 25000, true);
    // a repeated profiled run must certify the same optimum after resetting persistent buffers
    EXPECT_TRUE(profiled.converged);
    // profiling must preserve the iteration count instead of reusing the previous converged iterate
    EXPECT_EQ(profiled.iterations, first.iterations);
    // per-call timers must not change the gradient restart decisions
    EXPECT_EQ(profiled.restarts, first.restarts);
    // a fresh profiled run must visit the same matrix-vector calls as its unprofiled counterpart
    EXPECT_EQ(profiled.spmv_calls, first.spmv_calls);
    // corrected SpMV time must remain within the measured profiled iteration time
    EXPECT_LE(profiled.spmv_ns, profiled.full_ns);
    // empty-interval calibration must contribute a nonnegative reported overhead
    EXPECT_GE(profiled.timer_overhead_ns, 0.0);
}

// an incorrect backend must not certify itself through the final residual computation
TEST(PreallocatedReplay, IndependentFinalCertificate) {
    auto work = fdapde_bench::prepare_replay(identity_replay());
    const auto incorrect = [&](const double*, double* output) {
        std::copy(work.scaled_c.begin(), work.scaled_c.end(), output);
    };
    const auto result = fdapde_bench::run_fista(work, incorrect);
    // a forged zero backend gradient at the warm seed must fail the scalar metric residual oracle
    EXPECT_FALSE(result.converged);
    // the identity metric exposes the large true warm-seed stationarity error independently of the callback
    EXPECT_GT(result.kkt_relative, 0.1);
}

// preparation rejects incompatible vectors and missing positive diagonal entries before iteration begins
TEST(PreallocatedReplay, RejectInvalidPreparation) {
    auto input = identity_replay();
    input.c.pop_back();
    // a two-entry signal cannot match the declared three-dimensional metric
    EXPECT_THROW(fdapde_bench::prepare_replay(input), std::invalid_argument);
    input = identity_replay();
    std::fill(input.weight.begin(), input.weight.end(), 0.0);
    // a zero reference direction cannot define a finite relative weight error
    EXPECT_THROW(fdapde_bench::prepare_replay(input), std::invalid_argument);
    input = identity_replay();
    input.omega.pop_back();
    // removing the final identity entry leaves a zero diagonal and cannot define the scaled workspace
    EXPECT_THROW(fdapde_bench::prepare_replay(input), std::invalid_argument);
}
