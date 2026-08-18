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

#include <fdaPDE/execution.h>
#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <future>

namespace {

class OneWorkerEnvironment : public ::testing::Environment {
    void SetUp() override {
        fdapde::parallel_set_num_threads(1);
        ASSERT_EQ(fdapde::parallel_get_num_threads(), 1);
    }
};

[[maybe_unused]] ::testing::Environment* const execution_environment =
  ::testing::AddGlobalTestEnvironment(new OneWorkerEnvironment);

}   // namespace

TEST(ExecutionQueueCapacity, GatedBurstBeyondLocalQueueCapacityLosesNoTasks) {
    using namespace std::chrono_literals;

    std::promise<void> blocker_started;
    std::promise<void> release_blocker;
    std::shared_future<void> release = release_blocker.get_future().share();
    fdapde::parallel_execute([&] {
        blocker_started.set_value();
        release.wait();
    });
    ASSERT_EQ(blocker_started.get_future().wait_for(2s), std::future_status::ready);

    constexpr int task_count = 9'000;
    std::atomic<int> completed {0};
    for (int i = 0; i < task_count; ++i) {
        fdapde::parallel_execute([&] { completed.fetch_add(1, std::memory_order_relaxed); });
    }

    release_blocker.set_value();
    fdapde::parallel_join();
    EXPECT_EQ(completed.load(std::memory_order_relaxed), task_count);
}
