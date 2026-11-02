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

#include <array>
#include <cmath>
#include <future>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {

/// @brief configures the shared suite before the runtime singleton is initialized
class FourWorkerEnvironment : public ::testing::Environment {
    /// @brief selects four workers for runtime, graph and algorithm tests
    void SetUp() override {
        fdapde::parallel_set_num_threads(4);
        // checks the shared execution fixture uses four workers
        ASSERT_EQ(fdapde::parallel_get_num_threads(), 4);
    }
    /// @brief waits for outstanding work before the test process exits
    void TearDown() override { fdapde::parallel_join(); }
};

[[maybe_unused]] ::testing::Environment* const execution_environment =
  ::testing::AddGlobalTestEnvironment(new FourWorkerEnvironment);

/// @brief forces heap storage while retaining an observable shared resource
struct LargeLifetimeTask {
    std::array<int, 32> padding {};
    std::shared_ptr<int> resource;

    /// @brief increments the captured resource when the task executes
    void operator()() { ++*resource; }
};

}   // namespace

// verifies async returns values and transports exceptions
TEST(ExecutionRuntime, AsyncReturnsValuesAndTransportsExceptions) {
    auto result = fdapde::parallel_async([](int lhs, int rhs) { return lhs + rhs; }, 19, 23);
    // compares the asynchronous addition result with its known sum
    EXPECT_EQ(result.get(), 42);

    auto failure = fdapde::parallel_async([]() -> int { throw std::runtime_error("expected"); });
    // checks that the future rethrows the callable exception on the caller thread
    EXPECT_THROW(static_cast<void>(failure.get()), std::runtime_error);
    fdapde::parallel_join();
}

// verifies completed heap tasks release captured resources
TEST(ExecutionRuntime, CompletedHeapTasksReleaseCapturedResources) {
    auto first = std::make_shared<int>(0);
    auto second = std::make_shared<int>(0);
    auto asynchronous = std::make_shared<int>(0);
    std::weak_ptr<int> first_observer = first;
    std::weak_ptr<int> second_observer = second;
    std::weak_ptr<int> asynchronous_observer = asynchronous;

    fdapde::parallel_execute(LargeLifetimeTask {{}, first});
    fdapde::parallel_execute(LargeLifetimeTask {{}, second});
    auto result = fdapde::parallel_async([asynchronous] { return ++*asynchronous; });
    first.reset();
    second.reset();
    asynchronous.reset();
    // checks that the asynchronous callable ran once before releasing its captures
    EXPECT_EQ(result.get(), 1);
    fdapde::parallel_join();

    // checks the first heap task released its final shared ownership after join
    EXPECT_TRUE(first_observer.expired());
    // checks the second heap task released its final shared ownership after join
    EXPECT_TRUE(second_observer.expired());
    // checks the asynchronous task released its captured resource after join
    EXPECT_TRUE(asynchronous_observer.expired());
}

// verifies worker calls to global join are rejected
TEST(ExecutionRuntime, WorkerCallsToGlobalJoinAreRejected) {
    auto result = fdapde::parallel_async([] {
        try {
            fdapde::parallel_join();
        } catch (const std::logic_error&) { return true; }
        return false;
    });

    // checks the worker caught the forbidden global-join exception
    EXPECT_TRUE(result.get());
    fdapde::parallel_join();
}

// verifies discarding an unpublished task releases its capture without running it
TEST(ExecutionRuntime, DiscardReleasesUnpublishedTaskOwnership) {
    fdapde::internals::threaded_executor_impl executor(1);
    auto resource = std::make_shared<int>(0);
    std::weak_ptr<int> observer = resource;
    auto task = executor.allocate_task(0, LargeLifetimeTask {{}, resource});
    resource.reset();
    // the reserved callable retains its resource until explicit cancellation
    EXPECT_FALSE(observer.expired());
    task.discard();
    // destruction of the discarded callable releases the final shared owner
    EXPECT_TRUE(observer.expired());
    // discard clears the handle so cancellation cannot release it twice
    EXPECT_FALSE(task);
    task.discard();
    executor.join();
}

// verifies two-way parallelism uses one background worker and the attached caller
TEST(ExecutionRuntime, CallerCollaboratesWithinConfiguredParallelism) {
    fdapde::internals::threaded_executor_impl executor(2);
    const auto caller = std::this_thread::get_id();
    std::atomic<bool> caller_ran {false};
    std::atomic<bool> worker_ran {false};
    std::atomic<bool> invalid_id {false};
    executor.execute(fdapde::internals::task_parallel_for(), 0, 64, 1, [&](int) {
        const int id = fdapde::this_thread_id();
        if (id < 0 || id >= 2) invalid_id.store(true);
        if (std::this_thread::get_id() == caller) caller_ran.store(true);
    });
    auto asynchronous = executor.async([&] { worker_ran.store(fdapde::this_thread_id() == 0); });
    asynchronous.get();
    executor.join();
    // the final synchronous chunk always runs directly on the submitting caller
    EXPECT_TRUE(caller_ran.load());
    // asynchronous progress uses the only background context in a two-way pool
    EXPECT_TRUE(worker_ran.load());
    // every loop callback belongs to the worker or the reserved caller context
    EXPECT_FALSE(invalid_id.load());
    // attachment ownership is restored when the synchronous batch returns
    EXPECT_EQ(fdapde::this_thread_id(), fdapde::internals::main_thread_id);
}

// verifies child submissions from multiple external producers are included in global completion
TEST(ExecutionRuntime, ConcurrentPublishersRetainChildAccounting) {
    fdapde::internals::threaded_executor_impl executor(4);
    std::atomic<int> completed {0};
    std::vector<std::thread> producers;
    for (int producer = 0; producer < 4; ++producer) {
        producers.emplace_back([&] {
            for (int i = 0; i < 128; ++i) {
                executor.execute([&] {
                    ++completed;
                    executor.execute([&] { ++completed; });
                });
            }
        });
    }
    for (auto& producer : producers) producer.join();
    executor.join();
    // four producers submit 128 parents and one child per parent before global completion
    EXPECT_EQ(completed.load(), 1024);
}
