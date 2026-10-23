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
#include <chrono>
#include <memory>
#include <stdexcept>

namespace {

struct large_lifetime_task {
    std::array<int, 32> padding {};
    std::shared_ptr<int> resource;

    void operator()() { ++*resource; }
};

}   // namespace

TEST(execution, threaded_executor) {
    // future values and exceptions
    {
        auto result = fdapde::parallel_async([](int lhs, int rhs) { return lhs + rhs; }, 19, 23);
        EXPECT_EQ(result.get(), 42);

        auto failure = fdapde::parallel_async([]() -> int { throw std::runtime_error("expected"); });
        EXPECT_THROW(static_cast<void>(failure.get()), std::runtime_error);
        fdapde::parallel_join();
    }
    // completed tasks release captured resources
    {
        auto first = std::make_shared<int>(0);
        auto second = std::make_shared<int>(0);
        auto asynchronous = std::make_shared<int>(0);
        std::weak_ptr<int> first_observer = first;
        std::weak_ptr<int> second_observer = second;
        std::weak_ptr<int> asynchronous_observer = asynchronous;

        fdapde::parallel_execute(large_lifetime_task {{}, first});
        fdapde::parallel_execute(large_lifetime_task {{}, second});
        auto result = fdapde::parallel_async([asynchronous] { return ++*asynchronous; });
        first.reset();
        second.reset();
        asynchronous.reset();
        EXPECT_EQ(result.get(), 1);
        fdapde::parallel_join();

        EXPECT_TRUE(first_observer.expired());
        EXPECT_TRUE(second_observer.expired());
        EXPECT_TRUE(asynchronous_observer.expired());
    }
    // A worker join returns immediately without waiting for its own task.
    {
        auto result = fdapde::parallel_async([] {
            fdapde::parallel_join();
            return true;
        });

        EXPECT_TRUE(result.get());
        fdapde::parallel_join();
    }
    // Sleeping workers wake for new submissions while another worker remains blocked.
    {
        using namespace std::chrono_literals;
        fdapde::internals::threaded_executor_impl executor(4);
        std::promise<void> blocker_started;
        std::promise<void> release_blocker;
        auto release = release_blocker.get_future().share();
        executor.execute([&] {
            blocker_started.set_value();
            release.wait();
        });
        const auto started = blocker_started.get_future().wait_for(2s);
        if (started != std::future_status::ready) {
            release_blocker.set_value();
            executor.join();
        }
        ASSERT_EQ(started, std::future_status::ready);

        auto result = executor.async([] { return 42; });
        const auto ready = result.wait_for(2s);
        release_blocker.set_value();
        executor.join();
        ASSERT_EQ(ready, std::future_status::ready);
        EXPECT_EQ(result.get(), 42);

        // Repeatedly cross the empty-queue boundary and exercise dependency publication.
        for (int repeat = 0; repeat < 32; ++repeat) {
            auto value = executor.async([repeat] { return repeat; });
            EXPECT_EQ(value.get(), repeat);
            executor.join();
            std::atomic<int> completed {0};
            fdapde::TaskGraph graph;
            auto first = graph.add_node([&] { ++completed; });
            auto second = graph.add_node([&] { ++completed; });
            auto last = graph.add_node([&] { EXPECT_EQ(completed.load(), 2); });
            last.succeeds(first, second);
            executor.execute(fdapde::internals::task_graph(), graph);
            executor.join();
        }
    }
    // A nested join wakes when its last running child completes, even with no queued tasks.
    {
        using namespace std::chrono_literals;
        fdapde::internals::threaded_executor_impl executor(2);
        std::promise<void> child_started;
        std::promise<void> release_child;
        auto release = release_child.get_future().share();
        auto parent = executor.async([&] {
            executor.execute(fdapde::internals::task_parallel_for(), 0, 1, 1, [&](int) {
                child_started.set_value();
                release.wait();
            });
            return 42;
        });
        const auto started = child_started.get_future().wait_for(2s);
        release_child.set_value();
        ASSERT_EQ(started, std::future_status::ready);
        ASSERT_EQ(parent.wait_for(2s), std::future_status::ready);
        EXPECT_EQ(parent.get(), 42);
        executor.join();
    }
    // mailbox overflow preserves all submitted tasks
    {
        using namespace std::chrono_literals;
        fdapde::internals::threaded_executor_impl executor(1);

        std::promise<void> blocker_started;
        std::promise<void> release_blocker;
        std::shared_future<void> release = release_blocker.get_future().share();
        executor.execute([&] {
            blocker_started.set_value();
            release.wait();
        });
        ASSERT_EQ(blocker_started.get_future().wait_for(2s), std::future_status::ready);

        constexpr int task_count = 9'000;
        std::atomic<int> completed {0};
        for (int i = 0; i < task_count; ++i) {
            executor.execute([&] { completed.fetch_add(1, std::memory_order_relaxed); });
        }

        release_blocker.set_value();
        executor.join();
        EXPECT_EQ(completed.load(std::memory_order_relaxed), task_count);
    }
}
