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
    // prepared tasks can be discarded, submitted in batches and continued without a graph
    {
        fdapde::internals::threaded_executor_impl executor(1);
        auto resource = std::make_shared<int>(0);
        std::weak_ptr<int> observer = resource;
        auto discarded = executor.allocate_task(0, [resource] { ++*resource; });
        resource.reset();
        EXPECT_FALSE(observer.expired());
        executor.deallocate_task(discarded);
        EXPECT_TRUE(observer.expired());

        std::atomic<int> runs {0};
        auto continuation = executor.allocate_task(0, [&] { ++runs; });
        auto first = executor.allocate_task(0, [&] {
            ++runs;
            executor.submit_task(continuation);
        });
        auto second = executor.allocate_task(0, [&] { ++runs; });
        executor.submit_task({first, second});
        executor.join();
        EXPECT_EQ(runs.load(), 3);
    }
    // mailbox tasks remain accessible while their destination worker is blocked
    {
        using namespace std::chrono_literals;
        fdapde::internals::threaded_executor_impl executor(2);
        std::promise<int> started;
        std::promise<void> release_blocker, prepared_done, ordinary_done;
        auto started_worker = started.get_future();
        auto release = release_blocker.get_future().share();
        executor.execute([&] {
            started.set_value(fdapde::this_thread_id());
            release.wait();
        });
        const auto blocked = started_worker.wait_for(2s);
        const int destination = blocked == std::future_status::ready ? started_worker.get() : 0;
        auto prepared = executor.allocate_task(destination, [&] { prepared_done.set_value(); });
        executor.submit_task(prepared);
        const auto prepared_status = prepared_done.get_future().wait_for(2s);

        if (destination == 0) {
            executor.execute([] { });
        }
        executor.execute([&] { ordinary_done.set_value(); });
        const auto ordinary_status = ordinary_done.get_future().wait_for(2s);
        release_blocker.set_value();
        executor.join();
        EXPECT_EQ(blocked, std::future_status::ready);
        EXPECT_EQ(prepared_status, std::future_status::ready);
        EXPECT_EQ(ordinary_status, std::future_status::ready);
    }
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
    // an external group wait finishes while unrelated work remains blocked
    {
        using namespace std::chrono_literals;
        fdapde::internals::threaded_executor_impl executor(2);
        std::promise<void> release_blocker, child_started, release_child;
        auto blocker_release = release_blocker.get_future().share();
        auto child_release = release_child.get_future().share();
        auto blocker = executor.async([&] { blocker_release.wait(); });
        auto group = std::async(std::launch::async, [&] {
            executor.execute(fdapde::internals::task_parallel_for(), 0, 1, 1, [&](int) {
                child_started.set_value();
                child_release.wait();
            });
            return 42;
        });
        const auto started = child_started.get_future().wait_for(2s);
        release_child.set_value();
        const auto ready = group.wait_for(2s);
        release_blocker.set_value();
        executor.join();
        ASSERT_EQ(started, std::future_status::ready);
        ASSERT_EQ(ready, std::future_status::ready);
        EXPECT_EQ(group.get(), 42);
        blocker.get();
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
    // completion accounting across allocation owners and concurrent submissions
    {
        fdapde::internals::threaded_executor_impl executor(4);
        // continuations cross allocation owners; join must wait for the entire chain
        for (int repeat = 0; repeat < 32; ++repeat) {
            constexpr int count = 256;
            std::vector<fdapde::internals::worker::packaged_task> tasks(count);
            std::vector<int> output(count, 0);
            for (int i = 0; i < count; ++i) {
                tasks[i] = executor.allocate_task(3 - i % 4, [&, i] {
                    output[i] = i + 1;
                    if (i + 1 < count) executor.submit_task(tasks[i + 1]);
                });
            }
            executor.submit_task(tasks.front());
            executor.join();
            for (int i = 0; i < count; ++i) EXPECT_EQ(output[i], i + 1);
        }
        // join includes child tasks submitted by concurrent publishers
        for (int repeat = 0; repeat < 16; ++repeat) {
            std::atomic<int> completed {0};
            std::vector<std::thread> publishers;
            for (int producer = 0; producer < 4; ++producer) {
                publishers.emplace_back([&] {
                    for (int i = 0; i < 128; ++i) {
                        executor.execute([&] {
                            ++completed;
                            executor.execute([&] { ++completed; });
                        });
                    }
                });
            }
            for (auto& publisher : publishers) publisher.join();
            executor.join();
            EXPECT_EQ(completed.load(), 1024);
        }
    }
}
