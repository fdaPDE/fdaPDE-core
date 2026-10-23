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
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <future>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

struct large_copyable_task {
    std::array<int, 32> padding {};
    std::shared_ptr<std::vector<int>> observations;
    std::unique_ptr<int> local_count;

    explicit large_copyable_task(std::shared_ptr<std::vector<int>> observations_) :
        observations(std::move(observations_)), local_count(std::make_unique<int>(0)) { }

    large_copyable_task(const large_copyable_task& other) :
        padding(other.padding),
        observations(other.observations),
        local_count(std::make_unique<int>(*other.local_count)) { }

    large_copyable_task& operator=(const large_copyable_task& other) {
        if (this == &other) return *this;
        padding = other.padding;
        observations = other.observations;
        local_count = std::make_unique<int>(*other.local_count);
        return *this;
    }

    large_copyable_task(large_copyable_task&&) noexcept = default;
    large_copyable_task& operator=(large_copyable_task&&) noexcept = default;

    void operator()() { observations->push_back(++*local_count); }
};

struct throwing_graph_task {
    std::shared_ptr<int> resource;
    int* copies;
    int* fail_at;
    std::atomic<int>* runs;

    throwing_graph_task(std::shared_ptr<int> resource_, int& copies_, int& fail_at_, std::atomic<int>& runs_) :
        resource(std::move(resource_)), copies(&copies_), fail_at(&fail_at_), runs(&runs_) { }
    throwing_graph_task(const throwing_graph_task& other) :
        resource(other.resource), copies(other.copies), fail_at(other.fail_at), runs(other.runs) {
        if (++*copies == *fail_at) { throw std::runtime_error("graph task copy failed"); }
    }
    throwing_graph_task(throwing_graph_task&&) = default;
    void operator()() { runs->fetch_add(1, std::memory_order_relaxed); }
};

}   // namespace

TEST(execution, task_graph) {
    // dependency ordering
    {
        std::atomic<int> inputs {0};
        std::atomic<int> observed {-1};
        fdapde::TaskGraph graph;

        auto first = graph.add_node([&] { inputs.fetch_or(1, std::memory_order_release); });
        auto second = graph.add_node([&] { inputs.fetch_or(2, std::memory_order_release); });
        auto sink =
          graph.add_node([&] { observed.store(inputs.load(std::memory_order_acquire), std::memory_order_release); });
        sink.succeeds(first, second);

        EXPECT_EQ(graph.n_nodes(), 3);
        EXPECT_EQ(graph.n_edges(), 2);
        EXPECT_FALSE(graph.has_cycles());
        fdapde::parallel_execute(graph);
        EXPECT_EQ(observed.load(std::memory_order_acquire), 3);
    }
#ifndef FDAPDE_NO_DEBUG
    // Assertion failures run in a fresh process, independently of the active worker pool.
    ::testing::FLAGS_gtest_death_test_style = "threadsafe";
    // cycle rejection
    {
        std::atomic<int> runs {0};
        fdapde::TaskGraph graph;
        auto first = graph.add_node([&] { runs.fetch_add(1, std::memory_order_relaxed); });
        auto second = graph.add_node([&] { runs.fetch_add(1, std::memory_order_relaxed); });
        first.precedes(second);
        second.precedes(first);

        EXPECT_TRUE(graph.has_cycles());
        EXPECT_DEATH(fdapde::parallel_execute(graph), "acyclic");
        EXPECT_EQ(runs.load(std::memory_order_relaxed), 0);
    }
    // cross-graph edge rejection
    {
        fdapde::TaskGraph left;
        fdapde::TaskGraph right;
        auto left_node = left.add_node([] { });
        auto right_node = right.add_node([] { });

        EXPECT_DEATH(left_node.precedes(right_node), "assertion");
        EXPECT_EQ(left.n_edges(), 0);
        EXPECT_EQ(right.n_edges(), 0);
    }
#endif
    // independent copies of heap-backed tasks
    {
        auto observations = std::make_shared<std::vector<int>>();
        fdapde::TaskGraph original;
        original.add_node(large_copyable_task(observations));

        fdapde::TaskGraph constructed(original);
        fdapde::TaskGraph assigned;
        assigned = original;

        fdapde::parallel_execute(original);
        fdapde::parallel_execute(constructed);
        fdapde::parallel_execute(assigned);

        ASSERT_EQ(observations->size(), 3u);
        EXPECT_EQ(*observations, (std::vector<int> {1, 1, 1}));
    }
    // Preparation failures destroy all unpublished tasks and never run a partial graph.
    {
        auto resource = std::make_shared<int>(0);
        int copies = 0;
        int fail_at = 0;
        std::atomic<int> runs {0};
        fdapde::TaskGraph graph;
        auto first = graph.add_node(throwing_graph_task(resource, copies, fail_at, runs));
        auto second = graph.add_node(throwing_graph_task(resource, copies, fail_at, runs));
        auto last = graph.add_node(throwing_graph_task(resource, copies, fail_at, runs));
        last.succeeds(first, second);
        const auto owners = resource.use_count();

        auto preparation_failure = [&](int failure) {
            copies = 0;
            fail_at = failure;
            try {
                fdapde::parallel_execute(graph);
            } catch (const std::runtime_error& error) {
                return std::string(error.what()) == "graph task copy failed" && resource.use_count() == owners &&
                       runs.load(std::memory_order_relaxed) == 0;
            }
            return false;
        };
        for (int failure : {1, 2, 3}) {
            EXPECT_TRUE(preparation_failure(failure));
            EXPECT_TRUE(fdapde::parallel_async([&] { return preparation_failure(failure); }).get());
            fdapde::parallel_join();
            EXPECT_EQ(resource.use_count(), owners);
        }
        fail_at = 0;
        fdapde::parallel_execute(graph);
        fdapde::parallel_join();
        EXPECT_EQ(runs.load(std::memory_order_relaxed), 3);
        EXPECT_EQ(resource.use_count(), owners);
    }
    // Independent graph submissions share the ready queue without retaining stale links.
    {
        std::atomic<int> runs {0};
        std::vector<std::thread> submitters;
        for (int i = 0; i < 4; ++i) {
            submitters.emplace_back([&] {
                fdapde::TaskGraph graph;
                auto source = graph.add_node([&] { ++runs; });
                auto sink = graph.add_node([&] { ++runs; });
                sink.succeeds(source);
                for (int repeat = 0; repeat < 8; ++repeat) { fdapde::parallel_execute(graph); }
            });
        }
        for (auto& submitter : submitters) { submitter.join(); }
        fdapde::parallel_join();
        EXPECT_EQ(runs.load(), 64);
    }
    // dependent tasks release captured resources
    {
        auto source_resource = std::make_shared<int>(0);
        auto sink_resource = std::make_shared<int>(0);
        std::weak_ptr<int> source_observer = source_resource;
        std::weak_ptr<int> sink_observer = sink_resource;
        {
            fdapde::TaskGraph graph;
            auto source = graph.add_node([source_resource] { ++*source_resource; });
            auto sink = graph.add_node([sink_resource] { ++*sink_resource; });
            sink.succeeds(source);
            source_resource.reset();
            sink_resource.reset();
            fdapde::parallel_execute(graph);
        }
        fdapde::parallel_join();

        EXPECT_TRUE(source_observer.expired());
        EXPECT_TRUE(sink_observer.expired());
    }
    // graph waits exclude unrelated tasks
    {
        using namespace std::chrono_literals;

        std::promise<void> blocker_started;
        std::promise<void> release_blocker;
        std::shared_future<void> release = release_blocker.get_future().share();
        fdapde::parallel_execute([&] {
            blocker_started.set_value();
            release.wait();
        });
        ASSERT_EQ(blocker_started.get_future().wait_for(2s), std::future_status::ready);

        std::mutex mutex;
        std::condition_variable condition;
        bool graph_returned = false;
        bool returned_before_release = false;
        std::thread watchdog([&] {
            std::unique_lock<std::mutex> lock(mutex);
            returned_before_release = condition.wait_for(lock, 2s, [&] { return graph_returned; });
            lock.unlock();
            release_blocker.set_value();
        });

        std::atomic<int> graph_runs {0};
        fdapde::TaskGraph graph;
        graph.add_node([&] { graph_runs.fetch_add(1, std::memory_order_relaxed); });
        fdapde::parallel_execute(graph);
        {
            std::lock_guard<std::mutex> lock(mutex);
            graph_returned = true;
        }
        condition.notify_one();
        watchdog.join();
        fdapde::parallel_join();

        EXPECT_TRUE(returned_before_release);
        EXPECT_EQ(graph_runs.load(std::memory_order_relaxed), 1);
    }
    // nested parallel algorithms
    {
        std::vector<int> values(128, 0);
        fdapde::TaskGraph graph;
        graph.add_node([&] {
            fdapde::parallel_for(
              0, static_cast<int>(values.size()), [&](int i) { values[static_cast<std::size_t>(i)] = i + 1; });
        });

        fdapde::parallel_execute(graph);

        for (int i = 0; i < static_cast<int>(values.size()); ++i) {
            EXPECT_EQ(values[static_cast<std::size_t>(i)], i + 1);
        }
    }
}
