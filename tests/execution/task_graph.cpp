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
#include <thread>
#include <vector>

namespace {

/// @brief forces heap storage with a counter that must be copied independently
struct LargeCopyableTask {
    std::array<int, 32> padding {};
    std::shared_ptr<std::vector<int>> observations;
    std::unique_ptr<int> local_count;

    /// @brief initializes a private counter and shared observation sink
    explicit LargeCopyableTask(std::shared_ptr<std::vector<int>> observations_) :
        observations(std::move(observations_)), local_count(std::make_unique<int>(0)) { }

    /// @brief copies the private counter into separately owned storage
    LargeCopyableTask(const LargeCopyableTask& other) :
        padding(other.padding),
        observations(other.observations),
        local_count(std::make_unique<int>(*other.local_count)) { }

    /// @brief replaces the private counter with an independent copy
    LargeCopyableTask& operator=(const LargeCopyableTask& other) {
        if (this == &other) return *this;
        padding = other.padding;
        observations = other.observations;
        local_count = std::make_unique<int>(*other.local_count);
        return *this;
    }

    /// @brief moves the private counter and observation sink
    LargeCopyableTask(LargeCopyableTask&&) noexcept = default;
    /// @brief transfers the private counter and observation sink
    LargeCopyableTask& operator=(LargeCopyableTask&&) noexcept = default;

    /// @brief records the next private counter value in the shared sink
    void operator()() { observations->push_back(++*local_count); }
};

}   // namespace

// verifies dag honors dependencies
TEST(ExecutionTaskGraph, DagHonorsDependencies) {
    std::atomic<int> inputs {0};
    std::atomic<int> observed {-1};
    fdapde::TaskGraph graph;

    auto first = graph.add_node([&] { inputs.fetch_or(1, std::memory_order_release); });
    auto second = graph.add_node([&] { inputs.fetch_or(2, std::memory_order_release); });
    auto sink =
      graph.add_node([&] { observed.store(inputs.load(std::memory_order_acquire), std::memory_order_release); });
    sink.succeeds(first, second);

    // checks all three submitted callables are represented as graph nodes
    EXPECT_EQ(graph.n_nodes(), 3);
    // checks both predecessor edges were recorded
    EXPECT_EQ(graph.n_edges(), 2);
    // checks the constructed fork-join graph is acyclic
    EXPECT_FALSE(graph.has_cycles());
    fdapde::parallel_execute(graph);
    // checks the sink observed both predecessor writes through the dependency ordering
    EXPECT_EQ(observed.load(std::memory_order_acquire), 3);
}

// verifies cycles are rejected before any task runs
TEST(ExecutionTaskGraph, CyclesAreRejectedBeforeAnyTaskRuns) {
    std::atomic<int> runs {0};
    fdapde::TaskGraph graph;
    auto first = graph.add_node([&] { runs.fetch_add(1, std::memory_order_relaxed); });
    auto second = graph.add_node([&] { runs.fetch_add(1, std::memory_order_relaxed); });
    first.precedes(second);
    second.precedes(first);

    // checks the two opposite edges are detected as a cycle
    EXPECT_TRUE(graph.has_cycles());
    // checks cyclic execution fails with the public invalid_argument contract
    EXPECT_THROW(fdapde::parallel_execute(graph), std::invalid_argument);
    // checks rejection happened before any cyclic graph task executed
    EXPECT_EQ(runs.load(std::memory_order_relaxed), 0);
}

// verifies edges cannot cross graph ownership
TEST(ExecutionTaskGraph, EdgesCannotCrossGraphOwnership) {
    fdapde::TaskGraph left;
    fdapde::TaskGraph right;
    auto left_node = left.add_node([] { });
    auto right_node = right.add_node([] { });

    // checks nodes from different graphs cannot be connected
    EXPECT_THROW(left_node.precedes(right_node), std::invalid_argument);
    // checks rejected cross-graph insertion leaves the source graph unchanged
    EXPECT_EQ(left.n_edges(), 0);
    // checks rejected cross-graph insertion leaves the destination graph unchanged
    EXPECT_EQ(right.n_edges(), 0);
}

// verifies heap tasks are deep copied by construction and assignment
TEST(ExecutionTaskGraph, HeapTasksAreDeepCopiedByConstructionAndAssignment) {
    auto observations = std::make_shared<std::vector<int>>();
    fdapde::TaskGraph original;
    original.add_node(LargeCopyableTask(observations));

    fdapde::TaskGraph constructed(original);
    fdapde::TaskGraph assigned;
    assigned = original;

    fdapde::parallel_execute(original);
    fdapde::parallel_execute(constructed);
    fdapde::parallel_execute(assigned);

    // checks each original or copied graph produced exactly one observation
    ASSERT_EQ(observations->size(), 3u);
    // checks each deep-copied heap callable has an independent counter starting at one
    EXPECT_EQ(*observations, (std::vector<int> {1, 1, 1}));
}

// verifies dependency tasks release captured resources
TEST(ExecutionTaskGraph, DependencyTasksReleaseCapturedResources) {
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

    // checks the completed source task no longer owns its captured resource
    EXPECT_TRUE(source_observer.expired());
    // checks the completed successor task no longer owns its captured resource
    EXPECT_TRUE(sink_observer.expired());
}

// verifies execution waits for only the selected graph
TEST(ExecutionTaskGraph, ExecutionWaitsForOnlyTheSelectedGraph) {
    using namespace std::chrono_literals;

    std::promise<void> blocker_started;
    std::promise<void> release_blocker;
    std::shared_future<void> release = release_blocker.get_future().share();
    fdapde::parallel_execute([&] {
        blocker_started.set_value();
        release.wait();
    });
    // waits for unrelated blocking work to start before executing the selected graph
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

    // checks the selected graph returned before unrelated work was released
    EXPECT_TRUE(returned_before_release);
    // checks the selected graph callable executed exactly once
    EXPECT_EQ(graph_runs.load(std::memory_order_relaxed), 1);
}

// verifies graph nodes can run nested parallel algorithms
TEST(ExecutionTaskGraph, GraphNodesCanRunNestedParallelAlgorithms) {
    std::vector<int> values(128, 0);
    fdapde::TaskGraph graph;
    graph.add_node([&] {
        fdapde::parallel_for(
          0, static_cast<int>(values.size()), [&](int i) { values[static_cast<std::size_t>(i)] = i + 1; });
    });

    fdapde::parallel_execute(graph);

    for (int i = 0; i < static_cast<int>(values.size()); ++i) {
        // checks every nested loop write is visible after graph execution returns
        EXPECT_EQ(values[static_cast<std::size_t>(i)], i + 1);
    }
}

// verifies pooled and queued objects release captured resources
TEST(ExecutionOwnership, PooledAndQueuedObjectsReleaseCapturedResources) {
    auto pooled_resource = std::make_shared<int>(0);
    std::weak_ptr<int> pooled_observer = pooled_resource;
    {
        fdapde::internals::pool_allocator<std::shared_ptr<int>> allocator(1);
        auto* object = allocator.allocate(pooled_resource);
        pooled_resource.reset();
        // checks the allocated object owns the resource after external ownership is reset
        EXPECT_FALSE(pooled_observer.expired());
        allocator.deallocate(object);
        // checks recycling destroys the pooled object and releases its resource
        EXPECT_TRUE(pooled_observer.expired());
    }

    auto queued_resource = std::make_shared<int>(0);
    std::weak_ptr<int> queued_observer = queued_resource;
    {
        fdapde::internals::mpsc_queue<std::shared_ptr<int>> queue;
        queue.push(queued_resource);
        queued_resource.reset();
        // checks a pending queue node retains the submitted resource
        EXPECT_FALSE(queued_observer.expired());
    }
    // checks destroying the undrained queue releases its remaining resource
    EXPECT_TRUE(queued_observer.expired());
}

// verifies checked node access rejects both sides of the graph index range
TEST(ExecutionTaskGraph, NodeAccessRejectsInvalidIndices) {
    fdapde::TaskGraph graph;
    graph.add_node([] { });
    // a negative index must be rejected before indexing node storage
    EXPECT_THROW(graph.node(-1), std::out_of_range);
    // the end index must be rejected through the public subscript operator
    EXPECT_THROW(graph[1], std::out_of_range);
}
