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
#include <memory>
#include <vector>

namespace {

struct LargeCopyableTask {
    std::array<int, 32> padding {};
    std::shared_ptr<std::vector<int>> observations;
    std::unique_ptr<int> local_count;

    explicit LargeCopyableTask(std::shared_ptr<std::vector<int>> observations_) :
        observations(std::move(observations_)), local_count(std::make_unique<int>(0)) { }

    LargeCopyableTask(const LargeCopyableTask& other) :
        padding(other.padding),
        observations(other.observations),
        local_count(std::make_unique<int>(*other.local_count)) { }

    LargeCopyableTask& operator=(const LargeCopyableTask& other) {
        if (this == &other) return *this;
        padding = other.padding;
        observations = other.observations;
        local_count = std::make_unique<int>(*other.local_count);
        return *this;
    }

    LargeCopyableTask(LargeCopyableTask&&) noexcept = default;
    LargeCopyableTask& operator=(LargeCopyableTask&&) noexcept = default;

    void operator()() { observations->push_back(++*local_count); }
};

}   // namespace

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

    ASSERT_EQ(observations->size(), 3u);
    EXPECT_EQ(*observations, (std::vector<int> {1, 1, 1}));
}

TEST(ExecutionOwnership, PooledAndQueuedObjectsReleaseCapturedResources) {
    auto pooled_resource = std::make_shared<int>(0);
    std::weak_ptr<int> pooled_observer = pooled_resource;
    {
        fdapde::internals::pool_allocator<std::shared_ptr<int>> allocator(1);
        auto* object = allocator.allocate(pooled_resource);
        pooled_resource.reset();
        EXPECT_FALSE(pooled_observer.expired());
        allocator.deallocate(object);
        EXPECT_TRUE(pooled_observer.expired());
    }

    auto queued_resource = std::make_shared<int>(0);
    std::weak_ptr<int> queued_observer = queued_resource;
    {
        fdapde::internals::mpsc_queue<std::shared_ptr<int>> queue;
        queue.push(queued_resource);
        queued_resource.reset();
        EXPECT_FALSE(queued_observer.expired());
    }
    EXPECT_TRUE(queued_observer.expired());
}
