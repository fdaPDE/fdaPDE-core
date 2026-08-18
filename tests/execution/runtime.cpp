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
#include <stdexcept>

namespace {

class FourWorkerEnvironment : public ::testing::Environment {
    void SetUp() override {
        fdapde::parallel_set_num_threads(4);
        ASSERT_EQ(fdapde::parallel_get_num_threads(), 4);
    }
    void TearDown() override { fdapde::parallel_join(); }
};

[[maybe_unused]] ::testing::Environment* const execution_environment =
  ::testing::AddGlobalTestEnvironment(new FourWorkerEnvironment);

struct LargeLifetimeTask {
    std::array<int, 32> padding {};
    std::shared_ptr<int> resource;

    void operator()() { ++*resource; }
};

}   // namespace

TEST(ExecutionRuntime, AsyncReturnsValuesAndTransportsExceptions) {
    auto result = fdapde::parallel_async([](int lhs, int rhs) { return lhs + rhs; }, 19, 23);
    EXPECT_EQ(result.get(), 42);

    auto failure = fdapde::parallel_async([]() -> int { throw std::runtime_error("expected"); });
    EXPECT_THROW(static_cast<void>(failure.get()), std::runtime_error);
    fdapde::parallel_join();
}

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
    EXPECT_EQ(result.get(), 1);
    fdapde::parallel_join();

    EXPECT_TRUE(first_observer.expired());
    EXPECT_TRUE(second_observer.expired());
    EXPECT_TRUE(asynchronous_observer.expired());
}

TEST(ExecutionRuntime, WorkerCallsToGlobalJoinAreRejected) {
    auto result = fdapde::parallel_async([] {
        try {
            fdapde::parallel_join();
        } catch (const std::logic_error&) { return true; }
        return false;
    });

    EXPECT_TRUE(result.get());
    fdapde::parallel_join();
}
