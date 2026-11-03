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

#include <stdexcept>

TEST(execution, parallel_for) {
    // parallel_for: covers each iteration once.
    {
        const auto check = [](auto dispatcher) {
            using namespace std::chrono_literals;
            for (int begin : {-19, 0, 23}) {
                for (int count : {0, 1, 2, 3, 7, 17, 113}) {
                    std::vector<std::atomic<int>> visits(count);
                    auto result = fdapde::parallel_for(
                      begin, begin + count,
                      [&](int i) {
                          ASSERT_GE(i, begin);
                          ASSERT_LT(i, begin + count);
                          ++visits[i - begin];
                      },
                      dispatcher);
                    static_assert(std::is_same_v<decltype(result), fdapde::future<void>>);
                    EXPECT_TRUE(result.valid());
                    if (count == 0) { EXPECT_EQ(result.wait_for(0ms), std::future_status::ready); }
                    EXPECT_NO_THROW(result.wait());
                    for (const auto& value : visits) { EXPECT_EQ(value.load(), 1); }
                    EXPECT_NO_THROW(result.get());
                    EXPECT_FALSE(result.valid());
                }
            }
            std::atomic<int> calls {0};
            fdapde::parallel_for(0, 9, [&](int) { ++calls; }, dispatcher).get();
            EXPECT_EQ(calls.load(), 9);
        };
        check(fdapde::static_dispatch);
        check(fdapde::adaptive_dispatch);
    }
    // parallel_for: fixed steps.
    {
        const auto check = [](auto dispatcher) {
            const auto check = [dispatcher](int begin, int end, int step, const std::vector<int>& expected) {
                std::vector<std::atomic<int>> visits(end - begin);
                auto result = fdapde::parallel_for(
                  begin, end, step,
                  [&](int i) {
                      ASSERT_GE(i, begin);
                      ASSERT_LT(i, end);
                      ++visits[i - begin];
                  },
                  dispatcher);
                result.get();
                std::vector<int> observed;
                for (int i = begin; i < end; ++i) {
                    const int count = visits[i - begin].load();
                    EXPECT_LE(count, 1);
                    if (count != 0) { observed.push_back(i); }
                }
                EXPECT_EQ(observed, expected);
            };
            check(-5, 17, 3, {-5, -2, 1, 4, 7, 10, 13, 16});
            check(-8, 9, 4, {-8, -4, 0, 4, 8});
            check(1, 9, 2, {1, 3, 5, 7});
            check(4, 4, 3, {});
            check(5, 6, 9, {5});
            check(0, 17, 6, {0, 6, 12});
            check(-2, 3, 1, {-2, -1, 0, 1, 2});
            check(0, 17, std::numeric_limits<int>::max(), {0});
            const int lowest = std::numeric_limits<int>::min();
            const int highest = std::numeric_limits<int>::max();
            check(lowest, lowest + 5, 3, {lowest, lowest + 3});
            check(highest - 5, highest, 3, {highest - 5, highest - 2});

            std::atomic<int> calls {0};
            fdapde::parallel_for(0, 17, 2, [&](int) { ++calls; }, dispatcher).get();
            EXPECT_EQ(calls.load(), 9);
            auto result = fdapde::parallel_for(
              0, 17, 2,
              [](int i) {
                  if (i == 4) { throw std::runtime_error("stepped body failed"); }
              },
              dispatcher);
            EXPECT_NO_THROW(result.wait());
            EXPECT_THROW(result.get(), std::runtime_error);
        };
        check(fdapde::static_dispatch);
        check(fdapde::adaptive_dispatch);
    }
    // parallel_for: nested wait.
    {
        const auto check = [](auto dispatcher) {
            std::vector<std::atomic<int>> visits(7 * 11);
            auto result = fdapde::parallel_for(
              0, 7,
              [&](int outer) {
                  auto child =
                    fdapde::parallel_for(0, 11, [&](int inner) { ++visits[outer * 11 + inner]; }, dispatcher);
                  child.get();
              },
              dispatcher);
            result.get();
            for (const auto& value : visits) { EXPECT_EQ(value.load(), 1); }
            fdapde::parallel_join();
        };
        check(fdapde::static_dispatch);
        check(fdapde::adaptive_dispatch);
    }
    // parallel_for: reports body error through future.
    {
        const auto check = [](auto dispatcher) {
            auto resource = std::make_shared<int>(42);
            std::weak_ptr<int> observer = resource;
            auto result = fdapde::parallel_for(
              -2, 18,
              [held = std::move(resource)](int i) {
                  EXPECT_EQ(*held, 42);
                  if (i == 3) { throw std::runtime_error("body failed"); }
              },
              dispatcher);
            EXPECT_NO_THROW(result.wait());
            EXPECT_TRUE(result.valid());
            fdapde::parallel_join();
            EXPECT_TRUE(observer.expired());
            EXPECT_THROW(result.get(), std::runtime_error);
            EXPECT_FALSE(result.valid());
        };
        check(fdapde::static_dispatch);
        check(fdapde::adaptive_dispatch);
    }
    // parallel_for: owns callable until completion.
    {
        const auto check = [](auto dispatcher) {
            using namespace std::chrono_literals;
            if (fdapde::parallel_get_num_threads() == 1) { return; }
            std::latch release(1);
            std::promise<void> started;
            auto running = started.get_future();
            std::atomic<bool> first {true};
            std::atomic<int> completed {0};
            auto resource = std::make_shared<int>(42);
            std::weak_ptr<int> observer = resource;
            auto result = fdapde::parallel_for(
              0, 19,
              [held = std::move(resource), &release, &started, &first, &completed](int) {
                  if (first.exchange(false)) { started.set_value(); }
                  release.wait();
                  EXPECT_EQ(*held, 42);
                  ++completed;
              },
              dispatcher);
            EXPECT_EQ(running.wait_for(2s), std::future_status::ready);
            EXPECT_EQ(result.wait_for(0ms), std::future_status::timeout);
            EXPECT_FALSE(observer.expired());
            release.count_down();
            result.wait();
            EXPECT_EQ(completed.load(), 19);
            fdapde::parallel_join();
            EXPECT_TRUE(observer.expired());
            result.get();
        };
        check(fdapde::static_dispatch);
        check(fdapde::adaptive_dispatch);
    }
    // parallel_for: discarded future keeps work alive.
    {
        const auto check = [](auto dispatcher) {
            if (fdapde::parallel_get_num_threads() == 1) { return; }
            std::latch release(1);
            std::atomic<int> completed {0};
            auto resource = std::make_shared<int>(42);
            std::weak_ptr<int> observer = resource;
            {
                auto result = fdapde::parallel_for(
                  0, 19,
                  [held = std::move(resource), &release, &completed](int) {
                      release.wait();
                      EXPECT_EQ(*held, 42);
                      ++completed;
                  },
                  dispatcher);
            }
            EXPECT_FALSE(observer.expired());
            release.count_down();
            fdapde::parallel_join();
            EXPECT_EQ(completed.load(), 19);
            EXPECT_TRUE(observer.expired());
        };
        check(fdapde::static_dispatch);
        check(fdapde::adaptive_dispatch);
    }
    // parallel_for: lvalue and lvalue-qualified callable bodies.
    {
        using namespace std::chrono_literals;
        const auto check = [](auto partitioner) {
            for (int count : {0, 1, 257}) {
                std::vector<std::atomic<int>> visits(count);
                struct body_type {
                    std::vector<std::atomic<int>>& visits;
                    void operator()(int i) & { ++visits[i + 11]; }
                };
                body_type body {visits};
                auto result = fdapde::parallel_for(-11, -11 + count, body, partitioner);
                static_assert(std::is_same_v<decltype(result), fdapde::future<void>>);
                if (count == 0) { EXPECT_EQ(result.wait_for(0ms), std::future_status::ready); }
                result.get();
                for (auto& value : visits) {
                    EXPECT_EQ(value.load(), 1);
                    value.store(0);
                }
                fdapde::parallel_for(-11, -11 + count, 3, body_type {visits}, partitioner).get();
                for (int i = 0; i < count; ++i) { EXPECT_EQ(visits[i].load(), i % 3 == 0 ? 1 : 0); }
            }
        };
        check(fdapde::static_dispatch);
        check(fdapde::adaptive_dispatch);
    }
    // parallel_for: reversed ranges are empty with both dispatchers.
    {
        std::atomic<int> calls {0};
        fdapde::parallel_for(5, 4, [&](int) { ++calls; }, fdapde::static_dispatch).get();
        fdapde::parallel_for(5, 4, [&](int) { ++calls; }, fdapde::adaptive_dispatch).get();
        EXPECT_EQ(calls.load(), 0);
    }
}
