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

#include <forward_list>
#include <stdexcept>

TEST(execution, parallel_for_each) {
    // parallel_for_each: empty, singleton, forward, sentinel and const ranges.
    {
        using namespace std::chrono_literals;
        const auto check = [](auto partitioner) {
            for (int count : {0, 1, 257}) {
                std::vector<int> values(count);
                auto body = [](int& value) { ++value; };
                auto result = fdapde::parallel_for_each(values, body, partitioner);
                static_assert(std::is_same_v<decltype(result), fdapde::future<void>>);
                if (count == 0) { EXPECT_EQ(result.wait_for(0ms), std::future_status::ready); }
                result.get();
                for (int value : values) { EXPECT_EQ(value, 1); }
            }
            std::forward_list<int> values {0, 1, 2, 3, 4};
            fdapde::parallel_for_each(values, [](int& value) { ++value; }, partitioner).get();
            EXPECT_EQ((std::vector<int>(values.begin(), values.end())), (std::vector<int> {1, 2, 3, 4, 5}));
            auto view = std::ranges::subrange(std::counted_iterator(values.begin(), 5), std::default_sentinel);
            fdapde::parallel_for_each(view, [](int& value) { ++value; }, partitioner).get();
            EXPECT_EQ((std::vector<int>(values.begin(), values.end())), (std::vector<int> {2, 3, 4, 5, 6}));
            const int input[] {1, 2, 3, 4, 5};
            std::atomic<int> sum {0};
            fdapde::parallel_for_each(input, [&](const int& value) { sum += value; }, partitioner).get();
            EXPECT_EQ(sum.load(), 15);
        };
        check(fdapde::static_dispatch);
        check(fdapde::adaptive_dispatch);
    }
    // parallel_for_each: reports errors and releases captures.
    {
        const auto check = [](auto dispatcher) {
            std::vector<int> values(97);
            auto resource = std::make_shared<int>(42);
            std::weak_ptr<int> observer = resource;
            auto result = fdapde::parallel_for_each(
              values,
              [held = std::move(resource)](int&) {
                  EXPECT_EQ(*held, 42);
                  throw std::runtime_error("body failed");
              },
              dispatcher);
            EXPECT_NO_THROW(result.wait());
            fdapde::parallel_join();
            EXPECT_TRUE(observer.expired());
            EXPECT_THROW(result.get(), std::runtime_error);
            fdapde::parallel_for_each(values, [](int& value) { ++value; }, dispatcher).get();
            for (int value : values) { EXPECT_EQ(value, 1); }
        };
        check(fdapde::static_dispatch);
        check(fdapde::adaptive_dispatch);
    }
    // parallel_for_each: owns callable until async completion.
    {
        const auto check = [](auto dispatcher) {
            using namespace std::chrono_literals;
            if (fdapde::parallel_get_num_threads() == 1) { return; }
            std::vector<int> values(97);
            std::latch release(1);
            std::promise<void> started;
            auto running = started.get_future();
            std::atomic<bool> first {true};
            auto resource = std::make_shared<int>(42);
            std::weak_ptr<int> observer = resource;
            auto result = fdapde::parallel_for_each(
              values,
              [held = std::move(resource), &release, &started, &first](int& value) {
                  if (first.exchange(false)) { started.set_value(); }
                  release.wait();
                  value = *held;
              },
              dispatcher);
            EXPECT_EQ(running.wait_for(2s), std::future_status::ready);
            EXPECT_EQ(result.wait_for(0ms), std::future_status::timeout);
            EXPECT_FALSE(observer.expired());
            release.count_down();
            result.get();
            fdapde::parallel_join();
            EXPECT_TRUE(observer.expired());
            for (int value : values) { EXPECT_EQ(value, 42); }
        };
        check(fdapde::static_dispatch);
        check(fdapde::adaptive_dispatch);
    }
    // parallel_for_each: nested waits and future error propagation.
    {
        std::vector<std::vector<int>> rows(7, std::vector<int>(11));
        fdapde::parallel_for(
          0, int(rows.size()),
          [&](int row) {
              fdapde::parallel_for_each(rows[row], [](int& value) { ++value; }, fdapde::adaptive_dispatch).get();
          },
          fdapde::adaptive_dispatch)
          .get();
        for (const auto& row : rows) {
            for (int value : row) { EXPECT_EQ(value, 1); }
        }
        auto result = fdapde::parallel_for_each(
          rows, [](auto&) { throw std::runtime_error("body failed"); }, fdapde::adaptive_dispatch);
        EXPECT_NO_THROW(result.wait());
        EXPECT_THROW(result.get(), std::runtime_error);
        EXPECT_FALSE(result.valid());
    }
}
