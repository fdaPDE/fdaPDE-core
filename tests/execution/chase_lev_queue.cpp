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

#include <memory>
#include <set>

using namespace fdapde;

TEST(execution, chase_lev_queue) {
    // chase_lev_queue: empty pops and circular slot reuse preserve LIFO ordering.
    {
        internals::chase_lev_queue<int> queue(1024);   // force repeated circular-slot reuse
        const std::vector<int> batch {0, 1, 2, 3, 4, 5, 6};

        for (int round = 0; round < 128; ++round) {
            for (int i = 0; i < 8; ++i) EXPECT_FALSE(queue.pop_front().has_value());
            ASSERT_TRUE(queue.push_front(batch.begin(), batch.end()));
            for (int i = 6; i >= 0; --i) EXPECT_EQ(queue.pop_front(), i);
        }
    }
    // chase_lev_queue: capacity, empty batches and FIFO stealing.
    {
        const std::vector<int> batch {0, 1, 2, 3, 4, 5, 6};
        internals::chase_lev_queue<int> small(8);
        EXPECT_EQ(small.size(), 0);
        EXPECT_TRUE(small.push_front(batch.begin(), batch.begin()));
        EXPECT_TRUE(small.empty());
        ASSERT_TRUE(small.push_front(batch.begin(), batch.begin() + 1));
        EXPECT_EQ(small.size(), 1);
        EXPECT_FALSE(small.push_front(batch.begin(), batch.end()));
        EXPECT_EQ(small.size(), 1);
        EXPECT_EQ(small.pop_back(), 0);
        EXPECT_TRUE(small.empty());
        ASSERT_TRUE(small.push_front(batch.begin(), batch.end()));
        EXPECT_EQ(small.size(), 7);
        EXPECT_FALSE(small.push_front(batch.begin(), batch.begin() + 1));
        EXPECT_TRUE(small.push_front(batch.begin(), batch.begin()));
        for (int i = 0; i < 7; ++i) EXPECT_EQ(small.pop_back(), i);
        EXPECT_EQ(small.size(), 0);
        EXPECT_FALSE(small.pop_front().has_value());
    }

    // chase_lev_queue: simultaneous owner and thieves deliver each value exactly once.
    {
        const int num_elements = 100000;
        const int num_thieves = 2;
        internals::chase_lev_queue<int> queue(1024);
        std::atomic<bool> start_signal {false};
        std::atomic<bool> owner_done {false};

        // spawn consumer threads
        std::vector<std::vector<int>> thief_results(num_thieves);
        for (auto& v : thief_results) v.reserve(num_elements / num_thieves);
        std::vector<std::thread> thieves;
        for (int i = 0; i < num_thieves; ++i) {
            thieves.emplace_back([&, i]() {
                // spin until owner starts pushing to maximize immediate contention
                while (!start_signal.load(std::memory_order_acquire));
                // start stealing
                while (!owner_done.load(std::memory_order_acquire) || !queue.empty()) {
                    auto val = queue.pop_back();
                    if (val) { thief_results[i].push_back(*val); }
                }
            });
        }
        // producer thread
        std::vector<int> owner_results;
        owner_results.reserve(num_elements);
        start_signal.store(true, std::memory_order_release);
        for (int i = 1; i <= num_elements;) {
            // push
            const int count = std::min(i % 3 == 0 ? 7 : 1, num_elements - i + 1);
            if (count == 1) {
                while (!queue.push_front(i)) { std::this_thread::yield(); }
            } else {
                int values[7];
                for (int j = 0; j < count; ++j) values[j] = i + j;
                while (!queue.push_front(values, values + count)) std::this_thread::yield();
            }
            i += count;
            // simulate real work-stealing patterns: producer occasionally pops their own work
            if (i % 5 == 0) {
                auto val = queue.pop_front();
                if (val) { owner_results.push_back(*val); }
            }
        }
        owner_done.store(true, std::memory_order_release);
        // wait stealers to finish
        for (auto& t : thieves) { t.join(); }

        // check that every element 1, ..., n  found exactly once
        std::vector<int> registry(num_elements + 1, 0);
        for (int val : owner_results) {
            ASSERT_GE(val, 1);
            ASSERT_LE(val, num_elements);
            registry[val]++;
        }
        for (const auto& tr : thief_results) {
            for (int val : tr) {
                ASSERT_GE(val, 1);
                ASSERT_LE(val, num_elements);
                registry[val]++;
            }
        }
        for (int i = 1; i <= num_elements; ++i) { EXPECT_EQ(registry[i], 1); }   // each element found exactly once
    }
}
