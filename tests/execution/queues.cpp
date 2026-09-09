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
#include <gtest/gtest.h>   // testing framework

#include <set>
using namespace fdapde;

// verifies chase lev queue
TEST(execution, chase_lev_queue) {
    const int num_elements = 100000;
    const int num_thieves = 2;

    internals::chase_lev_queue<int> queue(1024);   // force repeated circular-slot reuse

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
    for (int i = 1; i <= num_elements; ++i) {
        // push
        while (!queue.push_front(i)) { std::this_thread::yield(); }
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
        // rejects corrupt owner values before using them as registry indices
        ASSERT_GE(val, 1);
        // bounds owner values by the largest published task id
        ASSERT_LE(val, num_elements);
        registry[val]++;
    }
    for (const auto& tr : thief_results) {
        for (int val : tr) {
            // rejects corrupt stolen values before using them as registry indices
            ASSERT_GE(val, 1);
            // bounds stolen values by the largest published task id
            ASSERT_LE(val, num_elements);
            registry[val]++;
        }
    }
    for (int i = 1; i <= num_elements; ++i) {
        // checks every published id appears exactly once across owner and thief results
        EXPECT_EQ(registry[i], 1);
    }   // each element found exactly once
}

// verifies mpsc queue
TEST(execution, mpsc_queue) {
    // single-thread correctness
    internals::mpsc_queue<int> queue;
    queue.push(10);
    queue.push(20);
    auto v1 = queue.pop();
    auto v2 = queue.pop();
    auto v3 = queue.pop();   // expected empty
    // checks the first pop contains a value before reading it
    ASSERT_TRUE(v1.has_value());
    // compares the first pop with the first submitted value
    EXPECT_EQ(v1.value(), 10);
    // checks the second pop contains a value before reading it
    ASSERT_TRUE(v2.has_value());
    // compares the second pop with the second submitted value
    EXPECT_EQ(v2.value(), 20);
    // checks the drained queue has no third value
    EXPECT_FALSE(v3.has_value());

    // multi-threaded test
    // P2P: simulate the exchange of messages from multiple threads, each owning a private mpsc queue, emulating a
    // nested parallelism burst, where tasks start to inject tasks to other workers
    {
        const int num_threads = 4;
        const int messages_per_thread = 20000;
        const int total_messages = num_threads * messages_per_thread;

        // one mpsc queue per thread
        std::vector<std::unique_ptr<internals::mpsc_queue<int>>> mailboxes;
        for (int i = 0; i < num_threads; ++i) {
            mailboxes.push_back(std::make_unique<internals::mpsc_queue<int>>(1024));
        }

        std::atomic<int> global_received_count {0};
        std::vector<std::thread> workers;
        std::vector<std::vector<int>> received(num_threads);

        for (int i = 0; i < num_threads; ++i) {
            workers.emplace_back([&, thread_id = i]() {
                // local random engine for picking peers
                std::mt19937 rng(thread_id);
                std::uniform_int_distribution<int> dist(0, num_threads - 1);

                int messages_sent = 0;

                while (messages_sent < messages_per_thread || global_received_count < total_messages) {
                    // producer case: send a message to a random peer's mailbox
                    if (messages_sent < messages_per_thread) {
                        int peer = dist(rng);
                        mailboxes[peer]->push(thread_id * messages_per_thread + messages_sent);
                        messages_sent++;
                    }
                    // consumer case: process messages from local mailbox
                    while (auto msg = mailboxes[thread_id]->pop()) {
                        if (msg) {
                            global_received_count.fetch_add(1, std::memory_order_relaxed);
                            received[thread_id].push_back(*msg);
                        }
                    }
                    if (messages_sent >= messages_per_thread && global_received_count < total_messages) {
                        std::this_thread::yield();
                    }
                }
            });
        }
        // wait for the test to finish
        for (auto& t : workers) { t.join(); }

        // check all messages exchanged
        // compares total deliveries with all peer-to-peer submissions
        EXPECT_EQ(global_received_count.load(), total_messages);
        // check no duplicates
        std::set<int> unique_vals;
        for (int i = 0; i < num_threads; ++i) { unique_vals.insert(received[i].begin(), received[i].end()); }
        // checks the set of delivered ids contains every submitted id exactly once
        EXPECT_EQ(unique_vals.size(), total_messages);
    }

    // multithreaded test
    // test the high contention of a single mpsc queue
    {
        internals::mpsc_queue<int> queue;
        const int num_producers = 3;
        const int items_per_producer = 10000;

        std::vector<std::thread> producers;
        for (int i = 0; i < num_producers; ++i) {
            producers.emplace_back([&, i]() {
                for (int j = 0; j < items_per_producer; ++j) { queue.push(i * items_per_producer + j); }
            });
        }
        // consumers
        std::vector<int> results;
        std::thread consumer([&]() {
            std::size_t total_expected = num_producers * items_per_producer;
            while (results.size() < total_expected) {
                auto val = queue.pop();
                if (val) {
                    results.push_back(*val);
                } else {
                    std::this_thread::yield();   // Wait for producers
                }
            }
        });
        // wait completion
        for (auto& t : producers) { t.join(); }
        consumer.join();

        // compares the single-consumer result count with all producer submissions
        EXPECT_EQ(results.size(), num_producers * items_per_producer);
        std::set<int> unique_vals(results.begin(), results.end());
        // checks high-contention deliveries contain no duplicate ids
        EXPECT_EQ(unique_vals.size(), num_producers * items_per_producer);
    }
}

// verifies the bounded deque reports saturation and keeps owner/thief order at circular wraparound
TEST(execution, chase_lev_capacity_and_wraparound) {
    internals::chase_lev_queue<int> queue(4);
    for (int round = 0; round < 20; ++round) {
        // the first item must fit in an empty deque
        EXPECT_TRUE(queue.push_front(round * 3));
        // the second item must fit without changing the oldest item
        EXPECT_TRUE(queue.emplace_front(round * 3 + 1));
        // the third item fills the usable capacity of a four-slot deque
        EXPECT_TRUE(queue.push_front(round * 3 + 2));
        // saturation must be reported without publishing the extra value
        EXPECT_FALSE(queue.push_front(-1));
        // thieves take the oldest published item
        EXPECT_EQ(queue.pop_back(), round * 3);
        // the owner takes the newest published item
        EXPECT_EQ(queue.pop_front(), round * 3 + 2);
        // the remaining item is claimed once by the thief
        EXPECT_EQ(queue.pop_back(), round * 3 + 1);
        // the final claim must leave the deque empty before slot reuse
        EXPECT_TRUE(queue.empty());
    }
    // an invalid internal buffer size is rejected with debug checks enabled
    EXPECT_THROW(internals::chase_lev_queue<int>(3), std::invalid_argument);
}
