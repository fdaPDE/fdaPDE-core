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
using namespace fdapde;

TEST(execution, chase_lev_queue) {
    const int num_items = 100000;
    const int num_stealers = std::thread::hardware_concurrency() - 1;

    internals::chase_lev_queue<int*> queue(1024);

    std::atomic<int> items_collected {0};
    std::atomic<bool> owner_done {false};
    std::vector<int*> results;
    results.reserve(num_items);
    std::mutex m;

    std::vector<int> data(num_items);
    for (int i = 0; i < num_items; ++i) { data[i] = i; }

    // stealer threads
    std::vector<std::thread> stealers;
    for (int i = 0; i < num_stealers; ++i) {
        stealers.emplace_back([&]() {
            while (!owner_done.load(std::memory_order_relaxed) || !queue.empty()) {
                if (auto item = queue.pop_back()) {
                    {
                        std::lock_guard<std::mutex> lock(m);
                        results.push_back(*item);
                    }
                    items_collected.fetch_add(1, std::memory_order_relaxed);
                } else {
                    std::this_thread::yield();
                }
            }
        });
    }

    // owner pushes items and occasionally pops from the front (LIFO)
    for (int i = 0; i < num_items; ++i) {
        queue.push_front(&data[i]);
        if (i % 3 == 0) {
            if (auto item = queue.pop_front()) {
                {
                    std::lock_guard<std::mutex> lock(m);
                    results.push_back(*item);
                }
                items_collected.fetch_add(1, std::memory_order_relaxed);
            }
        }
    }
    owner_done = true;
    // wait to finish
    for (auto& t : stealers) t.join();
    
    std::set<int*> unique_items(results.begin(), results.end());
    EXPECT_EQ(items_collected, num_items);
    EXPECT_EQ(unique_items.size(), num_items);
}

TEST(execution, mpsc_queue) {
    // single-thread correctness
    internals::mpsc_queue<int> queue;
    queue.push(10);
    queue.push(20);
    auto v1 = queue.pop();
    auto v2 = queue.pop();
    auto v3 = queue.pop();   // expected empty
    EXPECT_TRUE (v1.has_value());
    EXPECT_EQ(v1.value(), 10);
    EXPECT_TRUE (v2.has_value());
    EXPECT_EQ(v2.value(), 20);
    EXPECT_FALSE(v3.has_value());

    // multi-threaded test
    // P2P: simulate the exchange of messages from multiple threads, each owning a private mpsc queue, emulating a
    // nested parallelism burst, where tasks start to inject tasks to other workers
    {
        const int num_threads = std::thread::hardware_concurrency();
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
        EXPECT_EQ(global_received_count.load(), total_messages);
	// check no duplicates
        std::set<int> unique_vals;
        for (int i = 0; i < num_threads; ++i) { unique_vals.insert(received[i].begin(), received[i].end()); }
        EXPECT_EQ(unique_vals.size(), total_messages);
    }

    // multithreaded test
    // test the high contention of a single mpsc queue
    {
        internals::mpsc_queue<int> queue;
        const int num_producers = std::thread::hardware_concurrency() - 1;
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

        EXPECT_EQ(results.size(), num_producers * items_per_producer);
        std::set<int> unique_vals(results.begin(), results.end());
        EXPECT_EQ(unique_vals.size(), num_producers * items_per_producer);
    }
}
