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

#include <atomic>
#include <chrono>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

// Fail iterator preparation on the submitting thread; worker-side traversal remains non-throwing.
struct iterator_failure_state {
    int* begin;
    int copy_failure_offset;
    int assignment_failure_offset;
    std::thread::id submitting_thread = std::this_thread::get_id();
    bool armed = true;
    std::atomic<int> reads {0};
};

struct throwing_iterator {
    using iterator_category = std::forward_iterator_tag;
    using value_type = int;
    using difference_type = std::ptrdiff_t;
    using reference = int&;
    using pointer = int*;

    int* position = nullptr;
    iterator_failure_state* state = nullptr;

    throwing_iterator() = default;
    throwing_iterator(int* position_, iterator_failure_state* state_) : position(position_), state(state_) { }
    throwing_iterator(const throwing_iterator& other) : position(other.position), state(other.state) {
        fail_if_requested_(state ? state->copy_failure_offset : -1);
    }
    throwing_iterator(throwing_iterator&&) = default;
    throwing_iterator& operator=(throwing_iterator&&) = default;
    throwing_iterator& operator=(const throwing_iterator& other) {
        other.fail_if_requested_(other.state ? other.state->assignment_failure_offset : -1);
        position = other.position;
        state = other.state;
        return *this;
    }
    int& operator*() const {
        state->reads.fetch_add(1, std::memory_order_relaxed);
        return *position;
    }
    throwing_iterator& operator++() {
        ++position;
        return *this;
    }
    throwing_iterator operator++(int) {
        throwing_iterator previous(*this);
        ++*this;
        return previous;
    }
    friend bool operator==(const throwing_iterator& lhs, const throwing_iterator& rhs) {
        return lhs.position == rhs.position;
    }
   private:
    void fail_if_requested_(int offset) const {
        if (
          state && std::this_thread::get_id() == state->submitting_thread && state->armed && offset >= 0 &&
          position == state->begin + offset) {
            state->armed = false;
            throw std::runtime_error("iterator preparation failed");
        }
    }
};

struct throwing_range {
    using iterator = throwing_iterator;
    using const_iterator = throwing_iterator;
    using reference = int&;
    int values[3] {1, 2, 3};
    iterator_failure_state state;

    throwing_range(int copy_offset, int assignment_offset) : state {values, copy_offset, assignment_offset} { }
    iterator begin() { return iterator(values, &state); }
    iterator end() { return iterator(values + 3, &state); }
};

}   // namespace

TEST(execution, parallel_for) {
    // partitioning, custom steps, and empty ranges
    {
        std::vector<int> values(64, 0);

        fdapde::parallel_for(0, 64, 7, [&](int i) { values[static_cast<std::size_t>(i)] += 1; });
        fdapde::parallel_for(0, 64, [&](int i) { values[static_cast<std::size_t>(i)] += 2; });
        fdapde::parallel_for(
          0, 64, 3, [&](int i) { values[static_cast<std::size_t>(i)] += 4; }, [](int i) { return i + 2; });
        fdapde::parallel_for(4, 4, [&](int) { FAIL() << "empty ranges must not execute"; });
        fdapde::parallel_for(5, 4, [&](int) { FAIL() << "reversed ranges must not execute"; });

        for (int i = 0; i < 64; ++i) { EXPECT_EQ(values[static_cast<std::size_t>(i)], i % 2 == 0 ? 7 : 3); }
    }
    // Submission failures finish published chunks before propagating, including nested calls.
    {
        struct throwing_step {
            int* copies;
            int* advances;
            int copy_failure;
            int advance_failure;
            std::thread::id submitting_thread;

            throwing_step(int& copies_, int& advances_, int copy_failure_, int advance_failure_) :
                copies(&copies_),
                advances(&advances_),
                copy_failure(copy_failure_),
                advance_failure(advance_failure_),
                submitting_thread(std::this_thread::get_id()) { }
            throwing_step(const throwing_step& other) :
                copies(other.copies),
                advances(other.advances),
                copy_failure(other.copy_failure),
                advance_failure(other.advance_failure),
                submitting_thread(other.submitting_thread) {
                if (++*copies == copy_failure) { throw std::runtime_error("step copy failed"); }
            }
            throwing_step(throwing_step&&) = default;

            int operator()(int i) const {
                if (std::this_thread::get_id() == submitting_thread && ++*advances == advance_failure) {
                    throw std::runtime_error("step advance failed");
                }
                return i + 1;
            }
        };

        auto submission_failure = [](int copy_failure, int advance_failure, int expected_completed) {
            int copies = 0;
            int advances = 0;
            std::atomic<int> completed {0};
            throwing_step next(copies, advances, copy_failure, advance_failure);
            try {
                fdapde::parallel_for(
                  0, 3, 1,
                  [&](int) {
                      std::this_thread::sleep_for(std::chrono::milliseconds(10));
                      completed.fetch_add(1, std::memory_order_relaxed);
                  },
                  next);
            } catch (const std::runtime_error& error) {
                const std::string expected = copy_failure > 0 ? "step copy failed" : "step advance failed";
                return error.what() == expected && completed.load(std::memory_order_relaxed) == expected_completed;
            }
            return false;
        };

        // Fail before any task is queued, while preparing the second task, and after publishing the first task.
        EXPECT_TRUE(submission_failure(1, 0, 0));
        EXPECT_TRUE(submission_failure(2, 0, 1));
        EXPECT_TRUE(submission_failure(0, 5, 1));
        EXPECT_TRUE(fdapde::parallel_async([&] { return submission_failure(2, 0, 1); }).get());
        EXPECT_TRUE(fdapde::parallel_async([&] { return submission_failure(0, 5, 1); }).get());

        // Both overloads remain usable after a failed submission.
        std::vector<int> values(8, 0);
        fdapde::parallel_for(0, 8, 1, [&](int i) { ++values[i]; });
        fdapde::parallel_for(0, 8, 1, [&](int i) { ++values[i]; }, [](int i) { return i + 1; });
        for (int value : values) { EXPECT_EQ(value, 2); }
        fdapde::parallel_join();
    }
    // nested parallel algorithms
    {
        constexpr int outer_size = 8;
        constexpr int inner_size = 32;
        std::vector<int> values(static_cast<std::size_t>(outer_size * inner_size), 0);
        std::atomic<int> completed_rows {0};

        fdapde::parallel_for(0, outer_size, [&](int row) {
            const int begin = row * inner_size;
            const int end = begin + inner_size;
            fdapde::parallel_for(begin, end, [&](int i) { values[static_cast<std::size_t>(i)] = row + 1; });
            const int sum = fdapde::parallel_reduce(values.data() + begin, values.data() + end, 0, std::plus<>());
            if (sum == (row + 1) * inner_size) { completed_rows.fetch_add(1, std::memory_order_relaxed); }
        });

        EXPECT_EQ(completed_rows.load(std::memory_order_relaxed), outer_size);
        for (int row = 0; row < outer_size; ++row) {
            for (int column = 0; column < inner_size; ++column) {
                EXPECT_EQ(values[static_cast<std::size_t>(row * inner_size + column)], row + 1);
            }
        }
    }
}

TEST(execution, parallel_for_each) {
    // explicit and automatic grain sizes
    {
        std::vector<int> values(97, 0);

        fdapde::parallel_for_each(values, 11, [](int& value) { value += 1; });
        fdapde::parallel_for_each(values, [](int& value) { value += 2; });

        for (int value : values) { EXPECT_EQ(value, 3); }
    }
    // Failed iterator preparation waits for earlier chunks, including nested calls.
    {
        auto submission_failure = [](int copy_offset, int assignment_offset, int expected_reads, bool automatic_grain) {
            throwing_range range(copy_offset, assignment_offset);
            try {
                if (automatic_grain) {
                    fdapde::parallel_for_each(range, [](int& value) { ++value; });
                } else {
                    fdapde::parallel_for_each(range, 1, [](int& value) { ++value; });
                }
            } catch (const std::runtime_error& error) {
                return std::string(error.what()) == "iterator preparation failed" &&
                       range.state.reads.load(std::memory_order_relaxed) == expected_reads;
            }
            return false;
        };
        for (bool automatic_grain : {false, true}) {
            EXPECT_TRUE(submission_failure(1, -1, 0, automatic_grain));
            EXPECT_TRUE(submission_failure(2, -1, 1, automatic_grain));
            EXPECT_TRUE(submission_failure(-1, 1, 1, automatic_grain));
            EXPECT_TRUE(fdapde::parallel_async([&] { return submission_failure(2, -1, 1, automatic_grain); }).get());
            EXPECT_TRUE(fdapde::parallel_async([&] { return submission_failure(-1, 1, 1, automatic_grain); }).get());
        }
        std::vector<int> values(8, 0);
        fdapde::parallel_for_each(values, 1, [](int& value) { ++value; });
        for (int value : values) { EXPECT_EQ(value, 1); }
        fdapde::parallel_join();
    }
}

TEST(execution, parallel_reduce) {
    // reduction seed and input order
    {
        const std::vector<int> factors {2, 3, 4};
        EXPECT_EQ(fdapde::parallel_reduce(factors.begin(), factors.end(), 2, 5, std::multiplies<>()), 120);
        EXPECT_EQ(fdapde::parallel_reduce(factors.begin(), factors.end(), 5, std::multiplies<>()), 120);

        const int values[] {1, 2, 3};
        EXPECT_EQ(fdapde::parallel_reduce(values, values + 3, 2, 10, std::plus<>()), 16);
        EXPECT_EQ(fdapde::parallel_reduce(values, values, 42, std::plus<>()), 42);

        const std::vector<std::string> tokens {"a", "b", "c", "d"};
        EXPECT_EQ(
          fdapde::parallel_reduce(tokens.begin(), tokens.end(), 2, std::string("seed:"), std::plus<>()), "seed:abcd");
    }
    // Failed iterator preparation waits for earlier chunks, including nested calls.
    {
        auto submission_failure = [](int copy_offset, int assignment_offset, int expected_reads, bool automatic_grain) {
            throwing_range range(copy_offset, assignment_offset);
            try {
                if (automatic_grain) {
                    fdapde::parallel_reduce(range.begin(), range.end(), 0, std::plus<>());
                } else {
                    fdapde::parallel_reduce(range.begin(), range.end(), 1, 0, std::plus<>());
                }
            } catch (const std::runtime_error& error) {
                return std::string(error.what()) == "iterator preparation failed" &&
                       range.state.reads.load(std::memory_order_relaxed) == expected_reads;
            }
            return false;
        };
        for (bool automatic_grain : {false, true}) {
            EXPECT_TRUE(submission_failure(1, -1, 0, automatic_grain));
            EXPECT_TRUE(submission_failure(2, -1, 1, automatic_grain));
            EXPECT_TRUE(submission_failure(-1, 1, 1, automatic_grain));
            EXPECT_TRUE(fdapde::parallel_async([&] { return submission_failure(2, -1, 1, automatic_grain); }).get());
            EXPECT_TRUE(fdapde::parallel_async([&] { return submission_failure(-1, 1, 1, automatic_grain); }).get());
        }
        const std::vector<int> values {1, 2, 3};
        EXPECT_EQ(fdapde::parallel_reduce(values.begin(), values.end(), 1, 10, std::plus<>()), 16);
        fdapde::parallel_join();
    }
}
