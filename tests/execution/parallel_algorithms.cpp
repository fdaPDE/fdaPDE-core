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
#include <functional>
#include <numeric>
#include <string>
#include <vector>

// verifies parallel for supports partitioning stepping and empty ranges
TEST(ExecutionParallelAlgorithms, ParallelForSupportsPartitioningSteppingAndEmptyRanges) {
    std::vector<int> values(64, 0);

    fdapde::parallel_for(0, 64, 7, [&](int i) { values[static_cast<std::size_t>(i)] += 1; });
    fdapde::parallel_for(0, 64, [&](int i) { values[static_cast<std::size_t>(i)] += 2; });
    fdapde::parallel_for(
      0, 64, 3, [&](int i) { values[static_cast<std::size_t>(i)] += 4; }, [](int i) { return i + 2; });
    fdapde::parallel_for(4, 4, [&](int) {
        // fails if an empty range invokes its loop body
        FAIL() << "empty ranges must not execute";
    });
    fdapde::parallel_for(5, 4, [&](int) {
        // fails if a reversed range invokes its loop body
        FAIL() << "reversed ranges must not execute";
    });

    for (int i = 0; i < 64; ++i) {
        // compares each element with the expected unit-step and even-index contributions
        EXPECT_EQ(values[static_cast<std::size_t>(i)], i % 2 == 0 ? 7 : 3);
    }
}

// verifies parallel for each visits each element exactly once
TEST(ExecutionParallelAlgorithms, ParallelForEachVisitsEachElementExactlyOnce) {
    std::vector<int> values(97, 0);

    fdapde::parallel_for_each(values, 11, [](int& value) { value += 1; });
    fdapde::parallel_for_each(values, [](int& value) { value += 2; });

    for (int value : values) {
        // checks each element received both for-each updates exactly once
        EXPECT_EQ(value, 3);
    }
}

// verifies parallel reduce honors the initial value and raw pointers
TEST(ExecutionParallelAlgorithms, ParallelReduceHonorsTheInitialValueAndRawPointers) {
    const std::vector<int> factors {2, 3, 4};
    // checks explicit chunking multiplies the seed by all factors exactly once
    EXPECT_EQ(fdapde::parallel_reduce(factors.begin(), factors.end(), 2, 5, std::multiplies<>()), 120);
    // checks automatic chunking preserves the same multiplicative seed
    EXPECT_EQ(fdapde::parallel_reduce(factors.begin(), factors.end(), 5, std::multiplies<>()), 120);

    const int values[] {1, 2, 3};
    // checks reduction over raw pointers adds the seed once
    EXPECT_EQ(fdapde::parallel_reduce(values, values + 3, 2, 10, std::plus<>()), 16);
    // checks an empty pointer range returns the unchanged seed
    EXPECT_EQ(fdapde::parallel_reduce(values, values, 42, std::plus<>()), 42);

    const std::vector<std::string> tokens {"a", "b", "c", "d"};
    // compares concatenation with input order and a single leading seed
    EXPECT_EQ(
      fdapde::parallel_reduce(tokens.begin(), tokens.end(), 2, std::string("seed:"), std::plus<>()), "seed:abcd");
}

// verifies nested algorithms complete before their caller returns
TEST(ExecutionParallelAlgorithms, NestedAlgorithmsCompleteBeforeTheirCallerReturns) {
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

    // counts outer tasks whose nested reduction observed a fully completed inner loop
    EXPECT_EQ(completed_rows.load(std::memory_order_relaxed), outer_size);
    for (int row = 0; row < outer_size; ++row) {
        for (int column = 0; column < inner_size; ++column) {
            // checks every cell was written before the nested parallel call returned
            EXPECT_EQ(values[static_cast<std::size_t>(row * inner_size + column)], row + 1);
        }
    }
}

namespace {
/// @brief supplies a reduction value that cannot be initialized with an artificial default identity
struct Product {
    int value;
    /// @brief constructs the reduction value from an input factor
    explicit Product(int value_) : value(value_) { }
};
}   // namespace

// verifies partial reductions initialize from input elements when the result has no default constructor
TEST(ExecutionParallelAlgorithms, ReductionNeedsNoDefaultConstructedIdentity) {
    const std::vector<Product> factors {Product(2), Product(3), Product(4)};
    auto multiply = [](Product lhs, Product rhs) { return Product(lhs.value * rhs.value); };
    const auto product = fdapde::parallel_reduce(factors.begin(), factors.end(), 2, Product(5), multiply);
    // compares the result with the seed applied once to the three factors
    EXPECT_EQ(product.value, 120);
}

// verifies body failures are selected by index after all submitted chunks finish and the pool remains reusable
TEST(ExecutionParallelAlgorithms, ParallelForRethrowsLowestIndexAfterCompletion) {
    std::atomic<int> completed_chunks {0};
    bool failed = false;
    try {
        fdapde::parallel_for(0, 64, 8, [&](int i) {
            if (i == 5 || i == 24) throw std::runtime_error("failure " + std::to_string(i));
            if (i % 8 == 7) completed_chunks.fetch_add(1, std::memory_order_relaxed);
        });
    } catch (const std::runtime_error& error) {
        failed = true;
        // the lower failing index wins independently of which worker reports its exception first
        EXPECT_STREQ(error.what(), "failure 5");
        // both failed chunks stop early while the other six finish before the exception reaches this caller
        EXPECT_EQ(completed_chunks.load(std::memory_order_relaxed), 6);
    }
    // the invalid body must not be silently accepted or reported only through worker termination
    EXPECT_TRUE(failed);
    std::atomic<int> recovered {0};
    fdapde::parallel_for(0, 64, [&](int) { recovered.fetch_add(1, std::memory_order_relaxed); });
    // complete accounting lets a subsequent loop execute every iteration on the same worker pool
    EXPECT_EQ(recovered.load(std::memory_order_relaxed), 64);
}

// verifies nested failure propagation drains both inner and outer task groups without deadlocking workers
TEST(ExecutionParallelAlgorithms, NestedParallelForPropagatesFailuresAfterCompletion) {
    std::atomic<int> completed_rows {0};
    // the inner domain error must reach the external caller through the outer loop's failure handling
    EXPECT_THROW(
      fdapde::parallel_for(
        0, 8, 1,
        [&](int row) {
            fdapde::parallel_for(0, 16, 4, [&](int column) {
                if (row == 3 && column == 2) throw std::domain_error("invalid nested value");
            });
            completed_rows.fetch_add(1, std::memory_order_relaxed);
        }),
      std::domain_error);
    // the seven successful inner loops finish before the failed outer group releases its captured state
    EXPECT_EQ(completed_rows.load(std::memory_order_relaxed), 7);
}

namespace {
/// @brief distinguishes chunk-local step copies from the original used to partition the range
class FailChunkStep {
   public:
    /// @brief keeps the original step usable during range counting and partitioning
    FailChunkStep() = default;
    /// @brief marks each step copied into a chunk so only execution can trigger the injected failure
    FailChunkStep(const FailChunkStep&) noexcept : in_chunk_(true) { }
    /// @brief preserves the chunk marker while executor wrappers transfer the prepared callable
    FailChunkStep(FailChunkStep&&) noexcept = default;
    /// @brief fails at the selected chunk index independently of which thread executes that chunk
    int operator()(int i) const {
        if (in_chunk_ && i == 4) throw std::domain_error("invalid chunk step");
        return i + 2;
    }
   private:
    bool in_chunk_ = false;
};
}   // namespace

// verifies custom-step chunks report body and step failures without escaping worker execution
TEST(ExecutionParallelAlgorithms, SteppedParallelForRethrowsBodyAndStepFailures) {
    std::atomic<int> visited {0};
    bool failed = false;
    try {
        fdapde::parallel_for(
          0, 32, 3,
          [&](int i) {
              visited.fetch_add(1, std::memory_order_relaxed);
              if (i == 10 || i == 22) throw std::runtime_error("failure " + std::to_string(i));
          },
          [](int i) { return i + 2; });
    } catch (const std::runtime_error& error) {
        failed = true;
        // the first failing stepped index is selected even when another chunk fails sooner in wall-clock time
        EXPECT_STREQ(error.what(), "failure 10");
    }
    // the custom-step overload must expose its body error to the caller
    EXPECT_TRUE(failed);
    // failures occur at chunk ends, so all sixteen even indices finish before the loop returns
    EXPECT_EQ(visited.load(std::memory_order_relaxed), 16);
    // only chunk-local copies throw, so cooperative execution cannot hide the step failure from its chunk handler
    EXPECT_THROW(fdapde::parallel_for(0, 12, 3, [](int) { }, FailChunkStep()), std::domain_error);
}

namespace {
/// @brief injects a callable-construction failure after one stepped chunk has been published
class FailSecondStepCopy {
   public:
    /// @brief retains a copy counter shared by the caller and chunk-local step objects
    explicit FailSecondStepCopy(std::shared_ptr<std::atomic<int>> copies) : copies_(std::move(copies)) { }
    /// @brief lets the first chunk bind its step and rejects the next chunk before publication
    FailSecondStepCopy(const FailSecondStepCopy& other) : copies_(other.copies_) {
        if (copies_->fetch_add(1, std::memory_order_relaxed) == 1)
            throw std::runtime_error("step copy submission failure");
    }
    /// @brief moves a prepared step through executor wrappers without creating another copy
    FailSecondStepCopy(FailSecondStepCopy&&) noexcept = default;
    /// @brief advances the preparation and worker loops by one index
    int operator()(int i) const { return i + 1; }
   private:
    std::shared_ptr<std::atomic<int>> copies_;
};
}   // namespace

// verifies partial submission failure drains the first chunk before releasing its stack-bound loop body
TEST(ExecutionParallelAlgorithms, PartialSubmissionFailureDrainsPublishedChunks) {
    auto copies = std::make_shared<std::atomic<int>>(0);
    std::atomic<int> completed {0};
    // copying the second chunk's step throws after the first chunk is already owned by the executor
    EXPECT_THROW(
      fdapde::parallel_for(
        0, 12, 4, [&](int) { completed.fetch_add(1, std::memory_order_relaxed); }, FailSecondStepCopy(copies)),
      std::runtime_error);
    // exactly two construction attempts establish that failure occurred after one successful submission
    EXPECT_EQ(copies->load(std::memory_order_relaxed), 2);
    // all four callbacks from the published chunk complete before the captured local counter can be destroyed
    EXPECT_EQ(completed.load(std::memory_order_relaxed), 4);
    fdapde::parallel_for(0, 8, [&](int) { completed.fetch_add(1, std::memory_order_relaxed); });
    // releasing the failed reservation leaves the runtime usable for subsequent loops
    EXPECT_EQ(completed.load(std::memory_order_relaxed), 12);
}
