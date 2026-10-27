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
