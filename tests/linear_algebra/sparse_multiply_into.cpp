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

#include <fdaPDE/sparse_linear_algebra.h>
#include <gtest/gtest.h>

#include <cstdlib>
#include <limits>
#include <new>
#include <type_traits>
#include <utility>

namespace {
thread_local bool count_enabled = false;
thread_local std::size_t allocation_count = 0;

// counts allocations only while the supplied operation executes, restoring state after exceptions
template <typename Operation> std::size_t count_allocations(Operation operation) {
    allocation_count = 0;
    count_enabled = true;
    try {
        operation();
    } catch (...) {
        count_enabled = false;
        throw;
    }
    count_enabled = false;
    return allocation_count;
}

template <typename Matrix, typename Input, typename Output>
concept has_multiply_into =
  requires(const Matrix& matrix, const Input& input, Output& output) { matrix.multiply_into(input, output); };

using sparse_double = fdapde::SparseMatrix<double>;
using double_vector = fdapde::Vector<double, fdapde::Dynamic>;
using const_double_view = fdapde::VectorView<const double, fdapde::Dynamic>;
// a const input view can write into a preallocated native owner
static_assert(has_multiply_into<sparse_double, const_double_view, double_vector>);
// a const owner cannot be used as a mutable destination
static_assert(!has_multiply_into<sparse_double, double_vector, const double_vector>);
// a view of const coefficients cannot be used as a mutable destination
static_assert(!has_multiply_into<sparse_double, double_vector, const_double_view>);
// volatile coefficients cannot be borrowed as ordinary input storage
static_assert(!has_multiply_into<sparse_double, fdapde::VectorView<volatile double, fdapde::Dynamic>, double_vector>);
// volatile objects cannot provide the ordinary nonvolatile output interface
static_assert(!has_multiply_into<sparse_double, double_vector, volatile double_vector>);
// output coefficients retain the scalar promotion of the existing sparse product
static_assert(!has_multiply_into<sparse_double, double_vector, fdapde::Vector<float, fdapde::Dynamic>>);
// a statically known row-vector destination cannot satisfy the column-vector contract
static_assert(!has_multiply_into<sparse_double, double_vector, fdapde::Matrix<double, 1, 3>>);

// checks exact hand-computed rectangular products and repeated writes without storage growth
template <typename Scalar> void check_rectangular_product() {
    const fdapde::SparseMatrix<Scalar> matrix(
      3, 4,
      {
        {2, 0, Scalar(5) },
        {0, 2, Scalar(-1)},
        {1, 3, Scalar(4) },
        {0, 0, Scalar(1) },
        {1, 1, Scalar(3) },
        {0, 0, Scalar(1) }
    });
    const fdapde::Vector<Scalar, 4> input({Scalar(1), Scalar(2), Scalar(3), Scalar(4)});
    fdapde::Vector<Scalar, fdapde::Dynamic> output(3);
    auto* original_data = output.data();
    const auto allocations = count_allocations([&] {
        matrix.multiply_into(input, output);
        matrix.multiply_into(input, output);
    });
    // repeated writes into disjoint dynamic storage perform no allocation
    EXPECT_EQ(allocations, 0u);
    // the output buffer remains the exact preallocated storage
    EXPECT_EQ(output.data(), original_data);
    fdapde::Vector<Scalar, fdapde::Dynamic> independent;
    const auto allocating_allocations = count_allocations([&] { independent = matrix * input; });
    // the allocating public product is a positive control for the allocation counter
    EXPECT_GT(allocating_allocations, 0u);
    const Scalar expected[3] {Scalar(-1), Scalar(22), Scalar(5)};
    for (int row = 0; row < 3; ++row) {
        // manually summed rows include unsorted triplets and the duplicate coefficient at zero, zero
        EXPECT_EQ(output[row], expected[row]);
        // the retained allocating API computes the same exact row sums through the shared kernel
        EXPECT_EQ(independent[row], expected[row]);
    }
}
}   // namespace

#if defined(__GNUC__) || defined(__clang__)
#    define FDAPDE_SPMV_TEST_NOINLINE __attribute__((noinline))
#else
#    define FDAPDE_SPMV_TEST_NOINLINE
#endif

// intercepts ordinary scalar allocations used by native float and double vector storage
FDAPDE_SPMV_TEST_NOINLINE void* operator new(std::size_t size) {
    if (void* pointer = std::malloc(size == 0 ? 1 : size)) {
        if (count_enabled) ++allocation_count;
        return pointer;
    }
    throw std::bad_alloc();
}
// intercepts array allocations through the same counter
FDAPDE_SPMV_TEST_NOINLINE void* operator new[](std::size_t size) { return ::operator new(size); }
// releases storage allocated through the replacement allocation functions
FDAPDE_SPMV_TEST_NOINLINE void operator delete(void* pointer) noexcept { std::free(pointer); }
// releases array storage allocated through the replacement allocation functions
FDAPDE_SPMV_TEST_NOINLINE void operator delete[](void* pointer) noexcept { std::free(pointer); }
// supports compilers that emit sized scalar deallocation
FDAPDE_SPMV_TEST_NOINLINE void operator delete(void* pointer, std::size_t) noexcept { std::free(pointer); }
// supports compilers that emit sized array deallocation
FDAPDE_SPMV_TEST_NOINLINE void operator delete[](void* pointer, std::size_t) noexcept { std::free(pointer); }

#undef FDAPDE_SPMV_TEST_NOINLINE

// checks float and double CSR row sums against independent exact rectangular arithmetic
TEST(SparseMultiplyInto, RectangularFloatAndDouble) {
    check_rectangular_product<float>();
    check_rectangular_product<double>();
}

// checks const input and offset output views without allocating or touching surrounding coefficients
TEST(SparseMultiplyInto, OffsetViewsAndScalarPromotion) {
    const fdapde::SparseMatrix<float> matrix(
      3, 4,
      {
        {0, 0, 2 },
        {0, 2, -1},
        {1, 1, 3 },
        {1, 3, 4 },
        {2, 0, 5 }
    });
    const double input_storage[6] {99, 1, 2, 3, 4, 99};
    double output_storage[5] {91, 91, 91, 91, 91};
    const const_double_view input(input_storage + 1, 4);
    fdapde::VectorView<double, fdapde::Dynamic> output(output_storage + 1, 3);
    const auto allocations = count_allocations([&] { matrix.multiply_into(input, output); });
    // mixed float and double operands write directly into their promoted double output view
    EXPECT_EQ(allocations, 0u);
    const double expected[5] {91, -1, 22, 5, 91};
    for (int i = 0; i < 5; ++i) {
        // exact row sums and untouched guard coefficients bound the three-coefficient output view
        EXPECT_EQ(output_storage[i], expected[i]);
    }
    double adjacent_storage[7] {1, 2, 3, 4, 91, 91, 91};
    const const_double_view adjacent_input(adjacent_storage, 4);
    fdapde::VectorView<double, fdapde::Dynamic> adjacent_output(adjacent_storage + 4, 3);
    const auto adjacent_allocations = count_allocations([&] { matrix.multiply_into(adjacent_input, adjacent_output); });
    // touching range endpoints in one backing array are disjoint and require no temporary allocation
    EXPECT_EQ(adjacent_allocations, 0u);
    for (int row = 0; row < 3; ++row) {
        // adjacent output storage receives the same manually summed rectangular product
        EXPECT_EQ(adjacent_output[row], expected[row + 1]);
    }
}

// checks empty dimensions and absent rows overwrite existing values with the zero row sum
TEST(SparseMultiplyInto, EmptyDimensionsAndRows) {
    for (const auto shape : {
           std::pair {0, 0},
            std::pair {0, 4},
            std::pair {3, 0},
            std::pair {3, 4}
    }) {
        const sparse_double matrix(shape.first, shape.second);
        const double_vector input(shape.second);
        double_vector output(shape.first);
        for (int row = 0; row < output.size(); ++row) output[row] = 7;
        const auto allocations = count_allocations([&] { matrix.multiply_into(input, output); });
        // empty operands and empty CSR rows never need temporary storage
        EXPECT_EQ(allocations, 0u);
        for (int row = 0; row < output.size(); ++row) {
            // a row without stored coefficients overwrites the old sentinel with zero
            EXPECT_EQ(output[row], 0);
        }
    }
}

// checks shape failures precede any write and runtime-sized column matrices use the same public API
TEST(SparseMultiplyInto, RuntimeShapeContract) {
    const sparse_double matrix(
      2, 3,
      {
        {0, 0, 2},
        {1, 2, 3}
    });
    const fdapde::Matrix<double, fdapde::Dynamic, fdapde::Dynamic, fdapde::ColMajor> input(3, 1);
    fdapde::Matrix<double, fdapde::Dynamic, fdapde::Dynamic> output(2, 1);
    output(0, 0) = 7;
    output(1, 0) = 8;
    const double_vector short_input(2);
    // a mismatching input length is rejected before output is accessed
    EXPECT_THROW(matrix.multiply_into(short_input, output), std::invalid_argument);
    // the first output sentinel survives invalid input dimensions
    EXPECT_EQ(output(0, 0), 7);
    // the second output sentinel survives invalid input dimensions
    EXPECT_EQ(output(1, 0), 8);
    const fdapde::Matrix<double, fdapde::Dynamic, fdapde::Dynamic> wide_input(3, 2);
    // a matching row count cannot make a two-column input a vector
    EXPECT_THROW(matrix.multiply_into(wide_input, output), std::invalid_argument);
    fdapde::Matrix<double, fdapde::Dynamic, fdapde::Dynamic> wide_output(2, 2);
    // an equal row count cannot make a two-column output a vector
    EXPECT_THROW(matrix.multiply_into(input, wide_output), std::invalid_argument);
    double_vector short_output(1);
    // preallocated output length must match the sparse row count without resizing
    EXPECT_THROW(matrix.multiply_into(input, short_output), std::invalid_argument);
    const auto allocations = count_allocations([&] { matrix.multiply_into(input, output); });
    // runtime-sized native column matrices follow the same allocation-free path
    EXPECT_EQ(allocations, 0u);
    for (int row = 0; row < 2; ++row) {
        // the zero-filled valid input replaces each sentinel with its exact zero row sum
        EXPECT_EQ(output(row, 0), 0);
    }
}

// checks exact aliasing retains original input values until all sparse rows are evaluated
TEST(SparseMultiplyInto, ExactInputOutputAlias) {
    const sparse_double matrix(
      3, 3,
      {
        {0, 1, 2},
        {1, 2, 3},
        {2, 0, 4}
    });
    fdapde::Vector<double, 3> vector({1, 2, 3});
    const auto allocations = count_allocations([&] { matrix.multiply_into(vector, vector); });
    // an aliased call uses independent temporary storage, proving the counter detects allocation
    EXPECT_GT(allocations, 0u);
    const double expected[3] {4, 9, 4};
    for (int row = 0; row < 3; ++row) {
        // cyclic row dependencies are evaluated from the three original input coefficients
        EXPECT_EQ(vector[row], expected[row]);
    }
}

// checks both partial-overlap directions preserve input coefficients needed by later sparse rows
TEST(SparseMultiplyInto, PartialInputOutputAlias) {
    const sparse_double matrix(
      3, 4,
      {
        {0, 0, 2},
        {1, 1, 3},
        {2, 0, 4}
    });
    for (int output_offset : {0, 2}) {
        double storage[6] {99, 2, 3, 4, 5, 99};
        const const_double_view input(storage + 1, 4);
        fdapde::VectorView<double, fdapde::Dynamic> output(storage + output_offset, 3);
        const auto allocations = count_allocations([&] { matrix.multiply_into(input, output); });
        // each overlapping pointer range selects temporary output even when the starting addresses differ
        EXPECT_GT(allocations, 0u);
        const double expected[3] {4, 9, 8};
        for (int row = 0; row < 3; ++row) {
            // manually summed rows retain the original input despite writes on either side of its start
            EXPECT_EQ(output[row], expected[row]);
        }
    }
}

// checks an output view into mutable CSR values cannot corrupt coefficients needed by later rows
TEST(SparseMultiplyInto, SparseValuesOutputAlias) {
    sparse_double matrix(
      3, 3,
      {
        {0, 0, 2},
        {1, 1, 3},
        {2, 0, 4},
        {2, 2, 5}
    });
    const fdapde::Vector<double, 3> input({1, 2, 3});
    fdapde::VectorView<double, 3> output(&matrix.value_ref(1, 1));
    const auto allocations = count_allocations([&] { matrix.multiply_into(input, output); });
    // overlap with stored sparse coefficients also selects independent temporary output
    EXPECT_GT(allocations, 0u);
    const double expected[3] {2, 6, 19};
    for (int row = 0; row < 3; ++row) {
        // every row uses the original sparse coefficients before the final copy mutates their storage
        EXPECT_EQ(output[row], expected[row]);
    }
}

// checks checked integral arithmetic remains active in the preallocated path
TEST(SparseMultiplyInto, IntegralOverflow) {
    const fdapde::SparseMatrix<int> matrix(
      1, 1,
      {
        {0, 0, std::numeric_limits<int>::min()}
    });
    const fdapde::Vector<int, 1> input(-1);
    fdapde::Vector<int, 1> output(7);
    // the minimum signed coefficient multiplied by negative one is rejected before undefined overflow
    EXPECT_THROW(matrix.multiply_into(input, output), std::overflow_error);
    // no row result is written when its checked multiplication fails
    EXPECT_EQ(output(0, 0), 7);
}
