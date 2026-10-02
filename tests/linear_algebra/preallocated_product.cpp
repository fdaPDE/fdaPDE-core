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

#include <fdaPDE/dense_linear_algebra.h>
#include <gtest/gtest.h>

#include <array>
#include <cstdlib>
#include <new>

namespace allocation_probe {
thread_local bool active = false;
thread_local std::size_t count = 0;
}   // namespace allocation_probe

#if defined(__GNUC__) || defined(__clang__)
#    define FDAPDE_GEMV_TEST_NOINLINE __attribute__((noinline))
#else
#    define FDAPDE_GEMV_TEST_NOINLINE
#endif

// count coefficient-storage allocations without inlining the replacement allocation protocol
FDAPDE_GEMV_TEST_NOINLINE void* operator new(std::size_t size) {
    if (void* storage = std::malloc(size == 0 ? 1 : size)) {
        if (allocation_probe::active) ++allocation_probe::count;
        return storage;
    }
    throw std::bad_alloc();
}
FDAPDE_GEMV_TEST_NOINLINE void operator delete(void* storage) noexcept { std::free(storage); }
FDAPDE_GEMV_TEST_NOINLINE void operator delete(void* storage, std::size_t) noexcept { std::free(storage); }

#undef FDAPDE_GEMV_TEST_NOINLINE

namespace fdapde {
namespace {

/// @brief limits allocation instrumentation to the current thread and restores it after exceptions
struct AllocationGuard {
    /// @brief starts an empty allocation counter
    AllocationGuard() {
        allocation_probe::count = 0;
        allocation_probe::active = true;
    }
    /// @brief stops counting before test assertions or diagnostics execute
    ~AllocationGuard() { allocation_probe::active = false; }
};

/// @brief counts allocations made by the supplied public calls without counting their test assertions
template <typename Function> std::size_t allocations(Function function) {
    AllocationGuard guard;
    function();
    return allocation_probe::count;
}

/// @brief initializes bounded binary fractions by logical coordinates
template <typename Matrix> void fill_operand(Matrix& matrix) {
    using Scalar = typename Matrix::Scalar;
    for (int i = 0; i < matrix.rows(); ++i)
        for (int j = 0; j < matrix.cols(); ++j) matrix(i, j) = Scalar((7 * i + 3 * j) % 17 - 8) / Scalar(4);
}

/// @brief materializes an independent increasing-inner-index reference before any potentially aliased writes
template <typename Matrix, typename Input> auto ordered_reference(const Matrix& matrix, const Input& input) {
    using Scalar = std::common_type_t<typename Matrix::Scalar, typename Input::Scalar>;
    Vector<Scalar, Dynamic> result(matrix.rows());
    for (int i = 0; i < matrix.rows(); ++i) {
        Scalar value = 0;
        for (int k = 0; k < matrix.cols(); ++k) value += matrix(i, k) * input(k, 0);
        result[i] = value;
    }
    return result;
}

/// @brief compares every output coefficient against a reference saved before execution
template <typename Output, typename Reference>
void expect_coefficients(const Output& output, const Reference& reference) {
    // exact binary fractions retain the scalar accumulation result at each logical row
    ASSERT_EQ(output.rows(), reference.rows());
    for (int i = 0; i < output.rows(); ++i) {
        // every output coefficient must match the independent ordered reference
        EXPECT_EQ(output(i, 0), reference(i, 0));
    }
}

/// @brief verifies repeated direct and transposed products for one type and layout, including empty dimensions
template <typename Scalar, int Order> void check_layout() {
    for (const auto shape : {
           std::array {7, 5},
            std::array {1, 9},
            std::array {9, 1},
            std::array {7, 0},
            std::array {0, 5}
    }) {
        Matrix<Scalar, Dynamic, Dynamic, Order> matrix(shape[0], shape[1]);
        Vector<Scalar, Dynamic> input(shape[1]), transposed_input(shape[0]);
        Vector<Scalar, Dynamic> output(shape[0]), transposed_output(shape[1]);
        fill_operand(matrix);
        fill_operand(input);
        fill_operand(transposed_input);
        const auto reference = ordered_reference(matrix, input);
        const auto transposed_reference = ordered_reference(matrix.transpose(), transposed_input);
        const auto* address = output.data();
        const auto* transposed_address = transposed_output.data();
        const auto count = allocations([&] {
            for (int repeat = 0; repeat < 5; ++repeat) {
                matrix.multiply_into(input, output);
                matrix.transpose().multiply_into(transposed_input, transposed_output);
            }
        });
        // neither product allocates coefficient storage or materializes a transpose for disjoint buffers
        EXPECT_EQ(count, 0u);
        // direct multiplication keeps the persistent output bound to its original allocation
        EXPECT_EQ(output.data(), address);
        // transposed multiplication keeps the persistent output bound to its original allocation
        EXPECT_EQ(transposed_output.data(), transposed_address);
        // every direct-product coefficient follows the scalar increasing-k reference
        expect_coefficients(output, reference);
        // every transposed-product coefficient follows the scalar increasing-k reference
        expect_coefficients(transposed_output, transposed_reference);
    }
}

/// @brief identifies writable lvalue destinations for the explicit disjoint assignment contract
template <typename Destination, typename Source>
concept permits_disjoint_assignment =
  requires(Destination& destination, const Source& source) { destination.assign_disjoint(source); };
/// @brief rejects temporary destinations whose storage lifetime could be unclear to the caller
template <typename Destination, typename Source>
concept permits_temporary_disjoint_assignment =
  requires(Destination& destination, const Source& source) { std::move(destination).assign_disjoint(source); };
using assignment_matrix = Matrix<double, Dynamic, Dynamic>;
using assignment_view = MatrixView<double, Dynamic, Dynamic>;
using const_assignment_view = MatrixView<const double, Dynamic, Dynamic>;
// writable owners expose the explicit assignment contract
static_assert(permits_disjoint_assignment<assignment_matrix, const_assignment_view>);
// writable views also accept native const input views
static_assert(permits_disjoint_assignment<assignment_view, const_assignment_view>);
// a view with const coefficients cannot be assigned
static_assert(!permits_disjoint_assignment<const_assignment_view, assignment_matrix>);
// a const destination object cannot be assigned
static_assert(!permits_disjoint_assignment<const assignment_matrix, assignment_matrix>);
// temporary owners cannot expose the disjoint assignment contract
static_assert(!permits_temporary_disjoint_assignment<assignment_matrix, assignment_matrix>);
// temporary views also require a named lvalue destination
static_assert(!permits_temporary_disjoint_assignment<assignment_view, assignment_matrix>);

/// @brief checks fused affine and projected expressions with const inputs and persistent owner/view storage
template <int Order> void check_disjoint_assignment() {
    Matrix<double, Dynamic, Dynamic, Order> left(3, 5), right(3, 5), output(3, 5);
    fill_operand(left);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 5; ++j) right(i, j) = double((5 * i + 11 * j) % 13 - 6) / 8;
    MatrixView<const double, Dynamic, Dynamic, Order> left_view(left.data(), 3, 5);
    MatrixView<const double, Dynamic, Dynamic, Order> right_view(right.data(), 3, 5);
    const auto affine = left_view - 0.25 * right_view;
    const auto projected = affine.cwise().apply([](double value) { return value < 0 ? 0.0 : value; });
    const auto* address = output.data();
    const auto affine_count = allocations([&] {
        for (int repeat = 0; repeat < 5; ++repeat) output.assign_disjoint(affine);
    });
    // arbitrary affine expressions evaluate directly without allocating a coefficient snapshot
    EXPECT_EQ(affine_count, 0u);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 5; ++j) {
            // each coefficient retains the original affine arithmetic with const input views
            EXPECT_EQ(output(i, j), left(i, j) - 0.25 * right(i, j));
        }
    MatrixView<double, Dynamic, Dynamic, Order> output_view(output.data(), 3, 5);
    const auto projection_count = allocations([&] {
        for (int repeat = 0; repeat < 5; ++repeat) output_view.assign_disjoint(projected);
    });
    // coefficientwise delegation also evaluates into existing external storage without allocation
    EXPECT_EQ(projection_count, 0u);
    // owner assignment does not relocate the persistent destination
    EXPECT_EQ(output.data(), address);
    // view assignment preserves its original external storage binding
    EXPECT_EQ(output_view.data(), address);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 5; ++j) {
            const double value = left(i, j) - 0.25 * right(i, j);
            // projection evaluates every coefficient using the original disjoint operands
            EXPECT_EQ(output(i, j), value < 0 ? 0.0 : value);
        }
}

}   // namespace

// verifies preallocated owners, both layouts and ordered sums for float/double without allocation or resizing
TEST(PreallocatedProduct, OwnersAndTransposesReuseStorage) {
    check_layout<float, RowMajor>();
    check_layout<float, ColMajor>();
    check_layout<double, RowMajor>();
    check_layout<double, ColMajor>();
}

// verifies zero-copy const column-major input and unaligned external output with surrounding sentinels
TEST(PreallocatedProduct, ColumnMajorViewsRemainNativeAndUnaligned) {
    alignas(64) std::array<double, 17> matrix_storage;
    alignas(64) std::array<double, 5> input_storage;
    alignas(64) std::array<double, 7> output_storage;
    matrix_storage.fill(-41);
    input_storage.fill(-53);
    output_storage.fill(-67);
    MatrixView<double, 5, 3, ColMajor> writable_matrix(matrix_storage.data() + 1);
    VectorView<double, 3> writable_input(input_storage.data() + 1);
    fill_operand(writable_matrix);
    fill_operand(writable_input);
    MatrixView<const double, 5, 3, ColMajor> matrix(writable_matrix.data());
    VectorView<const double, 3> input(writable_input.data());
    VectorView<double, 5> output(output_storage.data() + 1);
    const auto reference = ordered_reference(matrix, input);
    const auto count = allocations([&] { matrix.multiply_into(input, output); });
    // the const native view directly reads the external coefficient buffer without allocation
    EXPECT_EQ(count, 0u);
    // one-double offset deliberately avoids sixteen-byte alignment
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(output.data()) % 16, sizeof(double));
    // the native mapped product matches every ordered reference coefficient
    expect_coefficients(output, reference);
    // the prefix sentinel detects writes before the output view
    EXPECT_EQ(output_storage.front(), -67);
    // the suffix sentinel detects writes after the output view
    EXPECT_EQ(output_storage.back(), -67);
    // borrowing the matrix view does not relocate the external input buffer
    EXPECT_EQ(matrix.data(), matrix_storage.data() + 1);
    const auto transposed_reference = ordered_reference(matrix.transpose(), output);
    const auto transposed_count = allocations([&] { matrix.transpose().multiply_into(output, writable_input); });
    // transposed const native views also avoid allocation and materialization for distinct external buffers
    EXPECT_EQ(transposed_count, 0u);
    // every transposed mapped coefficient follows the scalar reference computed before output writes
    expect_coefficients(writable_input, transposed_reference);
    // the input-buffer prefix remains outside the transposed output view
    EXPECT_EQ(input_storage.front(), -53);
    // the input-buffer suffix remains outside the transposed output view
    EXPECT_EQ(input_storage.back(), -53);
}

// verifies snapshots for partial input/output overlap and output ranges inside the original matrix
TEST(PreallocatedProduct, OverlapReadsOriginalOperands) {
    Matrix<double, 5, 3, ColMajor> matrix(5, 3);
    fill_operand(matrix);
    std::array<double, 7> storage {0.25, -0.5, 1.0, 0, 0, 0, 0};
    VectorView<double, 3> input(storage.data());
    VectorView<double, 5> output(storage.data() + 1);
    const auto reference = ordered_reference(matrix, input);
    matrix.multiply_into(input, output);
    // shifted output cannot overwrite input coefficients before the product has read them
    expect_coefficients(output, reference);
    Vector<double, 3> separate_input {0.25, -0.5, 1.0};
    const auto matrix_reference = ordered_reference(matrix, separate_input);
    VectorView<double, 5> matrix_output(matrix.data() + 1);
    matrix.multiply_into(separate_input, matrix_output);
    // output overlapping a matrix column uses the original matrix coefficients
    expect_coefficients(matrix_output, matrix_reference);
    Matrix<double, 5, 3, ColMajor> transposed_matrix(5, 3);
    fill_operand(transposed_matrix);
    Vector<double, 5> transposed_input({0.25, -0.5, 1.0, 0.75, -0.25});
    const auto transpose_reference = ordered_reference(transposed_matrix.transpose(), transposed_input);
    VectorView<double, 3> transposed_output(transposed_input.data() + 1);
    transposed_matrix.transpose().multiply_into(transposed_input, transposed_output);
    // transposed multiplication also snapshots inputs before writing an overlapping result
    expect_coefficients(transposed_output, transpose_reference);
}

// verifies unknown expression storage remains alias-safe and mixed scalar types preserve promotion
TEST(PreallocatedProduct, ExpressionsAndMixedTypesPreserveSemantics) {
    Matrix<double, 3, 3> matrix(3, 3);
    fill_operand(matrix);
    Vector<double, 3> input {0.5, -0.25, 1.0};
    const auto reference = ordered_reference(matrix + matrix, input);
    (matrix + matrix).multiply_into(input, input);
    // expression fallback preserves the original aliased vector while evaluating both matrix operands
    expect_coefficients(input, reference);
    Matrix<float, 5, 3, ColMajor> mixed_matrix(5, 3);
    Vector<double, 3> mixed_input {0.5, -0.25, 1.0};
    Vector<double, 5> mixed_output;
    fill_operand(mixed_matrix);
    const auto mixed_reference = ordered_reference(mixed_matrix, mixed_input);
    const auto count = allocations([&] { mixed_matrix.multiply_into(mixed_input, mixed_output); });
    // the plain mixed-scalar fallback still reuses preallocated promoted output storage
    EXPECT_EQ(count, 0u);
    // common scalar promotion matches the independent mixed-type scalar dot products
    expect_coefficients(mixed_output, mixed_reference);
}

// verifies permanent public dimension checks before any output coefficient changes
TEST(PreallocatedProduct, InvalidShapesLeaveOutputUnchanged) {
    Matrix<double, 3, 5> matrix(3, 5);
    Vector<double, 4> wrong_input;
    Vector<double, 5> input;
    Vector<double, 3> output {7, 8, 9};
    Vector<double, 2> wrong_output;
    Matrix<double, 1, 5> row_input;
    Matrix<double, 3, 2> matrix_output;
    // incompatible input length is rejected even when debug assertions are disabled
    EXPECT_THROW(matrix.multiply_into(wrong_input, output), std::invalid_argument);
    // incorrect output length is rejected rather than resized
    EXPECT_THROW(matrix.multiply_into(input, wrong_output), std::invalid_argument);
    // a row input cannot silently change the matrix-vector orientation
    EXPECT_THROW(matrix.multiply_into(row_input, output), std::invalid_argument);
    // a multi-column destination is rejected before its storage is touched
    EXPECT_THROW(matrix.multiply_into(input, matrix_output), std::invalid_argument);
    for (int i = 0; i < output.rows(); ++i) {
        // all successful-check failures preserve the original preallocated output
        EXPECT_EQ(output[i], 7 + i);
    }
}

// verifies explicit disjoint assignment fuses expressions with native const views and both destination layouts
TEST(PreallocatedAssignment, NativeViewsAndExpressionsReuseStorage) {
    check_disjoint_assignment<RowMajor>();
    check_disjoint_assignment<ColMajor>();
    Vector<float, Dynamic> input(17);
    Vector<double, Dynamic> output(17);
    fill_operand(input);
    VectorView<const float, Dynamic> view(input.data(), input.size());
    const auto* address = output.data();
    const auto count = allocations([&] { output.assign_disjoint(0.5 * view); });
    // vector execution preserves scalar conversion without a temporary allocation
    EXPECT_EQ(count, 0u);
    // the odd-length destination retains its original coefficient storage
    EXPECT_EQ(output.data(), address);
    for (int i = 0; i < 17; ++i) {
        // every promoted vector coefficient matches the same scalar multiplication
        EXPECT_EQ(output[i], 0.5 * input[i]);
    }
}

// verifies permanent exact-shape checks reject incompatible operands before writing or resizing
TEST(PreallocatedAssignment, InvalidShapesLeaveOutputUnchanged) {
    Matrix<double, Dynamic, Dynamic> output(3, 5), wrong_length(3, 4), wrong_orientation(5, 3);
    fill_operand(output);
    const auto original = output;
    const auto* address = output.data();
    // a different coefficient count is rejected rather than resizing the destination
    EXPECT_THROW(output.assign_disjoint(wrong_length), std::invalid_argument);
    // equal coefficient counts cannot silently change the matrix orientation
    EXPECT_THROW(output.assign_disjoint(wrong_orientation), std::invalid_argument);
    // coefficientwise delegation retains the same permanent exact-shape check
    EXPECT_THROW(output.assign_disjoint(wrong_orientation.cwise() + 1.0), std::invalid_argument);
    // rejected assignments preserve every original output coefficient
    EXPECT_EQ(output, original);
    // rejected assignments leave the original allocation bound to the destination
    EXPECT_EQ(output.data(), address);
    Vector<double, Dynamic> column(15);
    Matrix<double, 1, Dynamic> row(15);
    // the explicit contract rejects row-to-column assignment even when their lengths match
    EXPECT_THROW(column.assign_disjoint(row), std::invalid_argument);
}

}   // namespace fdapde
