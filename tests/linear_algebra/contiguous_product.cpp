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
#include <cstdint>
#include <limits>
#include <type_traits>

namespace fdapde {
namespace {

/// @brief compares every product coefficient with an independent scalar accumulation in increasing inner-index order
template <typename Result, typename Lhs, typename Rhs>
void expect_ordered_product(const Result& result, const Lhs& lhs, const Rhs& rhs) {
    using Scalar = std::common_type_t<typename Lhs::Scalar, typename Rhs::Scalar>;
    // the product output has one row for each left-operand row
    ASSERT_EQ(result.rows(), lhs.rows());
    // the product output has one column for each right-operand column
    ASSERT_EQ(result.cols(), rhs.cols());
    for (int i = 0; i < result.rows(); ++i) {
        for (int j = 0; j < result.cols(); ++j) {
            Scalar expected = 0;
            for (int k = 0; k < lhs.cols(); ++k) expected += lhs(i, k) * rhs(k, j);
            // binary-fraction inputs make the ordered scalar dot product an exact coefficient oracle
            EXPECT_EQ(result(i, j), expected);
        }
    }
}

/// @brief initializes bounded binary-fraction coefficients without depending on physical storage order
template <typename Matrix> void fill_product_operand(Matrix& matrix) {
    using Scalar = typename Matrix::Scalar;
    for (int i = 0; i < matrix.rows(); ++i) {
        for (int j = 0; j < matrix.cols(); ++j) matrix(i, j) = Scalar((7 * i + 3 * j) % 17 - 8) / Scalar(4);
    }
}

/// @brief checks rectangular and empty products for one scalar type and one combination of three storage orders
template <typename Scalar, int LhsOrder, int RhsOrder, int ResultOrder> void check_product_layout() {
    for (const auto shape : {
           std::array {3,  5,  7},
            std::array {17, 33, 9},
            std::array {5,  0,  7},
            std::array {0,  3,  7},
           std::array {5,  3,  0}
    }) {
        Matrix<Scalar, Dynamic, Dynamic, LhsOrder> lhs(shape[0], shape[1]);
        Matrix<Scalar, Dynamic, Dynamic, RhsOrder> rhs(shape[1], shape[2]);
        fill_product_operand(lhs);
        fill_product_operand(rhs);
        Matrix<Scalar, Dynamic, Dynamic, ResultOrder> result = lhs * rhs;
        // increasing-k scalar products verify every coordinate in the selected output layout
        expect_ordered_product(result, lhs, rhs);
        result.cwise() = Scalar(9);
        result = lhs * rhs;
        // assigning a product overwrites existing storage, including zero-inner-dimension coefficients
        expect_ordered_product(result, lhs, rhs);
    }
}

/// @brief exercises all left, right and result storage-order combinations for one scalar type
template <typename Scalar> void check_product_layouts() {
    check_product_layout<Scalar, RowMajor, RowMajor, RowMajor>();
    check_product_layout<Scalar, RowMajor, RowMajor, ColMajor>();
    check_product_layout<Scalar, RowMajor, ColMajor, RowMajor>();
    check_product_layout<Scalar, RowMajor, ColMajor, ColMajor>();
    check_product_layout<Scalar, ColMajor, RowMajor, RowMajor>();
    check_product_layout<Scalar, ColMajor, RowMajor, ColMajor>();
    check_product_layout<Scalar, ColMajor, ColMajor, RowMajor>();
    check_product_layout<Scalar, ColMajor, ColMajor, ColMajor>();
}

}   // namespace

// checks odd rectangular products, all operand/output layouts and empty dimensions for float and double
TEST(ContiguousProduct, LayoutsAndEmptyDimensionsFollowOrderedScalarProducts) {
    check_product_layouts<float>();
    check_product_layouts<double>();
}

// checks unaligned external operand/output storage and preserves surrounding sentinel coefficients
TEST(ContiguousProduct, UnalignedViewsPreserveSentinels) {
    alignas(64) std::array<double, 17> lhs_storage;
    alignas(64) std::array<double, 37> rhs_storage;
    alignas(64) std::array<double, 23> result_storage;
    lhs_storage.fill(-71.0);
    rhs_storage.fill(-83.0);
    result_storage.fill(-97.0);
    MatrixView<double, 3, 5, ColMajor> lhs(lhs_storage.data() + 1);
    MatrixView<double, 5, 7> rhs(rhs_storage.data() + 1);
    MatrixView<double, 3, 7, ColMajor> result(result_storage.data() + 1);
    fill_product_operand(lhs);
    fill_product_operand(rhs);
    // a one-double offset prevents the result view from satisfying sixteen-byte SIMD alignment
    ASSERT_EQ(reinterpret_cast<std::uintptr_t>(result.data()) % 16, sizeof(double));
    const MatrixView<const double, 3, 5, ColMajor> const_lhs(lhs.data());
    const MatrixView<const double, 5, 7> const_rhs(rhs.data());
    result = const_lhs * const_rhs;
    // external operands and output retain their logical coordinates despite unaligned physical storage
    expect_ordered_product(result, lhs, rhs);
    // the output prefix sentinel detects writes before the mapped coefficient range
    EXPECT_DOUBLE_EQ(result_storage.front(), -97.0);
    // the output suffix sentinel detects writes beyond its odd-sized mapped range
    EXPECT_DOUBLE_EQ(result_storage.back(), -97.0);
    // the left input prefix and suffix detect writes into read-only product operands
    EXPECT_TRUE(lhs_storage.front() == -71.0 && lhs_storage.back() == -71.0);
    // the right input prefix and suffix detect writes into read-only product operands
    EXPECT_TRUE(rhs_storage.front() == -83.0 && rhs_storage.back() == -83.0);
}

// checks owner aliases and overlapping external views against copies made before product assignment
TEST(ContiguousProduct, AliasedAssignmentsReadOriginalOperands) {
    Matrix<double, Dynamic, Dynamic> lhs(3, 5), rhs(5, 5);
    fill_product_operand(lhs);
    fill_product_operand(rhs);
    const auto original_lhs = lhs;
    lhs = lhs * rhs;
    // owner assignment snapshots the product before replacing its left input
    expect_ordered_product(lhs, original_lhs, rhs);

    std::array<double, 17> storage;
    MatrixView<double, 3, 5, ColMajor> source(storage.data());
    MatrixView<double, 3, 5, ColMajor> destination(storage.data() + 1);
    fill_product_operand(source);
    storage.back() = -101.0;
    const Matrix<double, 3, 5, ColMajor> original_source(source);
    destination = source * rhs;
    // overlap assignment uses the original left coefficients at every coordinate
    expect_ordered_product(destination, original_source, rhs);
    // the scalar after the destination range remains outside the overlapping product write
    EXPECT_DOUBLE_EQ(storage.back(), -101.0);
}

// checks nested expression, strided-block, integral and mixed-precision products through generic coefficient evaluation
TEST(ContiguousProduct, GenericOperandsRetainCoordinateEvaluationAndScalarPromotion) {
    Matrix<double, Dynamic, Dynamic> lhs(3, 5), rhs(5, 7);
    fill_product_operand(lhs);
    fill_product_operand(rhs);
    const auto doubled = lhs + lhs;
    const Matrix<double, Dynamic, Dynamic> nested = doubled * rhs;
    // an expression operand evaluates its doubled coefficients before the scalar dot-product oracle
    expect_ordered_product(nested, doubled, rhs);
    const auto product = lhs * rhs;
    const Matrix<double, Dynamic, Dynamic> nested_result = product + product;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 7; ++j) {
            double expected = 0;
            for (int k = 0; k < 5; ++k) expected += lhs(i, k) * rhs(k, j);
            // a product nested inside a sum retains the original per-coefficient expression evaluation
            EXPECT_DOUBLE_EQ(nested_result(i, j), 2.0 * expected);
        }
    }
    const auto block = rhs.block(0, 1, 5, 5);
    const Matrix<double, Dynamic, Dynamic> block_result = lhs * block;
    // a strided right block follows coordinate access instead of assuming packed plain storage
    expect_ordered_product(block_result, lhs, block);

    Matrix<int, 3, 5> integers;
    for (int i = 0; i < 3; ++i) {
        for (int k = 0; k < 5; ++k) integers(i, k) = i - 2 * k;
    }
    const Matrix<int, 5, 2> integer_rhs({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
    const Matrix<int, 3, 2> integral_result = integers * integer_rhs;
    // integer arithmetic retains its exact scalar accumulation rather than taking floating-point kernels
    expect_ordered_product(integral_result, integers, integer_rhs);
    const Matrix<float, 3, 5> floats(lhs);
    const Matrix<double, 3, 7> mixed_result = floats * rhs;
    // mixed float and double operands accumulate in their promoted double scalar type
    expect_ordered_product(mixed_result, floats, rhs);
}

// checks products copied into vectors with the opposite logical row or column orientation
TEST(ContiguousProduct, VectorResultsPreserveCoefficientOrderAcrossOrientations) {
    const Matrix<double, 1, 3> left_row({1, 2, 3});
    Matrix<double, 3, 5> right_matrix;
    fill_product_operand(right_matrix);
    const Vector<double, Dynamic> column_result(left_row * right_matrix);
    for (int j = 0; j < 5; ++j) {
        double expected = 0;
        for (int k = 0; k < 3; ++k) expected += left_row(0, k) * right_matrix(k, j);
        // a row-shaped product copied into a column vector retains the ordered column-coordinate oracle
        EXPECT_DOUBLE_EQ(column_result[j], expected);
    }
    Matrix<double, 5, 3> left_matrix;
    fill_product_operand(left_matrix);
    const Vector<double, 3> right_column({1, 2, 3});
    const Matrix<double, 1, Dynamic> row_result(left_matrix * right_column);
    for (int i = 0; i < 5; ++i) {
        double expected = 0;
        for (int k = 0; k < 3; ++k) expected += left_matrix(i, k) * right_column(k, 0);
        // a column-shaped product copied into a row vector retains the ordered row-coordinate oracle
        EXPECT_DOUBLE_EQ(row_result[i], expected);
    }
}

// checks static materialization, floating-point reduction order and existing specialized product executors
TEST(ContiguousProduct, ConstexprReductionOrderAndSpecializedExecutors) {
    constexpr auto fixed = [] {
        const Matrix<double, 2, 3, ColMajor> lhs({1, 2, 3, 4, 5, 6});
        const Matrix<double, 3, 2, ColMajor> rhs({1, 2, 3, 4, 5, 6});
        return Matrix<double, 2, 2, ColMajor>(lhs * rhs);
    }();
    // constant-evaluated materialization preserves the first and last explicit dot products
    static_assert(fixed(0, 0) == 22.0 && fixed(1, 1) == 64.0);

    const Matrix<double, 2, 5> cancellation({1.0e16, 1.0, -1.0e16, 2.0, 3.0, -1.0e16, 1.0, 1.0e16, 2.0, 3.0});
    const Matrix<double, 5, 3> ones(1.0);
    const Matrix<double, 2, 3> ordered = cancellation * ones;
    // cancellation-sensitive dot products retain increasing-k addition order without reassociation
    expect_ordered_product(ordered, cancellation, ones);

    const Vector<double, 3> column({2, 3, 4});
    const Matrix<double, 1, 5> row({1, 2, 3, 4, 5});
    const Matrix<double, 3, 5> outer = column * row;
    // column-times-row dispatch preserves the outer-product executor's coordinate values
    expect_ordered_product(outer, column, row);
    Matrix<double, 3, 5> dense;
    fill_product_operand(dense);
    const auto diagonal = column.as_diagonal();
    const Matrix<double, 3, 5> diagonal_result = diagonal * dense;
    // a diagonal operand retains its specialized product executor instead of reading packed storage as dense
    expect_ordered_product(diagonal_result, diagonal, dense);
    const Matrix<double, 3, 3> square({1, 2, 3, 4, 5, 6, 7, 8, 9});
    const auto lower = square.triangular_block<Lower>();
    const Matrix<double, 3, 5> triangular_result = lower * dense;
    // triangular masking remains active while the product executor visits only valid coefficients
    expect_ordered_product(triangular_result, lower, dense);
    dense(0, 0) = std::numeric_limits<double>::quiet_NaN();
    const Matrix<double, 3, 5> masked = diagonal * dense;
    // the diagonal executor skips masked zeros so an unrelated nan cannot contaminate a finite row
    EXPECT_DOUBLE_EQ(masked(1, 0), column[1] * dense(1, 0));
}

}   // namespace fdapde
