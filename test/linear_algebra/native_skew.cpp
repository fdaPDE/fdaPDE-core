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

#include <fdaPDE/linear_algebra.h>
#include <gtest/gtest.h>

#include <type_traits>
#include <utility>
#include <vector>

namespace {

namespace native = fdapde::linalg;

template <typename Matrix>
concept permits_temporary_skew = requires { Matrix {}.template as_skew_symmetric<native::Upper>(); };

template <typename Matrix>
concept permits_expression_skew =
  requires(Matrix& lhs, Matrix& rhs) { (lhs + rhs).template as_skew_symmetric<native::Upper>(); };

template <typename Matrix>
concept permits_coefficient_write = requires(Matrix& matrix) { matrix(0, 1) = 0.0; };

template <typename Matrix>
concept exposes_owning_rvalue_derived = requires(Matrix& matrix) { std::move(matrix).derived(); };

template <typename Matrix>
concept permits_owning_rvalue_copy_assignment = requires(Matrix& lhs, Matrix& rhs) { std::move(lhs) = rhs; };

template <typename Matrix>
concept permits_temporary_skew_addition = requires { Matrix {} + Matrix {}; };

template <typename Matrix>
concept permits_temporary_skew_scaling = requires { Matrix {} * 2.0; };

template <typename Matrix>
concept permits_matrix_compound_multiply = requires(Matrix& matrix) { matrix *= matrix; };

template <typename Matrix>
concept permits_scalar_compound_multiply = requires(Matrix& matrix) { matrix *= 2.0; };

using lifetime_matrix = native::Matrix<double, 3, 3>;
using skew_matrix = native::SkewSymmetricMatrix<double, 3>;
static_assert(!permits_temporary_skew<lifetime_matrix>);
static_assert(permits_expression_skew<lifetime_matrix>);
static_assert(!std::is_default_constructible_v<native::SkewSymmetricMatrixView<double, 3>>);
static_assert(
  std::is_default_constructible_v<native::SkewSymmetricMatrixView<double, fdapde::Dynamic, fdapde::Dynamic>>);
static_assert(!permits_coefficient_write<native::SkewSymmetricMatrixView<const double, 3>>);
static_assert(!exposes_owning_rvalue_derived<skew_matrix>);
static_assert(!permits_owning_rvalue_copy_assignment<skew_matrix>);
static_assert(!permits_temporary_skew_addition<skew_matrix>);
static_assert(!permits_temporary_skew_scaling<skew_matrix>);
static_assert(!permits_matrix_compound_multiply<skew_matrix>);
static_assert(permits_scalar_compound_multiply<skew_matrix>);
static_assert(native::is_skew_symmetric_matrix_v<const skew_matrix&>);

template <typename Actual, typename Expected>
void expect_matrix_near(const Actual& actual, const Expected& expected, double tolerance = 1.0e-12) {
    ASSERT_EQ(actual.rows(), expected.rows());
    ASSERT_EQ(actual.cols(), expected.cols());
    for (int i = 0; i < actual.rows(); ++i) {
        for (int j = 0; j < actual.cols(); ++j) {
            EXPECT_NEAR(static_cast<double>(actual(i, j)), static_cast<double>(expected(i, j)), tolerance);
        }
    }
}

TEST(NativeSkewSymmetricMatrix, PackedStorageMirroringAndViews) {
    const double packed[] = {1.0, 2.0, 3.0};
    skew_matrix matrix(packed);
    static_assert(skew_matrix::StorageSize == 3);
    EXPECT_EQ(matrix.storage_size(), 3);
    EXPECT_EQ(
      (native::Matrix<double, 3, 3>(matrix)),
      (native::Matrix<double, 3, 3>({0.0, 1.0, 2.0, -1.0, 0.0, 3.0, -2.0, -3.0, 0.0})));

    matrix(2, 0) = 7.0;
    EXPECT_DOUBLE_EQ(matrix.data()[1], -7.0);
    EXPECT_DOUBLE_EQ(static_cast<double>(matrix(0, 2)), -7.0);
    EXPECT_DOUBLE_EQ(static_cast<double>(matrix(2, 0)), 7.0);
    matrix(0, 1) = 5.0;
    EXPECT_DOUBLE_EQ(matrix.data()[0], 5.0);
    matrix(1, 1) = 0.0;
    EXPECT_DOUBLE_EQ(static_cast<double>(matrix(1, 1)), 0.0);
    matrix += matrix;
    matrix -= matrix;
    matrix += skew_matrix(packed);
    matrix *= 2.0;
    matrix /= 2.0;
    EXPECT_EQ(matrix, skew_matrix(packed));

    double view_storage[] = {9.0, 8.0, 7.0};
    native::SkewSymmetricMatrixView<double, 3> view(view_storage);
    double* const view_address = view.data();
    view = matrix;
    EXPECT_EQ(view.data(), view_address);
    EXPECT_EQ(view, matrix);

    double second_storage[] = {0.0, 0.0, 0.0};
    native::SkewSymmetricMatrixView<double, 3> second_view(second_storage);
    second_view = view;
    EXPECT_EQ(second_view.data(), second_storage);
    EXPECT_EQ(second_view, view);

    native::SkewSymmetricMatrixView<const double, 3> const_view(view_storage);
    static_assert(decltype(const_view)::ReadOnly == 1);
    EXPECT_DOUBLE_EQ(static_cast<double>(const_view(2, 0)), -2.0);
}

TEST(NativeSkewSymmetricMatrix, CastsArithmeticAndExpressionLifetimes) {
    native::Matrix<double, 3, 3> dense({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0});
    const auto upper = dense.template as_skew_symmetric<native::Upper>();
    const auto lower = dense.template as_skew_symmetric<native::Lower>();
    EXPECT_EQ(
      (native::Matrix<double, 3, 3>(upper)),
      (native::Matrix<double, 3, 3>({0.0, 2.0, 3.0, -2.0, 0.0, 6.0, -3.0, -6.0, 0.0})));
    EXPECT_EQ(
      (native::Matrix<double, 3, 3>(lower)),
      (native::Matrix<double, 3, 3>({0.0, -4.0, -7.0, 4.0, 0.0, -8.0, 7.0, 8.0, 0.0})));

    native::Matrix<double, 3, 3, native::ColMajor> column_major_dense(
      {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0});
    const auto column_major_upper = column_major_dense.template as_skew_symmetric<native::Upper>();
    static_assert(decltype(column_major_upper)::StorageOrder == native::ColMajor);
    EXPECT_EQ(
      (native::Matrix<double, 3, 3>(column_major_upper)),
      (native::Matrix<double, 3, 3>({0.0, 2.0, 3.0, -2.0, 0.0, 6.0, -3.0, -6.0, 0.0})));

    const auto temporary = (dense + dense).template as_skew_symmetric<native::Upper>();
    EXPECT_EQ(
      (native::Matrix<double, 3, 3>(temporary)),
      (native::Matrix<double, 3, 3>({0.0, 4.0, 6.0, -4.0, 0.0, 12.0, -6.0, -12.0, 0.0})));

    const skew_matrix matrix(upper);
    skew_matrix assigned;
    assigned = upper;
    EXPECT_EQ(assigned, matrix);
    const auto sum = matrix + matrix;
    const auto difference = sum - matrix;
    const auto left_scaled = 2.0 * difference;
    const auto right_scaled = difference * 3.0;
    const auto divided = right_scaled / 3.0;
    static_assert(native::is_skew_symmetric_matrix_v<decltype(sum)>);
    static_assert(native::is_skew_symmetric_matrix_v<decltype(difference)>);
    static_assert(native::is_skew_symmetric_matrix_v<decltype(left_scaled)>);
    static_assert(native::is_skew_symmetric_matrix_v<decltype(right_scaled)>);
    static_assert(native::is_skew_symmetric_matrix_v<decltype(divided)>);
    expect_matrix_near(difference, matrix);
    expect_matrix_near(left_scaled, 2.0 * matrix);
    expect_matrix_near(divided, matrix);

    const auto nested = (matrix + matrix) + (matrix - matrix);
    expect_matrix_near(nested, 2.0 * matrix);

    const auto product = matrix * matrix;
    static_assert(!native::is_skew_symmetric_matrix_v<decltype(product)>);
    EXPECT_EQ(
      (native::Matrix<double, 3, 3>(product)),
      (native::Matrix<double, 3, 3>({-13.0, -18.0, 12.0, -18.0, -40.0, -6.0, 12.0, -6.0, -45.0})));
}

TEST(NativeSkewSymmetricMatrix, DynamicAndPartialDynamicShapes) {
    native::SkewSymmetricMatrix<double, fdapde::Dynamic> dynamic(std::vector<double> {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    EXPECT_EQ(dynamic.rows(), 4);
    EXPECT_EQ(dynamic.cols(), 4);
    EXPECT_EQ(dynamic.storage_size(), 6);
    EXPECT_EQ(
      (native::Matrix<double, 4, 4>(dynamic)),
      (native::Matrix<double, 4, 4>(
        {0.0, 1.0, 2.0, 3.0, -1.0, 0.0, 4.0, 5.0, -2.0, -4.0, 0.0, 6.0, -3.0, -5.0, -6.0, 0.0})));

    dynamic.resize(3);
    EXPECT_EQ(dynamic.rows(), 3);
    EXPECT_EQ(dynamic.storage_size(), 3);
    EXPECT_EQ(
      (native::Matrix<double, 3, 3>(dynamic)),
      (native::Matrix<double, 3, 3>({0.0, 1.0, 2.0, -1.0, 0.0, 3.0, -2.0, -3.0, 0.0})));

    native::SkewSymmetricMatrix<double, 3, fdapde::Dynamic> partial_default;
    EXPECT_EQ(partial_default.rows(), 3);
    EXPECT_EQ(partial_default.cols(), 3);
    EXPECT_EQ(partial_default.storage_size(), 3);
    native::SkewSymmetricMatrix<double, fdapde::Dynamic, 3> other_partial_default;
    EXPECT_EQ(other_partial_default.rows(), 3);
    EXPECT_EQ(other_partial_default.cols(), 3);
    EXPECT_EQ(other_partial_default.storage_size(), 3);

    double view_storage[] = {4.0, 5.0, 6.0};
    native::SkewSymmetricMatrixView<double, fdapde::Dynamic, fdapde::Dynamic> dynamic_view(view_storage, 3);
    EXPECT_EQ(dynamic_view.rows(), 3);
    EXPECT_EQ(dynamic_view.cols(), 3);
    EXPECT_DOUBLE_EQ(static_cast<double>(dynamic_view(2, 1)), -6.0);
    dynamic_view = dynamic;
    EXPECT_EQ(dynamic_view.data(), view_storage);
    EXPECT_EQ(dynamic_view, dynamic);

    native::SkewSymmetricMatrix<double, 1> singleton;
    EXPECT_EQ(singleton.storage_size(), 0);
    EXPECT_DOUBLE_EQ(static_cast<double>(singleton(0, 0)), 0.0);
    native::SkewSymmetricMatrix<double, fdapde::Dynamic> dynamic_singleton(1);
    EXPECT_EQ(dynamic_singleton.storage_size(), 0);
    EXPECT_DOUBLE_EQ(static_cast<double>(dynamic_singleton(0, 0)), 0.0);

    native::SkewSymmetricMatrix<double, fdapde::Dynamic> resized_assignment(2);
    resized_assignment = dynamic;
    EXPECT_EQ(resized_assignment.rows(), 3);
    EXPECT_EQ(resized_assignment, dynamic);
}

}   // namespace
