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

#include <array>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

using namespace fdapde;

namespace {

template <typename MatrixType>
concept permits_temporary_skew = requires { MatrixType {}.template as_skew_symmetric<Upper>(); };

template <typename MatrixType>
concept permits_const_temporary_skew =
  requires(const MatrixType&& matrix) { std::move(matrix).template as_skew_symmetric<Upper>(); };

template <typename MatrixType>
concept permits_expression_skew =
  requires(MatrixType& lhs, MatrixType& rhs) { (lhs + rhs).template as_skew_symmetric<Upper>(); };

template <typename MatrixType>
concept permits_coefficient_write = requires(MatrixType& matrix) { matrix(0, 1) = 0.0; };

template <typename MatrixType>
concept exposes_owning_rvalue_derived = requires(MatrixType& matrix) { std::move(matrix).derived(); };

template <typename MatrixType>
concept permits_owning_rvalue_copy_assignment = requires(MatrixType& lhs, MatrixType& rhs) { std::move(lhs) = rhs; };

template <typename MatrixType>
concept permits_temporary_skew_addition = requires { MatrixType {} + MatrixType {}; };

template <typename MatrixType>
concept permits_temporary_skew_scaling = requires { MatrixType {} * 2.0; };

template <typename MatrixType>
concept permits_matrix_compound_multiply = requires(MatrixType& matrix) { matrix *= matrix; };

template <typename MatrixType>
concept permits_scalar_compound_multiply = requires(MatrixType& matrix) { matrix *= 2.0; };

using lifetime_matrix = Matrix<double, 3, 3>;
using skew_matrix = SkewSymmetricMatrix<double, 3, 3>;
using const_skew_view = SkewSymmetricMatrixView<const double, 3, 3>;

// checks at compile time: !permits_temporary_skew<lifetime_matrix>
static_assert(!permits_temporary_skew<lifetime_matrix>);
// checks at compile time: !permits_const_temporary_skew<lifetime_matrix>
static_assert(!permits_const_temporary_skew<lifetime_matrix>);
// checks at compile time: permits_expression_skew<lifetime_matrix>
static_assert(permits_expression_skew<lifetime_matrix>);
// checks at compile time: !std::is_default_constructible_v<SkewSymmetricMatrixView<double, 3, 3>>
static_assert(!std::is_default_constructible_v<SkewSymmetricMatrixView<double, 3, 3>>);
// checks at compile time: std::is_default_constructible_v<SkewSymmetricMatrixView<double, Dynamic, Dynamic>>
static_assert(std::is_default_constructible_v<SkewSymmetricMatrixView<double, Dynamic, Dynamic>>);
// checks at compile time: const_skew_view::ReadOnly == 1
static_assert(const_skew_view::ReadOnly == 1);
// checks at compile time: !permits_coefficient_write<const_skew_view>
static_assert(!permits_coefficient_write<const_skew_view>);
// checks at compile time: !exposes_owning_rvalue_derived<skew_matrix>
static_assert(!exposes_owning_rvalue_derived<skew_matrix>);
// checks at compile time: !permits_owning_rvalue_copy_assignment<skew_matrix>
static_assert(!permits_owning_rvalue_copy_assignment<skew_matrix>);
// checks at compile time: !permits_temporary_skew_addition<skew_matrix>
static_assert(!permits_temporary_skew_addition<skew_matrix>);
// checks at compile time: !permits_temporary_skew_scaling<skew_matrix>
static_assert(!permits_temporary_skew_scaling<skew_matrix>);
// checks at compile time: !permits_matrix_compound_multiply<skew_matrix>
static_assert(!permits_matrix_compound_multiply<skew_matrix>);
// checks at compile time: permits_scalar_compound_multiply<skew_matrix>
static_assert(permits_scalar_compound_multiply<skew_matrix>);
// checks at compile time: is_skew_symmetric_matrix_v<const skew_matrix&>
static_assert(is_skew_symmetric_matrix_v<const skew_matrix&>);

template <typename Actual, typename Expected>
void expect_matrix_near(const Actual& actual, const Expected& expected, double tolerance = 1.0e-12) {
    // compares actual.rows(), expected.rows() using eq semantics
    ASSERT_EQ(actual.rows(), expected.rows());
    // compares actual.cols(), expected.cols() using eq semantics
    ASSERT_EQ(actual.cols(), expected.cols());
    for (int i = 0; i < actual.rows(); ++i) {
        for (int j = 0; j < actual.cols(); ++j) {
            // compares the computed and expected values within the stated absolute tolerance
            EXPECT_NEAR(static_cast<double>(actual(i, j)), static_cast<double>(expected(i, j)), tolerance);
        }
    }
}

template <int StorageOrder> void check_dense_skew_views() {
    using matrix_type = Matrix<double, 3, 3, StorageOrder>;
    matrix_type dense({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0});
    const auto upper = dense.template as_skew_symmetric<Upper>();
    const auto lower = dense.template as_skew_symmetric<Lower>();
    // checks at compile time: decltype(upper)::StorageOrder == StorageOrder
    static_assert(decltype(upper)::StorageOrder == StorageOrder);
    // checks at compile time: decltype(lower)::ReadOnly == 1
    static_assert(decltype(lower)::ReadOnly == 1);
    // checks at compile time: !permits_coefficient_write<decltype(lower)>
    static_assert(!permits_coefficient_write<decltype(lower)>);
    expect_matrix_near(upper, matrix_type({0.0, 2.0, 3.0, -2.0, 0.0, 6.0, -3.0, -6.0, 0.0}));
    expect_matrix_near(lower, matrix_type({0.0, -4.0, -7.0, 4.0, 0.0, -8.0, 7.0, 8.0, 0.0}));

    const auto temporary_upper = (dense + dense).template as_skew_symmetric<Upper>();
    const auto temporary_lower = (dense + dense).template as_skew_symmetric<Lower>();
    expect_matrix_near(temporary_upper, matrix_type({0.0, 4.0, 6.0, -4.0, 0.0, 12.0, -6.0, -12.0, 0.0}));
    expect_matrix_near(temporary_lower, matrix_type({0.0, -8.0, -14.0, 8.0, 0.0, -16.0, 14.0, 16.0, 0.0}));

    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(upper(-1, 0)), std::out_of_range);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(lower(3, 0)), std::out_of_range);
    Matrix<double, Dynamic, Dynamic, StorageOrder> rectangular(2, 3);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(rectangular.template as_skew_symmetric<Upper>()), std::invalid_argument);
}

void check_packed_skew_contracts() {
    const double packed_data[] = {1.0, 2.0, 3.0};
    skew_matrix matrix(packed_data);
    const Matrix<double, 3, 3> expected({0.0, 1.0, 2.0, -1.0, 0.0, 3.0, -2.0, -3.0, 0.0});
    // checks at compile time: skew_matrix::StorageSize == 3
    static_assert(skew_matrix::StorageSize == 3);
    // compares matrix.storage_size(), 3 using eq semantics
    EXPECT_EQ(matrix.storage_size(), 3);
    expect_matrix_near(matrix, expected);

    matrix(2, 0) = 7.0;
    // compares matrix.data()[1], -7.0 using double_eq semantics
    EXPECT_DOUBLE_EQ(matrix.data()[1], -7.0);
    // compares static_cast<double>(matrix(0, 2)), -7.0 using double_eq semantics
    EXPECT_DOUBLE_EQ(static_cast<double>(matrix(0, 2)), -7.0);
    // compares static_cast<double>(matrix(2, 0)), 7.0 using double_eq semantics
    EXPECT_DOUBLE_EQ(static_cast<double>(matrix(2, 0)), 7.0);
    matrix(0, 1) = 5.0;
    // compares matrix.data()[0], 5.0 using double_eq semantics
    EXPECT_DOUBLE_EQ(matrix.data()[0], 5.0);
    matrix(1, 1) = 0.0;
    const std::array<double, 3> before_bad_diagonal {matrix.data()[0], matrix.data()[1], matrix.data()[2]};
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(matrix(1, 1) = 1.0, std::invalid_argument);
    // compares matrix.data()[i], before_bad_diagonal[static_cast<std::size_t>(i)] using double_eq semantics
    for (int i = 0; i < 3; ++i) EXPECT_DOUBLE_EQ(matrix.data()[i], before_bad_diagonal[static_cast<std::size_t>(i)]);

    const skew_matrix source(packed_data);
    const auto sum = source + source;
    const auto difference = sum - source;
    const auto left_scaled = 2.0 * difference;
    const auto right_scaled = difference * 3.0;
    const auto divided = right_scaled / 3.0;
    // checks at compile time: is_skew_symmetric_matrix_v<decltype(sum)>
    static_assert(is_skew_symmetric_matrix_v<decltype(sum)>);
    // checks at compile time: is_skew_symmetric_matrix_v<decltype(difference)>
    static_assert(is_skew_symmetric_matrix_v<decltype(difference)>);
    // checks at compile time: is_skew_symmetric_matrix_v<decltype(left_scaled)>
    static_assert(is_skew_symmetric_matrix_v<decltype(left_scaled)>);
    // checks at compile time: is_skew_symmetric_matrix_v<decltype(right_scaled)>
    static_assert(is_skew_symmetric_matrix_v<decltype(right_scaled)>);
    // checks at compile time: is_skew_symmetric_matrix_v<decltype(divided)>
    static_assert(is_skew_symmetric_matrix_v<decltype(divided)>);
    expect_matrix_near(difference, source);
    expect_matrix_near(left_scaled, Matrix<double, 3, 3>(source * 2.0));
    expect_matrix_near(divided, source);

    const auto product = source * source;
    // checks at compile time: !is_skew_symmetric_matrix_v<decltype(product)>
    static_assert(!is_skew_symmetric_matrix_v<decltype(product)>);
    expect_matrix_near(product, Matrix<double, 3, 3>(expected * expected));

    skew_matrix compounds(packed_data);
    compounds += source;
    compounds -= source;
    compounds *= 2.0;
    compounds /= 2.0;
    expect_matrix_near(compounds, source);

    std::array<double, 3> first_storage {};
    SkewSymmetricMatrixView<double, 3, 3> first_view(first_storage.data());
    const double* const first_address = first_view.data();
    first_view = source;
    // compares first_view.data(), first_address using eq semantics
    EXPECT_EQ(first_view.data(), first_address);
    expect_matrix_near(first_view, source);

    std::array<double, 3> second_storage {};
    SkewSymmetricMatrixView<double, 3, 3> second_view(second_storage.data());
    second_view = first_view;
    // compares second_view.data(), second_storage.data() using eq semantics
    EXPECT_EQ(second_view.data(), second_storage.data());
    expect_matrix_near(second_view, first_view);

    std::array<double, 3> temporary_storage {};
    const auto temporary_view = SkewSymmetricMatrixView<double, 3, 3>(temporary_storage.data()) = first_view;
    // compares temporary_view.data(), temporary_storage.data() using eq semantics
    EXPECT_EQ(temporary_view.data(), temporary_storage.data());
    expect_matrix_near(temporary_view, first_view);

    std::array<double, 4> overlap_storage {1.0, 2.0, 3.0, 0.0};
    SkewSymmetricMatrixView<double, 3, 3> overlap_source(overlap_storage.data());
    SkewSymmetricMatrixView<double, 3, 3> overlap_destination(overlap_storage.data() + 1);
    overlap_destination = overlap_source;
    // compares overlap_destination.data(), overlap_storage.data() + 1 using eq semantics
    EXPECT_EQ(overlap_destination.data(), overlap_storage.data() + 1);
    expect_matrix_near(overlap_destination, expected);

    const SkewSymmetricMatrixView<const double, 3, 3> const_view(first_storage.data());
    // checks at compile time: decltype(const_view)::ReadOnly == 1
    static_assert(decltype(const_view)::ReadOnly == 1);
    // compares static_cast<double>(const_view(2, 0)), -2.0 using double_eq semantics
    EXPECT_DOUBLE_EQ(static_cast<double>(const_view(2, 0)), -2.0);
}

void check_dynamic_skew_contracts() {
    SkewSymmetricMatrix<double, Dynamic, Dynamic> dynamic(std::vector<double> {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    // compares dynamic.rows(), 4 using eq semantics
    EXPECT_EQ(dynamic.rows(), 4);
    // compares dynamic.cols(), 4 using eq semantics
    EXPECT_EQ(dynamic.cols(), 4);
    // compares dynamic.storage_size(), 6 using eq semantics
    EXPECT_EQ(dynamic.storage_size(), 6);
    expect_matrix_near(
      dynamic,
      Matrix<double, 4, 4>({0.0, 1.0, 2.0, 3.0, -1.0, 0.0, 4.0, 5.0, -2.0, -4.0, 0.0, 6.0, -3.0, -5.0, -6.0, 0.0}));

    dynamic.resize(3, 3);
    // compares dynamic.rows(), 3 using eq semantics
    EXPECT_EQ(dynamic.rows(), 3);
    // compares dynamic.storage_size(), 3 using eq semantics
    EXPECT_EQ(dynamic.storage_size(), 3);
    expect_matrix_near(dynamic, Matrix<double, 3, 3>({0.0, 1.0, 2.0, -1.0, 0.0, 3.0, -2.0, -3.0, 0.0}));

    const SkewSymmetricMatrix<double, 3, Dynamic> partial_default;
    const SkewSymmetricMatrix<double, Dynamic, 3> other_partial_default;
    // compares partial_default.rows(), 3 using eq semantics
    EXPECT_EQ(partial_default.rows(), 3);
    // compares partial_default.cols(), 3 using eq semantics
    EXPECT_EQ(partial_default.cols(), 3);
    // compares other_partial_default.rows(), 3 using eq semantics
    EXPECT_EQ(other_partial_default.rows(), 3);
    // compares other_partial_default.cols(), 3 using eq semantics
    EXPECT_EQ(other_partial_default.cols(), 3);

    std::array<double, 3> view_storage {4.0, 5.0, 6.0};
    SkewSymmetricMatrixView<double, Dynamic, Dynamic> dynamic_view(view_storage.data(), 3, 3);
    // compares static_cast<double>(dynamic_view(2, 1)), -6.0 using double_eq semantics
    EXPECT_DOUBLE_EQ(static_cast<double>(dynamic_view(2, 1)), -6.0);
    dynamic_view = dynamic;
    // compares dynamic_view.data(), view_storage.data() using eq semantics
    EXPECT_EQ(dynamic_view.data(), view_storage.data());
    expect_matrix_near(dynamic_view + dynamic_view, Matrix<double, 3, 3>(dynamic + dynamic));

    const SkewSymmetricMatrix<double, 1, 1> singleton;
    // compares singleton.storage_size(), 0 using eq semantics
    EXPECT_EQ(singleton.storage_size(), 0);
    // compares static_cast<double>(singleton(0, 0)), 0.0 using double_eq semantics
    EXPECT_DOUBLE_EQ(static_cast<double>(singleton(0, 0)), 0.0);
    const SkewSymmetricMatrix<double, Dynamic, Dynamic> dynamic_singleton(1, 1);
    // compares dynamic_singleton.storage_size(), 0 using eq semantics
    EXPECT_EQ(dynamic_singleton.storage_size(), 0);
    // compares static_cast<double>(dynamic_singleton(0, 0)), 0.0 using double_eq semantics
    EXPECT_DOUBLE_EQ(static_cast<double>(dynamic_singleton(0, 0)), 0.0);
    const SkewSymmetricMatrixView<double, 1, 1> singleton_view(static_cast<double*>(nullptr));
    // compares static_cast<double>(singleton_view(0, 0)), 0.0 using double_eq semantics
    EXPECT_DOUBLE_EQ(static_cast<double>(singleton_view(0, 0)), 0.0);

    const SkewSymmetricMatrix<double, Dynamic, Dynamic> empty_owner;
    const SkewSymmetricMatrixView<double, Dynamic, Dynamic> empty_view;
    // compares empty_owner.rows(), 0 using eq semantics
    EXPECT_EQ(empty_owner.rows(), 0);
    // compares empty_owner.cols(), 0 using eq semantics
    EXPECT_EQ(empty_owner.cols(), 0);
    // compares empty_owner.storage_size(), 0 using eq semantics
    EXPECT_EQ(empty_owner.storage_size(), 0);
    // compares empty_view.rows(), 0 using eq semantics
    EXPECT_EQ(empty_view.rows(), 0);
    // compares empty_view.cols(), 0 using eq semantics
    EXPECT_EQ(empty_view.cols(), 0);
    // compares empty_view.data(), nullptr using eq semantics
    EXPECT_EQ(empty_view.data(), nullptr);

    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(
      static_cast<void>(SkewSymmetricMatrix<double, Dynamic, Dynamic>(std::vector<double> {1.0, 2.0})),
      std::invalid_argument);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(SkewSymmetricMatrix<double, Dynamic, Dynamic>(2, 3)), std::invalid_argument);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(SkewSymmetricMatrix<double, Dynamic, 3>(2, 2)), std::invalid_argument);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(
      static_cast<void>(SkewSymmetricMatrix<double, Dynamic, Dynamic>(
        std::numeric_limits<int>::max(), std::numeric_limits<int>::max())),
      std::length_error);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(
      static_cast<void>(SkewSymmetricMatrixView<double, Dynamic, Dynamic>(static_cast<double*>(nullptr), 3, 3)),
      std::invalid_argument);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(
      static_cast<void>(SkewSymmetricMatrixView<double, 3, 3>(static_cast<double*>(nullptr))), std::invalid_argument);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(dynamic(-1, 0)), std::out_of_range);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(std::as_const(dynamic)(3, 0)), std::out_of_range);

    SkewSymmetricMatrix<double, Dynamic, Dynamic> resize_target(3, 3);
    resize_target(2, 0) = 7.0;
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(resize_target.resize(2, 3), std::invalid_argument);
    // compares resize_target.rows(), 3 using eq semantics
    EXPECT_EQ(resize_target.rows(), 3);
    // compares static_cast<double>(resize_target(2, 0)), 7.0 using double_eq semantics
    EXPECT_DOUBLE_EQ(static_cast<double>(resize_target(2, 0)), 7.0);

    std::array<double, 1> assignment_storage {9.0};
    SkewSymmetricMatrixView<double, Dynamic, Dynamic> assignment_target(assignment_storage.data(), 2, 2);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(assignment_target = dynamic, std::invalid_argument);
    // compares assignment_target.rows(), 2 using eq semantics
    EXPECT_EQ(assignment_target.rows(), 2);
    // compares assignment_storage[0], 9.0 using double_eq semantics
    EXPECT_DOUBLE_EQ(assignment_storage[0], 9.0);
}

}   // namespace

// verifies skew symmetric through the public algebra API
TEST(linear_algebra, skew_symmetric) {
    check_dense_skew_views<RowMajor>();
    check_dense_skew_views<ColMajor>();
    check_packed_skew_contracts();
    check_dynamic_skew_contracts();
}
