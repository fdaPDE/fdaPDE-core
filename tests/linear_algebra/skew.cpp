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

// a temporary dense owner cannot lend a skew-symmetric wrapper
static_assert(!permits_temporary_skew<lifetime_matrix>);
// a const temporary dense owner cannot lend a skew-symmetric wrapper
static_assert(!permits_const_temporary_skew<lifetime_matrix>);
// a stored expression can safely retain a skew-symmetric wrapper
static_assert(permits_expression_skew<lifetime_matrix>);
// a fixed nonempty skew view requires an explicit storage binding
static_assert(!std::is_default_constructible_v<SkewSymmetricMatrixView<double, 3, 3>>);
// a dynamic skew view permits an initially empty binding
static_assert(std::is_default_constructible_v<SkewSymmetricMatrixView<double, Dynamic, Dynamic>>);
// a const-storage skew view advertises read-only access
static_assert(const_skew_view::ReadOnly == 1);
// a const-storage skew view rejects coefficient writes
static_assert(!permits_coefficient_write<const_skew_view>);
// a temporary skew owner cannot expose a dangling derived reference
static_assert(!exposes_owning_rvalue_derived<skew_matrix>);
// a temporary skew owner cannot return a borrow through copy assignment
static_assert(!permits_owning_rvalue_copy_assignment<skew_matrix>);
// addition cannot retain a temporary skew owner by reference
static_assert(!permits_temporary_skew_addition<skew_matrix>);
// scaling cannot retain a temporary skew owner by reference
static_assert(!permits_temporary_skew_scaling<skew_matrix>);
// matrix compound multiplication is unavailable because it need not preserve skew symmetry
static_assert(!permits_matrix_compound_multiply<skew_matrix>);
// scalar compound multiplication remains available because it preserves skew symmetry
static_assert(permits_scalar_compound_multiply<skew_matrix>);
// the skew-symmetric trait recognizes a const reference to an owner
static_assert(is_skew_symmetric_matrix_v<const skew_matrix&>);

template <typename Actual, typename Expected>
void expect_matrix_near(const Actual& actual, const Expected& expected, double tolerance = 1.0e-12) {
    // coefficient comparison requires matching row counts
    ASSERT_EQ(actual.rows(), expected.rows());
    // coefficient comparison requires matching column counts
    ASSERT_EQ(actual.cols(), expected.cols());
    for (int i = 0; i < actual.rows(); ++i) {
        for (int j = 0; j < actual.cols(); ++j) {
            // each coefficient agrees with the reference within the supplied absolute tolerance
            EXPECT_NEAR(static_cast<double>(actual(i, j)), static_cast<double>(expected(i, j)), tolerance);
        }
    }
}

template <int StorageOrder> void check_dense_skew_views() {
    using matrix_type = Matrix<double, 3, 3, StorageOrder>;
    matrix_type dense({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0});
    const auto upper = dense.template as_skew_symmetric<Upper>();
    const auto lower = dense.template as_skew_symmetric<Lower>();
    // a dense-backed skew wrapper retains the operand's storage order
    static_assert(decltype(upper)::StorageOrder == StorageOrder);
    // a skew wrapper borrowed from const storage is read-only
    static_assert(decltype(lower)::ReadOnly == 1);
    // a const dense-backed skew wrapper rejects coefficient writes
    static_assert(!permits_coefficient_write<decltype(lower)>);
    expect_matrix_near(upper, matrix_type({0.0, 2.0, 3.0, -2.0, 0.0, 6.0, -3.0, -6.0, 0.0}));
    expect_matrix_near(lower, matrix_type({0.0, -4.0, -7.0, 4.0, 0.0, -8.0, 7.0, 8.0, 0.0}));

    const auto temporary_upper = (dense + dense).template as_skew_symmetric<Upper>();
    const auto temporary_lower = (dense + dense).template as_skew_symmetric<Lower>();
    expect_matrix_near(temporary_upper, matrix_type({0.0, 4.0, 6.0, -4.0, 0.0, 12.0, -6.0, -12.0, 0.0}));
    expect_matrix_near(temporary_lower, matrix_type({0.0, -8.0, -14.0, 8.0, 0.0, -16.0, 14.0, 16.0, 0.0}));

    // runtime indices avoid GCC 14 diagnosing the unreachable access after the throwing guard
    volatile int negative_row = -1;
    volatile int past_last_row = 3;
    // skew-wrapper access rejects a negative row
    EXPECT_THROW(static_cast<void>(upper(negative_row, 0)), std::out_of_range);
    // const skew-wrapper access rejects a row equal to its dimension
    EXPECT_THROW(static_cast<void>(lower(past_last_row, 0)), std::out_of_range);
    Matrix<double, Dynamic, Dynamic, StorageOrder> rectangular(2, 3);
    // skew wrapping rejects a rectangular dense matrix
    EXPECT_THROW(static_cast<void>(rectangular.template as_skew_symmetric<Upper>()), std::invalid_argument);
}

void check_packed_skew_contracts() {
    const double packed_data[] = {1.0, 2.0, 3.0};
    skew_matrix matrix(packed_data);
    const Matrix<double, 3, 3> expected({0.0, 1.0, 2.0, -1.0, 0.0, 3.0, -2.0, -3.0, 0.0});
    // a fixed 3-by-3 skew matrix stores only three independent coefficients
    static_assert(skew_matrix::StorageSize == 3);
    // the runtime packed size agrees with the three independent upper entries
    EXPECT_EQ(matrix.storage_size(), 3);
    expect_matrix_near(matrix, expected);

    matrix(2, 0) = 7.0;
    // writing a reflected coefficient stores its negation in the packed upper triangle
    EXPECT_DOUBLE_EQ(matrix.data()[1], -7.0);
    // the upper entry reads the negative value stored by the reflected write
    EXPECT_DOUBLE_EQ(static_cast<double>(matrix(0, 2)), -7.0);
    // the lower entry reads the positive reflection of the packed upper entry
    EXPECT_DOUBLE_EQ(static_cast<double>(matrix(2, 0)), 7.0);
    matrix(0, 1) = 5.0;
    // compound proxy assignment updates the corresponding packed coefficient
    EXPECT_DOUBLE_EQ(matrix.data()[0], 5.0);
    matrix(1, 1) = 0.0;
    const std::array<double, 3> before_bad_diagonal {matrix.data()[0], matrix.data()[1], matrix.data()[2]};
    // writing a nonzero diagonal value is rejected to preserve skew symmetry
    EXPECT_THROW(matrix(1, 1) = 1.0, std::invalid_argument);
    // failed diagonal assignment leaves every packed coefficient unchanged
    for (int i = 0; i < 3; ++i) EXPECT_DOUBLE_EQ(matrix.data()[i], before_bad_diagonal[static_cast<std::size_t>(i)]);

    const skew_matrix source(packed_data);
    const auto sum = source + source;
    const auto difference = sum - source;
    const auto left_scaled = 2.0 * difference;
    const auto right_scaled = difference * 3.0;
    const auto divided = right_scaled / 3.0;
    // addition of skew matrices retains the skew-symmetric expression tag
    static_assert(is_skew_symmetric_matrix_v<decltype(sum)>);
    // subtraction of skew matrices retains the skew-symmetric expression tag
    static_assert(is_skew_symmetric_matrix_v<decltype(difference)>);
    // left scalar multiplication retains the skew-symmetric expression tag
    static_assert(is_skew_symmetric_matrix_v<decltype(left_scaled)>);
    // right scalar multiplication retains the skew-symmetric expression tag
    static_assert(is_skew_symmetric_matrix_v<decltype(right_scaled)>);
    // scalar division retains the skew-symmetric expression tag
    static_assert(is_skew_symmetric_matrix_v<decltype(divided)>);
    expect_matrix_near(difference, source);
    expect_matrix_near(left_scaled, Matrix<double, 3, 3>(source * 2.0));
    expect_matrix_near(divided, source);

    const auto product = source * source;
    // a general product of skew matrices does not claim skew symmetry
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
    // assignment into the first view retains its external storage binding
    EXPECT_EQ(first_view.data(), first_address);
    expect_matrix_near(first_view, source);

    std::array<double, 3> second_storage {};
    SkewSymmetricMatrixView<double, 3, 3> second_view(second_storage.data());
    second_view = first_view;
    // view-to-view assignment retains the second view's external storage binding
    EXPECT_EQ(second_view.data(), second_storage.data());
    expect_matrix_near(second_view, first_view);

    std::array<double, 3> temporary_storage {};
    const auto temporary_view = SkewSymmetricMatrixView<double, 3, 3>(temporary_storage.data()) = first_view;
    // assignment from a temporary view retains the destination storage binding
    EXPECT_EQ(temporary_view.data(), temporary_storage.data());
    expect_matrix_near(temporary_view, first_view);

    std::array<double, 4> overlap_storage {1.0, 2.0, 3.0, 0.0};
    SkewSymmetricMatrixView<double, 3, 3> overlap_source(overlap_storage.data());
    SkewSymmetricMatrixView<double, 3, 3> overlap_destination(overlap_storage.data() + 1);
    overlap_destination = overlap_source;
    // overlapping view assignment retains the destination's original offset
    EXPECT_EQ(overlap_destination.data(), overlap_storage.data() + 1);
    expect_matrix_near(overlap_destination, expected);

    const SkewSymmetricMatrixView<const double, 3, 3> const_view(first_storage.data());
    // a view of const packed storage advertises read-only access
    static_assert(decltype(const_view)::ReadOnly == 1);
    // a const view reads a lower entry as the negation of the stored upper entry
    EXPECT_DOUBLE_EQ(static_cast<double>(const_view(2, 0)), -2.0);
}

void check_dynamic_skew_contracts() {
    SkewSymmetricMatrix<double, Dynamic, Dynamic> dynamic(std::vector<double> {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    // resizing the dynamic skew owner updates its row count to four
    EXPECT_EQ(dynamic.rows(), 4);
    // resizing the dynamic skew owner keeps it square
    EXPECT_EQ(dynamic.cols(), 4);
    // four dimensions require six independent skew coefficients
    EXPECT_EQ(dynamic.storage_size(), 6);
    expect_matrix_near(
      dynamic,
      Matrix<double, 4, 4>({0.0, 1.0, 2.0, 3.0, -1.0, 0.0, 4.0, 5.0, -2.0, -4.0, 0.0, 6.0, -3.0, -5.0, -6.0, 0.0}));

    dynamic.resize(3, 3);
    // assignment resizes the dynamic skew owner back to three rows
    EXPECT_EQ(dynamic.rows(), 3);
    // three dimensions restore the packed size to three
    EXPECT_EQ(dynamic.storage_size(), 3);
    expect_matrix_near(dynamic, Matrix<double, 3, 3>({0.0, 1.0, 2.0, -1.0, 0.0, 3.0, -2.0, -3.0, 0.0}));

    const SkewSymmetricMatrix<double, 3, Dynamic> partial_default;
    const SkewSymmetricMatrix<double, Dynamic, 3> other_partial_default;
    // fixed columns determine the default partially dynamic owner's rows
    EXPECT_EQ(partial_default.rows(), 3);
    // fixed columns retain their declared extent in the default owner
    EXPECT_EQ(partial_default.cols(), 3);
    // fixed rows retain their declared extent in the default owner
    EXPECT_EQ(other_partial_default.rows(), 3);
    // fixed rows determine the default partially dynamic owner's columns
    EXPECT_EQ(other_partial_default.cols(), 3);

    std::array<double, 3> view_storage {4.0, 5.0, 6.0};
    SkewSymmetricMatrixView<double, Dynamic, Dynamic> dynamic_view(view_storage.data(), 3, 3);
    // a dynamic view reads the sign-correct reflection of its packed coefficient
    EXPECT_DOUBLE_EQ(static_cast<double>(dynamic_view(2, 1)), -6.0);
    dynamic_view = dynamic;
    // assignment preserves the dynamic view's external storage address
    EXPECT_EQ(dynamic_view.data(), view_storage.data());
    expect_matrix_near(dynamic_view + dynamic_view, Matrix<double, 3, 3>(dynamic + dynamic));

    const SkewSymmetricMatrix<double, 1, 1> singleton;
    // a fixed singleton skew matrix needs no stored coefficients
    EXPECT_EQ(singleton.storage_size(), 0);
    // a fixed singleton skew matrix exposes an implicit zero diagonal
    EXPECT_DOUBLE_EQ(static_cast<double>(singleton(0, 0)), 0.0);
    const SkewSymmetricMatrix<double, Dynamic, Dynamic> dynamic_singleton(1, 1);
    // a dynamic singleton skew matrix needs no stored coefficients
    EXPECT_EQ(dynamic_singleton.storage_size(), 0);
    // a dynamic singleton skew matrix exposes an implicit zero diagonal
    EXPECT_DOUBLE_EQ(static_cast<double>(dynamic_singleton(0, 0)), 0.0);
    const SkewSymmetricMatrixView<double, 1, 1> singleton_view(static_cast<double*>(nullptr));
    // a singleton view can read its implicit diagonal without coefficient storage
    EXPECT_DOUBLE_EQ(static_cast<double>(singleton_view(0, 0)), 0.0);

    const SkewSymmetricMatrix<double, Dynamic, Dynamic> empty_owner;
    const SkewSymmetricMatrixView<double, Dynamic, Dynamic> empty_view;
    // an empty skew owner has zero rows
    EXPECT_EQ(empty_owner.rows(), 0);
    // an empty skew owner has zero columns
    EXPECT_EQ(empty_owner.cols(), 0);
    // an empty skew owner allocates no packed coefficients
    EXPECT_EQ(empty_owner.storage_size(), 0);
    // an empty skew view has zero rows
    EXPECT_EQ(empty_view.rows(), 0);
    // an empty skew view has zero columns
    EXPECT_EQ(empty_view.cols(), 0);
    // an empty skew view permits a null storage binding
    EXPECT_EQ(empty_view.data(), nullptr);

    // construction rejects a packed length that cannot represent a skew matrix
    EXPECT_THROW(
      static_cast<void>(SkewSymmetricMatrix<double, Dynamic, Dynamic>(std::vector<double> {1.0, 2.0})),
      std::invalid_argument);
    // construction rejects a nonsquare runtime shape
    EXPECT_THROW(static_cast<void>(SkewSymmetricMatrix<double, Dynamic, Dynamic>(2, 3)), std::invalid_argument);
    // construction rejects runtime dimensions inconsistent with a static axis
    EXPECT_THROW(static_cast<void>(SkewSymmetricMatrix<double, Dynamic, 3>(2, 2)), std::invalid_argument);
    // construction rejects dimensions whose packed size exceeds the supported range
    EXPECT_THROW(
      static_cast<void>(SkewSymmetricMatrix<double, Dynamic, Dynamic>(
        std::numeric_limits<int>::max(), std::numeric_limits<int>::max())),
      std::length_error);
    // a nonempty dynamic skew view rejects a null storage pointer
    EXPECT_THROW(
      static_cast<void>(SkewSymmetricMatrixView<double, Dynamic, Dynamic>(static_cast<double*>(nullptr), 3, 3)),
      std::invalid_argument);
    // a nonempty fixed skew view rejects a null storage pointer
    EXPECT_THROW(
      static_cast<void>(SkewSymmetricMatrixView<double, 3, 3>(static_cast<double*>(nullptr))), std::invalid_argument);
    // mutable skew access rejects a negative row
    EXPECT_THROW(static_cast<void>(dynamic(-1, 0)), std::out_of_range);
    // const skew access rejects a row equal to its dimension
    EXPECT_THROW(static_cast<void>(std::as_const(dynamic)(3, 0)), std::out_of_range);

    SkewSymmetricMatrix<double, Dynamic, Dynamic> resize_target(3, 3);
    resize_target(2, 0) = 7.0;
    // resize rejects a nonsquare target shape
    EXPECT_THROW(resize_target.resize(2, 3), std::invalid_argument);
    // failed resize preserves the original row count
    EXPECT_EQ(resize_target.rows(), 3);
    // failed resize preserves the original reflected coefficient
    EXPECT_DOUBLE_EQ(static_cast<double>(resize_target(2, 0)), 7.0);

    std::array<double, 1> assignment_storage {9.0};
    SkewSymmetricMatrixView<double, Dynamic, Dynamic> assignment_target(assignment_storage.data(), 2, 2);
    // view assignment rejects a source of incompatible dimension
    EXPECT_THROW(assignment_target = dynamic, std::invalid_argument);
    // failed view assignment preserves the target's row count
    EXPECT_EQ(assignment_target.rows(), 2);
    // failed view assignment preserves the target's packed coefficient
    EXPECT_DOUBLE_EQ(assignment_storage[0], 9.0);
}

}   // namespace

// exercise skew reflection, packed proxies, structure-preserving arithmetic and borrowed storage
TEST(linear_algebra, skew_symmetric) {
    check_dense_skew_views<RowMajor>();
    check_dense_skew_views<ColMajor>();
    check_packed_skew_contracts();
    check_dynamic_skew_contracts();
}
