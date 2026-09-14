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
#include <type_traits>
#include <utility>
#include <vector>

using namespace fdapde;

namespace {

template <typename MatrixType>
concept permits_temporary_symmetric = requires { MatrixType {}.template as_symmetric<Lower>(); };

template <typename MatrixType>
concept permits_const_temporary_symmetric =
  requires(const MatrixType&& matrix) { std::move(matrix).template as_symmetric<Lower>(); };

template <typename MatrixType>
concept permits_expression_symmetric =
  requires(MatrixType& lhs, MatrixType& rhs) { (lhs + rhs).template as_symmetric<Lower>(); };

template <typename MatrixType>
concept permits_coefficient_write = requires(MatrixType& matrix) { matrix(0, 0) = 0.0; };

template <typename MatrixType>
concept exposes_owning_rvalue_derived = requires(MatrixType& matrix) { std::move(matrix).derived(); };

template <typename MatrixType>
concept permits_owning_rvalue_copy_assignment = requires(MatrixType& lhs, MatrixType& rhs) { std::move(lhs) = rhs; };

using lifetime_matrix = Matrix<double, 3, 3>;
using owning_symmetric = SymmetricMatrix<double, 3, 3>;
using const_symmetric_view = SymmetricMatrixView<const double, 3, 3>;

// a temporary dense owner cannot lend a symmetric wrapper
static_assert(!permits_temporary_symmetric<lifetime_matrix>);
// a const temporary dense owner cannot lend a symmetric wrapper
static_assert(!permits_const_temporary_symmetric<lifetime_matrix>);
// a stored expression can safely retain a symmetric wrapper
static_assert(permits_expression_symmetric<lifetime_matrix>);
// a temporary symmetric owner cannot expose a dangling derived reference
static_assert(!exposes_owning_rvalue_derived<owning_symmetric>);
// a temporary symmetric owner cannot return a borrow through copy assignment
static_assert(!permits_owning_rvalue_copy_assignment<owning_symmetric>);
// a fixed nonempty symmetric view requires an explicit storage binding
static_assert(!std::is_default_constructible_v<SymmetricMatrixView<double, 3, 3>>);
// a dynamic symmetric view permits an initially empty binding
static_assert(std::is_default_constructible_v<SymmetricMatrixView<double, Dynamic, Dynamic>>);
// a const-storage symmetric view advertises read-only access
static_assert(const_symmetric_view::ReadOnly == 1);
// a const-storage symmetric view rejects coefficient writes
static_assert(!permits_coefficient_write<const_symmetric_view>);
// the symmetric trait recognizes a const reference to an owner
static_assert(is_symmetric_matrix_v<const owning_symmetric&>);

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

template <int StorageOrder> void check_dense_symmetric_views() {
    using matrix_type = Matrix<double, 3, 3, StorageOrder>;
    matrix_type dense({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0});
    const auto lower = dense.template as_symmetric<Lower>();
    const auto upper = dense.template as_symmetric<Upper>();
    // a symmetric wrapper borrowed from a const matrix is read-only
    static_assert(decltype(lower)::ReadOnly == 1);
    // a const dense-backed symmetric wrapper rejects coefficient writes
    static_assert(!permits_coefficient_write<decltype(lower)>);
    expect_matrix_near(lower, matrix_type({1.0, 4.0, 7.0, 4.0, 5.0, 8.0, 7.0, 8.0, 9.0}));
    expect_matrix_near(upper, matrix_type({1.0, 2.0, 3.0, 2.0, 5.0, 6.0, 3.0, 6.0, 9.0}));

    const auto temporary = (dense + dense).template as_symmetric<Lower>();
    expect_matrix_near(temporary, matrix_type({2.0, 8.0, 14.0, 8.0, 10.0, 16.0, 14.0, 16.0, 18.0}));
    const auto temporary_upper = (dense + dense).template as_symmetric<Upper>();
    expect_matrix_near(temporary_upper, matrix_type({2.0, 4.0, 6.0, 4.0, 10.0, 12.0, 6.0, 12.0, 18.0}));

    const SymmetricMatrix<double, 3, 3> packed({1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    const Matrix<double, 3, 3> lower_value(lower);
    const Matrix<double, 3, 3> packed_value(packed);
    const Matrix<double, 3, 3> mixed_expected(lower_value + packed_value);
    expect_matrix_near(lower + packed, mixed_expected);
    expect_matrix_near(packed + lower, mixed_expected);

    // const symmetric-wrapper access rejects a negative row
    EXPECT_THROW(static_cast<void>(lower(-1, 0)), std::out_of_range);
    // symmetric-wrapper access rejects a row equal to its dimension
    EXPECT_THROW(static_cast<void>(upper(3, 0)), std::out_of_range);
    Matrix<double, Dynamic, Dynamic, StorageOrder> rectangular(2, 3);
    // symmetric wrapping rejects a rectangular dense matrix
    EXPECT_THROW(static_cast<void>(rectangular.template as_symmetric<Lower>()), std::invalid_argument);
}

void check_packed_symmetric_contracts() {
    const double packed_data[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    SymmetricMatrix<double, 3, 3> matrix(packed_data);
    const Matrix<double, 3, 3> expected({1.0, 2.0, 4.0, 2.0, 3.0, 5.0, 4.0, 5.0, 6.0});
    expect_matrix_near(matrix, expected);

    const auto sum = matrix + matrix;
    const auto difference = matrix - matrix;
    const auto left_scaled = 3.0 * matrix;
    const auto right_scaled = matrix * 3.0;
    const auto divided = matrix / 2.0;
    // addition of symmetric matrices retains the symmetric expression tag
    static_assert(is_symmetric_matrix_v<decltype(sum)>);
    // subtraction of symmetric matrices retains the symmetric expression tag
    static_assert(is_symmetric_matrix_v<decltype(difference)>);
    // left scalar multiplication retains the symmetric expression tag
    static_assert(is_symmetric_matrix_v<decltype(left_scaled)>);
    // right scalar multiplication retains the symmetric expression tag
    static_assert(is_symmetric_matrix_v<decltype(right_scaled)>);
    // scalar division retains the symmetric expression tag
    static_assert(is_symmetric_matrix_v<decltype(divided)>);
    expect_matrix_near(sum, Matrix<double, 3, 3>(expected + expected));
    expect_matrix_near(difference, Matrix<double, 3, 3>::Zero());
    expect_matrix_near(left_scaled, Matrix<double, 3, 3>(expected * 3.0));
    expect_matrix_near(right_scaled, Matrix<double, 3, 3>(expected * 3.0));
    expect_matrix_near(divided, Matrix<double, 3, 3>(expected / 2.0));

    const auto product = matrix * matrix;
    // a general product of symmetric matrices does not claim symmetry
    static_assert(!is_symmetric_matrix_v<decltype(product)>);
    expect_matrix_near(product, Matrix<double, 3, 3>(expected * expected));

    matrix(2, 0) = 8.0;
    // writing the reflected lower entry updates the corresponding upper entry
    EXPECT_DOUBLE_EQ(static_cast<double>(matrix(0, 2)), 8.0);
    const auto& const_matrix = matrix;
    // const access reads the same reflected coefficient after the write
    EXPECT_DOUBLE_EQ(static_cast<double>(const_matrix(2, 0)), 8.0);

    SymmetricMatrix<double, 3, 3> coefficientwise(matrix);
    coefficientwise.cwise() += 2.0;
    expect_matrix_near(coefficientwise, Matrix<double, 3, 3>({3.0, 4.0, 10.0, 4.0, 5.0, 7.0, 10.0, 7.0, 8.0}));

    SymmetricMatrix<double, Dynamic, Dynamic> dynamic_source(std::vector<double> {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    SymmetricMatrix<double, Dynamic, Dynamic> dynamic_target(2, 2);
    dynamic_target = dynamic_source;
    // assignment adopts the source's three-row shape
    EXPECT_EQ(dynamic_target.rows(), 3);
    // assignment preserves the source's square three-column shape
    EXPECT_EQ(dynamic_target.cols(), 3);
    expect_matrix_near(dynamic_target, expected);
    dynamic_target.resize(2, 2);
    // resizing a dynamic symmetric owner changes its row count to two
    EXPECT_EQ(dynamic_target.rows(), 2);
    dynamic_target = matrix;
    // subsequent assignment restores the source's three-row shape
    EXPECT_EQ(dynamic_target.rows(), 3);
    // subsequent assignment restores the source's reflected coefficient
    EXPECT_DOUBLE_EQ(static_cast<double>(dynamic_target(0, 2)), 8.0);
    const SymmetricMatrix<double, 3, Dynamic> partial_default;
    // fixed columns determine the default partially dynamic owner's row count
    EXPECT_EQ(partial_default.rows(), 3);
    // fixed columns retain their declared extent in the default owner
    EXPECT_EQ(partial_default.cols(), 3);

    std::array<double, 6> first_storage {};
    SymmetricMatrixView<double, 3, 3> first_view(first_storage.data());
    // symmetric views are stored by value in expression nodes
    static_assert(decltype(first_view)::NestAsRef == 0);
    const double* const first_address = first_view.data();
    first_view = matrix;
    // assignment into the first view preserves its external storage binding
    EXPECT_EQ(first_view.data(), first_address);
    expect_matrix_near(first_view, matrix);
    first_view(0, 2) = 11.0;
    // the first view exposes the assigned reflected coefficient
    EXPECT_DOUBLE_EQ(static_cast<double>(first_view(2, 0)), 11.0);

    std::array<double, 6> second_storage {};
    SymmetricMatrixView<double, 3, 3> second_view(second_storage.data());
    second_view = first_view;
    // assignment into the second view preserves its external storage binding
    EXPECT_EQ(second_view.data(), second_storage.data());
    expect_matrix_near(second_view, first_view);

    std::array<double, 6> temporary_storage {};
    const auto temporary_view = SymmetricMatrixView<double, 3, 3>(temporary_storage.data()) = first_view;
    // assignment from a temporary view preserves the destination storage binding
    EXPECT_EQ(temporary_view.data(), temporary_storage.data());
    expect_matrix_near(temporary_view, first_view);

    std::array<double, 7> overlap_storage {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0};
    SymmetricMatrixView<double, 3, 3> overlap_source(overlap_storage.data());
    SymmetricMatrixView<double, 3, 3> overlap_destination(overlap_storage.data() + 1);
    overlap_destination = overlap_source;
    expect_matrix_near(overlap_destination, Matrix<double, 3, 3>({1.0, 2.0, 4.0, 2.0, 3.0, 5.0, 4.0, 5.0, 6.0}));

    const SymmetricMatrixView<const double, 3, 3> const_view(first_storage.data());
    // a view of const packed storage advertises read-only access
    static_assert(decltype(const_view)::ReadOnly == 1);
    // a view of const packed storage rejects coefficient writes
    static_assert(!permits_coefficient_write<decltype(const_view)>);
    // the const view reads the assigned coefficient at its reflected coordinate
    EXPECT_DOUBLE_EQ(static_cast<double>(const_view(0, 2)), 11.0);
    SymmetricMatrixView<double, Dynamic, Dynamic> dynamic_view(first_storage.data(), 3, 3);
    expect_matrix_near(dynamic_view + dynamic_view, Matrix<double, 3, 3>(first_view + first_view));
    SymmetricMatrixView<double, Dynamic, Dynamic> empty_view;
    // an empty symmetric view has zero rows
    EXPECT_EQ(empty_view.rows(), 0);
    // an empty symmetric view has zero columns
    EXPECT_EQ(empty_view.cols(), 0);
    // an empty symmetric view permits a null storage binding
    EXPECT_EQ(empty_view.data(), nullptr);

    // construction rejects a packed length that is not a triangular number
    EXPECT_THROW(
      static_cast<void>(SymmetricMatrix<double, Dynamic, Dynamic>(std::vector<double> {1.0, 2.0})),
      std::invalid_argument);
    // construction rejects a nonsquare runtime shape
    EXPECT_THROW(static_cast<void>(SymmetricMatrix<double, Dynamic, Dynamic>(2, 3)), std::invalid_argument);
    // construction rejects runtime dimensions inconsistent with a static axis
    EXPECT_THROW(static_cast<void>(SymmetricMatrix<double, Dynamic, 3>(2, 2)), std::invalid_argument);
    // a nonempty dynamic symmetric view rejects a null storage pointer
    EXPECT_THROW(
      static_cast<void>(SymmetricMatrixView<double, Dynamic, Dynamic>(static_cast<double*>(nullptr), 2, 2)),
      std::invalid_argument);
    // mutable symmetric access rejects a negative row
    EXPECT_THROW(static_cast<void>(matrix(-1, 0)), std::out_of_range);
    // const symmetric access rejects a row equal to its dimension
    EXPECT_THROW(static_cast<void>(std::as_const(matrix)(3, 0)), std::out_of_range);

    SymmetricMatrix<double, Dynamic, Dynamic> resize_target(3, 3);
    resize_target(2, 0) = 7.0;
    // resize rejects a nonsquare target shape
    EXPECT_THROW(resize_target.resize(2, 3), std::invalid_argument);
    // failed resize preserves the original row count
    EXPECT_EQ(resize_target.rows(), 3);
    // failed resize preserves the original reflected coefficient
    EXPECT_DOUBLE_EQ(static_cast<double>(resize_target(2, 0)), 7.0);

    std::array<double, 3> assignment_storage {1.0, 2.0, 3.0};
    SymmetricMatrixView<double, Dynamic, Dynamic> assignment_target(assignment_storage.data(), 2, 2);
    // view assignment rejects a source with incompatible dimensions
    EXPECT_THROW(assignment_target = matrix, std::invalid_argument);
    // failed view assignment preserves the destination's row count
    EXPECT_EQ(assignment_target.rows(), 2);
    // failed view assignment preserves the destination's packed storage
    EXPECT_DOUBLE_EQ(assignment_storage[2], 3.0);
}

}   // namespace

// exercise symmetric reflection, packed storage, structure-preserving arithmetic and view contracts
TEST(linear_algebra, symmetric) {
    check_dense_symmetric_views<RowMajor>();
    check_dense_symmetric_views<ColMajor>();
    check_packed_symmetric_contracts();
}
