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
concept permits_temporary_triangular = requires { MatrixType {}.template triangular_block<Lower>(); };

template <typename MatrixType>
concept permits_expression_triangular =
  requires(MatrixType& lhs, MatrixType& rhs) { (lhs + rhs).template triangular_block<Lower>(); };

template <typename MatrixType>
concept permits_coefficient_write = requires(MatrixType& matrix) { matrix(0, 0) = 0.0; };

template <typename MatrixType>
concept permits_owning_rvalue_copy_assignment = requires(MatrixType& lhs, MatrixType& rhs) { std::move(lhs) = rhs; };

using lifetime_matrix = Matrix<double, 3, 3>;
using lifetime_lower = LowerTriangularMatrix<double, 3, 3>;
using const_lower_view = LowerTriangularMatrixView<const double, 3, 3>;

// a temporary dense owner cannot lend a triangular block
static_assert(!permits_temporary_triangular<lifetime_matrix>);
// a stored expression can safely retain its triangular block
static_assert(permits_expression_triangular<lifetime_matrix>);
// a temporary triangular owner cannot return a borrow through copy assignment
static_assert(!permits_owning_rvalue_copy_assignment<lifetime_lower>);
// a fixed nonempty triangular view requires an explicit storage binding
static_assert(!std::is_default_constructible_v<LowerTriangularMatrixView<double, 3, 3>>);
// a dynamic triangular view permits an initially empty binding
static_assert(std::is_default_constructible_v<LowerTriangularMatrixView<double, Dynamic, Dynamic>>);
// a const-storage triangular view advertises read-only access
static_assert(const_lower_view::ReadOnly == 1);
// a const-storage triangular view rejects coefficient writes
static_assert(!permits_coefficient_write<const_lower_view>);
// the triangular trait recognizes a const reference to an owner
static_assert(is_triangular_matrix_v<const lifetime_lower&>);

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

template <int StorageOrder> void check_dense_triangular_views() {
    Matrix<double, 3, 3, StorageOrder> dense({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0});
    auto lower = dense.template triangular_block<Lower>();
    auto upper = dense.template triangular_block<Upper>();
    expect_matrix_near(lower, Matrix<double, 3, 3>({1.0, 0.0, 0.0, 4.0, 5.0, 0.0, 7.0, 8.0, 9.0}));
    expect_matrix_near(upper, Matrix<double, 3, 3>({1.0, 2.0, 3.0, 0.0, 5.0, 6.0, 0.0, 0.0, 9.0}));

    lower(2, 1) = 12.0;
    lower(0, 2) = 99.0;
    // writing an in-triangle block coefficient updates the dense owner
    EXPECT_DOUBLE_EQ(dense(2, 1), 12.0);
    // the lower triangular block masks coefficients above the diagonal with zero
    EXPECT_DOUBLE_EQ(static_cast<double>(lower(0, 2)), 0.0);

    const auto& const_dense = dense;
    auto const_lower = const_dense.template triangular_block<Lower>();
    // a triangular block borrowed from a const matrix is read-only
    static_assert(decltype(const_lower)::ReadOnly == 1);
    // a const triangular block rejects coefficient writes
    static_assert(!permits_coefficient_write<decltype(const_lower)>);
    // the const block observes the earlier write into the lower triangle
    EXPECT_DOUBLE_EQ(static_cast<double>(const_lower(2, 1)), 12.0);

    // mutable block access rejects a negative row
    EXPECT_THROW(static_cast<void>(lower(-1, 0)), std::out_of_range);
    // const block access rejects a row equal to its dimension
    EXPECT_THROW(static_cast<void>(std::as_const(lower)(3, 0)), std::out_of_range);
}

void check_packed_triangular_contracts() {
    const double lower_data[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    const double upper_data[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    LowerTriangularMatrix<double, 3, 3> lower(lower_data);
    const UpperTriangularMatrix<double, 3, 3> upper(upper_data);

    expect_matrix_near(lower, Matrix<double, 3, 3>({1.0, 0.0, 0.0, 2.0, 3.0, 0.0, 4.0, 5.0, 6.0}));
    expect_matrix_near(upper, Matrix<double, 3, 3>({1.0, 2.0, 3.0, 0.0, 4.0, 5.0, 0.0, 0.0, 6.0}));
    lower(0, 2) = 99.0;
    // an out-of-triangle coefficient still reads as zero
    EXPECT_DOUBLE_EQ(static_cast<double>(lower(0, 2)), 0.0);

    const auto upper_sum = upper + upper;
    // addition of upper triangular matrices retains the triangular expression tag
    static_assert(is_triangular_matrix_v<decltype(upper_sum)>);
    expect_matrix_near(upper_sum, Matrix<double, 3, 3>({2.0, 4.0, 6.0, 0.0, 8.0, 10.0, 0.0, 0.0, 12.0}));

    const auto lower_square = lower * lower;
    // multiplication of lower triangular matrices retains the triangular expression tag
    static_assert(is_triangular_matrix_v<decltype(lower_square)>);
    expect_matrix_near(lower_square, Matrix<double, 3, 3>({1.0, 0.0, 0.0, 8.0, 9.0, 0.0, 38.0, 45.0, 36.0}));

    const auto mixed = lower * upper;
    // combining different triangles does not claim triangular structure
    static_assert(!is_triangular_matrix_v<decltype(mixed)>);
    expect_matrix_near(mixed, Matrix<double, 3, 3>({1.0, 2.0, 3.0, 2.0, 16.0, 21.0, 4.0, 28.0, 73.0}));

    const DiagonalMatrix<double, 3> diagonal({2.0, 3.0, 4.0});
    // left multiplication by a diagonal matrix preserves triangular structure
    static_assert(is_triangular_matrix_v<decltype(diagonal * lower)>);
    // right multiplication by a diagonal matrix preserves triangular structure
    static_assert(is_triangular_matrix_v<decltype(lower * diagonal)>);
    expect_matrix_near(diagonal * lower, Matrix<double, 3, 3>({2.0, 0.0, 0.0, 6.0, 9.0, 0.0, 16.0, 20.0, 24.0}));
    expect_matrix_near(lower * diagonal, Matrix<double, 3, 3>({2.0, 0.0, 0.0, 4.0, 9.0, 0.0, 8.0, 15.0, 24.0}));

    const Vector<double, 3> lower_rhs({1.0, 5.0, 32.0});
    expect_matrix_near(lower.solve(lower_rhs), Vector<double, 3>({1.0, 1.0, 23.0 / 6.0}));
    const Vector<double, Dynamic> dynamic_lower_rhs(std::vector<double> {1.0, 5.0, 32.0});
    expect_matrix_near(lower.solve(dynamic_lower_rhs), Vector<double, 3>({1.0, 1.0, 23.0 / 6.0}));
    const Vector<double, 3> upper_rhs({14.0, 23.0, 18.0});
    expect_matrix_near(upper.solve(upper_rhs), Vector<double, 3>({1.0, 2.0, 3.0}));

    const Matrix<double, 3, 2> solution({1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    const Matrix<double, 3, 2> lower_matrix_rhs(lower * solution);
    const Matrix<double, 3, 2> upper_matrix_rhs(upper * solution);
    expect_matrix_near(lower.solve(lower_matrix_rhs), solution);
    expect_matrix_near(upper.solve(upper_matrix_rhs), solution);

    LowerTriangularMatrix<double, Dynamic, Dynamic> dynamic(
      std::vector<double> {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0});
    // a dynamic triangular owner adopts the requested four-row shape
    ASSERT_EQ(dynamic.rows(), 4);
    const auto dynamic_inverse = dynamic.inverse();
    const Matrix<double, Dynamic, Dynamic> identity(dynamic * dynamic_inverse);
    Matrix<double, 4, 4> expected_identity;
    for (int i = 0; i < 4; ++i) expected_identity(i, i) = 1.0;
    expect_matrix_near(identity, expected_identity);

    LowerTriangularMatrix<double, Dynamic, 3> partial(3, 3);
    partial = lower;
    expect_matrix_near(partial, lower);
    const LowerTriangularMatrix<double, 3, Dynamic> partial_default;
    // fixed columns determine the default partially dynamic owner's row count
    EXPECT_EQ(partial_default.rows(), 3);
    // fixed columns retain their declared extent in the default owner
    EXPECT_EQ(partial_default.cols(), 3);

    std::array<double, 6> first_storage {};
    LowerTriangularMatrixView<double, 3, 3> first_view(first_storage.data());
    // triangular views are stored by value in expression nodes
    static_assert(decltype(first_view)::NestAsRef == 0);
    const double* const first_address = first_view.data();
    first_view = lower;
    // assignment into the first view preserves its external storage binding
    EXPECT_EQ(first_view.data(), first_address);
    expect_matrix_near(first_view, lower);

    std::array<double, 6> second_storage {};
    LowerTriangularMatrixView<double, 3, 3> second_view(second_storage.data());
    second_view = first_view;
    // assignment into the second view preserves its external storage binding
    EXPECT_EQ(second_view.data(), second_storage.data());
    expect_matrix_near(second_view, first_view);

    std::array<double, 7> overlap_storage {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0};
    LowerTriangularMatrixView<double, 3, 3> overlap_source(overlap_storage.data());
    LowerTriangularMatrixView<double, 3, 3> overlap_destination(overlap_storage.data() + 1);
    overlap_destination = overlap_source;
    expect_matrix_near(overlap_destination, Matrix<double, 3, 3>({1.0, 0.0, 0.0, 2.0, 3.0, 0.0, 4.0, 5.0, 6.0}));

    const LowerTriangularMatrixView<const double, 3, 3> const_view(first_storage.data());
    // a const triangular view reads the expected packed lower entry
    EXPECT_DOUBLE_EQ(static_cast<double>(const_view(2, 1)), 5.0);
    LowerTriangularMatrixView<double, Dynamic, Dynamic> dynamic_view(first_storage.data(), 3, 3);
    expect_matrix_near(
      dynamic_view + dynamic_view, Matrix<double, 3, 3>({2.0, 0.0, 0.0, 4.0, 6.0, 0.0, 8.0, 10.0, 12.0}));
    LowerTriangularMatrixView<double, Dynamic, Dynamic> empty_view;
    // an empty triangular view has zero rows
    EXPECT_EQ(empty_view.rows(), 0);
    // an empty triangular view has zero columns
    EXPECT_EQ(empty_view.cols(), 0);

    const Matrix<double, 3, 3> dense({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0});
    const auto dense_lower = dense.template triangular_block<Lower>();
    const Matrix<double, 3, 3> mixed_representation_expected({2.0, 0.0, 0.0, 6.0, 8.0, 0.0, 11.0, 13.0, 15.0});
    expect_matrix_near(dense_lower + lower, mixed_representation_expected);
    expect_matrix_near(lower + dense_lower, mixed_representation_expected);
    const auto temporary = (dense + dense).template triangular_block<Lower>();
    expect_matrix_near(temporary, Matrix<double, 3, 3>({2.0, 0.0, 0.0, 8.0, 10.0, 0.0, 14.0, 16.0, 18.0}));

    // construction rejects a packed length that is not a triangular number
    EXPECT_THROW(
      static_cast<void>(LowerTriangularMatrix<double, Dynamic, Dynamic>(std::vector<double> {1.0, 2.0})),
      std::invalid_argument);
    // construction rejects a nonsquare runtime shape
    EXPECT_THROW(static_cast<void>(LowerTriangularMatrix<double, Dynamic, Dynamic>(2, 3)), std::invalid_argument);
    // construction rejects runtime dimensions inconsistent with the fixed column count
    EXPECT_THROW(static_cast<void>(LowerTriangularMatrix<double, Dynamic, 3>(2, 2)), std::invalid_argument);
    // a nonempty dynamic triangular view rejects a null storage pointer
    EXPECT_THROW(
      static_cast<void>(LowerTriangularMatrixView<double, Dynamic, Dynamic>(static_cast<double*>(nullptr), 2, 2)),
      std::invalid_argument);
    // mutable owner access rejects a negative row
    EXPECT_THROW(static_cast<void>(lower(-1, 0)), std::out_of_range);
    // const owner access rejects a row equal to its dimension
    EXPECT_THROW(static_cast<void>(std::as_const(lower)(3, 0)), std::out_of_range);
    // triangular solve rejects a right-hand side with incompatible length
    EXPECT_THROW(static_cast<void>(lower.solve(Vector<double, Dynamic>(2))), std::invalid_argument);

    LowerTriangularMatrix<double, Dynamic, Dynamic> resize_target(3, 3);
    resize_target(2, 0) = 7.0;
    // resize rejects a nonsquare target shape
    EXPECT_THROW(resize_target.resize(2, 3), std::invalid_argument);
    // failed resize preserves the original row count
    EXPECT_EQ(resize_target.rows(), 3);
    // failed resize preserves the original lower-triangle coefficient
    EXPECT_DOUBLE_EQ(static_cast<double>(resize_target(2, 0)), 7.0);

    std::array<double, 3> assignment_storage {1.0, 2.0, 3.0};
    LowerTriangularMatrixView<double, Dynamic, Dynamic> assignment_target(assignment_storage.data(), 2, 2);
    // view assignment rejects a source with incompatible dimensions
    EXPECT_THROW(assignment_target = lower, std::invalid_argument);
    // failed view assignment preserves the destination's row count
    EXPECT_EQ(assignment_target.rows(), 2);
    // failed view assignment preserves its packed storage
    EXPECT_DOUBLE_EQ(assignment_storage[2], 3.0);

    Matrix<double, Dynamic, Dynamic> rectangular(2, 3);
    // triangular extraction rejects a rectangular dense matrix
    EXPECT_THROW(static_cast<void>(rectangular.template triangular_block<Lower>()), std::invalid_argument);

    const UpperTriangularMatrix<int, 2, 2> integral({1, 2, 3});
    // an integral triangular determinant preserves the integral scalar type
    static_assert(std::is_same_v<decltype(integral.determinant()), int>);
    // the integral determinant equals the exact product of diagonal entries
    EXPECT_EQ(integral.determinant(), 3);
}

}   // namespace

// exercise triangular masking, packed storage, arithmetic, solves and view assignment contracts
TEST(linear_algebra, triangular) {
    check_dense_triangular_views<RowMajor>();
    check_dense_triangular_views<ColMajor>();
    check_packed_triangular_contracts();
}
