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
#include <sstream>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

namespace native = fdapde::linalg;

template <typename Matrix>
concept permits_temporary_diagonal = requires { Matrix {}.diagonal(); };

template <typename Matrix>
concept permits_temporary_triangular = requires { Matrix {}.template triangular_block<native::Lower>(); };

template <typename Matrix>
concept permits_temporary_symmetric = requires { Matrix {}.template as_symmetric<native::Lower>(); };

template <typename Vector>
concept permits_temporary_as_diagonal = requires { Vector {}.as_diagonal(); };

template <typename Matrix>
concept permits_expression_structured_views = requires(Matrix& lhs, Matrix& rhs) {
    (lhs + rhs).diagonal();
    (lhs + rhs).template triangular_block<native::Lower>();
    (lhs + rhs).template as_symmetric<native::Lower>();
};

template <typename Vector>
concept permits_expression_as_diagonal = requires(Vector& lhs, Vector& rhs) { (lhs + rhs).as_diagonal(); };

template <typename Matrix>
concept permits_coefficient_write = requires(Matrix& matrix) { matrix(0, 0) = 0.0; };

template <typename Matrix>
concept exposes_owning_rvalue_derived = requires(Matrix& matrix) { std::move(matrix).derived(); };

template <typename Matrix>
concept permits_owning_rvalue_inverse = requires(Matrix& matrix) { std::move(matrix).inverse(); };

template <typename Matrix>
concept permits_compound_add = requires(Matrix& matrix) { matrix += matrix; };

template <typename Matrix>
concept permits_owning_rvalue_copy_assignment = requires(Matrix& lhs, Matrix& rhs) { std::move(lhs) = rhs; };

using lifetime_matrix = native::Matrix<double, 3, 3>;
using lifetime_vector = native::Vector<double, 3>;
static_assert(!permits_temporary_diagonal<lifetime_matrix>);
static_assert(!permits_temporary_triangular<lifetime_matrix>);
static_assert(!permits_temporary_symmetric<lifetime_matrix>);
static_assert(!permits_temporary_as_diagonal<lifetime_vector>);
static_assert(permits_expression_structured_views<lifetime_matrix>);
static_assert(permits_expression_as_diagonal<lifetime_vector>);
static_assert(!std::is_default_constructible_v<native::DiagonalMatrixView<double, 3>>);
static_assert(std::is_default_constructible_v<native::DiagonalMatrixView<double, fdapde::Dynamic>>);
static_assert(!std::is_default_constructible_v<native::LowerTriangularMatrixView<double, 3, 3>>);
static_assert(
  std::is_default_constructible_v<
    native::LowerTriangularMatrixView<double, fdapde::Dynamic, fdapde::Dynamic>>);
static_assert(!std::is_default_constructible_v<native::SymmetricMatrixView<double, 3, 3>>);
static_assert(
  std::is_default_constructible_v<native::SymmetricMatrixView<double, fdapde::Dynamic, fdapde::Dynamic>>);
static_assert(!std::is_default_constructible_v<native::OrthogonalMatrix<double, 2, 2>>);
static_assert(!std::is_default_constructible_v<native::OrthogonalMatrixView<double, 2, 2>>);
static_assert(!permits_coefficient_write<native::OrthogonalMatrix<double, 2, 2>>);
static_assert(!permits_coefficient_write<native::OrthogonalMatrixView<double, 2, 2>>);
static_assert(!permits_compound_add<native::OrthogonalMatrix<double, 2, 2>>);
static_assert(!permits_compound_add<native::OrthogonalMatrixView<double, 2, 2>>);
static_assert(!exposes_owning_rvalue_derived<native::DiagonalMatrix<double, 3>>);
static_assert(!exposes_owning_rvalue_derived<native::LowerTriangularMatrix<double, 3, 3>>);
static_assert(!exposes_owning_rvalue_derived<native::SymmetricMatrix<double, 3, 3>>);
static_assert(!exposes_owning_rvalue_derived<native::OrthogonalMatrix<double, 2, 2>>);
static_assert(!exposes_owning_rvalue_derived<native::PermutationMatrix<3, 3>>);
static_assert(!permits_owning_rvalue_inverse<native::DiagonalMatrix<double, 3>>);
static_assert(!permits_owning_rvalue_inverse<native::OrthogonalMatrix<double, 2, 2>>);
static_assert(!permits_owning_rvalue_inverse<native::PermutationMatrix<3, 3>>);
static_assert(!permits_owning_rvalue_copy_assignment<native::DiagonalMatrix<double, 3>>);
static_assert(!permits_owning_rvalue_copy_assignment<native::LowerTriangularMatrix<double, 3, 3>>);
static_assert(!permits_owning_rvalue_copy_assignment<native::SymmetricMatrix<double, 3, 3>>);
static_assert(!permits_owning_rvalue_copy_assignment<native::OrthogonalMatrix<double, 2, 2>>);
static_assert(!permits_owning_rvalue_copy_assignment<native::PermutationMatrix<3, 3>>);
static_assert(native::is_diagonal_matrix_v<const native::DiagonalMatrix<double, 3>&>);
static_assert(native::is_triangular_matrix_v<const native::LowerTriangularMatrix<double, 3, 3>&>);
static_assert(native::is_symmetric_matrix_v<const native::SymmetricMatrix<double, 3, 3>&>);
static_assert(native::is_orthogonal_matrix_v<const native::OrthogonalMatrix<double, 2, 2>&>);
static_assert(native::is_permutation_matrix_v<const native::PermutationMatrix<3, 3>&>);

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

TEST(NativeDiagonalMatrix, StorageViewsOperationsAndLifetimes) {
    native::Matrix<double, 3, 3> dense({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0});
    const native::Vector<double, 3> diagonal({2.0, 3.0, 4.0});
    dense.diagonal() = diagonal;
    EXPECT_DOUBLE_EQ(dense(0, 0), 2.0);
    EXPECT_DOUBLE_EQ(dense(1, 1), 3.0);
    EXPECT_DOUBLE_EQ(dense(2, 2), 4.0);

    const auto& const_dense = dense;
    auto const_diagonal = const_dense.diagonal();
    static_assert(decltype(const_diagonal)::ReadOnly == 1);
    static_assert(!permits_coefficient_write<decltype(const_diagonal)>);
    EXPECT_DOUBLE_EQ(const_diagonal[2], 4.0);

    const native::DiagonalMatrix<double, 3> matrix({2.0, 3.0, 4.0});
    EXPECT_DOUBLE_EQ(matrix.determinant(), 24.0);
    EXPECT_EQ(matrix, (native::Matrix<double, 3, 3>({2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0})));
    const native::DiagonalMatrix<double, 3> inverse(matrix.inverse());
    expect_matrix_near(
      inverse,
      native::Matrix<double, 3, 3>({0.5, 0.0, 0.0, 0.0, 1.0 / 3.0, 0.0, 0.0, 0.0, 0.25}));
    const auto sum = matrix + matrix;
    const native::DiagonalMatrix<double, 3> sum_inverse(std::move(sum).inverse());
    expect_matrix_near(
      sum_inverse,
      native::Matrix<double, 3, 3>({0.25, 0.0, 0.0, 0.0, 1.0 / 6.0, 0.0, 0.0, 0.0, 0.125}));
    EXPECT_EQ(
      (native::Matrix<double, 3, 3>((matrix + matrix) + matrix)),
      (native::Matrix<double, 3, 3>({6.0, 0.0, 0.0, 0.0, 9.0, 0.0, 0.0, 0.0, 12.0})));

    const native::Vector<double, 3> rhs({4.0, 9.0, 16.0});
    EXPECT_EQ(matrix.solve(rhs), (native::Vector<double, 3>({2.0, 3.0, 4.0})));

    const native::Matrix<double, 3, 3> operand({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0});
    EXPECT_EQ(
      (native::Matrix<double, 3, 3>(matrix * operand)),
      (native::Matrix<double, 3, 3>({2.0, 4.0, 6.0, 12.0, 15.0, 18.0, 28.0, 32.0, 36.0})));
    EXPECT_EQ(
      (native::Matrix<double, 3, 3>(operand * matrix)),
      (native::Matrix<double, 3, 3>({2.0, 6.0, 12.0, 8.0, 15.0, 24.0, 14.0, 24.0, 36.0})));

    native::DiagonalMatrix<double, fdapde::Dynamic> dynamic(std::vector<double> {1.0, 2.0, 3.0});
    EXPECT_EQ(dynamic.rows(), 3);
    dynamic.resize(4);
    EXPECT_EQ(dynamic.rows(), 4);
    EXPECT_EQ(dynamic.cols(), 4);

    std::array<double, 3> view_storage {5.0, 6.0, 7.0};
    native::DiagonalMatrixView<double, 3> view(view_storage.data());
    const double* const view_address = view.data();
    view = matrix;
    EXPECT_EQ(view.data(), view_address);
    EXPECT_EQ(view, matrix);
    std::array<double, 3> second_view_storage {};
    native::DiagonalMatrixView<double, 3> second_view(second_view_storage.data());
    second_view = view;
    EXPECT_EQ(second_view.data(), second_view_storage.data());
    EXPECT_EQ(second_view, view);
    native::DiagonalMatrixView<const double, 3> const_view(view_storage.data());
    static_assert(decltype(const_view)::ReadOnly == 1);
    static_assert(!permits_coefficient_write<decltype(const_view)>);
    EXPECT_DOUBLE_EQ(const_view(1, 1), 3.0);

    const auto temporary = (diagonal + diagonal).as_diagonal();
    EXPECT_EQ(
      (native::Matrix<double, 3, 3>(temporary)),
      (native::Matrix<double, 3, 3>({4.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 8.0})));
    const native::Vector<double, 3> temporary_dense_diagonal((operand + operand).diagonal());
    EXPECT_EQ(temporary_dense_diagonal, (native::Vector<double, 3>({2.0, 10.0, 18.0})));
}

TEST(NativeTriangularMatrix, PackedStorageViewsProductsAndSolves) {
    const double lower_data[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    const double upper_data[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    native::LowerTriangularMatrix<double, 3, 3> lower(lower_data);
    const native::UpperTriangularMatrix<double, 3, 3> upper(upper_data);
    EXPECT_EQ(lower, (native::Matrix<double, 3, 3>({1.0, 0.0, 0.0, 2.0, 3.0, 0.0, 4.0, 5.0, 6.0})));
    EXPECT_EQ(upper, (native::Matrix<double, 3, 3>({1.0, 2.0, 3.0, 0.0, 4.0, 5.0, 0.0, 0.0, 6.0})));
    lower(0, 2) = 99.0;
    EXPECT_DOUBLE_EQ(lower(0, 2), 0.0);

    const auto lower_square = lower * lower;
    static_assert(native::is_triangular_matrix_v<decltype(lower_square)>);
    EXPECT_EQ(
      (native::Matrix<double, 3, 3>(lower_square)),
      (native::Matrix<double, 3, 3>({1.0, 0.0, 0.0, 8.0, 9.0, 0.0, 38.0, 45.0, 36.0})));
    const native::DiagonalMatrix<double, 3> diagonal({2.0, 3.0, 4.0});
    const auto diagonal_lower = diagonal * lower;
    const auto lower_diagonal = lower * diagonal;
    static_assert(native::is_triangular_matrix_v<decltype(diagonal_lower)>);
    static_assert(native::is_triangular_matrix_v<decltype(lower_diagonal)>);
    EXPECT_EQ(
      (native::Matrix<double, 3, 3>(diagonal_lower)),
      (native::Matrix<double, 3, 3>({2.0, 0.0, 0.0, 6.0, 9.0, 0.0, 16.0, 20.0, 24.0})));
    EXPECT_EQ(
      (native::Matrix<double, 3, 3>(lower_diagonal)),
      (native::Matrix<double, 3, 3>({2.0, 0.0, 0.0, 4.0, 9.0, 0.0, 8.0, 15.0, 24.0})));
    const auto upper_doubled = upper + upper;
    EXPECT_EQ(
      (native::Matrix<double, 3, 3>(upper_doubled)),
      (native::Matrix<double, 3, 3>({2.0, 4.0, 6.0, 0.0, 8.0, 10.0, 0.0, 0.0, 12.0})));

    const auto mixed = lower * upper;
    static_assert(!native::is_triangular_matrix_v<decltype(mixed)>);
    EXPECT_EQ(
      (native::Matrix<double, 3, 3>(mixed)),
      (native::Matrix<double, 3, 3>({1.0, 2.0, 3.0, 2.0, 16.0, 21.0, 4.0, 28.0, 73.0})));

    const native::Vector<double, 3> lower_rhs({1.0, 5.0, 32.0});
    expect_matrix_near(lower.solve(lower_rhs), native::Vector<double, 3>({1.0, 1.0, 23.0 / 6.0}));
    const native::Vector<double, 3> upper_rhs({14.0, 23.0, 18.0});
    EXPECT_EQ(upper.solve(upper_rhs), (native::Vector<double, 3>({1.0, 2.0, 3.0})));
    const native::Vector<double, fdapde::Dynamic> dynamic_lower_rhs(
      std::vector<double> {1.0, 5.0, 32.0});
    expect_matrix_near(lower.solve(dynamic_lower_rhs), native::Vector<double, 3>({1.0, 1.0, 23.0 / 6.0}));

    const native::Matrix<double, 3, 2> expected_solution({1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    const native::Matrix<double, 3, 2> matrix_rhs(lower * expected_solution);
    EXPECT_EQ(lower.solve(matrix_rhs), expected_solution);
    const native::Matrix<double, 3, 2> upper_matrix_rhs(upper * expected_solution);
    EXPECT_EQ(upper.solve(upper_matrix_rhs), expected_solution);

    native::LowerTriangularMatrix<double, fdapde::Dynamic, fdapde::Dynamic> dynamic(
      std::vector<double> {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0});
    ASSERT_EQ(dynamic.rows(), 4);
    const auto dynamic_inverse = dynamic.inverse();
    const native::Matrix<double, fdapde::Dynamic, fdapde::Dynamic> identity(dynamic * dynamic_inverse);
    native::Matrix<double, 4, 4> expected_identity;
    for (int i = 0; i < 4; ++i) expected_identity(i, i) = 1.0;
    expect_matrix_near(identity, expected_identity);

    native::LowerTriangularMatrix<double, fdapde::Dynamic, 3> partial_dynamic(3, 3);
    partial_dynamic = lower;
    EXPECT_EQ(partial_dynamic, lower);
    const native::LowerTriangularMatrix<double, 3, fdapde::Dynamic> partial_default;
    EXPECT_EQ(partial_default.rows(), 3);
    EXPECT_EQ(partial_default.cols(), 3);

    std::array<double, 6> view_storage {};
    native::LowerTriangularMatrixView<double, 3, 3> view(view_storage.data());
    const double* const view_address = view.data();
    view = lower;
    EXPECT_EQ(view.data(), view_address);
    EXPECT_EQ(view, lower);
    std::array<double, 6> second_view_storage {};
    native::LowerTriangularMatrixView<double, 3, 3> second_view(second_view_storage.data());
    second_view = view;
    EXPECT_EQ(second_view.data(), second_view_storage.data());
    EXPECT_EQ(second_view, view);
    native::LowerTriangularMatrixView<const double, 3, 3> const_view(view_storage.data());
    static_assert(decltype(const_view)::ReadOnly == 1);
    static_assert(!permits_coefficient_write<decltype(const_view)>);
    EXPECT_DOUBLE_EQ(const_view(2, 1), 5.0);
    static_assert(decltype(view)::NestAsRef == 0);
    EXPECT_EQ(
      (native::Matrix<double, 3, 3>(view + view)),
      (native::Matrix<double, 3, 3>(lower * 2.0)));
    EXPECT_EQ(
      (native::Matrix<double, 3, 3>(view * 2.0)),
      (native::Matrix<double, 3, 3>(lower * 2.0)));

    native::LowerTriangularMatrixView<double, fdapde::Dynamic, fdapde::Dynamic> dynamic_view(
      view_storage.data(), 3, 3);
    EXPECT_EQ(
      (native::Matrix<double, fdapde::Dynamic, fdapde::Dynamic>(dynamic_view + dynamic_view)),
      (native::Matrix<double, 3, 3>(lower * 2.0)));

    const native::Matrix<double, 3, 3> dense({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0});
    const auto& const_dense = dense;
    auto const_triangular = const_dense.template triangular_block<native::Lower>();
    static_assert(decltype(const_triangular)::ReadOnly == 1);
    static_assert(!permits_coefficient_write<decltype(const_triangular)>);
    EXPECT_DOUBLE_EQ(const_triangular(2, 1), 8.0);
    const native::Matrix<double, 3, 3> dense_const_triangular(const_triangular);
    const native::Matrix<double, 3, 3> dense_lower(lower);
    const native::Matrix<double, 3, 3> mixed_representation_expected(dense_const_triangular + dense_lower);
    EXPECT_EQ(
      (native::Matrix<double, 3, 3>(const_triangular + lower)), mixed_representation_expected);
    EXPECT_EQ(
      (native::Matrix<double, 3, 3>(lower + const_triangular)), mixed_representation_expected);
    const auto temporary = (dense + dense).template triangular_block<native::Lower>();
    EXPECT_EQ(
      (native::Matrix<double, 3, 3>(temporary)),
      (native::Matrix<double, 3, 3>({2.0, 0.0, 0.0, 8.0, 10.0, 0.0, 14.0, 16.0, 18.0})));
}

TEST(NativeSymmetricMatrix, PackedStorageViewsArithmeticAndDynamicExpressions) {
    native::SymmetricMatrix<double, 3, 3> matrix({1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    const native::Matrix<double, 3, 3> expected({1.0, 2.0, 4.0, 2.0, 3.0, 5.0, 4.0, 5.0, 6.0});
    EXPECT_EQ(matrix, expected);
    matrix(2, 0) = 8.0;
    EXPECT_DOUBLE_EQ(matrix(0, 2), 8.0);

    const auto doubled = matrix + matrix;
    static_assert(native::is_symmetric_matrix_v<decltype(doubled)>);
    EXPECT_EQ((native::Matrix<double, 3, 3>(doubled)), (native::Matrix<double, 3, 3>(matrix * 2.0)));
    static_assert(!native::is_symmetric_matrix_v<decltype(matrix * matrix)>);

    std::array<double, 6> view_storage {};
    native::SymmetricMatrixView<double, 3, 3> view(view_storage.data());
    const double* const view_address = view.data();
    view = matrix;
    EXPECT_EQ(view.data(), view_address);
    EXPECT_EQ(view, matrix);
    std::array<double, 6> second_view_storage {};
    native::SymmetricMatrixView<double, 3, 3> second_view(second_view_storage.data());
    second_view = view;
    EXPECT_EQ(second_view.data(), second_view_storage.data());
    EXPECT_EQ(second_view, view);
    native::SymmetricMatrixView<const double, 3, 3> const_view(view_storage.data());
    static_assert(decltype(const_view)::ReadOnly == 1);
    static_assert(!permits_coefficient_write<decltype(const_view)>);
    EXPECT_DOUBLE_EQ(const_view(0, 2), 8.0);

    native::SymmetricMatrix<double, fdapde::Dynamic, fdapde::Dynamic> dynamic(2, 2);
    dynamic = matrix;
    EXPECT_EQ(dynamic.rows(), 3);
    EXPECT_EQ(dynamic.cols(), 3);
    EXPECT_DOUBLE_EQ(dynamic(0, 2), 8.0);
    const native::Matrix<double, fdapde::Dynamic, fdapde::Dynamic> dynamic_sum(dynamic + dynamic);
    EXPECT_EQ(dynamic_sum.rows(), 3);
    EXPECT_EQ(dynamic_sum, (native::Matrix<double, 3, 3>(matrix * 2.0)));
    const native::SymmetricMatrix<double, 3, fdapde::Dynamic> partial_default;
    EXPECT_EQ(partial_default.rows(), 3);
    EXPECT_EQ(partial_default.cols(), 3);

    static_assert(decltype(view)::NestAsRef == 0);
    EXPECT_EQ(
      (native::Matrix<double, 3, 3>(view + view)),
      (native::Matrix<double, 3, 3>(matrix * 2.0)));
    EXPECT_EQ(
      (native::Matrix<double, 3, 3>(view * 2.0)),
      (native::Matrix<double, 3, 3>(matrix * 2.0)));
    native::SymmetricMatrixView<double, fdapde::Dynamic, fdapde::Dynamic> dynamic_view(
      view_storage.data(), 3, 3);
    EXPECT_EQ(
      (native::Matrix<double, fdapde::Dynamic, fdapde::Dynamic>(dynamic_view + dynamic_view)),
      (native::Matrix<double, 3, 3>(matrix * 2.0)));

    native::Matrix<double, fdapde::Dynamic, fdapde::Dynamic> dense(
      native::Matrix<double, 3, 3>({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0}));
    const auto lower_temporary = (dense + dense).template as_symmetric<native::Lower>();
    EXPECT_EQ(
      (native::Matrix<double, fdapde::Dynamic, fdapde::Dynamic>(lower_temporary)),
      (native::Matrix<double, 3, 3>({2.0, 8.0, 14.0, 8.0, 10.0, 16.0, 14.0, 16.0, 18.0})));
    const auto upper_temporary = (dense + dense).template as_symmetric<native::Upper>();
    EXPECT_EQ(
      (native::Matrix<double, fdapde::Dynamic, fdapde::Dynamic>(upper_temporary)),
      (native::Matrix<double, 3, 3>({2.0, 4.0, 6.0, 4.0, 10.0, 12.0, 6.0, 12.0, 18.0})));
    const auto dense_symmetric = dense.template as_symmetric<native::Lower>();
    const native::Matrix<double, fdapde::Dynamic, fdapde::Dynamic> dense_symmetric_value(dense_symmetric);
    const native::Matrix<double, fdapde::Dynamic, fdapde::Dynamic> dense_matrix_value(matrix);
    const native::Matrix<double, fdapde::Dynamic, fdapde::Dynamic> mixed_representation_expected(
      dense_symmetric_value + dense_matrix_value);
    EXPECT_EQ(
      (native::Matrix<double, fdapde::Dynamic, fdapde::Dynamic>(dense_symmetric + matrix)),
      mixed_representation_expected);
    EXPECT_EQ(
      (native::Matrix<double, fdapde::Dynamic, fdapde::Dynamic>(matrix + dense_symmetric)),
      mixed_representation_expected);
}

TEST(NativeOrthogonalMatrix, InvariantConstructorsGroupOperationsAndViews) {
    const double rotation_data[] = {0.0, -1.0, 1.0, 0.0};
    const native::OrthogonalMatrix<double, 2, 2> rotation(rotation_data, native::checked);
    const auto squared = rotation * rotation;
    static_assert(native::is_orthogonal_matrix_v<decltype(squared)>);
    EXPECT_EQ(
      (native::Matrix<double, 2, 2>(squared)),
      (native::Matrix<double, 2, 2>({-1.0, 0.0, 0.0, -1.0})));
    EXPECT_NEAR(rotation.norm(), fdapde::sqrt(2.0), 1.0e-12);

    const native::Matrix<double, 2, 2> inverse(rotation.inverse());
    const native::Matrix<double, 2, 2> transpose(rotation.data().transpose());
    EXPECT_EQ(inverse, transpose);
    const native::Vector<double, 2> rhs({1.0, 0.0});
    EXPECT_EQ((native::Vector<double, 2>(rotation.solve(rhs))), (native::Vector<double, 2>({0.0, -1.0})));

    const native::OrthogonalMatrix<double, 2, 2, native::ColMajor> column_major(rotation_data, native::checked);
    EXPECT_EQ(column_major, rotation);
    static_assert(decltype(column_major)::StorageOrder == native::ColMajor);

    const native::OrthogonalMatrix<double, fdapde::Dynamic, fdapde::Dynamic> dynamic(
      std::vector<double> {0.0, -1.0, 1.0, 0.0}, native::checked);
    EXPECT_EQ(dynamic.rows(), 2);
    EXPECT_EQ(dynamic, rotation);

    const double basis_data[] = {1.0, 1.0, 0.0, 1.0};
    const native::OrthogonalMatrix<double, 2, 2> orthogonalized(basis_data, native::orthogonalize);
    const native::Matrix<double, 2, 2> gram(orthogonalized.data().transpose() * orthogonalized.data());
    expect_matrix_near(gram, native::Matrix<double, 2, 2>({1.0, 0.0, 0.0, 1.0}));
    const double close_basis_data[] = {1.0, 1.0, 1.0, 1.0 + 1.0e-5};
    const native::OrthogonalMatrix<double, 2, 2> reorthogonalized(close_basis_data, native::orthogonalize);
    const native::Matrix<double, 2, 2> close_gram(
      reorthogonalized.data().transpose() * reorthogonalized.data());
    expect_matrix_near(close_gram, native::Matrix<double, 2, 2>({1.0, 0.0, 0.0, 1.0}));

    const double unchecked_data[] = {1.0, 2.0, 3.0, 4.0};
    const native::OrthogonalMatrix<double, 2, 2> unchecked(unchecked_data, native::unchecked);
    EXPECT_DOUBLE_EQ(unchecked(1, 0), 3.0);

    std::array<double, 4> view_storage {0.0, -1.0, 1.0, 0.0};
    std::array<double, 4> identity_storage {1.0, 0.0, 0.0, 1.0};
    native::OrthogonalMatrixView<double, 2, 2> view(view_storage.data(), native::checked);
    native::OrthogonalMatrixView<double, 2, 2> identity_view(identity_storage.data(), native::checked);
    const double* const view_address = view.data().data();
    view = identity_view;
    EXPECT_EQ(view.data().data(), view_address);
    EXPECT_EQ(view, (native::Matrix<double, 2, 2>({1.0, 0.0, 0.0, 1.0})));
    const native::OrthogonalMatrixView<const double, 2, 2> const_view(identity_storage.data(), native::checked);
    static_assert(decltype(const_view)::ReadOnly == 1);
    EXPECT_DOUBLE_EQ(const_view(1, 1), 1.0);
}

TEST(NativePermutationMatrix, ValidationInverseCompositionAndActions) {
    const int permutation_data[] = {2, 0, 1};
    const native::PermutationMatrix<3, 3> permutation(permutation_data);
    EXPECT_EQ(permutation.determinant(), 1);
    EXPECT_NEAR(permutation.norm(), fdapde::sqrt(3.0), 1.0e-12);
    EXPECT_EQ(permutation.permutation(), (native::Vector<int, 3>({2, 0, 1})));

    const auto inverse = permutation.inverse();
    static_assert(native::is_permutation_matrix_v<decltype(inverse)>);
    EXPECT_EQ(inverse.permutation(), (native::Vector<int, 3>({1, 2, 0})));
    const int transposition_data[] = {1, 0, 2};
    const native::PermutationMatrix<3, 3> transposition(transposition_data);
    const auto composition = permutation * transposition;
    static_assert(native::is_permutation_matrix_v<decltype(composition)>);
    EXPECT_EQ(composition.permutation(), (native::Vector<int, 3>({2, 1, 0})));

    const native::Matrix<double, 3, 3> operand({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0});
    EXPECT_EQ(
      (native::Matrix<double, 3, 3>(permutation * operand)),
      (native::Matrix<double, 3, 3>({7.0, 8.0, 9.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0})));
    EXPECT_EQ(
      (native::Matrix<double, 3, 3>(operand * permutation)),
      (native::Matrix<double, 3, 3>({2.0, 3.0, 1.0, 5.0, 6.0, 4.0, 8.0, 9.0, 7.0})));

    const double identity_data[] = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    const native::OrthogonalMatrix<double, 3, 3> orthogonal_identity(identity_data, native::checked);
    static_assert(native::is_orthogonal_matrix_v<decltype(orthogonal_identity * permutation)>);
    static_assert(native::is_orthogonal_matrix_v<decltype(permutation * orthogonal_identity)>);
    EXPECT_EQ(
      (native::Matrix<double, 3, 3>(orthogonal_identity * permutation)),
      (native::Matrix<double, 3, 3>(permutation)));

    const native::DiagonalMatrix<double, 3> diagonal({2.0, 3.0, 4.0});
    const native::Matrix<double, 3, 3> dense_diagonal(diagonal);
    const native::Matrix<double, 3, 3> dense_permutation(permutation);
    EXPECT_EQ(
      (native::Matrix<double, 3, 3>(diagonal * permutation)),
      (native::Matrix<double, 3, 3>(dense_diagonal * dense_permutation)));
    EXPECT_EQ(
      (native::Matrix<double, 3, 3>(permutation * diagonal)),
      (native::Matrix<double, 3, 3>(dense_permutation * dense_diagonal)));

    const native::LowerTriangularMatrix<double, 3, 3> triangular({1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    const native::Matrix<double, 3, 3> dense_triangular(triangular);
    EXPECT_EQ(
      (native::Matrix<double, 3, 3>(triangular * permutation)),
      (native::Matrix<double, 3, 3>(dense_triangular * dense_permutation)));
    EXPECT_EQ(
      (native::Matrix<double, 3, 3>(permutation * triangular)),
      (native::Matrix<double, 3, 3>(dense_permutation * dense_triangular)));

    const native::SymmetricMatrix<double, 3, 3> symmetric({1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    const native::Matrix<double, 3, 3> dense_symmetric(symmetric);
    EXPECT_EQ(
      (native::Matrix<double, 3, 3>(symmetric * permutation)),
      (native::Matrix<double, 3, 3>(dense_symmetric * dense_permutation)));
    EXPECT_EQ(
      (native::Matrix<double, 3, 3>(permutation * symmetric)),
      (native::Matrix<double, 3, 3>(dense_permutation * dense_symmetric)));

    const native::PermutationMatrix<3, 3> identity;
    EXPECT_EQ(identity.permutation(), (native::Vector<int, 3>({0, 1, 2})));
    const native::PermutationMatrix<fdapde::Dynamic, fdapde::Dynamic> dynamic(
      std::vector<int> {1, 2, 0});
    EXPECT_EQ(dynamic.rows(), 3);
    EXPECT_EQ(dynamic.image(2), 0);

    EXPECT_EQ(transposition.determinant(), -1);

    const native::PermutationMatrix<fdapde::Dynamic, fdapde::Dynamic> empty;
    std::ostringstream stream;
    stream << empty;
    EXPECT_TRUE(stream.str().empty());
}

}   // namespace
