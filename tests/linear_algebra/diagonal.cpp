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
#include <type_traits>
#include <utility>
#include <vector>

namespace fdapde {
namespace {

template <typename MatrixType>
concept permits_temporary_diagonal = requires { MatrixType {}.diagonal(); };

template <typename MatrixType>
concept permits_const_temporary_diagonal = requires(const MatrixType&& matrix) {
    std::move(matrix).diagonal();
};

template <typename VectorType>
concept permits_temporary_as_diagonal = requires { VectorType {}.as_diagonal(); };

template <typename VectorType>
concept permits_const_temporary_as_diagonal = requires(const VectorType&& vector) {
    std::move(vector).as_diagonal();
};

template <typename MatrixType>
concept permits_expression_diagonal = requires(MatrixType& lhs, MatrixType& rhs) {
    (lhs + rhs).diagonal();
};

template <typename VectorType>
concept permits_expression_as_diagonal = requires(VectorType& lhs, VectorType& rhs) {
    (lhs + rhs).as_diagonal();
};

template <typename MatrixType>
concept permits_coefficient_write = requires(MatrixType& matrix) { matrix[0] = 0.0; };

template <typename MatrixType>
concept exposes_owning_rvalue_derived = requires(MatrixType& matrix) { std::move(matrix).derived(); };

template <typename MatrixType>
concept permits_owning_rvalue_inverse = requires(MatrixType& matrix) { std::move(matrix).inverse(); };

template <typename MatrixType>
concept permits_owning_rvalue_copy_assignment = requires(MatrixType& lhs, MatrixType& rhs) {
    std::move(lhs) = rhs;
};

using lifetime_matrix = Matrix<double, 3, 3>;
using lifetime_vector = Vector<double, 3>;
using owning_diagonal = DiagonalMatrix<double, 3>;
using mutable_dense_diagonal = decltype(std::declval<lifetime_matrix&>().diagonal());
using const_dense_diagonal = decltype(std::declval<const lifetime_matrix&>().diagonal());

static_assert(!permits_temporary_diagonal<lifetime_matrix>);
static_assert(!permits_const_temporary_diagonal<lifetime_matrix>);
static_assert(!permits_temporary_as_diagonal<lifetime_vector>);
static_assert(!permits_const_temporary_as_diagonal<lifetime_vector>);
static_assert(permits_expression_diagonal<lifetime_matrix>);
static_assert(permits_expression_as_diagonal<lifetime_vector>);
static_assert(mutable_dense_diagonal::ReadOnly == 0);
static_assert(const_dense_diagonal::ReadOnly == 1);
static_assert(!permits_coefficient_write<const_dense_diagonal>);
static_assert(!std::is_default_constructible_v<DiagonalMatrixView<double, 3>>);
static_assert(std::is_default_constructible_v<DiagonalMatrixView<double, Dynamic>>);
static_assert(!std::is_constructible_v<DiagonalMatrixView<double, 3>, double*, int>);
static_assert(!exposes_owning_rvalue_derived<owning_diagonal>);
static_assert(!permits_owning_rvalue_inverse<owning_diagonal>);
static_assert(!permits_owning_rvalue_copy_assignment<owning_diagonal>);
static_assert(is_diagonal_matrix_v<const owning_diagonal&>);
static_assert(std::is_same_v<decltype(std::declval<const DiagonalMatrix<float, 2>&>().determinant()), float>);
static_assert(!std::is_constructible_v<owning_diagonal, lifetime_matrix&>);

template <int StorageOrder> void check_diagonal_contracts() {
    using matrix_type = Matrix<double, 3, 3, StorageOrder>;
    matrix_type dense({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0});
    const Vector<double, 3> diagonal({2.0, 3.0, 4.0});
    dense.diagonal() = diagonal;
    EXPECT_EQ(dense, (matrix_type({2.0, 2.0, 3.0, 4.0, 3.0, 6.0, 7.0, 8.0, 4.0})));

    const auto& const_dense = dense;
    auto const_diagonal = const_dense.diagonal();
    static_assert(decltype(const_diagonal)::ReadOnly == 1);
    static_assert(!permits_coefficient_write<decltype(const_diagonal)>);
    EXPECT_DOUBLE_EQ(const_diagonal[2], 4.0);

    DiagonalMatrix<double, 3> matrix({2.0, 3.0, 4.0});
    EXPECT_DOUBLE_EQ(matrix.determinant(), 24.0);
    EXPECT_EQ(
      matrix,
      (matrix_type({2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0})));
    const DiagonalMatrix<double, 3> inverse(matrix.inverse());
    EXPECT_TRUE(almost_equal(
      inverse,
      matrix_type({0.5, 0.0, 0.0, 0.0, 1.0 / 3.0, 0.0, 0.0, 0.0, 0.25})));
    const auto sum = matrix + matrix;
    const DiagonalMatrix<double, 3> sum_inverse(std::move(sum).inverse());
    EXPECT_TRUE(almost_equal(
      sum_inverse,
      matrix_type({0.25, 0.0, 0.0, 0.0, 1.0 / 6.0, 0.0, 0.0, 0.0, 0.125})));
    EXPECT_EQ(
      (matrix_type((matrix + matrix) + matrix)),
      (matrix_type({6.0, 0.0, 0.0, 0.0, 9.0, 0.0, 0.0, 0.0, 12.0})));

    const Vector<double, 3> rhs({4.0, 9.0, 16.0});
    EXPECT_EQ(matrix.solve(rhs), (Vector<double, 3>({2.0, 3.0, 4.0})));
    const matrix_type operand({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0});
    EXPECT_EQ(
      (matrix_type(matrix * operand)),
      (matrix_type({2.0, 4.0, 6.0, 12.0, 15.0, 18.0, 28.0, 32.0, 36.0})));
    EXPECT_EQ(
      (matrix_type(operand * matrix)),
      (matrix_type({2.0, 6.0, 12.0, 8.0, 15.0, 24.0, 14.0, 24.0, 36.0})));

    matrix_type compound = operand;
    Vector<double, 3> threes({3.0, 3.0, 3.0});
    compound.diagonal() = Vector<double, 3>::Ones();
    compound.diagonal() += threes;
    compound.diagonal() *= 10.0;
    compound.diagonal() /= 10.0;
    compound.diagonal() -= threes;
    EXPECT_EQ(compound.diagonal(), (Vector<double, 3>::Ones()));

    const auto stored_dense_diagonal = (operand + operand).diagonal();
    EXPECT_EQ((Vector<double, 3>(stored_dense_diagonal)), (Vector<double, 3>({2.0, 10.0, 18.0})));

    std::array<double, 13> overlap_storage {
      1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0};
    MatrixView<double, 3, 3, StorageOrder> overlap_source(overlap_storage.data());
    MatrixView<double, 3, 3, StorageOrder> overlap_destination(overlap_storage.data() + 4);
    overlap_destination.diagonal() = overlap_source.diagonal();
    EXPECT_EQ(
      (Vector<double, 3>(overlap_destination.diagonal())),
      (Vector<double, 3>({1.0, 5.0, 9.0})));

    const Vector<double, 3> vector({2.0, 3.0, 4.0});
    const auto stored_diagonal_matrix = (vector + vector).as_diagonal();
    EXPECT_EQ(
      (matrix_type(stored_diagonal_matrix)),
      (matrix_type({4.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 8.0})));

    Matrix<double, Dynamic, Dynamic, StorageOrder> dynamic_dense(3, 3);
    dynamic_dense.diagonal() = diagonal;
    EXPECT_EQ(dynamic_dense.diagonal(), diagonal);
    Matrix<double, Dynamic, Dynamic, StorageOrder> rectangular(2, 3);
    EXPECT_THROW(static_cast<void>(rectangular.diagonal()), std::invalid_argument);

    Vector<double, Dynamic> dynamic_vector(std::vector<double> {1.0, 2.0, 3.0});
    const auto dynamic_wrapper = dynamic_vector.as_diagonal();
    EXPECT_EQ(dynamic_wrapper.rows(), 3);
    EXPECT_EQ(dynamic_wrapper.cols(), 3);
    EXPECT_EQ(dynamic_wrapper(2, 2), 3.0);

    DiagonalMatrix<double, Dynamic> dynamic(std::vector<double> {1.0, 2.0, 3.0});
    EXPECT_EQ(dynamic.rows(), 3);
    dynamic.resize(4);
    EXPECT_EQ(dynamic.rows(), 4);
    EXPECT_EQ(dynamic.cols(), 4);

    std::array<double, 3> view_storage {5.0, 6.0, 7.0};
    DiagonalMatrixView<double, 3> view(view_storage.data());
    auto fixed_view_diagonal = view.diagonal();
    fixed_view_diagonal[0] = 11.0;
    EXPECT_DOUBLE_EQ(view_storage[0], 11.0);
    const double* const view_address = view.data();
    view = matrix;
    EXPECT_EQ(view.data(), view_address);
    EXPECT_EQ(view, matrix);
    std::array<double, 3> second_view_storage {};
    DiagonalMatrixView<double, 3> second_view(second_view_storage.data());
    second_view = view;
    EXPECT_EQ(second_view.data(), second_view_storage.data());
    EXPECT_EQ(second_view, view);
    DiagonalMatrixView<const double, 3> const_view(view_storage.data());
    static_assert(decltype(const_view)::ReadOnly == 1);
    static_assert(!permits_coefficient_write<decltype(const_view)>);
    EXPECT_DOUBLE_EQ(const_view(1, 1), 3.0);
    auto fixed_const_view_diagonal = const_view.diagonal();
    static_assert(decltype(fixed_const_view_diagonal)::ReadOnly == 1);
    static_assert(!permits_coefficient_write<decltype(fixed_const_view_diagonal)>);
    EXPECT_DOUBLE_EQ(fixed_const_view_diagonal[1], 3.0);

    std::array<double, 3> dynamic_view_storage {8.0, 9.0, 10.0};
    DiagonalMatrixView<double, Dynamic> dynamic_view(dynamic_view_storage.data(), 3);
    auto dynamic_view_diagonal = dynamic_view.diagonal();
    EXPECT_EQ(dynamic_view_diagonal.size(), 3);
    dynamic_view_diagonal[1] = 12.0;
    EXPECT_DOUBLE_EQ(dynamic_view_storage[1], 12.0);
    const auto& const_dynamic_view = dynamic_view;
    auto const_dynamic_view_diagonal = const_dynamic_view.diagonal();
    static_assert(decltype(const_dynamic_view_diagonal)::ReadOnly == 1);
    static_assert(!permits_coefficient_write<decltype(const_dynamic_view_diagonal)>);
    EXPECT_DOUBLE_EQ(const_dynamic_view_diagonal[1], 12.0);

    DiagonalMatrixView<double, Dynamic> empty_view;
    EXPECT_EQ(empty_view.rows(), 0);
    EXPECT_EQ(empty_view.cols(), 0);
    EXPECT_EQ(empty_view.data(), nullptr);
    DiagonalMatrixView<double, Dynamic> explicit_empty_view(nullptr, 0);
    EXPECT_EQ(explicit_empty_view.rows(), 0);
    EXPECT_EQ(explicit_empty_view.cols(), 0);
    EXPECT_EQ(explicit_empty_view.data(), nullptr);
    auto explicit_empty_diagonal = explicit_empty_view.diagonal();
    EXPECT_EQ(explicit_empty_diagonal.size(), 0);
    EXPECT_EQ(explicit_empty_diagonal.data(), nullptr);
    const auto& const_explicit_empty_view = explicit_empty_view;
    auto const_explicit_empty_diagonal = const_explicit_empty_view.diagonal();
    EXPECT_EQ(const_explicit_empty_diagonal.size(), 0);
    EXPECT_EQ(const_explicit_empty_diagonal.data(), nullptr);
}

void check_diagonal_failure_contracts() {
    using dynamic_diagonal = DiagonalMatrix<double, Dynamic>;
    EXPECT_THROW(static_cast<void>(dynamic_diagonal(-1)), std::invalid_argument);

    dynamic_diagonal dynamic(std::vector<double> {1.0, 2.0});
    const dynamic_diagonal original = dynamic;
    EXPECT_THROW(dynamic.resize(-1), std::invalid_argument);
    EXPECT_EQ(dynamic, original);

    using fixed_diagonal = DiagonalMatrix<double, 3>;
    EXPECT_THROW(
      static_cast<void>(fixed_diagonal(std::vector<double> {1.0, 2.0})),
      std::invalid_argument);
    EXPECT_THROW(static_cast<void>(DiagonalMatrixView<double, 3>(nullptr)), std::invalid_argument);
    EXPECT_THROW(
      static_cast<void>(DiagonalMatrixView<double, Dynamic>(nullptr, 1)),
      std::invalid_argument);
    std::array<double, 1> one_value {1.0};
    EXPECT_THROW(
      static_cast<void>(DiagonalMatrixView<double, Dynamic>(one_value.data(), -1)),
      std::invalid_argument);

    const Vector<double, Dynamic> short_rhs(std::vector<double> {1.0});
    EXPECT_THROW(static_cast<void>(dynamic.solve(short_rhs)), std::invalid_argument);

    dynamic_diagonal other(std::vector<double> {1.0, 2.0, 3.0});
    EXPECT_THROW(static_cast<void>(dynamic + other), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(dynamic - other), std::invalid_argument);

    std::array<double, 2> destination_storage {7.0, 8.0};
    std::array<double, 3> source_storage {1.0, 2.0, 3.0};
    DiagonalMatrixView<double, Dynamic> destination(destination_storage.data(), 2);
    DiagonalMatrixView<double, Dynamic> source(source_storage.data(), 3);
    const double* const destination_address = destination.data();
    EXPECT_THROW(destination = source, std::invalid_argument);
    EXPECT_EQ(destination.data(), destination_address);
    EXPECT_DOUBLE_EQ(destination[0], 7.0);
    EXPECT_DOUBLE_EQ(destination[1], 8.0);

    const auto& const_dynamic = dynamic;
    EXPECT_THROW(static_cast<void>(dynamic(-1, 0)), std::out_of_range);
    EXPECT_THROW(static_cast<void>(dynamic(0, 2)), std::out_of_range);
    EXPECT_THROW(static_cast<void>(dynamic[-1]), std::out_of_range);
    EXPECT_THROW(static_cast<void>(dynamic[2]), std::out_of_range);
    EXPECT_THROW(static_cast<void>(const_dynamic(-1, 0)), std::out_of_range);
    EXPECT_THROW(static_cast<void>(const_dynamic(0, 2)), std::out_of_range);
    EXPECT_THROW(static_cast<void>(const_dynamic[-1]), std::out_of_range);
    EXPECT_THROW(static_cast<void>(const_dynamic[2]), std::out_of_range);

    Vector<double, Dynamic> vector(std::vector<double> {1.0, 2.0});
    auto wrapper = vector.as_diagonal();
    EXPECT_THROW(static_cast<void>(wrapper(-1, 0)), std::out_of_range);
    EXPECT_THROW(static_cast<void>(wrapper(0, 2)), std::out_of_range);

}

}   // namespace

TEST(linear_algebra, diagonal) {
    static constexpr Matrix<double, 2, 2> matrix({1.0, 2.0, 3.0, 4.0});
    static constexpr Vector<double, 2> one({1.0, 1.0});
    static_assert(matrix.diagonal().rows() == 2);
    static_assert(matrix.diagonal().cols() == 1);
    static_assert(matrix.diagonal().size() == 2);
    static_assert(matrix.diagonal() == Vector<double, 2>({1.0, 4.0}));
    static_assert((matrix.diagonal() + one) == Vector<double, 2>({2.0, 5.0}));

    check_diagonal_contracts<RowMajor>();
    check_diagonal_contracts<ColMajor>();
    check_diagonal_failure_contracts();
}

}   // namespace fdapde
