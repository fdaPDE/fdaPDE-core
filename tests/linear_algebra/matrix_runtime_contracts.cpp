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

#include <initializer_list>
#include <limits>
#include <vector>

namespace fdapde {

TEST(LinearAlgebraRuntimeContracts, DynamicSquareOperationsRemainAvailable) {
    Matrix<double, Dynamic, Dynamic> matrix = Matrix<double, 2, 2>({2, 1, 1, 3});
    const Matrix<double, 2, 2> zero = Matrix<double, 2, 2>::Zero();
    const Vector<double, 2> expected_diagonal({2, 3});
    const Matrix<double, Dynamic, Dynamic> inverse = matrix.inverse();

    EXPECT_EQ(matrix.symm_part(), matrix);
    EXPECT_EQ(matrix.skew_part(), zero);
    EXPECT_TRUE(almost_equal(inverse * matrix, Matrix<double, 2, 2>({1, 0, 0, 1})));
    EXPECT_EQ(matrix.determinant(), 5);
    EXPECT_EQ(matrix.diagonal(), expected_diagonal);
}

TEST(LinearAlgebraRuntimeContracts, SquareOperationsRejectRectangularMatrices) {
    Matrix<double, Dynamic, Dynamic> matrix(2, 3);

    EXPECT_THROW((void)matrix.symm_part(), std::invalid_argument);
    EXPECT_THROW((void)matrix.skew_part(), std::invalid_argument);
    EXPECT_THROW((void)matrix.inverse(), std::invalid_argument);
    EXPECT_THROW((void)matrix.determinant(), std::invalid_argument);
}

TEST(LinearAlgebraRuntimeContracts, DenseDiagonalViewChecksShapeAndIndexes) {
    Matrix<double, Dynamic, Dynamic> rectangular(2, 3);
    EXPECT_THROW((void)rectangular.diagonal(), std::invalid_argument);

    Matrix<double, 2, 2> matrix({1, 2, 3, 4});
    auto diagonal = matrix.diagonal();
    const auto& const_diagonal = diagonal;

    EXPECT_THROW((void)diagonal(-1, 0), std::out_of_range);
    EXPECT_THROW((void)diagonal(2, 0), std::out_of_range);
    EXPECT_THROW((void)diagonal(0, -1), std::out_of_range);
    EXPECT_THROW((void)diagonal(0, 1), std::out_of_range);
    EXPECT_THROW((void)diagonal[-1], std::out_of_range);
    EXPECT_THROW((void)diagonal[2], std::out_of_range);
    EXPECT_THROW((void)const_diagonal(-1, 0), std::out_of_range);
    EXPECT_THROW((void)const_diagonal(2, 0), std::out_of_range);
    EXPECT_THROW((void)const_diagonal(0, -1), std::out_of_range);
    EXPECT_THROW((void)const_diagonal(0, 1), std::out_of_range);
    EXPECT_THROW((void)const_diagonal[-1], std::out_of_range);
    EXPECT_THROW((void)const_diagonal[2], std::out_of_range);
}

TEST(LinearAlgebraRuntimeContracts, MatrixOwnerRejectsInvalidShapesAndSizes) {
    using dynamic_matrix = Matrix<int, Dynamic, Dynamic>;
    using partial_rows_matrix = Matrix<int, Dynamic, 3>;
    using partial_cols_matrix = Matrix<int, 2, Dynamic>;

    EXPECT_THROW(static_cast<void>(dynamic_matrix(-1, 2)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(partial_rows_matrix(2, 4)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(partial_cols_matrix(3, 2)), std::invalid_argument);
    EXPECT_THROW(
      static_cast<void>(dynamic_matrix(std::numeric_limits<int>::max(), 2)), std::length_error);

    partial_rows_matrix partial(2, 3);
    partial(1, 2) = 17;
    EXPECT_THROW(partial.resize(2, 4), std::invalid_argument);
    EXPECT_EQ(partial.rows(), 2);
    EXPECT_EQ(partial.cols(), 3);
    EXPECT_EQ(partial.size(), 6);
    EXPECT_EQ(partial(1, 2), 17);

    dynamic_matrix bounded(1, 2);
    bounded(0, 1) = 23;
    EXPECT_THROW(bounded.resize(std::numeric_limits<int>::max(), 2), std::length_error);
    EXPECT_EQ(bounded.rows(), 1);
    EXPECT_EQ(bounded.cols(), 2);
    EXPECT_EQ(bounded(0, 1), 23);
}

TEST(LinearAlgebraRuntimeContracts, MatrixOwnerChecksInputsAssignmentsAndIndexes) {
    using dynamic_matrix = Matrix<int, Dynamic, Dynamic>;
    using dynamic_vector = Vector<int, Dynamic>;
    using fixed_matrix = Matrix<int, 2, 3>;
    using fixed_vector = Vector<int, 3>;

    const std::vector<int> short_input {1, 2, 3, 4, 5};
    EXPECT_THROW(static_cast<void>(fixed_matrix(short_input)), std::invalid_argument);

    const dynamic_matrix wrong_matrix_shape(2, 4);
    EXPECT_THROW(static_cast<void>(fixed_matrix(wrong_matrix_shape)), std::invalid_argument);
    const dynamic_matrix non_vector_shape(2, 2);
    EXPECT_THROW(static_cast<void>(fixed_vector(non_vector_shape)), std::invalid_argument);

    fixed_vector vector({1, 2, 3});
    const std::initializer_list<int> short_vector {4, 5};
    EXPECT_THROW(vector = short_vector, std::invalid_argument);
    EXPECT_EQ(vector, fixed_vector({1, 2, 3}));

    fixed_matrix matrix({1, 2, 3, 4, 5, 6});
    const fixed_matrix& const_matrix = matrix;
    EXPECT_THROW(static_cast<void>(matrix(-1, 0)), std::out_of_range);
    EXPECT_THROW(static_cast<void>(const_matrix(0, 3)), std::out_of_range);
    const fixed_vector& const_vector = vector;
    EXPECT_THROW(static_cast<void>(vector[-1]), std::out_of_range);
    EXPECT_THROW(static_cast<void>(const_vector[3]), std::out_of_range);

    EXPECT_THROW(static_cast<void>(dynamic_vector::LinSpaced(1, 0, 1)), std::invalid_argument);
}

TEST(LinearAlgebraRuntimeContracts, ProceduralMatrixChecksShapeSizeAndIndexes) {
    const auto ones = [](int, int) { return 1; };
    using partial_procedural = ProceduralMatrix<decltype(ones), 2, Dynamic>;
    using dynamic_procedural = ProceduralMatrix<decltype(ones), Dynamic, Dynamic>;
    using fixed_procedural_vector = ProceduralMatrix<decltype(ones), 3, 1>;

    EXPECT_THROW(static_cast<void>(partial_procedural(3, 3, ones)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(fixed_procedural_vector(2, ones)), std::invalid_argument);
    EXPECT_THROW(
      static_cast<void>(dynamic_procedural(std::numeric_limits<int>::max(), 2, ones)),
      std::length_error);

    partial_procedural matrix(2, 3, ones);
    EXPECT_THROW(matrix.resize(3, 3), std::invalid_argument);
    EXPECT_EQ(matrix.rows(), 2);
    EXPECT_EQ(matrix.cols(), 3);
    EXPECT_THROW(static_cast<void>(matrix(2, 0)), std::out_of_range);
}

TEST(LinearAlgebraRuntimeContracts, MatrixViewRejectsInvalidRuntimeShapes) {
    using dynamic_matrix_view = MatrixView<int, Dynamic, Dynamic>;
    using partial_matrix_view = MatrixView<int, Dynamic, 3>;
    using dynamic_vector_view = MatrixView<int, Dynamic, 1>;
    using fixed_vector_view = MatrixView<int, 3, 1>;
    int data[6] {};

    dynamic_matrix_view empty;
    EXPECT_EQ(empty.rows(), 0);
    EXPECT_EQ(empty.cols(), 0);
    EXPECT_EQ(empty.data(), nullptr);

    EXPECT_THROW(static_cast<void>(dynamic_matrix_view(data, -1, 3)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(dynamic_matrix_view(data, 0, 3)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(dynamic_matrix_view(data, 2, 0)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(partial_matrix_view(data, 2, 2)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(dynamic_vector_view(data, 0)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(fixed_vector_view(data, 2)), std::invalid_argument);

    int destination_data[4] {1, 2, 3, 4};
    int source_data[6] {6, 5, 4, 3, 2, 1};
    dynamic_matrix_view destination(destination_data, 2, 2);
    dynamic_matrix_view source(source_data, 2, 3);
    int* const destination_binding = destination.data();
    EXPECT_THROW(destination = source, std::invalid_argument);
    EXPECT_EQ(destination.data(), destination_binding);
    EXPECT_EQ(destination.rows(), 2);
    EXPECT_EQ(destination.cols(), 2);
    EXPECT_EQ(destination(0, 0), 1);
    EXPECT_EQ(destination(1, 1), 4);
}

}   // namespace fdapde
