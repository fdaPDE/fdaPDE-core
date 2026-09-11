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
#include <initializer_list>
#include <iterator>
#include <type_traits>
#include <utility>
#include <vector>

namespace fdapde {
namespace {

// a fixed nonempty numeric view requires an explicit storage binding
static_assert(!std::is_default_constructible_v<MatrixView<int, 2, 2>>);
// a fully dynamic numeric view permits an initially empty binding
static_assert(std::is_default_constructible_v<MatrixView<int, Dynamic, Dynamic>>);
// a partially dynamic numeric view permits an initially empty binding
static_assert(std::is_default_constructible_v<MatrixView<int, Dynamic, 3>>);
// a mutable numeric view exposes writable scalar storage
static_assert(std::is_same_v<decltype(std::declval<MatrixView<int, 2, 3>&>().data()), int*>);
// a const numeric view object exposes only const scalar storage
static_assert(std::is_same_v<decltype(std::declval<const MatrixView<int, 2, 3>&>().data()), const int*>);
// a view of const scalars exposes only const scalar storage
static_assert(std::is_same_v<decltype(std::declval<MatrixView<const int, 2, 3>&>().data()), const int*>);

template <typename Matrix>
concept permits_left_temporary_add = requires(Matrix& named) { Matrix {} + named; };

template <typename Matrix>
concept permits_right_temporary_add = requires(Matrix& named) { named + Matrix {}; };

template <typename Matrix>
concept permits_left_temporary_subtract = requires(Matrix& named) { Matrix {} - named; };

template <typename Matrix>
concept permits_right_temporary_subtract = requires(Matrix& named) { named - Matrix {}; };

template <typename Matrix>
concept permits_temporary_scalar_multiply = requires { Matrix {} * 2.0; };

template <typename Matrix>
concept permits_scalar_temporary_multiply = requires { 2.0 * Matrix {}; };

template <typename Matrix>
concept permits_temporary_scalar_divide = requires { Matrix {} / 2.0; };

template <typename Matrix>
concept permits_left_temporary_product = requires(Matrix& named) { Matrix {} * named; };

template <typename Matrix>
concept permits_right_temporary_product = requires(Matrix& named) { named * Matrix {}; };

template <typename Matrix>
concept permits_left_temporary_kron = requires(Matrix& named) { kron(Matrix {}, named); };

template <typename Matrix>
concept permits_right_temporary_kron = requires(Matrix& named) { kron(named, Matrix {}); };

template <typename Vector>
concept permits_left_temporary_cross = requires(Vector& named) { Vector {}.cross(named); };

template <typename Vector>
concept permits_right_temporary_cross = requires(Vector& named) { named.cross(Vector {}); };

template <typename Matrix>
concept exposes_temporary_derived = requires { Matrix {}.derived(); };

template <typename Matrix>
concept permits_temporary_transpose = requires { Matrix {}.transpose(); };

template <typename Matrix>
concept permits_temporary_symm_part = requires { Matrix {}.symm_part(); };

template <typename Matrix>
concept permits_temporary_skew_part = requires { Matrix {}.skew_part(); };

template <typename Matrix>
concept permits_temporary_assignment_add = requires(Matrix& named) { (Matrix {} = named) + named; };

template <typename Matrix>
concept permits_temporary_add_assignment = requires(Matrix& named) { Matrix {} += named; };

template <typename Matrix>
concept permits_temporary_subtract_assignment = requires(Matrix& named) { Matrix {} -= named; };

template <typename Matrix>
concept permits_temporary_scalar_multiply_assignment = requires { Matrix {} *= 2.0; };

template <typename Matrix>
concept permits_temporary_scalar_divide_assignment = requires { Matrix {} /= 2.0; };

template <typename Matrix>
concept permits_temporary_product_assignment = requires(Matrix& named) { Matrix {} *= named; };

template <typename Matrix>
concept permits_temporary_initializer_assignment =
  requires(std::initializer_list<typename Matrix::Scalar> values) { Matrix {} = values; };

template <typename Matrix>
concept permits_safe_expression_chaining = requires(Matrix& a, Matrix& b, Matrix& c) {
    (a + b) + c;
    (a + b).transpose();
    (a + b).symm_part();
    (a + b).skew_part();
};

template <typename Vector>
concept permits_safe_cross_chaining = requires(Vector& a, Vector& b, Vector& c) { (a + b).cross(c - b); };

template <typename View>
concept permits_temporary_view_add =
  requires(View& named, std::add_pointer_t<typename View::Scalar> data) { View(data) + named; };

template <typename Matrix>
concept permits_temporary_static_block = requires(Matrix&& matrix) { std::move(matrix).template block<1, 1>(0, 0); };

template <typename Matrix>
concept permits_temporary_dynamic_block = requires(Matrix&& matrix) { std::move(matrix).block(0, 0, 1, 1); };

template <typename Matrix>
concept permits_temporary_row = requires(Matrix&& matrix) { std::move(matrix).row(0); };

template <typename Matrix>
concept permits_temporary_col = requires(Matrix&& matrix) { std::move(matrix).col(0); };

template <typename Matrix>
concept permits_temporary_static_top_rows = requires(Matrix&& matrix) { std::move(matrix).template top_rows<1>(); };

template <typename Matrix>
concept permits_temporary_dynamic_top_rows = requires(Matrix&& matrix) { std::move(matrix).top_rows(1); };

template <typename Matrix>
concept permits_temporary_static_bottom_rows =
  requires(Matrix&& matrix) { std::move(matrix).template bottom_rows<1>(); };

template <typename Matrix>
concept permits_temporary_dynamic_bottom_rows = requires(Matrix&& matrix) { std::move(matrix).bottom_rows(1); };

template <typename Matrix>
concept permits_temporary_static_left_cols = requires(Matrix&& matrix) { std::move(matrix).template left_cols<1>(); };

template <typename Matrix>
concept permits_temporary_dynamic_left_cols = requires(Matrix&& matrix) { std::move(matrix).left_cols(1); };

template <typename Matrix>
concept permits_temporary_static_right_cols = requires(Matrix&& matrix) { std::move(matrix).template right_cols<1>(); };

template <typename Matrix>
concept permits_temporary_dynamic_right_cols = requires(Matrix&& matrix) { std::move(matrix).right_cols(1); };

template <typename Matrix, int ExpectedReadOnly>
concept permits_all_temporary_block_accessors = requires(Matrix&& matrix) {
    requires(decltype(std::move(matrix).template block<1, 1>(0, 0))::ReadOnly == ExpectedReadOnly);
    requires(decltype(std::move(matrix).block(0, 0, 1, 1))::ReadOnly == ExpectedReadOnly);
    requires(decltype(std::move(matrix).row(0))::ReadOnly == ExpectedReadOnly);
    requires(decltype(std::move(matrix).col(0))::ReadOnly == ExpectedReadOnly);
    requires(decltype(std::move(matrix).template top_rows<1>())::ReadOnly == ExpectedReadOnly);
    requires(decltype(std::move(matrix).top_rows(1))::ReadOnly == ExpectedReadOnly);
    requires(decltype(std::move(matrix).template bottom_rows<1>())::ReadOnly == ExpectedReadOnly);
    requires(decltype(std::move(matrix).bottom_rows(1))::ReadOnly == ExpectedReadOnly);
    requires(decltype(std::move(matrix).template left_cols<1>())::ReadOnly == ExpectedReadOnly);
    requires(decltype(std::move(matrix).left_cols(1))::ReadOnly == ExpectedReadOnly);
    requires(decltype(std::move(matrix).template right_cols<1>())::ReadOnly == ExpectedReadOnly);
    requires(decltype(std::move(matrix).right_cols(1))::ReadOnly == ExpectedReadOnly);
};

template <typename Matrix>
concept permits_safe_temporary_block_access = requires(Matrix& a, Matrix& b) {
    (a + b).template block<1, 1>(0, 0);
    (a + b).row(0);
};

template <typename View>
concept permits_temporary_fixed_view_row =
  requires(std::add_pointer_t<typename View::Scalar> data) { View(data).row(0); };

template <typename View>
concept permits_temporary_dynamic_view_row =
  requires(std::add_pointer_t<typename View::Scalar> data) { View(data, 2, 2).row(0); };

template <typename Matrix>
concept permits_direct_temporary_vector_block =
  requires(Matrix&& matrix) { MatrixBlock<1, 1, Matrix> {std::move(matrix), 0}; };

template <typename Matrix>
concept permits_direct_temporary_static_block =
  requires(Matrix&& matrix) { MatrixBlock<1, 1, Matrix> {std::move(matrix), 0, 0}; };

template <typename Matrix>
concept permits_direct_temporary_dynamic_block =
  requires(Matrix&& matrix) { MatrixBlock<Dynamic, Dynamic, Matrix> {std::move(matrix), 0, 0, 1, 1}; };

template <typename Block>
concept permits_const_block_coefficient_write = requires(const Block& block) { block(0, 0) = 1.0; };

template <typename Block>
concept exposes_temporary_block_iterator = requires(Block&& block) { std::move(block).begin(); };

template <typename Block>
concept has_const_bidirectional_block_iterator =
  requires { requires std::bidirectional_iterator<typename Block::const_iterator>; };

template <typename Matrix>
concept permits_temporary_static_matrix_reshape =
  requires(Matrix&& matrix) { std::move(matrix).template reshape<1, 4>(); };

template <typename Matrix>
concept permits_temporary_static_vector_reshape =
  requires(Matrix&& matrix) { std::move(matrix).template reshape<4>(); };

template <typename Matrix>
concept permits_temporary_dynamic_matrix_reshape = requires(Matrix&& matrix) { std::move(matrix).reshape(1, 4); };

template <typename Matrix>
concept permits_temporary_dynamic_vector_reshape = requires(Matrix&& matrix) { std::move(matrix).reshape(4); };

template <typename Matrix, int ExpectedReadOnly>
concept permits_all_temporary_reshape_accessors = requires(Matrix&& matrix) {
    requires(decltype(std::move(matrix).template reshape<1, 4>())::ReadOnly == ExpectedReadOnly);
    requires(decltype(std::move(matrix).template reshape<4>())::ReadOnly == ExpectedReadOnly);
    requires(decltype(std::move(matrix).reshape(1, 4))::ReadOnly == ExpectedReadOnly);
    requires(decltype(std::move(matrix).reshape(4))::ReadOnly == ExpectedReadOnly);
};

template <typename Matrix>
concept permits_direct_temporary_static_reshape =
  requires(Matrix&& matrix) { ReshapeOp<1, 4, Matrix> {std::move(matrix)}; };

template <typename Matrix>
concept permits_direct_temporary_dynamic_reshape =
  requires(Matrix&& matrix) { ReshapeOp<Dynamic, Dynamic, Matrix> {std::move(matrix), 1, 4}; };

template <typename Matrix>
concept permits_direct_temporary_vector_reshape =
  requires(Matrix&& matrix) { ReshapeOp<Dynamic, 1, Matrix> {std::move(matrix), 4}; };

template <typename Reshape>
concept permits_reshape_coefficient_write = requires(Reshape& reshape) { reshape(0, 0) = 1.0; };

template <typename Reshape>
concept permits_const_reshape_coefficient_write = requires(const Reshape& reshape) { reshape(0, 0) = 1.0; };

template <typename Reshape>
concept permits_reshape_vector_write = requires(Reshape& reshape) { reshape[0] = 1.0; };

template <typename Reshape>
concept permits_const_reshape_vector_write = requires(const Reshape& reshape) { reshape[0] = 1.0; };

template <typename Reshape>
concept permits_reshape_assignment = requires(Reshape& lhs, const Reshape& rhs) { lhs = rhs; };

template <typename View, typename Rhs>
concept permits_view_assignment = requires(View& lhs, const Rhs& rhs) { lhs = rhs; };

template <typename View, typename Rhs>
concept permits_view_cwise_assignment = requires(View& lhs, Rhs& rhs) { lhs = rhs.cwise(); };

template <typename Matrix, int ExpectedReadOnly>
concept permits_temporary_cwise_accessor =
  requires(Matrix&& matrix) { requires(decltype(std::move(matrix).cwise())::ReadOnly == ExpectedReadOnly); };

template <typename Matrix>
concept permits_temporary_cwise = requires(Matrix&& matrix) { std::move(matrix).cwise(); };

template <typename Matrix>
concept permits_direct_temporary_cwise = requires(Matrix&& matrix) {
    MatrixCoeffWiseOp<Matrix, internals::identity_op> {std::move(matrix), internals::identity_op {}};
};

template <typename Matrix>
concept exposes_temporary_cwise_derived = requires(Matrix& matrix) { matrix.cwise().derived(); };

template <typename Matrix>
concept exposes_temporary_transformed_cwise_xpr = requires(Matrix& matrix) { matrix.cwise().sqrt().xpr(); };

template <typename Xpr>
concept exposes_writable_cwise_xpr = requires(Xpr& xpr) { xpr.xpr()(0, 0) = 1.0; };

template <typename Matrix>
concept permits_const_cwise_mutation = requires(const Matrix& matrix) { matrix.cwise() += 1.0; };

template <typename Matrix>
concept permits_transformed_cwise_mutation = requires(Matrix& matrix) { matrix.cwise().sqrt()(0, 0) = 1.0; };

template <typename Xpr>
concept permits_cwise_coefficient_write = requires(Xpr& xpr) { xpr(0, 0) = 1.0; };

template <typename Xpr>
concept permits_cwise_scalar_assignment = requires(Xpr& xpr) { xpr = 1.0; };

template <typename Xpr>
concept permits_cwise_scalar_compound = requires(Xpr& xpr) { xpr += 1.0; };

template <typename Matrix, int ExpectedReadOnly>
concept permits_temporary_rowwise_accessor =
  requires(Matrix&& matrix) { requires(decltype(std::move(matrix).rowwise())::ReadOnly == ExpectedReadOnly); };

template <typename Matrix, int ExpectedReadOnly>
concept permits_temporary_colwise_accessor =
  requires(Matrix&& matrix) { requires(decltype(std::move(matrix).colwise())::ReadOnly == ExpectedReadOnly); };

template <typename Matrix>
concept permits_direct_temporary_rowwise = requires(Matrix&& matrix) { MatrixRowWiseOp<Matrix> {std::move(matrix)}; };

template <typename Matrix>
concept permits_direct_temporary_colwise = requires(Matrix&& matrix) { MatrixColWiseOp<Matrix> {std::move(matrix)}; };

template <typename VectorWise, typename Rhs>
concept permits_vectorwise_assignment = requires(VectorWise& lhs, const Rhs& rhs) { lhs = rhs; };

template <typename VectorWise, typename Rhs>
concept permits_vectorwise_addition_assignment = requires(VectorWise& lhs, const Rhs& rhs) { lhs += rhs; };

template <typename VectorWise, typename Rhs>
concept permits_vectorwise_subtraction_assignment = requires(VectorWise& lhs, const Rhs& rhs) { lhs -= rhs; };

template <typename VectorWise, typename Rhs>
concept permits_temporary_vectorwise_assignment = requires(VectorWise&& lhs, const Rhs& rhs) { std::move(lhs) = rhs; };

template <typename VectorWise, typename Rhs>
concept permits_temporary_vectorwise_addition_assignment =
  requires(VectorWise&& lhs, const Rhs& rhs) { std::move(lhs) += rhs; };

template <typename VectorWise, typename Rhs>
concept permits_temporary_vectorwise_subtraction_assignment =
  requires(VectorWise&& lhs, const Rhs& rhs) { std::move(lhs) -= rhs; };

template <typename VectorWise, typename Rhs>
concept permits_vectorwise_comparison = requires(const VectorWise& lhs, const Rhs& rhs) {
    { lhs == rhs } -> std::same_as<bool>;
};

template <typename VectorWise, typename Xpr, typename Reduction>
concept permits_vectorwise_reduction =
  requires(const VectorWise& vectorwise, Xpr& xpr) { vectorwise.redux(xpr, 0.0, Reduction {}); };

template <typename VectorWise, typename Rhs>
inline constexpr bool vectorwise_temporary_mutations_return_values_v =
  !std::is_reference_v<decltype(std::declval<VectorWise&&>() = std::declval<const Rhs&>())> &&
  !std::is_reference_v<decltype(std::declval<VectorWise&&>() += std::declval<const Rhs&>())> &&
  !std::is_reference_v<decltype(std::declval<VectorWise&&>() -= std::declval<const Rhs&>())>;

/// @brief extracts the stored operation type from a partial reduction
template <typename Reduction> struct partial_reduction_traits;

/// @brief exposes the operation template argument for storage-trait assertions
template <typename Xpr, typename Op, int Axis>
struct partial_reduction_traits<internals::partial_matrix_redux_op<Xpr, Op, Axis>> {
    using operation_type = Op;
};

/// @brief preserves coefficient reference constness for writable-adaptor tests
struct writable_passthrough_op {
    /// @brief returns the mutable input reference unchanged
    constexpr double& operator()(double& value) const { return value; }
    /// @brief returns the const input reference unchanged
    constexpr const double& operator()(const double& value) const { return value; }
};

/// @brief adds a fixed offset at every reduction step
struct stateful_reduction_op {
    double offset;

    /// @brief adds the coefficient and stored offset to the accumulator
    constexpr double operator()(double accumulated, double value) const { return accumulated + value + offset; }
};

/// @brief tests reduction ownership with a movable but noncopyable operation
struct move_only_reduction_op {
    double offset;

    /// @brief stores the offset used at each reduction step
    constexpr explicit move_only_reduction_op(double offset_) : offset(offset_) { }
    /// @brief rejects copying to exercise move-only operation storage
    move_only_reduction_op(const move_only_reduction_op&) = delete;
    /// @brief moves the operation while preserving its offset
    constexpr move_only_reduction_op(move_only_reduction_op&&) = default;

    /// @brief adds the coefficient and stored offset to the accumulator
    constexpr double operator()(double accumulated, double value) const { return accumulated + value + offset; }
};

/// @brief requires a mutable coefficient reference to test reduction rejection
struct mutable_value_reduction_op {
    /// @brief accepts only mutable coefficient references for a negative compile-time check
    constexpr double operator()(double accumulated, double& value) const { return accumulated + value; }
};

/// @brief requires a non-const call operator to test reduction rejection
struct mutable_reduction_op {
    /// @brief requires a mutable operation object for a negative compile-time check
    constexpr double operator()(double accumulated, double value) { return accumulated + value; }
};

using lifetime_matrix = Matrix<double, 2, 2>;
using lifetime_const_matrix = const lifetime_matrix;
using lifetime_vector = Matrix<double, 3, 1>;
using lifetime_initializer_vector = Matrix<double, 2, 1>;
using lifetime_expression = decltype(std::declval<lifetime_matrix&>() + std::declval<lifetime_matrix&>());
using lifetime_view = MatrixView<double, 2, 2>;
using lifetime_const_view = MatrixView<const double, 2, 2>;
using lifetime_block = decltype(std::declval<lifetime_matrix&>().template block<1, 1>(0, 0));
using lifetime_const_block = decltype(std::declval<const lifetime_matrix&>().template block<1, 1>(0, 0));
using lifetime_const_view_block = decltype(std::declval<lifetime_const_view&>().template block<1, 1>(0, 0));
using lifetime_row = decltype(std::declval<lifetime_matrix&>().row(0));
using lifetime_row_reshape = decltype(std::declval<lifetime_matrix&>().template reshape<1, 4>());
using lifetime_column_reshape = decltype(std::declval<lifetime_matrix&>().template reshape<4>());
using lifetime_const_owner_reshape = decltype(std::declval<const lifetime_matrix&>().template reshape<1, 4>());
using lifetime_const_owner_column_reshape = decltype(std::declval<const lifetime_matrix&>().template reshape<4>());
using lifetime_const_scalar_view = MatrixView<const double, 2, 2>;
using lifetime_const_scalar_view_reshape =
  decltype(std::declval<lifetime_const_scalar_view&>().template reshape<1, 4>());
using lifetime_const_scalar_view_column_reshape =
  decltype(std::declval<lifetime_const_scalar_view&>().template reshape<4>());
using lifetime_cwise = decltype(std::declval<lifetime_matrix&>().cwise());
using lifetime_const_owner_cwise = decltype(std::declval<const lifetime_matrix&>().cwise());
using lifetime_transformed_cwise = decltype(std::declval<lifetime_matrix&>().cwise().sqrt());
using lifetime_writable_applied_cwise = decltype(std::declval<lifetime_cwise&>().apply(writable_passthrough_op {}));
using lifetime_cwise_mwise = decltype(std::declval<lifetime_cwise&>().mwise());
using lifetime_const_cwise_mwise = decltype(std::declval<const lifetime_cwise&>().mwise());
using lifetime_const_scalar_view_cwise = decltype(std::declval<lifetime_const_scalar_view&>().cwise());
using lifetime_symmetric_cwise = decltype(std::declval<SymmetricMatrix<double, 2, 2>&>().cwise());
using lifetime_triangular_cwise = decltype(std::declval<LowerTriangularMatrix<double, 2, 2>&>().cwise());
using lifetime_rowwise = decltype(std::declval<lifetime_matrix&>().rowwise());
using lifetime_colwise = decltype(std::declval<lifetime_matrix&>().colwise());
using lifetime_const_owner_rowwise = decltype(std::declval<const lifetime_matrix&>().rowwise());
using lifetime_const_owner_colwise = decltype(std::declval<const lifetime_matrix&>().colwise());
using lifetime_const_scalar_view_rowwise = decltype(std::declval<lifetime_const_scalar_view&>().rowwise());
using lifetime_const_scalar_view_colwise = decltype(std::declval<lifetime_const_scalar_view&>().colwise());
using lifetime_rowwise_sum = decltype(std::declval<lifetime_rowwise&>().sum());
using lifetime_colwise_sum = decltype(std::declval<lifetime_colwise&>().sum());
using lifetime_const_scalar_view_rowwise_sum = decltype(std::declval<lifetime_const_scalar_view_rowwise&>().sum());
using lifetime_rowwise_rhs = Matrix<double, 2, 1>;
using lifetime_colwise_rhs = Matrix<double, 1, 2>;
using lifetime_partial_row_source = Matrix<double, Dynamic, 2>;
using lifetime_partial_col_source = Matrix<double, 2, Dynamic>;
using lifetime_partial_rowwise = decltype(std::declval<lifetime_partial_row_source&>().rowwise());
using lifetime_partial_colwise = decltype(std::declval<lifetime_partial_col_source&>().colwise());
using lifetime_valid_partial_row_rhs = Matrix<double, Dynamic, 1>;
using lifetime_invalid_partial_row_rhs = Matrix<double, Dynamic, 2>;
using lifetime_valid_partial_col_rhs = Matrix<double, 1, Dynamic>;
using lifetime_invalid_partial_col_rhs = Matrix<double, 2, Dynamic>;
using lifetime_invalid_fixed_row_rhs = Matrix<double, 3, 1>;
using lifetime_invalid_fixed_col_rhs = Matrix<double, 1, 3>;
using lifetime_custom_reduction = decltype(std::declval<lifetime_rowwise&>().redux(
  std::declval<lifetime_matrix&>(), 0.0, std::declval<stateful_reduction_op&>()));
using lifetime_move_only_reduction = decltype(std::declval<lifetime_rowwise&>().redux(
  std::declval<lifetime_matrix&>(), 0.0, std::declval<move_only_reduction_op>()));
// addition rejects a temporary left owner to prevent a dangling operand
static_assert(!permits_left_temporary_add<lifetime_matrix>);
// addition rejects a temporary right owner to prevent a dangling operand
static_assert(!permits_right_temporary_add<lifetime_matrix>);
// subtraction rejects a temporary left owner to prevent a dangling operand
static_assert(!permits_left_temporary_subtract<lifetime_matrix>);
// subtraction rejects a temporary right owner to prevent a dangling operand
static_assert(!permits_right_temporary_subtract<lifetime_matrix>);
// right scalar multiplication cannot borrow a temporary matrix owner
static_assert(!permits_temporary_scalar_multiply<lifetime_matrix>);
// left scalar multiplication cannot borrow a temporary matrix owner
static_assert(!permits_scalar_temporary_multiply<lifetime_matrix>);
// scalar division cannot borrow a temporary matrix owner
static_assert(!permits_temporary_scalar_divide<lifetime_matrix>);
// matrix multiplication rejects a temporary left owner
static_assert(!permits_left_temporary_product<lifetime_matrix>);
// matrix multiplication rejects a temporary right owner
static_assert(!permits_right_temporary_product<lifetime_matrix>);
// the Kronecker multiplication rejects a temporary left owner
static_assert(!permits_left_temporary_kron<lifetime_matrix>);
// the Kronecker multiplication rejects a temporary right owner
static_assert(!permits_right_temporary_kron<lifetime_matrix>);
// a cross product rejects a temporary left vector owner
static_assert(!permits_left_temporary_cross<lifetime_vector>);
// a cross product rejects a temporary right vector owner
static_assert(!permits_right_temporary_cross<lifetime_vector>);
// a temporary owner cannot expose a dangling derived reference
static_assert(!exposes_temporary_derived<lifetime_matrix>);
// a temporary owner cannot lend a transpose expression
static_assert(!permits_temporary_transpose<lifetime_matrix>);
// a temporary owner cannot lend its symmetric-part expression
static_assert(!permits_temporary_symm_part<lifetime_matrix>);
// a temporary owner cannot lend its skew-part expression
static_assert(!permits_temporary_skew_part<lifetime_matrix>);
// assignment to a temporary owner cannot feed a borrowed addition expression
static_assert(!permits_temporary_assignment_add<lifetime_matrix>);
// addition assignment cannot return a borrow into a temporary owner
static_assert(!permits_temporary_add_assignment<lifetime_matrix>);
// subtraction assignment cannot return a borrow into a temporary owner
static_assert(!permits_temporary_subtract_assignment<lifetime_matrix>);
// scalar multiplication assignment cannot return a borrow into a temporary owner
static_assert(!permits_temporary_scalar_multiply_assignment<lifetime_matrix>);
// scalar division assignment cannot return a borrow into a temporary owner
static_assert(!permits_temporary_scalar_divide_assignment<lifetime_matrix>);
// matrix multiplication assignment cannot return a borrow into a temporary owner
static_assert(!permits_temporary_product_assignment<lifetime_matrix>);
// initializer assignment cannot return a borrow into a temporary vector owner
static_assert(!permits_temporary_initializer_assignment<lifetime_initializer_vector>);
// arithmetic and transpose chains retain expression nodes while borrowing live owners
static_assert(permits_safe_expression_chaining<lifetime_matrix>);
// cross-product chains retain temporary expression nodes while borrowing live vectors
static_assert(permits_safe_cross_chaining<lifetime_vector>);
// addition can safely store a temporary view of externally owned storage
static_assert(permits_temporary_view_add<MatrixView<double, 2, 2>>);
// a mutable view accepts assignment from a numeric owner
static_assert(permits_view_assignment<lifetime_view, lifetime_matrix>);
// a const-storage view rejects assignment from a numeric owner
static_assert(!permits_view_assignment<lifetime_const_view, lifetime_matrix>);
// a mutable coefficientwise view accepts assignment from a numeric owner
static_assert(permits_view_cwise_assignment<lifetime_view, lifetime_matrix>);
// a const-storage coefficientwise view rejects assignment from a numeric owner
static_assert(!permits_view_cwise_assignment<lifetime_const_view, lifetime_matrix>);
// a temporary owner cannot lend a statically sized block
static_assert(!permits_temporary_static_block<lifetime_matrix>);
// a temporary owner cannot lend a runtime-sized block
static_assert(!permits_temporary_dynamic_block<lifetime_matrix>);
// a temporary owner cannot lend a row
static_assert(!permits_temporary_row<lifetime_matrix>);
// a temporary owner cannot lend a column
static_assert(!permits_temporary_col<lifetime_matrix>);
// a temporary owner cannot lend a static prefix of rows
static_assert(!permits_temporary_static_top_rows<lifetime_matrix>);
// a temporary owner cannot lend a runtime prefix of rows
static_assert(!permits_temporary_dynamic_top_rows<lifetime_matrix>);
// a temporary owner cannot lend a static suffix of rows
static_assert(!permits_temporary_static_bottom_rows<lifetime_matrix>);
// a temporary owner cannot lend a runtime suffix of rows
static_assert(!permits_temporary_dynamic_bottom_rows<lifetime_matrix>);
// a temporary owner cannot lend a static prefix of columns
static_assert(!permits_temporary_static_left_cols<lifetime_matrix>);
// a temporary owner cannot lend a runtime prefix of columns
static_assert(!permits_temporary_dynamic_left_cols<lifetime_matrix>);
// a temporary owner cannot lend a static suffix of columns
static_assert(!permits_temporary_static_right_cols<lifetime_matrix>);
// a temporary owner cannot lend a runtime suffix of columns
static_assert(!permits_temporary_dynamic_right_cols<lifetime_matrix>);
// a const temporary owner cannot lend a statically sized block
static_assert(!permits_temporary_static_block<lifetime_const_matrix>);
// a const temporary owner cannot lend a runtime-sized block
static_assert(!permits_temporary_dynamic_block<lifetime_const_matrix>);
// a const temporary owner cannot lend a row
static_assert(!permits_temporary_row<lifetime_const_matrix>);
// a const temporary owner cannot lend a column
static_assert(!permits_temporary_col<lifetime_const_matrix>);
// a const temporary owner cannot lend a static prefix of rows
static_assert(!permits_temporary_static_top_rows<lifetime_const_matrix>);
// a const temporary owner cannot lend a runtime prefix of rows
static_assert(!permits_temporary_dynamic_top_rows<lifetime_const_matrix>);
// a const temporary owner cannot lend a static suffix of rows
static_assert(!permits_temporary_static_bottom_rows<lifetime_const_matrix>);
// a const temporary owner cannot lend a runtime suffix of rows
static_assert(!permits_temporary_dynamic_bottom_rows<lifetime_const_matrix>);
// a const temporary owner cannot lend a static prefix of columns
static_assert(!permits_temporary_static_left_cols<lifetime_const_matrix>);
// a const temporary owner cannot lend a runtime prefix of columns
static_assert(!permits_temporary_dynamic_left_cols<lifetime_const_matrix>);
// a const temporary owner cannot lend a static suffix of columns
static_assert(!permits_temporary_static_right_cols<lifetime_const_matrix>);
// a const temporary owner cannot lend a runtime suffix of columns
static_assert(!permits_temporary_dynamic_right_cols<lifetime_const_matrix>);
// a temporary expression supports all block accessors as read-only expressions
static_assert(permits_all_temporary_block_accessors<lifetime_expression, 1>);
// a temporary mutable view supports all block accessors with writable coefficients
static_assert(permits_all_temporary_block_accessors<lifetime_view, 0>);
// a const temporary view supports all block accessors with read-only coefficients
static_assert(permits_all_temporary_block_accessors<const lifetime_view, 1>);
// blocks of temporary expression nodes can safely retain live matrix operands
static_assert(permits_safe_temporary_block_access<lifetime_matrix>);
// a temporary fixed view can safely lend a row by storing its binding
static_assert(permits_temporary_fixed_view_row<MatrixView<double, 2, 2>>);
// a temporary dynamic view can safely lend a row by storing its binding
static_assert(permits_temporary_dynamic_view_row<MatrixView<double, Dynamic, Dynamic>>);
// direct vector-block construction rejects a temporary owner
static_assert(!permits_direct_temporary_vector_block<lifetime_matrix>);
// direct static-block construction rejects a temporary owner
static_assert(!permits_direct_temporary_static_block<lifetime_matrix>);
// direct runtime-block construction rejects a temporary owner
static_assert(!permits_direct_temporary_dynamic_block<lifetime_matrix>);
// direct vector-block construction rejects a const temporary owner
static_assert(!permits_direct_temporary_vector_block<lifetime_const_matrix>);
// direct static-block construction rejects a const temporary owner
static_assert(!permits_direct_temporary_static_block<lifetime_const_matrix>);
// direct runtime-block construction rejects a const temporary owner
static_assert(!permits_direct_temporary_dynamic_block<lifetime_const_matrix>);
// a block borrowed from a const owner is read-only
static_assert(lifetime_const_block::ReadOnly == 1);
// a block borrowed from a const-storage view is read-only
static_assert(lifetime_const_view_block::ReadOnly == 1);
// a const block object rejects coefficient writes
static_assert(!permits_const_block_coefficient_write<lifetime_block>);
// a const-storage view's block rejects coefficient writes
static_assert(!permits_const_block_coefficient_write<lifetime_const_view_block>);
// const block access returns a const scalar reference
static_assert(std::is_same_v<decltype(std::declval<const lifetime_block&>()(0, 0)), const double&>);
// access through a block of const storage returns a const scalar reference
static_assert(std::is_same_v<decltype(std::declval<lifetime_const_view_block&>()(0, 0)), const double&>);
// const row indexing returns a const scalar reference
static_assert(std::is_same_v<decltype(std::declval<const lifetime_row&>()[0]), const double&>);
// a temporary row wrapper cannot expose iterators tied to its expired state
static_assert(!exposes_temporary_block_iterator<lifetime_row>);
// mutable row iterators satisfy the standard bidirectional iterator requirements
static_assert(std::bidirectional_iterator<typename lifetime_row::iterator>);
// const row iterators satisfy the standard bidirectional iterator requirements
static_assert(has_const_bidirectional_block_iterator<lifetime_row>);
// dereferencing a const row iterator yields a const scalar reference
static_assert(std::is_same_v<std::iter_reference_t<typename lifetime_row::const_iterator>, const double&>);
using temporary_row_initializer_result =
  decltype(std::declval<lifetime_row&&>() = std::declval<const std::initializer_list<double>&>());
// initializer assignment to a temporary row returns a value rather than a dangling reference
static_assert(!std::is_lvalue_reference_v<temporary_row_initializer_result>);
// a temporary owner cannot lend a matrix reshape with static extents
static_assert(!permits_temporary_static_matrix_reshape<lifetime_matrix>);
// a temporary owner cannot lend a vector reshape with static extent
static_assert(!permits_temporary_static_vector_reshape<lifetime_matrix>);
// a temporary owner cannot lend a matrix reshape with runtime extents
static_assert(!permits_temporary_dynamic_matrix_reshape<lifetime_matrix>);
// a temporary owner cannot lend a vector reshape with runtime extent
static_assert(!permits_temporary_dynamic_vector_reshape<lifetime_matrix>);
// a const temporary owner cannot lend a matrix reshape with static extents
static_assert(!permits_temporary_static_matrix_reshape<lifetime_const_matrix>);
// a const temporary owner cannot lend a vector reshape with static extent
static_assert(!permits_temporary_static_vector_reshape<lifetime_const_matrix>);
// a const temporary owner cannot lend a matrix reshape with runtime extents
static_assert(!permits_temporary_dynamic_matrix_reshape<lifetime_const_matrix>);
// a const temporary owner cannot lend a vector reshape with runtime extent
static_assert(!permits_temporary_dynamic_vector_reshape<lifetime_const_matrix>);
// a temporary expression supports every reshape accessor with read-only coefficients
static_assert(permits_all_temporary_reshape_accessors<lifetime_expression, 1>);
// a temporary mutable view supports every reshape accessor with writable coefficients
static_assert(permits_all_temporary_reshape_accessors<lifetime_view, 0>);
// a const temporary view supports every reshape accessor with read-only coefficients
static_assert(permits_all_temporary_reshape_accessors<const lifetime_view, 1>);
// a temporary const-storage view supports every reshape accessor with read-only coefficients
static_assert(permits_all_temporary_reshape_accessors<lifetime_const_view, 1>);
// direct static-reshape construction rejects a temporary owner
static_assert(!permits_direct_temporary_static_reshape<lifetime_matrix>);
// direct runtime-reshape construction rejects a temporary owner
static_assert(!permits_direct_temporary_dynamic_reshape<lifetime_matrix>);
// direct vector-reshape construction rejects a temporary owner
static_assert(!permits_direct_temporary_vector_reshape<lifetime_matrix>);
// direct static-reshape construction rejects a const temporary owner
static_assert(!permits_direct_temporary_static_reshape<lifetime_const_matrix>);
// direct runtime-reshape construction rejects a const temporary owner
static_assert(!permits_direct_temporary_dynamic_reshape<lifetime_const_matrix>);
// direct vector-reshape construction rejects a const temporary owner
static_assert(!permits_direct_temporary_vector_reshape<lifetime_const_matrix>);
// a reshape borrowed from a const owner is read-only
static_assert(lifetime_const_owner_reshape::ReadOnly == 1);
// a reshape borrowed from const scalar storage is read-only
static_assert(lifetime_const_scalar_view_reshape::ReadOnly == 1);
// mutable row-reshape coordinate access returns a writable scalar reference
static_assert(std::is_same_v<decltype(std::declval<lifetime_row_reshape&>()(0, 0)), double&>);
// mutable row-reshape vector indexing returns a writable scalar reference
static_assert(std::is_same_v<decltype(std::declval<lifetime_row_reshape&>()[0]), double&>);
// mutable column-reshape vector indexing returns a writable scalar reference
static_assert(std::is_same_v<decltype(std::declval<lifetime_column_reshape&>()[0]), double&>);
// const row-reshape coordinate access returns a const scalar reference
static_assert(std::is_same_v<decltype(std::declval<const lifetime_row_reshape&>()(0, 0)), const double&>);
// const row-reshape vector indexing returns a const scalar reference
static_assert(std::is_same_v<decltype(std::declval<const lifetime_row_reshape&>()[0]), const double&>);
// const column-reshape vector indexing returns a const scalar reference
static_assert(std::is_same_v<decltype(std::declval<const lifetime_column_reshape&>()[0]), const double&>);
// a column reshape borrowed from a const owner returns const scalar references
static_assert(std::is_same_v<decltype(std::declval<lifetime_const_owner_column_reshape&>()[0]), const double&>);
// a column reshape borrowed from const storage returns const scalar references
static_assert(std::is_same_v<decltype(std::declval<lifetime_const_scalar_view_column_reshape&>()[0]), const double&>);
// a const row-reshape object rejects coordinate writes
static_assert(!permits_const_reshape_coefficient_write<lifetime_row_reshape>);
// a const column-reshape object rejects vector-index writes
static_assert(!permits_const_reshape_vector_write<lifetime_column_reshape>);
// a reshape borrowed from a const owner rejects coordinate writes
static_assert(!permits_reshape_coefficient_write<lifetime_const_owner_reshape>);
// a reshape borrowed from const storage rejects coordinate writes
static_assert(!permits_reshape_coefficient_write<lifetime_const_scalar_view_reshape>);
// a column reshape borrowed from a const owner rejects vector-index writes
static_assert(!permits_reshape_vector_write<lifetime_const_owner_column_reshape>);
// a column reshape borrowed from const storage rejects vector-index writes
static_assert(!permits_reshape_vector_write<lifetime_const_scalar_view_column_reshape>);
// a mutable reshape accepts assignment into its source storage
static_assert(permits_reshape_assignment<lifetime_row_reshape>);
// a reshape borrowed from a const owner rejects assignment
static_assert(!permits_reshape_assignment<lifetime_const_owner_reshape>);
using temporary_reshape_assignment_result =
  decltype(std::declval<lifetime_row_reshape&&>() = std::declval<const lifetime_row_reshape&>());
// assignment to a temporary reshape returns a value retaining its storage binding
static_assert(std::is_same_v<temporary_reshape_assignment_result, lifetime_row_reshape>);
// a temporary owner cannot lend a mutable coefficientwise adaptor
static_assert(!permits_temporary_cwise_accessor<lifetime_matrix, 0>);
// a const temporary owner cannot lend a read-only coefficientwise adaptor
static_assert(!permits_temporary_cwise_accessor<lifetime_const_matrix, 1>);
// a temporary owner cannot feed a coefficientwise expression chain
static_assert(!permits_temporary_cwise<lifetime_matrix>);
// a const temporary owner cannot feed a coefficientwise expression chain
static_assert(!permits_temporary_cwise<lifetime_const_matrix>);
// a temporary expression can be stored in a read-only coefficientwise adaptor
static_assert(permits_temporary_cwise_accessor<lifetime_expression, 1>);
// a temporary mutable view can be stored in a writable coefficientwise adaptor
static_assert(permits_temporary_cwise_accessor<lifetime_view, 0>);
// a const temporary view can be stored in a read-only coefficientwise adaptor
static_assert(permits_temporary_cwise_accessor<const lifetime_view, 1>);
// a temporary const-storage view can be stored in a read-only coefficientwise adaptor
static_assert(permits_temporary_cwise_accessor<lifetime_const_view, 1>);
// direct coefficientwise-adaptor construction rejects a temporary owner
static_assert(!permits_direct_temporary_cwise<lifetime_matrix>);
// direct coefficientwise-adaptor construction rejects a const temporary owner
static_assert(!permits_direct_temporary_cwise<lifetime_const_matrix>);
// a temporary coefficientwise adaptor cannot expose a dangling derived reference
static_assert(!exposes_temporary_cwise_derived<lifetime_matrix>);
// a temporary transformed adaptor cannot expose a dangling nested-expression reference
static_assert(!exposes_temporary_transformed_cwise_xpr<lifetime_matrix>);
// a transformed coefficientwise expression cannot expose a writable operand
static_assert(!exposes_writable_cwise_xpr<lifetime_transformed_cwise>);
// a plain adaptor over mutable storage advertises writable coefficients
static_assert(lifetime_cwise::ReadOnly == 0);
// a plain adaptor over a const owner advertises read-only coefficients
static_assert(lifetime_const_owner_cwise::ReadOnly == 1);
// a value-transforming adaptor advertises read-only coefficients
static_assert(lifetime_transformed_cwise::ReadOnly == 1);
// a plain adaptor over const scalar storage advertises read-only coefficients
static_assert(lifetime_const_scalar_view_cwise::ReadOnly == 1);
// a mutable plain adaptor permits coefficient writes
static_assert(permits_cwise_coefficient_write<lifetime_cwise>);
// a const-owner adaptor rejects coefficient writes
static_assert(!permits_cwise_coefficient_write<lifetime_const_owner_cwise>);
// a value-transforming adaptor rejects coefficient writes
static_assert(!permits_cwise_coefficient_write<lifetime_transformed_cwise>);
// a const-storage adaptor rejects coefficient writes
static_assert(!permits_cwise_coefficient_write<lifetime_const_scalar_view_cwise>);
// a reference-returning custom operation preserves writable coefficients
static_assert(lifetime_writable_applied_cwise::ReadOnly == 0);
// a reference-returning custom operation permits writes through the returned reference
static_assert(permits_cwise_coefficient_write<lifetime_writable_applied_cwise>);
// bulk scalar assignment is unavailable for a custom applied expression
static_assert(!permits_cwise_scalar_assignment<lifetime_writable_applied_cwise>);
// bulk scalar compound assignment is unavailable for a custom applied expression
static_assert(!permits_cwise_scalar_compound<lifetime_writable_applied_cwise>);
// switching a mutable plain adaptor back to matrix semantics preserves writability
static_assert(permits_cwise_coefficient_write<lifetime_cwise_mwise>);
// switching a const plain adaptor back to matrix semantics preserves read-only access
static_assert(!permits_cwise_coefficient_write<lifetime_const_cwise_mwise>);
// a symmetric proxy adaptor advertises writable coefficients
static_assert(lifetime_symmetric_cwise::ReadOnly == 0);
// a triangular proxy adaptor advertises writable coefficients
static_assert(lifetime_triangular_cwise::ReadOnly == 0);
// a symmetric proxy adaptor permits structure-aware coefficient writes
static_assert(permits_cwise_coefficient_write<lifetime_symmetric_cwise>);
// a triangular proxy adaptor permits structure-aware coefficient writes
static_assert(permits_cwise_coefficient_write<lifetime_triangular_cwise>);
// const owners cannot be mutated through coefficientwise adaptors
static_assert(!permits_const_cwise_mutation<lifetime_matrix>);
// computed coefficientwise transformations cannot mutate their source
static_assert(!permits_transformed_cwise_mutation<lifetime_matrix>);
// plain mutable coefficient access returns a scalar reference
static_assert(std::is_same_v<decltype(std::declval<lifetime_cwise&>()(0, 0)), double&>);
// const access to the nested expression returns a const owner reference
static_assert(std::is_same_v<decltype(std::declval<const lifetime_cwise&>().xpr()), const lifetime_matrix&>);
using lifetime_const_applied_cwise = decltype(std::declval<const lifetime_cwise&>().apply(internals::identity_op {}));
// a custom operation applied to const coefficients remains read-only
static_assert(lifetime_const_applied_cwise::ReadOnly == 1);
// a custom operation cannot bypass const coefficient access
static_assert(!permits_cwise_coefficient_write<lifetime_const_applied_cwise>);
using lifetime_integer_cwise_inverse = decltype(std::declval<Matrix<int, 1, 2>&>().cwise().inv());
using lifetime_cwise_comparison = decltype(std::declval<lifetime_matrix&>().cwise() < 1.0);
// coefficientwise inversion promotes integer input to double
static_assert(std::is_same_v<typename lifetime_integer_cwise_inverse::Scalar, double>);
// coefficientwise comparison produces Boolean scalar coefficients
static_assert(std::is_same_v<typename lifetime_cwise_comparison::Scalar, bool>);
using temporary_cwise_scalar_assignment_result = decltype(std::declval<lifetime_cwise&&>() = 1.0);
using temporary_cwise_scalar_compound_result = decltype(std::declval<lifetime_cwise&&>() += 1.0);
using temporary_cwise_expression_compound_result =
  decltype(std::declval<lifetime_cwise&&>() += std::declval<const lifetime_cwise&>());
// scalar assignment to a temporary adaptor returns a value retaining its binding
static_assert(std::is_same_v<temporary_cwise_scalar_assignment_result, lifetime_cwise>);
// scalar compound assignment to a temporary adaptor returns a value retaining its binding
static_assert(std::is_same_v<temporary_cwise_scalar_compound_result, lifetime_cwise>);
// expression compound assignment to a temporary adaptor returns a value retaining its binding
static_assert(std::is_same_v<temporary_cwise_expression_compound_result, lifetime_cwise>);
// a temporary owner cannot lend a rowwise adaptor
static_assert(!permits_temporary_rowwise_accessor<lifetime_matrix, 0>);
// a temporary owner cannot lend a columnwise adaptor
static_assert(!permits_temporary_colwise_accessor<lifetime_matrix, 0>);
// a const temporary owner cannot lend a rowwise adaptor
static_assert(!permits_temporary_rowwise_accessor<lifetime_const_matrix, 1>);
// a const temporary owner cannot lend a columnwise adaptor
static_assert(!permits_temporary_colwise_accessor<lifetime_const_matrix, 1>);
// a temporary expression can be stored in a read-only rowwise adaptor
static_assert(permits_temporary_rowwise_accessor<lifetime_expression, 1>);
// a temporary expression can be stored in a read-only columnwise adaptor
static_assert(permits_temporary_colwise_accessor<lifetime_expression, 1>);
// a temporary mutable view can be stored in a writable rowwise adaptor
static_assert(permits_temporary_rowwise_accessor<lifetime_view, 0>);
// a temporary mutable view can be stored in a writable columnwise adaptor
static_assert(permits_temporary_colwise_accessor<lifetime_view, 0>);
// a const temporary view can be stored in a read-only rowwise adaptor
static_assert(permits_temporary_rowwise_accessor<const lifetime_view, 1>);
// a const temporary view can be stored in a read-only columnwise adaptor
static_assert(permits_temporary_colwise_accessor<const lifetime_view, 1>);
// a temporary const-storage view can be stored in a read-only rowwise adaptor
static_assert(permits_temporary_rowwise_accessor<lifetime_const_view, 1>);
// a temporary const-storage view can be stored in a read-only columnwise adaptor
static_assert(permits_temporary_colwise_accessor<lifetime_const_view, 1>);
// direct rowwise-adaptor construction rejects a temporary owner
static_assert(!permits_direct_temporary_rowwise<lifetime_matrix>);
// direct columnwise-adaptor construction rejects a temporary owner
static_assert(!permits_direct_temporary_colwise<lifetime_matrix>);
// direct rowwise-adaptor construction rejects a const temporary owner
static_assert(!permits_direct_temporary_rowwise<lifetime_const_matrix>);
// direct columnwise-adaptor construction rejects a const temporary owner
static_assert(!permits_direct_temporary_colwise<lifetime_const_matrix>);
// a rowwise adaptor over mutable storage advertises writability
static_assert(lifetime_rowwise::ReadOnly == 0);
// a columnwise adaptor over mutable storage advertises writability
static_assert(lifetime_colwise::ReadOnly == 0);
// a rowwise adaptor over a const owner advertises read-only access
static_assert(lifetime_const_owner_rowwise::ReadOnly == 1);
// a columnwise adaptor over a const owner advertises read-only access
static_assert(lifetime_const_owner_colwise::ReadOnly == 1);
// a rowwise adaptor over const storage advertises read-only access
static_assert(lifetime_const_scalar_view_rowwise::ReadOnly == 1);
// a columnwise adaptor over const storage advertises read-only access
static_assert(lifetime_const_scalar_view_colwise::ReadOnly == 1);
// a row-sum expression exposes read-only computed coefficients
static_assert(lifetime_rowwise_sum::ReadOnly == 1);
// a column-sum expression exposes read-only computed coefficients
static_assert(lifetime_colwise_sum::ReadOnly == 1);
// row reduction of const double storage retains the double result type
static_assert(std::is_same_v<typename lifetime_const_scalar_view_rowwise_sum::Scalar, double>);
// rowwise assignment cannot write into a const owner
static_assert(!permits_vectorwise_assignment<lifetime_const_owner_rowwise, lifetime_rowwise_rhs>);
// rowwise addition assignment cannot write into a const owner
static_assert(!permits_vectorwise_addition_assignment<lifetime_const_owner_rowwise, lifetime_rowwise_rhs>);
// rowwise subtraction assignment cannot write into a const owner
static_assert(!permits_vectorwise_subtraction_assignment<lifetime_const_owner_rowwise, lifetime_rowwise_rhs>);
// a temporary rowwise adaptor cannot bypass const-owner assignment restrictions
static_assert(!permits_temporary_vectorwise_assignment<lifetime_const_owner_rowwise, lifetime_rowwise_rhs>);
// a temporary rowwise adaptor cannot bypass const-owner addition restrictions
static_assert(!permits_temporary_vectorwise_addition_assignment<lifetime_const_owner_rowwise, lifetime_rowwise_rhs>);
// a temporary rowwise adaptor cannot bypass const-owner subtraction restrictions
static_assert(!permits_temporary_vectorwise_subtraction_assignment<lifetime_const_owner_rowwise, lifetime_rowwise_rhs>);
// columnwise assignment cannot write into a const owner
static_assert(!permits_vectorwise_assignment<lifetime_const_owner_colwise, lifetime_colwise_rhs>);
// columnwise addition assignment cannot write into a const owner
static_assert(!permits_vectorwise_addition_assignment<lifetime_const_owner_colwise, lifetime_colwise_rhs>);
// columnwise subtraction assignment cannot write into a const owner
static_assert(!permits_vectorwise_subtraction_assignment<lifetime_const_owner_colwise, lifetime_colwise_rhs>);
// a temporary columnwise adaptor cannot bypass const-owner assignment restrictions
static_assert(!permits_temporary_vectorwise_assignment<lifetime_const_owner_colwise, lifetime_colwise_rhs>);
// a temporary columnwise adaptor cannot bypass const-owner addition restrictions
static_assert(!permits_temporary_vectorwise_addition_assignment<lifetime_const_owner_colwise, lifetime_colwise_rhs>);
// a temporary columnwise adaptor cannot bypass const-owner subtraction restrictions
static_assert(!permits_temporary_vectorwise_subtraction_assignment<lifetime_const_owner_colwise, lifetime_colwise_rhs>);
// partially dynamic rowwise assignment accepts a compatible fixed source extent
static_assert(permits_vectorwise_assignment<lifetime_partial_rowwise, lifetime_valid_partial_row_rhs>);
// partially dynamic rowwise assignment rejects a conflicting fixed source extent
static_assert(!permits_vectorwise_assignment<lifetime_partial_rowwise, lifetime_invalid_partial_row_rhs>);
// partially dynamic columnwise assignment accepts a compatible fixed source extent
static_assert(permits_vectorwise_assignment<lifetime_partial_colwise, lifetime_valid_partial_col_rhs>);
// partially dynamic columnwise assignment rejects a conflicting fixed source extent
static_assert(!permits_vectorwise_assignment<lifetime_partial_colwise, lifetime_invalid_partial_col_rhs>);
// fixed rowwise assignment rejects an incompatible source extent
static_assert(!permits_vectorwise_assignment<lifetime_rowwise, lifetime_invalid_fixed_row_rhs>);
// fixed columnwise assignment rejects an incompatible source extent
static_assert(!permits_vectorwise_assignment<lifetime_colwise, lifetime_invalid_fixed_col_rhs>);
// rowwise comparison rejects a source conflicting with its known static extent
static_assert(!permits_vectorwise_comparison<lifetime_partial_rowwise, lifetime_invalid_partial_row_rhs>);
// columnwise comparison rejects a source conflicting with its known static extent
static_assert(!permits_vectorwise_comparison<lifetime_partial_colwise, lifetime_invalid_partial_col_rhs>);
// temporary rowwise mutation returns values rather than dangling adaptor references
static_assert(vectorwise_temporary_mutations_return_values_v<lifetime_rowwise, lifetime_rowwise_rhs>);
// temporary columnwise mutation returns values rather than dangling adaptor references
static_assert(vectorwise_temporary_mutations_return_values_v<lifetime_colwise, lifetime_colwise_rhs>);
// a custom reduction stores its operation by value rather than by reference
static_assert(!std::is_reference_v<typename partial_reduction_traits<lifetime_custom_reduction>::operation_type>);
// a reduction can own a move-only operation
static_assert(std::is_move_constructible_v<lifetime_move_only_reduction>);
// reduction rejects an operation requiring mutable coefficient references
static_assert(!permits_vectorwise_reduction<lifetime_rowwise, lifetime_matrix, mutable_value_reduction_op>);
// reduction rejects an operation whose call requires a mutable operation object
static_assert(!permits_vectorwise_reduction<lifetime_rowwise, lifetime_matrix, mutable_reduction_op>);

template <int StorageOrder> void check_owner_behavior() {
    using fixed_matrix = Matrix<int, 2, 3, StorageOrder>;
    using dynamic_column = Matrix<int, Dynamic, 1, StorageOrder>;

    constexpr int fixed_input[6] {1, 2, 3, 4, 5, 6};
    constexpr fixed_matrix matrix(fixed_input);
    // constant evaluation retains the matrix's fixed two rows
    static_assert(matrix.rows() == 2);
    // constant evaluation retains the matrix's fixed three columns
    static_assert(matrix.cols() == 3);
    // initializer order places the sixth value at the final logical coordinate
    static_assert(matrix(1, 2) == 6);

    const std::array<int, 6> expected_storage =
      StorageOrder == RowMajor ? std::array<int, 6> {1, 2, 3, 4, 5, 6} : std::array<int, 6> {1, 4, 2, 5, 3, 6};
    // raw storage follows the expected physical order for the selected layout
    for (int i = 0; i < matrix.size(); ++i) { EXPECT_EQ(matrix.data()[i], expected_storage[i]); }

    const std::vector<int> input {1, 2, 3, 4, 5, 6};
    const fixed_matrix from_vector(input);
    // construction from a vector reproduces the initializer-constructed matrix
    EXPECT_EQ(from_vector, matrix);

    Matrix<int, Dynamic, 3, StorageOrder> dynamic_rows(2, 3);
    // a partially dynamic owner initially exposes six contiguous coefficients
    EXPECT_EQ(dynamic_rows.end() - dynamic_rows.begin(), 6);
    dynamic_rows.resize(4, 3);
    // resizing the dynamic row axis updates the row count to four
    EXPECT_EQ(dynamic_rows.rows(), 4);
    // resizing the dynamic row axis preserves the fixed three columns
    EXPECT_EQ(dynamic_rows.cols(), 3);
    // the resized row-dynamic owner exposes twelve contiguous coefficients
    EXPECT_EQ(dynamic_rows.end() - dynamic_rows.begin(), 12);

    Matrix<int, 2, Dynamic, StorageOrder> dynamic_cols(2, 3);
    // a column-dynamic owner initially exposes six contiguous coefficients
    EXPECT_EQ(dynamic_cols.end() - dynamic_cols.begin(), 6);
    dynamic_cols.resize(2, 4);
    // resizing the dynamic column axis preserves the fixed two rows
    EXPECT_EQ(dynamic_cols.rows(), 2);
    // resizing the dynamic column axis updates the column count to four
    EXPECT_EQ(dynamic_cols.cols(), 4);
    // the resized column-dynamic owner exposes eight contiguous coefficients
    EXPECT_EQ(dynamic_cols.end() - dynamic_cols.begin(), 8);

    dynamic_column column(std::vector<int> {1, 2, 3});
    // a dynamic column vector adopts its initial three-row length
    EXPECT_EQ(column.rows(), 3);
    // a dynamic column vector retains its single column
    EXPECT_EQ(column.cols(), 1);
    column = {4, 5};
    // assignment resizes the dynamic vector to the two-entry source
    EXPECT_EQ(column.rows(), 2);
    // vector assignment preserves the column orientation
    EXPECT_EQ(column.cols(), 1);
    // vector assignment copies the first source coefficient
    EXPECT_EQ(column[0], 4);
    // vector assignment copies the second source coefficient
    EXPECT_EQ(column[1], 5);

    const Matrix<int, 1, 3, StorageOrder> row({7, 8, 9});
    const Matrix<int, 3, 1, StorageOrder> fixed_column_from_row(row);
    // copying a row into a fixed column vector preserves its first coefficient
    EXPECT_EQ(fixed_column_from_row[0], 7);
    // copying a row into a fixed column vector preserves its middle coefficient
    EXPECT_EQ(fixed_column_from_row[1], 8);
    // copying a row into a fixed column vector preserves its final coefficient
    EXPECT_EQ(fixed_column_from_row[2], 9);

    const dynamic_column column_from_row(row);
    // copying a row into a dynamic column vector adopts length three
    EXPECT_EQ(column_from_row.rows(), 3);
    // copying a row into a dynamic column vector retains one column
    EXPECT_EQ(column_from_row.cols(), 1);
    // the reoriented dynamic copy preserves the first coefficient
    EXPECT_EQ(column_from_row[0], 7);
    // the reoriented dynamic copy preserves the middle coefficient
    EXPECT_EQ(column_from_row[1], 8);
    // the reoriented dynamic copy preserves the final coefficient
    EXPECT_EQ(column_from_row[2], 9);

    const Matrix<int, 1, 1, StorageOrder> scalar_vector(11);
    const dynamic_column column_from_scalar(scalar_vector);
    // copying a scalar expression into a column vector produces one row
    EXPECT_EQ(column_from_scalar.rows(), 1);
    // the singleton vector preserves the scalar expression's value
    EXPECT_EQ(column_from_scalar[0], 11);

    constexpr int OtherStorageOrder = StorageOrder == RowMajor ? ColMajor : RowMajor;
    const Matrix<int, 2, 3, OtherStorageOrder> other_order(matrix);
    // cross-order copying preserves an off-diagonal logical coefficient
    EXPECT_EQ(other_order(0, 1), 2);
    // cross-order copying preserves the final logical coefficient
    EXPECT_EQ(other_order(1, 2), 6);
}

template <int StorageOrder> void check_numeric_view_behavior() {
    using fixed_view = MatrixView<int, 2, 3, StorageOrder>;
    // assignment to a temporary numeric view returns a value with its storage binding
    static_assert(
      std::is_same_v<decltype(std::declval<fixed_view&&>() = std::declval<const fixed_view&>()), fixed_view>);

    std::array<int, 6> view_storage {};
    fixed_view view(view_storage.data());
    view(0, 1) = 7;
    constexpr int view_index = StorageOrder == RowMajor ? 1 : 2;
    // writing a view coordinate updates the expected physical storage offset
    EXPECT_EQ(view_storage[view_index], 7);
    const auto& const_view = view;
    // a const view object returns a const scalar reference
    static_assert(std::is_same_v<decltype(const_view(0, 0)), const int&>);

    std::array<int, 6> source_storage {};
    fixed_view source_view(source_storage.data());
    const auto source_alias = source_view;
    // copy construction of a view shares its source storage binding
    EXPECT_EQ(source_alias.data(), source_view.data());
    source_view(1, 2) = 11;
    int* const destination = view.data();
    view = source_view;
    // view-to-view assignment preserves the destination's storage address
    EXPECT_EQ(view.data(), destination);
    // view-to-view assignment copies the source's final coefficient
    EXPECT_EQ(view(1, 2), 11);

    const Matrix<int, 2, 3, StorageOrder> matrix({1, 2, 3, 4, 5, 6});
    view = matrix;
    // owner-to-view assignment preserves the destination's storage address
    EXPECT_EQ(view.data(), destination);
    // owner-to-view assignment copies the complete logical matrix
    EXPECT_EQ(view, matrix);

    std::array<int, 6> temporary_destination {};
    const auto assigned_temporary = fixed_view(temporary_destination.data()) = source_view;
    // assignment to a temporary view returns the original destination binding
    EXPECT_EQ(assigned_temporary.data(), temporary_destination.data());
    // the returned temporary view exposes the copied final coefficient
    EXPECT_EQ(assigned_temporary(1, 2), 11);

    std::array<int, 3> vector_storage {3, 2, 1};
    MatrixView<int, 1, Dynamic, StorageOrder> row_view(vector_storage.data(), 3);
    MatrixView<int, Dynamic, 1, StorageOrder> column_view(vector_storage.data(), 3);
    // a runtime row view retains one row
    EXPECT_EQ(row_view.rows(), 1);
    // a runtime row view adopts its three-coefficient length
    EXPECT_EQ(row_view.cols(), 3);
    // a runtime row view reads its final bound coefficient
    EXPECT_EQ(row_view(0, 2), 1);
    // a runtime column view adopts its three-coefficient length
    EXPECT_EQ(column_view.rows(), 3);
    // a runtime column view retains one column
    EXPECT_EQ(column_view.cols(), 1);
    // a runtime column view reads its final bound coefficient
    EXPECT_EQ(column_view(2, 0), 1);
}

template <int StorageOrder> void check_arithmetic_expression_nesting() {
    using matrix_type = Matrix<double, 2, 3, StorageOrder>;
    const matrix_type matrix({-4.0, 0.0, 2.0, 1.0, -3.0, 5.0});

    const matrix_type chained_sum = (matrix + matrix) + matrix;
    // a stored addition chain evaluates every coefficient to the explicit reference
    EXPECT_EQ(chained_sum, (matrix_type({-12.0, 0.0, 6.0, 3.0, -9.0, 15.0})));

    const Matrix<double, 3, 2, StorageOrder> transposed_sum = (matrix + matrix).transpose();
    // transposing a stored sum exchanges its coordinates correctly
    EXPECT_EQ(transposed_sum, (Matrix<double, 3, 2, StorageOrder>({-8.0, 2.0, 0.0, -6.0, 4.0, 10.0})));

    const Matrix<double, 2, 2, StorageOrder> gram = matrix * matrix.transpose();
    // a matrix-transpose product evaluates the expected Gram matrix
    EXPECT_EQ(gram, (Matrix<double, 2, 2, StorageOrder>({20.0, 6.0, 6.0, 35.0})));

    using square_matrix_type = Matrix<double, 2, 2, StorageOrder>;
    const square_matrix_type square({1.0, 2.0, 3.0, 4.0});
    const square_matrix_type symmetric = (square + square).symm_part();
    const square_matrix_type skew = (square + square).skew_part();
    // the symmetric-part expression matches the explicit reflected sum
    EXPECT_EQ(symmetric, (square_matrix_type({2.0, 5.0, 5.0, 8.0})));
    // the skew-part expression matches the explicit antisymmetric difference
    EXPECT_EQ(skew, (square_matrix_type({0.0, -1.0, 1.0, 0.0})));

    using vector_type = Matrix<double, 3, 1, StorageOrder>;
    const vector_type a({1.0, 0.0, 0.0});
    const vector_type b({0.0, 1.0, 0.0});
    const vector_type c({0.0, 0.0, 1.0});
    const vector_type cross = (a + b).cross(c - b);
    // a stored cross-product chain evaluates to the explicit three-component reference
    EXPECT_EQ(cross, (vector_type({1.0, -1.0, -1.0})));

    using cross_expression = decltype(a.cross(b));
    // the cross-product expression retains the requested storage order
    static_assert(cross_expression::StorageOrder == StorageOrder);
}

template <int StorageOrder> void check_assignment_alias_materialization() {
    using matrix_type = Matrix<double, 2, 2, StorageOrder>;

    matrix_type aliased({1.0, 2.0, 3.0, 4.0});
    aliased += aliased.transpose();
    // aliased addition reads the original transpose before overwriting the destination
    EXPECT_EQ(aliased, (matrix_type({2.0, 5.0, 5.0, 8.0})));

    aliased = matrix_type({1.0, 2.0, 3.0, 4.0});
    aliased -= aliased.transpose();
    // aliased subtraction reads the original transpose before overwriting the destination
    EXPECT_EQ(aliased, (matrix_type({0.0, -1.0, 1.0, 0.0})));

    aliased = matrix_type({1.0, 2.0, 3.0, 4.0});
    aliased *= aliased;
    // self multiplication uses the original matrix on both sides
    EXPECT_EQ(aliased, (matrix_type({7.0, 10.0, 15.0, 22.0})));

    matrix_type overlapping({1.0, 2.0, 3.0, 4.0});
    overlapping.template block<2, 2>(0, 0) = overlapping.template block<2, 2>(0, 0).transpose();
    // overlapping transpose assignment preserves every original coefficient
    EXPECT_EQ(overlapping, (matrix_type({1.0, 3.0, 2.0, 4.0})));

    const matrix_type source_owner({1.0, 2.0, 3.0, 4.0});
    double destination_data[4] {};
    MatrixView<const double, 2, 2, StorageOrder> source(source_owner.data());
    MatrixView<double, 2, 2, StorageOrder> destination(destination_data);
    destination = source;
    // assignment between overlapping storage views reproduces the source snapshot
    EXPECT_EQ(destination, source_owner);
}

template <int StorageOrder> void check_coefficientwise_behavior() {
    using matrix_type = Matrix<double, 2, 2, StorageOrder>;
    const matrix_type source({1.0, 4.0, 9.0, 16.0});

    matrix_type scalar_assigned;
    scalar_assigned.cwise() = 3.0;
    // coefficientwise scalar assignment fills every coefficient with three
    EXPECT_EQ(scalar_assigned, matrix_type(3.0));

    auto scalar_expression = source.cwise() + 1.0;
    const matrix_type shifted = scalar_expression;
    // coefficientwise scalar addition shifts the squared source values by one
    EXPECT_EQ(shifted, (matrix_type({2.0, 5.0, 10.0, 17.0})));
    const matrix_type reverse_difference = 20.0 - source.cwise();
    // scalar-minus-expression subtraction uses the intended operand order
    EXPECT_EQ(reverse_difference, (matrix_type({19.0, 16.0, 11.0, 4.0})));
    const matrix_type reciprocal = 144.0 / source.cwise();
    // scalar-divided-by-expression division produces the explicit reciprocal-scaled values
    EXPECT_EQ(reciprocal, (matrix_type({144.0, 36.0, 16.0, 9.0})));

    const Matrix<int, 1, 2, StorageOrder> integers({2, 4});
    const Matrix<double, 1, 2, StorageOrder> inverses = integers.cwise().inv();
    // integer coefficientwise inverse preserves fractional results through promotion
    EXPECT_EQ(inverses, (Matrix<double, 1, 2, StorageOrder>({0.5, 0.25})));

    const Matrix<bool, 2, 2, StorageOrder> comparison = source.cwise() < 10.0;
    // coefficientwise comparison materializes the explicit Boolean mask
    EXPECT_EQ(comparison, (Matrix<bool, 2, 2, StorageOrder>({true, true, true, false})));

    const Matrix<double, 1, 3, StorageOrder> row({1.0, 2.0, 3.0});
    const Matrix<double, 1, 3, StorageOrder> shifted_row = row.cwise() + 1.0;
    // coefficientwise row-vector arithmetic preserves vector indexing
    EXPECT_DOUBLE_EQ(shifted_row[2], 4.0);

    auto expression = (source + source).cwise().sqrt().mwise();
    const matrix_type expression_value = expression;
    // stored coefficientwise transformations match the explicit square-root reference
    EXPECT_TRUE(almost_equal(
      expression_value, matrix_type({fdapde::sqrt(2.0), fdapde::sqrt(8.0), fdapde::sqrt(18.0), fdapde::sqrt(32.0)})));

    double mutable_view_data[4] {1.0, 2.0, 3.0, 4.0};
    auto temporary_view_cwise = MatrixView<double, 2, 2, StorageOrder>(mutable_view_data).cwise();
    temporary_view_cwise += 1.0;
    // mutation through a temporary view adaptor updates every external scalar
    for (int i = 0; i < 4; ++i) { EXPECT_DOUBLE_EQ(mutable_view_data[i], i + 2.0); }

    auto temporary_const_view_cwise = MatrixView<const double, 2, 2, StorageOrder>(source.data()).cwise();
    const matrix_type const_view_value = temporary_const_view_cwise;
    // a const-storage adaptor materializes the original source matrix
    EXPECT_EQ(const_view_value, source);

    Matrix<double, Dynamic, 2, StorageOrder> partial_dynamic(2, 2);
    partial_dynamic = source;
    const Matrix<double, Dynamic, 2, StorageOrder> partial_shifted = partial_dynamic.cwise() + 1.0;
    // partially dynamic coefficientwise arithmetic matches the fixed-shape reference
    EXPECT_EQ(partial_shifted, shifted);

    matrix_type wrapped({1.0, 2.0, 3.0, 4.0});
    auto wrapped_cwise = wrapped.cwise();
    auto wrapped_mwise = wrapped_cwise.mwise();
    wrapped_mwise(0, 0) = 7.0;
    // writing through the plain adaptor updates its wrapped matrix coefficient
    EXPECT_DOUBLE_EQ(wrapped(0, 0), 7.0);

    const auto make_applied = [&source]() {
        auto offset = [value = 0.5](double x) { return x + value; };
        return source.cwise().apply(offset);
    };
    const matrix_type applied = make_applied();
    // a custom applied operation adds one half to every coefficient
    EXPECT_EQ(applied, (matrix_type({1.5, 4.5, 9.5, 16.5})));

    matrix_type add_alias({1.0, 2.0, 3.0, 4.0});
    add_alias.cwise() += add_alias.transpose().cwise();
    // aliased coefficientwise addition snapshots the transposed operand
    EXPECT_EQ(add_alias, (matrix_type({2.0, 5.0, 5.0, 8.0})));

    matrix_type subtract_alias({1.0, 2.0, 3.0, 4.0});
    subtract_alias.cwise() -= subtract_alias.transpose().cwise();
    // aliased coefficientwise subtraction snapshots the transposed operand
    EXPECT_EQ(subtract_alias, (matrix_type({0.0, -1.0, 1.0, 0.0})));

    matrix_type multiply_alias({1.0, 2.0, 3.0, 4.0});
    multiply_alias.cwise() *= multiply_alias.transpose().cwise();
    // aliased coefficientwise multiplication snapshots the transposed operand
    EXPECT_EQ(multiply_alias, (matrix_type({1.0, 6.0, 6.0, 16.0})));

    matrix_type divide_alias({1.0, 2.0, 3.0, 4.0});
    divide_alias.cwise() /= divide_alias.transpose().cwise();
    // aliased coefficientwise division snapshots the transposed operand
    EXPECT_EQ(divide_alias, (matrix_type({1.0, 2.0 / 3.0, 3.0 / 2.0, 1.0})));
}

template <int StorageOrder> void check_vectorwise_behavior() {
    using matrix_type = Matrix<double, 2, 3, StorageOrder>;
    using row_reduction_type = Matrix<double, 2, 1, StorageOrder>;
    using col_reduction_type = Matrix<double, 1, 3, StorageOrder>;
    const matrix_type source({1.0, 2.0, 3.0, 4.0, 5.0, 6.0});

    auto stored_rows = (source + source).rowwise();
    auto stored_row_sums = stored_rows.sum();
    const row_reduction_type row_sums = stored_row_sums;
    // rowwise reduction of a stored expression produces the two expected sums
    EXPECT_EQ(row_sums, (row_reduction_type({12.0, 30.0})));

    auto stored_cols = (source + source).colwise();
    auto stored_col_sums = stored_cols.sum();
    const col_reduction_type col_sums = stored_col_sums;
    // columnwise reduction of a stored expression produces the three expected sums
    EXPECT_EQ(col_sums, (col_reduction_type({10.0, 14.0, 18.0})));

    matrix_type row_view_owner({1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    auto temporary_view_rows = MatrixView<double, 2, 3, StorageOrder>(row_view_owner.data()).rowwise();
    temporary_view_rows += row_reduction_type({10.0, 20.0});
    // rowwise broadcast mutation updates the bound owner's rows
    EXPECT_EQ(row_view_owner, (matrix_type({11.0, 12.0, 13.0, 24.0, 25.0, 26.0})));

    matrix_type col_view_owner({1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    auto temporary_view_cols = MatrixView<double, 2, 3, StorageOrder>(col_view_owner.data()).colwise();
    temporary_view_cols += col_reduction_type({10.0, 20.0, 30.0});
    // columnwise broadcast mutation updates the bound owner's columns
    EXPECT_EQ(col_view_owner, (matrix_type({11.0, 22.0, 33.0, 14.0, 25.0, 36.0})));

    auto temporary_const_view_rows = MatrixView<const double, 2, 3, StorageOrder>(source.data()).rowwise();
    auto temporary_const_view_sum = temporary_const_view_rows.sum();
    const row_reduction_type const_view_sums = temporary_const_view_sum;
    // rowwise reduction over a const view preserves the expected sums
    EXPECT_EQ(const_view_sums, (row_reduction_type({6.0, 15.0})));

    stateful_reduction_op reducer {0.25};
    auto copied_reducer = source.rowwise().redux(source, 0.0, reducer);
    reducer.offset = 100.0;
    const row_reduction_type copied_reducer_result = copied_reducer;
    // a copied stateful reducer preserves its operation state
    EXPECT_EQ(copied_reducer_result, (row_reduction_type({6.75, 15.75})));

    auto move_only_reducer = source.rowwise().redux(source, 0.0, move_only_reduction_op {0.25});
    const row_reduction_type move_only_reducer_result = move_only_reducer;
    // a moved move-only reducer preserves its operation state
    EXPECT_EQ(move_only_reducer_result, (row_reduction_type({6.75, 15.75})));

    const auto make_reduction = [&source]() {
        auto local_reducer = [offset = 0.5](double accumulated, double value) { return accumulated + value + offset; };
        return source.rowwise().redux(source, 0.0, local_reducer);
    };
    auto stored_reduction = make_reduction();
    const row_reduction_type stored_reduction_result = stored_reduction;
    // a stored custom reduction preserves its bound operation and source
    EXPECT_EQ(stored_reduction_result, (row_reduction_type({7.5, 16.5})));

    matrix_type row_assign_alias({1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    row_assign_alias.rowwise() = row_assign_alias.col(0) + row_assign_alias.col(1);
    // aliased rowwise assignment evaluates its source before broadcasting
    EXPECT_EQ(row_assign_alias, (matrix_type({3.0, 3.0, 3.0, 9.0, 9.0, 9.0})));

    matrix_type row_add_alias({1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    row_add_alias.rowwise() += row_add_alias.col(0) + row_add_alias.col(1);
    // aliased rowwise addition evaluates its source before updating the owner
    EXPECT_EQ(row_add_alias, (matrix_type({4.0, 5.0, 6.0, 13.0, 14.0, 15.0})));

    matrix_type row_subtract_alias({1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    row_subtract_alias.rowwise() -= row_subtract_alias.col(0) + row_subtract_alias.col(1);
    // aliased rowwise subtraction evaluates its source before updating the owner
    EXPECT_EQ(row_subtract_alias, (matrix_type({-2.0, -1.0, 0.0, -5.0, -4.0, -3.0})));

    matrix_type col_assign_alias({1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    col_assign_alias.colwise() = col_assign_alias.row(0) + col_assign_alias.row(1);
    // aliased columnwise assignment evaluates its source before broadcasting
    EXPECT_EQ(col_assign_alias, (matrix_type({5.0, 7.0, 9.0, 5.0, 7.0, 9.0})));

    matrix_type col_add_alias({1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    col_add_alias.colwise() += col_add_alias.row(0) + col_add_alias.row(1);
    // aliased columnwise addition evaluates its source before updating the owner
    EXPECT_EQ(col_add_alias, (matrix_type({6.0, 9.0, 12.0, 9.0, 12.0, 15.0})));

    matrix_type col_subtract_alias({1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    col_subtract_alias.colwise() -= col_subtract_alias.row(0) + col_subtract_alias.row(1);
    // aliased columnwise subtraction evaluates its source before updating the owner
    EXPECT_EQ(col_subtract_alias, (matrix_type({-4.0, -5.0, -6.0, -1.0, -2.0, -3.0})));
}

template <int StorageOrder> void check_block_view_behavior() {
    using matrix_type = Matrix<double, 3, 4, StorageOrder>;
    matrix_type matrix({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0});

    Matrix<double, 1, 3, StorageOrder> row_vector({1.0, 2.0, 3.0});
    auto scalar_col = row_vector.col(2);
    // a column selected from a row vector has one row
    EXPECT_EQ(scalar_col.rows(), 1);
    // a column selected from a row vector has one column
    EXPECT_EQ(scalar_col.cols(), 1);
    // the scalar column view reads the selected vector coefficient
    EXPECT_DOUBLE_EQ(scalar_col(0, 0), 3.0);
    scalar_col(0, 0) = 4.0;
    // writing through the scalar column view updates the row vector
    EXPECT_DOUBLE_EQ(row_vector(0, 2), 4.0);

    Matrix<double, 3, 1, StorageOrder> column_vector({1.0, 2.0, 3.0});
    auto scalar_row = column_vector.row(2);
    // a row selected from a column vector has one row
    EXPECT_EQ(scalar_row.rows(), 1);
    // a row selected from a column vector has one column
    EXPECT_EQ(scalar_row.cols(), 1);
    // the scalar row view reads the selected vector coefficient
    EXPECT_DOUBLE_EQ(scalar_row(0, 0), 3.0);
    scalar_row(0, 0) = 4.0;
    // writing through the scalar row view updates the column vector
    EXPECT_DOUBLE_EQ(column_vector(2, 0), 4.0);

    const Matrix<double, 2, 2, StorageOrder> fixed = matrix.template block<2, 2>(1, 1);
    // a fixed block evaluates to the expected interior rectangle
    EXPECT_EQ(fixed, (Matrix<double, 2, 2, StorageOrder>({6.0, 7.0, 10.0, 11.0})));
    const Matrix<double, 3, 2, StorageOrder> dynamic = matrix.block(0, 2, 3, 2);
    // a runtime block evaluates to the expected right-hand rectangle
    EXPECT_EQ(dynamic, (Matrix<double, 3, 2, StorageOrder>({3.0, 4.0, 7.0, 8.0, 11.0, 12.0})));

    auto safe_expression_block = (matrix + matrix).template block<1, 2>(0, 0);
    const Matrix<double, 1, 2, StorageOrder> safe_expression_value = safe_expression_block;
    // a block of a temporary expression retains its computed coefficients
    EXPECT_EQ(safe_expression_value, (Matrix<double, 1, 2, StorageOrder>({2.0, 4.0})));

    auto temporary_const_view_row =
      MatrixView<const double, Dynamic, Dynamic, StorageOrder>(matrix.data(), 3, 4).row(1);
    const Matrix<double, 1, 4, StorageOrder> temporary_view_value = temporary_const_view_row;
    // a row of a temporary view retains the external storage binding
    EXPECT_EQ(temporary_view_value, (Matrix<double, 1, 4, StorageOrder>({5.0, 6.0, 7.0, 8.0})));

    double mutable_view_data[4] {1.0, 2.0, 3.0, 4.0};
    MatrixView<double, 2, 2, StorageOrder> mutable_view(mutable_view_data);
    auto temporary_mutable_view_row = MatrixView<double, 2, 2, StorageOrder>(mutable_view_data).row(1);
    // a row retained from a temporary mutable view remains writable
    static_assert(decltype(temporary_mutable_view_row)::ReadOnly == 0);
    temporary_mutable_view_row(0, 0) = 9.0;
    // writing the retained row updates the original view's storage
    EXPECT_DOUBLE_EQ(mutable_view(1, 0), 9.0);

    auto computed = matrix + matrix;
    auto computed_row = computed.row(0);
    // iterators over a computed row satisfy bidirectional iterator requirements
    static_assert(std::bidirectional_iterator<typename decltype(computed_row)::iterator>);
    double computed_sum = 0.0;
    for (const double value : computed_row) { computed_sum += value; }
    // iteration over the computed row sums the expected expression values
    EXPECT_DOUBLE_EQ(computed_sum, 20.0);

    auto row = matrix.row(1);
    int count = 0;
    double sum = 0.0;
    for (const double value : row) {
        ++count;
        sum += value;
    }
    // row iteration visits exactly the row's coefficient count
    EXPECT_EQ(count, row.size());
    // row iteration sums the four selected owner coefficients
    EXPECT_DOUBLE_EQ(sum, 26.0);

    matrix.row(2) = {30.0, 31.0, 32.0, 33.0};
    // initializer assignment to a temporary row writes the complete selected row
    EXPECT_EQ(matrix.row(2), (Matrix<double, 1, 4, StorageOrder>({30.0, 31.0, 32.0, 33.0})));

    matrix_type named_assignment({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0});
    auto destination = named_assignment.template block<2, 2>(0, 0);
    auto source = named_assignment.template block<2, 2>(1, 2);
    destination = source;
    // named block assignment changes only the expected destination rectangle
    EXPECT_EQ(named_assignment, (matrix_type({7.0, 8.0, 3.0, 4.0, 11.0, 12.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0})));

    matrix_type temporary_assignment({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0});
    temporary_assignment.template block<2, 2>(0, 0) = temporary_assignment.template block<2, 2>(1, 2);
    // temporary block assignment changes the same destination rectangle
    EXPECT_EQ(temporary_assignment, (matrix_type({7.0, 8.0, 3.0, 4.0, 11.0, 12.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0})));

    matrix_type view_owner({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0});
    MatrixView<double, 3, 4, StorageOrder> view(view_owner.data());
    double* const binding = view.data();
    auto view_destination = view.template block<2, 2>(0, 0);
    auto view_source = view.template block<2, 2>(1, 2);
    view_destination = view_source;
    // assignment through a block preserves its enclosing view's storage binding
    EXPECT_EQ(view.data(), binding);
    // assignment through a view-backed block updates only the selected owner rectangle
    EXPECT_EQ(view_owner, (matrix_type({7.0, 8.0, 3.0, 4.0, 11.0, 12.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0})));
}

template <int StorageOrder> void check_reshape_behavior() {
    using source_matrix = Matrix<double, 2, 3, StorageOrder>;
    using target_matrix = Matrix<double, 3, 2, StorageOrder>;
    source_matrix matrix({1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    const target_matrix expected = StorageOrder == RowMajor ? target_matrix({1.0, 2.0, 3.0, 4.0, 5.0, 6.0}) :
                                                              target_matrix({1.0, 5.0, 4.0, 3.0, 2.0, 6.0});

    // static reshape preserves the physical coefficient sequence in the new shape
    EXPECT_EQ((target_matrix(matrix.template reshape<3, 2>())), expected);
    // runtime reshape matches the same physical-order reference
    EXPECT_EQ((target_matrix(matrix.reshape(3, 2))), expected);
    const auto row = matrix.template reshape<1, 6>();
    const auto column = matrix.template reshape<6>();
    for (int i = 0; i < matrix.size(); ++i) {
        // row reshape follows the source's raw physical coefficient sequence
        EXPECT_DOUBLE_EQ(row[i], matrix.data()[i]);
        // column reshape follows the source's raw physical coefficient sequence
        EXPECT_DOUBLE_EQ(column[i], matrix.data()[i]);
    }
    const ReshapeOp<1, Dynamic, source_matrix> direct_row(matrix, matrix.size());
    // a directly constructed row reshape has one row
    EXPECT_EQ(direct_row.rows(), 1);
    // a directly constructed row reshape spans all source coefficients
    EXPECT_EQ(direct_row.cols(), matrix.size());
    // direct row-reshape indexing follows the raw physical coefficient sequence
    for (int i = 0; i < matrix.size(); ++i) { EXPECT_DOUBLE_EQ(direct_row[i], matrix.data()[i]); }

    auto expression_reshape = (matrix + matrix).template reshape<3, 2>();
    // reshape of a stored expression evaluates twice the expected coefficient sequence
    EXPECT_EQ((target_matrix(expression_reshape)), (target_matrix(expected + expected)));

    auto temporary_const_view_reshape =
      MatrixView<const double, 2, 3, StorageOrder>(matrix.data()).template reshape<3, 2>();
    // reshape of a temporary const view retains the expected coefficients
    EXPECT_EQ((target_matrix(temporary_const_view_reshape)), expected);

    double mutable_view_data[6] {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    auto temporary_mutable_view_reshape =
      MatrixView<double, Dynamic, Dynamic, StorageOrder>(mutable_view_data, 2, 3).reshape(3, 2);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 2; ++j) {
            const int k = StorageOrder == RowMajor ? i * 2 + j : j * 3 + i;
            // reshape of a temporary mutable view reads the mapped external scalar
            EXPECT_DOUBLE_EQ(temporary_mutable_view_reshape(i, j), mutable_view_data[k]);
        }
    }
    constexpr int mutated_index = StorageOrder == RowMajor ? 1 : 3;
    temporary_mutable_view_reshape(0, 1) = 19.0;
    // writing the retained reshape updates the expected external storage offset
    EXPECT_DOUBLE_EQ(mutable_view_data[mutated_index], 19.0);

    auto temporary_const_view_vector =
      MatrixView<const double, Dynamic, Dynamic, StorageOrder>(matrix.data(), 2, 3).reshape(6);
    for (int i = 0; i < matrix.size(); ++i) {
        // vector reshape of a temporary const view preserves physical order
        EXPECT_DOUBLE_EQ(temporary_const_view_vector[i], matrix.data()[i]);
    }

    source_matrix named_destination;
    const source_matrix named_source({6.0, 5.0, 4.0, 3.0, 2.0, 1.0});
    auto destination_reshape = named_destination.template reshape<1, 6>();
    auto source_reshape = named_source.template reshape<1, 6>();
    destination_reshape = source_reshape;
    // assignment through a named reshape copies the complete source
    EXPECT_EQ(named_destination, named_source);

    source_matrix reoriented_destination;
    reoriented_destination.template reshape<1, 6>() = named_source.template reshape<6>();
    // assignment between differently oriented reshapes preserves the intended source sequence
    EXPECT_EQ(reoriented_destination, named_source);

    source_matrix temporary_destination;
    temporary_destination.template reshape<1, 6>() = named_source.template reshape<1, 6>();
    // assignment through a temporary reshape copies the complete source
    EXPECT_EQ(temporary_destination, named_source);
    temporary_destination.template reshape<2, 3>() = matrix;
    // subsequent temporary-reshape assignment replaces the destination with the matrix values
    EXPECT_EQ(temporary_destination, matrix);

    Matrix<double, 2, 2, StorageOrder> aliased({1.0, 2.0, 3.0, 4.0});
    aliased.template reshape<2, 2>() = aliased.transpose();
    // aliased reshape assignment snapshots the source before transposing the destination
    EXPECT_EQ(aliased, (Matrix<double, 2, 2, StorageOrder>({1.0, 3.0, 2.0, 4.0})));

    double destination_data[4] {};
    double source_data[4] {4.0, 3.0, 2.0, 1.0};
    auto view_destination = MatrixView<double, 2, 2, StorageOrder>(destination_data).template reshape<1, 4>();
    auto view_source = MatrixView<double, 2, 2, StorageOrder>(source_data).template reshape<1, 4>();
    view_destination = view_source;
    // assignment between view-backed reshapes copies every external source coefficient
    for (int i = 0; i < 4; ++i) { EXPECT_DOUBLE_EQ(destination_data[i], source_data[i]); }
    view_destination[0] = 17.0;
    // writing the destination reshape updates the destination's external storage
    EXPECT_DOUBLE_EQ(destination_data[0], 17.0);
    // the same write leaves the independent source storage unchanged
    EXPECT_DOUBLE_EQ(source_data[0], 4.0);
}

}   // namespace

// exercise dense owner shape rules, physical storage order, resizing and vector reorientation
TEST(NativeDenseMatrix, OwnerShapeStorageAndVectorCopy) {
    check_owner_behavior<RowMajor>();
    check_owner_behavior<ColMajor>();

    const auto ones = [](int, int) { return 1; };
    ProceduralMatrix<decltype(ones), 2, Dynamic> procedural(2, 3, ones);
    procedural.resize(2, 4);
    // a procedural matrix adopts its requested two rows
    EXPECT_EQ(procedural.rows(), 2);
    // a procedural matrix adopts its requested four columns
    EXPECT_EQ(procedural.cols(), 4);
    // the resized procedural expression evaluates the expected coefficient
    EXPECT_EQ(procedural(1, 3), 1);
}

// exercise numeric view binding, write-through assignment, const access and both storage orders
TEST(NativeDenseMatrix, NumericViewBindingConstnessAndStorage) {
    check_numeric_view_behavior<RowMajor>();
    check_numeric_view_behavior<ColMajor>();
}

// exercise stored arithmetic chains, transpose, structured parts and cross products
TEST(NativeDenseMatrix, ArithmeticExpressionNesting) {
    check_arithmetic_expression_nesting<RowMajor>();
    check_arithmetic_expression_nesting<ColMajor>();
}

// exercise snapshot-based assignment when owners and views alias their operands
TEST(NativeDenseMatrix, AssignmentOperationsMaterializeAliases) {
    check_assignment_alias_materialization<RowMajor>();
    check_assignment_alias_materialization<ColMajor>();
}

// exercise coefficientwise scalar types, custom operations, constness, lifetimes and aliased writes
TEST(NativeDenseMatrix, CoefficientWiseAdaptorsAreTypedLifetimeSafeAndAliasSafe) {
    check_coefficientwise_behavior<RowMajor>();
    check_coefficientwise_behavior<ColMajor>();
}

// exercise vectorwise broadcasting, stateful reductions, constness, lifetimes and aliased writes
TEST(NativeDenseMatrix, VectorWiseAdaptorsAreLifetimeSafeAndAliasSafe) {
    check_vectorwise_behavior<RowMajor>();
    check_vectorwise_behavior<ColMajor>();
}

// exercise numeric block bounds, retained views, iteration and write-through assignment
TEST(NativeDenseMatrix, BlocksRemainBoundedViews) {
    check_block_view_behavior<RowMajor>();
    check_block_view_behavior<ColMajor>();
}

// exercise reshape physical order, retained views, constness and aliased assignment
TEST(NativeDenseMatrix, ReshapePreservesPhysicalOrderAndSafeViews) {
    check_reshape_behavior<RowMajor>();
    check_reshape_behavior<ColMajor>();
}

}   // namespace fdapde
