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

using namespace fdapde;

namespace {

template <typename MatrixType>
concept permits_coefficient_write = requires(MatrixType& matrix) { matrix(0, 0) = 0.0; };

template <typename MatrixType>
concept permits_data_write = requires(MatrixType& matrix) { matrix.data()(0, 0) = 0.0; };

template <typename MatrixType>
concept permits_compound_assignment = requires(MatrixType& lhs, MatrixType& rhs) { lhs += rhs; };

template <typename MatrixType>
concept exposes_owning_rvalue_derived = requires(MatrixType& matrix) { std::move(matrix).derived(); };

template <typename MatrixType>
concept permits_owning_rvalue_inverse = requires(MatrixType& matrix) { std::move(matrix).inverse(); };

template <typename MatrixType>
concept permits_owning_rvalue_assignment = requires(MatrixType& lhs, MatrixType& rhs) { std::move(lhs) = rhs; };

using fixed_orthogonal = OrthogonalMatrix<double, 2, 2>;
using fixed_orthogonal_view = OrthogonalMatrixView<double, 2, 2>;
using const_orthogonal_view = OrthogonalMatrixView<const double, 2, 2>;

// checks at compile time: !std::is_default_constructible_v<fixed_orthogonal>
static_assert(!std::is_default_constructible_v<fixed_orthogonal>);
// checks at compile time: !std::is_default_constructible_v<fixed_orthogonal_view>
static_assert(!std::is_default_constructible_v<fixed_orthogonal_view>);
// checks at compile time: std::is_default_constructible_v<OrthogonalMatrixView<double, Dynamic, Dynamic>>
static_assert(std::is_default_constructible_v<OrthogonalMatrixView<double, Dynamic, Dynamic>>);
// checks at compile time: fixed_orthogonal::ReadOnly == 1
static_assert(fixed_orthogonal::ReadOnly == 1);
// checks at compile time: fixed_orthogonal_view::ReadOnly == 1
static_assert(fixed_orthogonal_view::ReadOnly == 1);
// checks at compile time: const_orthogonal_view::ReadOnly == 1
static_assert(const_orthogonal_view::ReadOnly == 1);
// checks at compile time: !permits_coefficient_write<fixed_orthogonal>
static_assert(!permits_coefficient_write<fixed_orthogonal>);
// checks at compile time: !permits_coefficient_write<fixed_orthogonal_view>
static_assert(!permits_coefficient_write<fixed_orthogonal_view>);
// checks at compile time: !permits_coefficient_write<const_orthogonal_view>
static_assert(!permits_coefficient_write<const_orthogonal_view>);
// checks at compile time: !permits_compound_assignment<fixed_orthogonal>
static_assert(!permits_compound_assignment<fixed_orthogonal>);
// checks at compile time: !permits_compound_assignment<fixed_orthogonal_view>
static_assert(!permits_compound_assignment<fixed_orthogonal_view>);
// checks at compile time: !permits_data_write<fixed_orthogonal>
static_assert(!permits_data_write<fixed_orthogonal>);
// checks at compile time: !permits_data_write<fixed_orthogonal_view>
static_assert(!permits_data_write<fixed_orthogonal_view>);
// checks at compile time: !permits_data_write<const_orthogonal_view>
static_assert(!permits_data_write<const_orthogonal_view>);
// checks at compile time: !exposes_owning_rvalue_derived<fixed_orthogonal>
static_assert(!exposes_owning_rvalue_derived<fixed_orthogonal>);
// checks at compile time: !permits_owning_rvalue_inverse<fixed_orthogonal>
static_assert(!permits_owning_rvalue_inverse<fixed_orthogonal>);
// checks at compile time: !permits_owning_rvalue_assignment<fixed_orthogonal>
static_assert(!permits_owning_rvalue_assignment<fixed_orthogonal>);
// checks at compile time: is_orthogonal_matrix_v<const fixed_orthogonal&>
static_assert(is_orthogonal_matrix_v<const fixed_orthogonal&>);

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

template <int StorageOrder> void check_orthogonal_contracts() {
    using owner_type = OrthogonalMatrix<double, 2, 2, StorageOrder>;
    using dense_type = Matrix<double, 2, 2, StorageOrder>;
    using view_type = OrthogonalMatrixView<double, 2, 2, StorageOrder>;

    const double rotation_data[] = {0.0, -1.0, 1.0, 0.0};
    const owner_type rotation(rotation_data, checked);
    // checks at compile time: owner_type::StorageOrder == StorageOrder
    static_assert(owner_type::StorageOrder == StorageOrder);
    expect_matrix_near(rotation, dense_type({0.0, -1.0, 1.0, 0.0}));
    // compares the computed and expected values within the stated absolute tolerance
    EXPECT_NEAR(rotation.norm(), std::sqrt(2.0), 1.0e-12);

    const auto squared = rotation * rotation;
    // checks at compile time: is_orthogonal_matrix_v<decltype(squared)>
    static_assert(is_orthogonal_matrix_v<decltype(squared)>);
    expect_matrix_near(squared, dense_type({-1.0, 0.0, 0.0, -1.0}));
    expect_matrix_near(rotation.inverse(), dense_type({0.0, 1.0, -1.0, 0.0}));

    const Vector<double, 2> rhs({1.0, 0.0});
    expect_matrix_near(rotation.solve(rhs), Vector<double, 2>({0.0, -1.0}));
    const dense_type matrix_rhs({1.0, 2.0, 3.0, 4.0});
    expect_matrix_near(rotation.solve(matrix_rhs), dense_type({3.0, 4.0, -1.0, -2.0}));

    const OrthogonalMatrix<double, Dynamic, Dynamic, StorageOrder> dynamic(
      std::vector<double> {0.0, -1.0, 1.0, 0.0}, checked);
    const OrthogonalMatrix<double, 2, Dynamic, StorageOrder> partial(
      std::vector<double> {0.0, -1.0, 1.0, 0.0}, checked);
    // compares dynamic.rows(), 2 using eq semantics
    EXPECT_EQ(dynamic.rows(), 2);
    // compares partial.cols(), 2 using eq semantics
    EXPECT_EQ(partial.cols(), 2);
    expect_matrix_near(dynamic, rotation);
    expect_matrix_near(partial, rotation);

    const double basis_data[] = {1.0, 1.0, 0.0, 1.0};
    const owner_type orthogonalized(basis_data, orthogonalize);
    expect_matrix_near(orthogonalized.data().transpose() * orthogonalized.data(), IdentityMatrix<double, 2, 2>());
    const double close_basis_data[] = {1.0, 1.0, 1.0, 1.0 + 1.0e-5};
    const owner_type reorthogonalized(close_basis_data, orthogonalize);
    expect_matrix_near(
      reorthogonalized.data().transpose() * reorthogonalized.data(), IdentityMatrix<double, 2, 2>(), 1.0e-9);

    const double unchecked_data[] = {1.0, 2.0, 3.0, 4.0};
    const owner_type unchecked_owner(unchecked_data, unchecked);
    // compares unchecked_owner(1, 0), 3.0 using double_eq semantics
    EXPECT_DOUBLE_EQ(unchecked_owner(1, 0), 3.0);

    dense_type view_storage({0.0, -1.0, 1.0, 0.0});
    dense_type identity_storage({1.0, 0.0, 0.0, 1.0});
    view_type view(view_storage.data(), checked);
    view_type identity_view(identity_storage.data(), checked);
    const double* const view_address = view.data().data();
    view = identity_view;
    // compares view.data().data(), view_address using eq semantics
    EXPECT_EQ(view.data().data(), view_address);
    expect_matrix_near(view, identity_storage);
    view = rotation;
    // compares view.data().data(), view_address using eq semantics
    EXPECT_EQ(view.data().data(), view_address);
    expect_matrix_near(view, rotation);
    view = rotation * rotation;
    // compares view.data().data(), view_address using eq semantics
    EXPECT_EQ(view.data().data(), view_address);
    expect_matrix_near(view, dense_type({-1.0, 0.0, 0.0, -1.0}));
    const OrthogonalMatrixView<const double, 2, 2, StorageOrder> const_identity_view(identity_storage.data(), checked);
    view = const_identity_view;
    // compares view.data().data(), view_address using eq semantics
    EXPECT_EQ(view.data().data(), view_address);
    expect_matrix_near(view, identity_storage);

    std::array<double, 6> overlap_storage {1.0, 0.0, 0.0, 1.0, -1.0, 0.0};
    view_type overlap_source(overlap_storage.data(), checked);
    view_type overlap_target(overlap_storage.data() + 2, checked);
    const double* const overlap_address = overlap_target.data().data();
    overlap_target = overlap_source;
    // compares overlap_target.data().data(), overlap_address using eq semantics
    EXPECT_EQ(overlap_target.data().data(), overlap_address);
    expect_matrix_near(overlap_target, dense_type({1.0, 0.0, 0.0, 1.0}));

    const OrthogonalMatrixView<const double, 2, 2, StorageOrder> const_view(identity_storage.data(), checked);
    // compares const_view(1, 1), 1.0 using double_eq semantics
    EXPECT_DOUBLE_EQ(const_view(1, 1), 1.0);
    OrthogonalMatrixView<double, Dynamic, Dynamic, StorageOrder> dynamic_view(identity_storage.data(), 2, 2, checked);
    expect_matrix_near(dynamic_view, identity_storage);

    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(owner_type(std::vector<double> {1.0, 0.0, 0.0}, checked)), std::invalid_argument);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(owner_type(std::vector<double> {1.0, 0.0, 0.0}, unchecked)), std::invalid_argument);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(owner_type(unchecked_data, checked)), std::invalid_argument);
    const double rank_deficient_data[] = {1.0, 2.0, 2.0, 4.0};
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(owner_type(rank_deficient_data, orthogonalize)), std::invalid_argument);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(view_type(static_cast<double*>(nullptr), checked)), std::invalid_argument);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(view_type(static_cast<double*>(nullptr), unchecked)), std::invalid_argument);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(
      static_cast<void>(
        OrthogonalMatrixView<double, Dynamic, Dynamic, StorageOrder>(identity_storage.data(), 1, 2, unchecked)),
      std::invalid_argument);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(
      static_cast<void>(
        OrthogonalMatrixView<double, Dynamic, Dynamic, StorageOrder>(identity_storage.data(), 1, 2, checked)),
      std::invalid_argument);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(rotation(-1, 0)), std::out_of_range);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(view(2, 0)), std::out_of_range);

    std::array<double, 9> larger_identity {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    OrthogonalMatrixView<double, Dynamic, Dynamic, StorageOrder> mismatched_target(
      identity_storage.data(), 2, 2, checked);
    OrthogonalMatrixView<double, Dynamic, Dynamic, StorageOrder> mismatched_source(
      larger_identity.data(), 3, 3, checked);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(mismatched_target = mismatched_source, std::invalid_argument);
    // compares mismatched_target.rows(), 2 using eq semantics
    EXPECT_EQ(mismatched_target.rows(), 2);
}

}   // namespace

// verifies orthogonal through the public algebra API
TEST(linear_algebra, orthogonal) {
    check_orthogonal_contracts<RowMajor>();
    check_orthogonal_contracts<ColMajor>();
}
