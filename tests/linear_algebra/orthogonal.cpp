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
concept permits_coefficient_write = requires(MatrixType& matrix) { matrix(0, 0) = 0.0; };

template <typename MatrixType>
concept permits_data_write = requires(MatrixType& matrix) { matrix.data()(0, 0) = 0.0; };

template <typename MatrixType>
concept permits_compound_assignment = requires(MatrixType& lhs, MatrixType& rhs) { lhs += rhs; };

template <typename MatrixType>
concept exposes_owning_rvalue_derived = requires(MatrixType& matrix) { std::move(matrix).derived(); };

template <typename MatrixType>
concept permits_owning_rvalue_inverse = requires(MatrixType& matrix) { std::move(matrix).inv(); };

template <typename MatrixType>
concept permits_owning_rvalue_assignment = requires(MatrixType& lhs, MatrixType& rhs) { std::move(lhs) = rhs; };

using fixed_orthogonal = OrthogonalMatrix<double, 2>;
using fixed_orthogonal_view = OrthogonalMatrixView<double, 2>;
using const_orthogonal_view = OrthogonalMatrixView<const double, 2>;

// arbitrary matrices cannot bypass validation through the removed checked tag
static_assert(!std::is_constructible_v<fixed_orthogonal, Matrix<double, 2, 2>, decltype(checked)>);
// arbitrary matrices cannot bypass validation through the removed unchecked tag
static_assert(!std::is_constructible_v<fixed_orthogonal, Matrix<double, 2, 2>, decltype(unchecked)>);
// raw storage cannot request the removed checked view overload
static_assert(!std::is_constructible_v<fixed_orthogonal_view, double*, decltype(checked)>);
// raw storage cannot request an unchecked view binding
static_assert(!std::is_constructible_v<fixed_orthogonal_view, double*, decltype(unchecked)>);

// a fixed orthogonal owner requires explicit coefficients and a construction policy
static_assert(!std::is_default_constructible_v<fixed_orthogonal>);
// a fixed nonempty orthogonal view requires an explicit storage binding
static_assert(!std::is_default_constructible_v<fixed_orthogonal_view>);
// a dynamic orthogonal view also requires an explicit verified binding
static_assert(!std::is_default_constructible_v<OrthogonalMatrixView<double, Dynamic>>);
// an orthogonal owner advertises immutable coefficient access
static_assert(fixed_orthogonal::ReadOnly == 1);
// a mutable-storage orthogonal view still advertises immutable coefficient access
static_assert(fixed_orthogonal_view::ReadOnly == 1);
// a const-storage orthogonal view advertises immutable coefficient access
static_assert(const_orthogonal_view::ReadOnly == 1);
// an owner forbids coefficient writes that could break orthogonality
static_assert(!permits_coefficient_write<fixed_orthogonal>);
// a mutable-storage view forbids coefficient writes that could break orthogonality
static_assert(!permits_coefficient_write<fixed_orthogonal_view>);
// a const-storage view forbids coefficient writes
static_assert(!permits_coefficient_write<const_orthogonal_view>);
// an orthogonal owner does not expose unrestricted compound assignment
static_assert(!permits_compound_assignment<fixed_orthogonal>);
// an orthogonal view does not expose unrestricted compound assignment
static_assert(!permits_compound_assignment<fixed_orthogonal_view>);
// an owner cannot bypass immutability through its data accessor
static_assert(!permits_data_write<fixed_orthogonal>);
// a mutable-storage view cannot bypass immutability through its data accessor
static_assert(!permits_data_write<fixed_orthogonal_view>);
// a const-storage view cannot expose writable raw coefficients
static_assert(!permits_data_write<const_orthogonal_view>);
// a temporary owner cannot expose a derived reference that would dangle
static_assert(!exposes_owning_rvalue_derived<fixed_orthogonal>);
// a temporary owner cannot lend the transpose used by inverse
static_assert(!permits_owning_rvalue_inverse<fixed_orthogonal>);
// a temporary owner cannot return a borrow through assignment
static_assert(!permits_owning_rvalue_assignment<fixed_orthogonal>);
// the orthogonal type trait recognizes a const reference to an owner
static_assert(is_orthogonal_matrix_v<const fixed_orthogonal&>);

template <typename Actual, typename Expected>
void expect_matrix_near(const Actual& actual, const Expected& expected, double tolerance = 1.0e-12) {
    // coefficient comparison requires the same number of rows
    ASSERT_EQ(actual.rows(), expected.rows());
    // coefficient comparison requires the same number of columns
    ASSERT_EQ(actual.cols(), expected.cols());
    for (int i = 0; i < actual.rows(); ++i) {
        for (int j = 0; j < actual.cols(); ++j) {
            // each logical coefficient agrees with the reference within the supplied absolute tolerance
            EXPECT_NEAR(static_cast<double>(actual(i, j)), static_cast<double>(expected(i, j)), tolerance);
        }
    }
}

template <int StorageOrder> void check_orthogonal_contracts() {
    using owner_type = OrthogonalMatrix<double, 2, StorageOrder>;
    using dense_type = Matrix<double, 2, 2, StorageOrder>;
    using view_type = OrthogonalMatrixView<double, 2, StorageOrder>;

    const double rotation_data[] = {0.0, -1.0, 1.0, 0.0};
    const owner_type rotation(rotation_data);
    // the orthogonal owner retains its requested storage order
    static_assert(owner_type::StorageOrder == StorageOrder);
    expect_matrix_near(rotation, dense_type({0.0, -1.0, 1.0, 0.0}));
    // a two-dimensional rotation has Frobenius norm sqrt(2)
    EXPECT_NEAR(rotation.norm(), std::sqrt(2.0), 1.0e-12);

    const auto squared = rotation * rotation;
    // the product of orthogonal matrices retains the orthogonal expression tag
    static_assert(is_orthogonal_matrix_v<decltype(squared)>);
    expect_matrix_near(squared, dense_type({-1.0, 0.0, 0.0, -1.0}));
    // inversion transposes the known quarter-turn coefficients without a factorization
    expect_matrix_near(rotation.inv(), dense_type({0.0, 1.0, -1.0, 0.0}));

    const Vector<double, 2> rhs({1.0, 0.0});
    expect_matrix_near(rotation.solve(rhs), Vector<double, 2>({0.0, -1.0}));
    const dense_type matrix_rhs({1.0, 2.0, 3.0, 4.0});
    expect_matrix_near(rotation.solve(matrix_rhs), dense_type({3.0, 4.0, -1.0, -2.0}));

    const OrthogonalMatrix<double, Dynamic, StorageOrder> dynamic(std::vector<double> {0.0, -1.0, 1.0, 0.0});
    // a dynamic orthogonal owner adopts the two-row input shape
    EXPECT_EQ(dynamic.rows(), 2);
    // one static order determines both orthogonal axes independently of storage order
    static_assert(owner_type::Order == 2 && owner_type::Rows == owner_type::Cols);
    expect_matrix_near(dynamic, rotation);

    const double basis_data[] = {1.0, 1.0, 0.0, 1.0};
    const owner_type orthogonalized(basis_data, orthogonalize);
    expect_matrix_near(orthogonalized.data().transpose() * orthogonalized.data(), IdentityMatrix<double, 2, 2>());
    const double close_basis_data[] = {1.0, 1.0, 1.0, 1.0 + 1.0e-5};
    const owner_type reorthogonalized(close_basis_data, orthogonalize);
    expect_matrix_near(
      reorthogonalized.data().transpose() * reorthogonalized.data(), IdentityMatrix<double, 2, 2>(), 1.0e-9);

    const double nonorthogonal_data[] = {1.0, 2.0, 3.0, 4.0};

    dense_type view_storage({0.0, -1.0, 1.0, 0.0});
    dense_type identity_storage({1.0, 0.0, 0.0, 1.0});
    view_type view(view_storage.data());
    view_type identity_view(identity_storage.data());
    const double* const view_address = view.data().data();
    view = identity_view;
    // view-to-view assignment keeps the destination storage binding
    EXPECT_EQ(view.data().data(), view_address);
    expect_matrix_near(view, identity_storage);
    view = rotation;
    // assignment from an owner keeps the destination storage binding
    EXPECT_EQ(view.data().data(), view_address);
    expect_matrix_near(view, rotation);
    view = rotation * rotation;
    // assignment from an orthogonal product keeps the destination storage binding
    EXPECT_EQ(view.data().data(), view_address);
    expect_matrix_near(view, dense_type({-1.0, 0.0, 0.0, -1.0}));
    const OrthogonalMatrixView<const double, 2, StorageOrder> const_identity_view(identity_storage.data());
    view = const_identity_view;
    // assignment from a const-storage view keeps the destination storage binding
    EXPECT_EQ(view.data().data(), view_address);
    expect_matrix_near(view, identity_storage);

    std::array<double, 6> overlap_storage {1.0, 0.0, 0.0, 1.0, -1.0, 0.0};
    view_type overlap_source(overlap_storage.data());
    view_type overlap_target(overlap_storage.data() + 2);
    const double* const overlap_address = overlap_target.data().data();
    overlap_target = overlap_source;
    // overlapping assignment keeps the destination bound to its original offset
    EXPECT_EQ(overlap_target.data().data(), overlap_address);
    expect_matrix_near(overlap_target, dense_type({1.0, 0.0, 0.0, 1.0}));

    const OrthogonalMatrixView<const double, 2, StorageOrder> const_view(identity_storage.data());
    // a const orthogonal view reads the identity matrix's final diagonal entry
    EXPECT_DOUBLE_EQ(const_view(1, 1), 1.0);
    OrthogonalMatrixView<double, Dynamic, StorageOrder> dynamic_view(identity_storage.data(), 2, 2);
    expect_matrix_near(dynamic_view, identity_storage);

    // construction rejects three coefficients for a 2-by-2 owner
    EXPECT_THROW(static_cast<void>(owner_type(std::vector<double> {1.0, 0.0, 0.0})), std::invalid_argument);
    // construction rejects coefficients that are not orthogonal
    EXPECT_THROW(static_cast<void>(owner_type(nonorthogonal_data)), std::invalid_argument);
    const double rank_deficient_data[] = {1.0, 2.0, 2.0, 4.0};
    // orthogonalization rejects a rank-deficient basis
    EXPECT_THROW(static_cast<void>(owner_type(rank_deficient_data, orthogonalize)), std::invalid_argument);
    // construction rejects a null pointer for a nonempty view
    EXPECT_THROW(static_cast<void>(view_type(static_cast<double*>(nullptr))), std::invalid_argument);
    // construction rejects a nonsquare view shape
    EXPECT_THROW(
      static_cast<void>(OrthogonalMatrixView<double, Dynamic, StorageOrder>(identity_storage.data(), 1, 2)),
      std::invalid_argument);
    // owner coefficient access rejects a negative row
    EXPECT_THROW(static_cast<void>(rotation(-1, 0)), std::out_of_range);
    // view coefficient access rejects a row equal to its extent
    EXPECT_THROW(static_cast<void>(view(2, 0)), std::out_of_range);

    std::array<double, 9> larger_identity {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    OrthogonalMatrixView<double, Dynamic, StorageOrder> mismatched_target(identity_storage.data(), 2, 2);
    OrthogonalMatrixView<double, Dynamic, StorageOrder> mismatched_source(larger_identity.data(), 3, 3);
    // view assignment rejects a source with a different shape
    EXPECT_THROW(mismatched_target = mismatched_source, std::invalid_argument);
    // failed view assignment preserves the destination's row extent
    EXPECT_EQ(mismatched_target.rows(), 2);
}

template <int StorageOrder> void check_matrix_orthogonalization() {
    using dense_type = Matrix<double, 2, 2, StorageOrder>;
    dense_type directions({1., 1., 1., 0.});
    const dense_type original(directions);
    const double h = 1. / std::sqrt(2.);
    const dense_type expected({h, h, h, -h});
    const auto basis = std::as_const(directions).orthogonalize();
    // the operation returns an owner preserving the scalar, static dimensions and storage order
    static_assert(std::is_same_v<std::remove_cv_t<decltype(basis)>, OrthogonalMatrix<double, 2, StorageOrder>>);
    // these columns have analytically known orthonormal directions
    expect_matrix_near(basis, expected);
    // constructing the basis leaves all source coefficients untouched
    expect_matrix_near(directions, original);
    // the Gram matrix verifies both unit column norms and zero cross-column inner products
    expect_matrix_near(basis.transpose() * basis, IdentityMatrix<double, 2, 2>());

    const Matrix<double, Dynamic, Dynamic, StorageOrder> dynamic(original);
    const auto dynamic_basis = dynamic.orthogonalize();
    // runtime dimensions and the requested layout remain encoded in the returned owner
    static_assert(
      std::is_same_v<std::remove_cv_t<decltype(dynamic_basis)>, OrthogonalMatrix<double, Dynamic, StorageOrder>>);
    // runtime shape and coefficients agree with the same analytic basis
    expect_matrix_near(dynamic_basis, expected);

    const Matrix<double, Dynamic, 2, StorageOrder> fixed_columns(original);
    const auto fixed_order_basis = fixed_columns.orthogonalize();
    // a known column count determines the result's single order even when input rows are dynamic
    static_assert(
      std::is_same_v<std::remove_cv_t<decltype(fixed_order_basis)>, OrthogonalMatrix<double, 2, StorageOrder>>);
    // inferred fixed order produces the same analytic orthonormal basis
    expect_matrix_near(fixed_order_basis, expected);

    const MatrixView<const double, 2, 2, StorageOrder> view(directions.data());
    const auto view_basis = view.orthogonalize();
    // a read-only external view can produce an independent orthogonal owner
    expect_matrix_near(view_basis, expected);
    const auto expression_basis = (directions + original).orthogonalize();
    // doubling both columns preserves their analytically known orthonormal directions
    expect_matrix_near(expression_basis, expected);
    const auto temporary_basis = dense_type({1., 1., 1., 0.}).orthogonalize();
    // eager ownership keeps the result valid after the temporary input is destroyed
    expect_matrix_near(temporary_basis, expected);

    directions(0, 0) = 7.;
    // changing the original buffer cannot alter the basis obtained from its view
    expect_matrix_near(view_basis, expected);
}

}   // namespace

// exercise immutable orthogonal owners, checked construction and overlap-safe views in both storage orders
TEST(linear_algebra, orthogonal) {
    check_orthogonal_contracts<RowMajor>();
    check_orthogonal_contracts<ColMajor>();
}

// matrix orthogonalization owns its result and preserves inputs across layouts, views and expressions
TEST(linear_algebra, orthogonalize_matrix) {
    check_matrix_orthogonalization<RowMajor>();
    check_matrix_orthogonalization<ColMajor>();
}

// permanent validation rejects inputs that cannot establish a finite square orthonormal basis
TEST(linear_algebra, orthogonalize_invalid_matrix) {
    const Matrix<double, 2, 2> dependent({1., 2., 2., 4.});
    // collinear columns cannot produce a full-rank orthogonal matrix
    EXPECT_THROW(static_cast<void>(dependent.orthogonalize()), std::invalid_argument);
    const Matrix<double, Dynamic, Dynamic> rectangular(2, 3);
    // runtime rectangular dimensions cannot represent an orthogonal owner
    EXPECT_THROW(static_cast<void>(rectangular.orthogonalize()), std::invalid_argument);
    const Matrix<double, 2, 2> nonfinite({std::numeric_limits<double>::quiet_NaN(), 1., 1., 0.});
    // a nonfinite column cannot satisfy the orthonormality check
    EXPECT_THROW(static_cast<void>(nonfinite.orthogonalize()), std::invalid_argument);
}
