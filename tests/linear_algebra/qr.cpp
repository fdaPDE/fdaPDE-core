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

#include <limits>
#include <type_traits>
#include <utility>

namespace {

using namespace fdapde;

template <typename Decomposition>
concept permits_rvalue_q_factor = requires(Decomposition decomposition) { std::move(decomposition).Q(); };

template <typename Decomposition>
concept permits_rvalue_r_factor = requires(Decomposition decomposition) { std::move(decomposition).R(); };

using fixed_qr = HouseholderQR<double, 3, 2>;
static_assert(std::is_default_constructible_v<fixed_qr>);
static_assert(!permits_rvalue_q_factor<fixed_qr>);
static_assert(!permits_rvalue_r_factor<fixed_qr>);

template <typename MatrixType, typename Decomposition>
void expect_qr_reconstruction(
  const MatrixType& source, const Decomposition& decomposition, double tolerance = 1.0e-12) {
    ASSERT_TRUE(decomposition.computed());
    ASSERT_EQ(decomposition.Q().rows(), source.rows());
    ASSERT_EQ(decomposition.Q().cols(), source.rows());
    ASSERT_EQ(decomposition.R().rows(), source.rows());
    ASSERT_EQ(decomposition.R().cols(), source.cols());

    const Matrix<double, Dynamic, Dynamic> dense_source(source);
    const Matrix<double, Dynamic, Dynamic> reconstructed(decomposition.Q() * decomposition.R());
    const double scale = dense_source.norm();
    const double denominator = scale > 0.0 ? scale : 1.0;
    EXPECT_LT((reconstructed - dense_source).norm() / denominator, tolerance);

    const Matrix<double, Dynamic, Dynamic> q(decomposition.Q());
    Matrix<double, Dynamic, Dynamic> identity(q.rows(), q.rows());
    identity.set_zero();
    for (int i = 0; i < identity.rows(); ++i) identity(i, i) = 1.0;
    EXPECT_LT((q.transpose() * q - identity).norm(), tolerance);
}

template <int StorageOrder> void check_qr_shapes_lifetime_and_reconstruction() {
    const Matrix<double, 4, 3, StorageOrder> tall(
      {1.0, 2.0, -1.0, 2.0, 0.0, 3.0, -1.0, 4.0, 2.0, 3.0, -2.0, 1.0});
    const HouseholderQR tall_qr(tall);
    static_assert(decltype(tall_qr)::Rows == 4 && decltype(tall_qr)::Cols == 3);
    EXPECT_EQ(tall_qr.rank(), 3);
    expect_qr_reconstruction(tall, tall_qr);

    const Matrix<double, 2, 3, StorageOrder> wide_source({1.0, -2.0, 3.0, 4.0, 1.0, -1.0});
    const Matrix<double, Dynamic, Dynamic, StorageOrder> wide(wide_source);
    const HouseholderQR wide_qr(wide);
    static_assert(decltype(wide_qr)::Rows == Dynamic && decltype(wide_qr)::Cols == Dynamic);
    EXPECT_EQ(wide_qr.rank(), 2);
    expect_qr_reconstruction(wide, wide_qr);

    const Matrix<double, 3, 2, StorageOrder> partial_source({1.0, 2.0, 3.0, -1.0, 2.0, 4.0});
    const Matrix<double, Dynamic, 2, StorageOrder> partial(partial_source);
    const HouseholderQR partial_qr(partial);
    static_assert(decltype(partial_qr)::Rows == Dynamic && decltype(partial_qr)::Cols == 2);
    EXPECT_EQ(partial_qr.rank(), 2);
    expect_qr_reconstruction(partial, partial_qr);

    const MatrixView<const double, 3, 2, StorageOrder> const_view(partial_source.data());
    const HouseholderQR const_view_qr(const_view);
    static_assert(std::is_same_v<typename decltype(const_view_qr)::Scalar, double>);
    expect_qr_reconstruction(const_view, const_view_qr);

    const auto retained_qr = [] {
        const Matrix<double, 3, 2, StorageOrder> source({1.0, 2.0, 3.0, -1.0, 2.0, 4.0});
        const Matrix<double, 3, 2, StorageOrder> zero = Matrix<double, 3, 2, StorageOrder>::Zero();
        return HouseholderQR(source + zero);
    }();
    expect_qr_reconstruction(partial_source, retained_qr);
}

template <int StorageOrder> void check_qr_rank_and_scale_contracts() {
    const Matrix<double, 3, 2, StorageOrder> zero = Matrix<double, 3, 2, StorageOrder>::Zero();
    const HouseholderQR zero_qr(zero);
    EXPECT_EQ(zero_qr.rank(), 0);
    expect_qr_reconstruction(zero, zero_qr);

    const Matrix<double, 4, 3, StorageOrder> rank_deficient(
      {1.0, 2.0, 3.0, 2.0, 4.0, 1.0, 3.0, 6.0, -1.0, 4.0, 8.0, 2.0});
    const HouseholderQR rank_deficient_qr(rank_deficient);
    EXPECT_EQ(rank_deficient_qr.rank(), 2);
    expect_qr_reconstruction(rank_deficient, rank_deficient_qr);

    const Matrix<double, 3, 3, StorageOrder> leading_zero_column(
      {0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0});
    const HouseholderQR shifted_qr(leading_zero_column);
    EXPECT_EQ(shifted_qr.rank(), 2);
    expect_qr_reconstruction(leading_zero_column, shifted_qr);

    for (const double scale : {1.0e-150, 1.0e150}) {
        const Matrix<double, 3, 2, StorageOrder> matrix(
          {scale, 2.0 * scale, -3.0 * scale, scale, 2.0 * scale, -scale});
        const HouseholderQR qr(matrix);
        EXPECT_EQ(qr.rank(), 2);
        expect_qr_reconstruction(matrix, qr);
    }

    const Matrix<double, 2, 2, StorageOrder> mixed_scale({1.0, 0.0, 0.0, 1.0e-160});
    const HouseholderQR mixed_scale_qr(mixed_scale);
    EXPECT_EQ(mixed_scale_qr.rank(), 1);
    expect_qr_reconstruction(mixed_scale, mixed_scale_qr);
}

template <int StorageOrder> void check_qr_invalid_input_contracts() {
    HouseholderQR<double, Dynamic, Dynamic> reusable;
    EXPECT_FALSE(reusable.computed());
    EXPECT_EQ(reusable.rank(), 0);

    const Matrix<double, 2, 2, StorageOrder> valid({1.0, 2.0, 3.0, 4.0});
    reusable.compute(valid);
    EXPECT_TRUE(reusable.computed());
    EXPECT_EQ(reusable.rank(), 2);

    Matrix<double, Dynamic, Dynamic, StorageOrder> nonfinite(valid);
    nonfinite(0, 0) = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(reusable.compute(nonfinite), std::invalid_argument);
    EXPECT_FALSE(reusable.computed());
    EXPECT_EQ(reusable.rank(), 0);

    nonfinite(0, 0) = std::numeric_limits<double>::infinity();
    EXPECT_THROW(reusable.compute(nonfinite), std::invalid_argument);
    EXPECT_FALSE(reusable.computed());
    EXPECT_EQ(reusable.rank(), 0);

    const Matrix<double, Dynamic, Dynamic, StorageOrder> empty;
    EXPECT_THROW(reusable.compute(empty), std::invalid_argument);
    EXPECT_FALSE(reusable.computed());
    EXPECT_EQ(reusable.rank(), 0);

    HouseholderQR<double, 3, 2> fixed_shape;
    Matrix<double, Dynamic, Dynamic, StorageOrder> wrong_shape(2, 3);
    EXPECT_THROW(fixed_shape.compute(wrong_shape), std::invalid_argument);
    EXPECT_FALSE(fixed_shape.computed());
    EXPECT_EQ(fixed_shape.rank(), 0);
}

TEST(linear_algebra, householder_qr) {
    check_qr_shapes_lifetime_and_reconstruction<RowMajor>();
    check_qr_shapes_lifetime_and_reconstruction<ColMajor>();
    check_qr_rank_and_scale_contracts<RowMajor>();
    check_qr_rank_and_scale_contracts<ColMajor>();
    check_qr_invalid_input_contracts<RowMajor>();
    check_qr_invalid_input_contracts<ColMajor>();
}

}   // namespace
