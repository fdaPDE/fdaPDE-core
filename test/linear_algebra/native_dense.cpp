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
#include <iterator>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

namespace native = fdapde::linalg;

template <typename Matrix>
concept permits_temporary_add = requires(Matrix& named) { Matrix {} + named; };

template <typename Matrix>
concept permits_temporary_cwise = requires { Matrix {}.cwise(); };

template <typename Matrix>
concept permits_safe_expression_chaining = requires(Matrix& a, Matrix& b, Matrix& c) { (a + b) + c; };

template <typename Matrix>
concept permits_temporary_bool_and = requires(Matrix& named) { Matrix {} & named; };

template <typename Mask, typename Matrix>
concept permits_temporary_select_branch = requires(Mask& mask, Matrix& named) { mask.select(Matrix {}, named); };

template <typename Matrix>
concept exposes_temporary_derived = requires { Matrix {}.derived(); };

template <typename Matrix>
concept permits_temporary_assignment_add = requires(Matrix& named) { (Matrix {} = named) + named; };

template <typename Mask>
concept permits_temporary_bool_compound = requires(Mask& named) { (Mask {} |= named) & named; };

template <typename Matrix>
concept permits_direct_temporary_rowwise = requires { native::MatrixRowWiseOp<Matrix>(Matrix {}); };

template <typename Matrix>
concept exposes_temporary_cwise_derived = requires(Matrix& named) { named.cwise().derived(); };

template <typename Matrix>
concept exposes_temporary_transformed_xpr = requires(Matrix& named) { named.cwise().sqrt().xpr(); };

template <typename Matrix>
concept permits_const_cwise_mutation = requires(const Matrix& matrix) { matrix.cwise() += 1.0; };

template <typename Matrix>
concept permits_const_reshape_assignment = requires(const Matrix& dst, Matrix& src) {
    dst.template reshape<1, 4>() = src.template reshape<1, 4>();
};

template <typename Matrix>
concept permits_const_rowwise_assignment = requires(const Matrix& dst, Matrix& src) {
    dst.rowwise() = src.rowwise().sum();
};

template <typename View>
concept permits_const_numeric_coefficient_write = requires(const View& view) { view(0, 0) = 1.0; };

template <typename View>
concept permits_const_boolean_coefficient_write = requires(const View& view) { view(0, 0) = true; };

template <typename View>
concept exposes_temporary_block_iterator = requires(View&& view) { std::move(view).begin(); };

using lifetime_matrix = native::Matrix<double, 2, 2>;
using lifetime_mask = native::Matrix<bool, 2, 2>;
using lifetime_block = decltype(std::declval<lifetime_matrix&>().template block<1, 1>(0, 0));
using lifetime_reshape = decltype(std::declval<lifetime_matrix&>().template reshape<1, 4>());
using lifetime_bool_block = decltype(std::declval<lifetime_mask&>().template block<1, 1>(0, 0));
using lifetime_bool_reshape = decltype(std::declval<lifetime_mask&>().template reshape<1, 4>());
static_assert(!permits_temporary_add<lifetime_matrix>);
static_assert(!permits_temporary_cwise<lifetime_matrix>);
static_assert(permits_safe_expression_chaining<lifetime_matrix>);
static_assert(!permits_temporary_bool_and<lifetime_mask>);
static_assert(!permits_temporary_select_branch<lifetime_mask, lifetime_matrix>);
static_assert(!exposes_temporary_derived<lifetime_matrix>);
static_assert(!exposes_temporary_derived<lifetime_mask>);
static_assert(!permits_temporary_assignment_add<lifetime_matrix>);
static_assert(!permits_temporary_bool_compound<lifetime_mask>);
static_assert(!permits_direct_temporary_rowwise<lifetime_matrix>);
static_assert(!exposes_temporary_cwise_derived<lifetime_matrix>);
static_assert(!exposes_temporary_transformed_xpr<lifetime_matrix>);
static_assert(!std::is_lvalue_reference_v<decltype(std::declval<lifetime_matrix&>().cwise() += 1.0)>);
static_assert(!std::is_lvalue_reference_v<decltype(std::declval<lifetime_matrix&>().cwise() = 1.0)>);
static_assert(!permits_const_cwise_mutation<lifetime_matrix>);
static_assert(!permits_const_reshape_assignment<lifetime_matrix>);
static_assert(!permits_const_rowwise_assignment<lifetime_matrix>);
static_assert(!permits_const_numeric_coefficient_write<lifetime_block>);
static_assert(!permits_const_numeric_coefficient_write<lifetime_reshape>);
static_assert(!permits_const_boolean_coefficient_write<lifetime_bool_block>);
static_assert(!permits_const_boolean_coefficient_write<lifetime_bool_reshape>);
static_assert(!exposes_temporary_block_iterator<lifetime_block>);
static_assert(decltype(std::declval<const lifetime_matrix&>().cwise())::ReadOnly == 1);
static_assert(decltype(std::declval<const lifetime_matrix&>().template reshape<1, 4>())::ReadOnly == 1);
static_assert(decltype(std::declval<lifetime_matrix&>().transpose())::ReadOnly == 1);
static_assert(decltype(std::declval<lifetime_matrix&>().rowwise().sum())::ReadOnly == 1);
static_assert(native::is_matrix_v<lifetime_matrix&>);
static_assert(native::is_matrix_v<const lifetime_matrix&>);
static_assert(native::is_vector_v<const native::Matrix<double, 2, 1>&>);
static_assert(!native::is_vector_v<int>);
static_assert(native::is_boolean_matrix_v<lifetime_mask&>);
static_assert(native::is_boolean_vector_v<const native::Matrix<bool, 1, 2>&>);
static_assert(!native::is_boolean_vector_v<int>);
static_assert(!std::is_default_constructible_v<native::MatrixView<double, 2, 2>>);
static_assert(std::is_default_constructible_v<native::MatrixView<double, fdapde::Dynamic, fdapde::Dynamic>>);
static_assert(!std::is_default_constructible_v<native::MatrixView<bool, 2, 2>>);
static_assert(std::is_default_constructible_v<native::MatrixView<bool, fdapde::Dynamic, fdapde::Dynamic>>);

template <int StorageOrder> void check_construction_and_storage() {
    using matrix_type = native::Matrix<double, 2, 3, StorageOrder>;
    constexpr matrix_type compile_time_matrix({1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    static_assert(compile_time_matrix.rows() == 2);
    static_assert(compile_time_matrix.cols() == 3);
    static_assert(compile_time_matrix(1, 2) == 6.0);

    matrix_type matrix = compile_time_matrix;
    EXPECT_EQ(matrix.size(), 6);
    EXPECT_DOUBLE_EQ(matrix(1, 2), 6.0);

    const std::array<double, 6> expected = StorageOrder == native::RowMajor
      ? std::array<double, 6> {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}
      : std::array<double, 6> {1.0, 4.0, 2.0, 5.0, 3.0, 6.0};
    for (int i = 0; i < matrix.size(); ++i) EXPECT_DOUBLE_EQ(matrix.data()[i], expected[i]);

    matrix(1, 2) = 9.0;
    EXPECT_DOUBLE_EQ(matrix(1, 2), 9.0);

    native::Matrix<double, fdapde::Dynamic, fdapde::Dynamic, StorageOrder> dynamic;
    dynamic = compile_time_matrix;
    EXPECT_EQ(dynamic.rows(), 2);
    EXPECT_EQ(dynamic.cols(), 3);
    EXPECT_DOUBLE_EQ(dynamic(1, 2), 6.0);

    native::Matrix<double, fdapde::Dynamic, 3, StorageOrder> dynamic_rows(2, 3);
    EXPECT_EQ(dynamic_rows.rows(), 2);
    EXPECT_EQ(dynamic_rows.cols(), 3);
    EXPECT_EQ(static_cast<int>(dynamic_rows.end() - dynamic_rows.begin()), 6);

    native::Matrix<double, 2, fdapde::Dynamic, StorageOrder> dynamic_cols(2, 3);
    EXPECT_EQ(dynamic_cols.rows(), 2);
    EXPECT_EQ(dynamic_cols.cols(), 3);
    EXPECT_EQ(static_cast<int>(dynamic_cols.end() - dynamic_cols.begin()), 6);

    native::Matrix<double, fdapde::Dynamic, 1, StorageOrder> initialized_vector;
    initialized_vector = {1.0, 2.0, 3.0};
    EXPECT_EQ(initialized_vector.rows(), 3);
    EXPECT_EQ(initialized_vector.cols(), 1);
    EXPECT_EQ(initialized_vector, (native::Matrix<double, 3, 1, StorageOrder>({1.0, 2.0, 3.0})));

    const auto ones_functor = [](int, int) { return 1.0; };
    native::ProceduralMatrix<decltype(ones_functor), 2, fdapde::Dynamic> procedural(2, 3, ones_functor);
    procedural.resize(2, 4);
    EXPECT_EQ(procedural.rows(), 2);
    EXPECT_EQ(procedural.cols(), 4);
    EXPECT_DOUBLE_EQ(procedural(1, 3), 1.0);

    constexpr int other_order = StorageOrder == native::RowMajor ? native::ColMajor : native::RowMajor;
    native::Matrix<double, 2, 3, other_order> other = compile_time_matrix;
    EXPECT_DOUBLE_EQ(other(0, 1), 2.0);
    EXPECT_DOUBLE_EQ(other(1, 2), 6.0);

    std::array<double, 6> view_storage {};
    native::MatrixView<double, 2, 3, StorageOrder> view(view_storage.data());
    view(0, 1) = 7.0;
    constexpr int view_index = StorageOrder == native::RowMajor ? 1 : 2;
    EXPECT_DOUBLE_EQ(view_storage[view_index], 7.0);
    const auto& const_view = view;
    static_assert(std::is_same_v<decltype(const_view(0, 0)), const double&>);
    EXPECT_DOUBLE_EQ(const_view(0, 1), 7.0);

    std::array<double, 6> source_storage {};
    native::MatrixView<double, 2, 3, StorageOrder> source_view(source_storage.data());
    source_view(1, 2) = 11.0;
    view = source_view;
    EXPECT_EQ(view.data(), view_storage.data());
    EXPECT_DOUBLE_EQ(view(1, 2), 11.0);
    view = compile_time_matrix;
    EXPECT_EQ(view.data(), view_storage.data());
    EXPECT_DOUBLE_EQ(view(1, 2), 6.0);
}

template <int StorageOrder> void check_arithmetic_and_reductions() {
    using matrix_type = native::Matrix<double, 2, 3, StorageOrder>;
    const matrix_type matrix({-4.0, 0.0, 2.0, 1.0, -3.0, 5.0});

    const matrix_type sum = matrix + matrix;
    const matrix_type expected_sum({-8.0, 0.0, 4.0, 2.0, -6.0, 10.0});
    EXPECT_EQ(sum, expected_sum);
    const native::Matrix<double, 3, 2, StorageOrder> transposed_sum = (matrix + matrix).transpose();
    EXPECT_EQ(transposed_sum, (native::Matrix<double, 3, 2, StorageOrder>({-8.0, 2.0, 0.0, -6.0, 4.0, 10.0})));

    const native::Matrix<double, 2, 2, StorageOrder> gram = matrix * matrix.transpose();
    EXPECT_EQ(gram, (native::Matrix<double, 2, 2, StorageOrder>({20.0, 6.0, 6.0, 35.0})));
    EXPECT_DOUBLE_EQ(matrix.sum(), 1.0);
    EXPECT_DOUBLE_EQ(matrix.prod(), 0.0);
    EXPECT_DOUBLE_EQ(matrix.mean(), 1.0 / 6.0);
    EXPECT_DOUBLE_EQ(matrix.squared_norm(), 55.0);
    EXPECT_DOUBLE_EQ(matrix.inf_norm(), 5.0);
    EXPECT_DOUBLE_EQ(matrix.max(), 5.0);
    EXPECT_DOUBLE_EQ(matrix.min(), -4.0);
    EXPECT_EQ(matrix.rowwise().sum(), (native::Matrix<double, 2, 1, StorageOrder>({-2.0, 3.0})));
    EXPECT_EQ(matrix.colwise().sum(), (native::Matrix<double, 1, 3, StorageOrder>({-3.0, -3.0, 7.0})));

    native::Matrix<double, 2, 3, StorageOrder> broadcast;
    broadcast.rowwise() = native::Matrix<double, 2, 1, StorageOrder>({1.0, 2.0});
    EXPECT_EQ(broadcast, (native::Matrix<double, 2, 3, StorageOrder>({1.0, 1.0, 1.0, 2.0, 2.0, 2.0})));

    const matrix_type zero;
    EXPECT_DOUBLE_EQ(zero.inf_norm(), 0.0);

    const matrix_type negative({-4.0, -2.0, -3.0, -9.0, -8.0, -7.0});
    EXPECT_DOUBLE_EQ(negative.max(), -2.0);
    EXPECT_EQ(negative.rowwise().max(), (native::Matrix<double, 2, 1, StorageOrder>({-2.0, -7.0})));
    EXPECT_EQ(
      negative.colwise().max(), (native::Matrix<double, 1, 3, StorageOrder>({-4.0, -2.0, -3.0})));

    const matrix_type quotient = matrix / 2;
    EXPECT_EQ(quotient, (matrix_type({-2.0, 0.0, 1.0, 0.5, -1.5, 2.5})));

    native::Matrix<double, 2, 2, StorageOrder> aliased({1.0, 2.0, 3.0, 4.0});
    aliased += aliased.transpose();
    EXPECT_EQ(aliased, (native::Matrix<double, 2, 2, StorageOrder>({2.0, 5.0, 5.0, 8.0})));
    aliased = native::Matrix<double, 2, 2, StorageOrder>({1.0, 2.0, 3.0, 4.0});
    aliased *= aliased;
    EXPECT_EQ(aliased, (native::Matrix<double, 2, 2, StorageOrder>({7.0, 10.0, 15.0, 22.0})));

    native::Matrix<double, 2, 2, StorageOrder> cwise_aliased({1.0, 2.0, 3.0, 4.0});
    cwise_aliased.cwise() += cwise_aliased.transpose().cwise();
    EXPECT_EQ(cwise_aliased, (native::Matrix<double, 2, 2, StorageOrder>({2.0, 5.0, 5.0, 8.0})));

    native::Matrix<double, 2, 2, StorageOrder> cwise_source({1.0, 4.0, 9.0, 16.0});
    cwise_source.cwise() += 1.0;
    EXPECT_EQ(cwise_source, (native::Matrix<double, 2, 2, StorageOrder>({2.0, 5.0, 10.0, 17.0})));
    cwise_source.cwise() -= 1.0;
    native::Matrix<double, 2, 2, StorageOrder> scalar_assigned;
    scalar_assigned.cwise() = 3.0;
    EXPECT_EQ(scalar_assigned, (native::Matrix<double, 2, 2, StorageOrder>(3.0)));
    const native::Matrix<double, 2, 2, StorageOrder> shifted = cwise_source.cwise() + 1.0;
    EXPECT_EQ(shifted, (native::Matrix<double, 2, 2, StorageOrder>({2.0, 5.0, 10.0, 17.0})));
    const native::Matrix<double, 2, 2, StorageOrder> reciprocal = 18.0 / cwise_source.cwise();
    EXPECT_EQ(reciprocal, (native::Matrix<double, 2, 2, StorageOrder>({18.0, 4.5, 2.0, 1.125})));
    const native::Matrix<double, 2, 2, StorageOrder> roots = -cwise_source.cwise().sqrt();
    EXPECT_EQ(roots, (native::Matrix<double, 2, 2, StorageOrder>({-1.0, -2.0, -3.0, -4.0})));
    native::Matrix<bool, 2, 2, StorageOrder> comparison_mask = cwise_source.cwise() < 10.0;
    EXPECT_EQ(
      comparison_mask,
      (native::Matrix<bool, 2, 2, StorageOrder>({true, true, true, false})));
    comparison_mask = cwise_source.cwise() >= 9.0;
    EXPECT_EQ(
      comparison_mask,
      (native::Matrix<bool, 2, 2, StorageOrder>({false, false, true, true})));
    const native::Matrix<double, 4, 1, StorageOrder> reshaped_roots =
      cwise_source.template reshape<4, 1>().cwise().sqrt();
    const native::Matrix<double, 4, 1, StorageOrder> expected_reshaped_roots = StorageOrder == native::RowMajor
      ? native::Matrix<double, 4, 1, StorageOrder>({1.0, 2.0, 3.0, 4.0})
      : native::Matrix<double, 4, 1, StorageOrder>({1.0, 3.0, 2.0, 4.0});
    EXPECT_EQ(reshaped_roots, expected_reshaped_roots);

    const native::Matrix<double, 1, 3, StorageOrder> row({1.0, 2.0, 3.0});
    const native::Matrix<double, 1, 3, StorageOrder> shifted_row = row.cwise() + 1.0;
    EXPECT_DOUBLE_EQ(shifted_row[2], 4.0);
    const native::Matrix<double, fdapde::Dynamic, 1, StorageOrder> column_from_row(row);
    EXPECT_EQ(column_from_row.rows(), 3);
    EXPECT_EQ(column_from_row.cols(), 1);
    EXPECT_EQ(column_from_row, (native::Matrix<double, 3, 1, StorageOrder>({1.0, 2.0, 3.0})));

    constexpr native::Matrix<double, 1, 1, StorageOrder> one(1.0);
    constexpr native::Matrix<double, 1, 1, StorageOrder> exponential = one.cwise().exp();
    constexpr native::Matrix<double, 1, 1, StorageOrder> logarithm = one.cwise().log();
    static_assert(exponential[0] > 2.71828 && exponential[0] < 2.71829);
    static_assert(logarithm[0] > -1.0e-8 && logarithm[0] < 1.0e-8);
    constexpr native::Matrix<double, 1, 1, StorageOrder> smallest(
      std::numeric_limits<double>::denorm_min());
    constexpr native::Matrix<double, 1, 1, StorageOrder> smallest_log = smallest.cwise().log();
    static_assert(smallest_log[0] > -745.0 && smallest_log[0] < -744.0);
    constexpr native::Matrix<double, 1, 1, StorageOrder> near_underflow(-744.0);
    constexpr native::Matrix<double, 1, 1, StorageOrder> subnormal_exp = near_underflow.cwise().exp();
    static_assert(subnormal_exp[0] > 0.0 && subnormal_exp[0] < std::numeric_limits<double>::min());

    const native::Matrix<int, 1, 2, StorageOrder> integers({2, 4});
    const native::Matrix<double, 1, 2, StorageOrder> inverses = integers.cwise().inv();
    EXPECT_EQ(inverses, (native::Matrix<double, 1, 2, StorageOrder>({0.5, 0.25})));

    static_assert(decltype(matrix + matrix)::StorageOrder == StorageOrder);
    static_assert(std::is_same_v<typename decltype(cwise_source.cwise() < 2.0)::Scalar, bool>);
}

template <int StorageOrder> void check_blocks() {
    using matrix_type = native::Matrix<double, 3, 4, StorageOrder>;
    matrix_type matrix({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0});

    const auto fixed = matrix.template block<2, 2>(1, 1);
    EXPECT_EQ(fixed, (native::Matrix<double, 2, 2, StorageOrder>({6.0, 7.0, 10.0, 11.0})));
    const auto dynamic = matrix.block(0, 2, 3, 2);
    EXPECT_EQ(dynamic, (native::Matrix<double, 3, 2, StorageOrder>({3.0, 4.0, 7.0, 8.0, 11.0, 12.0})));
    EXPECT_EQ(matrix.row(2), (native::Matrix<double, 1, 4, StorageOrder>({9.0, 10.0, 11.0, 12.0})));
    EXPECT_EQ(matrix.col(0), (native::Matrix<double, 3, 1, StorageOrder>({1.0, 5.0, 9.0})));

    auto computed = matrix + matrix;
    auto computed_row = computed.row(0);
    static_assert(std::bidirectional_iterator<typename decltype(computed_row)::iterator>);
    double computed_sum = 0.0;
    for (const double value : computed_row) computed_sum += value;
    EXPECT_DOUBLE_EQ(computed_sum, 20.0);

    const auto row_reshape = matrix.template reshape<1, 12>();
    const auto col_reshape = matrix.template reshape<12, 1>();
    for (int i = 0; i < matrix.size(); ++i) {
        EXPECT_DOUBLE_EQ(row_reshape[i], matrix.data()[i]);
        EXPECT_DOUBLE_EQ(col_reshape[i], matrix.data()[i]);
    }

    matrix.template block<2, 2>(0, 0) = native::Matrix<double, 2, 2, StorageOrder>({20.0, 21.0, 22.0, 23.0});
    EXPECT_DOUBLE_EQ(matrix(0, 0), 20.0);
    EXPECT_DOUBLE_EQ(matrix(1, 1), 23.0);

    matrix.template block<2, 2>(0, 0) = matrix.template block<2, 2>(1, 2);
    EXPECT_EQ(
      (matrix.template block<2, 2>(0, 0)),
      (native::Matrix<double, 2, 2, StorageOrder>({7.0, 8.0, 11.0, 12.0})));

    native::Matrix<double, 2, 2, StorageOrder> overlapping({1.0, 2.0, 3.0, 4.0});
    overlapping.template block<2, 2>(0, 0) = overlapping.template block<2, 2>(0, 0).transpose();
    EXPECT_EQ(overlapping, (native::Matrix<double, 2, 2, StorageOrder>({1.0, 3.0, 2.0, 4.0})));

    native::Matrix<double, 2, 3, StorageOrder> reshape_dst;
    const native::Matrix<double, 2, 3, StorageOrder> reshape_src({1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    reshape_dst.template reshape<1, 6>() = reshape_src.template reshape<1, 6>();
    EXPECT_EQ(reshape_dst, reshape_src);

    native::Matrix<double, 4, 1, StorageOrder> vector_reshape_dst;
    const native::Matrix<double, 2, 2, StorageOrder> vector_reshape_src({1.0, 2.0, 3.0, 4.0});
    vector_reshape_dst.template reshape<2, 2>() = vector_reshape_src;
    EXPECT_EQ((vector_reshape_dst.template reshape<2, 2>()), vector_reshape_src);

    matrix.row(2) = {30.0, 31.0, 32.0, 33.0};
    auto row = matrix.row(2);
    static_assert(std::bidirectional_iterator<typename decltype(row)::iterator>);
    EXPECT_EQ(row, (native::Matrix<double, 1, 4, StorageOrder>({30.0, 31.0, 32.0, 33.0})));
    int count = 0;
    for ([[maybe_unused]] const double value : row) ++count;
    EXPECT_EQ(count, row.size());

    const matrix_type& const_matrix = matrix;
    const auto const_block = const_matrix.template block<2, 2>(0, 0);
    static_assert(decltype(const_block)::ReadOnly == 1);
    static_assert(std::is_same_v<decltype(const_block(0, 0)), const double&>);
}

template <int StorageOrder> void check_boolean_packing() {
    using exact_pack = native::Matrix<bool, 8, 8, StorageOrder>;
    exact_pack bits;
    EXPECT_EQ(bits.bitpacks(), 1);
    bits(1, 6) = true;
    constexpr int expected_bit = StorageOrder == native::RowMajor ? 14 : 49;
    EXPECT_NE(bits.bitpack(0) & (typename exact_pack::bitpack_t(1) << expected_bit), 0u);
    const exact_pack copy = bits;
    EXPECT_EQ(copy, bits);
    const exact_pack full(true);
    EXPECT_TRUE(full.all());
    EXPECT_EQ(full.count(), 64);
    exact_pack assigned;
    assigned = full;
    EXPECT_EQ(assigned, full);

    native::Matrix<bool, fdapde::Dynamic, fdapde::Dynamic, StorageOrder> dynamic_assigned(1, 1);
    const native::Matrix<bool, fdapde::Dynamic, fdapde::Dynamic, StorageOrder> dynamic_source(2, 3, true);
    dynamic_assigned = dynamic_source;
    EXPECT_EQ(dynamic_assigned.rows(), 2);
    EXPECT_EQ(dynamic_assigned.cols(), 3);
    EXPECT_TRUE(dynamic_assigned.all());

    exact_pack proxy_bits;
    proxy_bits(0, 1) = true;
    proxy_bits(0, 0) = proxy_bits(0, 1);
    proxy_bits(0, 1) = false;
    EXPECT_TRUE(bool(proxy_bits(0, 0)));
    EXPECT_FALSE(bool(proxy_bits(0, 1)));

    native::Matrix<bool, fdapde::Dynamic, fdapde::Dynamic, StorageOrder> empty;
    EXPECT_EQ(empty.bitpacks(), 0);
    EXPECT_EQ(empty, empty);
    EXPECT_FALSE(empty.any());
    EXPECT_TRUE(empty.all());
    EXPECT_EQ(empty.count(), 0);
    EXPECT_TRUE(empty.which(true).empty());

    using dynamic_row = native::Matrix<bool, 1, fdapde::Dynamic, StorageOrder>;
    const auto row_zeros = dynamic_row::Zero(5);
    const auto row_ones = dynamic_row::Ones(5);
    EXPECT_EQ(row_zeros.rows(), 1);
    EXPECT_EQ(row_zeros.cols(), 5);
    EXPECT_EQ(row_ones.rows(), 1);
    EXPECT_EQ(row_ones.cols(), 5);
    EXPECT_FALSE((native::Matrix<bool, 1, fdapde::Dynamic, StorageOrder>(row_zeros).any()));
    EXPECT_TRUE((native::Matrix<bool, 1, fdapde::Dynamic, StorageOrder>(row_ones).all()));

    for (const int size : {1, 63, 64, 65, 130}) {
        native::Matrix<bool, fdapde::Dynamic, 1, StorageOrder> packed(size);
        EXPECT_EQ(packed.bitpacks(), (size + int(packed.PackSize) - 1) / int(packed.PackSize));
        packed[size - 1] = true;
        EXPECT_EQ(packed.count(), 1);
    }

    std::vector<bool> packed_input(130, false);
    for (const int index : {0, 63, 64, 129}) packed_input[index] = true;
    const native::Matrix<bool, fdapde::Dynamic, 1, StorageOrder> packed_from_vector(packed_input);
    EXPECT_EQ(packed_from_vector.bitpacks(), 3);
    EXPECT_EQ(packed_from_vector.count(), 4);
    for (const int index : {0, 63, 64, 129}) EXPECT_TRUE(bool(packed_from_vector[index]));

    const native::Matrix<bool, 2, 2, StorageOrder> fixed_from_vector(
      std::vector<bool> {true, false, false, true});
    EXPECT_EQ(fixed_from_vector.bitpacks(), 1);
    EXPECT_EQ(fixed_from_vector.count(), 2);

    native::Matrix<bool, fdapde::Dynamic, 1, StorageOrder> resized(10);
    resized[9] = true;
    resized.resize(5);
    EXPECT_EQ(resized.count(), 0);
    resized.resize(10);
    EXPECT_FALSE(bool(resized[9]));

    native::Matrix<bool, 5, 13, StorageOrder> overflow;
    EXPECT_EQ(overflow.bitpacks(), 2);
    EXPECT_FALSE(overflow.any());
    overflow.set();
    EXPECT_TRUE(overflow.all());
    EXPECT_EQ(overflow.count(), 65);
    overflow(4, 12) = false;
    EXPECT_FALSE(overflow.all());
    EXPECT_EQ(overflow.count(), 64);
    overflow.clear();
    EXPECT_FALSE(overflow.any());

    const native::Matrix<bool, 2, 2, StorageOrder> diagonal({true, false, false, true});
    const native::Matrix<bool, 2, 2, StorageOrder> anti_diagonal({false, true, true, false});
    EXPECT_EQ(~diagonal, anti_diagonal);
    EXPECT_EQ(diagonal & anti_diagonal, (native::Matrix<bool, 2, 2, StorageOrder>()));
    EXPECT_EQ(diagonal | anti_diagonal, (native::Matrix<bool, 2, 2, StorageOrder>(true)));
    EXPECT_EQ(diagonal ^ anti_diagonal, (native::Matrix<bool, 2, 2, StorageOrder>(true)));
    const auto full_expression = diagonal | anti_diagonal;
    EXPECT_EQ((full_expression.template reshape<1, 4>().count()), 4);

    native::Matrix<bool, 1, 4, StorageOrder> bool_aliased({true, false, true, true});
    bool_aliased.right_cols(3) &= bool_aliased.left_cols(3);
    EXPECT_EQ(bool_aliased, (native::Matrix<bool, 1, 4, StorageOrder>({true, false, false, true})));

    const native::Matrix<double, 2, 2, StorageOrder> lhs({1.0, 2.0, 3.0, 4.0});
    const native::Matrix<double, 2, 2, StorageOrder> rhs({10.0, 20.0, 30.0, 40.0});
    const native::Matrix<double, 2, 2, StorageOrder> selected = diagonal.select(lhs, rhs);
    EXPECT_EQ(selected, (native::Matrix<double, 2, 2, StorageOrder>({1.0, 20.0, 30.0, 4.0})));
    const native::Matrix<double, 2, 2, StorageOrder> inverse_selected = (~diagonal).select(lhs, rhs);
    EXPECT_EQ(inverse_selected, (native::Matrix<double, 2, 2, StorageOrder>({10.0, 2.0, 3.0, 40.0})));

    const native::Matrix<bool, 2, 3, StorageOrder> layout({false, true, true, true, false, false});
    EXPECT_EQ(layout.which(true), (std::vector<int> {1, 2, 3}));
    EXPECT_EQ(
      layout.template right_cols<2>(), (native::Matrix<bool, 2, 2, StorageOrder>({true, true, false, false})));
    EXPECT_EQ(layout.right_cols(1), (native::Matrix<bool, 2, 1, StorageOrder>({true, false})));
    const auto bool_row = layout.template reshape<1, 6>();
    const auto bool_col = layout.template reshape<6, 1>();
    for (int i = 0; i < layout.size(); ++i) {
        const bool expected =
          (layout.bitpack(0) & (typename decltype(layout)::bitpack_t(1) << i)) != 0;
        EXPECT_EQ(bool(bool_row[i]), expected);
        EXPECT_EQ(bool(bool_col[i]), expected);
    }
    EXPECT_EQ(bool_row.count(), layout.count());
    EXPECT_EQ(bool_col.count(), layout.count());

    native::Matrix<bool, 2, 4, StorageOrder> bool_blocks(
      {false, false, true, false, false, false, false, true});
    bool_blocks.template block<2, 2>(0, 0) = bool_blocks.template block<2, 2>(0, 2);
    const auto bool_block = bool_blocks.template left_cols<2>();
    EXPECT_EQ(
      bool_block,
      (native::Matrix<bool, 2, 2, StorageOrder>({true, false, false, true})));
    EXPECT_TRUE(bool_block.any());
    EXPECT_FALSE(bool_block.all());
    EXPECT_EQ(bool_block.count(), 2);

    native::Matrix<bool, 2, 3, StorageOrder> bool_reshape_dst;
    bool_reshape_dst.template reshape<1, 6>() = layout.template reshape<1, 6>();
    EXPECT_EQ(bool_reshape_dst, layout);

    constexpr int other_order = StorageOrder == native::RowMajor ? native::ColMajor : native::RowMajor;
    const native::Matrix<bool, 2, 3, other_order> other_layout({false, true, true, true, false, false});
    EXPECT_FALSE((layout ^ other_layout).any());

    const double nan = std::numeric_limits<double>::quiet_NaN();
    const native::Matrix<double, 2, 2, StorageOrder> nan_matrix({0.0, nan, 2.0, nan});
    const auto matrix_mask = native::nan_indicator(nan_matrix);
    EXPECT_EQ(matrix_mask.rows(), 2);
    EXPECT_EQ(matrix_mask.cols(), 2);
    EXPECT_EQ(matrix_mask.count(), 2);
    EXPECT_TRUE(bool(matrix_mask(0, 1)));
    EXPECT_TRUE(bool(matrix_mask(1, 1)));

    const std::vector<double> nan_vector {nan, 1.0, nan};
    const auto vector_mask = native::nan_indicator(nan_vector);
    EXPECT_EQ(vector_mask.rows(), 3);
    EXPECT_EQ(vector_mask.cols(), 1);
    EXPECT_EQ(vector_mask.count(), 2);
    EXPECT_TRUE(bool(vector_mask(0, 0)));
    EXPECT_TRUE(bool(vector_mask(2, 0)));

    using view_type = native::MatrixView<bool, 2, 2, StorageOrder>;
    using view_word = typename view_type::bitpack_t;
    static_assert(std::is_constructible_v<view_type, view_word*>);
    static_assert(!std::is_constructible_v<view_type, unsigned char*>);
    view_word view_storage = 0;
    view_type safe_view(&view_storage);
    safe_view(1, 0) = true;
    constexpr int view_bit = StorageOrder == native::RowMajor ? 2 : 1;
    EXPECT_TRUE(bool(safe_view(1, 0)));
    EXPECT_NE(view_storage & (view_word(1) << view_bit), view_word(0));

    view_word source_view_storage = 0;
    view_type source_view(&source_view_storage);
    source_view(0, 1) = true;
    safe_view = source_view;
    EXPECT_EQ(safe_view.data(), &view_storage);
    EXPECT_TRUE(bool(safe_view(0, 1)));
    safe_view = diagonal;
    EXPECT_EQ(safe_view.data(), &view_storage);
    EXPECT_TRUE(bool(safe_view(0, 0)));
    EXPECT_FALSE(bool(safe_view(0, 1)));

    native::MatrixView<bool, fdapde::Dynamic, fdapde::Dynamic, StorageOrder> empty_view;
    empty_view.set();
    empty_view.clear();
}

}   // namespace

TEST(NativeDenseMatrix, ConstructionAccessAndStorageOrder) {
    check_construction_and_storage<native::RowMajor>();
    check_construction_and_storage<native::ColMajor>();
}

TEST(NativeDenseMatrix, ArithmeticAndReductions) {
    check_arithmetic_and_reductions<native::RowMajor>();
    check_arithmetic_and_reductions<native::ColMajor>();
}

TEST(NativeDenseMatrix, BlocksAreViews) {
    check_blocks<native::RowMajor>();
    check_blocks<native::ColMajor>();
}

TEST(NativeDenseBoolean, PackingLogicAndSelection) {
    check_boolean_packing<native::RowMajor>();
    check_boolean_packing<native::ColMajor>();
}
