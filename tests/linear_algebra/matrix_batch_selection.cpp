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

namespace {
using namespace fdapde;

// fixed simplex selections copy indices inline while preserving the borrowed coefficient and cache slots
TEST(MatrixBatchSelection, FixedIndicesRetainOrderAndBorrowedCaches) {
    using Point = SPDMatrix<double, 3, Cache::Log>;
    MatrixBatch<Point> points(3);
    points[0] = Matrix<double, 3, 3>({2., 0., 0., 0., 3., 0., 0., 0., 4.});
    points[1] = Matrix<double, 3, 3>({5., 0., 0., 0., 6., 0., 0., 0., 7.});
    points[2] = Matrix<double, 3, 3>({8., 0., 0., 0., 9., 0., 0., 0., 10.});
    std::array<int, 3> indices {2, 0, 2};
    const auto selected = points.select(indices);
    indices[0] = 1;
    // the array extent selects inline index storage at compile time
    static_assert(decltype(selected)::IndexCount == 3);
    // copying indices retains the selected order after the caller mutates its array
    EXPECT_EQ(selected[0].data(), points[2].data());
    // repeated positions retain the same source cache rather than copying cached logarithms
    EXPECT_EQ(selected[2].cache().data(), points[2].cache().data());
    points[0] = Matrix<double, 3, 3>({11., 0., 0., 0., 12., 0., 0., 0., 13.});
    // a later assignment to the borrowed owner remains visible through the persistent selection
    EXPECT_EQ(selected[1](0, 0), 11.);
    const auto mapped =
      points.select(std::array<int, 4> {2, 0, 1, 2}).map([](const auto& value) { return value(0, 0); });
    const auto result = mapped.eval();
    // the deferred map owns the temporary selection and preserves its four copied indices
    EXPECT_EQ(result[2](0, 0), 5.);
}

// fixed spans and native vectors share inline storage while runtime containers retain arbitrary counts
TEST(MatrixBatchSelection, FixedAndDynamicExtentsHaveTheSameSelectionContract) {
    MatrixBatch<Matrix<int, 1, 1>> points(3);
    for (int i = 0; i < 3; ++i) points[i](0, 0) = i + 1;
    const std::array<int, 3> indices {2, 0, 1};
    const auto span_selection = points.select(std::span<const int, 3>(indices));
    const auto vector_selection = points.select(Vector<int, 3>({2, 0, 1}));
    const auto dynamic_selection = points.select(std::vector<int> {2, 0, 1});
    // a fixed span preserves its compile-time extent instead of falling back to a heap vector
    static_assert(decltype(span_selection)::IndexCount == 3);
    // native fixed vectors expose a static coefficient count for the same inline path
    static_assert(decltype(vector_selection)::IndexCount == 3);
    // variable-length containers keep the unrestricted dynamic fallback
    static_assert(decltype(dynamic_selection)::IndexCount == Dynamic);
    for (std::size_t i = 0; i < indices.size(); ++i) {
        // each fixed span position matches the independently selected runtime index vector
        EXPECT_EQ(span_selection[i](0, 0), dynamic_selection[i](0, 0));
        // native fixed-vector indices use the same checked order as runtime indices
        EXPECT_EQ(vector_selection[i](0, 0), dynamic_selection[i](0, 0));
    }
    // negative fixed indices are rejected before indexing the borrowed batch
    EXPECT_THROW(static_cast<void>(points.select(std::array<int, 1> {-1})), std::out_of_range);
    // a fixed upper-bound violation is rejected by the same public validation as dynamic indices
    EXPECT_THROW(static_cast<void>(points.select(std::array<int, 1> {3})), std::out_of_range);
    const auto empty = points.select(std::array<int, 0> {});
    // a zero-extent selection has no entries without losing the source matrix shape
    EXPECT_EQ(empty.size(), std::size_t {0});
    // preserving rows allows an empty expression to materialize with the correct shape
    EXPECT_EQ(empty.rows(), 1);
}
}   // namespace
