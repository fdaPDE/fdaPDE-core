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

#include <fdaPDE/geometry.h>
#include <gtest/gtest.h>

namespace lattice_testing {

using fdapde::Dynamic;
using fdapde::Matrix;

Matrix<double, Dynamic, Dynamic> points(std::initializer_list<std::array<double, 2>> coordinates) {
    Matrix<double, Dynamic, Dynamic> result(coordinates.size(), 2);
    int i = 0;
    for (const auto& point : coordinates) {
        result(i, 0) = point[0];
        result(i, 1) = point[1];
        ++i;
    }
    return result;
}

void expect_boundary(
  const Matrix<double, Dynamic, Dynamic>& actual, std::initializer_list<std::array<double, 2>> expected_coordinates) {
    const auto expected = points(expected_coordinates);
    ASSERT_EQ(actual.rows(), expected.rows());
    ASSERT_EQ(actual.cols(), expected.cols());
    EXPECT_EQ(actual, expected);
}

}   // namespace lattice_testing

TEST(mesh_generation, extracts_square_and_l_shaped_boundaries) {
    using namespace lattice_testing;
    const fdapde::Vector<double, 2> anchor(0, 0);
    const auto single = fdapde::square_lattice_boundary(
      points({
        {0, 0}
    }),
      2.0, anchor);
    expect_boundary(
      single, {
                {-1, -1},
                {1,  -1},
                {1,  1 },
                {-1, 1 }
    });

    const auto l_shape = fdapde::square_lattice_boundary(
      points({
        {0, 0},
        {2, 0},
        {0, 2}
    }),
      2.0, anchor);
    expect_boundary(
      l_shape, {
                 {-1, -1},
                 {3,  -1},
                 {3,  1 },
                 {1,  1 },
                 {1,  3 },
                 {-1, 3 }
    });
    const auto mesh = fdapde::constrained_delaunay(l_shape);
    EXPECT_NEAR(mesh.measure(), 12.0, 1e-14);
    EXPECT_EQ(mesh.n_boundary_edges(), l_shape.rows());
}

TEST(mesh_generation, selects_the_largest_component_deterministically) {
    using namespace lattice_testing;
    const fdapde::Vector<double, 2> anchor(0, 0);
    const auto largest = fdapde::square_lattice_boundary(
      points({
        {0,  0},
        {2,  0},
        {0,  2},
        {10, 0},
        {12, 0}
    }),
      2.0, anchor);
    expect_boundary(
      largest, {
                 {-1, -1},
                 {3,  -1},
                 {3,  1 },
                 {1,  1 },
                 {1,  3 },
                 {-1, 3 }
    });

    const auto tie = fdapde::square_lattice_boundary(
      points({
        {0,  0},
        {2,  0},
        {10, 0},
        {12, 0}
    }),
      2.0, anchor);
    expect_boundary(
      tie, {
             {-1, -1},
             {3,  -1},
             {3,  1 },
             {-1, 1 }
    });

    const auto corner_contact = fdapde::square_lattice_boundary(
      points({
        {0, 0},
        {2, 2}
    }),
      2.0, anchor);
    expect_boundary(
      corner_contact, {
                        {-1, -1},
                        {1,  -1},
                        {1,  1 },
                        {-1, 1 }
    });
}

TEST(mesh_generation, has_explicit_half_open_cell_ties) {
    using namespace lattice_testing;
    const fdapde::Vector<double, 2> anchor(0, 0);
    expect_boundary(
      fdapde::square_lattice_boundary(
        points({
          {1, 0}
    }),
        2.0, anchor),
      {{1, -1}, {3, -1}, {3, 1}, {1, 1}});
    expect_boundary(
      fdapde::square_lattice_boundary(
        points({
          {std::nextafter(1.0, -std::numeric_limits<double>::infinity()), 0}
    }),
        2.0, anchor),
      {{-1, -1}, {1, -1}, {1, 1}, {-1, 1}});
    expect_boundary(
      fdapde::square_lattice_boundary(
        points({
          {-1, 0}
    }),
        2.0, anchor),
      {{-1, -1}, {1, -1}, {1, 1}, {-1, 1}});
    expect_boundary(
      fdapde::square_lattice_boundary(
        points({
          {std::nextafter(-1.0, -std::numeric_limits<double>::infinity()), 0}
    }),
        2.0, anchor),
      {{-3, -1}, {-1, -1}, {-1, 1}, {-3, 1}});
}

TEST(mesh_generation, fills_holes_and_handles_degree_four_pinches) {
    using namespace lattice_testing;
    const fdapde::Vector<double, 2> anchor(0, 0);
    Matrix<double, Dynamic, Dynamic> ring_points(8, 2);
    int k = 0;
    for (int j = 0; j < 3; ++j) {
        for (int i = 0; i < 3; ++i) {
            if (i == 1 && j == 1) continue;
            ring_points(k, 0) = 2 * i;
            ring_points(k, 1) = 2 * j;
            ++k;
        }
    }
    const auto filled = fdapde::square_lattice_boundary(ring_points, 2.0, anchor);
    expect_boundary(
      filled, {
                {-1, -1},
                {5,  -1},
                {5,  5 },
                {-1, 5 }
    });
    EXPECT_NEAR(fdapde::constrained_delaunay(filled).measure(), 36.0, 1e-14);

    const auto pinch = fdapde::square_lattice_boundary(
      points({
        {0, 0 },
        {0, -2},
        {2, -2},
        {4, -2},
        {4, 0 },
        {4, 2 },
        {2, 2 }
    }),
      2.0, anchor);
    expect_boundary(
      pinch, {
               {-1, -3},
               {5,  -3},
               {5,  3 },
               {1,  3 },
               {1,  1 },
               {-1, 1 }
    });
}

TEST(mesh_generation, is_repeatable_and_translation_equivariant) {
    using namespace lattice_testing;
    const fdapde::Vector<double, 2> anchor(0, 0);
    const auto first = fdapde::square_lattice_boundary(
      points({
        {0, 0},
        {2, 0},
        {0, 2},
        {2, 0}
    }),
      2.0, anchor);
    const auto shuffled = fdapde::square_lattice_boundary(
      points({
        {2, 0},
        {0, 2},
        {0, 0},
        {2, 0}
    }),
      2.0, anchor);
    EXPECT_EQ(first, shuffled);

    const auto translated = fdapde::square_lattice_boundary(
      points({
        {10, 10},
        {12, 10}
    }),
      2.0);
    expect_boundary(
      translated, {
                    {9,  9 },
                    {13, 9 },
                    {13, 11},
                    {9,  11}
    });

    const auto default_anchor = fdapde::square_lattice_boundary(
      points({
        {5, 2},
        {7, 2},
        {5, 0}
    }),
      2.0);
    expect_boundary(
      default_anchor, {
                        {4, -1},
                        {6, -1},
                        {6, 1 },
                        {8, 1 },
                        {8, 3 },
                        {4, 3 }
    });

    const fdapde::Vector<double, 2> fractional_anchor(1.25, -2.5);
    const auto fractional = fdapde::square_lattice_boundary(
      points({
        {1.25, -2.5},
        {1.45, -2.5}
    }),
      0.2, fractional_anchor);
    ASSERT_EQ(fractional.rows(), 4);
    const auto expected_fractional = points({
      {1.15, -2.6},
      {1.55, -2.6},
      {1.55, -2.4},
      {1.15, -2.4}
    });
    for (int i = 0; i < fractional.rows(); ++i) {
        EXPECT_NEAR(fractional(i, 0), expected_fractional(i, 0), 1e-14);
        EXPECT_NEAR(fractional(i, 1), expected_fractional(i, 1), 1e-14);
    }
}

TEST(mesh_generation, rejects_invalid_input) {
    using namespace lattice_testing;
    Matrix<double, Dynamic, Dynamic> empty(0, 2);
    EXPECT_THROW(fdapde::square_lattice_boundary(empty, 1.0), std::invalid_argument);
    Matrix<double, Dynamic, Dynamic> wrong_columns(1, 3);
    EXPECT_THROW(fdapde::square_lattice_boundary(wrong_columns, 1.0), std::invalid_argument);
    EXPECT_THROW(
      fdapde::square_lattice_boundary(
        points({
          {0, 0}
    }),
        0.0),
      std::invalid_argument);
    EXPECT_THROW(
      fdapde::square_lattice_boundary(
        points({
          {0, 0}
    }),
        -1.0),
      std::invalid_argument);
    EXPECT_THROW(
      fdapde::square_lattice_boundary(
        points({
          {0, 0}
    }),
        std::numeric_limits<double>::quiet_NaN()),
      std::invalid_argument);
    EXPECT_THROW(
      fdapde::square_lattice_boundary(
        points({
          {0, 0}
    }),
        std::numeric_limits<double>::infinity()),
      std::invalid_argument);

    auto nonfinite = points({
      {0, 0}
    });
    nonfinite(0, 1) = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(fdapde::square_lattice_boundary(nonfinite, 1.0), std::invalid_argument);
    EXPECT_THROW(
      fdapde::square_lattice_boundary(
        points({
          {0, 0}
    }),
        1.0, fdapde::Vector<double, 2>(std::numeric_limits<double>::infinity(), 0.0)),
      std::invalid_argument);
    EXPECT_THROW(
      fdapde::square_lattice_boundary(
        points({
          {1e16, 1e16}
    }),
        1.0, fdapde::Vector<double, 2>(1e16, 1e16)),
      std::invalid_argument);
}
