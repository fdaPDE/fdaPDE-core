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

TEST(svg, parses_numeric_grammar_and_relative_commands) {
    const auto rings = fdapde::svg_document_rings(
      R"svg(<?xml version="1.0"?><svg viewBox="0 0 20 20"><path d="M1e1,-2.5 L.5-.25 l+1,+2 z"/></svg>)svg", 2);
    ASSERT_EQ(rings.size(), 1);
    ASSERT_EQ(rings[0].rows(), 3);
    EXPECT_DOUBLE_EQ(rings[0](0, 0), 10.0);
    EXPECT_DOUBLE_EQ(rings[0](0, 1), 2.5);
    EXPECT_DOUBLE_EQ(rings[0](1, 0), 0.5);
    EXPECT_DOUBLE_EQ(rings[0](1, 1), 0.25);
    EXPECT_DOUBLE_EQ(rings[0](2, 0), 1.5);
    EXPECT_DOUBLE_EQ(rings[0](2, 1), -1.75);

    const auto implicit_lines = fdapde::svg_document_rings(R"svg(<svg><path d='M0 0 2 0 2 2 0 2Z'/></svg>)svg", 2);
    ASSERT_EQ(implicit_lines[0].rows(), 4);
    EXPECT_DOUBLE_EQ(implicit_lines[0](2, 0), 2.0);
    EXPECT_DOUBLE_EQ(implicit_lines[0](2, 1), -2.0);
}

TEST(svg, flattens_cubics_and_converts_to_cartesian_coordinates) {
    const auto rings = fdapde::svg_document_rings(R"svg(<svg><path d="M0 0 C0 1 1 1 1 0 L0 0 Z"/></svg>)svg", 2);
    ASSERT_EQ(rings.size(), 1);
    ASSERT_EQ(rings[0].rows(), 3);
    EXPECT_DOUBLE_EQ(rings[0](0, 0), 0.0);
    EXPECT_DOUBLE_EQ(rings[0](0, 1), 0.0);
    EXPECT_DOUBLE_EQ(rings[0](1, 0), 0.5);
    EXPECT_DOUBLE_EQ(rings[0](1, 1), -0.75);
    EXPECT_DOUBLE_EQ(rings[0](2, 0), 1.0);
    EXPECT_DOUBLE_EQ(rings[0](2, 1), 0.0);
}

TEST(svg, preserves_multiple_paths_and_subpaths_in_source_order) {
    const auto rings = fdapde::svg_document_rings(
      R"svg(<svg xmlns="http://www.w3.org/2000/svg">
        <!-- synthetic independent regions -->
        <path d="M0 0L1 0L0 1Z M2 2l1 0 0 1z"/>
        <g><path fill="none" d='M4 4L5 4L4 5z'/></g>
      </svg>)svg",
      3);
    ASSERT_EQ(rings.size(), 3);
    EXPECT_DOUBLE_EQ(rings[0](0, 0), 0.0);
    EXPECT_DOUBLE_EQ(rings[1](0, 0), 2.0);
    EXPECT_DOUBLE_EQ(rings[2](0, 0), 4.0);
    EXPECT_DOUBLE_EQ(rings[2](0, 1), -4.0);
    for (const auto& ring : rings) {
        const auto mesh = fdapde::constrained_delaunay(ring);
        EXPECT_EQ(mesh.n_boundary_edges(), ring.rows());
        EXPECT_NEAR(mesh.measure(), 0.5, 1e-14);
    }
}

TEST(svg, composes_with_constrained_delaunay_for_lines_and_curves) {
    const auto square = fdapde::svg_document_rings(
      R"svg(<svg width="2cm" viewBox="0 0 99 99"><path d="M0 0L2 0L2 2L0 2Z"/></svg>)svg", 4);
    const auto square_mesh = fdapde::constrained_delaunay(square[0]);
    EXPECT_NEAR(square_mesh.measure(), 4.0, 1e-14);
    EXPECT_EQ(square_mesh.n_boundary_edges(), 4);

    const auto curved =
      fdapde::svg_document_rings(R"svg(<svg><path d="M0 0 C0 1 1 1 1 0 L1 -1 L0 -1 Z"/></svg>)svg", 8);
    ASSERT_EQ(curved.size(), 1);
    const auto curved_mesh = fdapde::constrained_delaunay(curved[0]);
    EXPECT_EQ(curved_mesh.n_boundary_edges(), curved[0].rows());
    EXPECT_GT(curved_mesh.measure(), 1.0);
    EXPECT_EQ(curved, fdapde::svg_document_rings(R"svg(<svg><path d="M0 0 C0 1 1 1 1 0 L1 -1 L0 -1 Z"/></svg>)svg", 8));
}

TEST(svg, parses_compact_repeated_relative_cubics_from_multiple_paths) {
    const auto rings = fdapde::svg_document_rings(
      R"svg(<?xml version="1.0"?>
        <svg viewBox="0 0 8 8">
          <defs><style>.outline { fill: none; }</style></defs>
          <path class="outline" d="M1 1c.5 0 1 .5 1 1 0 .5-.5 1-1 1-.5 0-1-.5-1-1 0-.5.5-1 1-1Z"/>
          <path d="M5,1c.5,0,1,.5,1,1 0,.5-.5,1-1,1-.5,0-1-.5-1-1 0-.5.5-1 1-1z"/>
          <path d="M1 5c.5 0 1 .5 1 1 0 .5-.5 1-1 1-.5 0-1-.5-1-1 0-.5.5-1 1-1Z"/>
          <path d="M5 5c.5 0 1 .5 1 1 0 .5-.5 1-1 1-.5 0-1-.5-1-1 0-.5.5-1 1-1Z"/>
        </svg>)svg",
      4);
    ASSERT_EQ(rings.size(), 4);
    for (const auto& ring : rings) {
        ASSERT_EQ(ring.rows(), 16);
        const auto resampled = fdapde::resample_polygon_ring(ring, 0.3, 8);
        const auto mesh = fdapde::constrained_delaunay(resampled);
        EXPECT_EQ(mesh.n_boundary_edges(), resampled.rows());
        EXPECT_GT(mesh.measure(), 0.0);
    }
    EXPECT_DOUBLE_EQ(rings[0](0, 0), 1.0);
    EXPECT_DOUBLE_EQ(rings[1](0, 0), 5.0);
    EXPECT_DOUBLE_EQ(rings[2](0, 1), -5.0);
    EXPECT_DOUBLE_EQ(rings[3](0, 1), -5.0);
}

TEST(svg, rejects_unsupported_or_malformed_documents) {
    const auto rejects = [](std::string_view document) {
        EXPECT_THROW(fdapde::svg_document_rings(document, 2), std::invalid_argument);
    };
    rejects("");
    rejects("<path d='M0 0L1 0L0 1Z'/>");
    rejects("<path d='M0 0L1 0L0 1Z'/><svg></svg>");
    rejects("<svg></svg>");
    rejects("<svg><path d='M0 0L1 0L0 1Z'/>");
    rejects("<svg></svg><svg><path d='M0 0L1 0L0 1Z'/></svg>");
    rejects("<svg><path d=''/></svg>");
    rejects("<svg><path d='M0 0L1 0L0 1Z' x:d='M0 0L1 0L0 1Z'/></svg>");
    rejects("<svg><path d='M0 0L1 0L0 1'/></svg>");
    rejects("<svg><path d='M0 0L1Z'/></svg>");
    rejects("<svg><path d='M0 0L1 0L0 1Z 2 2'/></svg>");
    rejects("<svg><path d='M0,,0L1 0L0 1Z'/></svg>");
    rejects("<svg><path d='M0 0L1e999 0L0 1Z'/></svg>");
    rejects("<svg><path d='M0 0L1 0L0 1Z&amp;'/></svg>");
    rejects("<svg transform='scale(2)'><path d='M0 0L1 0L0 1Z'/></svg>");
    rejects("<svg><g transform='translate(1 2)'><path d='M0 0L1 0L0 1Z'/></g></svg>");
    rejects("<svg><rect x='0' y='0' width='1' height='1'/></svg>");
    rejects("<svg><path d='M0 0L1 0L0 1Z'/><image href='fixture.png'/></svg>");
    rejects("<svg>unsupported text<path d='M0 0L1 0L0 1Z'/></svg>");
    rejects("<svg><defs><path d='M0 0L1 0L0 1Z'/></defs><path d='M2 0L3 0L2 1Z'/></svg>");
    rejects("<!DOCTYPE svg><svg><path d='M0 0L1 0L0 1Z'/></svg>");
    rejects("<svg><path d='M0 0L1 0L0 1Z'/");
    rejects("garbage<svg><path d='M0 0L1 0L0 1Z'/></svg>");
    rejects("<svg><path d='M0 0L1 0L0 1Z'/></svg>trailing");
    rejects("<svg><defs><path d='M0 0L1 0L0 1Z'/></wrong></defs></svg>");
    rejects("<svg><g><path d='M0 0L1 0L0 1Z'/></svg>");
    rejects("<svg><svg><path d='M0 0L1 0L0 1Z'/></svg></svg>");
    EXPECT_THROW(fdapde::svg_document_rings("<svg><path d='M0 0L1 0L0 1Z'/></svg>", 0), std::invalid_argument);

    for (char command : std::string("HhVvQqTtSsAaXx")) {
        const std::string document = std::string("<svg><path d='M0 0") + command + "1 1L0 1Z'/></svg>";
        rejects(document);
    }
}

TEST(svg, rejects_degenerate_subpaths) {
    const auto rejects = [](std::string_view data) {
        const std::string document = std::string("<svg><path d='") + std::string(data) + "'/></svg>";
        EXPECT_THROW(fdapde::svg_document_rings(document, 4), std::invalid_argument);
    };
    rejects("M0 0L1 0Z");
    rejects("M0 0L1 0L1 0L0 1Z");
    rejects("M0 0L1 0L0 1L1 0Z");
    rejects("M0 0C0 0 0 0 0 0L1 0L0 1Z");
    rejects("M0 0L1 0L0 1M2 2L3 2L2 3Z");
}
