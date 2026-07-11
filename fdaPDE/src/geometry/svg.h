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

#ifndef __FDAPDE_GEOMETRY_SVG_H__
#define __FDAPDE_GEOMETRY_SVG_H__

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include "header_check.h"

namespace fdapde {
namespace internals {
namespace svg {

// functional sources: RGCCA stable@4ff7f7b3b4d6b0f1ed984151d79b3f48a9f83299 and
// fdaPDE-test-bench develop-svg@29aed29d8700d137e5874ea2ada462fff9cc62d1. This is an independent,
// dependency-free rewrite of the supported path and Cartesian-coordinate behavior.

struct point_t {
    double x;
    double y;
};

inline bool equal(const point_t& lhs, const point_t& rhs) { return lhs.x == rhs.x && lhs.y == rhs.y; }

class path_parser {
   public:
    path_parser(std::string_view data, int cubic_subdivisions) : data_(data), subdivisions_(cubic_subdivisions) { }

    std::vector<std::vector<point_t>> parse() {
        while (true) {
            skip_whitespace();
            if (position_ == data_.size()) break;
            const char token = data_[position_];
            if (is_command(token)) {
                command_ = token;
                ++position_;
                number_precedes_ = false;
                if (command_ == 'Z' || command_ == 'z') {
                    close_ring();
                    command_ = 0;
                } else if (!is_supported(command_)) {
                    throw std::invalid_argument("Unsupported SVG path command.");
                }
                continue;
            }
            if (command_ == 0) { throw std::invalid_argument("SVG path data requires a command before coordinates."); }

            switch (command_) {
            case 'M':
            case 'm':
                move_to(command_ == 'm');
                break;
            case 'L':
            case 'l':
                line_to(command_ == 'l');
                break;
            case 'C':
            case 'c':
                cubic_to(command_ == 'c');
                break;
            default:
                throw std::invalid_argument("Unsupported SVG path command.");
            }
        }
        if (ring_open_) { throw std::invalid_argument("SVG subpaths must be explicitly closed with Z."); }
        if (rings_.empty()) { throw std::invalid_argument("SVG path data contains no closed subpath."); }
        return rings_;
    }
   private:
    static bool is_command(char value) { return (value >= 'A' && value <= 'Z') || (value >= 'a' && value <= 'z'); }

    static bool is_supported(char command) {
        return command == 'M' || command == 'm' || command == 'L' || command == 'l' || command == 'C' || command == 'c';
    }

    static bool is_digit(char value) { return value >= '0' && value <= '9'; }

    void skip_whitespace() {
        while (position_ < data_.size()) {
            const char value = data_[position_];
            if (value != ' ' && value != '\t' && value != '\n' && value != '\r' && value != '\f') break;
            ++position_;
        }
    }

    double number() {
        skip_whitespace();
        if (position_ < data_.size() && data_[position_] == ',') {
            if (!number_precedes_) { throw std::invalid_argument("SVG path data contains a misplaced comma."); }
            ++position_;
            skip_whitespace();
            if (position_ == data_.size() || data_[position_] == ',') {
                throw std::invalid_argument("SVG path data contains a misplaced comma.");
            }
        }
        const std::size_t begin = position_;
        if (position_ < data_.size() && (data_[position_] == '+' || data_[position_] == '-')) ++position_;
        bool has_digits = false;
        while (position_ < data_.size() && is_digit(data_[position_])) {
            has_digits = true;
            ++position_;
        }
        if (position_ < data_.size() && data_[position_] == '.') {
            ++position_;
            while (position_ < data_.size() && is_digit(data_[position_])) {
                has_digits = true;
                ++position_;
            }
        }
        if (!has_digits) { throw std::invalid_argument("SVG path data contains an invalid number."); }
        if (position_ < data_.size() && (data_[position_] == 'e' || data_[position_] == 'E')) {
            ++position_;
            if (position_ < data_.size() && (data_[position_] == '+' || data_[position_] == '-')) ++position_;
            const std::size_t exponent = position_;
            while (position_ < data_.size() && is_digit(data_[position_])) ++position_;
            if (position_ == exponent) { throw std::invalid_argument("SVG path data contains an invalid exponent."); }
        }

        std::string_view lexeme = data_.substr(begin, position_ - begin);
        if (!lexeme.empty() && lexeme.front() == '+') lexeme.remove_prefix(1);
        double value = 0.0;
        const auto result = std::from_chars(lexeme.data(), lexeme.data() + lexeme.size(), value);
        if (result.ec != std::errc() || result.ptr != lexeme.data() + lexeme.size() || !std::isfinite(value)) {
            throw std::invalid_argument("SVG path data contains a non-finite or unrepresentable number.");
        }
        number_precedes_ = true;
        return value;
    }

    point_t point(bool relative) {
        point_t point {number(), number()};
        if (relative) {
            point.x += current_.x;
            point.y += current_.y;
        }
        if (!std::isfinite(point.x) || !std::isfinite(point.y)) {
            throw std::invalid_argument("SVG path arithmetic exceeded the finite coordinate range.");
        }
        return point;
    }

    void append(point_t point) {
        if (!current_ring_.empty() && equal(current_ring_.back(), point)) {
            throw std::invalid_argument("SVG subpath contains adjacent duplicate vertices.");
        }
        current_ring_.push_back(point);
        current_ = point;
    }

    void move_to(bool relative) {
        if (ring_open_) { throw std::invalid_argument("SVG subpaths must be explicitly closed before a new M."); }
        append(point(relative));
        start_ = current_;
        ring_open_ = true;
        command_ = relative ? 'l' : 'L';
    }

    void line_to(bool relative) {
        if (!ring_open_) { throw std::invalid_argument("SVG line command requires an open subpath."); }
        append(point(relative));
    }

    void cubic_to(bool relative) {
        if (!ring_open_) { throw std::invalid_argument("SVG cubic command requires an open subpath."); }
        const point_t origin = current_;
        const point_t control_one = point(relative);
        const point_t control_two = point(relative);
        const point_t end = point(relative);
        for (int step = 1; step <= subdivisions_; ++step) {
            const double t = static_cast<double>(step) / subdivisions_;
            const double s = 1.0 - t;
            const point_t sample {
              s * s * s * origin.x + 3.0 * s * s * t * control_one.x + 3.0 * s * t * t * control_two.x +
                t * t * t * end.x,
              s * s * s * origin.y + 3.0 * s * s * t * control_one.y + 3.0 * s * t * t * control_two.y +
                t * t * t * end.y};
            if (!std::isfinite(sample.x) || !std::isfinite(sample.y)) {
                throw std::invalid_argument("SVG cubic flattening exceeded the finite coordinate range.");
            }
            append(sample);
        }
        current_ = end;
    }

    void close_ring() {
        if (!ring_open_) { throw std::invalid_argument("SVG close command requires an open subpath."); }
        if (equal(current_ring_.back(), start_)) current_ring_.pop_back();
        if (current_ring_.size() < 3) {
            throw std::invalid_argument("SVG subpath needs at least three distinct vertices.");
        }
        std::set<std::array<double, 2>> vertices;
        for (const point_t& point : current_ring_) {
            if (!vertices.insert({point.x, point.y}).second) {
                throw std::invalid_argument("SVG subpath contains duplicate vertices.");
            }
        }
        rings_.push_back(std::move(current_ring_));
        current_ring_.clear();
        current_ = start_;
        ring_open_ = false;
    }

    std::string_view data_;
    int subdivisions_;
    std::size_t position_ = 0;
    char command_ = 0;
    bool number_precedes_ = false;
    bool ring_open_ = false;
    point_t current_ {0.0, 0.0};
    point_t start_ {0.0, 0.0};
    std::vector<point_t> current_ring_;
    std::vector<std::vector<point_t>> rings_;
};

using attribute_t = std::pair<std::string_view, std::string_view>;

struct tag_t {
    std::string_view name;
    std::vector<attribute_t> attributes;
    bool self_closing;
};

inline bool is_name_start(char value) {
    return (value >= 'A' && value <= 'Z') || (value >= 'a' && value <= 'z') || value == '_' || value == ':';
}

inline bool is_name_character(char value) {
    return is_name_start(value) || (value >= '0' && value <= '9') || value == '-' || value == '.';
}

inline void skip_whitespace(std::string_view text, std::size_t& position) {
    while (position < text.size()) {
        const char value = text[position];
        if (value != ' ' && value != '\t' && value != '\n' && value != '\r') break;
        ++position;
    }
}

inline std::string_view local_name(std::string_view name) {
    const std::size_t colon = name.rfind(':');
    return colon == std::string_view::npos ? name : name.substr(colon + 1);
}

inline tag_t parse_tag(std::string_view text) {
    std::size_t position = 0;
    skip_whitespace(text, position);
    if (position == text.size() || !is_name_start(text[position])) {
        throw std::invalid_argument("SVG contains a malformed element tag.");
    }
    const std::size_t name_begin = position++;
    while (position < text.size() && is_name_character(text[position])) ++position;
    const std::string_view name = text.substr(name_begin, position - name_begin);
    std::vector<attribute_t> attributes;
    bool self_closing = false;
    while (true) {
        skip_whitespace(text, position);
        if (position == text.size()) break;
        if (text[position] == '/') {
            self_closing = true;
            ++position;
            skip_whitespace(text, position);
            if (position != text.size()) { throw std::invalid_argument("SVG contains a malformed self-closing tag."); }
            break;
        }
        if (!is_name_start(text[position])) { throw std::invalid_argument("SVG contains a malformed attribute."); }
        const std::size_t attribute_begin = position++;
        while (position < text.size() && is_name_character(text[position])) ++position;
        const std::string_view attribute = text.substr(attribute_begin, position - attribute_begin);
        skip_whitespace(text, position);
        if (position == text.size() || text[position] != '=') {
            throw std::invalid_argument("SVG attributes must have quoted values.");
        }
        ++position;
        skip_whitespace(text, position);
        if (position == text.size() || (text[position] != '\'' && text[position] != '"')) {
            throw std::invalid_argument("SVG attributes must have quoted values.");
        }
        const char quote = text[position++];
        const std::size_t value_begin = position;
        const std::size_t value_end = text.find(quote, position);
        if (value_end == std::string_view::npos) {
            throw std::invalid_argument("SVG contains an unterminated attribute.");
        }
        const std::string_view value = text.substr(value_begin, value_end - value_begin);
        position = value_end + 1;
        if (std::any_of(attributes.begin(), attributes.end(), [&](const attribute_t& entry) {
                return entry.first == attribute;
            })) {
            throw std::invalid_argument("SVG element contains a duplicate attribute.");
        }
        attributes.emplace_back(attribute, value);
    }
    return {name, std::move(attributes), self_closing};
}

inline std::size_t tag_end(std::string_view document, std::size_t begin) {
    char quote = 0;
    for (std::size_t position = begin; position < document.size(); ++position) {
        const char value = document[position];
        if (quote != 0) {
            if (value == quote) quote = 0;
        } else if (value == '\'' || value == '"') {
            quote = value;
        } else if (value == '>') {
            return position;
        }
    }
    throw std::invalid_argument("SVG contains an unterminated element tag.");
}

inline std::optional<std::string_view> attribute(const std::vector<attribute_t>& attributes, std::string_view name) {
    std::optional<std::string_view> value;
    for (const attribute_t& entry : attributes) {
        if (local_name(entry.first) != name) continue;
        if (value.has_value()) { throw std::invalid_argument("SVG element contains a duplicate local attribute."); }
        value = entry.second;
    }
    return value;
}

inline bool whitespace_only(std::string_view text) {
    std::size_t position = 0;
    skip_whitespace(text, position);
    return position == text.size();
}

}   // namespace svg
}   // namespace internals

/**
 * @brief Parse explicitly closed polygonal and cubic path rings from an in-memory SVG document.
 *
 * The supported path commands are `M/m`, `L/l`, `C/c`, and `Z/z`; cubic curves are sampled uniformly at the requested
 * positive number of subdivisions. Multiple path elements and subpaths are returned separately in source order as
 * unclosed rings. SVG y coordinates are negated to produce Cartesian coordinates. Transforms, entities, DOCTYPE,
 * non-path geometry, open subpaths, and malformed or unsupported path data are rejected. `viewBox`, dimensions,
 * presentation attributes, CSS, and fill rules do not alter coordinates. The returned rings are not classified as
 * outer boundaries or holes; each independent simple ring can be passed to `constrained_delaunay`.
 */
inline std::vector<Matrix<double, Dynamic, Dynamic>>
svg_document_rings(std::string_view document, int cubic_subdivisions) {
    using namespace internals::svg;
    if (cubic_subdivisions <= 0) { throw std::invalid_argument("SVG cubic subdivisions must be positive."); }
    if (document.empty()) { throw std::invalid_argument("SVG document cannot be empty."); }
    if (document.find('&') != std::string_view::npos) {
        throw std::invalid_argument("SVG entities are not supported.");
    }

    bool has_svg = false;
    bool closed_svg = false;
    bool inside_svg = false;
    std::vector<std::string_view> element_stack;
    std::vector<Matrix<double, Dynamic, Dynamic>> result;
    std::size_t position = 0;
    while (true) {
        const std::size_t begin = document.find('<', position);
        if (begin == std::string_view::npos) {
            if (!inside_svg && !whitespace_only(document.substr(position))) {
                throw std::invalid_argument("SVG document contains text outside its root element.");
            }
            break;
        }
        if (!inside_svg && !whitespace_only(document.substr(position, begin - position))) {
            throw std::invalid_argument("SVG document contains text outside its root element.");
        }
        if (document.substr(begin, 4) == "<!--") {
            const std::size_t end = document.find("-->", begin + 4);
            if (end == std::string_view::npos) { throw std::invalid_argument("SVG contains an unterminated comment."); }
            position = end + 3;
            continue;
        }
        if (document.substr(begin, 2) == "<?") {
            const std::size_t end = document.find("?>", begin + 2);
            if (end == std::string_view::npos) {
                throw std::invalid_argument("SVG contains an unterminated processing instruction.");
            }
            position = end + 2;
            continue;
        }
        if (document.substr(begin, 2) == "<!") {
            throw std::invalid_argument("SVG declarations, entities, and CDATA are not supported.");
        }

        const std::size_t end = tag_end(document, begin + 1);
        std::string_view tag = document.substr(begin + 1, end - begin - 1);
        position = end + 1;
        std::size_t tag_position = 0;
        skip_whitespace(tag, tag_position);
        if (tag_position < tag.size() && tag[tag_position] == '/') {
            ++tag_position;
            skip_whitespace(tag, tag_position);
            if (tag_position == tag.size() || !is_name_start(tag[tag_position])) {
                throw std::invalid_argument("SVG contains a malformed closing tag.");
            }
            const std::size_t name_begin = tag_position++;
            while (tag_position < tag.size() && is_name_character(tag[tag_position])) ++tag_position;
            const std::string_view name = local_name(tag.substr(name_begin, tag_position - name_begin));
            skip_whitespace(tag, tag_position);
            if (tag_position != tag.size()) { throw std::invalid_argument("SVG contains a malformed closing tag."); }
            if (element_stack.empty() || element_stack.back() != name) {
                throw std::invalid_argument("SVG contains a mismatched closing tag.");
            }
            if (name == "svg") {
                inside_svg = false;
                closed_svg = true;
            }
            element_stack.pop_back();
            continue;
        }

        const tag_t parsed = parse_tag(tag);
        const std::string_view name = local_name(parsed.name);
        const auto& attributes = parsed.attributes;
        if (attribute(attributes, "transform").has_value()) {
            throw std::invalid_argument("SVG transforms are not supported.");
        }
        if (name == "svg") {
            if (has_svg || !element_stack.empty()) {
                throw std::invalid_argument("SVG document contains multiple or nested svg root elements.");
            }
            has_svg = true;
            inside_svg = true;
        } else if (!inside_svg) {
            throw std::invalid_argument("SVG document contains an element outside its svg root.");
        }
        if (
          name == "rect" || name == "circle" || name == "ellipse" || name == "line" || name == "polyline" ||
          name == "polygon" || name == "use") {
            throw std::invalid_argument("SVG contains unsupported non-path geometry.");
        }
        if (name == "path") {
            const auto data = attribute(attributes, "d");
            if (!data.has_value() || data->empty()) {
                throw std::invalid_argument("SVG path requires a nonempty d attribute.");
            }
            const auto rings = path_parser(*data, cubic_subdivisions).parse();
            for (const auto& ring : rings) {
                Matrix<double, Dynamic, Dynamic> matrix(ring.size(), 2);
                for (int i = 0; i < int(ring.size()); ++i) {
                    matrix(i, 0) = ring[i].x;
                    matrix(i, 1) = -ring[i].y;
                }
                result.push_back(std::move(matrix));
            }
        }
        if (!parsed.self_closing) {
            element_stack.push_back(name);
        } else if (name == "svg") {
            inside_svg = false;
            closed_svg = true;
        }
    }
    if (!element_stack.empty()) { throw std::invalid_argument("SVG document contains an unclosed element tag."); }
    if (!has_svg || !closed_svg) { throw std::invalid_argument("SVG document requires a closed svg root element."); }
    if (result.empty()) { throw std::invalid_argument("SVG document contains no supported path rings."); }
    return result;
}

}   // namespace fdapde

#endif   // __FDAPDE_GEOMETRY_SVG_H__
