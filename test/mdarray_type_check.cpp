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

#include <fdaPDE/geoframe.h>

#include <cassert>
#include <sstream>

// verify that dependent all-dynamic extents retain their exact public types
using fdapde::Dynamic;
using fdapde::full_dynamic_extent_t;
using fdapde::MdExtents;
// the empty index sequence must preserve a zero-dimensional extent type
static_assert(std::is_same_v<full_dynamic_extent_t<0>, MdExtents<>>);
// one sequence index must yield one dynamic dimension
static_assert(std::is_same_v<full_dynamic_extent_t<1>, MdExtents<Dynamic>>);
// column storage must retain exactly two dynamic dimensions
static_assert(std::is_same_v<full_dynamic_extent_t<2>, MdExtents<Dynamic, Dynamic>>);
// higher-order blocks must retain all dimensions without a special case
static_assert(std::is_same_v<full_dynamic_extent_t<3>, MdExtents<Dynamic, Dynamic, Dynamic>>);

// exercise dependent column storage and missing-value masks through the GeoFrame header
int main() {
    fdapde::internals::scalar_data_layer data;
    data.append_vec("numeric", std::vector<double> {1.0, std::numeric_limits<double>::quiet_NaN(), 3.0});
    data.append_vec("text", std::vector<std::string> {"first", "NA", "last"});
    const auto& read_only = data;
    auto numeric_mask = read_only.col<double>("numeric").nan();
    // the numeric mask must identify only the deliberately inserted floating-point NaN
    assert(!numeric_mask(0, 0) && numeric_mask(1, 0) && !numeric_mask(2, 0));
    auto text_mask = read_only.col<std::string>("text").nan();
    // the text mask must identify only the documented missing-value encoding
    assert(!text_mask(0, 0) && text_mask(1, 0) && !text_mask(2, 0));
    data.col<double>("numeric")(0, 0) = 2.0;
    // a mutable view must update the same storage read by its const counterpart
    assert(read_only.col<double>("numeric")(0, 0) == 2.0);
    std::ostringstream output;
    output << read_only;
    // the formatter must render missing values from the column masks
    assert(output.str().find("NA") != std::string::npos);
}
