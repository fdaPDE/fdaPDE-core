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

#ifndef __FDAPDE_IO_MODULE_H__
#define __FDAPDE_IO_MODULE_H__

// clang-format off

#include <charconv>
#include <filesystem>
#include <fstream>

// basic parsing
#include "src/io/batched_istream.h"
#include "src/io/parsing.h"
#include "src/io/csv.h"
#include "src/io/txt.h"

// geometry dependent parsing
#include "geometry.h"    // pull geometry module first
#include "src/io/shp.h"

// clang-format on

namespace fdapde {

template <typename T>
internals::table_reader<T> read_table(const std::string& filename, bool header = true, bool index_col = true) {
    std::filesystem::path filepath(filename);
    if (!std::filesystem::exists(filepath)) { throw std::runtime_error("File " + filename + " not found."); }

    // dispatch to table_reader depending on extension
    std::string ext = filepath.extension();
    if (ext == ".csv") { return read_csv<T>(filename, header, index_col); }
    if (ext == ".txt") { return read_txt<T>(filename, header, index_col); }
    throw std::runtime_error("Unable to parse file " + filename + ": unsupported format.");
}

}   // namespace fdapde

#endif   // __FDAPDE_IO_MODULE_H__
