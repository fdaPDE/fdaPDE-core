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

#ifndef __FDAPDE_CSV_H__
#define __FDAPDE_CSV_H__

#include "header_check.h"

namespace fdapde {
  
// parser for CSV, Comma Separated Values (RFC 4180 compliant)
template <typename T>
internals::table_reader<T> read_csv(const std::string& filename, bool header = true, bool index_col = true) {
    internals::table_reader<T> csv(filename.c_str(), header, ',', index_col, true, 4);
    return csv;
}

// writes container to csv file
template <typename T>
void write_csv(
  const std::string& filename, const T& data, int rows, const std::vector<std::string>& colnames, bool by_rows = true) {
    internals::table_writer<T> csv(filename, ",");
    csv.write(data, rows, colnames, by_rows);
    return;
}
template <typename T>
void write_csv(const std::string& filename, const T& data, int rows, int cols, bool by_rows = true) {
    return write_csv(filename, data, rows, seq("V", cols), by_rows);
}
template <typename T> void write_csv(const std::string& filename, const T& data, const std::string& colname) {
    return write_csv(filename, data, data.size(), std::vector<std::string> {colname});
}
template <typename T> void write_csv(const std::string& filename, const T& data) {
    return write_csv(filename, data, "V1");
}

}   // namespace fdapde

#endif // __FDAPDE_CSV_H__
