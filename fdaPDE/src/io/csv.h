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
    internals::table_reader<T> csv(
      filename.c_str(), header, /* sep = */ ',', index_col, /* skip_quote = */ true, /* chunksize = */ 100000);
    return csv;
}

#ifdef __FDAPDE_HAS_EIGEN__

template <typename DataT>
    requires(internals::is_eigen_dense_xpr_v<DataT>)
void write_csv(const std::string& filename, const DataT& data, const std::vector<std::string>& colnames) {
    fdapde_assert(data.cols() == colnames.size());
    const static Eigen::IOFormat CSVFormat(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
    std::ofstream file(filename);
    for (int i = 0; i < colnames.size() - 1; ++i) { file << colnames[i] << ", "; }
    file << colnames.back() << "\n";
    if (file.is_open()) {
        file << data.format(CSVFormat);
        file.close();
    }
    return;
}

template <typename DataT>
    requires(internals::is_eigen_dense_xpr_v<DataT>)
void write_csv(const std::string& filename, const DataT& data) {
    return write_csv(filename, data, seq("V", data.cols()));
}

#endif

template <typename DataT>
    requires(!internals::is_eigen_dense_xpr_v<DataT> && internals::is_vector_like_v<DataT>)
void write_csv(
  const std::string& filename, const DataT& data, int rows, int cols, const std::vector<std::string>& colnames,
  bool by_rows = true) {
    fdapde_assert(data.size() % (rows * cols) == 0 && cols == colnames.size());
    std::ofstream file(filename);
    for (int i = 0; i < colnames.size() - 1; ++i) { file << colnames[i] << ", "; }
    file << colnames.back() << "\n";

    int inner = by_rows ? cols : 1;
    int outer = by_rows ? 1 : cols;
    if(file.is_open()) {
        file << std::setprecision(16);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols - 1; ++j) { file << data[i * inner + j * outer] << ", "; }
            file << data[by_rows ? ((i + 1) * cols - 1) : (i + (cols - 1) * cols + 1)] << "\n";
        }
    }
    return;
}
template <typename DataT>
    requires(!internals::is_eigen_dense_xpr_v<DataT> && internals::is_vector_like_v<DataT>)
void write_csv(const std::string& filename, const DataT& data, int rows, int cols, bool by_rows = true) {
    return write_csv(filename, data, rows, cols, seq("V", cols), by_rows);
}
template <typename DataT>
    requires(!internals::is_eigen_dense_xpr_v<DataT> && internals::is_vector_like_v<DataT>)
void write_csv(
  const std::string& filename, const DataT& data, const std::vector<std::string>& colnames, bool by_rows = true) {
    return write_csv(filename, data, data.size(), 1, colnames, by_rows);
}
template <typename DataT>
    requires(!internals::is_eigen_dense_xpr_v<DataT> && internals::is_vector_like_v<DataT>)
void write_csv(const std::string& filename, const DataT& data, bool by_rows = true) {
    return write_csv(filename, data, data.size(), 1, seq("V", 1), by_rows);
}

}   // namespace fdapde

#endif // __FDAPDE_CSV_H__
