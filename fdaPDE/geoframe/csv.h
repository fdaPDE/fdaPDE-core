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

#ifndef __CSV_H__
#define __CSV_H__

#include <limits>
#include <vector>

#include "batched_istream.h"
#include "parsing.h"

namespace fdapde {
  
// parser for CSV, Comma Separated Values (RFC 4180 compliant)
template <typename T> class CSVFile {
   private:
    template <typename CharBuff>
        requires(internals::is_char_buff<CharBuff>)
    T parse_value_(const CharBuff& token) const {
        // check if token is recognized as na
        if (std::find(na_values_.begin(), na_values_.end(), token) != na_values_.end()) {
            return std::numeric_limits<T>::quiet_NaN();
        }
	// parse token as numeric
        if constexpr (std::is_same_v<T, double>) { return internals::stod(token); }
        if constexpr (std::is_same_v<T, int>) { return internals::stoi(token); }
        return T {};
    }

    std::string_view& skipquote_(bool skip_quote, std::string_view& token) const {
        if (skip_quote) { [[likely]]
            if (!token.empty() && token.front() == '"') token.remove_prefix(1);
            if (!token.empty() && token.back()  == '"') token.remove_suffix(1);
        }
        return token;
    }
    // parsed data
    std::vector<T> data_ {};
    std::size_t n_cols_ = 0, n_rows_ = 0;
    std::vector<std::string> colnames_ {};

    std::vector<std::string> na_values_ = {"NA", "NaN", "nan"};
   public:
    CSVFile() = default;
    CSVFile(
      const char* filename, bool header, char sep, bool index_col, bool skip_quote = true, std::size_t chunksize = 4) :
        n_cols_(0), n_rows_(0), colnames_() {
        parse(filename, header, sep, index_col, skip_quote, chunksize);
    }
    CSVFile(const char* filename, bool index_col, bool skip_quote = true, std::size_t chunksize = 4) :
        CSVFile(filename, true, ',', index_col, skip_quote, chunksize) { }
    CSVFile(const std::string& filename, bool index_col, bool skip_quote = true, std::size_t chunksize = 4) :
        CSVFile(filename.c_str(), index_col, skip_quote, chunksize) { }

    Eigen::Map<const DMatrix<T, Eigen::RowMajor>> as_matrix() const {
        return Eigen::Map<const DMatrix<T, Eigen::RowMajor>>(data_.data(), n_rows_, n_cols_);
    }
    // modifiers
    void set_na_values(const std::vector<std::string>& na_values) { na_values_ = na_values; }
    std::size_t cols() const { return n_cols_; }
    std::size_t rows() const { return n_rows_; }
    const std::vector<T>& data() const { return data_; }
    const std::vector<std::string>& colnames() const { return colnames_; }
    // parsing function
    void parse(
      const char* filename, bool header = true, char sep = ',', bool index_col = false, bool skip_quote = true,
      std::size_t chunksize = 4) {
        if (!std::filesystem::exists(filename))
            throw std::runtime_error("file " + std::string(filename) + " not found.");
        auto stream = batched_istream(filename, chunksize); 
        bool header_ = header;
	std::size_t col_id = 0;
        std::string last_token;

        while (stream) {
            stream.read();
            const char* buff = stream.data();
            // tokenize input stream
            internals::token_stream token_stream_(buff, stream.size(), sep);

	    // PS: bug when file doesn't fit in chunksize
	    
            while (token_stream_) {
                auto line = token_stream_.get_line();
                if (header_) { [[unlikely]]   // header parsing logic
                    header_ = false;
                    while (line.has_token()) {
                        std::string_view& token = skipquote_(skip_quote, line.get_token());
                        if (index_col == false && n_cols_ != 0) { colnames_.push_back(std::string(token)); }
                        n_cols_++;
                        ++line;
                    }
                } else {   // data parsing logic
                    while (line.has_token()) {
                        if (index_col == false && col_id == 0) {   // skip first column
                        } else {
                            std::string_view& token = skipquote_(skip_quote, line.get_token());
                            if (line.eof()) {   // skip parsing and wait for next block
                                last_token = token;
                            } else {
                                if (!last_token.empty()) {
                                    last_token = last_token + std::string(token);   // merge tokens
                                    data_.push_back(parse_value_(last_token));
                                    last_token.clear();
                                } else if (!token.empty()) {
                                    data_.push_back(parse_value_(token));
                                }
                            }
                        }
                        if (!line.eof() ) { col_id = (col_id + 1) % n_cols_; }
                        ++line;
                    }
                }
            }
        }
        // process evantual last token of the last block of the stream
        if (!last_token.empty()) { data_.push_back(parse_value_(last_token)); }
        if (index_col == false) n_cols_ = n_cols_ - 1;		
        if (data_.size() % n_cols_ != 0) throw std::invalid_argument("csv parsing error.");
        n_rows_ = data_.size() / n_cols_;
        return;
    }
};

  // bug with index_col == true
template <typename T> CSVFile<T> read_csv(const std::string& filename, bool header = true, bool index_col = false) {
    CSVFile<T> csv(filename.c_str(), header, /* sep = */ ',', index_col, /* skip_quote = */ true, /* chunksize = */ 1000);
    return csv;
}

}   // namespace fdapde

#endif // __CSV_H__
