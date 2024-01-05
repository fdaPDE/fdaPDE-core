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

#ifndef __CSV_READER_H__
#define __CSV_READER_H__

#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace fdapde{
namespace core{

// a simpler parser for .csv files of type T
template <typename T>
class CSVReader{
 private:
  std::unordered_map<std::string, T> reserved_tokens_ {
    {"NA",  std::numeric_limits<T>::quiet_NaN()},
    {"NaN", std::numeric_limits<T>::quiet_NaN()},
    {"nan", std::numeric_limits<T>::quiet_NaN()}
  };
  std::vector<char> filter_ = {' ', '"'}; // toekns removed from each line
  
  // split s in a vector of strings according to sep
  std::vector<std::string> split_string(std::string s, const std::string& sep) const {
      std::vector<std::string> splitted_string;
      // keep string position
      size_t j = 0;
      while(j != std::string::npos){
	j = s.find(sep);
	// store string in token list
	splitted_string.push_back(s.substr(0, j));
	s.erase(0, j+sep.length());
      }    
      return splitted_string;
  }
  // remove any char in c_vect from s
  std::string remove_char(const std::string& s, const std::vector<char>& c_vect) const {
    std::string result = "";
    for(char c : s){ 
      if (std::find(c_vect.begin(), c_vect.end(), c) == c_vect.end()) { // char c is not to be removed
	result += c;
      }
    }
    return result;
  }
  // remove any occurence of j from s
  std::string remove_char(const std::string& s, char j) const {
    std::string result = "";
    for(char c : s) if (c != j) result += c;
    return result;
  }
  
 public:
  CSVReader() = default;

  // parse files with a dense structure
  template <typename U>
  typename std::enable_if<std::is_same<U, Eigen::Dense>::value, DMatrix<T>>::type
  parse_file(const std::string& file) const {
    // open file
    std::ifstream file_stream;
    file_stream.open(file);
    // deduce size of resulting matrix
    std::string line;
    getline(file_stream, line); // read first line
    int cols = split_string(line, ",").size() - 1; // first column is row index column
    int rows = 0;
    while(getline(file_stream, line)) { rows++; }

    // reserve space
    DMatrix<T> parsed_file;
    parsed_file.resize(rows, cols);
  
    // parse
    file_stream.clear();
    file_stream.seekg(0); // reset stream
    getline(file_stream, line); // skip first line
    // read file until EOF
    int row = 0;
    T val;
    while(getline(file_stream, line)){
      // split CSV line in tokens
      std::vector<std::string> parsed_line = split_string(line, ",");
      for(int col = 1; col < parsed_line.size(); ++col){ // skip first column (row index column)
	std::string data_token = remove_char(parsed_line[col], filter_);
	// detect token type
	auto token_type = reserved_tokens_.find(data_token);
	if(token_type != reserved_tokens_.end()) { // reserved token found
	  parsed_file(row, col-1) = token_type->second;
	} else {
	  std::istringstream ss(data_token);
	  ss >> parsed_file(row, col-1);
	}
      }
      row++;
    }
    // close file and return
    file_stream.close();
    return parsed_file;
  }

  // parse files with a 3-column (row,col,value) sparse format structure
  template <typename U>
  typename std::enable_if<std::is_same<U, Eigen::Sparse>::value, SpMatrix<T>>::type
  parse_file(const std::string& file) const {
    // open file
    std::ifstream file_stream;
    file_stream.open(file);
    // check if file is in 3-column format
    std::string line;
    getline(file_stream, line); // read first line
    if(split_string(line, ",").size() - 1 != 3) {
      throw std::runtime_error(".csv file not in sparse 3-column format");
    }
    // prepare triplet list
    std::vector<Eigen::Triplet<T>> triplet_list;
    // triplets data structures
    Eigen::Index row = 0, col = 0, n_row = 0, n_col = 0;
    T val{};
    // read file until EOF
    while(getline(file_stream, line)){
      // split CSV line in tokens
      std::vector<std::string> parsed_line = split_string(line, ",");
      // get row and column indexes
      std::istringstream ss_row(remove_char(parsed_line[1], filter_));
      std::istringstream ss_col(remove_char(parsed_line[2], filter_));
      ss_row >> row;
      n_row = (n_row < row) ? row : n_row;
      ss_col >> col;
      n_col = (n_col < col) ? col : n_col;
      // parse value token
      std::string value_token = remove_char(parsed_line[3], filter_);
      auto token_type = reserved_tokens_.find(value_token);
      if(token_type != reserved_tokens_.end()) { // reserved token found
	val = token_type->second;
      } else {
	std::istringstream ss_val(value_token);
	ss_val >> val;
      }    
      // store triplet
      triplet_list.emplace_back(row-1, col-1, val);
    }
    // assemble sparse matrix and return
    SpMatrix<T> parsed_file(n_row, n_col);
    parsed_file.setFromTriplets(triplet_list.begin(), triplet_list.end());
    parsed_file.makeCompressed();    
    file_stream.close();
    return parsed_file;
  }
};

}}
  
#endif // __CSV_READER_H__
