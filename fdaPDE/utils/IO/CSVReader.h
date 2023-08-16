#ifndef __CSV_READER_H__
#define __CSV_READER_H__

#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Sparse>

template <typename T> using CSVinternal = std::unordered_map<std::string, std::vector<T>>;

// class encoding a parsed csv file
template <typename T>
class CSVFile{
private:
  CSVinternal<T> parsedFile; // parsed file
protected:
  std::ifstream file;                     // read only stream
  std::vector<std::string> columnNames_;  // reference to column names for fast access
  
public:
  CSVFile() = default;
  CSVFile(const std::string& file_) { file.open(file_); }
  
  // output of CSVReader
  void setParsedFile(const CSVinternal<T>& parsedFile_) { parsedFile = parsedFile_; }
  // open and close input stream
  void openFile(const std::string& file_) { file.open(file_); }
  void closeFile(void) { file.close(); }
  // get input stream
  std::ifstream& getInputStream() { return file; }

  // add new column to parsed csv file
  void addColumnName(std::string name_) { columnNames_.push_back(name_); }
  
  // converts the raw parsed file in eigen dense matrix
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> toEigen();
  // get raw parsed file
  CSVinternal<T> getRawParsedFile() const { return parsedFile; }
  // get j-th row of file (packed as a std::vector)
  std::vector<T> row(std::size_t idx) const {
    std::vector<T> result;
    // cycle over parsed information
    for(auto& it : columnNames_){
      result.push_back(parsedFile.at(it)[idx]);
    }
    return result;
  }
  
  std::size_t rows() const { return parsedFile.at(columnNames_[0]).size(); }
  std::size_t cols() const { return parsedFile.size(); }
};

template <typename T>
class CSVSparseFile : public CSVFile<T>{
private:
  CSVinternal<std::vector<T>> parsedFile;
public:
  // constructor
  CSVSparseFile() = default;
  CSVSparseFile(const std::string& file_) : CSVFile<T>(file_) {};
  
  void setParsedFile(const CSVinternal<std::vector<T>>& parsedFile_) { parsedFile = parsedFile_; }
  // converts the raw parsed file in eigen dense matrix
  Eigen::SparseMatrix<T> toEigen();
  // get raw parsed file
  CSVinternal<std::vector<T>> getRawParsedFile() const { return parsedFile; }
  // get j-th row of file (packed as a std::vector)
  std::vector<T> row(std::size_t idx) const {
    std::vector<T> result;
    // cycle over parsed information
    for(auto& it : this->columnNames_){
      result.insert(result.begin(), parsedFile.at(it)[idx].begin(), parsedFile.at(it)[idx].end());
    }
    return result;
  }

};

// a simple utility class to parse simple .csv files of type T
template <typename T>
class CSVReader{
 private:
  // split a string in a vector of strings according to a separator
  std::vector<std::string> split(std::string input_, std::string separator);
  // remove any char in the supplied vector from string s
  std::string remove(std::string s, const std::vector<char>& vect) const;
  // remove a single char from string s
  std::string remove(std::string s, char j) const;
  
 public:
  CSVReader() = default;

  // parse the file returning a CSVFile whose internal has a vector for each column in the .csv
  CSVFile<T> parseFile(const std::string& file_);
  CSVSparseFile<T> parseSparseFile(const std::string& file_);
};

#include "CSVReader.tpp"

// utility functions to import .csv into Eigen dynamic matrices without the burden of defining CSV objects
template <typename T>
DMatrix<T> read_csv(const std::string& file_name){
  CSVReader<T> reader{};
  CSVFile<T> parsed_file;
  parsed_file = reader.parseFile(file_name);
  return parsed_file.toEigen();
}

#endif // __CSV_READER_H__
