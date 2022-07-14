#ifndef __CSV_READER_H__
#define __CSV_READER_H__

#include <cstddef>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>
#include <iostream>
#include <Eigen/Core>
#include <set>

template <typename T> using CSVinternal = std::unordered_map<std::string, std::vector<T>>;

template <typename T>
class CSVFile{
private:
  
  std::ifstream file;                     // read only stream
  std::vector<std::string> columnNames_;  // reference to column names for fast access

  CSVinternal<T> parsedFile;              // parsed file

public:
  CSVFile(const std::string& file_) {
    file.open(file_);
  }

  // output of CSVReader
  void setParsedFile(const CSVinternal<T>& parsedFile_) { parsedFile = parsedFile_; }
  // close input stream
  void closeFile(void) { file.close(); }
  // get input stream
  std::ifstream& getInputStream() { return file; }

  void addColumnName(std::string name_) { columnNames_.push_back(name_); }
  
  // an utility for converting parsed file into Eigen matrix
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> toEigen();
  // get raw parsed file
  CSVinternal<T> getRawParsedFile() { return parsedFile; }
};

template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> CSVFile<T>::toEigen() {
  // get number of columns
  unsigned int numColumns = parsedFile.size();
  // get number of rows (equal for every column)
  unsigned int numRows = parsedFile.at(columnNames_[0]).size();
  
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> result;
  result.resize(numRows, numColumns);
  
  // fill eigen matrix
  size_t i = 0;
  for(std::string key : columnNames_){
    for(size_t j = 0; j < parsedFile.at(key).size(); ++j){
      result(j,i) = parsedFile.at(key)[j];
    }
    ++i;
  }
  
  return result;
}

// a super simple class to parse and read simple .csv files
// (works with output of write.csv() from R)
class CSVReader{

 private:
  // split a string in a vector of strings according to a separator
  std::vector<std::string> split(std::string input_, std::string separator);
  // remove char j from string s
  std::string remove(std::string s, const char j);

 public:
  CSVReader() = default;

  // parse the file returning a CSVFile whose internal has a vector for each column in the .csv
  template <typename T> CSVFile<T> parseFile(const std::string& file_);
};

std::vector<std::string> CSVReader::split(std::string input_, std::string separator) {
  std::vector<std::string> parsedString;

  // keep string position
  size_t j = 0;
  while(j != std::string::npos){
    j = input_.find(separator);
    // store string in token list
    parsedString.push_back(input_.substr(0, j));
    input_.erase(0, j+separator.length());
  }
  
  return parsedString;
}

std::string CSVReader::remove(std::string s, const char j) {
  std::string result = "";
  for(char c : s){ 
    if (c != j) result += c;
  }
  return result;
}

// parse a file by converting its entries according to type T (commonly int,
// double). Observe that the produced map will contain only entries of the
// specified type
template <typename T>
CSVFile<T> CSVReader::parseFile(const std::string& file_){
  // create a CSVFile object
  CSVFile<T> csv(file_);
  // get input stream
  std::ifstream& file = csv.getInputStream();

  // internal data structure of CSVFile
  CSVinternal<T> parsedFile;
  // set column by reading first line
  std::string line;
  getline(file, line); // read line from stream
  std::vector<std::string> columnNames = split(line, ",");
  
  // set header
  for(size_t j = 1; j < columnNames.size(); ++j){
    std::string c_name = remove(columnNames[j], '\"');
    csv.addColumnName(c_name);
    parsedFile[c_name]; // value initialize the column
  }
  
  // read file until end line by line
  while(getline(file, line)){
    // split CSV line in tokens
    std::vector<std::string> parsedLine = split(line, ",");

    // fill data structure
    for(size_t j = 1; j < columnNames.size(); ++j){
      // convert string to type T
      std::istringstream ss(parsedLine[j]);
      T dataPoint;
      ss >> dataPoint;
      // insert value
      parsedFile[remove(columnNames[j], '\"')].push_back(dataPoint);
    }
  }

  // save result
  csv.setParsedFile(parsedFile);
  csv.closeFile();
  
  return csv;
}

#endif // __CSV_READER_H__
