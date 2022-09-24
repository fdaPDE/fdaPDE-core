// split a string in a vector of substrings, using "separator" as delimiter
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

// remove from a given string s all occurencies of strings matching vect
std::string CSVReader::remove(std::string s, const std::vector<char>& vect) const {
  std::string result = "";
  for(char c : s){ 
    if (std::find(vect.begin(), vect.end(), c) == vect.end()) // char c is not to be removed
      result += c;
  }
  return result;
}

// remove from a given string s all chars j
std::string CSVReader::remove(std::string s, char j) const {
  std::string result = "";
  for(char c : s) if (c != j) result += c;
  return result;
}

// move a CSVFile into an Eigen::Matrix
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

// parse a file by casting its entries to type T
// Observe that the produced map can contain only entries of the specified type (not allowed to parse non-homogeneous files)
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
    // the following list removes all listed chars for each parsed line (chars here cause istringstream to
    // not cast correctly the string into type T, hence must be removed)
    std::vector<char> filter({' ', '"'});
    for(size_t j = 1; j < columnNames.size(); ++j){
      // apply filters to parsed string (remove unwanted chars)
      std::string filtered = remove(parsedLine[j], filter);
      // convert string to type T
      std::istringstream ss(filtered);
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

// move a CSVSparseFile into an Eigen::SparseMatrix
template <typename T>
Eigen::SparseMatrix<T> CSVSparseFile<T>::toEigen() {
  // return an adjacency matrix, hence we set as 1 the corresponding entry
  std::list<Eigen::Triplet<T>> tripletList;
  std::size_t n = parsedFile.at(this->columnNames_[0]).size(); // size of matrix

  for(auto col : parsedFile){
    for(std::size_t i = 0; i < n; ++i){
      for(T j : col.second[i]){
	if(j != std::numeric_limits<T>::quiet_NaN()){ 
	  // we push pair (i, j-1) since j indexes parsed from raw files starts from 1.
	  tripletList.push_front(Eigen::Triplet<T>(i, j-1, 1));
	}
      }
    }
  }
  // fill sparse matrix
  Eigen::SparseMatrix<T> result(n, n);
  result.setFromTriplets(tripletList.begin(), tripletList.end());
  result.makeCompressed();

  return result;
}

// parse a file which encodes some sparse structure. Usefull if you later want to move the .csv into an Eigen::SparseMatrix<T>
template <typename T>
CSVSparseFile<T> CSVReader::parseSparseFile(const std::string& file_){
  // create a CSVFile object
  CSVSparseFile<T> csv(file_);
  // get input stream
  std::ifstream& file = csv.getInputStream();

  // internal data structure of CSVFile
  CSVinternal<std::vector<T>> parsedFile;

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

    // fill data structure (start from j = 1 to skip first column containing row IDs)
    for(size_t j = 1; j < columnNames.size(); ++j){
      // perform proper parsing depending on token shape
      if(parsedLine[j].find("c(") != std::string::npos){
	// token is of kind c(id1, id2, ..., idn)
	std::string filteredString = remove(parsedLine[j], {'c', '(', ')'});
	// filteredString is now a comma separated list of indexes
	std::vector<std::string> parsedFilteredString = split(filteredString, ";");
	std::vector<T> data{};
	if(parsedFilteredString.empty()){
	  data.push_back(std::numeric_limits<T>::quiet_NaN()); // if list is empty (token of kind c()) signal to skip this row by inserting NaN
	}else{
	  for(std::string value : parsedFilteredString){
	    if(value != ""){ // skip possible empty substrings coming from filtering
	      std::istringstream ss(value);
	      T dataPoint;
	      ss >> dataPoint;
	      // insert value
	      data.push_back(dataPoint);
	    }
	  }
	}
	// insert value
	parsedFile[remove(columnNames[j], '\"')].push_back(data);
      }else{
	// token is an index range id1:id2
	std::vector<std::string> parsedIndexRange = split(parsedLine[j], ":");
	// move id1 and id2 to integers
	std::vector<int> indexRange{};
	for(std::string index : parsedIndexRange){
	  std::istringstream ss(index);
	  T idx;
	  ss >> idx;
	  // insert value
	  indexRange.push_back(idx);
	}
	std::vector<T> data{};
	for(int i = indexRange[0]; i <= indexRange[1]; ++i){
	  data.push_back(i);
	}
	// insert value
	parsedFile[remove(columnNames[j], '\"')].push_back(data);
      }
    }
  }
  // save result
  csv.setParsedFile(parsedFile);
  csv.closeFile();
  
  return csv;
}