// construct a mesh from .csv files
template <unsigned int M, unsigned int N>
Mesh<M,N>::Mesh(const std::string& pointsFile,    const std::string& edgesFile, const std::string& trianglesFile,
		const std::string& neighborsFile, const std::string& boundaryMarkersFile){
  std::cout << pointsFile << std::endl;
  // open and parse CSV files
  CSVReader reader;
  CSVFile<double> points = reader.parseFile<double>(pointsFile);
  CSVFile<int> edges     = reader.parseFile<int>(edgesFile);
  CSVFile<int> triangles = reader.parseFile<int>(trianglesFile);
  
  // load neighboring informations
  typename std::conditional<
    !is_linear_network<M, N>::value, CSVFile<int>, CSVSparseFile<int>
    >::type neighbors;
  if constexpr(!is_linear_network<M,N>::value){
    neighbors = reader.parseFile<int>(neighborsFile);
    // move parsed file to eigen dense matrix, recall that a negative value means no neighbor
    neighbors_ = neighbors.toEigen();
    neighbors_ = (neighbors_.array() - 1).matrix();
  }else{
    // activate proper parsing to handle sparse matrix storage of neighboring information in case of linear network meshes.
    // in this case the csv stores a 2 column table where column i contains the list of indexes of neighboring elements attached
    // to node i of the linear element (a linear element has only 2 nodes)
    neighbors = reader.parseSparseFile<int>(neighborsFile);
    neighbors_ = neighbors.toEigen(); // .toEigen() of CSVSparseFile already subtract 1 to indexes for reaglignment
  }
  
  CSVFile<int> boundary  = reader.parseFile<int>(boundaryMarkersFile);
  
  // bring parsed informations to matrix-like structures
  points_  = points.toEigen();
  numNodes = points_.rows();
  // compute mesh limits
  for(size_t dim = 0; dim < N; ++dim){
    meshRange[dim].first  = points_.col(dim).minCoeff();
    meshRange[dim].second = points_.col(dim).maxCoeff();

    minMeshRange[dim] = meshRange[dim].first;
    kMeshRange[dim] = 1/(meshRange[dim].second - meshRange[dim].first);
  }
    
  // need to subtract 1 from all indexes since triangle indexes start from 1
  // C++ start counting from 0 instead
  edges_ = edges.toEigen();
  edges_ = (edges_.array() - 1).matrix();

  triangles_  = triangles.toEigen();
  triangles_  = (triangles_.array() -1).matrix();
  numElements = triangles_.rows();
   
  boundaryMarkers_ = boundary.toEigen();
  return;
}

// build and provides a nice abstraction for an element given its ID
template <unsigned int M, unsigned int N>
std::shared_ptr<Element<M,N>> Mesh<M,N>::requestElementById(unsigned int ID) const {
  // in the following use auto to take advantage of eigen acceleration
  // get the indexes of vertices from triangles_
  auto pointIndexes = triangles_.row(ID);
  // get neighbors information
  auto elementNeighbors = neighbors_.row(ID);
  
  // prepare element
  std::array<SVector<N>, N_VERTICES(M,N)> coords;
  std::array<std::pair<unsigned, SVector<N>>, N_VERTICES(M,N)> FEsupport;
  // number of neighbors may be not known at compile time in case linear network elements are employed, use a dynamic
  // data structure to handle 1.5D case as well transparently
  std::vector<int> neighbors;
  std::array<std::pair<unsigned, unsigned>, N_VERTICES(M,N)> boundaryMarkers;
  
  for(size_t i = 0; i < pointIndexes.size(); ++i){
    SVector<N> vertex(points_.row(pointIndexes[i]));
    coords[i] = vertex;
    FEsupport[i] = std::make_pair(pointIndexes[i], vertex);

    if constexpr(!is_linear_network<M, N>::value){
      // from triangle documentation: The first neighbor of triangle i is opposite the first corner of triangle i, and so on.
      // by storing neighboring informations as they come from triangle we have that neighbor[0] is the
      // triangle adjacent to the face opposite to coords[0]
      neighbors.push_back(elementNeighbors[i]);
    }
    // store boundary informations
    boundaryMarkers[i] = std::make_pair(pointIndexes[i], boundaryMarkers_(pointIndexes[i]));
  }
  // fill neighboring information for the linear network element case
  if constexpr(is_linear_network<M, N>::value){
    for(Eigen::SparseMatrix<int>::InnerIterator SpMat_it(neighbors_, ID); SpMat_it; ++SpMat_it){
      neighbors.push_back(SpMat_it.row()); // neighbors_ is stored in ColumnMajor mode
    }
  }
  // return shared pointer to the element
  return std::make_shared<Element<M,N>>(ID, FEsupport, coords, neighbors, boundaryMarkers);
}
