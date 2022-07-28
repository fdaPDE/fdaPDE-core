// construct a mesh from .csv files
template <unsigned int M, unsigned int N, unsigned int R>
Mesh<M,N,R>::Mesh(const std::string& points,    const std::string& edges, const std::string& elements,
		  const std::string& neighbors, const std::string& boundary){
  // open and parse CSV files
  CSVReader reader;
  CSVFile<double> pointsData = reader.parseFile<double>(points);
  CSVFile<int> edgesData = reader.parseFile<int>(edges);
  CSVFile<int> elementsData = reader.parseFile<int>(elements);
  CSVFile<int> boundaryData = reader.parseFile<int>(boundary);
  
  // load neighboring informations
  typename std::conditional<
    !is_linear_network<M, N>::value, CSVFile<int>, CSVSparseFile<int>
    >::type neighborsData;
  if constexpr(!is_linear_network<M,N>::value){
    neighborsData = reader.parseFile<int>(neighbors);
    // move parsed file to eigen dense matrix, recall that a negative value means no neighbor
    neighbors_ = neighborsData.toEigen();
    neighbors_ = (neighbors_.array() - 1).matrix();
  }else{
    // activate proper parsing to handle sparse matrix storage of neighboring information in case of linear network meshes.
    // in this case the csv stores a 2 column table where column i contains the list of indexes of neighboring elements attached
    // to node i of the linear element (a linear element has only 2 nodes)
    neighborsData = reader.parseSparseFile<int>(neighbors);
    neighbors_ = neighborsData.toEigen(); // .toEigen() of CSVSparseFile already subtract 1 to indexes for reaglignment
  }  
  
  // bring parsed informations to matrix-like structures
  points_   = pointsData.toEigen();
  numNodes_ = points_.rows();
  // compute mesh limits
  for(size_t dim = 0; dim < N; ++dim){
    range_[dim].first  = points_.col(dim).minCoeff();
    range_[dim].second = points_.col(dim).maxCoeff();

    minRange_[dim] = range_[dim].first;
    kk_[dim] = 1/(range_[dim].second - range_[dim].first);
  }
    
  // need to subtract 1 from all indexes since triangle indexes start from 1
  // C++ start counting from 0 instead
  edges_ = edgesData.toEigen();
  edges_ = (edges_.array() - 1).matrix();

  elements_    = elementsData.toEigen();
  elements_    = (elements_.array() -1).matrix();
  numElements_ = elements_.rows();
   
  boundary_ = boundaryData.toEigen();
  return;
}

// build and provides a nice abstraction for an element given its ID
template <unsigned int M, unsigned int N, unsigned int R>
std::shared_ptr<Element<M,N,R>> Mesh<M,N,R>::element(unsigned int ID) const {
  // in the following use auto to take advantage of eigen acceleration
  // get the indexes of vertices from triangles_
  auto pointData = elements_.row(ID);
  // get neighbors information
  auto neighboringData = neighbors_.row(ID);
  
  // prepare element
  std::array<std::size_t, ct_nnodes(M,R)> nodeIDs{};
  std::array<SVector<N>,  ct_nnodes(M,R)> coords{};
  // number of neighbors may be not known at compile time in case linear network elements are employed, use a dynamic
  // data structure to handle 1.5D case as well transparently
  std::vector<int> neighbors{};
  std::array<std::size_t, ct_nnodes(M,R)> boundary{};
  
  for(size_t i = 0; i < pointData.size(); ++i){
    SVector<N> node(points_.row(pointData[i])); // coordinates of node
    coords[i]   = node;
    // global ID of the node in the mesh
    nodeIDs[i]  = pointData[i];
    // boundary informations, boundary[i] == 1 <-> node with ID pointData[i] is on boundary
    boundary[i] = boundary_(pointData[i]);

    if constexpr(!is_linear_network<M, N>::value){
      // from triangle documentation: The first neighbor of triangle i is opposite the first corner of triangle i, and so on.
      // by storing neighboring informations as they come from triangle we have that neighbor[0] is the
      // triangle adjacent to the face opposite to coords[0]. This is true for any mesh different from a network mesh
      neighbors.push_back(neighboringData[i]);
    }
  }
  // fill neighboring information for the linear network element case
  if constexpr(is_linear_network<M, N>::value){
    for(Eigen::SparseMatrix<int>::InnerIterator SpMat_it(neighbors_, ID); SpMat_it; ++SpMat_it){
      neighbors.push_back(SpMat_it.row()); // neighbors_ is stored in ColumnMajor mode
    }
  }
  // return shared pointer to the element
  return std::make_shared<Element<M,N>>(ID, nodeIDs, coords, neighbors, boundary);
}
