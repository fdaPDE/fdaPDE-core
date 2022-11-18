// builds a node enumeration for the support of a basis of order R. This fills both the elements_ table
// and recompute the boundary informations. (support only for order 2 basis)
template <unsigned int M, unsigned int N, unsigned int R>
void Mesh<M,N,R>::compute_basis_support(const DMatrix<int>& boundary) {
  // algorithm initialization
  int next = numNodes_; // next valid ID to assign
  std::unordered_map<int, int> assigned; // map of already assigned IDs
  std::size_t col = 0; // column of the elements_ table to change
  std::vector<bool> onBoundary(numNodes_, false);
  
  // start enumeration
  for(std::size_t elem = 0; elem < elements_.rows(); ++elem){
    // consider all pairs of nodes (all edges)
    for(std::size_t i = 0; i < M+1; ++i){
      for(std::size_t j = i+1; j < M+1; ++j){
	// check if edge (elements_[i], elements_[j]) has an already assigned number
	std::pair<int, int> edge = std::minmax(elements_(elem,i), elements_(elem,j));
	
	// map edge to a unique index: (l,h) -> l*numNodes_ + h;
	int z = edge.first*numNodes_ + edge.second;
	auto it = assigned.find(z);
	if(it != assigned.end()){ // there is an already assigned index
	  elements_(elem, M+1+col) = it->second;
	  assigned.erase(it); // free space, an order 2 node cannot be shared more than 2 times!
	}else{
	  elements_(elem, M+1+col) = next;
	  assigned[z] = next;
	  // new node is on boundary iff both endpoints of its edge are on boundary
	  if(boundary(edge.first,0) && boundary(edge.second,0)) onBoundary.emplace_back(true);
	  next++; // increase next ID to assign
	}
	col++;
      }
    }
    col = 0; // reset column counter
  }
  
  // adjust boundary informations
  boundary_.resize(next,1);
  boundary_.topRows(boundary.rows()) = boundary;
  for(std::size_t i = numNodes_; i < next; ++i){
    if(onBoundary[i-numNodes_]) boundary_(i,0) = 1;
    else boundary_(i,0) = 0;
  }
  // store degrees of freedom
  dof_ = next;
  
  return;
}

// construct directly from raw eigen matrix (used from wrappers)
template <unsigned int M, unsigned int N, unsigned int R>
Mesh<M,N,R>::Mesh(const DMatrix<double>& points, const DMatrix<int>& edges, const DMatrix<int>& elements,
		  const typename neighboring_structure<M, N>::type& neighbors, const DMatrix<int>& boundary) :
  points_(points), neighbors_(neighbors) {
  // realign indexes (we assume index coming from mesh generator to be greater or equal to 1, C++ starts count from 0)
  neighbors_ = (neighbors_.array() - 1).matrix();

  // elements_ is used also to support the strucural information required for the definition of a finite element
  // basis of order R on this mesh. In particular the elements_ matrix has the following data layout
  // 
  //    | ----- M + 1 columns ----- | ---- ct_nnodes(M,R) - (M+1) columns ---- |
  //    |                           |                                          |
  //    |  elements' vertices. MESH |   extra nodes needed for the definition  |
  //    |    will mainly use this   |   of a finite elements basis of order R  |
  //    |    portion of the table   |                                          |
  //    | ------------------------- | ---------------------------------------- |
  //
  // keep in mind that nodes which are not vertices are not stored inside the mesh object but only an enumeration
  // of them coherent with the mesh topology is built. Nor Element objects produced by a call to element() will contain
  // such informations.
  elements_.resize(elements.rows(), ct_nnodes(M,R));
  elements_.leftCols(elements_.cols()) = (elements.array() -1).matrix();

  // store number of nodes and number of elements
  numNodes_ = points_.rows();
  numElements_ = elements_.rows();
  
  // for order 1 meshes the functional basis is built over the same vertices which define the mesh geometry, nothing to do
  if constexpr(R > 1) compute_basis_support(boundary);
  else boundary_ = boundary;
    
  // compute mesh limits
  for(size_t dim = 0; dim < N; ++dim){
    range_[dim].first  = points_.col(dim).minCoeff();
    range_[dim].second = points_.col(dim).maxCoeff();

    minRange_[dim] = range_[dim].first;
    kk_[dim] = 1/(range_[dim].second - range_[dim].first);
  }

  // scan the whole mesh and precompute here once all elements' abstractions for fast access
  fill_cache();
  // end of initialization
  return;
}

// construct a mesh from .csv files
template <unsigned int M, unsigned int N, unsigned int R>
Mesh<M,N,R>::Mesh(const std::string& points,    const std::string& edges, const std::string& elements,
		  const std::string& neighbors, const std::string& boundary){
  // open and parse CSV files
  CSVReader<double> Dreader; CSVReader<int> Ireader;
  CSVFile<double> pointsData = Dreader.parseFile(points);
  CSVFile<int> edgesData = Ireader.parseFile(edges);
  CSVFile<int> elementsData = Ireader.parseFile(elements);
  CSVFile<int> boundaryData = Ireader.parseFile(boundary);
  
  // load neighboring informations
  typename std::conditional<
    !is_linear_network<M, N>::value, CSVFile<int>, CSVSparseFile<int>
    >::type neighborsData;
  if constexpr(!is_linear_network<M,N>::value){
    neighborsData = Ireader.parseFile(neighbors);
    // move parsed file to eigen dense matrix, recall that a negative value means no neighbor
    neighbors_ = neighborsData.toEigen();
    neighbors_ = (neighbors_.array() - 1).matrix();
  }else{
    // activate proper parsing to handle sparse matrix storage of neighboring information in case of linear network meshes.
    // in this case the csv stores a 2 column table where column i contains the list of indexes of neighboring elements attached
    // to node i of the linear element (a linear element has only 2 nodes)
    neighborsData = Ireader.parseSparseFile(neighbors);
    neighbors_ = neighborsData.toEigen(); // .toEigen() of CSVSparseFile already subtract 1 to indexes for reaglignment
  }  
  
  // bring parsed informations to matrix-like structures
  points_   = pointsData.toEigen();
  numNodes_ = points_.rows();
  // compute mesh range
  for(size_t dim = 0; dim < N; ++dim){
    range_[dim].first  = points_.col(dim).minCoeff();
    range_[dim].second = points_.col(dim).maxCoeff();

    minRange_[dim] = range_[dim].first;
    kk_[dim] = 1/(range_[dim].second - range_[dim].first);
  }
    
  // need to subtract 1 from all indexes since triangle indexes start from 1
  // C++ start counting from 0 instead
  elements_.resize(elementsData.rows(), ct_nnodes(M,R));
  elements_.leftCols(elementsData.cols()) = (elementsData.toEigen().array() -1).matrix();
  numElements_ = elements_.rows();
  
  // for order 1 meshes the functional basis is built over the same vertices which define the mesh geometry, nothing to do
  if constexpr(R > 1) compute_basis_support(boundaryData.toEigen());
  else boundary_ = boundaryData.toEigen();
  
  // scan the whole mesh and precompute here once all elements' abstractions for fast access
  fill_cache();
  // end of initialization
  return;
}

// fill the cache_ data structure with pointers to element objects
template <unsigned int M, unsigned int N, unsigned int R>
void Mesh<M,N,R>::fill_cache() {
  // reserve space for cache
  cache_.resize(numElements_);
  
  // cycle over all possible elements' ID
  for(std::size_t ID = 0; ID < numElements_; ++ID){
    // in the following use auto to take advantage of eigen acceleration
    // get the indexes of vertices from triangles_
    auto pointData = elements_.row(ID);
    // get neighbors information
    auto neighboringData = neighbors_.row(ID);
  
    // prepare element
    std::array<std::size_t, ct_nvertices(M)> nodeIDs{};
    std::array<SVector<N>,  ct_nvertices(M)> coords{};
    // number of neighbors may be not known at compile time in case linear network elements are employed, use a dynamic
    // data structure to handle 1.5D case as well transparently
    std::vector<int> neighbors{};
    std::array<std::size_t, ct_nvertices(M)> boundary{};
  
    for(size_t i = 0; i < ct_nvertices(M); ++i){
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
    // cache constructed element
    cache_[ID] = std::make_shared<Element<M,N,R>>(ID, nodeIDs, coords, neighbors, boundary);
  }
}

// provides the element abstraction given its ID
template <unsigned int M, unsigned int N, unsigned int R>
std::shared_ptr<Element<M,N,R>> Mesh<M,N,R>::element(unsigned int ID) const {
  // return cached pointer to the element
  return cache_[ID];
}

// extract from raw information the mesh node with given ID (ID starts from 0)
template <unsigned int M, unsigned int N, unsigned int R>
SVector<N> Mesh<M,N,R>::node(unsigned int ID) const { return points_.row(ID); }
