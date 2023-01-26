// builds a node enumeration for the support of a basis of order R. This fills both the elements_ table
// and recompute the boundary informations. (support only for order 2 basis)
template <unsigned int M, unsigned int N, unsigned int R>
void Mesh<M,N,R>::DOFenumerate(const DMatrix<int>& boundary) {
  // algorithm initialization
  int next = numNodes_; // next valid ID to assign
  // map of already assigned IDs
  std::unordered_map<std::pair<int, int>, std::array<int, n_dof_per_edge>, fdaPDE::pair_hash> assigned;
  std::size_t col = 0; // current column of the elements_ table to change
  std::vector<bool> onBoundary(ct_nnodes(M,R)*numNodes_, false);
  
  // start enumeration
  for(std::size_t elem = 0; elem < elements_.rows(); ++elem){
    // consider all pairs of nodes (all edges)
    for(std::size_t i = 0; i < n_vertices; ++i){
      for(std::size_t j = i+1; j < n_vertices; ++j){
	// check if edge (elements_[i], elements_[j]) has an already assigned number
	std::pair<int, int> edge = std::minmax(elements_(elem,i), elements_(elem,j));
	auto it = assigned.find(edge);
	if(it != assigned.end()){ // there is an already assigned index
	  for(std::size_t z = 0; z < n_dof_per_edge; ++z, ++col){
	    elements_(elem, M+1+col) = it->second[z];
	  }
	}else{
	  for(std::size_t z = 0; z < n_dof_per_edge; ++z, ++col, ++next){
	    elements_(elem, M+1+col) = next;
	    assigned[edge][z] = next;
	    // new node is on boundary iff both endpoints of its edge are on boundary
	    if(boundary(edge.first,0) && boundary(edge.second,0)) onBoundary[next] = true;
	  }
	}
      }
    }
    // insert not-shared dofs if required (nodes internal to the current element)
    for(std::size_t i = 0; i < n_dof_internal; ++i, ++col, ++next){
      elements_(elem, M+1+col) = next;
    }
    col = 0; // reset column counter
  }
  dof_ = next; // store degrees of freedom
  // adjust boundary informations
  boundary_.resize(dof_,1);
  boundary_.topRows(boundary.rows()) = boundary;
  for(std::size_t i = numNodes_; i < dof_; ++i){ // adjust for new nodes of the enumeration
    if(onBoundary[i]) boundary_(i,0) = 1;
    else boundary_(i,0) = 0;
  }
  return;
}

// construct directly from raw eigen matrix (used from wrappers)
template <unsigned int M, unsigned int N, unsigned int R>
Mesh<M,N,R>::Mesh(const DMatrix<double>& points, const DMatrix<int>& edges, const DMatrix<int>& elements,
		  const typename neighboring_structure<M, N>::type& neighbors, const DMatrix<int>& boundary) :
  points_(points), neighbors_(neighbors) {
  // realign indexes (we assume index coming from mesh generator to be greater or equal to 1, C++ starts count from 0)
  if constexpr(!is_linear_network<M,N>::value)
    neighbors_ = (neighbors_.array() - 1).matrix();
  else
    neighbors_ = neighbors; // adjacency matrix is directly given as input as sparse matrix
  
  // compute dof_table
  elements_.resize(elements.rows(), ct_nnodes(M,R));
  elements_.leftCols(elements.cols()) = (elements.array() -1).matrix();
  // store number of nodes and number of elements
  numNodes_ = points_.rows();
  numElements_ = elements_.rows();
  if constexpr(R > 1) DOFenumerate(boundary);
  else{
    // for order 1 meshes the functional basis is built over the same vertices which define the mesh geometry, nothing to do
    // set boundary structure as coming from data and dof as number of mesh nodes
    boundary_ = boundary;
    dof_ = numNodes_;
  }
    
  // compute mesh limits
  for(size_t dim = 0; dim < N; ++dim){
    range_[dim].first  = points_.col(dim).minCoeff();
    range_[dim].second = points_.col(dim).maxCoeff();
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
  // in the following subtract 1 for index realignment
  
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
  }

  // compute dof_table
  elements_.resize(elementsData.rows(), ct_nnodes(M,R));
  elements_.leftCols(elementsData.cols()) = (elementsData.toEigen().array() - 1).matrix();
  // store number of elements
  numElements_ = elements_.rows();  
  if constexpr(R > 1) DOFenumerate(boundaryData.toEigen());
  else{
    // for order 1 meshes the functional basis is built over the same vertices which define the mesh geometry, nothing to do
    // set boundary structure as coming from data and dof as number of mesh nodes
    boundary_ = boundaryData.toEigen();
    dof_ = numNodes_;
  }
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
    // degrees of freedom associated with this element
    auto pointData = elements_.row(ID);
    auto neighboringData = neighbors_.row(ID); // neighboring structure
  
    // prepare element
    std::array<std::size_t, ct_nvertices(M)> nodeIDs{};
    std::array<SVector<N>,  ct_nvertices(M)> coords{};
    // number of neighbors may be not known at compile time in case linear network elements are employed, use a dynamic
    // data structure to handle 1.5D case as well transparently
    std::vector<int> neighbors{};
    // boundary informations, the element is on boundary <-> at least one node with ID pointData[i] is on boundary
    bool boundary = false;
    
    for(size_t i = 0; i < ct_nvertices(M); ++i){
      SVector<N> node(points_.row(pointData[i])); // coordinates of node
      coords[i]   = node;
      // global ID of the node in the mesh
      nodeIDs[i]  = pointData[i];
      boundary |= (boundary_(pointData[i]) == 1);

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

// provides the element abstraction given its ID (returns cached pointer to the element)
template <unsigned int M, unsigned int N, unsigned int R>
std::shared_ptr<Element<M,N,R>> Mesh<M,N,R>::element(unsigned int ID) const { return cache_[ID]; }

// extract from raw information the mesh node with given ID (ID starts from 0)
template <unsigned int M, unsigned int N, unsigned int R>
SVector<N> Mesh<M,N,R>::node(unsigned int ID) const { return points_.row(ID); }

// produce the matrix of dof coordinates
template <unsigned int M, unsigned int N, unsigned int R>
DMatrix<double> Mesh<M,N,R>::dofCoords() const {
  if constexpr (R == 1)
    return points_; // for order 1 meshes dofs coincide with vertices
  else {
    // allocate space
    DMatrix<double> coords;
    coords.resize(dof_, N);
    coords.topRows(numNodes_) = points_; // copy coordinates of elements' vertices
    std::unordered_set<std::size_t> visited; // set of already visited dofs
    // define reference element
    std::array<SVector<M+1>, ct_nnodes(M,R)> refCoords = ReferenceElement<M,R>().bary_coords;

    // cycle over all mesh elements
    for(std::size_t i = 0; i < elements_.rows(); ++i){
      // extract dofs related to element with ID i
      auto dofs = elements_.row(i);
      auto e = *cache_[i]; // take pointer to current physical element
      for(std::size_t j = ct_nvertices(M); j < ct_nnodes(M,R); ++j){ // cycle only on non-vertex points
	if(visited.find(dofs[j]) == visited.end()){ // not yet mapped dof
	  // map points from reference to physical element
	  coords.row(dofs[j]) = e.barycentricMatrix()*refCoords[j].template tail<M>() + e.coords()[0];
	  visited.insert(dofs[j]);
	}
      }
    }
    return coords;
  }
}
