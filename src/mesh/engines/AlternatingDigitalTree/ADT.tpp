// constructor
template <unsigned int M, unsigned int N, unsigned int R>
ADT<M,N,R>::ADT(const Mesh<M,N,R>& mesh) : mesh_(mesh){
  // move mesh elements to 2N dimensional points
  std::vector<std::pair<SVector<2*N>, unsigned int>> data;
  data.reserve(mesh_.elements()); // avoid useless reallocations at runtime
  // computation of normalization constants
  for(std::size_t dim = 0; dim < N; ++dim){
    normalization_[dim] = 1.0/(mesh_.range()[dim].second - mesh_.range()[dim].first);
  }
  
  for(const auto& element : mesh_){
    // compute bounding box
    std::pair<SVector<N>, SVector<N>> boundingBox = element->boundingBox();
    // create 2N dimensional point
    SVector<2*N> elementToPoint;
      
    // scale dimensions in the unit hypercube
    // point scaling means to apply the following linear transformation to each dimension of the point
    // scaledPoint[dim] = (point[dim] - meshRange[dim].first)/(meshRange[dim].second - meshRange[dim].first)
    for(size_t dim = 0; dim < N; ++dim){
      boundingBox.first[dim]  = (boundingBox.first[dim]  - mesh_.range()[dim].first)*normalization_[dim];
      boundingBox.second[dim] = (boundingBox.second[dim] - mesh_.range()[dim].first)*normalization_[dim];
    }
    elementToPoint << boundingBox.first, boundingBox.second;
    data.push_back(std::make_pair(elementToPoint, element->ID()));
  }
  // set up internal data structure
  init(data);
}
  
// data needs not to be rescaled before calling this method. Rescaling of the points in the unit hypercube is handled
// here as part of the ADT construction
// initializes the ADT
template <unsigned int M, unsigned int N, unsigned int R>
void ADT<M,N,R>::init(const std::vector<std::pair<SVector<2*N>, unsigned int>>& data) {
  // initialization
  SVector<2*N> left_lower_corner = SVector<2*N>::Zero(), right_upper_corner = SVector<2*N>::Ones();
  tree = Tree(ADTnode<2*N>(data[0].second, data[0].first, std::make_pair(left_lower_corner, right_upper_corner)));
  
  // process all points inside data one by one and insert them in the correct position
  for(size_t j = 1; j < data.size(); ++j){
    SVector<2*N> nodeData    = data[j].first;
    unsigned int nodeID      = data[j].second;
    rectangle<2*N> nodeRange = std::make_pair(left_lower_corner, right_upper_corner);

    // traverse the tree based on coordinates of data and insert the corresponding node at right position in the tree
    node_ptr<ADTnode<2*N>> current = tree.getNode(0); // root node
    
    bool inserted = false;           // stop iterating when an insertion point has been found
    unsigned int iteration = 1;      // defines the granularity of the split. split points are located at (0.5)^iteration
    std::array<double,2*N> offset{}; // keep track of the splits of the domain at each iteration
  
    // search for the right insertion location in the tree
    while(!inserted){
      for(size_t dim = 0; dim < 2*N; ++dim){     // cycle over dimensions
	double split_point = offset[dim] + std::pow(0.5, iteration); // split point
	if(nodeData[dim] < split_point){
      	  nodeRange.second[dim] = split_point;   // shrink node range on the left
	  if(tree.insert(ADTnode<2*N>(nodeID, nodeData, nodeRange), current->getKey(), LinkDirection::LEFT ) != nullptr){ // O(1) operation
	    inserted = true;                     // stop searching for location
	    break;
	  }else
	    current = current->getChildren()[0]; // move to left child
	}
	else{
	  nodeRange.first[dim] = split_point;    // shrink node range on the right
	  if(tree.insert(ADTnode<2*N>(nodeID, nodeData, nodeRange), current->getKey(), LinkDirection::RIGHT) != nullptr){ // O(1) operation
	    inserted = true;                     // stop searching for location
	    break;
	  }else{
	    current = current->getChildren()[1]; // move to right child
	    offset[dim] += std::pow(0.5, iteration);
	  }
	}
      }
      // virtually perform an half split of the hyper-cube
      iteration++;
    }
  }
  // construction ended
  return;
}

// a searching range (here called query) is supplied as a pair of points (a,b) where a is the
// lower-left corner and b the upper-right corner of the query rectangle. This method find all the points
// which are contained in a given query
template <unsigned int M, unsigned int N, unsigned int R>
std::list<unsigned int> ADT<M,N,R>::geometricSearch(const Query<2*N> &query) const {
  std::list<unsigned int> searchResult;
  // use a stack to assist the searching process
  std::stack< node_ptr<ADTnode<2*N>> > stack;
  stack.push(tree.getNode(0)); // start from root
  
  while(!stack.empty()){
    node_ptr<ADTnode<2*N>> current = stack.top();
    stack.pop();
    // add to solution if point is contained in query range
    if(query.contains(current->getData().point_))
      searchResult.push_back(current->getData().elementID_);
    // get children at node
    std::array<node_ptr<ADTnode<2*N>>, 2> children = current->getChildren();
    
    bool left_child_test  = children[0] != nullptr ? query.intersect(children[0]->getData().range_) : false;
    bool right_child_test = children[1] != nullptr ? query.intersect(children[1]->getData().range_) : false;
    
    if(left_child_test)        // test if left  child range intersects query range
      stack.push(children[0]); 
    if(right_child_test)       // test if right child range intersects query range
      stack.push(children[1]); 
  }
  // search completed
  return searchResult;
}

// once mesh elements are mapped as points in a 2N dimensional space, the problem of searching for the
// element containing a given point can be solved as a geometric search problem in a 2N dimensional space
template <unsigned int M, unsigned int N, unsigned int R>
template <typename... Args>
std::shared_ptr<Element<M,N,R>> ADT<M,N,R>::search(const SVector<N> &point, Args&... args) const {  
  // point scaling means to apply the following linear transformation to each dimension of the point
  // scaledPoint[dim] = (point[dim] - meshRange[dim].first)/(meshRange[dim].second - meshRange[dim].first)
  SVector<N> scaledPoint;
  for(size_t dim = 0; dim < N; ++dim){
    scaledPoint[dim] = (point[dim] - mesh_.range()[dim].first)*normalization_[dim];
  }
  // build search query
  SVector<2*N> lower, upper;
  lower << SVector<N>::Zero(), scaledPoint;
  upper << scaledPoint, SVector<N>::Ones();
  rectangle<2*N> query = std::make_pair(lower,upper);
    
  // perform search (now the problem has been transformed to the one of searching for the set
  // of points contained in the range of the query. See "(J. Bonet, J. Peraire) 1991
  // An alternating digital tree (ADT) algorithm for 3D geometric searching and intersection problems"
  // for details)
  std::list<unsigned int> searchResult = geometricSearch(Query<2*N>(query));
  // exhaustively scan the query results to get the searched mesh element
  for(unsigned int ID : searchResult){
    std::shared_ptr<Element<M,N,R>> element = mesh_.element(ID);
    if(element->contains(point)){
      (args(element, point), ...); // parameter pack expansion to call functor on the pair (element, point).
      return element;
    }
  }
  // no element found (this will rise an Address bounday error at runtime, rise an exception instead)
  return nullptr;
}
