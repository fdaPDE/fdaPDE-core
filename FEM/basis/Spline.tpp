// build a spline tree given the knot vector and a pair of indexes i,j,
// where i denotes the knot the spline refers to and j the spline order
Spline::Spline(const std::vector<double>& knotVector, int i, int j) : knotVector_(knotVector), i_(i), j_(j) {
  // build a pair from (i,j)
  typedef std::pair<int, int> pairID;
  pairID rootID = std::make_pair(i,j);
    
  // insert root in spline tree
  Tree<SplineNode> splineTree = Tree<SplineNode>(SplineNode());
    
  // use a queue structure to assist the spline build process
  std::queue<std::pair<unsigned int, pairID>> queue{};
  queue.push(std::make_pair(splineTree.getRoot()->getKey(),rootID)); // push root to queue
    
  // spline construction
  while(!queue.empty()){
    std::pair<unsigned int, pairID> currentNode = queue.front();
    queue.pop();
    
    int nodeID = currentNode.first;           // identifier of this node
    pairID currentPair = currentNode.second;  // pair (i,j) of spline node N_ij(x)

    // build left spline node
    SplineNode leftNode  = SplineNode(knotVector[currentPair.first],
				      knotVector[currentPair.first + currentPair.second]);

    auto left_ptr = splineTree.insert(leftNode, nodeID, LinkDirection::LEFT);

    
    // build right spline node
    SplineNode rightNode = SplineNode(knotVector[currentPair.first + 1 + currentPair.second],
				      knotVector[currentPair.first + 1]);
    auto right_ptr = splineTree.insert(rightNode, nodeID, LinkDirection::RIGHT);      
      
    // push child nodes to queue for later processing if children nodes are not leafs
    if(currentPair.second - 1 > 0){
      queue.push(std::make_pair(left_ptr->getKey(),
				std::make_pair(currentPair.first,
					       currentPair.second - 1)));
      queue.push(std::make_pair(right_ptr->getKey(),
				std::make_pair(currentPair.first + 1,
					       currentPair.second - 1)));
    }
  }
  spline_.push_back(std::make_tuple(1, splineTree, i, j));
  return;
}

double Spline::eval(const Tree<SplineNode>& tree, double x) const {
  // perform a DFS search to collect the spline evaluation
  SplineEvaluator evaluator = SplineEvaluator(tree.getRoot(), x);
  tree.DFS(evaluator);
  return evaluator.result_;
}

// evaluate a spline at a given point.
double Spline::operator()(double x) const {
  double result;
  for(const auto& tree : spline_)
    result += std::get<0>(tree)*eval(std::get<1>(tree), x);
  return result;
}

// compute derivative of spline. The derivative of a spline is just another spline with the following expression
// d/dx N_ij(x) = j/(u_i+j - u_i)*N_i,j-1(x)  - j/(u_i+j+1 - u_i+1)*N_i+1,j-1(x)
//                |-------------||---------|  |--------------------||-----------|
//                     alpha         N_L               beta              N_R
// this method just cycles over the single spline subtrees computing for each of them a pair of low order splines as described
// in the equation above
Spline Spline::gradient() const {
  std::vector<std::tuple<double, Tree<SplineNode>, int, int>> gradientSpline;
  
  for(const auto& s : spline_){
    int i = std::get<2>(s), j = std::get<3>(s);
    // build low order splines
    Spline N_L(knotVector_, i,   j-1);
    Spline N_R(knotVector_, i+1, j-1);
    
    // compute adjustment coefficients and push result in gradientSpline vector
    if(knotVector_[i+j] - knotVector_[i] != 0){
      double alpha = j/(knotVector_[i+j] - knotVector_[i]);
      gradientSpline.push_back(std::make_tuple(alpha, N_L[0], i,   j-1));    
    }
    if(knotVector_[i+j+1] - knotVector_[i+1] != 0){
      double beta  = -j/(knotVector_[i+j+1] - knotVector_[i+1]);
      gradientSpline.push_back(std::make_tuple(beta,  N_R[0], i+1, j-1));
    }
  }

  return Spline(knotVector_, gradientSpline);
}
