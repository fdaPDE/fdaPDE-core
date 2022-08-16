// a node of a spline tree (an N_ij object in the above schema)
class SplineNode {
 private:
  double k1_, k2_; // knots u_i   and u_i+j
  double d1_ = 0;  // for efficiency reasons precompute 1/(u_i+j - u_i)

 public:
  // constructor
  SplineNode() = default;
 SplineNode(double k1, double k2) : k1_(k1), k2_(k2), d1_(1/(k2_ - k1_)){ }

  // for a spline of order j get the knot span [u_i, u_i+j)
  std::pair<double, double> getKnotSpan() const {
    return std::make_pair(k1_, k2_);
  }
  
  double operator()(double x) const { return (x - k1_)*d1_; }

  // returns true if the point x belongs to the knot span of N_ij
  bool contains(double x) const {
    if(k1_ < k2_) return (x >= k1_ && x < k2_) ? true : false;
    else          return (x >= k2_ && x < k1_) ? true : false;
  }
};

// functor passed to the DFS visit. See Tree.h for more details
struct SplineEvaluator{
  double result_ = 0;
  std::map<int, double> partialMap_{};
  double partialProduct_ = 1;
  double inputPoint_;
    
SplineEvaluator(const node_ptr<SplineNode>& root_ptr, double inputPoint) : inputPoint_(inputPoint) {
  partialMap_[root_ptr->getKey()] = 1;
};

  // add result only if N_i0(x) = 1, that is if x is contained in the knot span [u_i, u_i+1)
  void leafAction(const node_ptr<SplineNode>& currentNode) {
    if(currentNode->getData().contains(inputPoint_))
      result_ += partialProduct_;

    // restore partialProduct to the one of the father
    partialProduct_ = partialMap_[currentNode->getFather()->getKey()];
    return;
  }

  /* A spline is defined by the recursive Cox-De boor formula

     [(x-u_i)/(u_i+j - u_i)]*N_i,j-1(x) or [(u_i+j+1 - x)/(u_i+j+1 - u_i+1)]*N_i+1,j-1(x)

     At the end of recursion the spline equation can be seens as N_ij = w0*N_00 + w1*N_10 + ... + wM*N_M0     
     During DFS visit coefficients w0, w1, ..., wM are computed from root to leafs. firstVisitAction() just develops the coefficients
     product of each term N_i0.
  */
  void firstVisitAction(const node_ptr<SplineNode>& currentNode) {
    // update partial product
    partialProduct_ *= currentNode->getData()(inputPoint_);
    partialMap_[currentNode->getKey()] = partialProduct_;
    return;
  }

  // go back in the recursion tree
  void lastVisitAction(const node_ptr<SplineNode>& currentNode) {
    // restore to partialProduct of the father
    partialProduct_ = partialMap_[currentNode->getFather()->getKey()];
    return;
  }
};

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
