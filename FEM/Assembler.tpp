// stiff matrix for laplacian operator
template <unsigned int M, unsigned int N, unsigned int R, typename B, typename I>
template <typename E>
Eigen::SparseMatrix<double> Assembler<M, N, R, B, I>::assemble(const E& bilinearForm) {

  std::vector<Eigen::Triplet<double>> tripletList;  // store triplets (node_i, node_j, integral_value)
  Eigen::SparseMatrix<double> stiffnessMatrix;      // stiffness matrix is sparse due to the local support of basis functions

  // properly preallocate memory to avoid reallocations
  tripletList.reserve(n_basis*n_basis*mesh_.elements());
  stiffnessMatrix.resize(mesh_.nodes(), mesh_.nodes());

  // cycle over all mesh elements
  for(const auto& e : mesh_){
    // consider all pair of nodes
    for(size_t i = 0; i < n_basis; ++i){
      for(size_t j = 0; j < n_basis; ++j){
	if constexpr(is_symmetric<decltype(bilinearForm)>::value){
	  // compute only half of the discretization matrix if the operator is symmetric
	  if(e->nodeIDs()[i] >= e->nodeIDs()[j]){
	    // any integral computation for the construction of the stiffness matrix is performed on the reference element
	    double value = integrator_.integrate(referenceBasis_, *e, i, j, bilinearForm);
	    
	    // From Eigen doucmentation: The input list of triplets does not have to be sorted, and can contains duplicated elements.
	    // In any case, the result is a sorted and compressed sparse matrix where the duplicates have been summed up.
	    // Linearity of the integral is implicitly used during matrix construction by eigen!
	    tripletList.emplace_back(e->nodeIDs()[i], e->nodeIDs()[j], value);
	  }
	}else{
	  // not any optimization to perform in the general case
	  double value = integrator_.integrate(referenceBasis_, *e, i, j, bilinearForm);
	  tripletList.emplace_back(e->nodeIDs()[i], e->nodeIDs()[j], value);
	}
      }
    }
  }
  // stiff matrix assembled
  stiffnessMatrix.setFromTriplets(tripletList.begin(), tripletList.end());
  stiffnessMatrix.prune(std::numeric_limits<double>::epsilon() * 10); // remove almost zero entries
  
  // return just half of the discretization matrix if the form is symmetric (lower triangular part)
  if constexpr(is_symmetric<decltype(bilinearForm)>::value)
    return stiffnessMatrix.selfadjointView<Eigen::Lower>();
  else
    return stiffnessMatrix;
};

template <unsigned int M, unsigned int N, unsigned int R, typename B, typename I>
Eigen::Matrix<double, Eigen::Dynamic, 1> Assembler<M, N, R, B, I>::forcingTerm(const Eigen::Matrix<double, Eigen::Dynamic, 1>& f) {

  Eigen::Matrix<double, Eigen::Dynamic, 1> result{};
  result.resize(mesh_.nodes(), 1); // there are as many basis functions as number of nodes in the mesh
  result.fill(0); // init result vector to zero
  
  // build forcing vector
  for(const auto& e : mesh_){
    // integrate on each node
    std::array<std::size_t, ct_nnodes(M,R)> nodes = e->nodeIDs();
    for(size_t i = 0; i < n_basis; ++i){
      if(!mesh_.isOnBoundary(nodes[i])){ // skip computation if node is a boundary node
	auto phi_i = (*basis_[e->ID()])[i];
	// f[e->ID()] is the value of the discretized forcing field (given as datum) over the current element
	auto functional = f[e->ID()]*phi_i; // functional to integrate
	
	// perform integration and store result exploiting additiviy of the integral
	result[e->nodeIDs()[i]] += integrator_.integrate(*e, functional);
      }else{
	result[e->nodeIDs()[i]] += 0;
      }
    }
  }
  return result;
}
