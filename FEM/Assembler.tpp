// assembly for the discretization matrix of a general bilinear form L
template <unsigned int M, unsigned int N, unsigned int R, typename B, typename I>
template <typename E>
Eigen::SparseMatrix<double> Assembler<M, N, R, B, I>::assemble(const E& bilinearForm) {

  std::vector<Eigen::Triplet<double>> tripletList;  // store triplets (node_i, node_j, integral_value)
  Eigen::SparseMatrix<double> discretizationMatrix; // discretization matrix is sparse due to the local support of basis functions

  // properly preallocate memory to avoid reallocations
  tripletList.reserve(n_basis*n_basis*mesh_.elements());
  discretizationMatrix.resize(mesh_.nodes(), mesh_.nodes());

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
  // matrix assembled
  discretizationMatrix.setFromTriplets(tripletList.begin(), tripletList.end());
  discretizationMatrix.makeCompressed();
  
  // return just half of the discretization matrix if the form is symmetric (lower triangular part)
  if constexpr(is_symmetric<decltype(bilinearForm)>::value)
    return discretizationMatrix.selfadjointView<Eigen::Lower>();
  else
    return discretizationMatrix;
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
	// perform integration on reference element and store result exploiting additiviy of the integral
	result[e->nodeIDs()[i]] += integrator_.integrate(*e, f, referenceBasis_[i]); // \int_e [f*\psi]
      }else{
	result[e->nodeIDs()[i]] += 0;	// implicitly force homogeneous boundary conditions
      }
    }
  }
  return result;
}
