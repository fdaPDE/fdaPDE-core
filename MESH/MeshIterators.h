// allow range-for loop over mesh elements
struct iterator{
private:
  friend Mesh;
  const Mesh* meshContainer_; // pointer to mesh object
  int index_; // keep track of current iteration during for-loop
  // constructor
iterator(const Mesh* container, int index)
  : meshContainer_(container), index_(index) {}; 
public:
  // just increment the current iteration and return this iterator
  iterator& operator++() {
    ++index_;
    return *this;
  }
  // dereference the iterator means to create Element object at current index
  std::shared_ptr<Element<M,N,R>> operator*() {
    return meshContainer_->element(index_);
  }
  // two iterators are different when their indexes are different
  friend bool operator!=(const iterator& lhs, const iterator& rhs) {
    return lhs.index_ != rhs.index_;
  }
  // const version to enable const auto& syntax
  std::shared_ptr<Element<M,N,R>> operator*() const {
    return meshContainer_->element(index_);
  }
};

// iterator allowing to loop on the boundary IDs only
struct boundary_iterator{
private:
  friend Mesh;
  const Mesh* meshContainer_;
  int index_; // current boundary node
  // constructor
boundary_iterator(const Mesh* container, int index)
  : meshContainer_(container), index_(index) {};
public:
  // fetch next boundary node
  boundary_iterator& operator++() {
    index_++;
    // scan until all nodes have been visited or a boundary node is not found
    for(; index_ < meshContainer_->dof_ && meshContainer_->isOnBoundary(index_) != true; ++index_);
    return *this;
  }
  // dereference returns a pair containing the ID of the boundary node and its physical coordinate
  int operator*() const {
    return index_;
  }
  // two iterators are different when their indexes are different
  friend bool operator!=(const boundary_iterator& lhs, const boundary_iterator& rhs) {
    return lhs.index_ != rhs.index_;
  }
};
