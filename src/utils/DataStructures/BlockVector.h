#ifndef __BLOCK_VECTOR_H__
#define __BLOCK_VECTOR_H__

namespace fdaPDE{

  // a vector made of blocks of equal size (used for space-time problems)
  template <typename T>
  class BlockVector {
  private:
    DVector<T> data_;
    std::size_t n_; // number of blocks
    std::size_t m_; // size of single block
  public:
    // constructor
    BlockVector(std::size_t n, std::size_t m) : n_(n), m_(m) { data_ = DVector<T>::Zero(n*m); }
    auto operator()(std::size_t i) { return data_.block(i*m_,0, m_,1); } // access to i-th block
    auto operator()(std::size_t i, std::size_t j) { return data_.block(i*m_,0, j*m_,1); } // access to blocks (i, i+j)

    auto head(std::size_t i) { return data_.block(0,0, i*m_, 1); } // access to first i blocks
    auto tail(std::size_t i) { return data_.block((n_-i)*m_,0, i*m_, 1); } // access to last i blocks
    // getter to internal data
    const DVector<T>& get() const { return data_; }
  };

}

#endif // __BLOCK_VECTOR_H__
