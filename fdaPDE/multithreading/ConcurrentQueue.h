#ifndef __CONCURRENT_QUEUE_H__
#define __CONCURRENT_QUEUE_H__

#include <mutex>
#include <queue>
#include <optional>

namespace fdaPDE {
namespace multithreading {

  // a C++17 thread-safe std::queue implementation. Acts as a wrapper to std::queue with imposed thread safety
  template <typename T>
  class ConcurrentQueue {
  private:
    typedef std::queue<T> Container;
    Container queue_;
    std::mutex mutex_;
  public:
    using value_type      = typename Container::value_type;
    using reference       = typename Container::reference;
    using const_reference = typename Container::const_reference;
    using size_type       = typename Container::size_type;
    using container_type  =          Container;    

    // construct empty queue
    ConcurrentQueue() = default;
    // construct using a range of value_type objects
    template <typename Iterator>
    ConcurrentQueue(Iterator first, Iterator last) : queue_(first, last) {};
    
    // current number of elements in the queue, can be queried by any thread
    size_type size() const { return queue_.size(); }
    bool empty() const { return queue_.empty(); }
    
    // inserts element at the end
    void push(const value_type& value) { 
      std::lock_guard<std::mutex> lock(mutex_);
      queue_.push(value);
    }

    // constructs element in-place at the end
    template <typename... Args>
    void emplace(Args&&... args){
      std::lock_guard<std::mutex> lock(mutex_);
      queue_.emplace(args...);
    }
    
    // removes and returns the first element, returns an empty optional if queue is empty
    std::optional<value_type> pop() {
      std::lock_guard<std::mutex> lock(mutex_);
      if(queue_.empty()){
	return std::nullopt; // return empty optional
      }
      // get first element and remove from queue
      value_type value = queue_.front();
      queue_.pop();
      return std::optional<value_type>(value);
    }
    // removes and place into buffer the first element, notifies with a boolean if extraction was ok
    bool pop(value_type& buff) {
      std::lock_guard<std::mutex> lock(mutex_);
      if(queue_.empty()){
	return false; // nothing to extract
      }
      // get first element and remove from queue
      buff = queue_.front();
      queue_.pop();
      return true;
    }
    
    // atomically removes all elements from the queue
    void clear() {
      std::lock_guard<std::mutex> lock(mutex_);
      while(!queue_.empty()){
	queue_.pop();
      }
    }
  };

}}

#endif // __CONCURRENT_QUEUE_H__
