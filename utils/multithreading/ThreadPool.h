#ifndef __THREAD_POOL_H__
#define __THREAD_POOL_H__

#include <thread>
#include <future>

#include "ConcurrentQueue.h"

namespace fdaPDE{
namespace multithreading {

  // thread pool implementation for parallel execution of jobs
  class ThreadPool {
  private:
    // possible states for a worker thread in the pool
    enum State { IDLE, BUSY };
    std::vector<State> state_;
    
    ConcurrentQueue<std::function<void()>> task_queue_; 
    std::vector<std::thread> thread_pool_;

    std::mutex mutex_;
    std::condition_variable pending_;
    
    bool kill_ = false; // if asserted true, ThreadPool stops its execution
    bool sync_ = false; // if asserted true, sync on the task_queue is requested
    std::mutex sync_mutex_;
    std::condition_variable sync_condition_;

    // main thread loop, this gets executed by any spawned thread
    void execute(std::size_t thread_id) {
      while(true) {
	std::function<void()> task; // task's function object to execute
	{
	  // wait on condition variable
	  std::unique_lock<std::mutex> lock(mutex_);
	  pending_.wait(lock, [this, &task] {
	    // if there is some pending job or kill signal arrived, stop wait
	    return kill_ || task_queue_.pop(task); 
	  });
	} // release lock
	
	if(kill_) return; // stop if meanwhile a kill signal has been issued
	state_[thread_id] = State::BUSY;
	task(); // execute task
	state_[thread_id] = State::IDLE;
	if(sync_){ // notifies caller if pending for synchronization
	  std::lock_guard<std::mutex> lock(sync_mutex_);
	  sync_condition_.notify_one();
	}
      }
    }
    
  public:
    // constructor
    ThreadPool() : ThreadPool(std::thread::hardware_concurrency()) {};
    ThreadPool(std::size_t n_thread) : state_(n_thread, State::IDLE) {
      // starts pool of n_threads
      thread_pool_.reserve(n_thread);
      for(std::size_t i = 0; i < n_thread; ++i){
	thread_pool_.emplace_back(std::bind(&ThreadPool::execute, this, i));
      }
    }

    // avoid copy/move-constructors and copy/move assignments
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool(ThreadPool&&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;
    ThreadPool& operator=(ThreadPool&&) = delete;
    
    // send work to the pool
    template <typename F, typename... Args>
    auto send_async(F&& f, Args&&... args) -> std::future<decltype(f(args...))> {      
      // wrap the callable target for asynchronous execution
      typedef std::packaged_task<decltype(f(args...))()> task_type;
      std::shared_ptr<task_type> task = std::make_shared<task_type>([=](){ return f(args...); });
      std::future<decltype(f(args...))> future = task->get_future(); // get future associated with promised result
      
      // erase specific type of requested job
      std::function<void()> wrapped_task = [task]() -> void { (*task)(); };

      // put task on queue (concurrent push/pop handled by the queue)
      task_queue_.push(wrapped_task);
      // notify one thread, if any waiting for job
      std::lock_guard<std::mutex> lock(mutex_);
      pending_.notify_one();
      return future;
    }

    // blocks caller until all job sent is done
    void sync() {
      std::unique_lock<std::mutex> lock(sync_mutex_);
      sync_ = true; // caller put on wait
      sync_condition_.wait(lock, [this] {
	return task_queue_.empty() &&
	  std::all_of(state_.begin(), state_.end(), [](State s) { return s == State::IDLE; });
      });
    }

    // terminate the pool, if some thread is executing a job, it will terminate at the end of the execution
    void shutdown() {
      {
	std::lock_guard<std::mutex> lock(mutex_);
	kill_ = true;
      } // release lock
      pending_.notify_all(); // wake-up all threads, they will exit because of kill_ == true
      // wait for thread to finish
      for(auto& th : thread_pool_){
	th.join();
      }
      return;
    }
  };
  
}}

#endif // __THREAD_POOL_H__
