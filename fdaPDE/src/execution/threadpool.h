// This file is part of fdaPDE, a C++ library for physics-informed
// spatial and functional data analysis.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#ifndef __FDAPDE_EXECUTION_THREADPOOL_H__
#define __FDAPDE_EXECUTION_THREADPOOL_H__

#include "header_check.h"

namespace fdapde {
namespace internals {

// implementation of the random stealing algorithm
// * "Blumofe, R. D., & Leiserson, C. E. (1999). Scheduling multithreaded computations by work stealing. Journal of
//    the ACM (JACM), 46(5), 720-748."
struct random_stealing_policy {
    explicit random_stealing_policy(std::size_t n, int probes = 2) :
        probes_(probes), dist_(0, n - 1), rng_(std::random_device {}()) { }

    template <typename TryStealFunctor>
    std::optional<internals::task_handle*> pick(int self, TryStealFunctor&& try_steal) {
        for (int k = 0; k < probes_; ++k) {
            int victim = dist_(rng_);
            if (victim != self) {
                if (auto task = try_steal(victim)) return task;
            }
        }
        return std::nullopt;
    }
   private:
    int probes_;
    std::uniform_int_distribution<int> dist_;
    mutable std::mt19937 rng_;   // data-race here
};
  
}   // namespace internals
  
class ThreadPool {
    using StealPolicy = internals::random_stealing_policy;
    using WorkerPool = std::vector<std::unique_ptr<internals::worker>>;
   public:
    // constructor
    ThreadPool() : ThreadPool(std::thread::hardware_concurrency()) { }
    explicit ThreadPool(int size) : n_workers_(size), init_latch_(size + 1), steal_policy_(size) {
        workers_.reserve(n_workers_);
        internals::tls_worker_id = 0;   // set main thread worker id to zero
        // start workers
        for (int i = 0; i < n_workers_; i++) {
            workers_.emplace_back(std::make_unique<internals::worker>(1 + i, this));
        }
        // wait workers to be ready, avoids threadpool destruction before worker construction
        init_latch_.arrive_and_wait();
    }
    // observers
    int n_workers() const { return n_workers_; }
  
    // submits f(args...) for asynchronous execution. returns a std::future holding the result
    template <typename F, typename... Args> [[nodiscard]] auto submit(F&& f, Args&&... args) {
        using ret_t = decltype(f(args...));
        auto packaged_task =
          std::make_shared<std::packaged_task<ret_t()>>([f_ = f, ... args_ = args]() { return f_(args_...); });
        // dispatch task for execution
        dispatch_(std::move([packaged_task]() { (*packaged_task)(); }));
        return packaged_task->get_future();
    };
    // execute f(args...) asynchronously. doesn't wait for any result
    template <typename F, typename... Args>
        requires(!std::is_same_v<std::decay_t<F>, TaskGraph>)
    void execute(F&& f, Args&&... args) {
        dispatch_(std::move([f_ = f, ... args_ = args]() { f_(args_...); }));
    }
    // execute a TaskGraph object
    void execute(const TaskGraph& tg) {
        // first load all nodes
        std::unordered_map<internals::task_handle*, internals::task_handle*> task_ptr_map_;
        std::vector<internals::task_handle*> ptr_vec_;
        std::vector<int> worker_id;
        for (std::size_t i = 0; i < tg.nodes(); ++i) {
            // we here need an alloc_task which allocates the task but doesn't submit it to the task_queue_
            internals::task_handle* ptr = tg.adjacency_[i]->task_;
            worker_id.push_back(select_worker_());
            internals::task_handle* stable_ptr = workers_[worker_id.back()]->allocate_task(*(tg.adjacency_[i]->task_)); // qui copiamo, per non lasciare il grafo in uno stato indefinito
            ptr_vec_.push_back(stable_ptr);
            task_ptr_map_.emplace(ptr, stable_ptr);
        }
        // replace internal pointers with stable worker-local pointers
        for (std::size_t i = 0; i < ptr_vec_.size(); ++i) {
            for (std::size_t j = 0; j < ptr_vec_[i]->required_by().size(); ++j) {
                ptr_vec_[i]->required_by()[j] = task_ptr_map_[ptr_vec_[i]->required_by()[j]];
            }
        }

	std::vector<bool> runnable(tg.nodes(), false);
	for(std::size_t i = 0; i < tg.nodes(); ++i) { runnable[i] = ptr_vec_[i]->runnable(); }
	
	// increase counter first
	task_count_.fetch_add((int)ptr_vec_.size(), std::memory_order_acq_rel);

        // send tasks (without allocation)
        for (std::size_t i = 0; i < ptr_vec_.size(); ++i) {
	  if (runnable[i]) {
                workers_[worker_id[i]]->dispatch_handle(ptr_vec_[i]);
                workers_[worker_id[i]]->wake_up();
            }
        }
    }

    // woorker coordination utilities
    void on_task_complete(internals::task_handle* task) {
        if (task_count_.fetch_sub(1, std::memory_order_release) == 1) {
            std::lock_guard<std::mutex> lock(join_m_);
            join_cv_.notify_one();
        } else {
            // decrease ref count, dispatch completed successors back for execution
            for (internals::task_handle* deps : task->required_by()) {
                if (deps->ref_count_fetch_sub(1, std::memory_order_release) == 1) { dispatch_handle_(deps); }
            }
        }
        // dealloca il task
        return;
    }
    void on_worker_ready() { init_latch_.arrive_and_wait(); }
    bool on_worker_idle () { return task_count_.load(std::memory_order_acquire) != 0; }

    std::optional<internals::task_handle*> try_steal(int thief_id) {
        return steal_policy_.pick(thief_id, [&](int victim) { return workers_[victim]->try_steal(); });
    }

    void join() {
        std::unique_lock<std::mutex> lock(join_m_);
        join_cv_.wait(lock, [&]() { return task_count_.load(std::memory_order_acquire) == 0; });
    }

    // stops all running threads. not yet completed tasks are losts
    void stop() {
        for (auto& worker : workers_) { worker->stop(); }
        for (auto& worker : workers_) { worker->join(); }
        return;
    }
    // destructor
    ~ThreadPool() { stop(); }
   private:
    // dispatches task to worker
    std::atomic<int> curr_worker_ {0};   // worker to which to dispatch next task

    int select_worker_() { return curr_worker_.fetch_add(1, std::memory_order_release) % n_workers_; }

    template <typename Task_>
        requires(!std::is_same_v<std::decay_t<Task_>, internals::task_handle>)
    void dispatch_(Task_&& task) {
        // round-robin worker selection
        int worker_id = select_worker_();
        workers_[worker_id]->submit_task(internals::task_handle(std::move(task), worker_id));
        task_count_.fetch_add(1, std::memory_order_release);
        workers_[worker_id]->wake_up();
        return;
    }

    void dispatch_handle_(internals::task_handle* handle) {
        int worker_id = select_worker_();
        workers_[worker_id]->dispatch_handle(handle);
        workers_[worker_id]->wake_up();
        return;
    }

    WorkerPool workers_;                // worker-pool
    const int n_workers_;               // number of workers in the pool
    std::latch init_latch_;             // workers synchronization barrier at startup
    std::atomic<int> task_count_ {0};   // overall number of active tasks
    StealPolicy steal_policy_;

    // joining logic
    std::mutex join_m_;
    std::condition_variable join_cv_;  
};

  
// namespace internals {

// struct threadpool_executor {
//     static auto& instance() {
//         // intentionally leaked to guarantee threadpool teardown at program termination
//         static ThreadPool<>* tp = new ThreadPool<>();
//         return *tp;
//     }
// };

// }   // namespace internals

// // public API

// number of available threads
// int num_threads() { return internals::threadpool_executor::instance().n_workers(); }

}   // namespace fdapde

#endif   // __FDAPDE_EXECUTION_THREADPOOL_H__
