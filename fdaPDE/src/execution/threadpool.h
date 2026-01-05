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
        probes_(probes), dist_(0, static_cast<int>(n - 1)) { }

    template <typename TryStealFunctor>
    std::optional<task_handle*> pick(int self, TryStealFunctor&& try_steal) {
        for (int k = 0; k < probes_; ++k) {
            int victim = dist_(tls_rng());
            if (victim != self) {
                if (auto task = try_steal(victim)) return task;
            }
        }
        return std::nullopt;
    }
   private:
    static std::mt19937& tls_rng() {   // thread local rng to avoid races
        thread_local std::mt19937 rng{std::random_device{}()};
        return rng;
    }
    int probes_;
    std::uniform_int_distribution<int> dist_;
};

// implementation of the round-robin scheduling algorithm
struct round_robin_scheduling_policy {
    explicit round_robin_scheduling_policy(std::size_t n) : size_(n) {}

    int pick() { return curr_.fetch_add(1, std::memory_order_relaxed) % size_; }
   private:
    std::atomic<int> curr_ {0};
    std::size_t size_;
};

}   // namespace internals

struct ThreadPool {
    using worker_type = internals::worker;
    using worker_pointer = std::unique_ptr<worker_type>;
    using task_type = typename worker_type::task_type;
    using task_pointer = typename worker_type::task_pointer;
    using size_type = std::size_t;
    using stealing_policy = internals::random_stealing_policy;
    using scheduling_policy = internals::round_robin_scheduling_policy;

    // constructor
    ThreadPool() : ThreadPool(std::thread::hardware_concurrency()) { }
    explicit ThreadPool(size_type size) :
        n_workers_(size), init_latch_(size + 1), stealing_policy_(size), scheduling_policy_(size) {
        workers_.reserve(n_workers_);
        internals::tls_worker_id = 0;   // set main thread worker id to zero
        // start workers
        for (size_type i = 0; i < n_workers_; i++) {
            workers_.emplace_back(std::make_unique<internals::worker>(1 + i, this));
        }
        // wait workers to be ready, avoids threadpool destruction before worker construction
        init_latch_.arrive_and_wait();
    }
    // observers
    size_type size() const { return n_workers_; }
    bool is_active() const { return active_.load(std::memory_order_acquire); }

    // submits f(args...) for asynchronous execution. returns a std::future holding the result
    template <typename F, typename... Args> [[nodiscard]] auto submit(F&& f, Args&&... args) {
        using ret_t = decltype(f(args...));
        auto packaged_task =
          std::make_shared<std::packaged_task<ret_t()>>([f_ = f, ... args_ = args]() { return f_(args_...); });
        // dispatch task for execution
        dispatch_task_(std::move([packaged_task]() { (*packaged_task)(); }));
        return packaged_task->get_future();
    };
    // execute f(args...) asynchronously. doesn't wait for any result
    template <typename F, typename... Args>
        requires(!std::is_same_v<std::decay_t<F>, TaskGraph>)
    void execute(F&& f, Args&&... args) {
        dispatch_task_(std::move([f_ = f, ... args_ = args]() { f_(args_...); }));
    }
    // execute a TaskGraph object
    void execute(const TaskGraph& tg) {
        const size_type num_nodes = tg.nodes();
        if (num_nodes == 0) return;

        std::unordered_map<task_pointer, task_pointer> task_map;
        std::vector<int> worker_vec(num_nodes);
        std::vector<task_pointer> ready_tasks;
	// copy task graph to stable memory
        for (size_type i = 0; i < num_nodes; ++i) {
            int w_id = scheduling_policy_.pick();
            worker_vec[i] = w_id;
	    task_pointer old_ptr = tg.adjacency_[i]->task_;
	    task_pointer new_ptr = workers_[w_id]->allocate_task(*old_ptr);
            task_map[old_ptr] = new_ptr;
        }
        // replace old dependency pointers with stable worker-local pointers
        for (const auto& [_, new_ptr] : task_map) {
            for (auto& old_ptr : new_ptr->required_by()) { old_ptr = task_map[old_ptr]; }
            if (new_ptr->runnable()) { ready_tasks.push_back(new_ptr); }
        }
        // notify workers and start execution
        std::unique_lock<std::mutex> lock(m_);
        task_count_ += num_nodes;   // increase task count
        lock.unlock();
        for (size_type i = 0; i < ready_tasks.size(); ++i) {
            workers_[worker_vec[i]]->enqueue_task(ready_tasks[i]);
        }
        cv_.notify_all();
        return;
    }
    // blocks caller until all tasks queued in the pool have been executed
    void join() {
        std::unique_lock<std::mutex> lock(m_);
        cv_.wait(lock, [&]() { return task_count_ == 0; });
    }
    // stops all running threads. not yet completed tasks are losts
    void stop() {
        std::unique_lock<std::mutex> lock(m_);
        active_.store(false, std::memory_order_release);
        lock.unlock();
        cv_.notify_all();
        for (auto& worker : workers_) { worker->join(); }
        return;
    }
    // destructor
    ~ThreadPool() { stop(); }
   private:
    // woorker coordination utilities
    friend worker_type;
    // invoked by a worker to signal the completion of a task
    void on_task_complete(task_pointer task) {
        std::unique_lock<std::mutex> lock(m_);
        task_count_--;
        if (task_count_ == 0) {
            cv_.notify_all();
            return;
        }
        lock.unlock();
        // decrease ref count, dispatch completed successors back for execution
        for (task_pointer task_ptr : task->required_by()) {
            if (task_ptr->ref_count_fetch_sub(1, std::memory_order_release) == 1) { renqueue_task_(task_ptr); }
        }
	// deallocate
        workers_[task->allocation_context().value()]->deallocate_task(task);
        return;
    }
    // invoked by a worker to signal its readyness
    void on_worker_ready() { init_latch_.arrive_and_wait(); }
    // invoked by a worker to decide whether to move to an idle state
    void on_worker_idle() {
        std::unique_lock<std::mutex> lock(m_);
        cv_.wait(lock, [&]() { return (task_count_ != 0) || !active_; });
    }
    // invoked by a worker to perform a stealing attempt
    std::optional<task_pointer> try_steal(int thief_id) {
        return stealing_policy_.pick(thief_id, [&](int victim) { return workers_[victim]->try_steal(); });
    }

    // allocates and submits task to the pool. notifies all workers for execution
    template <typename Task>
        requires(!std::is_same_v<std::decay_t<Task>, task_pointer>)
    void dispatch_task_(Task&& task) {
        int w_id = scheduling_policy_.pick();
        workers_[w_id]->submit_task(task_type(std::move(task), w_id));
        std::unique_lock<std::mutex> lock(m_);
        task_count_++;
        lock.unlock();
        cv_.notify_all();
        return;
    }
    // submits an already allocated task for execution.
    void renqueue_task_(task_pointer task) {
        int w_id = scheduling_policy_.pick();
        workers_[w_id]->enqueue_task(task);
        cv_.notify_all();
        return;
    }

    std::vector<worker_pointer> workers_;   // worker pool
    const size_type n_workers_;             // size of the pool
    std::latch init_latch_;                 // workers synchronization barrier at startup
    stealing_policy stealing_policy_;       // stealing algorithm
    scheduling_policy scheduling_policy_;   // scheduling algorithm
    std::atomic<bool> active_ {true};       // asserted false to indicate pool deactivation
    size_type task_count_ = 0;              // overall number of active (executed + pending) tasks
    std::mutex m_;
    std::condition_variable cv_;
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
