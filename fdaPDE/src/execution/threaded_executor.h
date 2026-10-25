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

#ifndef __FDAPDE_EXECUTION_THREADED_EXECUTOR_H__
#define __FDAPDE_EXECUTION_THREADED_EXECUTOR_H__

#include "header_check.h"

namespace fdapde {
namespace internals {

// executor configuration is mutable only before the first task initializes the worker pool
struct threaded_executor_config_t {
    std::mutex m_;
    bool initialized = false;
    int num_threads = static_cast<int>(
      std::min(fdapde::available_concurrency(), static_cast<std::size_t>(std::numeric_limits<int>::max())));
};
inline threaded_executor_config_t threaded_executor_config;

// implementation of the random stealing algorithm
// * "blumofe, r. d., & leiserson, c. e. (1999). scheduling multithreaded computations by work stealing. journal of the
//    acm (jacm), 46(5), 720-748"
struct random_stealing_policy {
    explicit random_stealing_policy(std::size_t n, int probes = 2) :
        probes_(probes), dist_(0, static_cast<int>(n - 1)) { }

    template <typename TryStealFunctor> std::optional<task_handle*> pick(int self, TryStealFunctor&& try_steal) {
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
        thread_local std::mt19937 rng {std::random_device {}()};
        return rng;
    }
    int probes_;
    std::uniform_int_distribution<int> dist_;
};

// implementation of the round-robin scheduling algorithm
struct round_robin_scheduling_policy {
    explicit round_robin_scheduling_policy(std::size_t n) : size_(n) { }

    int pick() { return static_cast<int>(curr_.fetch_add(1, std::memory_order_relaxed) % size_); }
   private:
    std::atomic<std::size_t> curr_ {0};
    std::size_t size_;
};

// multi-threaded runtime system with randomized work-stealing load balancing
struct threaded_executor_impl {
    using worker_type = internals::worker;
    using worker_pointer = std::unique_ptr<worker_type>;
    using task_type = typename worker_type::task_type;
    using task_pointer = typename worker_type::task_pointer;
    using packaged_task = typename worker_type::packaged_task;
    using size_type = std::size_t;
    using stealing_policy = internals::random_stealing_policy;
    using scheduling_policy = internals::round_robin_scheduling_policy;

    // constructor
    threaded_executor_impl() : threaded_executor_impl(fdapde::available_concurrency()) { }
    explicit threaded_executor_impl(size_type size) :
        n_workers_(size), init_latch_(n_workers_ + 1), stealing_policy_(n_workers_), scheduling_policy_(n_workers_) {
        workers_.reserve(n_workers_);
        // start workers
        for (size_type i = 0; i < n_workers_; i++) {
            try {
                workers_.emplace_back(std::make_unique<internals::worker>(i, this));
            } catch (...) {
                active_.store(false, std::memory_order_release);
                // supply fake arrivals for workers that never started, plus the coordinator thread
                init_latch_.count_down(n_workers_ + 1 - workers_.size());
                for (auto& worker : workers_) {
                    if (worker->joinable()) { worker->join(); }
                }
                throw;
            }
        }
        // wait workers to be ready, avoids executor destruction before worker construction
        init_latch_.arrive_and_wait();
    }
    // observers
    size_type size() const { return n_workers_; }
    bool is_active() const { return active_.load(std::memory_order_acquire); }
    // submits f(args...) for asynchronous execution. returns a std::future holding the result
    template <typename F, typename... Args>
        requires(std::is_invocable_v<std::decay_t<F>&, std::decay_t<Args>&...>)
    [[nodiscard]] auto async(F&& f, Args&&... args) {
        using ret_t = std::invoke_result_t<std::decay_t<F>&, std::decay_t<Args>&...>;
        auto bound_task = [f_ = std::forward<F>(f),
                           args_ = std::tuple<std::decay_t<Args>...>(std::forward<Args>(args)...)]() mutable -> ret_t {
            return std::apply([&f_](auto&... stored_args) -> ret_t { return std::invoke(f_, stored_args...); }, args_);
        };
        auto packaged_task = std::make_shared<std::packaged_task<ret_t()>>(std::move(bound_task));
        auto result = packaged_task->get_future();
        dispatch_task_([packaged_task]() { (*packaged_task)(); });
        return result;
    }
    // executes a callable f(args...) asynchronously. doesn't wait for any result
    template <typename F, typename... Args>
        requires(
          std::is_invocable_v<std::decay_t<F>&, std::decay_t<Args>&...> &&
          std::is_copy_constructible_v<std::decay_t<F>> && (std::is_copy_constructible_v<std::decay_t<Args>> && ...))
    void execute(F&& f, Args&&... args) {
        auto bound_task = [f_ = std::forward<F>(f),
                           args_ = std::tuple<std::decay_t<Args>...>(std::forward<Args>(args)...)]() mutable {
            std::apply([&f_](auto&... stored_args) { std::invoke(f_, stored_args...); }, args_);
        };
        dispatch_task_(std::move(bound_task));
    }
    // executes a runnable task object
    template <typename Task, typename... Args>
        requires(requires(Task task, threaded_executor_impl* executor, Args... args) { task.run(executor, args...); })
    auto execute(Task&& task, Args&&... args) {
        return task.run(this, std::forward<Args>(args)...);
    }
    // low-level task handling
    // reserve both task and mailbox storage before submission
    template <typename Task> packaged_task allocate_task(int worker, Task&& task) {
        return workers_[worker]->allocate_task(std::forward<Task>(task));
    }
    // destroy a packaged task that was never submitted
    void deallocate_task(packaged_task task) {
        workers_[task.task->allocation_context().value()]->deallocate_task(task);
    }
    // submit a packaged task without allocating queue storage
    void submit_task(packaged_task task) {
        {
            std::lock_guard<std::mutex> lock(m_);
            workers_[task.task->allocation_context().value()]->submit_task(task);
        }
        work_cv_.notify_one();
        if (completion_waiters_.load() != 0) cv_.notify_all();
    }
    // publish a prepared batch of tasks
    void submit_task(const std::vector<packaged_task>& tasks) {
        {
            std::lock_guard<std::mutex> lock(m_);
            for (packaged_task task : tasks) {
                auto& owner = workers_[task.task->allocation_context().value()];
                owner->submit_task(task);
            }
        }
        work_cv_.notify_all();
        if (completion_waiters_.load() != 0) cv_.notify_all();
    }
    // blocks caller until all tasks queued in the pool have been executed
    void join() {
        if (this_thread_id() != main_thread_id) return;
        std::unique_lock<std::mutex> lock(m_);
        completion_wait registration(join_waiters_);
        cv_.wait(lock, [&]() {
            return std::all_of(
              workers_.begin(), workers_.end(), [](const auto& worker) { return !worker->has_registered_tasks(); });
        });
    }
    // sends the calling worker back to its worker loop, until condition is not met. used in collaboartive waits
    template <typename Condition>
        requires(requires(Condition cond) {
            { cond() } -> std::convertible_to<bool>;
        })
    void active_join(int worker_id, Condition&& cond) {
        while (cond()) {
            if (worker_id != main_thread_id && workers_[worker_id]->try_execute_one(this)) continue;
            std::unique_lock<std::mutex> lock(m_);
            completion_wait registration(completion_waiters_);
            cv_.wait(lock, [&] {
                return !cond() || (worker_id != main_thread_id && has_queued_tasks_()) ||
                       !active_.load(std::memory_order_acquire);
            });
        }
        return;
    }
    // stops all running threads. not yet completed tasks are losts
    void stop() {
        std::unique_lock<std::mutex> lock(m_);
        active_.store(false, std::memory_order_release);
        lock.unlock();
        cv_.notify_all();
        work_cv_.notify_all();
        for (auto& worker : workers_) { worker->join(); }
        return;
    }
    // destructor
    ~threaded_executor_impl() { stop(); }
   private:
    // woorker coordination utilities
    friend worker_type;
    // tells completing tasks that a caller in join or active_join may need waking up when they finish
    struct completion_wait {
        explicit completion_wait(std::atomic<size_type>& count) : count_(count) {
            ++count_;
            std::atomic_thread_fence(std::memory_order_seq_cst);
        }
        ~completion_wait() { --count_; }
        std::atomic<size_type>& count_;
    };
    // invoked by a worker to signal completion of task, notify possibly waiting workers
    void on_task_complete(task_pointer task) {
        const bool owner_done = workers_[task->allocation_context().value()]->delete_task(task);
        std::atomic_thread_fence(std::memory_order_seq_cst);
        if (completion_waiters_.load() != 0 || (owner_done && join_waiters_.load() != 0)) {
            // handshake with predicate checks under m_ to prevent a missed completion wakeup
            std::lock_guard<std::mutex> lock(m_);
            cv_.notify_all();
        }
    }
    // invoked by a worker to signal its readyness
    void on_worker_ready() { init_latch_.arrive_and_wait(); }
    // invoked by a worker to decide whether to move to an idle state
    void on_worker_idle() {
        std::unique_lock<std::mutex> lock(m_);
        work_cv_.wait(lock, [&]() { return has_queued_tasks_() || !active_.load(std::memory_order_acquire); });
    }
    // invoked by a worker to perform a stealing attempt
    std::optional<task_pointer> try_steal(int thief_id) {
        if (auto task = stealing_policy_.pick(thief_id, [&](int victim) { return workers_[victim]->try_steal(); })) {
            return task;
        }
        // exhaust available victims before parking, rather than retrying the global idle lock after two misses
        for (size_type offset = 1; offset < n_workers_; ++offset) {
            if (auto task = workers_[(thief_id + offset) % n_workers_]->try_steal()) { return task; }
        }
        return std::nullopt;
    }
    // read worker-local queue counts only while coordinating parking
    bool has_queued_tasks_() {
        return std::any_of(
          workers_.begin(), workers_.end(), [](const auto& worker) { return worker->has_queued_tasks(); });
    }
    // allocates and submits task to the pool, waking an idle worker
    template <typename Task> void dispatch_task_(Task&& task) {
        int w_id = scheduling_policy_.pick();
        packaged_task prepared = allocate_task(w_id, task_type(std::forward<Task>(task), w_id));
        try {
            submit_task(prepared);
        } catch (...) {
            deallocate_task(prepared);
            throw;
        }
        return;
    }

    std::vector<worker_pointer> workers_;             // worker pool
    const size_type n_workers_;                       // size of the pool
    std::latch init_latch_;                           // workers synchronization barrier at startup
    stealing_policy stealing_policy_;                 // stealing algorithm
    scheduling_policy scheduling_policy_;             // scheduling algorithm
    alignas(64) std::atomic<bool> active_ {true};     // asserted false to indicate pool deactivation
    std::atomic<size_type> join_waiters_ {0};         // number of workers waiting for a global join predicate
    std::atomic<size_type> completion_waiters_ {0};   // callers currently waiting in a collaborative wait
    alignas(64) std::mutex m_;
    std::condition_variable cv_;        // completion and collaborative waits
    std::condition_variable work_cv_;   // idle workers
};

struct threaded_executor {
    // intentionally leaked to guarantee executor teardown at program termination (avoid crashes due to active threads
    // accessing the executor after its destruction); memory is statically allocated to satisfy leak detectors
    static auto& instance() {
        static std::aligned_storage_t<sizeof(threaded_executor_impl), alignof(threaded_executor_impl)> storage;
        static threaded_executor_impl* exec = [] {
            std::lock_guard<std::mutex> lock(threaded_executor_config.m_);
            auto* result = new (&storage) threaded_executor_impl(threaded_executor_config.num_threads);
            threaded_executor_config.initialized = true;
            return result;
        }();
        return *exec;
    }
};

}   // namespace internals

// public api

template <typename Task, typename... Args> struct is_runnable_task {
    static constexpr bool value = requires(Task task, internals::threaded_executor_impl* executor, Args... args) {
        { task.run(executor, args...) } -> std::same_as<void>;
    };
};
template <typename Task, typename... Args>
static constexpr bool is_runnable_task_v = is_runnable_task<Task, Args...>::value;

// set number of worker threads
inline void parallel_set_num_threads(int num_threads) {
    fdapde_assert(num_threads > 0, "number of requested threads must be positive");
    std::lock_guard<std::mutex> lock(internals::threaded_executor_config.m_);
    if (internals::threaded_executor_config.initialized) {
        throw std::logic_error("thread count cannot change after executor initialization");
    }
    internals::threaded_executor_config.num_threads = num_threads;
}
// get number of worker threads
inline int parallel_get_num_threads() {
    std::lock_guard<std::mutex> lock(internals::threaded_executor_config.m_);
    return internals::threaded_executor_config.num_threads;
}
// executes a callable object asynchronously
template <typename F, typename... Args>
    requires(
      std::is_invocable_v<std::decay_t<F>&, std::decay_t<Args>&...> && std::is_copy_constructible_v<std::decay_t<F>> &&
      (std::is_copy_constructible_v<std::decay_t<Args>> && ...))
void parallel_execute(F&& f, Args&&... args) {
    internals::threaded_executor::instance().execute(std::forward<F>(f), std::forward<Args>(args)...);
}
// executes a callable object asynchronously returning a std::future object for synchronization and result retrieval
template <typename F, typename... Args>
    requires(std::is_invocable_v<std::decay_t<F>&, std::decay_t<Args>&...>)
auto parallel_async(F&& f, Args&&... args) {
    return internals::threaded_executor::instance().async(std::forward<F>(f), std::forward<Args>(args)...);
}
// explicitly joins the executor until all work is completed
inline void parallel_join() { internals::threaded_executor::instance().join(); }

}   // namespace fdapde

#endif   // __fdapde_execution_threaded_executor_h__
