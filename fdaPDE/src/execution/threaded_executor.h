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
inline std::mutex parallel_config_mutex;
inline bool parallel_executor_initialized = false;
inline int parallel_num_threads = static_cast<int>(
  std::min(fdapde::available_concurrency(), static_cast<std::size_t>(std::numeric_limits<int>::max())));

// implementation of the random stealing algorithm
// * "Blumofe, R. D., & Leiserson, C. E. (1999). Scheduling multithreaded computations by work stealing. Journal of
//    the ACM (JACM), 46(5), 720-748."
/// @brief samples other workers as possible work-stealing victims
struct random_stealing_policy {
    /// @brief configures the victim range and number of stealing attempts
    explicit random_stealing_policy(std::size_t n, int probes = 2) :
        probes_(probes), dist_(0, static_cast<int>(n - 1)) { }

    /// @brief selects a worker according to this policy
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
    /// @brief returns the calling thread random generator
    static std::mt19937& tls_rng() {   // thread local rng to avoid races
        thread_local std::mt19937 rng {std::random_device {}()};
        return rng;
    }
    int probes_;
    std::uniform_int_distribution<int> dist_;
};

/// @brief distributes submissions cyclically over the workers
struct round_robin_scheduling_policy {
    /// @brief initializes cyclic scheduling over the supplied worker count
    explicit round_robin_scheduling_policy(std::size_t n) : size_(n) { }

    /// @brief selects a worker according to this policy
    int pick() { return static_cast<int>(curr_.fetch_add(1, std::memory_order_relaxed) % size_); }
   private:
    std::atomic<std::size_t> curr_ {0};
    std::size_t size_;
};

/// @brief coordinates task ownership, accounting and a work-stealing worker pool
struct threaded_executor_impl {
    using worker_type = internals::worker;
    using worker_pointer = std::unique_ptr<worker_type>;
    using task_type = typename worker_type::task_type;
    using task_pointer = typename worker_type::task_pointer;
    using size_type = std::size_t;
    using stealing_policy = internals::random_stealing_policy;
    using scheduling_policy = internals::round_robin_scheduling_policy;

    /// @brief starts a pool sized by available concurrency
    threaded_executor_impl() : threaded_executor_impl(fdapde::available_concurrency()) { }
    /// @brief starts the requested worker count and waits for startup readiness
    explicit threaded_executor_impl(size_type size) :
        n_workers_(validate_size_(size)),
        init_latch_(n_workers_ + 1),
        stealing_policy_(n_workers_),
        scheduling_policy_(n_workers_) {
        workers_.reserve(n_workers_);
        // start workers
        for (size_type i = 0; i < n_workers_; i++) {
            workers_.emplace_back(std::make_unique<internals::worker>(i, this));
        }
        // wait workers to be ready, avoids executor destruction before worker construction
        init_latch_.arrive_and_wait();
    }

    /// @brief returns the number of worker threads
    size_type size() const { return n_workers_; }
    /// @brief reports whether workers should continue running
    bool is_active() const { return active_.load(std::memory_order_acquire); }
    /// @brief submits a bound callable and returns its result or exception through a future
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
    /// @brief dispatches a bound callable or runs an executor-aware task
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
    /// @brief dispatches a bound callable or runs an executor-aware task
    template <typename Task, typename... Args>
        requires(requires(Task task, threaded_executor_impl* executor, Args... args) { task.run(executor, args...); })
    auto execute(Task&& task, Args&&... args) {
        return task.run(this, std::forward<Args>(args)...);
    }
    /// @brief constructs a callable in the owning worker pool
    template <typename Task> task_pointer allocate_task(int worker, Task&& task) {
        return workers_[worker]->allocate_task(std::forward<Task>(task));
    }
    /// @brief publishes an allocated task to an inbound worker queue
    void enqueue_task(task_pointer task) {
        int worker = scheduling_policy_.pick();
        workers_[worker]->enqueue_task(task);
    }
    /// @brief wakes all workers and callers waiting for task completion
    void notify_all() { cv_.notify_all(); }
    /// @brief reserves group accounting before any task becomes visible to workers
    void reserve_tasks(size_type task_count) {
        std::lock_guard<std::mutex> lock(m_);
        fdapde_assert(
          task_count <= std::numeric_limits<size_type>::max() - task_count_, std::overflow_error,
          "execution task counter overflow");
        task_count_ += task_count;
    }
    /// @brief waits for all submitted work and rejects calls from executor workers
    void join() {
        fdapde_strong_assert(
          this_thread_id() == main_thread_id, std::logic_error,
          "parallel_join cannot be called from an executor worker");
        std::unique_lock<std::mutex> lock(m_);
        cv_.wait(lock, [&]() { return task_count_ == 0; });
    }
    /// @brief helps execute work until the supplied pending-work predicate becomes false
    template <typename Condition>
        requires(requires(Condition cond) {
            { cond() } -> std::convertible_to<bool>;
        })
    void active_join(int worker_id, Condition&& cond) {
        while (cond()) {
            if (worker_id != main_thread_id) {
                workers_[worker_id]->try_execute_one(this);
            } else {
                std::unique_lock<std::mutex> lock(m_);
                cv_.wait(lock, [&]() { return !cond() || !active_.load(std::memory_order_acquire); });
            }
        }
        return;
    }
    /// @brief stops and joins workers without completing pending tasks
    void stop() {
        std::unique_lock<std::mutex> lock(m_);
        active_.store(false, std::memory_order_release);
        lock.unlock();
        cv_.notify_all();
        for (auto& worker : workers_) { worker->join(); }
        return;
    }

    /// @brief stops and joins the worker pool
    ~threaded_executor_impl() { stop(); }
   private:
    /// @brief checks the internal worker count before constructing the pool
    static size_type validate_size_(size_type size) {
        fdapde_assert(
          size > 0 && size <= static_cast<size_type>(std::numeric_limits<int>::max()), std::invalid_argument,
          "executor size must be a positive int");
        return size;
    }
    // woorker coordination utilities
    friend worker_type;
    /// @brief publishes ready successors, destroys the task and releases its accounting
    void on_task_complete(task_pointer task) {
        // decrease ref count, dispatch completed successors back for execution
        for (task_pointer task_ptr : task->inverse_dependencies()) {
            if (task_ptr->ref_count_fetch_sub(1, std::memory_order_acq_rel) == 1) { renqueue_task_(task_ptr); }
        }
        const int allocation_context = task->allocation_context().value();
        workers_[allocation_context]->deallocate_task(task);

        std::lock_guard<std::mutex> lock(m_);
        fdapde_assert(task_count_ > 0, std::logic_error, "completed task was not accounted for");
        task_count_--;
        cv_.notify_all();
        return;
    }
    /// @brief waits until every worker has completed startup
    void on_worker_ready() { init_latch_.arrive_and_wait(); }
    /// @brief waits until work is pending or the executor stops
    void on_worker_idle() {
        std::unique_lock<std::mutex> lock(m_);
        cv_.wait(lock, [&]() { return (task_count_ != 0) || !active_; });
    }
    /// @brief attempts to claim a task from another worker
    std::optional<task_pointer> try_steal(int thief_id) {
        return stealing_policy_.pick(thief_id, [&](int victim) { return workers_[victim]->try_steal(); });
    }

    /// @brief allocates and accounts for one task before publishing it with rollback on failure
    template <typename Task> void dispatch_task_(Task&& task) {
        int w_id = scheduling_policy_.pick();
        task_pointer task_ptr = workers_[w_id]->allocate_task(task_type(std::forward<Task>(task), w_id));
        // reserve accounting before publication and release the task if reservation fails
        try {
            reserve_tasks(1);
        } catch (...) {
            workers_[w_id]->deallocate_task(task_ptr);
            throw;
        }
        try {
            workers_[w_id]->enqueue_task(task_ptr);
        } catch (...) {
            workers_[w_id]->deallocate_task(task_ptr);
            std::lock_guard<std::mutex> lock(m_);
            task_count_--;
            cv_.notify_all();
            throw;
        }
        cv_.notify_all();
        return;
    }
    /// @brief publishes an already accounted successor and wakes the workers
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

/// @brief provides the process-lifetime executor singleton
struct threaded_executor {
    /// @brief initializes once and returns the process-lifetime executor
    static auto& instance() {
        static std::aligned_storage_t<sizeof(threaded_executor_impl), alignof(threaded_executor_impl)> storage;
        static threaded_executor_impl* exec = [] {
            std::lock_guard<std::mutex> lock(parallel_config_mutex);
            auto* result = new (&storage) threaded_executor_impl(parallel_num_threads);
            parallel_executor_initialized = true;
            return result;
        }();
        return *exec;
    }
};

}   // namespace internals

/// @brief detects tasks implementing the runtime run protocol
template <typename Task, typename... Args> struct is_runnable_task {
    static constexpr bool value = requires(Task task, internals::threaded_executor_impl* executor, Args... args) {
        { task.run(executor, args...) } -> std::same_as<void>;
    };
};
template <typename Task, typename... Args>
static constexpr bool is_runnable_task_v = is_runnable_task<Task, Args...>::value;

/// @brief sets a positive worker count before the first executor use
inline void parallel_set_num_threads(int num_threads) {
    fdapde_strong_assert(num_threads > 0, std::invalid_argument, "parallel thread count must be positive");
    std::lock_guard<std::mutex> lock(internals::parallel_config_mutex);
    fdapde_strong_assert(
      !internals::parallel_executor_initialized, std::logic_error,
      "parallel thread count cannot change after executor initialization");
    internals::parallel_num_threads = num_threads;
}
/// @brief returns the configured worker count under the configuration lock
inline int parallel_get_num_threads() {
    std::lock_guard<std::mutex> lock(internals::parallel_config_mutex);
    return internals::parallel_num_threads;
}
/// @brief submits a callable asynchronously or executes and waits for a task graph
template <typename F, typename... Args>
    requires(
      std::is_invocable_v<std::decay_t<F>&, std::decay_t<Args>&...> && std::is_copy_constructible_v<std::decay_t<F>> &&
      (std::is_copy_constructible_v<std::decay_t<Args>> && ...))
void parallel_execute(F&& f, Args&&... args) {
    internals::threaded_executor::instance().execute(std::forward<F>(f), std::forward<Args>(args)...);
}
/// @brief returns a future transporting the submitted callable result or exception
template <typename F, typename... Args>
    requires(std::is_invocable_v<std::decay_t<F>&, std::decay_t<Args>&...>)
auto parallel_async(F&& f, Args&&... args) {
    return internals::threaded_executor::instance().async(std::forward<F>(f), std::forward<Args>(args)...);
}
/// @brief waits for all submitted work and rejects calls from executor workers
inline void parallel_join() { internals::threaded_executor::instance().join(); }

}   // namespace fdapde

#endif   // __FDAPDE_EXECUTION_THREADED_EXECUTOR_H__
