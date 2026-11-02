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

// executor global status, mutable only before the first threaded_executor initialization
/// @brief provides access to the process-lifetime worker pool
struct threaded_executor;
/// @brief serializes executor configuration and freezes it on startup
struct threaded_executor_global_state_t {
    /// @brief starts with mutable configuration and detected parallelism
    threaded_executor_global_state_t() = default;
    friend threaded_executor;

    /// @brief validates and sets total parallelism while configuration remains mutable
    void set_num_threads(int num_threads) {
        std::lock_guard<std::mutex> lock(m_);
        fdapde_strong_assert(num_threads > 0, std::invalid_argument, "number of requested threads must be positive");
        fdapde_strong_assert(!is_ready_, std::logic_error, "thread count cannot change after executor initialization");
        num_threads_ = num_threads;
    }
    /// @brief fixes affinity before the runtime first initializes
    void set_affinity(bool enabled) {
        std::lock_guard<std::mutex> lock(m_);
        fdapde_strong_assert(!is_ready_, std::logic_error, "affinity cannot change after executor initialization");
        cpu_affinity_enabled = enabled;
    }
    /// @brief returns configured total parallelism
    int num_threads() const { return num_threads_.load(std::memory_order_relaxed); }
   private:
    /// @brief freezes configuration while its mutex is held by executor initialization
    void set_ready_() {
        // requires m_ to be externally acquired
        is_ready_ = true;
    }
    mutable std::mutex m_;
    bool is_ready_ = false;
    std::atomic<int> num_threads_ = fdapde::available_concurrency();
};
inline threaded_executor_global_state_t threaded_executor_global_state;

// implementation of the random stealing algorithm
// * "blumofe, r. d., & leiserson, c. e. (1999). scheduling multithreaded computations by work stealing. journal of the
//    acm (jacm), 46(5), 720-748"
/// @brief selects stealing victims with a randomized starting context
struct random_stealing_policy {
    /// @brief configures the range of potential stealing victims
    explicit random_stealing_policy(int size) : size_(size) { }
    /// @brief selects candidate contexts according to this scheduling policy
    template <typename TryStealFunctor> std::optional<task_handle*> pick(int self, TryStealFunctor&& try_steal) {
        int victim_id = static_cast<int>(random_() % size_);
        for (int k = 0; k < size_; ++k) {
            if (victim_id != self) {
                if (auto task = try_steal(victim_id)) { return task; }
            }
            if (++victim_id == size_) victim_id = 0;   // increment-and-wrap, avoid integer division
        }
        return std::nullopt;
    }
   private:
    // a cheap per-thread random number generator
    /// @brief advances a thread-local generator for low-cost stealing victim selection
    static std::uint64_t random_() {
        thread_local std::uint64_t state =
          0x9e3779b97f4a7c15ULL ^ static_cast<std::uint64_t>(std::hash<std::thread::id> {}(std::this_thread::get_id()));
        state ^= state << 13;
        state ^= state >> 07;
        state ^= state << 17;
        return state;
    }
    const int size_;
};

// implementation of the round-robin scheduling algorithm
/// @brief distributes external submissions cyclically over background contexts
struct round_robin_scheduling_policy {
    /// @brief configures the cyclic range of background submission contexts
    explicit round_robin_scheduling_policy(std::size_t n) : size_(n) { }
    /// @brief selects candidate contexts according to this scheduling policy
    int pick() {
        auto current = curr_.load(std::memory_order_relaxed);
        for (;;) {
            auto next = current + 1 == size_ ? 0 : current + 1;   // increment-and-wrap, avoid integer division
            if (curr_.compare_exchange_weak(current, next, std::memory_order_relaxed)) {
                return static_cast<int>(current);
            }
        }
    }
   private:
    std::atomic<std::size_t> curr_ {0};
    std::size_t size_;
};

// executor-owned task storage, execution contexts, and physical background threads
/// @brief coordinates allocation, context scheduling and background worker lifetime
struct threaded_executor_impl {
   private:
    using context_type = execution_context;
    using context_pointer = std::unique_ptr<context_type>;
    using allocator_type = pool_allocator<task_handle>;
    using stealing_policy = internals::random_stealing_policy;
    using scheduling_policy = internals::round_robin_scheduling_policy;
    // spin-loop and backoff constants
    static constexpr int min_spin_retention_us = 250;
    static constexpr int max_spin_retention_us = 1000;
    static constexpr int max_spin_pauses = 32;
    static constexpr int backoff_factor = 2;
   public:
    using size_type = std::size_t;

    // an handle to submit and discard task execution
    /// @brief retains a task and its unpublished queue reservation
    struct task_submission_handle {
        /// @brief constructs an empty unpublished task handle
        task_submission_handle() = default;

        // queue task for parallel execution
        /// @brief publishes this previously reserved task for execution
        void submit() { executor_->submit_task(*this); }
        // discard unsubmitted tasks. task execution is not preempted, hence submitted tasks cannot be discarded
        /// @brief reclaims an unpublished task and clears its reservation handle
        void discard() {
            if (*this) executor_->deallocate_task(*this);
        }
        explicit operator bool() const { return task_ != nullptr; }
        /// @brief returns this task submission handle for reservation operations
        task_submission_handle* operator->() noexcept { return this; }
        /// @brief returns the context owning this reserved task
        int context_id() const { return task_->owner_id(); }
       private:
        friend struct threaded_executor_impl;
        friend struct execution_context;
        /// @brief binds an allocated callable to its executor before queue reservation
        task_submission_handle(threaded_executor_impl* executor, task_handle* task) :
            executor_(executor), task_(task) { }

        threaded_executor_impl* executor_ = nullptr;         // executor controlling this task
        task_handle* task_ = nullptr;                        // the actual unit of work
        context_type::task_buffer_pointer data_ = nullptr;   // memory location storing this task
    };

    /// @brief starts a pool with detected available parallelism
    threaded_executor_impl() : threaded_executor_impl(fdapde::available_concurrency()) { }
    /// @brief starts background workers and reserves one context for the caller
    explicit threaded_executor_impl(size_type size) :
        stealing_policy_(std::max<size_type>(2, validate_size_(size))),
        scheduling_policy_(std::max<size_type>(1, size - 1)),
        size_(size),
        n_workers_(std::max<size_type>(1, size - 1)),   // retain asynchronous progress at P=1
        init_latch_(n_workers_ + 1),
        active_(true),
        idle_cnt_(0),
        epoch_(0),
        join_cnt_(0),
        active_join_cnt_(0) {
        // affinity is best-effort, including on unsupported systems
        std::vector<int> cpus;
        try {
            if (cpu_affinity_enabled) cpus = allowed_cpus();
        } catch (...) {
            // no affinity capability available on the hosting OS
        }
        contexts_.reserve(n_workers_ + 1);
        workers_.reserve(n_workers_);
        for (size_type i = 0; i <= n_workers_; ++i) {
            contexts_.push_back(std::make_unique<context_type>(static_cast<int>(i), i < n_workers_));
        }
        for (size_type i = 0; i < n_workers_; ++i) {
            try {
                workers_.emplace_back([this, i, &cpus] {
                    try {
                        // pin cpus before startup barrier
                        if (!cpus.empty()) bind({cpus[i % cpus.size()]});
                    } catch (...) {
                        // retain OS scheduling on failure
                    }
                    worker_loop_(static_cast<int>(i));
                });
            } catch (...) {
                active_.store(false, std::memory_order_release);
                // release started workers when the remaining threads could not be created
                init_latch_.count_down(n_workers_ + 1 - i);
                for (auto& worker : workers_) worker.join();
                throw;
            }
        }
        init_latch_.arrive_and_wait();
    }
    /// @brief returns the configured parallelism
    size_type size() const { return size_; }
    /// @brief reports whether background workers should continue running
    bool is_active() const { return active_.load(std::memory_order_acquire); }

    // submits f(args...) asynchronously and returns its result through a future
    /// @brief submits a bound callable and transports its result or exception through a future
    template <typename F, typename... Args>
        requires(std::is_invocable_v<std::decay_t<F>&, std::decay_t<Args>&...>)
    [[nodiscard]] auto async(F&& f, Args&&... args) {
        using ret_t = std::invoke_result_t<std::decay_t<F>&, std::decay_t<Args>&...>;
        using args_t = std::tuple<std::decay_t<Args>...>;
        auto task = [f_ = std::forward<F>(f), args_ = args_t(std::forward<Args>(args)...)]() mutable -> ret_t {
            return std::apply([&f_](auto&... stored_args) -> ret_t { return std::invoke(f_, stored_args...); }, args_);
        };
        auto packaged_task = std::make_shared<std::packaged_task<ret_t()>>(std::move(task));
        auto result = packaged_task->get_future();
        // submit task for execution
        auto handle = allocate_task(this_thread_id(), [packaged_task]() { (*packaged_task)(); });
        try {
            handle.submit();
        } catch (...) {
            handle.discard();
            throw;
        }
        return result;
    }
    // submits f(args...) without a result or an implicit wait
    /// @brief submits a bound callable for asynchronous execution
    template <typename F, typename... Args>
        requires(
          std::is_invocable_v<std::decay_t<F>&, std::decay_t<Args>&...> &&
          std::is_copy_constructible_v<std::decay_t<F>> && (std::is_copy_constructible_v<std::decay_t<Args>> && ...))
    void execute(F&& f, Args&&... args) {
        using args_t = std::tuple<std::decay_t<Args>...>;
        auto task = [f_ = std::forward<F>(f), args_ = args_t(std::forward<Args>(args)...)]() mutable {
            std::apply([&f_](auto&... stored_args) { std::invoke(f_, stored_args...); }, args_);
        };
        // submit task for execution
        auto handle = allocate_task(this_thread_id(), std::move(task));
        try {
            handle.submit();
        } catch (...) {
            handle.discard();
            throw;
        }
    }

    /// @brief executes an internal algorithm descriptor on this pool
    template <typename Task, typename... Args>
        requires(requires(Task task, threaded_executor_impl* executor, Args... args) { task.run(executor, args...); })
    auto execute(Task&& task, Args&&... args) {
        return task.run(this, std::forward<Args>(args)...);
    }

    // allocate task storage and reserve queue resources without publication
    /// @brief allocates a callable and reserves execution capacity before publication
    template <typename Task> task_submission_handle allocate_task(int context_id, Task&& task) {
        if (context_id < 0) { context_id = scheduling_policy_.pick(); }   // external thread
        auto* pointer = task_pool_.allocate(std::forward<Task>(task));
        try {
            task_submission_handle handle(this, pointer);
            contexts_[context_id]->reserve_task(handle);
            return handle;
        } catch (...) {
            task_pool_.deallocate(pointer);
            throw;
        }
    }
    // discard an unpublished task, release its reservation, and clear this handle
    /// @brief reclaims an unpublished callable and its queue reservation
    void deallocate_task(task_submission_handle& task) {
        if (!task) return;
        const auto discarded = std::exchange(task, {});
        contexts_[discarded.context_id()]->cancel_task(discarded);
        task_pool_.deallocate(discarded.task_);
    }
    // publish through the owning context, then notify the runtime
    /// @brief publishes reserved tasks and notifies workers once per submission
    void submit_task(task_submission_handle task) {
        contexts_[task.context_id()]->publish_task(task);
        epoch_.fetch_add(1, std::memory_order_release);
        notify();
    }
    // publish consecutive tasks for each context issuing a single notification at completed batch submission
    /// @brief publishes reserved tasks and notifies workers once per submission
    void submit_task(const std::vector<task_submission_handle>& tasks) {
        for (std::size_t first = 0; first < tasks.size();) {
            const int context_id = tasks[first].context_id();
            auto last = first + 1;
            while (last < tasks.size()) {
                if (tasks[last].context_id() != context_id) break;
                ++last;
            }
            contexts_[context_id]->publish_task(tasks.begin() + first, tasks.begin() + last);
            first = last;
        }
        epoch_.fetch_add(1, std::memory_order_release);
        notify_all();
    }
   private:
    /// @brief temporarily owns the reserved caller context and restores external thread identity
    struct attachment_handle {
        /// @brief acquires the reserved caller context when available
        explicit attachment_handle(threaded_executor_impl* executor) noexcept : executor_(nullptr) {
            if (tls_worker_id == external_thread && executor->contexts_.back()->try_attach()) {
                executor_ = executor;
                tls_worker_id = static_cast<int>(executor_->n_workers_);
            }
        }
        /// @brief disables copying or moving of owned execution storage
        attachment_handle(const attachment_handle&) = delete;
        /// @brief disables assignment of owned execution storage
        attachment_handle& operator=(const attachment_handle&) = delete;
        /// @brief releases the caller context and restores external thread identity
        ~attachment_handle() { detach(); }
        /// @brief reports whether this handle owns the reserved caller context
        bool attached() const noexcept { return executor_ != nullptr; }
        /// @brief reports whether this handle owns the caller context
        explicit operator bool() const noexcept { return attached(); }
        /// @brief releases ownership of the reserved caller context
        void detach() noexcept {
            if (!executor_) return;
            tls_worker_id = external_thread;
            executor_->contexts_.back()->detach();
            executor_ = nullptr;
        }
       private:
        threaded_executor_impl* executor_;
    };
    /// @brief counts parked waiters with a handshake before their predicate is observed
    struct wait_guard {
        /// @brief initializes parked waiters with a handshake before their predicate is observed
        explicit wait_guard(std::atomic<size_type>& count) : count_(count) {
            ++count_;
            std::atomic_thread_fence(std::memory_order_seq_cst);
        }
        /// @brief releases the parked-waiter registration
        ~wait_guard() { --count_; }
       private:
        std::atomic<size_type>& count_;
    };
   public:
    // called by an externally managed thread to check if it can join this executor
    /// @brief acquires the reserved caller context when it is available
    [[nodiscard]] attachment_handle try_attach() noexcept { return attachment_handle(this); }
    // blocks external callers until all published work has completed
    /// @brief waits for global task completion and rejects calls from executor workers
    void join() {
        fdapde_strong_assert(
          this_thread_id() == main_thread_id, std::logic_error,
          "parallel_join cannot be called from an executor worker");
        std::unique_lock<std::mutex> lock(m_);
        wait_guard wait_on(join_cnt_);
        join_cv_.wait(lock, [&]() {
            const auto epoch = epoch_.load(std::memory_order_acquire);
            const bool empty = std::all_of(
              contexts_.begin(), contexts_.end(), [](const auto& context) { return !context->has_unfinished_tasks(); });
            return empty && epoch_.load(std::memory_order_acquire) == epoch;
        });
    }
    // attached callers execute work while cond() is true; external callers wait passively
    /// @brief helps execute work while the group predicate remains true or waits as an external caller
    template <typename Condition>
        requires(requires(Condition cond) {
            { cond() } -> std::convertible_to<bool>;
        })
    void active_join(int context_id, Condition&& cond) {
        if (context_id < 0) {
            // external detatched threads are put in passive join
            std::unique_lock<std::mutex> lock(m_);
            wait_guard wait_on(active_join_cnt_);
            join_cv_.wait(lock, [&] { return !cond(); });
            return;
        }
        // exponential backoff strategy
        using clock = std::chrono::steady_clock;
        auto retention = std::chrono::microseconds(min_spin_retention_us);
        while (cond()) {
            if (try_execute_one_(context_id)) continue;
            const auto deadline = clock::now() + retention;
            int pauses = 1;
            while (cond()) {
                for (int i = 0; i < pauses; ++i) {
                    if (!cond()) return;
                    cpu_yield_();   // lightweight CPU yield
                }
                if (!cond()) return;
                if (try_execute_one_(context_id)) {
                    retention = std::min(retention * backoff_factor, std::chrono::microseconds(max_spin_retention_us));
                    break;
                }
                if (clock::now() >= deadline) {
                    retention = std::max(retention / backoff_factor, std::chrono::microseconds(min_spin_retention_us));
                    // no work found during backoff, go idle
                    std::unique_lock<std::mutex> lock(m_);
                    wait_guard wait_on(active_join_cnt_);
                    join_cv_.wait(lock, [&] { return !cond() || are_ready_tasks_available_() || !is_active(); });
                    break;
                }
                pauses = std::min(pauses * backoff_factor, max_spin_pauses);
            }
        }
    }
    // stops background threads; queued tasks may remain unexecuted
    /// @brief stops and joins background workers without draining queued tasks
    void stop() {
        std::unique_lock<std::mutex> lock(m_);
        active_.store(false, std::memory_order_release);
        lock.unlock();
        join_cv_.notify_all();
        idle_cv_.notify_all();
        for (auto& worker : workers_) worker.join();
    }
    /// @brief wakes one idle worker and collaborative waiters after publication
    void notify() noexcept { notify_(false); }   // wakes one idle worker and all collaborative join waiters
    /// @brief wakes all idle workers and collaborative waiters after batch publication
    void notify_all() noexcept { notify_(true); }   // wakes all idle workers and all collaborative join waiters
    // wakes both global and collaborative join waiters
    /// @brief wakes global and collaborative waiters after completion without losing notifications
    void notify_all_join() noexcept {
        std::atomic_thread_fence(std::memory_order_seq_cst);
        if (join_cnt_.load() != 0 || active_join_cnt_.load() != 0) {
            // handshake with wait predicates under m_ to prevent a missed completion wakeup
            std::lock_guard<std::mutex> lock(m_);
            join_cv_.notify_all();
        }
    }
    /// @brief releases background workers before executor storage is destroyed
    ~threaded_executor_impl() {
        if (is_active()) stop();
    }
   private:
    /// @brief validates internal pool size before worker arithmetic
    static size_type validate_size_(size_type size) {
        fdapde_assert(
          size > 0 && size <= static_cast<size_type>(std::numeric_limits<int>::max()), std::invalid_argument,
          "executor size must be a positive int");
        return size;
    }
    /// @brief runs one local or stolen task and releases its capture and completion count
    bool try_execute_one_(int context_id) noexcept {
        auto task = contexts_[context_id]->try_fetch_task();
        if (!task) {
            // no local work, try steal work from other context
            task = stealing_policy_.pick(context_id, [&](int victim) { return contexts_[victim]->try_steal(); });
        }
        if (!task) return false;   // no work found, return

        // run assigned task and update task accounting once finished
        (*task)->run();
        const int owner_id = (*task)->owner_id();
        task_pool_.deallocate(*task);
        if (contexts_[owner_id]->complete_task()) { notify_all_join(); }
        return true;
    }
    // loop executed by executor managed physical-threads
    /// @brief steals and runs tasks with bounded spinning before parking on ready work
    void worker_loop_(int context_id) {
        tls_worker_id = context_id;
        init_latch_.arrive_and_wait();
        using clock = std::chrono::steady_clock;
        auto retention = std::chrono::microseconds(min_spin_retention_us);
        while (is_active()) {
            // try fetch work, possible do a stealing attempt
            if (try_execute_one_(context_id)) continue;
            // no work found, exponential backoff
            const auto deadline = clock::now() + retention;
            int pauses = 1;
            while (is_active()) {
                for (int i = 0; i < pauses; ++i) { cpu_yield_(); }   // lightweight CPU yield
                if (try_execute_one_(context_id)) {
                    retention = std::min(retention * backoff_factor, std::chrono::microseconds(max_spin_retention_us));
                    break;
                }
                if (clock::now() >= deadline) {
                    retention = std::max(retention / backoff_factor, std::chrono::microseconds(min_spin_retention_us));
                    // no work found during backoff, go idle
                    std::unique_lock<std::mutex> lock(m_);
                    wait_guard wait_on(idle_cnt_);
                    idle_cv_.wait(lock, [&] { return are_ready_tasks_available_() || !is_active(); });
                    break;
                }
                pauses = std::min(pauses * backoff_factor, max_spin_pauses);
            }
        }
    }
    /// @brief wakes idle workers and collaborative waiters using their predicate mutex
    void notify_(bool notify_all) noexcept {
        std::atomic_thread_fence(std::memory_order_seq_cst);
        const bool idle = idle_cnt_.load() != 0;
        const bool join = active_join_cnt_.load() != 0;
        if (!idle && !join) return;
        std::lock_guard<std::mutex> lock(m_);
        if (idle) { notify_all ? idle_cv_.notify_all() : idle_cv_.notify_one(); }
        if (join) join_cv_.notify_all();
    }
    /// @brief reports whether any execution context holds runnable work
    bool are_ready_tasks_available_() const {
        return std::any_of(
          contexts_.begin(), contexts_.end(), [](const auto& context) { return context->has_ready_tasks(); });
    }
    // hints the processor to relax during spinning. emits machine-level yield/pause instructions
    /// @brief hints that the processor may relax during a bounded spin interval
    void cpu_yield_() noexcept {
#if defined(__has_builtin)
#    if __has_builtin(__builtin_ia32_pause)
        __builtin_ia32_pause();
#    elif __has_builtin(__builtin_arm_yield)
        __builtin_arm_yield();
#    elif defined(__riscv_zihintpause) && __has_builtin(__builtin_riscv_pause)
        __builtin_riscv_pause();
#    elif defined(__aarch64__) || (defined(__arm__) && defined(__ARM_ARCH) && __ARM_ARCH >= 7)
        __asm__ __volatile__("yield");
#    elif defined(__riscv_zihintpause)
        __asm__ __volatile__("pause");
#    endif
#endif
    }

    allocator_type task_pool_;                 // shared physical memory holding tasks
    stealing_policy stealing_policy_;          // randomized stealing functor
    scheduling_policy scheduling_policy_;      // round-robin scheduler
    std::vector<context_pointer> contexts_;    // background execution contexts
    std::vector<std::thread> workers_;         // physical threads internally managed by the executor
    const size_type size_;                     // configured parallelism, n_workers_ != size for size = 1
    const size_type n_workers_;                // number of internal threads
    std::latch init_latch_;                    // startup barrier
    std::atomic<bool> active_;                 // asserted false during executor shutdown
    std::atomic<size_type> idle_cnt_;          // background workers waiting for work
    std::atomic<std::uint64_t> epoch_;         // publication epoch, incremented once per submission call
    std::atomic<size_type> join_cnt_;          // external callers waiting in join()
    std::atomic<size_type> active_join_cnt_;   // callers parked in active_join()
    std::condition_variable idle_cv_;
    std::condition_variable join_cv_;
    std::mutex m_;
};

/// @brief provides access to the process-lifetime worker pool
struct threaded_executor {
    // intentionally never destroyed to keep background threads from accessing destroyed executor storage
    /// @brief starts the process-lifetime executor once, optionally capping its configured parallelism
    static auto& instance(int max_threads = std::numeric_limits<int>::max()) {
        static std::aligned_storage_t<sizeof(threaded_executor_impl), alignof(threaded_executor_impl)> storage;
        static threaded_executor_impl* exec = [max_threads] {
            std::lock_guard<std::mutex> lock(threaded_executor_global_state.m_);
            const int threads = std::min(max_threads, threaded_executor_global_state.num_threads());
            auto* result = new (&storage) threaded_executor_impl(threads);
            threaded_executor_global_state.num_threads_ = threads;
            threaded_executor_global_state.set_ready_();
            return result;
        }();
        return *exec;
    }
};

}   // namespace internals

/// @brief detects algorithm descriptors accepting an executor and arguments
template <typename Task, typename... Args> struct is_runnable_task {
    static constexpr bool value = requires(Task task, internals::threaded_executor_impl* executor, Args... args) {
        { task.run(executor, args...) } -> std::same_as<void>;
    };
};
template <typename Task, typename... Args>
static constexpr bool is_runnable_task_v = is_runnable_task<Task, Args...>::value;

// set total available parallelism
/// @brief sets positive total parallelism before runtime initialization
inline void parallel_set_num_threads(int num_threads) {
    internals::threaded_executor_global_state.set_num_threads(num_threads);
}
/// @brief enables best-effort background affinity before runtime initialization
inline void parallel_set_affinity(bool enabled) { internals::threaded_executor_global_state.set_affinity(enabled); }

// get total available parallelism
/// @brief returns configured total parallelism
inline int parallel_get_num_threads() { return internals::threaded_executor_global_state.num_threads(); }

/// @brief starts the executor within a positive concurrency cap or reuses an existing pool that respects it
inline int parallel_initialize(int max_threads) {
    fdapde_strong_assert(max_threads > 0, std::invalid_argument, "parallelism limit must be positive");
    const auto threads = internals::threaded_executor::instance(max_threads).size();
    fdapde_strong_assert(
      threads <= static_cast<std::size_t>(max_threads), std::logic_error,
      "existing executor exceeds the requested parallelism limit");
    return static_cast<int>(threads);
}
// executes a callable object asynchronously
/// @brief submits a copyable callable without waiting for its completion
template <typename F, typename... Args>
    requires(
      std::is_invocable_v<std::decay_t<F>&, std::decay_t<Args>&...> && std::is_copy_constructible_v<std::decay_t<F>> &&
      (std::is_copy_constructible_v<std::decay_t<Args>> && ...))
void parallel_execute(F&& f, Args&&... args) {
    internals::threaded_executor::instance().execute(std::forward<F>(f), std::forward<Args>(args)...);
}
// executes a callable object asynchronously returning a std::future object for synchronization and result retrieval
/// @brief submits a callable and returns a future carrying its result or exception
template <typename F, typename... Args>
    requires(std::is_invocable_v<std::decay_t<F>&, std::decay_t<Args>&...>)
auto parallel_async(F&& f, Args&&... args) {
    return internals::threaded_executor::instance().async(std::forward<F>(f), std::forward<Args>(args)...);
}
// explicitly joins the executor until all work is completed
/// @brief waits for all submitted work from outside executor workers
inline void parallel_join() { internals::threaded_executor::instance().join(); }

}   // namespace fdapde

#endif   // __FDAPDE_EXECUTION_THREADED_EXECUTOR_H__
