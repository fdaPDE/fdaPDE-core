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
struct threaded_executor;
struct threaded_executor_global_state_t {
    threaded_executor_global_state_t() = default;
    friend threaded_executor;

    // setters
    void set_num_threads(int num_threads) {
        std::lock_guard<std::mutex> lock(m_);
        fdapde_assert(num_threads > 0, "number of requested threads must be positive");
        fdapde_assert(!is_ready_     , "thread count cannot change after executor initialization");
        num_threads_ = num_threads;
    }
    // getters
    int num_threads() const { return num_threads_.load(std::memory_order_relaxed); }
   private:
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
struct random_stealing_policy {
    explicit random_stealing_policy(int size) : size_(size) { }
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
struct round_robin_scheduling_policy {
    explicit round_robin_scheduling_policy(std::size_t n) : size_(n) { }
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
    struct task_submission_handle {
        task_submission_handle() = default;
      
        // queue task for parallel execution
        void submit() { executor_->submit_task(*this); }
        // discard unsubmitted tasks. task execution is not preempted, hence submitted tasks cannot be discarded
        void discard() {
            if (*this) executor_->deallocate_task(*this);
        }
        // getters
        explicit operator bool() const { return task_ != nullptr; }
        task_submission_handle* operator->() noexcept { return this; }
        int context_id() const { return task_->owner_id(); }
       private:
        friend struct threaded_executor_impl;
        friend struct execution_context;
        task_submission_handle(threaded_executor_impl* executor, task_handle* task) :
            executor_(executor), task_(task) { }

        threaded_executor_impl* executor_ = nullptr;         // executor controlling this task
        task_handle* task_ = nullptr;                        // the actual unit of work
        context_type::task_buffer_pointer data_ = nullptr;   // memory location storing this task
    };

    threaded_executor_impl() : threaded_executor_impl(fdapde::available_concurrency()) { }
    explicit threaded_executor_impl(size_type size) :
        stealing_policy_(size + 1),
        scheduling_policy_(size),
        n_workers_(size),
        init_latch_(n_workers_ + 1),
        active_(true),
        idle_cnt_(0),
        epoch_(0),
        join_cnt_(0),
        active_join_cnt_(0) {
        contexts_.reserve(n_workers_ + 1);
        workers_.reserve(n_workers_);
        for (size_type i = 0; i <= n_workers_; ++i) {
            contexts_.push_back(std::make_unique<context_type>(static_cast<int>(i), i < n_workers_));
        }
        for (size_type i = 0; i < n_workers_; ++i) {
            try {
                workers_.emplace_back([this, i] { worker_loop_(static_cast<int>(i)); });
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
    size_type size() const { return n_workers_; }
    bool is_active() const { return active_.load(std::memory_order_acquire); }

    // submits f(args...) asynchronously and returns its result through a future
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

    // allocate task storage and reserve queue resources without publication
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
    void deallocate_task(task_submission_handle& task) {
        if (!task) return;
        const auto discarded = std::exchange(task, {});
        contexts_[discarded.context_id()]->cancel_task(discarded);
        task_pool_.deallocate(discarded.task_);
    }
    // publish through the owning context, then notify the runtime
    void submit_task(task_submission_handle task) {
        contexts_[task.context_id()]->publish_task(task);
        epoch_.fetch_add(1, std::memory_order_release);
        notify();
    }
    // publish consecutive tasks for each context issuing a single notification at completed batch submission
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
    // RAII ownership of the caller's reserved context
    struct attachment_handle {
        explicit attachment_handle(threaded_executor_impl* executor) noexcept : executor_(nullptr) {
            if (tls_worker_id == external_thread && executor->contexts_.back()->try_attach()) {
                executor_ = executor;
                tls_worker_id = static_cast<int>(executor_->n_workers_);
            }
        }
        attachment_handle(const attachment_handle&) = delete;
        attachment_handle& operator=(const attachment_handle&) = delete;
        ~attachment_handle() { detach(); }
        bool attached() const noexcept { return executor_ != nullptr; }
        explicit operator bool() const noexcept { return attached(); }
        void detach() noexcept {
            if (!executor_) return;
            tls_worker_id = external_thread;
            executor_->contexts_.back()->detach();
            executor_ = nullptr;
        }
       private:
        threaded_executor_impl* executor_;
    };
    struct wait_guard {
        explicit wait_guard(std::atomic<size_type>& count) : count_(count) {
            ++count_;
            std::atomic_thread_fence(std::memory_order_seq_cst);
        }
        ~wait_guard() { --count_; }
       private:
        std::atomic<size_type>& count_;
    };
   public:
    // called by an externally managed thread to check if it can join this executor
    [[nodiscard]] attachment_handle try_attach() noexcept { return attachment_handle(this); }
    // blocks external callers until all published work has completed
    void join() {
        if (tls_worker_id != external_thread) return;
        std::unique_lock<std::mutex> lock(m_);
        wait_guard wait_on(join_cnt_);
        join_cv_.wait(lock, [&]() {
            const auto epoch = epoch_.load(std::memory_order_acquire);
            const bool empty = std::all_of(contexts_.begin(), contexts_.end(), [](const auto& context) {
                return !context->has_unfinished_tasks();
            });
            return empty && epoch_.load(std::memory_order_acquire) == epoch;
        });
    }
    // attached callers execute work while cond() is true; external callers wait passively
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
                    retention = std::min(
                      retention * backoff_factor, std::chrono::microseconds(max_spin_retention_us));
                    break;
                }
                if (clock::now() >= deadline) {
                    retention = std::max(
                      retention / backoff_factor, std::chrono::microseconds(min_spin_retention_us));
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
    void stop() {
        std::unique_lock<std::mutex> lock(m_);
        active_.store(false, std::memory_order_release);
        lock.unlock();
        join_cv_.notify_all();
        idle_cv_.notify_all();
        for (auto& worker : workers_) worker.join();
    }
    void notify() noexcept { notify_(false); }      // wakes one idle worker and all collaborative join waiters
    void notify_all() noexcept { notify_(true); }   // wakes all idle workers and all collaborative join waiters
    // wakes both global and collaborative join waiters
    void notify_all_join() noexcept {
        std::atomic_thread_fence(std::memory_order_seq_cst);
        if (join_cnt_.load() != 0 || active_join_cnt_.load() != 0) {
            // handshake with wait predicates under m_ to prevent a missed completion wakeup
            std::lock_guard<std::mutex> lock(m_);
            join_cv_.notify_all();
        }
    }
    ~threaded_executor_impl() {
        if (is_active()) stop();
    }
   private:
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
                    retention = std::min(
                      retention * backoff_factor, std::chrono::microseconds(max_spin_retention_us));
                    break;
                }
                if (clock::now() >= deadline) {
                    retention = std::max(
                      retention / backoff_factor, std::chrono::microseconds(min_spin_retention_us));
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
    void notify_(bool notify_all) noexcept {
        std::atomic_thread_fence(std::memory_order_seq_cst);
        const bool idle = idle_cnt_.load() != 0;
        const bool join = active_join_cnt_.load() != 0;
        if (!idle && !join) return;
        std::lock_guard<std::mutex> lock(m_);
        if (idle) { notify_all ? idle_cv_.notify_all() : idle_cv_.notify_one(); }
        if (join) join_cv_.notify_all();
    }
    // check if there is some executable work in the executor
    bool are_ready_tasks_available_() const {
        return std::any_of(contexts_.begin(), contexts_.end(), [](const auto& context) {
            return context->has_ready_tasks();
        });
    }
    // hints the processor to relax during spinning. emits machine-level yield/pause instructions
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

struct threaded_executor {
    // intentionally never destroyed to keep background threads from accessing destroyed executor storage
    static auto& instance() {
        static std::aligned_storage_t<sizeof(threaded_executor_impl), alignof(threaded_executor_impl)> storage;
        static threaded_executor_impl* exec = [] {
            std::lock_guard<std::mutex> lock(threaded_executor_global_state.m_);
            auto* result = new (&storage) threaded_executor_impl(threaded_executor_global_state.num_threads());
            threaded_executor_global_state.set_ready_();
            return result;
        }();
        return *exec;
    }
};

}   // namespace internals

// public API

template <typename Task, typename... Args> struct is_runnable_task {
    static constexpr bool value = requires(Task task, internals::threaded_executor_impl* executor, Args... args) {
        { task.run(executor, args...) } -> std::same_as<void>;
    };
};
template <typename Task, typename... Args>
static constexpr bool is_runnable_task_v = is_runnable_task<Task, Args...>::value;

// set number of worker threads
inline void parallel_set_num_threads(int num_threads) {
    internals::threaded_executor_global_state.set_num_threads(num_threads);
}
// get number of worker threads
inline int parallel_get_num_threads() { return internals::threaded_executor_global_state.num_threads(); }
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

#endif   // __FDAPDE_EXECUTION_THREADED_EXECUTOR_H__
