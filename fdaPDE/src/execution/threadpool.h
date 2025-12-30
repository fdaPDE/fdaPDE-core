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

inline thread_local int tls_worker_id = -1;   // worker logical index

}   // namespace internals

// returns the index of the worker with the largest job count
struct max_load_stealing {
    max_load_stealing(int) { }

    template <typename WorkLoadT> std::optional<int> pick(const WorkLoadT& workload) {
        const int n = workload.size();
        int i = 0, j = workload[0].load(std::memory_order_acquire);   // current optimum and optimal value
        for (int k = 1; k < n; k++) {
            int m = workload[k].load(std::memory_order_acquire);
            if (m > j) {
                i = k;
                j = m;
            }
        }
        if (j == 0) { return std::nullopt; }
        return i;
    };
};

// returns a random worker among the ones having at least one job
struct random_stealing {
    explicit random_stealing(int n) : rng_(std::random_device {}()) { idxs_.resize(n); }

    template <typename WorkLoadT> std::optional<int> pick(const WorkLoadT& workload) {
        int i = 0;
        for (int k = 0; k < workload.size(); k++) {
            if (workload[k].load(std::memory_order_acquire) > 0) { idxs_[i++] = k; }
        }
        if (i == 0) { return std::nullopt; }
        if (i == 1) { return idxs_[0]; }
        std::uniform_int_distribution<> random_int(0, i - 1);
        return idxs_[random_int(rng_)];
    }
   private:
    std::vector<int> idxs_;
    mutable std::mt19937 rng_;
};

// returns a random worker among the workers in the top half of the busiest ones
struct top_half_random_stealing {
    explicit top_half_random_stealing(int n) : rng_(std::random_device {}()) { idxs_.resize(n); }

    template <typename WorkLoadT> std::optional<int> pick(const WorkLoadT& workload) {
        int i = 0;
        for (int k = 0; k < workload.size(); k++) {
            int m = workload[k].load(std::memory_order_acquire);
            if (m > 0) {
                auto& [worker, load] = idxs_[i];
                worker = k;
                load = m;
                ++i;
            }
        }
        if (i == 0) { return std::nullopt; }
        // sort in decreasing number of jobs order
        std::sort(idxs_.begin(), idxs_.begin() + i, [](std::pair<int, int>& a, std::pair<int, int>& b) {
            return a.second > b.second;
        });
        if (i < 4) {
            return idxs_[0].first;
        } else {
            std::uniform_int_distribution<> random_int(0, i / 2);
            return idxs_[random_int(rng_)].first;
        }
    }
   private:
    std::vector<std::pair<int, int>> idxs_;   // (worker index, worker queue size)
    mutable std::mt19937 rng_;
};

// returns the next worker index in round-robin order
struct round_robin_scheduling {
    round_robin_scheduling(int) : idx_(0) { }

    template <typename WorkLoadT> int pick(const WorkLoadT& workload) {
        int next = idx_.load(std::memory_order_acquire);
        idx_.store((idx_ + 1) % workload.size(), std::memory_order_release);
        return next;
    }
   private:
    std::atomic<int> idx_;
};

// returns the worker with the smallest job count
struct least_loaded_scheduling {
    least_loaded_scheduling(int) { }

    template <typename WorkLoadT> int pick(const WorkLoadT& workload) {
        const int n = workload.size();
        int i = 0, j = workload[0].load(std::memory_order_acquire);
        for (int k = 1; k < n; k++) {
            int m = workload[k].load(std::memory_order_acquire);
            if (m < j) {
                i = k;
                j = m;
            }
        }
        return i;
    }
};

namespace internals {

// implementation of the static-sized Chase-Lev circular buffer queue:
// * "Chase, D. and Lev, Y. (2005). Dynamic circular work-stealing deque. In Proceedings of the seventeenth annual
//    ACM symposium on Parallelism in algorithms and architectures (pp. 21-28)."
// * "Le, N. M., Pop, A., Cohen, A., \and Zappa, F. (2013). Correct and efficient work-stealing for weak memory
//    models. ACM SIGPLAN Notices, 48(8), 69-80."
template <typename T>
//    requires(std::is_trivially_copyable_v<T> && sizeof(T) <= sizeof(void*) && std::atomic<T>::is_always_lock_free)
struct chase_lev_queue {
    // any instance of type T must be copy/move-able atomically (a copy/move operation can be performed by the compiler
    // issuing a single mov instruction). observe that pointers fall in this category

    // constructors
    chase_lev_queue() : chase_lev_queue(4096) { }
    explicit chase_lev_queue(std::int64_t capacity) :
        buffer_(capacity), capacity_(capacity), mask_(capacity - 1), bottom_(0), top_(0) {
        fdapde_assert(capacity > 0 && (capacity & (capacity - 1)) == 0);   // require capacity power of two
    }
    // avoid copy/move-semantic
    chase_lev_queue(const chase_lev_queue&) = delete;
    chase_lev_queue(chase_lev_queue&&) = delete;
    chase_lev_queue& operator=(const chase_lev_queue&) = delete;
    chase_lev_queue& operator=(chase_lev_queue&&) = delete;

    // returns first element of the container, nullopt if container empty. the element is removed
    // only owning thread pops from buffer's front
    std::optional<T> pop_front() {
        std::int64_t b = bottom_.load(std::memory_order_relaxed) - 1;
        bottom_.store(b, std::memory_order_relaxed);
        std::atomic_thread_fence(std::memory_order_seq_cst);
        std::int64_t t = top_.load(std::memory_order_relaxed);

        if (t <= b) {
            if (t == b) {
                // last queue element
                if (!top_.compare_exchange_strong(t, t + 1, std::memory_order_seq_cst, std::memory_order_relaxed)) {
                    // lost race with concurrent pop_back (stealing attempt)
                    bottom_.store(b + 1, std::memory_order_relaxed);
                    return std::nullopt;
                }
                bottom_.store(b + 1, std::memory_order_relaxed);
            }
            std::int64_t idx = b & mask_;
            return std::move(buffer_[idx]);   // atomic read
        } else {
            // empty queue
            bottom_.store(b + 1, std::memory_order_relaxed);
            return std::nullopt;
        }
    }

    // appends a copy of value to the end of the container. aborts if container full
    // only owning thread push at buffer's back
    template <typename T_>
        requires(std::is_convertible_v<T_, T>)
    bool push_front(T_ value) {
        std::int64_t b = bottom_.load(std::memory_order_relaxed);
        std::int64_t t = top_.load(std::memory_order_acquire);
        // abort if queue is full
        if (b >= t + capacity_ - 1) { return false; }
        // write
        std::int64_t idx = b & mask_;
        buffer_[idx] = std::move(value);
        std::atomic_thread_fence(std::memory_order_release);
        bottom_.store(b + 1, std::memory_order_relaxed);   // sync write
        return true;
    }
    // constructs a new element at the end of the container. aborts if container full
    // only owning thread push at buffer's back
    template <typename... Args>
        requires(std::is_constructible_v<T, Args...>)
    bool emplace_front(Args&&... args) {
        T value(std::forward<Args>(args)...);

        std::int64_t b = bottom_.load(std::memory_order_relaxed);
        std::int64_t t = top_.load(std::memory_order_acquire);
        // abort if queue is full
        if (b >= t + capacity_ - 1) { return false; }
        // write
        std::int64_t idx = b & mask_;
        buffer_[idx] = std::move(value);
        std::atomic_thread_fence(std::memory_order_release);
        bottom_.store(b + 1, std::memory_order_relaxed);   // sync write
        return true;
    }
    // returns last element of the container, nullopt if container empty. the element is removed
    // this method is invoked as a result of a work-stealing attempt
    std::optional<T> pop_back() {
        std::int64_t t = top_.load(std::memory_order_acquire);
        std::atomic_thread_fence(std::memory_order_seq_cst);
        std::int64_t b = bottom_.load(std::memory_order_acquire);
        // abort if queue is empty
        if (t >= b) { return std::nullopt; }
        // atomic read
        std::int64_t idx = t & mask_;
        // try to claim the element by incrementing top
        if (!top_.compare_exchange_strong(t, t + 1, std::memory_order_seq_cst, std::memory_order_relaxed)) {
            return std::nullopt;   // lost race
        }
        return std::move(buffer_[idx]);   // we move now, after having win the CAS
    }
    // observers
    bool empty() const {
        std::int64_t t = top_.load(std::memory_order_acquire), b = bottom_.load(std::memory_order_acquire);
        return t >= b;
    }
   private:
    std::vector<T> buffer_;
    const std::int64_t capacity_;   // maximum buffer size (as power of 2)
    const int mask_;                // 0b(capacity_ - 1), allows fast modulo capacity_

    alignas(64) std::atomic<std::int64_t> bottom_;   // number of performed pushes
    alignas(64) std::atomic<std::int64_t> top_;      // number of performed pops
};

// unbounded lock-free MPSC queue
template <typename T>
    requires(std::is_move_constructible_v<T>)
class mpsc_queue {
    struct node_t {
        std::atomic<node_t*> next;
        T data;

        node_t() : next(nullptr) { }
        template <typename T_>
            requires(std::is_constructible_v<T, T_>)
        node_t(T_&& data_) : next(nullptr), data(std::forward<T_>(data_)) { }
    };
   public:
    mpsc_queue() : head_(nullptr), tail_(nullptr) {
        node_t* root = new node_t();
        head_ = root;
        tail_.store(root, std::memory_order_relaxed);
    }

    // multi-producer push
    void push(T&& value) {
        node_t* node = new node_t(std::move(value));
        node_t* prev = tail_.exchange(node, std::memory_order_acq_rel);
        prev->next.store(node, std::memory_order_release);
        return;
    }
    // single consumer pop
    std::optional<T> pop() {
        node_t* next = head_->next.load(std::memory_order_acquire);
        if (next == nullptr) { return std::nullopt; }

        T value = std::move(next->data);
        delete head_;
        head_ = next;
        return std::move(value);
    }

    ~mpsc_queue() {
        node_t* next = head_;
        while (head_->next) {
            next = head_->next;
            delete (head_);
        }
        delete (next);
    }
   private:
    node_t* head_;
    std::atomic<node_t*> tail_;
};

}   // namespace internals

template <typename SchedulingStrategy = round_robin_scheduling, typename StealingStrategy = random_stealing>
class ThreadPool {
    using scheduling_t = std::decay_t<SchedulingStrategy>;
    using stealing_t = std::decay_t<StealingStrategy>;
    static constexpr int buffer_size = 4096;

    // array of std::atomic<T>
    template <typename T> struct atomic_array {
       private:
        // a copiable/movable wrapper around a std::atomic<T> object
        struct atomic_t {
            atomic_t() : v_() { }
            atomic_t(const std::atomic<T>& v) : v_(v.load()) { }
            atomic_t(const atomic_t& other) : v_(other.v_.load()) { }
            atomic_t& operator=(const atomic_t& other) {
                v_ = other.v_.load();
                return *this;
            }
            void store(const T& value, std::memory_order order = std::memory_order_seq_cst) { v_.store(value, order); }
            T load(std::memory_order order = std::memory_order_seq_cst) const noexcept { return v_.load(order); }
            const std::atomic<T>& operator*() const { return v_; }
            std::atomic<T>& operator*() { return v_; }
           private:
            std::atomic<T> v_;
        };
        std::vector<atomic_t> data_;
       public:
        explicit atomic_array(int size) : data_(size) { }
        template <typename T_>
            requires(std::is_convertible_v<T, T_>)
        atomic_array(int size, T_ value) : data_(size) {
            for (int i = 0; i < size; ++i) { data_[i].store(value); }
        }
        // observers
        int size() const { return data_.size(); }
        const std::atomic<T>& operator[](int i) const {
            fdapde_assert(i >= 0 && i < size());
            return *data_[i];
        }
        // modifiers
        std::atomic<T>& operator[](int i) {
            fdapde_assert(i >= 0 && i < size());
            return *data_[i];
        }
    };

    struct task_t {
        using fn_t = void (*)(void*);
        static constexpr int buffer_size = 64;

        task_t() noexcept = default;
        // copy semantic
        task_t(const task_t&) = delete;
        task_t& operator=(const task_t&) = delete;
        // move semantic
        task_t(task_t&& other) noexcept :
            fn_(std::exchange(other.fn_, nullptr)),
            rm_(std::exchange(other.rm_, nullptr)),
            mv_(std::exchange(other.mv_, nullptr)),
            sb_(std::exchange(other.sb_, 0)) {
            if (sb_) {
                if (mv_) { mv_(storage_.buff_, other.storage_.buff_); }
            } else {
                storage_.data_ = std::exchange(other.storage_.data_, nullptr);
            }
        }
        task_t& operator=(task_t&& other) noexcept {
            if (this == &other) return *this;
            if (rm_) { rm_(sb_ ? (void*)storage_.buff_ : storage_.data_); }   // clean-up to prevent leaks

            fn_ = std::exchange(other.fn_, nullptr);
            rm_ = std::exchange(other.rm_, nullptr);
            mv_ = std::exchange(other.mv_, nullptr);
            sb_ = std::exchange(other.sb_, 0);
            if (sb_) {
                if (mv_) { mv_(storage_.buff_, other.storage_.buff_); }
            } else {
                storage_.data_ = std::exchange(other.storage_.data_, nullptr);
            }
            return *this;
        }
        template <typename F>
            requires(!std::is_same_v<std::decay_t<F>, task_t> && std::is_invocable_v<F>)
        explicit task_t(F&& f) {
            using Fn = std::decay_t<F>;
            constexpr bool sb = sizeof(F) <= buffer_size && alignof(Fn) <= alignof(union U);
            sb_ = sb;
            if constexpr (sb) {
                // stack allocation for small task object
                new (storage_.buff_) Fn(std::forward<F>(f));
            } else {
                // if task cannot fit in small buffer, resort to heap allocation
                storage_.data_ = new Fn(std::forward<F>(f));
            }
            // type-erased function handlers
            fn_ = [](void* ptr) { (*reinterpret_cast<Fn*>(ptr))(); };
            rm_ = [](void* ptr) noexcept {
                if (ptr == nullptr) { return; }
                (*reinterpret_cast<Fn*>(ptr)).~Fn();
                if constexpr (!sb) { delete (reinterpret_cast<Fn*>(ptr)); }   // free resources
            };
            mv_ = [](void* dst, void* src) noexcept {
                new (dst) Fn(std::move(*reinterpret_cast<Fn*>(src)));
                (*reinterpret_cast<Fn*>(src)).~Fn();
            };
        }
        // invoke
        void operator()() { fn_(sb_ ? (void*)storage_.buff_ : storage_.data_); }
        void operator()() const { fn_(sb_ ? (void*)storage_.buff_ : storage_.data_); }
        ~task_t() {
            if (rm_) { rm_(sb_ ? (void*)storage_.buff_ : storage_.data_); }
        }
       private:
        fn_t fn_ = nullptr;
        fn_t rm_ = nullptr;
        void (*mv_)(void*, void*);
        union alignas(std::max_align_t) U {
            void* data_;
            std::byte buff_[buffer_size];   // small buffer optimization
        } storage_ {};
        bool sb_ = false;
    };

    // work executor: a wrapper around a std::thread and a ring_buffer of type-erased jobs
    struct worker_t {
        template <typename WorkerLoopT>
        worker_t(int i, int queue_size, WorkerLoopT worker_loop, ThreadPool* tp) :
            i_(i), queue_(queue_size), running_(true), tp_(tp), thread_(worker_loop) { }

        // worker coordination
        void wake_up() { idle_cv_.notify_one(); }
        void join() { thread_.join(); }
        bool joinable() const { return thread_.joinable(); }
        void to_idle() {
            std::unique_lock<std::mutex> lock(idle_m_);
            idle_cv_.wait(lock, [&]() {
                return tp_->workload_[i_].load(std::memory_order_acquire) > 0 ||
                       !running_.load(std::memory_order_acquire);
            });
        }
        void stop() {
            running_.store(false, std::memory_order_release);
            wake_up();   // signals possible waiting worker
        }
        // observers
        bool is_running() const { return running_.load(std::memory_order_acquire) == true; }
        // task handling
        std::optional<task_t> try_fetch_task() {
            // check mailbox
            auto&& mail = mailbox_.pop();
            if (mail) {
                if (queue_.empty()) {
                    return mail;   // bypass queue if already empty ------------ batched insertion?
                } else {
                    queue_.push_front(std::move(*mail));
                }
            }
            return queue_.pop_front();
        }

        void submit_task(task_t&& task) {
            // submit task to mailbox, as only this worker can push to queue_'s front (avoid break Chase-Lev invariant)
            return mailbox_.push(std::move(task));
        }
        std::optional<task_t> try_steal() { return queue_.pop_back(); }
       private:
        int i_;

        internals::chase_lev_queue<task_t> queue_;
        internals::mpsc_queue<task_t> mailbox_;
        std::atomic<bool> running_;
        ThreadPool* tp_;
        std::thread thread_;

        // worker coordination
        std::mutex idle_m_;
        std::condition_variable idle_cv_;
    };

    // worker loop
    void execute_task_(const task_t& task, int i) {
        task();
        workload_[i].fetch_sub(1, std::memory_order_release);
        // n_tasks_.fetch_sub(1, std::memory_order_release);  --- removed for performances
        // notify anyone waiting on global join()
        // std::lock_guard<std::mutex> lock(join_m_);
        // join_cv_.notify_all();
        return;
    }
    void worker_loop_(int i) {
        internals::tls_worker_id = i;   // register worker global id
        init_latch_.arrive_and_wait();
        // loop logic
        while (workers_[i]->is_running()) {
            // 1. check worker queue
            auto task = workers_[i]->try_fetch_task();   // pop-front
            if (task) {
                execute_task_(*task, i);
            } else {
                // 2. try steal work from busy workers
                std::optional<int> j = stealer_.pick(workload_);
                if (j && (*j != i)) {
                    auto task = workers_[j.value()]->try_steal();
                    if (task) {
                        execute_task_(*task, j.value());
                    } else {
                        // 3. nothing to do, move to idle state
                        workers_[i]->to_idle();
                    }
                }
            }
        }
        return;
    }

    // submits and schedules one task
    template <typename Task> void submit_task_(Task&& task) {
        // 1. query scheduler to pick worker
        int i = scheduler_.pick(workload_);
        // 2. submit task
        workers_[i]->submit_task(task_t(task));
        workload_[i].fetch_add(1, std::memory_order_release);
        // n_tasks_.fetch_add(1, std::memory_order_release); --- removed for performances
        // 3. notify worker, if waiting
        workers_[i]->wake_up();
        return;
    }

    std::vector<std::shared_ptr<worker_t>> workers_;
    int n_workers_;
    scheduling_t scheduler_;
    std::latch init_latch_;
    // stealing logic
    atomic_array<int> workload_;   // current workload distribution
    stealing_t stealer_;
    // joining logic
    std::atomic<int> n_tasks_;   // overall number of active tasks
    std::mutex join_m_;
    std::condition_variable join_cv_;
   public:
    // constructor
    ThreadPool() : ThreadPool(std::thread::hardware_concurrency()) { }
    explicit ThreadPool(int size) :
        n_workers_(size), scheduler_(size), init_latch_(size + 1), workload_(size, 0), stealer_(size) {
        workers_.reserve(n_workers_);
        // start workers
        for (int i = 0; i < n_workers_; i++) {
            workers_.emplace_back(std::make_shared<worker_t>(i, buffer_size, [&, this, i] { worker_loop_(i); }, this));
            workload_[i].store(0, std::memory_order_release);
        }
        // wait workers to be ready, avoids threadpool destruction before worker construction
        init_latch_.arrive_and_wait();
    }
    // observers
    int n_workers() const { return n_workers_; }
    const std::atomic<int>& n_tasks() const { return n_tasks_; }

    // submits f(args...) for asynchronous execution. returns a std::future holding the result
    template <typename F, typename... Args>
    [[nodiscard]] auto submit(F&& f, Args&&... args) -> std::future<decltype(f(args...))> {
        // type-erase task type to send to the threadpool
        using ret_t = decltype(f(args...));
        std::shared_ptr<std::packaged_task<ret_t()>> packaged_task =
          std::make_shared<std::packaged_task<ret_t()>>([f_ = f, ... args_ = args]() mutable { return f_(args_...); });
        std::function<void()> task = [packaged_task]() { (*packaged_task)(); };

        submit_task_(task);
        return packaged_task->get_future();
    };
    // execute f(args...) asynchronously. doesn't wait for any result
    template <typename F, typename... Args> void execute(F&& f, Args&&... args) {
        auto task = [f_ = f, ... args_ = args]() mutable { f_(args_...); };
        submit_task_(task);
        return;
    }

    // executes the range [begin, end) in parallel, submits one task per iteration
    template <typename F, typename Iterator>
        requires(std::is_same_v<std::invoke_result_t<F, Iterator>, void> && requires(Iterator it) { ++it; })
    void parallel_for(Iterator begin, Iterator end, F&& f) {
        for (Iterator j = begin; j != end; ++j) { execute(f, j); }
        join();
    }
    // executes the range [begin, end) in parallel with custom increment logic, submits one task per iteration
    template <typename F, typename Iterator>
        requires(
          std::is_same_v<std::invoke_result_t<F, Iterator>, void> && requires(Iterator it, Iterator jt) { it += jt; })
    void parallel_for(Iterator begin, Iterator end, std::function<Iterator(Iterator)> inc, F&& f) {
        for (Iterator j = begin; j < end; j += inc(j)) { execute(f, j); }
        join();
    }

    // splits [begin, end) into contiguous chunks of size grain_size and submits one task per chunk
    template <typename F, typename Iterator>
        requires(
          std::is_same_v<std::invoke_result_t<F, Iterator>, void> &&
          requires(Iterator it, Iterator jt, int k) {
              { it - jt } -> std::convertible_to<int>;
              it += k;
              ++it;
              k < it;
          })
    void parallel_for(Iterator begin, Iterator end, int grain_size, F&& f) {
        fdapde_assert(grain_size > 0);
        const int n = end - begin;
        if (n == 0) return;   // nothing to loop

        grain_size = (grain_size < n) ? grain_size : n;
        std::atomic<int> group_ptr {1};
        for (Iterator j = begin; j < end; j += grain_size) {
            Iterator k = ((j + grain_size) < end) ? (j + grain_size) : end;
            // send job for [i, k) blocked range execution
            auto loop_body = [&, k_ = k, j_ = j, f_ = f]() {
                for (Iterator it = j_; it < k_; ++it) { f_(it); }
                if (group_ptr.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                    // last chunk of this parallel_for task group
                    std::unique_lock<std::mutex> lock(join_m_);
                    join_cv_.notify_all();
                }
            };
            submit_task_(loop_body);
            group_ptr.fetch_add(1, std::memory_order_release);
        }
        // return fast if main thread is already the last
        if (group_ptr.fetch_sub(1, std::memory_order_acq_rel) == 1) { return; }
        // wait for this task group
        std::unique_lock<std::mutex> lock(join_m_);
        join_cv_.wait(lock, [&] { return group_ptr.load(std::memory_order_acquire) == 0; });
        return;
    }
    // splits container range into contiguous chunks whose size is adaptively chosen. submits one task per chunk
    template <typename Container, typename F> void parallel_for_each(Container&& c, F&& f) {
        using iterator_t = typename std::decay_t<Container>::iterator;
        iterator_t begin = c.begin();
        iterator_t end = c.end();
        const int n = std::distance(begin, end);
        if (n == 0) return;   // nothing to loop

        int grain_size = std::max(1.0, double(n) / (4 * workers_.size()));   // TBB auto-partitioning
        for (iterator_t j = begin; j < end; j += grain_size) {
            iterator_t k = ((j + grain_size) < end) ? (j + grain_size) : end;
            // send job for [i, k) blocked range execution
            execute([=, f_ = f, this]() mutable {
                for (iterator_t it = j; it < k; ++it) { f_(*it); }   // pass dereferenced iterator
            });
        }
        join();
    }

    // waits until all submitted tasks complete
    void join() {
        //         // work-helping: instead of blocking the caller, temporarily use it as member of the pool
        //         // while (n_tasks_.load(std::memory_order_acquire) > 0) {
        //         //     bool busy = false;
        //         //     for (int i = 0; i < n_workers_; ++i) {
        //         //         auto task = workers_[i]->try_steal();
        //         //         if (task) {
        //         //             execute_task_(task, i);
        //         //             busy = true;
        //         //             break;
        //         //         }
        //         //     }
        //         //     if (!busy && n_tasks_.load(std::memory_order_acquire) > 0) {
        //                 // std::unique_lock<std::mutex> lock(join_m_);
        //                 // join_cv_.wait(lock, [&] { return n_tasks_.load(std::memory_order_acquire) == 0; });
        //         //     }
        //         // }
        // while (n_tasks_.load(std::memory_order_acquire) > 0) {
        //         std::this_thread::yield();
        //     }
        //       // n_tasks_.store(0);
        //         return;
    }
    // stops all running threads. not yet completed tasks are losts. explicitly call join() to wait for all sent tasks
    void stop() {
        for (auto& worker : workers_) { worker->stop(); }
        for (auto& worker : workers_) { worker->join(); }
        return;
    }
    // destructor
    ~ThreadPool() { stop(); }
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
// logical identifier of running thread
int this_thread_id() noexcept { return internals::tls_worker_id; }
// // executes the range [begin, end) in parallel, submits one task per iteration
// template <typename Iterator, typename F> void parallel_for(Iterator begin, Iterator end, F&& f) {
//     internals::threadpool_executor::instance().parallel_for(begin, end, f);
// }
// // executes the range [begin, end) in parallel with custom increment logic, submits one task per iteration
// template <typename Iterator, typename F>
// void parallel_for(Iterator begin, Iterator end, std::function<Iterator(Iterator)> inc, F&& f) {
//     internals::threadpool_executor::instance().parallel_for(begin, end, inc, f);
// }
// // splits [begin, end) into contiguous chunks of size grain_size and submits one task per chunk
// template <typename Iterator, typename F> void parallel_for(Iterator begin, Iterator end, int grain_size, F&& f) {
//     internals::threadpool_executor::instance().parallel_for(begin, end, grain_size, f);
// }
// // parallel range-for over container
// template <typename Container, typename F> void parallel_for_each(Container&& c, F&& f) {
//     internals::threadpool_executor::instance().parallel_for_each(c, f);
// }

}   // namespace fdapde

#endif   // __FDAPDE_EXECUTION_THREADPOOL_H__
