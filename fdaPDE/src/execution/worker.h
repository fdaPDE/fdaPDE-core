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

#ifndef __FDAPDE_EXECUTION_WORKER_H__
#define __FDAPDE_EXECUTION_WORKER_H__

#include "header_check.h"

namespace fdapde {
namespace internals {

inline thread_local int tls_worker_id = -1;   // worker logical index
  
// pooled-object (free-list based) allocator supporting concurrent lock-free deallocation
template <typename T> struct pool_allocator {
   private:
    struct slot_t {
        alignas(T) std::byte storage[sizeof(T)];
        slot_t* next;
    };
    // contiguous memory region of slot_t items
    class block_t {
        slot_t* data_;
        block_t* next_;
       public:
        explicit block_t(std::size_t block_sz) : data_(nullptr), next_(nullptr) { data_ = new slot_t[block_sz]; }
        ~block_t() { delete[] data_; }
        // observers
        slot_t* data() const { return data_; }
        block_t* next() const { return next_; }
        // accessors
        const T& operator[](int i) const { return *reinterpret_cast<const T*>(data_[i].storage); }
        T& operator[](int i) { return *reinterpret_cast<T*>(data_[i].storage); }
        // modifiers
        void set_next(block_t* next) { next_ = next; }
    };
   public:
    pool_allocator() : pool_allocator(1024) { }
    explicit pool_allocator(std::size_t block_sz) :
        block_sz_(block_sz), used_slots_(0), free_list_(nullptr), shared_free_list_(nullptr) {
        base_ = new block_t(block_sz_);
        block_list_ = base_;
    }
    // disable copy semantic
    pool_allocator(const pool_allocator&) = delete;
    pool_allocator& operator=(const pool_allocator&) = delete;

    // constructs object of type T and returns pointer to its reserved memory region. not thread-safe
    template <typename... Args> T* allocate(Args&&... args) {
        // fetch memory from free_list, if available
        if (free_list_ == nullptr) { free_list_ = shared_free_list_.exchange(nullptr, std::memory_order_acquire); }
        if (free_list_ != nullptr) { return construct_at_free_slot_(std::forward<Args>(args)...); }
        // allocate memory if available space exhausted
        if (used_slots_ >= block_sz_) {
            block_t* new_block = new block_t(block_sz_);
            block_list_->set_next(new_block);
            block_list_ = new_block;
            used_slots_ = 0;
        }
        T* ptr = std::addressof((*block_list_)[used_slots_]);
        used_slots_++;
        new (ptr) T(std::forward<Args>(args)...);
        return ptr;
    }
    // concurrently deallocate memory reserved to ptr
    void deallocate(T* ptr) {
        // the ptr memory layout is the one of a slot_t, here is safe to reinterpret ptr as a slot_t*
        slot_t* slot = reinterpret_cast<slot_t*>(ptr);
        slot_t* head = shared_free_list_.load(std::memory_order_acquire);
        // lock-free retry loop: append this slot to the shared free list
        do {
            slot->next = head;
        } while (
          !shared_free_list_.compare_exchange_strong(head, slot, std::memory_order_release, std::memory_order_relaxed));
        return;
    }
    ~pool_allocator() {
        block_t* next = base_;
        while (next != nullptr) {
            block_t* tmp = next->next();
            delete next;
            next = tmp;
        }
    }
   private:
    template <typename... Args> T* construct_at_free_slot_(Args&&... args) {
        slot_t* slot = free_list_;
        free_list_ = slot->next;
        T* ptr = reinterpret_cast<T*>(std::addressof(slot->storage));
        new (ptr) T(std::forward<Args>(args)...);
        return ptr;
    }
    const std::uint64_t block_sz_;
    std::uint64_t used_slots_;                // number of used slots in last block
    block_t* block_list_;                     // list of allocated memory blocks
    slot_t* free_list_;                       // already allocated slots available for writing
    std::atomic<slot_t*> shared_free_list_;   // slots concurrently freed by other threads
    block_t* base_;                           // first block of the pool
};
  
// implementation of the static-sized Chase-Lev circular buffer queue:
// * "Chase, D. and Lev, Y. (2005). Dynamic circular work-stealing deque. In Proceedings of the seventeenth annual
//    ACM symposium on Parallelism in algorithms and architectures (pp. 21-28)."
// * "Le, N. M., Pop, A., Cohen, A., \and Zappa, F. (2013). Correct and efficient work-stealing for weak memory
//    models. ACM SIGPLAN Notices, 48(8), 69-80."
template <typename T> struct chase_lev_queue {
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

// unbounded lock-free multi-producer/single-consumer queue with free-list memory allocator
// * "Michael, M. and Scott, M. (1996). Simple, fast, and practical non-blocking and blocking concurrent queue
//    algorithms. In Proceedings of the fifteenth annual ACM symposium on Principles of distributed computing (pp
//    267-275)."
template <typename T>
    requires(std::is_move_constructible_v<T> && std::is_default_constructible_v<T>)
class mpsc_queue {
    // queue node
    struct node_t {
        node_t() : next_(nullptr), data_() { }
        template <typename T_>
            requires(std::is_constructible_v<T, T_>)
        explicit node_t(T_&& data) : next_(nullptr), data_(std::forward<T_>(data)) { }
        // observers
        std::atomic<node_t*>& next() { return next_; }
        const T& data() const { return data_; }
       private:
        std::atomic<node_t*> next_;   // next queue node
        T data_;                      // stored data
    };
   public:
    using value_type = T;
    using allocator_type = pool_allocator<node_t>;

    mpsc_queue() : mpsc_queue(512) { }
    explicit mpsc_queue(std::size_t allocator_block_sz) :
        head_(nullptr), tail_(nullptr), allocator_(allocator_block_sz) {
        node_t* root = allocator_.allocate();
        head_ = root;
        tail_.store(root, std::memory_order_relaxed);
    }

    // multi-producer push, pushes the element value to the end of the queue
    template <typename T_> void push(T_&& value) {
        node_t* node = allocator_.allocate(std::forward<T_>(value));
        node_t* prev = tail_.exchange(node, std::memory_order_acq_rel);
        prev->next().store(node, std::memory_order_release);
        return;
    }
    // single consumer pop, removes an element from the front of the queue. returns nullopt if the queue is empty
    std::optional<value_type> pop() {
        node_t* next = head_->next().load(std::memory_order_acquire);
        if (next == nullptr) { return std::nullopt; }

        value_type value = std::move(next->data());
        allocator_.deallocate(head_);
        head_ = next;
        return std::move(value);
    }
   private:
    node_t* head_;
    std::atomic<node_t*> tail_;
    allocator_type allocator_;
};
  
// logical execution component mapped to a physical execution unit (hardware thread)
struct worker {
    static constexpr int task_queue_size = 4096;
    // constructor
    template <typename ThreadPool_>
    worker(int worker_id, ThreadPool_* tp) :
        worker_id_(worker_id), task_queue_(task_queue_size), mailbox_(), running_(true), thread_([this, tp] {
            run_(tp);
        }) { }

    // worker coordination
    void wake_up() { idle_cv_.notify_one(); }
    void join() { thread_.join(); }
    bool joinable() const { return thread_.joinable(); }
    void stop() {
        std::lock_guard<std::mutex> lock(idle_m_);
        running_.store(false, std::memory_order_release);
	idle_cv_.notify_one();
    }
    // observers
    bool is_running() const { return running_.load(std::memory_order_acquire); }
    // pubilc task handling
    void submit_task(task_handle&& task) {
        task_handle* task_ptr = task_pool_.allocate(std::move(task));
        mailbox_.push(std::move(task_ptr));
    }

    template <typename Task> task_handle* allocate_task(Task&& task) {
        return task_pool_.allocate(std::forward<Task>(task));
    }
    void dispatch_handle(task_handle* task) { mailbox_.push(task); }

    std::optional<task_handle*> try_steal() { return task_queue_.pop_back(); }
   private:
    // fetches a task. The task is obtained either from the local task queue or from the inbound mailbox.
    // returns nullopt if no task is available for execution
    std::optional<task_handle*> try_fetch_task_() {
        auto&& task = task_queue_.pop_front();
        if (task) { return std::move(task); }
        // check mailbox
        auto&& mail = mailbox_.pop();
        if (mail) {
            // drain mailbox
            auto&& tmp = mailbox_.pop();
            while (tmp.has_value()) {
                task_queue_.push_front(std::move(*tmp));
                tmp = mailbox_.pop();
            }
            return mail;
        } else {
            return std::nullopt;
        }
    }
    // for a runnable task, acquires, executes and notifies its completion
    template <typename ThreadPool_> void try_execute_task_(ThreadPool_* tp, task_handle* task) {
        if (!task->runnable()) return;
        task->run();
        tp->on_task_complete(task);
	return;
    }
    // worker loop
    template <typename ThreadPool_> void run_(ThreadPool_* tp) {
        internals::tls_worker_id = worker_id_;   // register worker global id
        tp->on_worker_ready();
        // loop logic
        while (is_running()) {
            // check worker queue
            std::optional<task_handle*> task = try_fetch_task_();
            // if a task is not runnable, it is removed from any working queue but not from its task_pool. the task will
            // be re-enqueued as a result of a notification event (e.g., task ref_count hits 0) and re-executed
            if (task) {
                try_execute_task_(tp, *task);
            } else {
                // try steal work from busy workers
                std::optional<task_handle*> task = tp->try_steal(worker_id_);
                if (task) {
                    try_execute_task_(tp, *task);
                } else {
                    // nothing to do, query the threadpool to establish if we have to move to idle state
                    std::unique_lock<std::mutex> lock(idle_m_);
                    idle_cv_.wait(
                      lock, [&]() { return tp->on_worker_idle() || !running_.load(std::memory_order_acquire); });
                }
            }
        }
        return;
    }

    const int worker_id_;                        // worker identifier
    pool_allocator<task_handle> task_pool_;      // memory allocator for task storage
    chase_lev_queue<task_handle*> task_queue_;   // tasks pending for execution, amenable to work stealing
    mpsc_queue<task_handle*> mailbox_;           // externally submitted tasks
    std::atomic<bool> running_;                  // logical indicating wheter to stop execution
    std::thread thread_;                         // OS managed thread
    // idle state
    std::mutex idle_m_;
    std::condition_variable idle_cv_;
};

}   // namespace internals

// logical identifier of running thread
inline int this_worker_id() noexcept { return internals::tls_worker_id; }
  
}   // namespace fdapde

#endif   // __FDAPDE_EXECUTION_WORKER_H__
