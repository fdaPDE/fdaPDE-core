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

static constexpr int main_thread_id = -1;
inline thread_local int tls_worker_id = main_thread_id;   // worker logical index

// concurrent pooled-object allocator, with fast lock-free deallocatin path.
template <typename T> struct pool_allocator {
    using value_type = T;
    using pointer = value_type*;
    using const_pointer = const value_type*;
    using reference = T&;
    using const_reference = const T&;
    using size_type = std::uint64_t;
    using difference_type = std::int64_t;
   private:
    struct slot_type {
        alignas(value_type) std::byte storage[sizeof(value_type)];
        slot_type* next;
    };
    // contiguous memory region of slot_type items
    class block_type {
        slot_type* data_;
        block_type* next_;
       public:
        explicit block_type(size_type block_sz) : data_(nullptr), next_(nullptr) { data_ = new slot_type[block_sz]; }
        ~block_type() { delete[] data_; }
        // observers
        slot_type* data() const { return data_; }
        block_type* next() const { return next_; }
        // accessors
        const_reference operator[](int i) const { return *reinterpret_cast<const_pointer>(data_[i].storage); }
        reference operator[](int i) { return *reinterpret_cast<pointer>(data_[i].storage); }
        // modifiers
        void set_next(block_type* next) { next_ = next; }
    };
   public:
    // constructors
    pool_allocator() : pool_allocator(1024) { }
    explicit pool_allocator(size_type block_sz) :
        block_sz_(block_sz), used_slots_(0), free_list_(nullptr), shared_free_list_(nullptr) {
        base_ = new block_type(block_sz_);
        block_list_ = base_;
    }
    // disable copy semantic
    pool_allocator(const pool_allocator&) = delete;
    pool_allocator& operator=(const pool_allocator&) = delete;

    // constructs object of type T and returns pointer to its reserved memory region. not thread-safe
    template <typename... Args> pointer allocate(Args&&... args) {
        std::lock_guard<std::mutex> lock(m_);
        // fetch memory from free_list, if available
        if (free_list_ == nullptr) { free_list_ = shared_free_list_.exchange(nullptr, std::memory_order_acquire); }
        if (free_list_ != nullptr) { return construct_at_free_slot_(std::forward<Args>(args)...); }
        // allocate memory if available space exhausted
        if (used_slots_ >= block_sz_) {
            block_type* new_block = new block_type(block_sz_);
            block_list_->set_next(new_block);
            block_list_ = new_block;
            used_slots_ = 0;
        }
        pointer ptr = std::addressof((*block_list_)[used_slots_]);
        used_slots_++;
        new (ptr) value_type(std::forward<Args>(args)...);
        return ptr;
    }
    // concurrently deallocate memory reserved to ptr
    void deallocate(pointer ptr) {
        // the ptr memory layout is the one of a slot_type, here is safe to reinterpret ptr as a slot_type*
        slot_type* slot = reinterpret_cast<slot_type*>(ptr);
        slot_type* head = shared_free_list_.load(std::memory_order_acquire);
        // lock-free retry loop: append this slot to the shared free list
        do {
            slot->next = head;
        } while (
          !shared_free_list_.compare_exchange_strong(head, slot, std::memory_order_release, std::memory_order_relaxed));
        return;
    }
    ~pool_allocator() {
        block_type* next = base_;
        while (next != nullptr) {
            block_type* tmp = next->next();
            delete next;
            next = tmp;
        }
    }
   private:
    template <typename... Args> pointer construct_at_free_slot_(Args&&... args) {
        slot_type* slot = free_list_;
        free_list_ = slot->next;
        pointer ptr = reinterpret_cast<pointer>(std::addressof(slot->storage));
        new (ptr) value_type(std::forward<Args>(args)...);
        return ptr;
    }
    const size_type block_sz_;
    size_type used_slots_;                       // number of used slots in last block
    block_type* block_list_;                     // list of allocated memory blocks
    slot_type* free_list_;                       // already allocated slots available for writing
    std::atomic<slot_type*> shared_free_list_;   // slots concurrently freed by other threads
    block_type* base_;                           // first block of the pool
    std::mutex m_;
};

// implementation of the static-sized Chase-Lev circular buffer queue:
// * "Chase, D. and Lev, Y. (2005). Dynamic circular work-stealing deque. In Proceedings of the seventeenth annual
//    ACM symposium on Parallelism in algorithms and architectures (pp. 21-28)."
// * "Le, N. M., Pop, A., Cohen, A., \and Zappa, F. (2013). Correct and efficient work-stealing for weak memory
//    models. ACM SIGPLAN Notices, 48(8), 69-80."
template <typename T>
    requires(std::is_trivially_copyable_v<T> && std::atomic<T>::is_always_lock_free)
struct chase_lev_queue {
    // any instance of type T must be copy/move-able atomically. pointers fall in this category
    using value_type = T;
    using allocator_type = std::allocator<T>;
    using reference = value_type&;
    using const_reference = const value_type&;
    using pointer = typename std::allocator_traits<allocator_type>::pointer;
    using const_pointer = typename std::allocator_traits<allocator_type>::const_pointer;
    using size_type = std::uint64_t;
    using difference_type = std::int64_t;

    // constructors
    chase_lev_queue() : chase_lev_queue(4096) { }
    explicit chase_lev_queue(size_type capacity) :
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
    std::optional<value_type> pop_front() {
        difference_type b = bottom_.load(std::memory_order_relaxed) - 1;
        bottom_.store(b, std::memory_order_relaxed);
        std::atomic_thread_fence(std::memory_order_seq_cst);
        difference_type t = top_.load(std::memory_order_relaxed);

        if (std::cmp_less_equal(t, b)) {
            if (std::cmp_equal(t, b)) {
                // last queue element
                if (!top_.compare_exchange_strong(t, t + 1, std::memory_order_seq_cst, std::memory_order_relaxed)) {
                    // lost race with concurrent pop_back (stealing attempt)
                    bottom_.store(b + 1, std::memory_order_relaxed);
                    return std::nullopt;
                }
                bottom_.store(b + 1, std::memory_order_relaxed);
            }
            difference_type idx = b & mask_;
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
        requires(std::is_convertible_v<T_, value_type>)
    bool push_front(T_&& value) {
        difference_type b = bottom_.load(std::memory_order_relaxed);
        difference_type t = top_.load(std::memory_order_acquire);
        // abort if queue is full
        if (std::cmp_greater_equal(b - t, capacity_ - 1)) { return false; }
        // write
        difference_type idx = b & mask_;
        buffer_[idx] = std::move(value);
        std::atomic_thread_fence(std::memory_order_release);
        bottom_.store(b + 1, std::memory_order_relaxed);   // sync write
        return true;
    }
    // constructs a new element at the end of the container. aborts if container full
    // only owning thread push at buffer's back
    template <typename... Args>
        requires(std::is_constructible_v<value_type, Args...>)
    bool emplace_front(Args&&... args) {
        value_type value(std::forward<Args>(args)...);

        difference_type b = bottom_.load(std::memory_order_relaxed);
        difference_type t = top_.load(std::memory_order_acquire);
        // abort if queue is full
        if (std::cmp_greater_equal(b - t, capacity_ - 1)) { return false; }
        // write
        difference_type idx = b & mask_;
        buffer_[idx] = std::move(value);
        std::atomic_thread_fence(std::memory_order_release);
        bottom_.store(b + 1, std::memory_order_relaxed);   // sync write
        return true;
    }
    // returns last element of the container, nullopt if container empty. the element is removed
    // this method is invoked as a result of a work-stealing attempt
    std::optional<value_type> pop_back() {
        difference_type t = top_.load(std::memory_order_acquire);
        std::atomic_thread_fence(std::memory_order_seq_cst);
        difference_type b = bottom_.load(std::memory_order_acquire);
        // abort if queue is empty
        if (std::cmp_greater_equal(t, b)) { return std::nullopt; }
        // atomic read
        difference_type idx = t & mask_;
        // try to claim the element by incrementing top
        if (!top_.compare_exchange_strong(t, t + 1, std::memory_order_seq_cst, std::memory_order_relaxed)) {
            return std::nullopt;   // lost race
        }
        return std::move(buffer_[idx]);   // we move now, after having win the CAS
    }
    // observers
    bool empty() const {
        difference_type t = top_.load(std::memory_order_acquire), b = bottom_.load(std::memory_order_acquire);
        return t >= b;
    }
   private:
    std::vector<value_type> buffer_;
    const size_type capacity_;     // maximum buffer size (as power of 2)
    const difference_type mask_;   // 0b(capacity_ - 1), allows fast modulo capacity_

    alignas(64) std::atomic<difference_type> bottom_;   // number of performed pushes
    alignas(64) std::atomic<difference_type> top_;      // number of performed pops
};

// unbounded lock-free multi-producer/single-consumer queue with free-list memory allocator
// * "Michael, M. and Scott, M. (1996). Simple, fast, and practical non-blocking and blocking concurrent queue
//    algorithms. In Proceedings of the fifteenth annual ACM symposium on Principles of distributed computing (pp
//    267-275)."
template <typename T>
    requires(std::is_move_constructible_v<T> && std::is_default_constructible_v<T>)
struct mpsc_queue {
    using value_type = T;
    using reference = value_type&;
    using const_reference = const value_type&;
   private:
    // linked-list like node with single successor
    struct node_type {
        node_type() : next(nullptr), data() { }
        template <typename T_>
            requires(std::is_constructible_v<value_type, T_>)
        explicit node_type(T_&& data_) : next(nullptr), data(std::forward<T_>(data_)) { }
        // observers
        std::atomic<node_type*> next;   // next queue node
        value_type data;                // stored data
    };
   public:
    using allocator_type = pool_allocator<node_type>;
    using pointer = typename allocator_type::pointer;
    using const_pointer = typename allocator_type::const_pointer;
    using size_type = std::uint64_t;
    using difference_type = std::int64_t;

    mpsc_queue() : mpsc_queue(512) { }
    explicit mpsc_queue(size_type allocator_block_sz) : head_(nullptr), tail_(nullptr), allocator_(allocator_block_sz) {
        pointer root = allocator_.allocate();
        head_ = root;
        tail_.store(root, std::memory_order_relaxed);
    }

    // multi-producer push, pushes the element value to the end of the queue
    template <typename T_>
        requires(std::is_constructible_v<value_type, T_>)
    void push(T_&& value) {
        pointer node = allocator_.allocate(std::forward<T_>(value));
        pointer prev = tail_.exchange(node, std::memory_order_acq_rel);
        prev->next.store(node, std::memory_order_release);
        return;
    }
    // single consumer pop, removes an element from the front of the queue. returns nullopt if the queue is empty
    std::optional<value_type> pop() {
        pointer next = head_->next.load(std::memory_order_acquire);
        if (next == nullptr) { return std::nullopt; }

        value_type value = std::move(next->data);
        allocator_.deallocate(head_);
        head_ = next;
        return std::move(value);
    }
   private:
    pointer head_;
    alignas(64) std::atomic<pointer> tail_;
    allocator_type allocator_;

  std::mutex m_;
  
};

// logical execution component mapped to a physical execution unit (hardware thread)
struct worker {
    using task_type = task_handle;
    using allocator_type = pool_allocator<task_type>;
    using task_pointer = typename allocator_type::pointer;
    using task_const_pointer = typename allocator_type::const_pointer;
    using task_queue_type = chase_lev_queue<task_pointer>;
    using task_buffer_type = mpsc_queue<task_pointer>;
    static constexpr int task_queue_size = 8192;   // 2^13

    // constructor
    template <typename Executor>
    worker(int worker_id, Executor* e) :
        worker_id_(worker_id), task_queue_(task_queue_size), task_buffer_(), thread_([this, e] { run_(e); }) { }

    // thread coordination
    void join() { thread_.join(); }
    bool joinable() const { return thread_.joinable(); }
    // task handling
    // allocates and forward task into the worker task_pool
    template <typename Task> task_pointer allocate_task(Task&& task) {
        task_pointer task_ptr = task_pool_.allocate(std::forward<Task>(task));
        if (!task_ptr->allocation_context().has_value()) { task_ptr->set_allocation_context(worker_id_); }
        return task_ptr;
    }
    // enqueue task for execution. task must point to some stable address
    void enqueue_task(task_pointer task) { task_buffer_.push(task); }
    // allocates and forwards task into the worker task_pool. enqueue the task for execution
    template <typename Task> void submit_task(Task&& task) {
        task_pointer task_ptr = allocate_task(std::forward<Task>(task));
        enqueue_task(task_ptr);
        return;
    }
    // frees memory holded by task
    void deallocate_task(task_pointer task) { task_pool_.deallocate(task); }
    // allows an external worker to perform a stealing attempt on this worker task_queue
    std::optional<task_pointer> try_steal() { return task_queue_.pop_back(); }
    // tries to execute a task, if any. usec in active join loops
    template <typename Executor> void try_execute_one(Executor* e) {
        std::optional<task_pointer> task = try_fetch_task_();
        if (task) {
            try_execute_task_(e, *task);
        } else {
            std::optional<task_pointer> task = e->try_steal(worker_id_);
            if (task) { try_execute_task_(e, *task); }
        }
        return;
    }
   private:
    // fetches a task. The task is obtained either from the local task queue or from the inbound task_buffer.
    // returns nullopt if no task is available for execution
    std::optional<task_pointer> try_fetch_task_() {
        auto&& task = task_queue_.pop_front();
        if (task) { return std::move(task); }
        // check task_buffer
        auto&& mail = task_buffer_.pop();
        if (mail) {
            // drain task_buffer
            auto&& tmp = task_buffer_.pop();
            while (tmp.has_value()) {
                task_queue_.push_front(std::move(*tmp));
                tmp = task_buffer_.pop();
            }
            return mail;
        } else {
            return std::nullopt;
        }
    }
    // for a runnable task, acquires, executes and notifies its completion
    template <typename Executor> void try_execute_task_(Executor* e, task_pointer task) {
        if (!task->runnable()) {
            // a non runnable task is removed from any working queue but not from its task_pool. the task will
            // be eventually re-enqueued and executed as a result of a notification event
            return;
        }
        task->run();
        e->on_task_complete(task);
        return;
    }
    // worker loop
    template <typename Executor> void run_(Executor* e) {
        tls_worker_id = worker_id_;   // register worker global id
        e->on_worker_ready();
        // loop logic
        while (e->is_active()) {
            // check worker local queue
            std::optional<task_pointer> task = try_fetch_task_();
            if (task) {
                try_execute_task_(e, *task);
            } else {
                // try steal work from busy workers
                std::optional<task_pointer> task = e->try_steal(worker_id_);
                if (task) {
                    try_execute_task_(e, *task);
                } else {
                    e->on_worker_idle();   // nothing to do, query the executor to decide if move to idle
                }
            }
        }
        return;
    }

    const int worker_id_;            // worker identifier
    allocator_type task_pool_;       // memory allocator for task storage
    task_queue_type task_queue_;     // tasks pending for execution, amenable to work stealing
    task_buffer_type task_buffer_;   // externally submitted tasks
    std::thread thread_;             // OS managed thread
};

}   // namespace internals

// logical identifier of running thread
inline int this_thread_id() noexcept { return internals::tls_worker_id; }
  
}   // namespace fdapde

#endif   // __FDAPDE_EXECUTION_WORKER_H__
