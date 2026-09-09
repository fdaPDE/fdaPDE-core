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

/// @brief serializes allocation and recycling of explicitly destroyed pooled objects
template <typename T> struct pool_allocator {
    using value_type = T;
    using pointer = value_type*;
    using const_pointer = const value_type*;
    using reference = T&;
    using const_reference = const T&;
    using size_type = std::uint64_t;
    using difference_type = std::int64_t;
   private:
    /// @brief stores an aligned object or a link to the next reusable slot
    struct slot_type {
        alignas(value_type) std::byte storage[sizeof(value_type)];
        slot_type* next;
    };
    /// @brief owns a contiguous group of allocator slots
    class block_type {
        slot_type* data_;
        block_type* next_;
       public:
        /// @brief allocates a block containing the requested number of aligned slots
        explicit block_type(size_type block_sz) : data_(nullptr), next_(nullptr) { data_ = new slot_type[block_sz]; }
        /// @brief releases owned storage and resources
        ~block_type() { delete[] data_; }

        /// @brief returns the first slot in this block
        slot_type* data() const { return data_; }
        /// @brief returns the next allocator block
        block_type* next() const { return next_; }

        /// @brief links the next allocator block
        void set_next(block_type* next) { next_ = next; }
    };
   public:
    /// @brief starts an empty pool with 1024 slots per block
    pool_allocator() : pool_allocator(1024) { }
    /// @brief starts an empty pool with the supplied positive block size
    explicit pool_allocator(size_type block_sz) : block_sz_(block_sz), used_slots_(0), free_list_(nullptr) {
        base_ = new block_type(block_sz_);
        block_list_ = base_;
    }
    /// @brief disables construction that would duplicate runtime ownership
    pool_allocator(const pool_allocator&) = delete;
    /// @brief disables assignment of runtime ownership
    pool_allocator& operator=(const pool_allocator&) = delete;

    /// @brief constructs an object in a new or recycled slot under the allocator lock
    template <typename... Args> pointer allocate(Args&&... args) {
        std::lock_guard<std::mutex> lock(m_);
        // fetch memory from free_list, if available
        if (free_list_ != nullptr) { return construct_at_free_slot_(std::forward<Args>(args)...); }
        // allocate memory if available space exhausted
        if (used_slots_ >= block_sz_) {
            block_type* new_block = new block_type(block_sz_);
            block_list_->set_next(new_block);
            block_list_ = new_block;
            used_slots_ = 0;
        }
        void* storage = block_list_->data()[used_slots_].storage;
        used_slots_++;
        return ::new (storage) value_type(std::forward<Args>(args)...);
    }
    /// @brief destroys the object before returning its slot to the free list
    void deallocate(pointer ptr) {
        std::destroy_at(ptr);
        // the ptr memory layout is the one of a slot_type, here is safe to reinterpret ptr as a slot_type*
        slot_type* slot = reinterpret_cast<slot_type*>(ptr);
        std::lock_guard<std::mutex> lock(m_);
        slot->next = free_list_;
        free_list_ = slot;
        return;
    }
    /// @brief releases slot blocks after callers have destroyed live objects
    ~pool_allocator() {
        block_type* next = base_;
        while (next != nullptr) {
            block_type* tmp = next->next();
            delete next;
            next = tmp;
        }
    }
   private:
    /// @brief constructs an object in the first reusable slot
    template <typename... Args> pointer construct_at_free_slot_(Args&&... args) {
        slot_type* slot = free_list_;
        free_list_ = slot->next;
        return ::new (static_cast<void*>(slot->storage)) value_type(std::forward<Args>(args)...);
    }
    const size_type block_sz_;
    size_type used_slots_;     // number of used slots in last block
    block_type* block_list_;   // list of allocated memory blocks
    slot_type* free_list_;     // already allocated slots available for writing
    block_type* base_;         // first block of the pool
    std::mutex m_;
};

// implementation of the static-sized Chase-Lev circular buffer queue:
// * "Chase, D. and Lev, Y. (2005). Dynamic circular work-stealing deque. In Proceedings of the seventeenth annual
//    ACM symposium on Parallelism in algorithms and architectures (pp. 21-28)."
// * "Le, N. M., Pop, A., Cohen, A., \and Zappa, F. (2013). Correct and efficient work-stealing for weak memory
//    models. ACM SIGPLAN Notices, 48(8), 69-80."
/// @brief provides a bounded single-owner deque with concurrent stealing and atomic circular slots
template <typename T>
    requires(std::is_trivially_copyable_v<T> && std::atomic<T>::is_always_lock_free)
struct chase_lev_queue {
    // slots must support lock-free atomic loads and stores, as task pointers do
    using value_type = T;
    using allocator_type = std::allocator<T>;
    using reference = value_type&;
    using const_reference = const value_type&;
    using pointer = typename std::allocator_traits<allocator_type>::pointer;
    using const_pointer = typename std::allocator_traits<allocator_type>::const_pointer;
    using size_type = std::uint64_t;
    using difference_type = std::int64_t;

    /// @brief constructs an empty deque with 4096 circular slots
    chase_lev_queue() : chase_lev_queue(4096) { }
    /// @brief constructs an empty deque with a power-of-two slot count
    explicit chase_lev_queue(size_type capacity) :
        buffer_(capacity), capacity_(capacity), mask_(capacity - 1), bottom_(0), top_(0) {
        fdapde_assert(
          capacity != 0 && (capacity & (capacity - 1)) == 0, std::invalid_argument,
          "Chase-Lev queue capacity must be a power of two");
    }
    /// @brief disables construction that would duplicate runtime ownership
    chase_lev_queue(const chase_lev_queue&) = delete;
    /// @brief disables construction that would duplicate runtime ownership
    chase_lev_queue(chase_lev_queue&&) = delete;
    /// @brief disables assignment of runtime ownership
    chase_lev_queue& operator=(const chase_lev_queue&) = delete;
    /// @brief disables assignment of runtime ownership
    chase_lev_queue& operator=(chase_lev_queue&&) = delete;

    /// @brief lets the owner claim the newest item or returns no value
    std::optional<value_type> pop_front() {
        difference_type b = bottom_.load(std::memory_order_relaxed) - 1;
        bottom_.store(b, std::memory_order_relaxed);
        std::atomic_thread_fence(std::memory_order_seq_cst);
        difference_type t = top_.load(std::memory_order_relaxed);

        if (std::cmp_less_equal(t, b)) {
            difference_type idx = b & mask_;
            value_type value = buffer_[idx].load(std::memory_order_relaxed);
            if (std::cmp_equal(t, b)) {
                // last queue element
                if (!top_.compare_exchange_strong(t, t + 1, std::memory_order_seq_cst, std::memory_order_relaxed)) {
                    // lost race with concurrent pop_back (stealing attempt)
                    bottom_.store(b + 1, std::memory_order_relaxed);
                    return std::nullopt;
                }
                bottom_.store(b + 1, std::memory_order_relaxed);
            }
            return value;
        } else {
            // empty queue
            bottom_.store(b + 1, std::memory_order_relaxed);
            return std::nullopt;
        }
    }

    /// @brief lets the owner publish an item or reports insufficient capacity
    template <typename T_>
        requires(std::is_convertible_v<T_, value_type>)
    bool push_front(T_&& value) {
        difference_type b = bottom_.load(std::memory_order_relaxed);
        difference_type t = top_.load(std::memory_order_acquire);
        // abort if queue is full
        if (std::cmp_greater_equal(b - t, capacity_ - 1)) { return false; }
        // write
        difference_type idx = b & mask_;
        buffer_[idx].store(value_type(std::forward<T_>(value)), std::memory_order_relaxed);
        bottom_.store(b + 1, std::memory_order_release);
        return true;
    }
    /// @brief constructs and publishes an owner item or reports insufficient capacity
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
        buffer_[idx].store(value, std::memory_order_relaxed);
        bottom_.store(b + 1, std::memory_order_release);
        return true;
    }
    /// @brief reads and claims the oldest item before its circular slot can be reused
    std::optional<value_type> pop_back() {
        difference_type t = top_.load(std::memory_order_acquire);
        std::atomic_thread_fence(std::memory_order_seq_cst);
        difference_type b = bottom_.load(std::memory_order_acquire);
        // abort if queue is empty
        if (std::cmp_greater_equal(t, b)) { return std::nullopt; }
        difference_type idx = t & mask_;
        // read before advancing top because the owner can immediately reuse the claimed circular slot
        value_type value = buffer_[idx].load(std::memory_order_relaxed);
        // try to claim the element by incrementing top
        if (!top_.compare_exchange_strong(t, t + 1, std::memory_order_seq_cst, std::memory_order_relaxed)) {
            return std::nullopt;   // lost race
        }
        return value;
    }

    /// @brief reports whether the observed deque indices contain no item
    bool empty() const {
        difference_type t = top_.load(std::memory_order_acquire), b = bottom_.load(std::memory_order_acquire);
        return t >= b;
    }
    /// @brief reports whether the observed deque occupancy reaches usable capacity
    bool full() const {
        difference_type t = top_.load(std::memory_order_acquire), b = bottom_.load(std::memory_order_relaxed);
        return std::cmp_greater_equal(b - t, capacity_ - 1);
    }
   private:
    std::vector<std::atomic<value_type>> buffer_;
    const size_type capacity_;     // maximum buffer size (as power of 2)
    const difference_type mask_;   // 0b(capacity_ - 1), allows fast modulo capacity_

    alignas(64) std::atomic<difference_type> bottom_;   // number of performed pushes
    alignas(64) std::atomic<difference_type> top_;      // number of performed pops
};

// unbounded lock-free multi-producer/single-consumer queue with free-list memory allocator
// * "Michael, M. and Scott, M. (1996). Simple, fast, and practical non-blocking and blocking concurrent queue
// algorithms. In Proceedings of the fifteenth annual ACM symposium on Principles of distributed computing (pp
// 267-275)."
/// @brief provides a linked multiple-producer single-consumer queue with pooled nodes
template <typename T>
    requires(std::is_move_constructible_v<T> && std::is_default_constructible_v<T>)
struct mpsc_queue {
    using value_type = T;
    using reference = value_type&;
    using const_reference = const value_type&;
   private:
    /// @brief stores one value and the atomic link to its successor
    struct node_type {
        /// @brief stores one value and the atomic link to its successor
        node_type() : next(nullptr), data() { }
        /// @brief stores one value and the atomic link to its successor
        template <typename T_>
            requires(std::is_constructible_v<value_type, T_>)
        explicit node_type(T_&& data_) : next(nullptr), data(std::forward<T_>(data_)) { }

        std::atomic<node_type*> next;   // next queue node
        value_type data;                // stored data
    };
   public:
    using allocator_type = pool_allocator<node_type>;
    using pointer = typename allocator_type::pointer;
    using const_pointer = typename allocator_type::const_pointer;
    using size_type = std::uint64_t;
    using difference_type = std::int64_t;

    /// @brief constructs an empty queue using 512 allocator slots per block
    mpsc_queue() : mpsc_queue(512) { }
    /// @brief constructs the sentinel node using the supplied allocator block size
    explicit mpsc_queue(size_type allocator_block_sz) : head_(nullptr), tail_(nullptr), allocator_(allocator_block_sz) {
        pointer root = allocator_.allocate();
        head_ = root;
        tail_.store(root, std::memory_order_relaxed);
    }

    /// @brief publishes a new tail node from any producer
    template <typename T_>
        requires(std::is_constructible_v<value_type, T_>)
    void push(T_&& value) {
        pointer node = allocator_.allocate(std::forward<T_>(value));
        pointer prev = tail_.exchange(node, std::memory_order_acq_rel);
        prev->next.store(node, std::memory_order_release);
        return;
    }
    /// @brief moves the next value out and recycles the old head node
    std::optional<value_type> pop() {
        pointer next = head_->next.load(std::memory_order_acquire);
        if (next == nullptr) { return std::nullopt; }

        value_type value = std::move(next->data);
        allocator_.deallocate(head_);
        head_ = next;
        return std::move(value);
    }
    /// @brief destroys all remaining queue nodes before releasing their pool
    ~mpsc_queue() {
        while (head_ != nullptr) {
            pointer next = head_->next.load(std::memory_order_relaxed);
            allocator_.deallocate(head_);
            head_ = next;
        }
    }
   private:
    pointer head_;
    alignas(64) std::atomic<pointer> tail_;
    allocator_type allocator_;
};

/// @brief executes local or stolen tasks on one runtime thread
struct worker {
    using task_type = task_handle;
    using allocator_type = pool_allocator<task_type>;
    using task_pointer = typename allocator_type::pointer;
    using task_const_pointer = typename allocator_type::const_pointer;
    using task_queue_type = chase_lev_queue<task_pointer>;
    using task_buffer_type = mpsc_queue<task_pointer>;
    static constexpr int task_queue_size = 8192;   // 2^13

    /// @brief starts a worker thread with its task pool and inbound queue
    template <typename Executor>
    worker(int worker_id, Executor* e) :
        worker_id_(worker_id), task_queue_(task_queue_size), task_buffer_(), thread_([this, e] { run_(e); }) { }

    /// @brief joins this worker thread
    void join() { thread_.join(); }
    /// @brief reports whether the worker thread can be joined
    bool joinable() const { return thread_.joinable(); }

    /// @brief constructs a callable in the owning worker pool
    template <typename Task> task_pointer allocate_task(Task&& task) {
        task_pointer task_ptr = task_pool_.allocate(std::forward<Task>(task));
        if (!task_ptr->allocation_context().has_value()) { task_ptr->set_allocation_context(worker_id_); }
        return task_ptr;
    }
    /// @brief publishes an allocated task to an inbound worker queue
    void enqueue_task(task_pointer task) { task_buffer_.push(task); }
    /// @brief allocates and publishes a task in this worker
    template <typename Task> void submit_task(Task&& task) {
        task_pointer task_ptr = allocate_task(std::forward<Task>(task));
        enqueue_task(task_ptr);
        return;
    }
    /// @brief destroys and recycles a task in this worker pool
    void deallocate_task(task_pointer task) { task_pool_.deallocate(task); }
    /// @brief attempts to claim a task from another worker
    std::optional<task_pointer> try_steal() { return task_queue_.pop_back(); }
    /// @brief executes one local or stolen runnable task if available
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
    /// @brief takes local work or drains inbound work up to the deque capacity
    std::optional<task_pointer> try_fetch_task_() {
        auto&& task = task_queue_.pop_front();
        if (task) { return std::move(task); }
        // check task_buffer
        auto&& mail = task_buffer_.pop();
        if (mail) {
            // drain task_buffer
            while (!task_queue_.full()) {
                auto tmp = task_buffer_.pop();
                if (!tmp.has_value()) break;
                [[maybe_unused]] const bool pushed = task_queue_.push_front(std::move(*tmp));
                fdapde_assert(pushed, std::logic_error, "owner queue capacity changed during drain");
            }
            return mail;
        } else {
            return std::nullopt;
        }
    }
    /// @brief executes a runnable task and reports completion to its executor
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
    /// @brief registers the worker and processes tasks until the executor stops
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
    std::thread thread_;             // oS managed thread
};

}   // namespace internals

/// @brief returns the worker index or the external-thread sentinel
inline int this_thread_id() noexcept { return internals::tls_worker_id; }

}   // namespace fdapde

#endif   // __FDAPDE_EXECUTION_WORKER_H__
