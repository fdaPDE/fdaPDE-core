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

#ifndef __FDAPDE_EXECUTION_CHASE_LEV_QUEUE_H__
#define __FDAPDE_EXECUTION_CHASE_LEV_QUEUE_H__

#include "header_check.h"

namespace fdapde {
namespace internals {

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
        fdapde_assert(
          capacity != 0 && (capacity & (capacity - 1)) == 0, "Chase-Lev queue capacity must be a power of two");
    }
    // avoid copy/move-semantic
    chase_lev_queue(const chase_lev_queue&) = delete;
    chase_lev_queue(chase_lev_queue&&) = delete;
    chase_lev_queue& operator=(const chase_lev_queue&) = delete;
    chase_lev_queue& operator=(chase_lev_queue&&) = delete;

    // returns first element of the container, nullopt if container empty. the element is removed
    // only owning thread pops from buffer's front
    std::optional<value_type> pop_front() {
        difference_type b = bottom_.load(std::memory_order_relaxed);
        // Only the owner can push: an empty observation needs no bottom write or fence.
        if (b <= top_.load(std::memory_order_acquire)) return std::nullopt;
        --b;
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
        buffer_[idx].store(value_type(std::forward<T_>(value)), std::memory_order_relaxed);
        bottom_.store(b + 1, std::memory_order_release);
        return true;
    }
    // push a range of elements at buffer's back
    template <typename Iterator>
        requires(std::forward_iterator<Iterator> && std::is_convertible_v<std::iter_reference_t<Iterator>, value_type>)
    bool push_front(Iterator first, Iterator last) {
        const size_type count = std::ranges::distance(first, last);
        difference_type b = bottom_.load(std::memory_order_relaxed);
        difference_type t = top_.load(std::memory_order_acquire);
        if (count >= capacity_ || std::cmp_greater_equal(b - t + static_cast<difference_type>(count), capacity_)) {
            return false;
        }
        for (size_type i = 0; first != last; ++first, ++i) {
            buffer_[(b + difference_type(i)) & mask_].store(*first, std::memory_order_relaxed);
        }
        bottom_.store(b + count, std::memory_order_release);
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
        buffer_[idx].store(value, std::memory_order_relaxed);
        bottom_.store(b + 1, std::memory_order_release);
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
        difference_type idx = t & mask_;
        // Read before advancing top: after a successful claim the owner may immediately reuse this circular slot.
        value_type value = buffer_[idx].load(std::memory_order_relaxed);
        // try to claim the element by incrementing top
        if (!top_.compare_exchange_strong(t, t + 1, std::memory_order_seq_cst, std::memory_order_relaxed)) {
            return std::nullopt;   // lost race
        }
        return value;
    }
    // observers
    bool empty() const {
        difference_type t = top_.load(std::memory_order_acquire), b = bottom_.load(std::memory_order_acquire);
        return t >= b;
    }
    size_type size() const {
        difference_type b = bottom_.load(std::memory_order_relaxed);
        difference_type t = top_.load(std::memory_order_acquire);
        return std::max<difference_type>(b - t, 0);
    }
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

}   // namespace internals
}   // namespace fdapde

#endif   // __FDAPDE_EXECUTION_CHASE_LEV_QUEUE_H__
