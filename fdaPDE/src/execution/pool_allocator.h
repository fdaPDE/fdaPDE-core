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

#ifndef __FDAPDE_EXECUTION_POOL_ALLOCATOR_H__
#define __FDAPDE_EXECUTION_POOL_ALLOCATOR_H__

#include "header_check.h"

namespace fdapde {
namespace internals {

// shared allocator with per-thread memory pools and atomic cross-thread returns
// each allocating physical thread owns a memory_pool in the allocator, avoiding coordination overhead due to memory
// sharing. memory pools are allocated on-demand.
template <typename T> struct pool_allocator {
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    using size_type = std::uint64_t;
    using difference_type = std::int64_t;
   private:
    struct memory_pool;
    // the basis addressable memory unit inside a memory_pool
    struct memory_slot {
        alignas(T) std::byte storage[sizeof(T)];   // actual stored data
        memory_pool* owner;                        // pool to which this slot belong
        memory_slot* next;                         // next slot in the pool
    };
    // per-thread reserved memory, organized in contiguous blocks of memory slots
    struct memory_pool {
        explicit memory_pool(std::uint64_t thread) : owner_id(thread) { }

        const std::uint64_t owner_id;       // ID of the thread owning this pool
        memory_slot* free_list = nullptr;   // pointer to the first available thread's memory slot in this allocator
        std::atomic<memory_slot*> returned {nullptr};         // pointer to freed memory_slots
        std::vector<std::unique_ptr<memory_slot[]>> blocks;   // physical memory
    };
    // per-thread allocator ID -> memory pool mapping
    struct pool_binding {
        std::weak_ptr<int> lifetime;   // whether the allocator still exists
        memory_pool* pool;             // pointer to physical memory
    };
    inline static thread_local std::unordered_map<std::uint64_t, pool_binding> tls_bindings_;
    // memory_pool address and ID of the allocator most recently accessed by this thread
    inline static thread_local memory_pool* tls_last_accessed_memory_pool_ = nullptr;
    inline static thread_local std::uint64_t tls_last_accessed_allocator_id_ =
      std::numeric_limits<std::uint64_t>::max();
    // thread unique identifier across memory pools
    inline static std::atomic<std::uint64_t>  next_tls_thread_id_ = 0;
    inline static thread_local const std::uint64_t tls_thread_id_ = next_tls_thread_id_.fetch_add(1);

    memory_pool& get_tls_memory_pool_() {
        if (tls_last_accessed_allocator_id_ == this_allocator_id_) { return *tls_last_accessed_memory_pool_; }
        // check if this thread already has reserved memory in this allocator
        auto it = tls_bindings_.find(this_allocator_id_);
        if (it != tls_bindings_.end()) {
            tls_last_accessed_allocator_id_ = it->first;
            tls_last_accessed_memory_pool_  = it->second.pool;
            return *tls_last_accessed_memory_pool_;
        }

	// this thread never allocated here, prepare a new memory pool
	// clean thread's bindings
        for (auto old = tls_bindings_.begin(); old != tls_bindings_.end();) {
            old = old->second.lifetime.expired() ? tls_bindings_.erase(old) : std::next(old);
        }
        // create new memory pool
        auto storage = std::make_unique<memory_pool>(tls_thread_id_);
	// transfer ownership to this allocator
        auto* ptr = storage.get();
        {
            std::lock_guard<std::mutex> lock(m_);
            pools_.push_back(std::move(storage));
        }
	// update thread bindings
        tls_bindings_.emplace(this_allocator_id_, pool_binding {lifetime_, ptr});
        tls_last_accessed_allocator_id_ = this_allocator_id_;
        tls_last_accessed_memory_pool_  = ptr;
        return *ptr;
    }
   public:
    pool_allocator() : pool_allocator(1024) { }
    explicit pool_allocator(size_type block_size) : block_size_(block_size) {
        fdapde_assert(block_size > 0, "allocator block size must be positive");
    }
    pool_allocator(const pool_allocator&) = delete;
    pool_allocator& operator=(const pool_allocator&) = delete;

    template <typename... Args> pointer allocate(Args&&... args) {
        auto& memory_pool = get_tls_memory_pool_();
        if (!memory_pool.free_list) {   // no free slots, try to request from returned slots
            memory_pool.free_list = memory_pool.returned.exchange(nullptr, std::memory_order_acquire);
        }
        if (!memory_pool.free_list) {   // no free slots, allocate new memory block
            auto block = std::make_unique<memory_slot[]>(block_size_);
            auto* first = block.get();
            for (size_type i = 0; i < block_size_; ++i) {
                first[i].owner = std::addressof(memory_pool);
                first[i].next = i + 1 < block_size_ ? first + i + 1 : nullptr;
            }
            memory_pool.blocks.push_back(std::move(block));
            memory_pool.free_list = first;
        }
        // allocate object in memory
        auto* slot = memory_pool.free_list;
        memory_pool.free_list = slot->next;
        try {
            return new (static_cast<void*>(slot->storage)) T(std::forward<Args>(args)...);
        } catch (...) {
            slot->next = memory_pool.free_list;
            memory_pool.free_list = slot;
            throw;
        }
    }
    void deallocate(pointer object) {
        // free object memory
        std::destroy_at(object);
	// update memory pool
        auto* slot = reinterpret_cast<memory_slot*>(object);
        auto& owner = *slot->owner;
        if (owner.owner_id == tls_thread_id_) {
            slot->next = owner.free_list;
            owner.free_list = slot;
        } else {
            auto* head = owner.returned.load(std::memory_order_relaxed);
            do {
                slot->next = head;
            } while (
              !owner.returned.compare_exchange_weak(head, slot, std::memory_order_release, std::memory_order_relaxed));
        }
	return;
    }
   private:
    inline static std::atomic<std::uint64_t> next_allocator_id_ {0};
    const std::uint64_t this_allocator_id_ = next_allocator_id_.fetch_add(1, std::memory_order_relaxed);

    const size_type block_size_;
    std::shared_ptr<int> lifetime_ = std::make_shared<int>(0);
    std::mutex m_;
    std::vector<std::unique_ptr<memory_pool>> pools_;
};

}   // namespace internals
}   // namespace fdapde

#endif   // __FDAPDE_EXECUTION_POOL_ALLOCATOR_H__
