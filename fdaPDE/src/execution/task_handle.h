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

#ifndef __FDAPDE_EXECUTION_TASK_HANDLE_H__
#define __FDAPDE_EXECUTION_TASK_HANDLE_H__

#include "header_check.h"

namespace fdapde {
namespace internals {

// type-erasing task handle with small buffer optimization
class task_handle {
    static constexpr int buffer_size = 64;
   public:
    task_handle() noexcept = default;
    // copy semantic
    task_handle(const task_handle& other) :
        fn_(other.fn_),
        rm_(other.rm_),
        mv_(other.mv_),
        cp_(other.cp_),
        sb_(other.sb_),
        owner_id_(other.owner_id_) {
        if (sb_) {
            if (cp_) { cp_(storage_.buff_, other.storage_.buff_); }
        } else {
            if (cp_) { cp_(&storage_.data_, other.storage_.data_); }
        }
    }
    task_handle& operator=(const task_handle& other) {
        if (this == &other) return *this;
        task_handle copy(other);
        *this = std::move(copy);
        return *this;
    }
    // move semantic
    task_handle(task_handle&& other) noexcept :
        fn_(std::exchange(other.fn_, nullptr)),
        rm_(std::exchange(other.rm_, nullptr)),
        mv_(std::exchange(other.mv_, nullptr)),
        cp_(std::exchange(other.cp_, nullptr)),
        sb_(std::exchange(other.sb_, 0)),
        owner_id_(other.owner_id_)
    {
        if (sb_) {
            if (mv_) { mv_(storage_.buff_, other.storage_.buff_); }
        } else {
            storage_.data_ = std::exchange(other.storage_.data_, nullptr);
        }
    }
    task_handle& operator=(task_handle&& other) noexcept {
        if (this == &other) return *this;
        if (rm_) { rm_(sb_ ? (void*)storage_.buff_ : storage_.data_); }   // clean-up to prevent leaks

        fn_ = std::exchange(other.fn_, nullptr);
        rm_ = std::exchange(other.rm_, nullptr);
        mv_ = std::exchange(other.mv_, nullptr);
        cp_ = std::exchange(other.cp_, nullptr);
        sb_ = std::exchange(other.sb_, 0);
        if (sb_) {
            if (mv_) { mv_(storage_.buff_, other.storage_.buff_); }
        } else {
            storage_.data_ = std::exchange(other.storage_.data_, nullptr);
        }
        owner_id_ = other.owner_id_;
        return *this;
    }
    template <typename F, typename OwnerId>
        requires(
          !std::is_same_v<std::decay_t<F>, task_handle> && std::is_invocable_v<std::decay_t<F>&> &&
          std::is_copy_constructible_v<std::decay_t<F>> &&
          std::is_constructible_v<std::optional<int>, OwnerId>)
    task_handle(F&& f, OwnerId owner_id) : owner_id_(owner_id) {
        using Fn = std::decay_t<F>;
        constexpr bool sb =
          sizeof(Fn) <= buffer_size && alignof(Fn) <= alignof(union U) && std::is_nothrow_move_constructible_v<Fn>;
        sb_ = sb;
        if constexpr (sb) {   // stack allocation for small task object
            new (storage_.buff_) Fn(std::forward<F>(f));
            rm_ = [](void* ptr) noexcept {
                if (ptr) { reinterpret_cast<Fn*>(ptr)->~Fn(); }
            };
            cp_ = [](void* dst, const void* src) { new (dst) Fn(*reinterpret_cast<const Fn*>(src)); };
        } else {   // if task cannot fit in small buffer, resort to heap allocation
            storage_.data_ = new Fn(std::forward<F>(f));
            rm_ = [](void* ptr) noexcept {
                if (ptr) { delete reinterpret_cast<Fn*>(ptr); }
            };
            cp_ = [](void* dst, const void* src) {
                *reinterpret_cast<void**>(dst) = new Fn(*reinterpret_cast<const Fn*>(src));
            };
        }   // type-erased function handlers
        fn_ = [](void* ptr) { (*reinterpret_cast<Fn*>(ptr))(); };

        mv_ = [](void* dst, void* src) noexcept {
            new (dst) Fn(std::move(*reinterpret_cast<Fn*>(src)));
            (*reinterpret_cast<Fn*>(src)).~Fn();
        };
    }
    template <typename F>
        requires(
          !std::is_same_v<std::decay_t<F>, task_handle> && std::is_invocable_v<std::decay_t<F>&> &&
          std::is_copy_constructible_v<std::decay_t<F>>)
    explicit task_handle(F&& f) : task_handle(std::forward<F>(f), std::nullopt) { }

    // invoke
    void run() { fn_(sb_ ? (void*)storage_.buff_ : storage_.data_); }
    // ownership managment
    int owner_id() const { return owner_id_.has_value() ? owner_id_.value() : -1; }
    void set_owner(int owner_id) { owner_id_ = owner_id; }
    // destructor (task destroyed only after completion)
    ~task_handle() {
        if (rm_) { rm_(sb_ ? (void*)storage_.buff_ : storage_.data_); }
    }
   private:
    void (*fn_)(void*) = nullptr;
    void (*rm_)(void*) = nullptr;
    void (*mv_)(void*, void*) = nullptr;
    void (*cp_)(void*, const void*) = nullptr;
    union alignas(std::max_align_t) U {
        void* data_ = nullptr;
        std::byte buff_[buffer_size];   // small buffer optimization
    } storage_ {};
    bool sb_ = false;

    // task properties
    std::optional<int> owner_id_;   // execution context responsible for this task's accounting
};

}   // namespace internals
}   // namespace fdapde

#endif   // __FDAPDE_EXECUTION_TASK_HANDLE_H__
