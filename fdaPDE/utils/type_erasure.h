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

#ifndef __TYPE_ERASURE_H__
#define __TYPE_ERASURE_H__
#include<utility>
#include "assert.h"
#include "traits.h"

namespace fdapde {

// a set of utilities for generating type-erasing polymorphic object wrappers, supporting different storage schemes

// heterogeneous type and value list
template <auto... Vs> struct ValueList { };
template <typename... Ts> struct TypeList { };

// get I-th type from TypeList
template <int I, int N, typename T> struct TypeListGetImpl { };
template <int I, int N, typename T, typename... Ts> struct TypeListGetImpl<I, N, TypeList<T, Ts...>> {
    using type = typename std::conditional<N == I, T, typename TypeListGetImpl<I, N + 1, TypeList<Ts...>>::type>::type;
};
template <int I, int N, typename T> struct TypeListGetImpl<I, N, TypeList<T>> {
    using type = T;
};

template <int I, typename T> struct get { };
template <int I, typename... Ts> struct get<I, TypeList<Ts...>> {
    static_assert(I < sizeof...(Ts), "index out of bound");
    using type = typename TypeListGetImpl<I, 0, TypeList<Ts...>>::type;
};

// size of TypeList
template <typename T> struct size { };
template <typename... Ts> struct size<TypeList<Ts...>> {
    static constexpr int value = sizeof...(Ts);
};
template <auto... Vs> struct size<ValueList<Vs...>> {
    static constexpr int value = sizeof...(Vs);
};

// register method signature in virtual table
template <typename T, auto FuncPtr, std::size_t... Is> auto load_method(std::index_sequence<Is...>) {
    using Signature = fn_ptr_traits<FuncPtr>;
    return static_cast<typename Signature::FnPtrType>(
      [](void* obj, std::tuple_element_t<Is, typename Signature::ArgsType>... args) -> typename Signature::RetType {
          return std::mem_fn(FuncPtr)(*reinterpret_cast<T*>(obj), args...);   // cast to pointer to T done here
      });
}
// recursively initializes virtual table
template <int N, typename T, auto FuncPtr, auto... Vs> void init_vtable(ValueList<FuncPtr, Vs...>, void** vtable) {
    vtable[N] =
      reinterpret_cast<void*>(load_method<T, FuncPtr>(std::make_index_sequence<fn_ptr_traits<FuncPtr>::n_args> {}));
    init_vtable<N + 1, T, Vs...>(ValueList<Vs...>(), vtable);
}
template <int N, typename T, auto FuncPtr> void init_vtable(ValueList<FuncPtr>, void** vtable) {   // end of recursion
    vtable[N] =
      reinterpret_cast<void*>(load_method<T, FuncPtr>(std::make_index_sequence<fn_ptr_traits<FuncPtr>::n_args> {}));
}

namespace internals {
// implementation of different storage strategies for the holded type erased object
// inspired from boost.te (https://github.com/boost-ext/te)

struct shared_storage {
    std::shared_ptr<void> ptr_ = nullptr;
    void* ptr() const { return ptr_.get(); }
    // constructors
    shared_storage() = default;
    template <typename T> shared_storage(const T& obj) : ptr_(std::make_shared<T>(obj)) {};
};

struct heap_storage {
    void* ptr_ = nullptr;
    void (*del)(void*) = nullptr;
    void* (*copy)(const void*) = nullptr;

    heap_storage() = default;
    template <typename T> heap_storage(const T& obj) : ptr_(new T(obj)) {
        // store delete and copy function pointers
        del = [](void* ptr) { delete reinterpret_cast<T*>(ptr); };
        copy = [](const void* ptr) -> void* { return new T(*reinterpret_cast<const T*>(ptr)); };
    };
    // copy construct/assign
    heap_storage(const heap_storage& other) :
        ptr_(other.ptr_ == nullptr ? nullptr : other.copy(other.ptr_)), del(other.del), copy(other.copy) {};
    heap_storage& operator=(const heap_storage& other) {
        // free memory
        if (ptr_) del(ptr_);
        // deep copy data from other
        ptr_ = other.ptr_ == nullptr ? nullptr : other.copy(other.ptr_);
        del = other.del;
        copy = other.copy;
        return *this;
    }
    // move semantic
    heap_storage(heap_storage&& other) :
        ptr_(std::exchange(other.ptr_, nullptr)), del(std::exchange(other.del, nullptr)),
        copy(std::exchange(other.copy, nullptr)) {};
    heap_storage& operator=(heap_storage&& other) {
        // free memory
        if (ptr_) del(ptr_);
        // move data from other to this
        ptr_ = std::exchange(other.ptr_, nullptr);
        del  = std::exchange(other.del,  nullptr);
        copy = std::exchange(other.copy, nullptr);
        return *this;
    }
    // getter to holded data
    void* ptr() const { return ptr_; };

    ~heap_storage() {
        // free memory and restore status
        if (ptr_) del(ptr_);
        ptr_ = nullptr;
    }
};

}   // namespace internals

struct vtable_handler {
    void** vtable_ = nullptr;
    int size_ = 0;

    // copies virtual table vt1 into vt2
    void** vtable_copy(const vtable_handler& vt1, vtable_handler& vt2) {
        // free second memory and allocate new memory
        if (vt2.vtable_ == nullptr) delete[] vt2.vtable_;
        vt2.vtable_ = nullptr;
        vt2.vtable_ = new void*[vt1.size_];
        // copy
        vt2.size_ = vt1.size_;
        for (std::size_t i = 0; i < vt2.size_; ++i) { vt2.vtable_[i] = vt1.vtable_[i]; }
        return vt2.vtable_;
    }

    vtable_handler() = default;
    // copy construct/assign
    vtable_handler(const vtable_handler& other) :
        vtable_(other.vtable_ == nullptr ? nullptr : vtable_copy(other, *this)) {};
    vtable_handler& operator=(const vtable_handler& other) {
        if (vtable_ == nullptr) delete[] vtable_;
        vtable_copy(other, *this);
        return *this;
    }
    // move semantic
    vtable_handler(vtable_handler&& other) :
        vtable_(std::exchange(other.vtable_, nullptr)), size_(std::exchange(other.size_, 0)) {};
    vtable_handler& operator=(vtable_handler&& other) {
        if (vtable_ == nullptr) delete[] vtable_;
        // move data from other to this
        vtable_ = std::exchange(other.vtable_, nullptr);
        size_ = std::exchange(other.size_, 0);
        return *this;
    }

    ~vtable_handler() {
        if (vtable_ == nullptr) delete[] vtable_;
        vtable_ = nullptr;
    }

    virtual void* data() const = 0;   // pointer to stored object
};
  
template <typename I, typename StorageType = internals::heap_storage> class erase : vtable_handler, public I {
    // initializes virtual table
    template <typename T> void _vtable_init(const T& obj) {
        typedef typename std::decay<T>::type T_;
        static_assert(
          std::is_destructible<T_>::value &&
          (std::is_copy_constructible<T_>::value || std::is_move_constructible<T_>::value));
        // assign virtual table
        vtable_ = new void*[size<typename I::fn_ptrs<T_>>::value];
        size_ = size<typename I::fn_ptrs<T_>>::value;
        init_vtable<0, T_>(typename I::fn_ptrs<T_>(), vtable_);
    }
   public:
    erase() = default;
    // constructor and assignment from arbitrary type T
    template <typename T> erase(const T& obj) : data_(obj) { _vtable_init(obj); }
    template <typename T> erase& operator=(const T& obj) {
        data_ = StorageType(obj);
        _vtable_init(obj);
        return *this;
    }
    virtual void* data() const override { return data_.ptr(); }

    virtual ~erase() = default;
   private:
    StorageType data_ {};   // pointer to holded object
};

// invoke function pointer
template <typename RetType, int N, typename T, typename... Args> RetType invoke(T&& obj, Args... args) {
    fdapde_assert(N < reinterpret_cast<const vtable_handler&>(obj).size_ + 1);
    return reinterpret_cast<RetType (*)(void*, Args...)>(reinterpret_cast<const vtable_handler&>(obj).vtable_[N])(
      reinterpret_cast<const vtable_handler&>(obj).data(), args...);
}

// alias for function member pointers
template <auto... Vs> using mem_fn_ptrs = ValueList<Vs...>;
  
}   // namespace fdapde

#endif   // __TYPE_ERASURE_H__
