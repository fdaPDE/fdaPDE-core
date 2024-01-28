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

#include <typeinfo>
#include <typeindex>
#include <unordered_map>
#include <utility>
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

// merge an arbitrary number of value lists
template <typename V1, typename V2> struct MergeImpl { };
template <auto... Vs1, auto... Vs2> struct MergeImpl<ValueList<Vs1...>, ValueList<Vs2...>> {
    using type = ValueList<Vs1..., Vs2...>; // metafunction to merge a pair of value lists
};
template <typename... Vs> struct merge { };
template <typename V1, typename V2, typename... Vs> struct merge<V1, V2, Vs...> {
    using type = typename merge<typename MergeImpl<V1, V2>::type, Vs...>::type;
};
template <typename V> struct merge<V> {   // end of recursion
    using type = V;
};

// register method signature in virtual table
template <typename T, auto FuncPtr, std::size_t... Is> auto load_method(std::index_sequence<Is...>) {
    using Signature = fn_ptr_traits<FuncPtr>;
    return static_cast<typename Signature::FnPtrType>(
      [](void* obj, std::tuple_element_t<Is, typename Signature::ArgsType>&&... args) -> typename Signature::RetType {
          return std::mem_fn(FuncPtr)(   // cast to pointer to T done here
            *reinterpret_cast<T*>(obj), std::forward<std::tuple_element_t<Is, typename Signature::ArgsType>>(args)...);
      });
}
// recursively initializes virtual table
template <int N, typename T, auto FuncPtr, auto V, auto... Vs>
void init_vtable(ValueList<FuncPtr, V, Vs...>, void** vtable) {
    vtable[N] =
      reinterpret_cast<void*>(load_method<T, FuncPtr>(std::make_index_sequence<fn_ptr_traits<FuncPtr>::n_args> {}));
    init_vtable<N + 1, T, V, Vs...>(ValueList<V, Vs...>(), vtable);
}
template <int N, typename T, auto FuncPtr> void init_vtable(ValueList<FuncPtr>, void** vtable) {   // end of recursion
    vtable[N] =
      reinterpret_cast<void*>(load_method<T, FuncPtr>(std::make_index_sequence<fn_ptr_traits<FuncPtr>::n_args> {}));
}

// implementation of different storage strategies for the holded type erased object
// inspired from boost.te (https://github.com/boost-ext/te)

struct shared_storage {
    std::shared_ptr<void> ptr_ = nullptr;
    void* ptr() { return ptr_.get(); }
    const void* ptr() const { return ptr_.get(); }

    shared_storage() = default;
    template <typename T> shared_storage(const T& obj) : ptr_(std::make_shared<T>(obj)) {};
};

struct non_owning_storage {
    const void* ptr_ = nullptr;
    void* ptr() { return const_cast<void*>(ptr_); }
    const void* ptr() const { return ptr_; }

    non_owning_storage() = default;
    template <typename T> non_owning_storage(T& obj) : ptr_(&obj) {};
    template <typename T> non_owning_storage(const T& obj) : ptr_(&obj) {};
    // copy construct/assign
    non_owning_storage(const non_owning_storage& other) : ptr_(other.ptr_) { }
    non_owning_storage& operator=(const non_owning_storage& other) {
        ptr_ = other.ptr_;
        return *this;
    }
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
        ptr_(other.ptr_ == nullptr ? nullptr : other.copy(other.ptr_)), del(other.del), copy(other.copy) { };
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
    void* ptr() { return ptr_; };
    const void* ptr() const { return ptr_; }
  
    ~heap_storage() {
        // free memory and restore status
        if (ptr_) del(ptr_);
        ptr_ = nullptr;
    }
};

struct vtable_handler {
    void** vtable_ = nullptr;
    int size_;
    std::unordered_map<std::type_index, int> offset_table_ {};

    // copies virtual table vt1 into vt2
    void** vtable_copy(const vtable_handler& vt1, vtable_handler& vt2) {
        vt2.size_ = vt1.size_;
        vt2.vtable_ = new void*[vt2.size_];
        for (std::size_t i = 0; i < vt2.size_; ++i) { vt2.vtable_[i] = vt1.vtable_[i]; }
        return vt2.vtable_;
    }

    vtable_handler() : vtable_(nullptr), size_(0) {};
    // copy construct/assign
    vtable_handler(const vtable_handler& other) :
        vtable_(other.vtable_ == nullptr ? nullptr : vtable_copy(other, *this)), offset_table_(other.offset_table_) { };
    vtable_handler& operator=(const vtable_handler& other) {
        if (vtable_) delete[] vtable_;
        vtable_copy(other, *this);
	offset_table_ = other.offset_table_;
        return *this;
    }
    // move semantic
    vtable_handler(vtable_handler&& other) :
        vtable_(std::exchange(other.vtable_, nullptr)), size_(std::exchange(other.size_, 0)),
        offset_table_(std::move(other.offset_table_)) {};
    vtable_handler& operator=(vtable_handler&& other) {
        if (vtable_) delete[] vtable_;
        // move data from other to this
        vtable_ = std::exchange(other.vtable_, nullptr);
        size_ = std::exchange(other.size_, 0);
        offset_table_ = std::move(other.offset_table_);
        return *this;
    }

    ~vtable_handler() {
        if(vtable_) delete[] vtable_;
        vtable_ = nullptr;
	size_ = 0;
    }

    operator bool() const { return size_ != 0; }
    virtual void* __data() = 0;   // pointer to stored object
    virtual const void* __data() const = 0;
};
  
template <typename StorageType, typename... I> class erase : vtable_handler, public I... {
    // initializes virtual table
    template <typename T> void _vtable_init(const T& obj) {
        typedef typename std::decay<T>::type T_;
        static_assert(
          std::is_destructible<T_>::value &&
          (std::is_copy_constructible<T_>::value || std::is_move_constructible<T_>::value));
        // assign virtual table
	size_ = (size<typename I::template fn_ptrs<T_>>::value + ...);
	delete[] vtable_;
        vtable_ = new void*[size_];
        init_vtable<0, T_>(typename merge<typename I::template fn_ptrs<T_>...>::type(), vtable_);
	// initialize offset table
        int base_ = 0;
        auto load_offset = [this, &base_](const auto& i) -> void {
            offset_table_[typeid(std::decay_t<decltype(i)>)] = base_;
            base_ += size<typename std::decay<decltype(i)>::type::template fn_ptrs<T_>>::value;
        };
        std::apply([load_offset](const auto&... type) { (load_offset(type), ...); }, std::tuple<I...>());
    }
   public:
    erase() = default;
    erase(const erase& other) : vtable_handler(other), data_(other.data_) { };
    erase& operator=(const erase& other) {
      vtable_handler::operator=(other);
      data_ = other.data_;
      return *this;
    }
    erase(erase&& other) : vtable_handler(other), data_(std::move(other.data_)) {}
    erase& operator=(erase&& other) {
      vtable_handler::operator=(other);
      data_ = std::move(other.data_);
      return *this;
    };
    // constructor and assignment from arbitrary type T
    template <typename T> erase(const T& obj) : data_(obj) { _vtable_init(obj); }
    template <typename T> erase& operator=(const T& obj) {
        data_ = StorageType(obj);
        _vtable_init(obj);
        return *this;
    }

    virtual void* __data() override { return data_.ptr(); }
    virtual const void* __data() const override { return data_.ptr(); }
    operator bool() const { return vtable_handler::operator bool(); }
    
    virtual ~erase() = default;
   private:
    StorageType data_ {};    // pointer to holded object
};

// invoke function pointer (T is deduced to the type of the interface)
template <typename RetType, int N, typename T, typename... Args> RetType invoke(T&& obj, Args&&... args) {
    auto& vtable = reinterpret_cast<const vtable_handler&>(obj);
    short offset = vtable.offset_table_.at(typeid(std::decay_t<T>)) + N;
    return reinterpret_cast<RetType (*)(const void*, Args&&...)>(vtable.vtable_[offset])(
      vtable.__data(), std::forward<Args>(args)...);
}
  
// alias for function member pointers
template <auto... Vs> using mem_fn_ptrs = ValueList<Vs...>;
  
}   // namespace fdapde

#endif   // __TYPE_ERASURE_H__
