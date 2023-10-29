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

namespace fdapde {

// a set of utilities for generating type-erasing polymorphic object wrappers, supporting different storage schemes

// heterogeneous value list
template <auto... Vs> struct ValueList { };
template <auto... Vs> using BindingsList = ValueList<Vs...>;

// size of value list
template <auto... Vs> constexpr int size(ValueList<Vs...>) { return sizeof...(Vs); }

// heterogeneous type list
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
template <typename... Ts> constexpr int size(TypeList<Ts...>) { return sizeof...(Ts); }

template <typename... Ts> struct interface_signature {
    using Signature = TypeList<Ts...>;
};
  
template <typename... T> struct method_signature { };
template <typename R, typename... Args> struct method_signature<R(Args...)> {
    using RetType = R;                                // return type
    using ArgsType = std::tuple<Args...>;             // arguments parameter pack
    typedef RetType (*FuncPtrType)(void*, Args...);   // void* requested by vtable
    static constexpr int n_args = sizeof...(Args);
};

// transform a TypeList into a std::tuple
template <typename T> struct tuplify { };
template <typename... Ts> struct tuplify<TypeList<Ts...>> {
    using type = std::tuple<typename Ts::FuncPtrType...>;
};

template <typename T, typename Signature, auto FuncPtr, std::size_t... Is>
auto load_method(std::index_sequence<Is...>) {
    using ArgsTuple = typename Signature::ArgsType;
    return std::make_tuple(static_cast<typename Signature::FuncPtrType>(
      [](void* obj, std::tuple_element_t<Is, ArgsTuple>... args) -> typename Signature::RetType {
          return std::mem_fn(FuncPtr)(*reinterpret_cast<T*>(obj), args...);   // cast to pointer to T done here
      }));
}

template <int N, typename T, typename I, auto FuncPtr, auto... Vs> auto init_vtable(ValueList<FuncPtr, Vs...>) {
    using MethodSignature = typename get<N, typename I::Signature>::type;
    return std::tuple_cat(
      load_method<T, MethodSignature, FuncPtr>(std::make_index_sequence<MethodSignature::n_args> {}),
      init_vtable<N + 1, T, I, Vs...>(ValueList<Vs...>()));
}

template <int N, typename T, typename I, auto FuncPtr> auto init_vtable(ValueList<FuncPtr> list) {
    using MethodSignature = typename get<N, typename I::Signature>::type;
    return load_method<T, MethodSignature, FuncPtr>(std::make_index_sequence<MethodSignature::n_args> {});
}

namespace internals {
// inspired from boost.te (https://github.com/boost-ext/te)

// shallow-copy semantic
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
    template <typename T> heap_storage(const T& obj) : ptr_(new T {obj}) {
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
        // deep copy
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
        del  = std::exchange(other.del , nullptr);
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

struct TypeErasureBase {
    void* vtable_ = nullptr;   // pointer to virtual table
    virtual void* data() const = 0;
};

template <typename I, typename StorageType = internals::heap_storage> class TypeErasure : TypeErasureBase, public I {
    template <typename T> void init(const T& obj) {
        typedef typename std::decay<T>::type T_;
        typedef std::add_pointer_t<decltype(init_vtable<0, T_, I>(typename I::Bindings<T_>()))> VTableType;
        static_assert(size(typename I::Bindings<T_>()) == size(typename I::Signature()));
        static_assert(
          std::is_destructible<T_>::value &&
          (std::is_copy_constructible<T_>::value || std::is_move_constructible<T_>::value));

        vtable_ = new auto(init_vtable<0, T, I>(typename I::Bindings<T_>()));
    }
   public:
    TypeErasure() = default;
    // constructor and assignment from arbitrary type T
    template <typename T> TypeErasure(const T& obj) : data_(obj) { init(obj); }
    template <typename T> TypeErasure& operator=(const T& obj) {
        data_ = StorageType(obj);
        init(obj);
        return *this;
    }
    explicit operator bool() const { return vtable_ == nullptr; }
    // getters
    void* vtable() { return vtable_; }
    virtual void* data() const override { return data_.ptr(); }

    ~TypeErasure() = default;
   private:
    StorageType data_ {};   // pointer to holded object
};

template <int N, typename T, typename... Args> auto invoke(T&& t, Args... args) {
    typedef typename tuplify<typename std::decay_t<T>::Signature>::type* VTablePtrType;
    return std::get<N>(*reinterpret_cast<VTablePtrType>(reinterpret_cast<const TypeErasureBase&>(t).vtable_))(
      reinterpret_cast<const TypeErasureBase&>(t).data(), args...);
}

}   // namespace fdapde

#endif   // __TYPE_ERASURE_H__
