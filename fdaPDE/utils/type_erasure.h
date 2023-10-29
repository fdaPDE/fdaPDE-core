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

// a set of utilities for generating type-erasing polymorphic object wrappers

// heterogeneous value list
template <auto... Vs> struct ValueList { };
template <auto... Vs> using BindingsList = ValueList<Vs...>;

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

class TypeErasureBase {
   protected:
    std::shared_ptr<void> ptr_ = nullptr;   // pointer to holded object
    void* vtable_ = nullptr;                // pointer to virtual table
   public:
    void* get() { return ptr_.get(); }
    void* vtable() { return vtable_; }
};

template <typename... Ts> struct interface_signature : public TypeErasureBase {
    using Signature = TypeList<Ts...>;
};

template <typename I> struct TypeErasure : public I::Interface {
    typedef typename I::Interface Base;
    using Base::ptr_;
    using Base::vtable_;

    TypeErasure() = default;
    template <typename T> TypeErasure(const T& obj) {
        typedef std::add_pointer_t<decltype(init_vtable<0, T, Base>(typename I::Bindings<T>()))> VTableType;
        ptr_ = std::make_shared<T>(obj);
        vtable_ = new auto(init_vtable<0, T, Base>(typename I::Bindings<T>()));
        // store deleter
        del = [](void* obj) { delete reinterpret_cast<VTableType>(obj); };
    }
    template <typename T> TypeErasure& operator=(const T& obj) {
        typedef std::add_pointer_t<decltype(init_vtable<0, T, Base>(typename I::Bindings<T>()))> VTableType;
        ptr_ = std::make_shared<T>(obj);
        vtable_ = new auto(init_vtable<0, T, Base>(typename I::Bindings<T>()));
        // store deleter
        del = [](void* obj) { delete reinterpret_cast<VTableType>(obj); };
        return *this;
    }
    explicit operator bool() const { return vtable_ == nullptr; }   // check if in a valid state

    ~TypeErasure() {
        if (vtable_) del(vtable_);
        vtable_ = nullptr;
    }
   private:
    void (*del)(void*) = nullptr;   // function pointer to deleter
};

template <int N, typename T, typename... Args> auto invoke(T&& t, Args... args) {
    return std::get<N>(*reinterpret_cast<typename tuplify<typename std::decay_t<T>::Signature>::type*>(t.vtable()))(
      t.get(), args...);
}

}   // namespace fdapde

#endif   // __TYPE_ERASURE_H__
