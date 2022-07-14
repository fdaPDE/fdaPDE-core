// template recursion to generate an N-dimensional for loop at compile time
 
// recursive step
template <int I, int... N>
struct Looper{
    template <typename F, typename... X>
    constexpr void operator()(F& f, X... x) {
        for (int i = 0; i < I; ++i) {
            Looper<N...>()(f, x..., i);
        }
    }
};

// base case (end of recursion)
template <int I>
struct Looper<I>{
    template <typename F, typename ...X>
    constexpr void operator()(F& f, X... x) {
        for (int i = 0; i < I; ++i) {
            f(x..., i);
        }
    }
};
