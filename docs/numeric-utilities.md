# Numeric utilities

Include `<fdaPDE/utility.h>` to use scalar utilities in runtime code and C++20 constant expressions. Include `<fdaPDE/linear_algebra.h>` for native dense matrix types and the existing Eigen-backed algebra modules.

```cpp
#include <fdaPDE/utility.h>

static_assert(fdapde::sqrt(1e-20) > 0.99999999999999e-10);
static_assert(fdapde::min(2, 1.5) == 1.5);
static_assert(fdapde::ldexp(0.75f, 3) == 6.0f);
```

`sqrt` supports floating-point scalars, preserves signed zero and positive infinity, and returns NaN for negative arguments or NaN. Runtime evaluation uses `std::sqrt`; constant evaluation normalizes the input before Heron's iteration.

`frexp` decomposes a `double` into a signed fraction and binary exponent. Finite nonzero fractions have magnitude in `[0.5, 1)`, including subnormal inputs. Zero and nonfinite arguments are returned unchanged with exponent zero. Runtime finite decomposition uses `std::frexp`.

`ldexp` scales a floating-point scalar by an integer power of two. It supports `float`, `double`, and `long double`, preserves signed zero and nonfinite inputs, and handles subnormal results and exponents wider than `int`. Runtime evaluation uses `std::ldexp` after bounding the exponent. Constant evaluation normalizes first and rounds subnormal results once at the final multiplication.

`log` returns positive infinity for positive infinity, negative infinity for either signed zero, and NaN for negative arguments or NaN. Finite `log` and `exp` use polynomial approximations rather than correctly rounded implementations. Tests bound the logarithm's absolute error by `1e-9` at the listed representative scales; they do not establish a uniform error bound over every floating-point input. `log1p` and `log1pexp` compose these approximations, with separate branches for small or large arguments.

`pow` accepts integer exponents, including the minimum signed exponent and wide integer exponents. As with built-in arithmetic, intermediate overflow limits constant evaluation. `min` and `max` promote their arguments to a common type and participate only when the corresponding scalar comparison is valid.

The existing comparison conventions remain unchanged: `sign(x)` reports nonnegativity as zero or one; `greater_equal` and `less_equal` compare a difference against relative tolerance and therefore exclude equal nonzero values at the default positive tolerance. The strict `greater_than` and `less_than` helpers remain available to existing consumers. Integer division requires a nonzero divisor; factorial and combinations are limited by their integer result range. `ceil` and `floor` require inputs whose integral part fits in `long`.

Expression nesting metadata remains internal. `ref_select` preserves the declared `NestAsRef` policy; it does not itself establish lifetime safety for temporary owning expressions.

# Verification

From the repository root:

```sh
cmake -S tests -B build/numeric-debug -DCMAKE_BUILD_TYPE=Debug
cmake --build build/numeric-debug --parallel
ctest --test-dir build/numeric-debug --output-on-failure
```

The tests cover numeric constant evaluation and runtime boundaries, nesting traits, Eigen vector dispatch, stable dense expressions, KD-tree boundary queries, and analytic P1 finite-element mass assembly. Compile targets check `utility.h`, `linear_algebra.h`, and `core.h` independently. Debug assertions stay enabled in the numeric tests, including CMake Release builds. The pre-existing focused disabled-assert test is separate.

For GCC on the macOS SDK whose Mach headers use `_Static_assert` in C++, configure with `-DCMAKE_CXX_FLAGS=-D_Static_assert=static_assert`. This is a toolchain workaround, not a change to fdaPDE assertion behavior. Sanitizer builds can use `-DCMAKE_CXX_FLAGS='-fsanitize=address,undefined -fno-omit-frame-pointer'` with Clang.
